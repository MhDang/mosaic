"""Dream's denoising loop, with an optional automaton constraint.

The loop of Dream's ``generation_utils._sample`` without the Hugging Face
generation plumbing: pad the prompt with masks, then for each step predict x0 at
the masked positions and commit the most confident ones on a linear timestep
schedule. Unconstrained, x0 and its confidence come from ``alg``; constrained,
x0 is drawn from LM x automaton over the whole response and the confidence is
its constrained marginal (``commit_by=constrained``) or the LM's probability
of the drawn token (``model``).
"""

import torch
import torch.distributions as dists
import torch.nn.functional as F

from .constraint import check_matcher, resolve_commit_by

#: Unconstrained ways to propose x0 and rank positions.
ALGS = ("origin", "maskgit_plus", "topk_margin", "entropy")


@torch.no_grad()
def generate(model, input_ids, mask_token_id, *, steps, max_new_tokens,
             temperature=0.0, top_p=None, top_k=None, alg="entropy", alg_temp=None,
             eps=1e-3, block_length=None, matcher=None, commit_by="constrained"):
    """Generate ``max_new_tokens`` after ``input_ids`` (bz, prompt_len). Returns the canvas.

    block_length: decode semi-autoregressively in blocks of this many tokens
        (``steps`` split evenly between them); None means one block. The model
        and the constraint always see the whole response; only commits are
        restricted to the active block.
    matcher: a fresh :class:`~mosaic.matcher.ConstraintMatcher` bounded to ``max_new_tokens``,
        or None (unconstrained).
    """
    commit_by = resolve_commit_by(commit_by)
    check_matcher(matcher, max_new_tokens)
    if matcher is None and alg not in ALGS:
        raise ValueError(f"alg must be one of {ALGS}, got {alg!r}")

    prompt_len = input_ids.shape[1]
    max_length = prompt_len + max_new_tokens
    x = F.pad(input_ids, (0, max_length - prompt_len), value=mask_token_id)

    gen_region_len = max_length - prompt_len
    if block_length is None or block_length <= 0 or block_length >= gen_region_len:
        block_length = gen_region_len
    num_blocks = max(1, (gen_region_len + block_length - 1) // block_length)
    steps_per_block = max(1, steps // num_blocks)
    # per-block timestep schedule, from t=1 down to eps
    timesteps = torch.linspace(1, eps, steps_per_block + 1, device=x.device)

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = min(block_start + block_length, max_length)

        for inner_step in range(steps_per_block):
            mask_index_all = x == mask_token_id
            block_pos_mask = torch.zeros_like(x, dtype=torch.bool)
            block_pos_mask[:, block_start:block_end] = True
            mask_index = mask_index_all & block_pos_mask  # masks in the active block
            if not mask_index.any():
                break

            logits = model(x, "full", None).logits
            # Dream predicts position i from position i-1 (AR initialisation)
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            mask_logits = logits[mask_index]
            t = timesteps[inner_step]
            s = timesteps[inner_step + 1]
            is_last_inner = inner_step == steps_per_block - 1

            if matcher is None and alg == "origin":
                p_transfer = 1 - s / t if not is_last_inner else 1
                x0 = torch.zeros_like(x[mask_index], device=x.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=x.device) < p_transfer
                _, x0[transfer_index_t_s] = sample_tokens(
                    mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k
                )
                x[mask_index] = x0.clone()
                continue

            if matcher is not None:
                x0_gen, marginal = matcher.propose_x0(
                    logits[:, prompt_len:], x[:, prompt_len:], mask_token_id, temperature=temperature,
                    return_marginal=commit_by == "constrained",
                )
                if marginal is None:
                    p_gen = F.softmax(logits[:, prompt_len:, :], dim=-1)
                    marginal = torch.gather(p_gen, dim=-1, index=x0_gen.unsqueeze(-1)).squeeze(-1)
                gen_active_mask = mask_index[:, prompt_len:]
                x0 = x0_gen[gen_active_mask]
                confidence = marginal[gen_active_mask]
            elif alg == "maskgit_plus":
                confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
            elif alg == "topk_margin":
                confidence, x0 = sample_tokens(
                    mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True
                )
            else:  # entropy
                confidence, x0 = sample_tokens(
                    mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True
                )

            num_mask_token = mask_index.sum() / mask_index.shape[0]
            # at least one per step while masks remain; the block's last step commits the rest
            if not is_last_inner:
                number_transfer_tokens = max(1, int(num_mask_token * (1 - s / t)))
            else:
                number_transfer_tokens = int(num_mask_token)
            full_confidence = torch.full_like(x, -torch.inf, device=x.device, dtype=logits.dtype)
            full_confidence[mask_index] = confidence.to(dtype=full_confidence.dtype)
            if number_transfer_tokens > 0:
                if alg_temp is None or alg_temp == 0:
                    _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                else:
                    full_confidence = full_confidence / alg_temp
                    full_confidence = F.softmax(full_confidence, dim=-1)
                    transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                x_ = torch.zeros_like(x, device=x.device, dtype=torch.long) + mask_token_id
                x_[mask_index] = x0.clone()
                row_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(transfer_index)
                x[row_indices, transfer_index] = x_[row_indices, transfer_index]

    return x


# ---- unconstrained proposals (Dream's sample_tokens) ----


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None,
                  margin_confidence=False, neg_entropy=False):
    """Propose x0 per row of ``logits``; returns (confidence, x0).

    Confidence is the drawn token's probability, the top-1 minus top-2 margin,
    or the distribution's negentropy.
    """
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = _top_p_logits(logits, top_p)
    if top_k is not None:
        logits = _top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[:, 0] - sorted_probs[:, 1]

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


def _top_p_logits(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # shift right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def _top_k_logits(logits, top_k):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
