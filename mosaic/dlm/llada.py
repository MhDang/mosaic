"""LLaDA's denoising loop, with an optional automaton constraint.

LLaDA's reference ``generate``: the response is decoded in blocks of
``block_length``, each with ``steps / num_blocks`` steps that commit an equal
share of its masks. Each step proposes x0 everywhere (Gumbel-max sampling at
``temperature``, or LM x automaton when constrained) and commits the
highest-confidence masked positions of the active block. ``remasking``
chooses the confidence: the model's probability of the proposed token
(``low_confidence``; its constrained marginal under ``commit_by=constrained``)
or uniform noise (``random``).
"""

import numpy as np
import torch
import torch.nn.functional as F

from .constraint import check_matcher, resolve_commit_by

REMASKING = ("low_confidence", "random")


@torch.no_grad()
def generate(model, prompt, mask_id, *, steps=64, gen_length=128, block_length=32,
             temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
             vocab_size=None, matcher=None, commit_by="constrained"):
    """Generate ``gen_length`` tokens after ``prompt`` (bz, prompt_len). Returns the canvas.

    vocab_size: tokens the unconstrained proposal may draw from (the tokenizer's
        vocabulary; the model's logits are padded beyond it).
    matcher: a fresh :class:`~mosaic.matcher.ConstraintMatcher` bounded to ``gen_length``,
        or None (unconstrained).
    """
    commit_by = resolve_commit_by(commit_by)
    check_matcher(matcher, gen_length)
    if remasking not in REMASKING:
        raise NotImplementedError(remasking)

    with torch.autocast(device_type="cuda", enabled=False):
        prompt_len = prompt.shape[1]
        x = torch.full(
            (prompt.shape[0], prompt_len + gen_length), mask_id,
            dtype=torch.long, device=prompt.device,
        )
        x[:, :prompt_len] = prompt.clone()
        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        for num_block in range(num_blocks):
            end_idx = prompt_len + (num_block + 1) * block_length
            block_mask_index = x[:, prompt_len + num_block * block_length:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == mask_id

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    logits = model(torch.cat([x, un_x], dim=0)).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                p_original = F.softmax(logits, dim=-1)
                marginal = None
                if matcher is None:
                    gen_logits = logits[:, prompt_len:, :vocab_size]
                    x0_gen = torch.argmax(add_gumbel_noise(gen_logits, temperature), dim=-1)
                else:
                    x0_gen, marginal = matcher.propose_x0(
                        logits[:, prompt_len:], x[:, prompt_len:], mask_id, temperature=temperature,
                        return_marginal=commit_by == "constrained",
                    )
                x0 = torch.cat((x[:, :prompt_len], x0_gen), dim=-1)

                if remasking == "low_confidence":
                    if marginal is not None:
                        p_prompt = torch.gather(
                            p_original[:, :prompt_len, :], dim=-1,
                            index=x0[:, :prompt_len].unsqueeze(-1),
                        ).squeeze(-1)
                        x0_p = torch.cat([p_prompt, marginal], dim=1)
                    else:
                        x0_p = torch.gather(p_original, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                else:  # random
                    x0_p = torch.rand(x0.shape, device=x0.device)

                # never commit beyond the current block
                x0_p[:, end_idx:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_indices] = x0[j, select_indices]
        return x


def add_gumbel_noise(logits, temperature):
    """Gumbel-max sampling: argmax of the result draws from softmax(logits / temperature)."""
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """How many masks to commit at each of ``steps`` steps: an even split, remainder first."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps).clone()
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        num_transfer_tokens[indices.unsqueeze(0) < remainder] += 1
    return num_transfer_tokens.to(torch.int64)
