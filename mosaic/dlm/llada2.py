"""LLaDA2's block-diffusion loop (LLaDA2.0), with an optional constraint matcher.

LLaDA2 decodes blocks of ``block_length`` tokens left to right; blocks are
aligned to absolute positions, so the first one also holds the tail of the
prompt. Attention is causal across blocks and bidirectional within one. Each
step proposes x0 for the active block and commits every masked position whose
confidence exceeds ``threshold``, or the single most confident one if none
does; the block is done when no mask is left. This is SGLang's
``LowConfidence`` rule, and the reference ``generate``'s with
``steps == block_length``. Decoding stops after a block that contains EOS.

Unconstrained, x0 is the argmax (or a sample at ``temperature``) and its
confidence the model's probability. With a :class:`~mosaic.matcher.ConstraintMatcher`,
x0 comes from ``matcher.propose_x0`` and the confidence is the model's probability
(``commit_by=model``, the default) or its constrained marginal (``constrained``):
under a threshold rule, near-1 marginals on the tokens the constraint forces would
commit too much per step.
"""

import torch
import torch.nn.functional as F

from .constraint import resolve_commit_by


@torch.no_grad()
def generate(model, input_ids, mask_id, *, gen_length=256, block_length=32,
             threshold=0.95, temperature=0.0, eos_id=None, max_steps_per_block=None,
             matcher=None, commit_by="model"):
    """Generate after ``input_ids`` (1, prompt_len). Returns the generated ids (1, n).

    The output stops at (and includes) the first ``eos_id``, or is
    ``gen_length`` tokens long.
    """
    commit_by = resolve_commit_by(commit_by)
    device = model.device
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]
    num_blocks = (prompt_len + gen_length + block_length - 1) // block_length
    total_len = num_blocks * block_length
    max_steps = max_steps_per_block or block_length + 1

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    attn_mask = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)[None, None]
        .log()
        .to(model.dtype)
    )
    position_ids = torch.arange(total_len, device=device)[None]
    x = torch.full((1, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    for block in range(prompt_len // block_length, num_blocks):
        start, end = block * block_length, (block + 1) * block_length
        gen_from = max(prompt_len - start, 0)  # first generated position in the block
        for _ in range(max_steps):
            active = x[0, start:end] == mask_id
            if not active.any():
                break
            logits = model(
                x[:, :end], attention_mask=attn_mask[:, :, :end, :end],
                position_ids=position_ids[:, :end],
            ).logits[0, start:end].float()  # (block_length, V)

            x0 = x[0, start:end].clone()
            conf = torch.full((block_length,), -torch.inf, device=device)
            if matcher is None:
                probs = F.softmax(logits / temperature if temperature > 0 else logits, dim=-1)
                if temperature > 0:
                    tok = torch.multinomial(probs, 1).squeeze(-1)
                else:
                    tok = probs.argmax(dim=-1)
                x0 = tok
                conf = probs.gather(-1, tok[:, None]).squeeze(-1)
            else:
                x0_gen, marginal = matcher.propose_x0(
                    logits[gen_from:], x[0, start + gen_from:end], mask_id,
                    temperature=temperature, return_marginal=commit_by == "constrained",
                )
                x0[gen_from:] = x0_gen
                if marginal is None:
                    marginal = F.softmax(logits[gen_from:], dim=-1).gather(-1, x0_gen[:, None]).squeeze(-1)
                conf[gen_from:] = marginal.float()

            conf = torch.where(active, conf, torch.full_like(conf, -torch.inf))
            commit = conf > threshold
            if not commit.any():
                commit[conf.argmax()] = True
            commit &= active
            x[0, start:end] = torch.where(commit, x0, x[0, start:end])

        if matcher is not None:
            assert matcher.accept_tokens(x[0, start + gen_from:end]), "constraint rejected a block"
        if eos_id is not None and (x[0, prompt_len:end] == eos_id).any():
            break

    gen = x[0, prompt_len:prompt_len + gen_length]
    eos = (gen == eos_id).nonzero() if eos_id is not None else []
    n = int(eos[0]) + 1 if len(eos) else gen_length
    return gen[:n][None]
