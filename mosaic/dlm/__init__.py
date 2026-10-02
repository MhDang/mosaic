"""Diffusion LMs and their (optionally constrained) denoising loops."""

import torch

from . import dream, llada, llada2
from .constraint import COMMIT_BY_OPTIONS, resolve_commit_by
from .models import MODELS, load_model, load_tokenizer, model_family


def generate(model, family, input_ids, mask_token_id, settings, matcher=None,
             commit_by=None, vocab_size=None):
    """Run ``family``'s denoising loop with the generation ``settings``. Returns the canvas.

    settings (dict-like): ``max_new_tokens``, ``steps``, ``temperature`` and
    ``block_length`` for all families; Dream also reads ``alg``, ``alg_temp``,
    ``top_p``, ``top_k``, ``eps``; LLaDA reads ``remasking``, ``cfg_scale``;
    LLaDA2 reads ``threshold`` and ``eos_id`` (it runs until each block is
    unmasked, so ``steps`` is unused).
    matcher: a :class:`~mosaic.matcher.ConstraintMatcher` for the request, bounded to
        ``max_new_tokens``, or None for unconstrained.
    commit_by: ``constrained`` or ``model`` (see :data:`COMMIT_BY_OPTIONS`); None: the
        loop's default, ``model`` for LLaDA2 and ``constrained`` for Dream and LLaDA.
    """
    loop_args = {} if commit_by is None else {"commit_by": commit_by}
    if family == "dream":
        return dream.generate(
            model, input_ids, mask_token_id,
            steps=settings["steps"],
            max_new_tokens=settings["max_new_tokens"],
            temperature=settings.get("temperature", 0.0),
            top_p=settings.get("top_p"),
            top_k=settings.get("top_k"),
            alg=settings.get("alg", "entropy"),
            alg_temp=settings.get("alg_temp"),
            eps=settings.get("eps", 1e-3),
            block_length=settings.get("block_length"),
            matcher=matcher,
            **loop_args,
        )
    if family == "llada":
        return llada.generate(
            model, input_ids, mask_token_id,
            steps=settings["steps"],
            gen_length=settings["max_new_tokens"],
            block_length=settings.get("block_length", 32),
            temperature=settings.get("temperature", 0.0),
            cfg_scale=settings.get("cfg_scale", 0.0),
            remasking=settings.get("remasking", "low_confidence"),
            vocab_size=vocab_size,
            matcher=matcher,
            **loop_args,
        )
    if family == "llada2":
        assert input_ids.shape[0] == 1, "the LLaDA2 loop decodes one sequence at a time"
        out = llada2.generate(
            model, input_ids, mask_token_id,
            gen_length=settings["max_new_tokens"],
            block_length=settings.get("block_length", 32),
            threshold=settings.get("threshold", 0.95),
            temperature=settings.get("temperature", 0.0),
            eos_id=settings.get("eos_id"),
            matcher=matcher,
            **loop_args,
        )
        return torch.cat([input_ids, out], dim=1)
    raise ValueError(f"Unknown model family {family!r}")


__all__ = [
    "COMMIT_BY_OPTIONS", "MODELS", "generate", "load_model", "load_tokenizer", "model_family",
    "resolve_commit_by",
]
