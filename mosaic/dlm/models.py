"""Loading the diffusion LMs (Dream, LLaDA, LLaDA2) from the Hugging Face Hub."""

import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from . import compat

#: Short names for the supported checkpoints (``benchmarks/bfcl/run_hf.py --model``).
MODELS = {
    "dream": "Dream-org/Dream-v0-Instruct-7B",
    "llada": "GSAI-ML/LLaDA-8B-Instruct",
    "llada2": "inclusionAI/LLaDA2.0-mini",
}

#: The Hub commits the checkpoints load from. Their modeling code runs here
#: (``trust_remote_code``), so a pinned commit keeps an upstream change from
#: changing mosaic's outputs; these are each repository's latest as of 2026-09-30.
REVISIONS = {
    "Dream-org/Dream-v0-Instruct-7B": "05334cb9faaf763692dcf9d8737c642be2b2a6ae",
    "GSAI-ML/LLaDA-8B-Instruct": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
    "inclusionAI/LLaDA2.0-mini": "dad945cac317da394b390f82c7b40691d8a881ed",
}

#: LLaDA2's mask token, when the config does not name one.
LLADA2_MASK_ID = 156895

DTYPES = {
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
    "float32": torch.float32, "fp32": torch.float32,
}


def model_family(path):
    """``dream``, ``llada`` or ``llada2``: which denoising loop a checkpoint uses."""
    p = path.lower()
    if "dream" in p:
        return "dream"
    if "llada2" in p:
        return "llada2"
    if "llada" in p:
        return "llada"
    raise ValueError(f"Cannot tell whether {path!r} is a Dream or a LLaDA model")


def load_model(name_or_path, dtype="bfloat16", device="cuda"):
    """Load a model and its tokenizer. Returns (model, tokenizer, family, mask_token_id).

    All three families ship their modeling code on the Hub (``trust_remote_code``),
    loaded at the commit in ``REVISIONS``; on transformers 5 it is adapted first
    (:mod:`mosaic.dlm.compat`). LLaDA2 loads through ``AutoModelForCausalLM`` (its
    ``AutoModel`` has no LM head). The tokenizer pads on the left.
    """
    path = MODELS.get(name_or_path, name_or_path)
    family = model_family(path)
    revision = REVISIONS.get(path)
    auto = AutoModelForCausalLM if family == "llada2" else AutoModel
    # transformers 4.56 renamed ``torch_dtype`` to ``dtype``
    major, minor = map(int, transformers.__version__.split(".")[:2])
    dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    if major >= 5:
        compat.adapt_hub_code(path, revision)
    model = auto.from_pretrained(path, revision=revision, trust_remote_code=True,
                                 **{dtype_kw: DTYPES[str(dtype)]})
    if major >= 5:
        compat.reset_rotary_buffers(model)
    model = model.to(device).eval()
    tokenizer = load_tokenizer(path)
    mask_token_id = getattr(model.config, "mask_token_id", None)
    if mask_token_id is None and family == "llada2":
        mask_token_id = LLADA2_MASK_ID
    return model, tokenizer, family, mask_token_id


def load_tokenizer(name_or_path):
    """A checkpoint's tokenizer, at the commit in ``REVISIONS``, padding on the left."""
    path = MODELS.get(name_or_path, name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(path, revision=REVISIONS.get(path), trust_remote_code=True)
    tokenizer.padding_side = "left"
    return tokenizer
