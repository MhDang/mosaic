"""Running Dream's, LLaDA's and LLaDA2's Hub code on transformers 5.

The three models ship their modeling code on the Hugging Face Hub, written for
transformers 4. On transformers 5 it either fails to load or, worse, loads with
uninitialized rotary frequencies and generates garbage. :func:`adapt_hub_code`
(before ``from_pretrained``) and :func:`reset_rotary_buffers` (after) fix the
five differences; with them the models generate exactly what they do on
transformers 4.57. :func:`~mosaic.dlm.models.load_model` calls them on transformers
5 only.
"""

import functools
import inspect
import sys
from types import MappingProxyType

import torch
from transformers import AutoConfig, GenerationConfig, PretrainedConfig, modeling_rope_utils
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_utils import PreTrainedModel


def adapt_hub_code(path, revision=None):
    """Import ``path``'s Hub classes and adapt them to transformers 5, before loading:

    - the ``"default"`` rotary embedding, which transformers 5 dropped from
      ``ROPE_INIT_FUNCTIONS``, is put back;
    - a model's ``tie_weights`` and a generation config's ``validate`` ignore
      the keyword arguments transformers 5 added;
    - a model's ``all_tied_weights_keys`` (set by ``post_init``, which these
      models never call; none of them ties weights) and a config's
      ``use_cache`` (no longer kept on model configs) get defaults.
    """
    modeling_rope_utils.ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope_parameters)
    config = AutoConfig.from_pretrained(path, revision=revision, trust_remote_code=True)
    for ref in (getattr(config, "auto_map", None) or {}).values():
        get_class_from_dynamic_module(ref if isinstance(ref, str) else ref[0], path, revision=revision)
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("transformers_modules."):
            continue
        for cls in list(vars(module).values()):
            if not (isinstance(cls, type) and cls.__module__ == module_name):
                continue
            if issubclass(cls, GenerationConfig):
                _ignore_new_kwargs(cls, "validate")
            if issubclass(cls, PreTrainedModel):
                _ignore_new_kwargs(cls, "tie_weights")
                if "all_tied_weights_keys" not in vars(cls):
                    cls.all_tied_weights_keys = MappingProxyType({})  # read-only: loading only reads it
            if issubclass(cls, PretrainedConfig) and not hasattr(cls, "use_cache"):
                cls.use_cache = False  # the denoising loops never cache


@torch.no_grad()
def reset_rotary_buffers(model):
    """Recompute ``model``'s rotary frequencies after loading.

    transformers 5 builds the model on the meta device and fills in only the
    checkpoint's weights, so the ``inv_freq`` buffers the Hub code computes at
    construction stay uninitialized memory.
    """
    for module in model.modules():
        if hasattr(module, "inv_freq") and hasattr(module, "rope_init_fn"):
            inv_freq, attention_scaling = module.rope_init_fn(module.config, module.inv_freq.device)
            module.inv_freq.copy_(inv_freq)
            if torch.is_tensor(getattr(module, "original_inv_freq", None)):
                module.original_inv_freq = module.inv_freq.clone()
            if hasattr(module, "attention_scaling"):
                module.attention_scaling = attention_scaling


# ---- helpers ----


def _default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):
    """transformers 4's ``_compute_default_rope_parameters``: (inv_freq, attention scaling)."""
    base = getattr(config, "rope_theta", None)
    if base is None:
        base = (getattr(config, "rope_parameters", None) or {}).get("rope_theta")
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
    exponent = torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim
    return 1.0 / (base ** exponent), 1.0


def _ignore_new_kwargs(cls, name):
    """Let ``cls.<name>`` (defined in the Hub code itself) drop keyword arguments it does not take."""
    fn = vars(cls).get(name)
    if fn is None or getattr(fn, "_ignores_new_kwargs", False):
        return
    params = inspect.signature(fn).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        return fn(self, *args, **{k: v for k, v in kwargs.items() if k in params})

    wrapper._ignores_new_kwargs = True
    setattr(cls, name, wrapper)
