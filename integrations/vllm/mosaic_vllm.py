"""vLLM plugin: DiffusionGemma decoding under a JSON-Schema constraint.

The repo's vLLM env installs it (``UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync
--extra vllm``; see integrations/vllm/README.md) as a ``vllm.general_plugins``
entry point, so vLLM loads it in every process. Point ``MOSAIC_PATH`` at the mosaic
checkout:

    export MOSAIC_PATH=$PWD

A request that carries ``extra_args={"json_schema": <schema>}`` (``vllm_xargs``
on the OpenAI server; a JSON string also works) is decoded under the schema.
Other requests decode exactly as stock vLLM, in the same batches.

A thinking request adds ``"reasoning": true`` (the model's own markers, Gemma's
``<|channel>thought\n`` ... ``<channel|>``) or ``"reasoning": [open, close]``: it
thinks freely, then answers under the schema, in one request. The thinking is
the model's own: its canvases go through vLLM's step with only special tokens
blocked (the open marker is allowed at the very start, the close marker
anywhere), so it cannot end its turn before answering. When a thinking canvas
converges holding the close marker, it is not committed: everything up to and
including the marker is pinned, the rest is redrawn, and the canvas is denoised
again with the schema's constraint on the positions after the marker; the
constraint then holds until the answer ends. A thinking
that has not closed within ``max_thinking_tokens`` (an extra arg; by default the
request's max_tokens less one canvas) is closed: its next canvas starts with
the close marker. The answer must end within ``max_answer_tokens`` after the
close marker (default one canvas). When vLLM preempts a request to free KV cache,
it later adds it back as a new request holding what it had written; the plugin
picks up from those tokens (they count toward the thinking limit, and an answer
already begun continues under the constraint).

DiffusionGemma denoises a canvas by redrawing every position at every step,
keeping the lowest-entropy positions up to an entropy bound and re-randomizing
the rest, until the canvas is stable and confident; the canvas is then
committed and the next one starts. For a constrained request, each step of a
canvas uses mosaic's matcher API (``mosaic/matcher.py``, the API the SGLang
integration uses):

- the x0 draw is joint over the canvas, from (model at the scheduled
  temperature) x (schema) (``BatchConstraintMatcher.batch_propose_x0``),
  starting where the committed canvases left the automaton;
- the entropy bound, the confidence and stability tests and the
  self-conditioning read the exact constrained marginals
  (``BatchConstraintMatcher.batch_marginal_log``) instead of the model's;
- a converged canvas is committed as a temperature-0 proposal from those
  marginals (a valid string, where the per-position argmax need not be), and
  fed to the matcher when it commits, so an output longer than a canvas
  continues under the constraint.

Everything else (the temperature schedule, the acceptance rule, the stop test,
the commit cycle) is vLLM's own step, which constrained requests share with
the rest of the batch.

Settings (environment variables, read in each process):
  MOSAIC_MATMUL_DTYPE  float64 (default) or float32: the parallel sampler's products;
                       fp32 underflows on DiffusionGemma's peaked logits
  MOSAIC_CACHE_DIR     compiled automata on disk, reused across processes and runs
  MOSAIC_CACHE_SIZE    compiled schemas kept on each GPU (default 8)
  MOSAIC_ROWS_PER_CALL constrained canvases per step call (default 2): each holds a few
                       (canvas, vocabulary) fp32 tensors outside vLLM's memory budget
  MOSAIC_TILE_MEMORY_GB the free memory DiffusionGemma's sampler is told it has (default 6),
                       the same on every worker (see _SameFreeMemory); it sizes the
                       sampler's tiles of requests (6 GiB: 2 canvases of 128 per tile)
  MOSAIC_COMMIT_RAW    1 (default): a committed canvas is encoded without the
                       self-conditioning block, as in HF; 0: vLLM's own
  MOSAIC_ANCHOR_WINDOW 1 (default): a denoising canvas's sliding window is anchored
                       at the canvas start, as in HF; 0: vLLM's own
  MOSAIC_EOT_TOKEN, MOSAIC_EOS_TOKEN
                       the end-of-turn token after the value and the padding after
                       it (default for Gemma chat templates: <turn|>, <pad>)

Tested with vLLM 0.29.1rc1.dev551+g1b3b88ec2 and google/diffusiongemma-26B-A4B-it,
on one GPU (part of the weights in host memory) and tensor parallel 2.
"""

import json
import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

#: The ``extra_args`` key a request passes its schema under.
SCHEMA_KEY = "json_schema"
#: The ``extra_args`` key of a thinking request's markers: true (the model's own) or [open, close].
REASONING_KEY = "reasoning"
#: The ``extra_args`` key bounding a thinking request's thinking, in tokens.
MAX_THINKING_KEY = "max_thinking_tokens"
#: The ``extra_args`` key bounding a thinking request's answer after the close marker, in tokens.
MAX_ANSWER_KEY = "max_answer_tokens"
#: DiffusionGemma's (Gemma's) thinking markers.
GEMMA_REASONING = ("<|channel>thought\n", "<channel|>")
#: A free canvas position, in the matcher API's mask convention.
FREE = -1

#: canvas slot -> ConstraintMatcher, in this process
_MATCHERS = {}
#: canvas slot -> _Thinking, for thinking requests
_THINKING = {}
#: canvas slot -> the tokens a request had written before vLLM preempted it, while it is added back there
_WRITTEN = {}
_compiler = None
_tokenizer = None  # (name, revision) of the served model's tokenizer
_stock_step = None


def register():
    """The plugin entry point: patch DiffusionGemma's sampler in this process."""
    mosaic = os.environ.get("MOSAIC_PATH")
    if mosaic and mosaic not in sys.path:
        sys.path.insert(0, mosaic)
    from vllm.model_executor.models import diffusion_gemma as dg

    if getattr(dg, "_mosaic", False):
        return
    dg._mosaic = True
    if os.environ.get("MOSAIC_ANCHOR_WINDOW", "1") != "0":
        _anchor_canvas_window()
    if os.environ.get("MOSAIC_COMMIT_RAW", "1") != "0":
        _commit_without_self_conditioning(dg)
    tile_memory = float(os.environ.get("MOSAIC_TILE_MEMORY_GB", 6)) * 2**30
    dg.current_platform = _SameFreeMemory(dg.current_platform, int(tile_memory))
    global _stock_step
    _stock_step = dg._compiled_sample_step
    dg._compiled_sample_step = _sample_step

    custom_sampler = dg.DiffusionGemmaModelState.custom_sampler
    model_add_request = dg.DiffusionGemmaModelState.add_request
    add_request = dg.DiffusionSampler.add_request
    remove_request = dg.DiffusionGemmaRequestStates.remove_request

    def _custom_sampler(self, sampler):
        global _tokenizer
        model_config = self.vllm_config.model_config
        _tokenizer = (model_config.tokenizer, model_config.tokenizer_revision)
        return custom_sampler(self, sampler)

    def _model_add_request(self, req_index, new_req_data):
        # vLLM preempts a request (to free KV cache) by removing it, and resumes it as a new request whose
        # tokens are its prompt and what it had written; the sampler's add_request, next, picks them up
        model_add_request(self, req_index, new_req_data)
        written = list((new_req_data.prefill_token_ids or [])[new_req_data.prompt_len:])
        if written:
            _WRITTEN[req_index] = written
        else:
            _WRITTEN.pop(req_index, None)

    def _add_request(self, req_idx, sampling_params):
        add_request(self, req_idx, sampling_params)
        _MATCHERS.pop(req_idx, None)
        _THINKING.pop(req_idx, None)
        written = _WRITTEN.pop(req_idx, None)
        extra = getattr(sampling_params, "extra_args", None) or {}
        schema = extra.get(SCHEMA_KEY)
        if schema is None:
            return
        try:
            compiler = _get_compiler(self.embed_weight.device)
            reasoning = _reasoning_markers(extra.get(REASONING_KEY), compiler.tokenizer)
            if reasoning is not None:  # the constraint is loaded when the thinking closes (see _close_thinking)
                markers = _marker_ids(reasoning, compiler.tokenizer)
                if markers is None:
                    logger.error("mosaic: the close marker %r is not one token; the request decodes unconstrained",
                                 reasoning[1])
                    return
                limits = [extra.get(MAX_THINKING_KEY), extra.get(MAX_ANSWER_KEY)]
                _THINKING[req_idx] = _Thinking(schema, *markers, sampling_params.max_tokens,
                                               *(None if v is None else int(v) for v in limits))
                if written:
                    _resume_thinking(req_idx, written, self.canvas_length)
                return
            compiled = compiler.compile_json_schema(schema)
        except torch.OutOfMemoryError:
            raise
        except Exception as e:  # an unsupported schema decodes unconstrained, not a crashed engine
            logger.error("mosaic: cannot compile a request's schema, decoding it unconstrained: %s", e)
            return
        from mosaic.matcher import ConstraintMatcher

        _MATCHERS[req_idx] = ConstraintMatcher(compiled, max_new_tokens=sampling_params.max_tokens)
        if written and not _MATCHERS[req_idx].accept_tokens(written):
            logger.warning("mosaic: a resumed request's output is rejected by its constraint (slot %d)", req_idx)

    def _remove_request(self, slot_idx):
        remove_request(self, slot_idx)
        _MATCHERS.pop(slot_idx, None)
        _THINKING.pop(slot_idx, None)
        _WRITTEN.pop(slot_idx, None)

    dg.DiffusionGemmaModelState.custom_sampler = _custom_sampler
    dg.DiffusionGemmaModelState.add_request = _model_add_request
    dg.DiffusionSampler.add_request = _add_request
    dg.DiffusionGemmaRequestStates.remove_request = _remove_request


# ---- matching HF DiffusionGemma (every request) ----


#: (module, [(old, new), ...]): each ``old`` must occur exactly once.
_WINDOW_EDITS = [
    ("triton_attention_helpers", [(
        "        q_abs = context_len + qpos_lo\n",
        "        q_abs = context_len + qpos_lo\n"
        "        if USE_PER_SEQ_CAUSAL or (not USE_CAUSAL):  # mosaic: a canvas's window starts at the canvas\n"
        "            q_abs = context_len\n",
    )]),
    ("triton_unified_attention", [
        ("        query_abs_pos = context_len + query_pos[:, None]\n",
         "        query_abs_pos = context_len + query_pos[:, None]\n"
         "        if USE_PER_SEQ_CAUSAL:  # mosaic: a bidirectional canvas's window is anchored at its start\n"
         "            query_abs_pos = tl.where(tl.load(per_seq_causal_ptr + seq_idx), query_abs_pos,\n"
         "                                     context_len + 0 * query_pos[:, None])\n"
         "        elif not USE_CAUSAL:\n"
         "            query_abs_pos = context_len + 0 * query_pos[:, None]\n"),
        ("            qpos_lo = q_block_local_idx * BLOCK_Q\n",
         "            qpos_lo = q_block_local_idx * BLOCK_Q\n"
         "            if USE_PER_SEQ_CAUSAL:  # mosaic: as above\n"
         "                qpos_lo = tl.where(tl.load(per_seq_causal_ptr + seq_idx), qpos_lo, 0)\n"
         "            elif not USE_CAUSAL:\n"
         "                qpos_lo = 0\n"),
        ("from vllm.v1.attention.ops.triton_attention_helpers import (",
         "from mosaic_triton_attention_helpers import ("),
    ]),
]


def _anchor_canvas_window():
    """Make a bidirectional (denoising) canvas's sliding window HF DiffusionGemma's.

    In HF's DiffusionGemma (the release), every canvas query in a
    sliding-window layer sees the last ``sliding_window - 1`` context tokens
    before the canvas, plus the whole canvas: the window is anchored at the
    canvas start. vLLM's Triton attention centres the window on each query
    position, so canvas position j loses the oldest j context tokens (up to
    127 of 1023), which changes the model's predictions on a canvas the model
    itself wrote. Causal attention (prefill, commits) is
    unchanged. vLLM's attention modules are copied with the edits into a
    temporary directory (Triton reads kernel source from files) and the
    Triton backend is pointed at the copy.
    """
    import importlib
    import tempfile

    from vllm.v1.attention.ops import triton_attention_helpers

    ops_dir = os.path.dirname(triton_attention_helpers.__file__)
    out_dir = tempfile.mkdtemp(prefix="mosaic-triton-")
    for name, edits in _WINDOW_EDITS:
        with open(os.path.join(ops_dir, name + ".py")) as f:
            src = f.read()
        for old, new in edits:
            if src.count(old) != 1:
                raise RuntimeError(f"mosaic: cannot anchor the canvas window: {name}.py has changed ({old.strip()!r})")
            src = src.replace(old, new)
        with open(os.path.join(out_dir, "mosaic_" + name + ".py"), "w") as f:
            f.write(src)
    sys.path.insert(0, out_dir)
    patched = importlib.import_module("mosaic_triton_unified_attention")
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.attention.ops import triton_unified_attention

    triton_attn.unified_attention = patched.unified_attention
    triton_unified_attention.unified_attention = patched.unified_attention


def _commit_without_self_conditioning(dg):
    """Encode a committed canvas as HF DiffusionGemma does: raw embeddings, no self-conditioning block.

    In HF a finished canvas is encoded by the encoder (the causal text model):
    its scaled embeddings go straight into the layers. vLLM encodes it with a
    commit forward of the same model, and applies the self-conditioning block to
    every request that has logits, commit steps included: with a zero signal
    the block reduces to ``post_norm``, which rescales the committed tokens'
    embeddings to unit RMS (from ~1.8) and so changes the keys and values every
    later canvas attends to. Commit steps skip the block here.
    """
    apply_sc = dg.DiffusionGemmaModelState._apply_self_conditioning

    def _apply_self_conditioning(self, decode_slots_np, decode_idx_np, query_start_loc_np, inputs_embeds,
                                 sc_embeds):
        if len(decode_slots_np):
            phase = self.diffusion_states.is_encoder_phase
            committing = phase[torch.as_tensor(decode_slots_np, device=phase.device)].cpu().numpy()
            decode_slots_np, decode_idx_np = decode_slots_np[~committing], decode_idx_np[~committing]
        return apply_sc(self, decode_slots_np, decode_idx_np, query_start_loc_np, inputs_embeds, sc_embeds)

    dg.DiffusionGemmaModelState._apply_self_conditioning = _apply_self_conditioning


class _SameFreeMemory:
    """``current_platform`` for DiffusionGemma's sampler, reporting the same free memory on every worker.

    The sampler sizes its tiles of requests from the GPU's free memory, and every
    tile makes one self-conditioning all-reduce. Under tensor parallelism the
    workers' free memory differs (other processes, allocator state), so they can
    split a batch into different tiles and pair one worker's all-reduce with
    another's for other requests: the self-conditioning then mixes requests and
    the model's output turns to garbage (stock requests included). A fixed
    figure makes the tiling identical everywhere.
    """

    def __init__(self, platform, free_bytes):
        self._platform = platform
        self._free = free_bytes

    def mem_get_info(self):
        return self._free, self._platform.mem_get_info()[1]

    def __getattr__(self, name):
        return getattr(self._platform, name)


# ---- requests: the compiler, thinking requests ----


def _get_compiler(device):
    global _compiler
    if _compiler is None:
        from transformers import AutoTokenizer

        from mosaic.matcher import ConstraintCompiler

        name, revision = _tokenizer
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=True)
        gemma = "<turn|>" in tokenizer.get_vocab()
        eot = os.environ.get("MOSAIC_EOT_TOKEN") or ("<turn|>" if gemma else None)
        eos = os.environ.get("MOSAIC_EOS_TOKEN") or ("<pad>" if gemma else None)
        fp32 = os.environ.get("MOSAIC_MATMUL_DTYPE", "float64").lower() in ("float32", "fp32")
        _compiler = ConstraintCompiler(
            tokenizer, eos_token=eos, eot_token=eot, device=device,
            matmul_dtype=None if fp32 else torch.float64, cache_dir=os.environ.get("MOSAIC_CACHE_DIR"),
            cache_size=int(os.environ.get("MOSAIC_CACHE_SIZE", 8)),
        )
    return _compiler


def _reasoning_markers(value, tokenizer):
    """A request's ``reasoning`` extra arg as an (open, close) pair, or None.

    true: the model's own markers (Gemma's); a pair of strings: those. A JSON
    string of either also works (``vllm_xargs`` values are safest as strings).
    """
    if isinstance(value, str):
        value = json.loads(value)
    if value is None or value is False:
        return None
    if value is True:
        if GEMMA_REASONING[1] not in tokenizer.get_vocab():
            raise ValueError("reasoning=true needs a model whose thinking markers are known (Gemma's); "
                             "pass [open, close] instead")
        return GEMMA_REASONING
    open_text, close_text = value
    return str(open_text), str(close_text)


def _marker_ids(reasoning, tokenizer):
    """(the open marker's first token, the close marker's token), or None if the close marker is not one token."""
    opening = tokenizer.encode(reasoning[0], add_special_tokens=False)
    close = tokenizer.encode(reasoning[1], add_special_tokens=False)
    return (opening[0], close[0]) if len(close) == 1 else None


class _Thinking:
    """A thinking request's progress: thinking (vLLM's own step, special tokens blocked) until it closes, then
    answering under the schema's constraint, which is loaded then."""

    def __init__(self, schema, open_id, close_id, max_tokens, max_thinking=None, max_answer=None):
        self.schema, self.open_id, self.close_id = schema, open_id, close_id
        self.max_tokens, self.max_thinking, self.max_answer = max_tokens, max_thinking, max_answer
        self.answering = False
        self.committed = 0  # tokens committed so far
        self.offset = 0  # in the canvas that closed the thinking: where the answer starts

    def thinking_limit(self, canvas_len):
        """Tokens the thinking may take: by default all but one canvas, kept for the answer."""
        if self.max_thinking is not None:
            return self.max_thinking
        return None if self.max_tokens is None else self.max_tokens - canvas_len

    def out_of_tokens(self, canvas_len):
        """Whether the thinking must close in its next canvas: one more thinking canvas would pass the limit."""
        limit = self.thinking_limit(canvas_len)
        return limit is not None and self.committed + canvas_len > limit


# ---- the decode step ----


def _sample_step(logits, decode_slots, decode_idx, all_slots, valid_canvas_len, canvas, argmax_canvas,
                 step_tensor, is_encoder_phase, confident_tensor, sc_embeds, embed_weight, normalizer,
                 history, history_len_tensor, max_steps_tensor, pin_mask, seed_canvas, read_only,
                 sampled, num_sampled, draft_tokens, **config):
    """vLLM's decode step for a tile of requests, constrained rows under their schemas.

    Replaces ``_compiled_sample_step`` (same arguments and return). In order: the
    canvases committing now are fed to their matchers; every row but the
    constrained ones goes through vLLM's own step (a thinking request's with its
    special tokens blocked); a thinking canvas that converged on its close marker
    is started over for its answer; then the constrained rows go through
    :func:`_constrained_step`.
    """
    state = (canvas, argmax_canvas, step_tensor, is_encoder_phase, confident_tensor, sc_embeds,
             embed_weight, normalizer, history, history_len_tensor, max_steps_tensor, pin_mask,
             seed_canvas, read_only, sampled, num_sampled, draft_tokens)
    if not _MATCHERS and not _THINKING:
        return _stock_step(logits, decode_slots, decode_idx, all_slots, valid_canvas_len, *state, **config)
    slots = decode_slots.tolist()
    ours = [i for i, s in enumerate(slots) if s in _MATCHERS or s in _THINKING]
    if not ours:
        return _stock_step(logits, decode_slots, decode_idx, all_slots, valid_canvas_len, *state, **config)

    n, W = len(slots), config["CL"]
    commit = is_encoder_phase[decode_slots].tolist()
    valid_len = valid_canvas_len.tolist()
    out_of_thinking = _on_commits([i for i in ours if commit[i]], slots, argmax_canvas, valid_len, pin_mask, W)
    thinking = {i for i in ours if not commit[i] and slots[i] in _THINKING and not _THINKING[slots[i]].answering}
    # only a request resumed after a preemption can be out of thinking tokens here (_resume_thinking)
    out_of_thinking += [i for i in thinking if _THINKING[slots[i]].out_of_tokens(W)]
    ours = [i for i in ours if not commit[i] and i not in thinking]
    if not ours and not thinking and not out_of_thinking:
        return _stock_step(logits, decode_slots, decode_idx, all_slots, valid_canvas_len, *state, **config)

    logits = logits.view(n, W, -1)
    scaled = torch.empty(n, W, logits.shape[-1], device=logits.device, dtype=torch.float32)
    others = [i for i in range(n) if i not in set(ours)]
    if others:
        idx = torch.tensor(others, device=logits.device)
        rows = logits[idx]  # a copy
        for j, i in enumerate(others):
            if i in thinking:  # the model's own thinking, without the special tokens that would end it early
                _block_special_tokens(rows[j], _THINKING[slots[i]])
        scaled[idx] = _stock_step(rows.reshape(len(others) * W, -1), decode_slots[idx], decode_idx[idx],
                                  all_slots, valid_canvas_len[idx], *state, **config).view(len(others), W, -1)
    switch = dict(slots=slots, W=W, vocab_size=config["vocab_size"], canvas=canvas, step_tensor=step_tensor,
                  is_encoder_phase=is_encoder_phase, confident_tensor=confident_tensor, sc_embeds=sc_embeds,
                  history_len_tensor=history_len_tensor, pin_mask=pin_mask, seed_canvas=seed_canvas,
                  draft_tokens=draft_tokens)
    _switch_closed_thinking(thinking, out_of_thinking, slots, argmax_canvas, is_encoder_phase, valid_len, switch)
    if not ours:
        return scaled
    # a few rows per call: the step holds several (rows, CL, vocab) tensors outside vLLM's memory budget
    per_call = int(os.environ.get("MOSAIC_ROWS_PER_CALL", 2))
    for c in range(0, len(ours), per_call):
        chunk = ours[c:c + per_call]
        idx = torch.tensor(chunk, device=logits.device)
        offsets = [_THINKING[slots[i]].offset if slots[i] in _THINKING else 0 for i in chunk]
        scaled[idx] = _constrained_step([_MATCHERS[slots[i]] for i in chunk], offsets, logits[idx], decode_slots[idx],
                                        decode_idx[idx], all_slots, valid_canvas_len[idx], *state, **config)
    return scaled


def _on_commits(rows, slots, argmax_canvas, valid_len, pin_mask, W):
    """The rows whose canvas commits this step (it converged last step), before vLLM's step encodes it.

    An answer canvas is fed to its matcher (from where the answer starts, in the
    canvas that closed the thinking); a thinking canvas is only counted. Returns
    the thinking requests that reach their thinking limit with this commit.
    """
    out_of_thinking = []
    for i in rows:
        s, block = slots[i], argmax_canvas[slots[i], : valid_len[i]]
        t, m = _THINKING.get(s), _MATCHERS.get(s)
        phase = "thinking" if t is not None and not t.answering else "answer"
        if phase == "thinking":
            t.committed += valid_len[i]
            if t.out_of_tokens(W):
                out_of_thinking.append(i)
        else:
            answer = block[t.offset:] if t is not None else block
            if answer.numel() and not m.accept_tokens(answer):
                logger.warning("mosaic: a committed canvas is rejected by its constraint (slot %d)", s)
            if t is not None:
                t.committed += valid_len[i]
                t.offset = 0
                pin_mask[s] = False  # the pinned thinking belonged to the canvas that closed it
    return out_of_thinking


def _switch_closed_thinking(thinking, out_of_thinking, slots, argmax_canvas, is_encoder_phase, valid_len, switch):
    """After vLLM's step: start each thinking canvas that converged holding the close marker over for its
    answer (its thinking through the marker pinned, the rest under the constraint), and each thinking out of
    tokens with its next canvas closed."""
    for i in thinking:
        if i not in out_of_thinking and bool(is_encoder_phase[slots[i]]):
            t = _THINKING[slots[i]]
            block = argmax_canvas[slots[i], : valid_len[i]].tolist()
            if t.close_id in block:
                prefix = block[: block.index(t.close_id) + 1]
                # ending with the marker, the canvas is all thinking: it commits as it is, the answer starts next
                _close_thinking(i, prefix, redraw=len(prefix) < len(block), **switch)
    for i in out_of_thinking:
        _close_thinking(i, [_THINKING[slots[i]].close_id], **switch)


# ---- thinking canvases ----


#: A blocked token's logit: far below any real one, so its probability is exactly 0 after the softmax. Not
#: -inf: vLLM's step computes each position's entropy as -sum(p log p), and 0 * -inf is NaN, which would keep
#: every canvas from ever counting as confident (it would run to the step cap).
_BLOCKED = -1e9


def _block_special_tokens(rows, thinking):
    """Mask a thinking canvas's logits (CL, V) in place: no special token but the close marker, and the open
    marker only at the very start of the output."""
    kept = rows[:, [thinking.close_id, thinking.open_id]].clone()
    rows[:, _special_ids(rows.device)] = _BLOCKED
    rows[:, thinking.close_id] = kept[:, 0]
    if thinking.committed == 0:
        rows[0, thinking.open_id] = kept[0, 1]


def _close_thinking(i, prefix, slots, W, vocab_size, canvas, step_tensor, is_encoder_phase, confident_tensor,
                    sc_embeds, history_len_tensor, pin_mask, seed_canvas, draft_tokens, redraw=True):
    """Start row ``i``'s canvas over with ``prefix`` (its thinking through the close marker) pinned and the
    rest free, and its request answering under the schema's constraint from right after the prefix.
    redraw=False: the canvas is all thinking and commits as it is; the answer starts with the next one."""
    s = slots[i]
    t = _THINKING[s]
    k = len(prefix)
    if not _start_answer(s, t.committed + k, W):
        return
    t.offset = k
    if not redraw:
        return
    pin_mask[s] = False
    pin_mask[s, :k] = True
    seed_canvas[s, :k] = torch.tensor(prefix, device=seed_canvas.device, dtype=seed_canvas.dtype)
    fresh = torch.randint(0, vocab_size, (W,), device=canvas.device, dtype=canvas.dtype)
    fresh = torch.where(pin_mask[s, :W], seed_canvas[s, :W].to(fresh.dtype), fresh)
    canvas[s] = fresh
    draft_tokens[s, :W] = fresh.to(draft_tokens.dtype)
    is_encoder_phase[s] = False
    step_tensor[s] = 0
    history_len_tensor[s] = 0
    confident_tensor[s] = False
    sc_embeds[s] = 0


def _start_answer(s, thought, W):
    """Slot ``s``'s thinking took ``thought`` tokens, through the close marker: its answer goes under the
    schema's constraint from here. False if the schema cannot be compiled (the request answers unconstrained)."""
    t = _THINKING[s]
    from mosaic.matcher import ConstraintMatcher

    try:
        compiled = _compiler.compile_json_schema(t.schema)
    except torch.OutOfMemoryError:
        raise
    except Exception as e:  # an unsupported schema: the request answers unconstrained
        logger.error("mosaic: cannot compile a request's schema, answering unconstrained: %s", e)
        _THINKING.pop(s, None)
        return False
    # the answer must end within max_answer_tokens (by default one canvas), or the whitespace the value may
    # start or end with could put it off indefinitely
    budget = t.max_answer if t.max_answer is not None else W
    if t.max_tokens is not None:
        budget = min(budget, t.max_tokens - thought)
    _MATCHERS[s] = ConstraintMatcher(compiled, max_new_tokens=budget)
    t.answering = True
    return True


def _resume_thinking(s, written, W):
    """A thinking request vLLM preempted, added back in slot ``s`` with the tokens it had ``written``: count them
    toward its thinking limit (the step closes the thinking if they reach it), and if the thinking had closed,
    continue the answer under the constraint after what it had written of it."""
    t = _THINKING[s]
    t.committed = len(written)
    if t.close_id not in written:
        return
    k = written.index(t.close_id) + 1
    if _start_answer(s, k, W) and k < len(written) and not _MATCHERS[s].accept_tokens(written[k:]):
        logger.warning("mosaic: a resumed request's answer is rejected by its constraint (slot %d)", s)


_SPECIAL_IDS = {}


def _special_ids(device):
    """The tokenizer's special tokens (EOS and EOT included) as ids on ``device``: never inside a thinking."""
    if device not in _SPECIAL_IDS:
        c = _compiler
        names = list(c.banned_tokens) + [c.eos_token, c.eot_token]
        ids = sorted({c.tokenizer.convert_tokens_to_ids(t) for t in names})
        _SPECIAL_IDS[device] = torch.tensor(ids, device=device)
    return _SPECIAL_IDS[device]


# ---- the constrained step ----


@torch.no_grad()
def _constrained_step(matchers, offsets, logits, decode_slots, decode_idx, all_slots, valid_canvas_len, canvas,
                      argmax_canvas, step_tensor, is_encoder_phase, confident_tensor, sc_embeds, embed_weight,
                      normalizer, history, history_len_tensor, max_steps_tensor, pin_mask, seed_canvas,
                      read_only, sampled, num_sampled, draft_tokens, *, max_denoising_steps, t_min, t_max,
                      confidence_threshold, vocab_size, CL, ST, entropy_bound, sc_vocab_start, sc_vocab_end,
                      tp_size, tp_group_name, compute_sc=True):
    """vLLM's ``_compiled_sample_step`` for denoising rows, with the model's distribution constrained.

    Follows vLLM's phases one for one (``vllm/model_executor/models/diffusion_gemma.py``
    at 1b3b88ec2); what differs is marked "constrained". logits: (n, CL, V). offsets: per row,
    where the constraint starts in the canvas (the positions before hold pinned thinking).
    Returns the constrained log-marginals, in place of the tempered logits.
    """
    n = decode_slots.shape[0]
    device = decode_slots.device

    # ---- Phase 1: Temperature schedule ----
    steps_f = step_tensor[decode_slots].float()
    remaining = (max_denoising_steps - steps_f).clamp(min=1.0)
    temp = t_min + (t_max - t_min) * (remaining / max_denoising_steps)
    scaled = logits.float() / temp[:, None, None].clamp(min=1e-10)

    # ---- Phases 2-3, constrained: a joint draw, and the exact marginals ----
    pins = pin_mask[decode_slots]
    seeds = seed_canvas[decode_slots]
    tokens = torch.where(pins, seeds, torch.full_like(seeds, FREE))
    from mosaic.matcher import BatchConstraintMatcher

    # one sampler call per schema (not one shared call), the draws the plugin was validated with
    batch = BatchConstraintMatcher(across_constraints=False)
    # Under tensor parallelism every worker holds its own copy of each canvas and computes this step
    # from the same logits and random stream, with the same result
    new_tokens = _propose(batch, matchers, offsets, scaled, tokens, temperature=1.0)
    log_probs = _marginals(batch, matchers, offsets, scaled, tokens)
    probs = log_probs.exp()
    argmax_tokens = log_probs.argmax(dim=-1)

    token_entropy = -torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs)).sum(dim=-1)
    mean_entropy = token_entropy.mean(dim=-1)
    confident_tensor[decode_slots] = mean_entropy < confidence_threshold

    # ---- Phase 4: Entropy-bound acceptance mask ----
    sorted_ent, sorted_idx = torch.sort(token_entropy, dim=-1)
    cumsum_ent = torch.cumsum(sorted_ent, dim=-1)
    cummax_ent = torch.cummax(sorted_ent, dim=-1).values
    sorted_mask = (cumsum_ent - cummax_ent) <= entropy_bound
    eb_mask = torch.zeros_like(sorted_mask)
    eb_mask.scatter_(1, sorted_idx, sorted_mask)

    # ---- Phase 5: Post-sample (every row denoises) ----
    step_tensor[decode_slots] = (steps_f + 1).to(step_tensor.dtype)
    random_tokens = torch.randint(0, vocab_size, (n, CL), device=device, dtype=canvas.dtype)
    denoise_canvas = torch.where(eb_mask, new_tokens.to(canvas.dtype), random_tokens)
    canvas[decode_slots] = torch.where(pins, seeds, denoise_canvas)

    hist_len = history_len_tensor[decode_slots]
    write_pos = hist_len % ST
    for i in range(ST):
        write_here = (write_pos == i).unsqueeze(1)
        history[decode_slots, i] = torch.where(write_here, argmax_tokens, history[decode_slots, i])
    argmax_canvas[decode_slots] = argmax_tokens.to(argmax_canvas.dtype)
    new_hist_len = hist_len + 1
    history_len_tensor[decode_slots] = new_hist_len

    # ---- Phase 6: Stability + convergence ----
    ref = history[decode_slots, 0]
    mismatch = torch.zeros(n, device=device, dtype=torch.int32)
    for h in range(1, ST):
        mismatch = mismatch + (ref != history[decode_slots, h]).sum(dim=-1).int()
    stable = mismatch == 0
    converged = (stable & confident_tensor[decode_slots] & (new_hist_len >= ST)) | (
        step_tensor[decode_slots] >= max_steps_tensor[decode_slots]
    )
    is_encoder_phase[decode_slots] = converged

    # constrained: a converged canvas commits as a valid string
    done = [i for i, c in enumerate(converged.tolist()) if c]
    if done:
        rows = torch.tensor(done, device=device)
        valid = _propose(batch, [matchers[i] for i in done], [offsets[i] for i in done], log_probs[rows],
                         tokens[rows], temperature=0.0)
        argmax_canvas[decode_slots[rows]] = valid.to(argmax_canvas.dtype)

    emit = converged & read_only[decode_slots]
    sampled[decode_idx] = argmax_canvas[decode_slots].to(sampled.dtype) * emit[:, None]
    num_sampled[decode_idx] = (emit * valid_canvas_len).to(num_sampled.dtype)

    # self-conditioning from the constrained marginals
    sc_keep = (~is_encoder_phase[decode_slots])[:, None, None]
    if compute_sc:
        local_probs = probs[..., sc_vocab_start:sc_vocab_end].to(embed_weight.dtype)
        soft_embeds = torch.matmul(local_probs, embed_weight[: sc_vocab_end - sc_vocab_start])
        if tp_size > 1:
            soft_embeds = torch.ops.vllm.all_reduce(soft_embeds, group_name=tp_group_name)
        soft_embeds = soft_embeds * normalizer
        sc_pin = (~pins).unsqueeze(-1)
        sc_embeds[decode_slots] = (soft_embeds * sc_keep * sc_pin).to(sc_embeds.dtype)
    else:
        sc_embeds[decode_slots] = 0

    newly_converged = converged.unsqueeze(1)
    canvas[decode_slots] = torch.where(newly_converged, argmax_canvas[decode_slots], canvas[decode_slots])
    is_encoder_phase[decode_slots] &= ~read_only[decode_slots]

    # ---- Phase 7: Copy canvas → draft_tokens for all slots ----
    draft_tokens[all_slots, :CL] = canvas[all_slots]
    return log_probs


def _offset_groups(offsets):
    """{offset: rows with it}."""
    groups = {}
    for r, k in enumerate(offsets):
        groups.setdefault(k, []).append(r)
    return groups


def _propose(batch, matchers, offsets, scores, tokens, temperature):
    """``batch_propose_x0`` with each row's constraint from its offset on; earlier positions keep their tokens."""
    x0 = tokens.clone()
    for k, rows in _offset_groups(offsets).items():
        if k >= tokens.shape[1]:
            continue
        idx = torch.tensor(rows, device=tokens.device)
        x, _ = batch.batch_propose_x0([matchers[r] for r in rows], scores[idx, k:], tokens[idx, k:], FREE,
                                      temperature=temperature, return_marginal=False)
        x0[idx, k:] = x.to(x0.dtype)
    return x0


def _marginals(batch, matchers, offsets, scores, tokens):
    """``batch_marginal_log`` with each row's constraint from its offset on; earlier positions are a point mass
    on their (pinned) tokens."""
    out = torch.full(scores.shape, -torch.inf, device=scores.device)
    for k, rows in _offset_groups(offsets).items():
        idx = torch.tensor(rows, device=scores.device)
        if k < tokens.shape[1]:
            out[idx, k:] = batch.batch_marginal_log([matchers[r] for r in rows], scores[idx, k:], tokens[idx, k:],
                                                    FREE)
        if k:
            out[idx, :k] = torch.full_like(out[idx, :k], -torch.inf).scatter_(
                -1, tokens[idx, :k].long().unsqueeze(-1), 0.0)
    return out
