"""SGLang dLLM algorithm: LowConfidence decoding under a JSON-Schema constraint.

A drop-in file: put it (or a symlink) in ``sglang/srt/dllm/algorithm/`` and
point ``MOSAIC_PATH`` at the mosaic checkout, then launch with

    MOSAIC_PATH=/path/to/mosaic python -m sglang.launch_server \\
        --model-path inclusionAI/LLaDA2.0-mini --trust-remote-code \\
        --dllm-algorithm ConstrainedLowConfidence

Requests that carry a JSON Schema are decoded under it: either OpenAI-style
``response_format`` / ``json_schema`` in the sampling params (SGLang also
compiles it with its grammar backend, which the dLLM path does not use; this
file patches the dLLM scheduler to admit such requests at all), or
``custom_params={"json_schema": ...}`` (skips that). Other requests decode
exactly as ``LowConfidence``.

Each denoising step of a block, a constrained request gets its x0 from
``ConstraintMatcher.propose_x0``: an exact sample of the whole block from
(model) x (schema), starting where the request's accepted tokens left the
automaton and leaving the output completable within ``max_new_tokens``. The
commit rule is LowConfidence's: every masked position whose confidence exceeds
``threshold``, else the single most confident one. The confidence is the
model's probability of the proposed token (``commit_by: model``, default)
or the proposal's exact constrained marginal (``constrained``). Under a threshold
rule the marginal, often near 1 on tokens the schema forces, commits more tokens
per step: fewer forward passes, lower accuracy. When a block is done its tokens
are accepted into the matcher.

Schemas are compiled in the background: as soon as a request is queued, worker
processes compile its schema into a disk cache, and a thread loads it onto the
GPU once the request nears the front of the queue. The request is admitted into
a batch when that is done, so a new schema never stalls the requests already
decoding.

Optional ``--dllm-algorithm-config`` YAML keys: ``threshold`` (0.95),
``commit_by`` (model | constrained), ``temperature`` (0.0), ``sampler``
(parallel | sequential), ``matmul_dtype`` (float32 | float64), ``cache_dir`` (where
compiled schemas are saved and reused; default a temporary directory),
``compile_workers`` (the CPU count minus 4, at most 32; compiling a schema takes
~1 s of one CPU core; the workers start with the engine),
``load_ahead`` (4; queued requests whose constraint is loaded ahead of time),
``batch_constraints`` (true; requests with different schemas share one sampler
call per step), ``cache_size`` (8; compiled schemas kept on the GPU, each an
edges x vocabulary matrix, ~250 MB for a BFCL function).

Works with both of SGLang's dLLM schedules: first-done-first-out (the default;
one step per call, the state carried by the request) and synchronous
(``--no-dllm-fdfo``). Tested with SGLang 0.5.20 and LLaDA2.0-mini.
"""

import logging
import os
import sys
from collections import OrderedDict, defaultdict
from typing import Any, List, Optional

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_mosaic = os.environ.get("MOSAIC_PATH")
if _mosaic and _mosaic not in sys.path:
    sys.path.insert(0, _mosaic)

from mosaic.sampling.parallel import to_device_async, warn_nan_counts  # noqa: E402
from mosaic.matcher import BatchConstraintMatcher, ConstraintCompiler, ConstraintMatcher  # noqa: E402

logger = logging.getLogger(__name__)

#: The ``custom_params`` key a request can pass its schema under.
SCHEMA_KEY = "json_schema"
#: Matchers of finished requests are dropped; at most this many are kept.
MAX_TRACKED = 4096

#: The algorithm instance of this process, for the scheduler hook.
_ACTIVE = []


def request_schema(req) -> Optional[Any]:
    """The JSON Schema a request asks for, or None."""
    sp = req.sampling_params
    if sp.custom_params and sp.custom_params.get(SCHEMA_KEY) is not None:
        return sp.custom_params[SCHEMA_KEY]
    return sp.json_schema


def _expose_batch_requests():
    """Make each batch's requests visible to the algorithm as ``forward_batch.mosaic_reqs``.

    SGLang hands a dLLM algorithm the ForwardBatch, which carries request ids
    but not their sampling params; wrap the worker's dLLM forward to attach
    the ScheduleBatch's requests.
    """
    from sglang.srt.managers.tp_worker import TpModelWorker

    forward = TpModelWorker._forward_batch_generation_dllm
    if getattr(forward, "_mosaic", False):
        return

    def _forward(self, forward_batch, batch=None):
        forward_batch.mosaic_reqs = None if batch is None else list(batch.reqs)
        return forward(self, forward_batch, batch)

    _forward._mosaic = True
    TpModelWorker._forward_batch_generation_dllm = _forward


def _patch_admission():
    """Admit requests into dLLM batches once their constraint is ready.

    Wraps the dLLM scheduler's batch builder to (1) move ``json_schema``
    requests whose SGLang grammar is compiled out of the grammar queue: the
    autoregressive scheduler does this, the dLLM one does not, so they would
    wait forever; (2) start compiling the schema of every queued request in
    the background, and keep a request queued until its constraint is loaded.
    """
    from sglang.srt.managers.scheduler import Scheduler

    get_new_batch = Scheduler.get_new_batch_dllm
    if getattr(get_new_batch, "_mosaic", False):
        return

    def _get_new_batch(self, running_batch):
        if self.grammar_manager.has_waiting_grammars():
            for req in self.grammar_manager.get_ready_grammar_requests():
                self._add_request_to_queue(req)
        algo = _ACTIVE[-1] if _ACTIVE else None
        if algo is not None:
            algo.start(self.server_args)  # once, as the engine starts
        if algo is None or not self.waiting_queue:
            return get_new_batch(self, running_batch)
        algo.prefetch(self.waiting_queue, self.server_args)
        queued = self.waiting_queue
        held = {id(req) for req in queued if not algo.is_ready(req)}
        if not held:
            return get_new_batch(self, running_batch)
        self.waiting_queue = [req for req in queued if id(req) not in held]
        try:
            return get_new_batch(self, running_batch)
        finally:  # put the held requests back, in their places
            left = {id(req) for req in self.waiting_queue}
            was = {id(req) for req in queued}
            self.waiting_queue = [req for req in queued if id(req) in held or id(req) in left] + [
                req for req in self.waiting_queue if id(req) not in was
            ]

    _get_new_batch._mosaic = True
    Scheduler.get_new_batch_dllm = _get_new_batch


def _default_compile_workers():
    """All but four of the CPUs this process may use (SGLang's own processes need
    some), at most 32 (each worker holds torch and the tokenizer in memory)."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 8
    return min(32, max(1, cpus - 4))


class ConstrainedLowConfidence(DllmAlgorithm):
    """LowConfidence whose x0 proposals follow each request's JSON Schema."""

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        cfg = config.algorithm_config
        self.threshold = cfg.get("threshold", 0.95)
        self.commit_by = str(cfg.get("commit_by", "model")).lower()
        self.temperature = cfg.get("temperature", 0.0)
        self.sampler = cfg.get("sampler", "parallel")
        self.matmul_dtype = torch.float64 if str(cfg.get("matmul_dtype", "")) in ("float64", "fp64") else None
        self.cache_dir = cfg.get("cache_dir")
        self.cache_size = cfg.get("cache_size", 8)
        self.compile_workers = cfg.get("compile_workers", _default_compile_workers())
        self.load_ahead = max(1, cfg.get("load_ahead", 4))
        self.batch_constraints = bool(cfg.get("batch_constraints", True))
        self._batch = BatchConstraintMatcher(across_constraints=self.batch_constraints)
        self._tokenizer_path = None
        self._compiler = None
        self._started = False
        self._compiling = {}  # rid -> (req, Future of its CompiledConstraint), loading
        self._precompiling = {}  # rid -> (req, None): schema sent to the disk compile only
        self._matchers = OrderedDict()  # rid -> (req, ConstraintMatcher), across a request's blocks
        _expose_batch_requests()
        _patch_admission()
        _ACTIVE.append(self)

    def run(self, model_runner, forward_batch, algo_states=None):
        if self._tokenizer_path is None:
            self._set_tokenizer_path(model_runner.server_args)
        # The blocks on the host, read before the forward is queued (the GPU is idle
        # here): the step decides everything host-side from them, so that it queues
        # its work behind the forward without waiting for it.
        self._blocks = forward_batch.input_ids.view(forward_batch.batch_size, self.block_size).tolist()
        # Blocks the last step completed are fed to their matchers here too, while the
        # host sync that takes costs nothing.
        if algo_states is not None:
            finished = [i for i, state in enumerate(algo_states)
                        if state is not None and state["matcher"] is not None and not state["accepted"]
                        and self.mask_id not in self._blocks[i]]
            if finished:
                self._accept_blocks(forward_batch.rids, algo_states, self._blocks, finished)
        return super().run(model_runner, forward_batch, algo_states)

    # ---- admission (called by the scheduler hook) ----

    def start(self, server_args) -> None:
        """Load the tokenizer and spawn the compile workers when the engine starts,
        not when the first constrained request arrives."""
        if not self._started:
            self._started = True
            self._set_tokenizer_path(server_args)
            self._get_compiler().start_workers()

    def prefetch(self, reqs, server_args) -> None:
        """Get the constraints of these queued requests ready, in queue order.

        Every schema starts compiling into the disk cache at once; the first
        ``load_ahead`` constrained requests also have theirs loaded onto the
        GPU, whose memory is spent only on requests about to run.
        """
        if self._tokenizer_path is None:
            self._set_tokenizer_path(server_args)
        loading = 0
        for req in reqs:
            if req.rid in self._matchers:
                continue
            if req.rid in self._compiling:
                loading += 1
                continue
            schema = request_schema(req)
            if schema is None:
                continue
            if loading < self.load_ahead:
                self._compiling[req.rid] = (req, self._get_compiler().compile_json_schema_async(schema))
                loading += 1
            elif req.rid not in self._precompiling:
                self._get_compiler().precompile_json_schema(schema)
                self._precompiling[req.rid] = (req, None)

    def is_ready(self, req) -> bool:
        """Whether ``req`` can join a batch: unconstrained, or its constraint is loaded (or failed)."""
        entry = self._compiling.get(req.rid)
        if entry is not None:
            return entry[1].done()
        return req.rid in self._matchers or request_schema(req) is None

    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        bz = forward_batch.batch_size
        # the block's committed tokens (the prompt's tail) come first
        starts = [sum(t != self.mask_id for t in block) for block in self._host_blocks(forward_batch)]
        reqs = getattr(forward_batch, "mosaic_reqs", None) or [None] * bz
        for tracked in (self._matchers, self._compiling, self._precompiling):
            for rid in [rid for rid, (req, _) in tracked.items() if req.finished()]:
                del tracked[rid]
        return [
            {"matcher": self._matcher_for(req), "start": start, "accepted": False}
            for req, start in zip(reqs, starts)
        ]

    def step(self, forward_batch: ForwardBatch, full_logits: torch.Tensor,
             states: List[Any]) -> List[bool]:
        bz = forward_batch.batch_size
        vocab_size = full_logits.shape[-1]
        logits = full_logits.view(bz, self.block_size, vocab_size)
        input_ids = forward_batch.input_ids.view(bz, self.block_size)
        masked = input_ids == self.mask_id
        # No host sync until the end: the forward may still be running on the GPU.
        blocks = self._host_blocks(forward_batch)
        done = [self.mask_id not in block for block in blocks]

        # LowConfidence's proposal, kept for unconstrained rows
        x0 = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

        # constrained rows, batched by where their generated positions start
        groups = defaultdict(list)
        finished = []  # rows whose block is complete and not yet fed to their matcher (by run())
        for i, state in enumerate(states):
            if state["matcher"] is None:
                continue
            if done[i]:
                if not state["accepted"]:
                    finished.append(i)
                continue
            groups[state["start"]].append(i)
        if finished:
            self._accept_blocks(forward_batch.rids, states, blocks, finished)
        nan_counts = []
        for s, rows in groups.items():
            idx = to_device_async(rows, logits.device)
            x_gen, marginal = self._batch.batch_propose_x0(
                [states[i]["matcher"] for i in rows], logits[idx, s:], input_ids[idx, s:], self.mask_id,
                temperature=self.temperature, return_marginal=self.commit_by == "constrained",
                nan_counts=nan_counts,
            )
            if marginal is None:  # commit_by=model: the model's probability of the proposal
                marginal = torch.gather(probs[idx, s:], -1, x_gen.unsqueeze(-1)).squeeze(-1)
            x0[idx, s:] = x_gen
            confidence[idx, s:] = marginal.to(confidence.dtype)

        # LowConfidence: commit masked positions above the threshold, else the top one
        confidence = torch.where(masked, confidence, -float("inf"))
        transfer = confidence > self.threshold
        has_transfer = transfer.sum(dim=1) > 0
        top1 = torch.zeros_like(transfer)
        top1[torch.arange(bz, device=top1.device), confidence.argmax(dim=1)] = True
        transfer = torch.where(has_transfer.unsqueeze(-1), transfer, top1)

        x0 = torch.where(masked, x0, input_ids)
        new_input_ids = torch.where(transfer, x0, input_ids)
        # In place, to keep the input_ids tensor identity (CUDA-graph safe).
        forward_batch.input_ids.copy_(new_input_ids.view(-1))
        # The step's one host sync, now that all its work is queued (the caller reads
        # the new blocks right after anyway): the next step's blocks, and the checks.
        self._blocks = new_input_ids.tolist()
        warn_nan_counts(nan_counts)
        return done

    # ---- helpers ----

    def _matcher_for(self, req) -> Optional[ConstraintMatcher]:
        """The request's matcher, created on its first block; None if unconstrained."""
        if req is None:
            return None
        if req.rid in self._matchers:
            return self._matchers[req.rid][1]
        schema = request_schema(req)
        if schema is None:
            return None
        self.prefetch([req], None)  # normally done while the request was queued
        _, future = self._compiling.pop(req.rid)
        self._precompiling.pop(req.rid, None)
        try:
            compiled = future.result()  # ready unless the request skipped the queue
        except torch.OutOfMemoryError:
            raise
        except Exception as e:  # an unsupported schema decodes unconstrained, not a crashed server
            logger.error("mosaic: cannot compile the schema of request %s, decoding it unconstrained: %s",
                         req.rid, e)
            return None
        matcher = ConstraintMatcher(compiled, max_new_tokens=req.sampling_params.max_new_tokens)
        self._matchers[req.rid] = (req, matcher)
        while len(self._matchers) > MAX_TRACKED:
            self._matchers.popitem(last=False)
        return matcher

    def _set_tokenizer_path(self, server_args):
        if server_args is not None:
            self._tokenizer_path = server_args.tokenizer_path or server_args.model_path

    def _get_compiler(self) -> ConstraintCompiler:
        if self._compiler is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path, trust_remote_code=True)
            # an explicit index: the loader thread must not fall back to GPU 0
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
            self._compiler = ConstraintCompiler(
                tokenizer, sampler=self.sampler, device=device, matmul_dtype=self.matmul_dtype,
                cache_dir=self.cache_dir, cache_size=self.cache_size,
                compile_workers=self.compile_workers,
            )
        return self._compiler

    def _host_blocks(self, forward_batch):
        """The batch's blocks as host lists: read in ``run`` before the forward, or left by
        the previous step (the synchronous schedule runs several steps per ``run``)."""
        blocks = getattr(self, "_blocks", None)
        if blocks is None or len(blocks) != forward_batch.batch_size:  # not called through run()
            blocks = forward_batch.input_ids.view(forward_batch.batch_size, self.block_size).tolist()
        return blocks

    def _accept_blocks(self, rids, states, blocks, rows):
        """Feed these rows' finished blocks (host lists) to their matchers."""
        blocks = [blocks[i][states[i]["start"]:] for i in rows]
        for i, ok in zip(rows, BatchConstraintMatcher.batch_accept_tokens([states[i]["matcher"] for i in rows], blocks)):
            states[i]["accepted"] = True
            if not ok:
                logger.warning("mosaic: block rejected by the constraint for request %s", rids[i])


Algorithm = ConstrainedLowConfidence
