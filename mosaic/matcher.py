"""Constrained x0 proposals for diffusion LMs, as a small engine-facing API.

The shape follows xgrammar: compile a constraint once per schema, keep one
matcher per request, and call it from the engine's decoding loop. Where an
autoregressive grammar matcher fills a next-token bitmask, a diffusion matcher
proposes the whole block at once:

    compiler = ConstraintCompiler(tokenizer)
    compiled = compiler.compile_json_schema(schema)          # cached per schema
    # or any grammar built with mosaic.grammar: compiler.compile_grammar(char_nfa, key)
    matcher = ConstraintMatcher(compiled, max_new_tokens=256)

    # at every denoising step of a block (positions after what was accepted):
    x0, marginal = matcher.propose_x0(block_logits, block_tokens, mask_token_id)
    # ... the engine commits some positions of x0, e.g. marginal > threshold ...
    # once the block is final:
    matcher.accept_tokens(block_tokens)

    # an engine's batch: the same calls on many requests' matchers at once
    batch = BatchConstraintMatcher()
    x0, marginal = batch.batch_propose_x0(matchers, logits, tokens, mask_token_id)
    BatchConstraintMatcher.batch_accept_tokens(matchers, blocks)

    # a sampler that redraws every position at every step (DiffusionGemma's)
    # also needs each position's whole constrained distribution:
    log_marginal = matcher.marginal_log(block_logits, block_tokens, mask_token_id)
    log_marginals = BatchConstraintMatcher.batch_marginal_log(matchers, logits, tokens, mask_token_id)

``propose_x0`` draws x0 exactly from (the model's per-position distribution) x
(the constraint), given the positions of the block already committed, starting
from the automaton state the accepted prefix reached, and requiring that the
output can still be finished in the tokens left after the block. Any subset of
x0 can be committed: x0 itself is a valid continuation, so the next step always
has one.

A serving engine should not stall on a schema it has not seen:
``compiler.compile_json_schema_async(schema)`` compiles in worker processes and
loads in a background thread, returning a Future.
"""

import hashlib
import json
import multiprocessing
import os
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

import torch

from .tokenizers import compile_automaton, resolve_tokenizer_tokens, special_tokens, tokenizer_key
from .sampling import SAMPLERS, ParallelSampler
from .sampling.parallel import sample_many, to_device_async
from .sampling.automaton import UNREACHABLE
from .grammar.json_schema import JSONBuilder


class ConstraintCompiler:
    """Compiles constraints for one tokenizer, with an LRU cache: JSON Schemas
    (:meth:`compile_json_schema`), or any character automaton (:meth:`compile_grammar`).

    eos_token: the token that pads the output once it is complete. eot_token:
        the token that must follow the value before the padding. Both default
        to what :func:`~mosaic.tokenizers.resolve_tokenizer_tokens` gives for
        Dream / LLaDA / LLaDA2 tokenizers, else to the tokenizer's EOS. Every
        special token of the tokenizer is excluded from string contents.
    sampler: ``parallel`` (O(log L) depth) or ``sequential`` (O(L)).
    cache_dir: if given, compiled automata are also saved there (per tokenizer
        and schema) and loaded back instead of recompiled; compiling a schema
        takes ~0.5 s, loading it milliseconds.
    compile_workers: worker processes for :meth:`compile_json_schema_async`
        (0: compile in its background thread instead). They need a disk cache;
        a temporary directory is used if ``cache_dir`` is None.
    """

    def __init__(self, tokenizer, eos_token=None, eot_token=None, sampler="parallel",
                 device="cuda", max_depth=3, cache_size=64, matmul_dtype=None, cache_dir=None,
                 compile_workers=0):
        self.tokenizer = tokenizer
        try:
            default_eos, default_eot, _ = resolve_tokenizer_tokens(tokenizer)
        except NotImplementedError:
            default_eos, default_eot = tokenizer.eos_token, None
        self.eos_token = eos_token or default_eos
        self.eot_token = eot_token or default_eot or self.eos_token
        self.sampler_cls = SAMPLERS[sampler]
        self.device = device
        self.max_depth = max_depth
        self.matmul_dtype = matmul_dtype
        self._cache = OrderedDict()
        self._cache_size = cache_size
        self.compile_workers = compile_workers
        if compile_workers > 0 and cache_dir is None:
            cache_dir = tempfile.mkdtemp(prefix="mosaic-constraints-")
        self.cache_dir = cache_dir
        self._lock = threading.RLock()  # the memory cache and the in-flight compiles
        self._pending = {}  # schema key -> Future, while it compiles / loads
        self._disk_jobs = {}  # schema key -> worker job compiling it into the disk cache
        self._loader = None  # background thread: loads compiled automata onto the device
        self._workers = None  # process pool: compiles
        self.banned_tokens = [t for t in special_tokens(tokenizer)
                              if t not in (self.eos_token, self.eot_token)]
        # Everything besides the schema that changes the automaton. Computed once:
        # len() of a large tokenizer takes tens of milliseconds.
        settings = json.dumps([tokenizer.name_or_path, len(tokenizer), self.eos_token,
                               self.eot_token, self.banned_tokens, self.max_depth])
        self._settings_key = hashlib.sha256(settings.encode()).hexdigest()[:16]

    def compile_json_schema(self, schema):
        """A constraint accepting exactly the JSON values that satisfy ``schema``."""
        if isinstance(schema, str):
            schema = json.loads(schema)
        key = _schema_key(schema)

        def build():
            builder = JSONBuilder(
                set(map(chr, range(256))),
                max_depth=self.max_depth,
                eos_token=self.eos_token,
                eot_token=self.eot_token,
                banned_tokens=self.banned_tokens,
            )
            return builder.build_value(schema)

        return self._compile(key, self._path(key), build)

    def compile_grammar(self, char_nfa, key=None):
        """A constraint from a character automaton, like xgrammar's ``compile_grammar``.

        char_nfa: built with :mod:`mosaic.grammar`'s primitives or a builder (e.g.
            :class:`~mosaic.grammar.json_schema.JSONBuilder`), or a function of no
            arguments returning it, called only if the constraint is not cached.
            It must spell out its own end (EOT, EOS padding) and banned tokens.
        key: the grammar's name in the caches: the memory cache, and the directory
            ``cache_dir/key`` on disk, so it must identify the tokenizer as well.
            None: compile it every time and cache nothing.
        """
        path = None if key is None or self.cache_dir is None else os.path.join(self.cache_dir, key)
        return self._compile(key, path, char_nfa)

    def _compile(self, key, path, char_nfa):
        """The constraint ``key`` from the memory cache, else from the disk cache ``path``,
        else compiled from ``char_nfa`` (called first if it is a function) and saved there."""
        with self._lock:
            if key is not None and key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        if path is not None and os.path.isfile(os.path.join(path, "config.json")):
            sampler = self.sampler_cls.from_pretrained(path, device=self.device)
        else:
            if callable(char_nfa):
                char_nfa = char_nfa()
            sampler = compile_automaton(char_nfa, self.tokenizer, save_directory=path,
                                        cls=self.sampler_cls, device=self.device)
        sampler = sampler.to(self.device).eval()
        if isinstance(sampler, ParallelSampler):
            sampler.matmul_dtype = self.matmul_dtype
        # host-side tables the decoding steps use, made here (maybe in the loader thread)
        # rather than with a host sync in the middle of a step
        sampler.steps_to_accept()
        sampler._edges_numpy()
        compiled = CompiledConstraint(sampler, key, self.tokenizer.convert_tokens_to_ids(self.eos_token))
        if key is not None:
            with self._lock:
                self._cache[key] = compiled
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return compiled

    def compile_json_schema_async(self, schema):
        """:meth:`compile_json_schema` in the background; returns a ``concurrent.futures.Future``.

        A schema in the memory cache resolves at once. Otherwise, with
        ``compile_workers``, a worker process compiles it into the disk cache
        (compiling is mostly Python, so in a thread it would hold the caller up
        through the GIL) and a background thread then loads it onto ``device``.
        Concurrent calls for one schema share a Future.
        """
        if isinstance(schema, str):
            schema = json.loads(schema)
        key = _schema_key(schema)
        with self._lock:
            if key in self._cache:
                future = Future()
                future.set_result(self._cache[key])
                return future
            if key in self._pending:
                return self._pending[key]
            future = self._pending[key] = Future()
        job = self._compile_to_disk(schema, key)
        if job is not None:
            job.add_done_callback(
                lambda job: self._loader_thread().submit(self._resolve, schema, key, future, job)
            )
        else:
            self._loader_thread().submit(self._resolve, schema, key, future)
        return future

    def precompile_json_schema(self, schema):
        """Start compiling ``schema`` into the disk cache, without loading it.

        For a schema needed later (a queued request's): the later
        :meth:`compile_json_schema_async` then only loads it, while the device
        memory is spent only on schemas about to be used. Returns the worker's
        Future, or None if there is nothing to do (no ``compile_workers``, or
        the schema is already compiled).
        """
        if isinstance(schema, str):
            schema = json.loads(schema)
        key = _schema_key(schema)
        with self._lock:
            if key in self._cache or key in self._pending:
                return None
        return self._compile_to_disk(schema, key)

    def _compile_to_disk(self, schema, key):
        """The worker job compiling ``schema`` into the disk cache (started if needed), or None."""
        path = self._path(key)
        if self.compile_workers == 0 or os.path.isfile(os.path.join(path, "config.json")):
            return None
        with self._lock:
            job = self._disk_jobs.get(key)
            if job is None:
                job = self._disk_jobs[key] = self._worker_pool().submit(_compile_in_worker, schema)
                job.add_done_callback(lambda _: self._forget_disk_job(key))
            return job

    def _forget_disk_job(self, key):
        with self._lock:
            self._disk_jobs.pop(key, None)

    def _resolve(self, schema, key, future, job=None):
        """Background thread: load (or compile) ``schema`` and complete ``future``."""
        try:
            if job is not None:
                job.result()  # re-raise a worker's error
            compiled = self.compile_json_schema(schema)
            device = torch.device(self.device)
            if device.type == "cuda":  # the tensors are complete before any other stream reads them
                torch.cuda.current_stream(device).synchronize()
            future.set_result(compiled)
        except BaseException as e:
            future.set_exception(e)
        finally:
            with self._lock:
                self._pending.pop(key, None)

    def start_workers(self):
        """Spawn every compile worker now rather than on the first compile: each
        loads torch and the tokenizer, which takes seconds."""
        if self.compile_workers > 0:
            pool = self._worker_pool()
            for _ in range(self.compile_workers):  # the pool spawns a worker for each waiting task
                pool.submit(int)

    def close(self):
        """Stop the compile workers and the loader thread (each starts again on its next use)."""
        with self._lock:
            workers, loader = self._workers, self._loader
            self._workers = self._loader = None
        if workers is not None:
            workers.shutdown()
        if loader is not None:
            loader.shutdown()

    def _loader_thread(self):
        with self._lock:
            if self._loader is None:
                self._loader = ThreadPoolExecutor(1, thread_name_prefix="mosaic-load")
            return self._loader

    def _worker_pool(self):
        with self._lock:
            if self._workers is None:
                settings = dict(eos_token=self.eos_token, eot_token=self.eot_token,
                                max_depth=self.max_depth, cache_dir=self.cache_dir)
                self._workers = ProcessPoolExecutor(
                    self.compile_workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_worker, initargs=(self.tokenizer.name_or_path, settings),
                )
            return self._workers

    def _path(self, key):
        return None if self.cache_dir is None else os.path.join(self.cache_dir, self._disk_key(key))

    def compile_builtin_json_grammar(self):
        """Any JSON value (nesting up to ``max_depth``)."""
        return self.compile_json_schema({})

    @property
    def tokenizer_key(self):
        return tokenizer_key(self.tokenizer)

    def _disk_key(self, schema_key):
        """Cache subdirectory: the compiler's settings, then the schema."""
        return os.path.join(self._settings_key, schema_key)


def _schema_key(schema):
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


# ---- compile workers (separate processes) ----

_worker_compiler = None


def _init_worker(tokenizer_path, settings):
    """Process-pool initializer: a CPU compiler writing into the shared disk cache."""
    global _worker_compiler
    from transformers import AutoTokenizer

    parent = os.getppid()

    def exit_with_parent():  # never outlive the engine, however it goes down
        while os.getppid() == parent:
            time.sleep(1.0)
        os._exit(0)

    threading.Thread(target=exit_with_parent, daemon=True).start()
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    _worker_compiler = ConstraintCompiler(tokenizer, device="cpu", cache_size=1, **settings)


def _compile_in_worker(schema):
    _worker_compiler.compile_json_schema(schema)


class CompiledConstraint:
    """A compiled constraint: the token automaton, shared by every request using it.

    eos_token_id: the padding token, which may only follow a complete output;
    matchers with a ``max_new_tokens`` bound force it past the bound.
    """

    def __init__(self, sampler, key, eos_token_id=None):
        self.sampler = sampler
        self.key = key
        self.eos_token_id = eos_token_id

    @property
    def num_states(self):
        return self.sampler.num_nodes

    @property
    def num_edges(self):
        return self.sampler.num_edges

    @property
    def vocab_size(self):
        return self.sampler.vocab_size


class ConstraintMatcher:
    """Per-request state: where the accepted prefix left the automaton.

    max_new_tokens bounds the output length; blocks are then conditioned on the
    output being completable within it, and a block running past it has EOS
    forced there (the constraint's EOS padding). None means no bound.
    """

    def __init__(self, compiled, max_new_tokens=None):
        self.compiled = compiled
        self.max_new_tokens = max_new_tokens
        self.reset()

    def reset(self):
        self.node = self.compiled.sampler.initial_node_id
        self.num_accepted = 0

    # ---- decoding ----

    @torch.no_grad()
    def propose_x0(self, logits, tokens, mask_token_id, temperature=0.0,
                   return_marginal=True):
        """Propose x0 for the next block of positions.

        logits: (L, V_model) the model's logits for the L positions following the
            accepted prefix, or (n, L, V_model) for n independent proposals from
            this same state (e.g. several samples of one prompt). tokens: (L,) or
            (n, L), their current contents; ``mask_token_id`` marks free
            positions, anything else is committed and kept.
        Returns (x0, marginal or None), shaped like ``tokens``: x0 drawn from
        model x constraint (tokens at ``temperature``; 0 = argmax given the
        sampled state path), and each proposed token's exact constrained
        marginal probability. One sampler call draws all n rows.
        """
        single = tokens.dim() == 1
        if single:
            logits, tokens = logits.unsqueeze(0), tokens.unsqueeze(0)
        x0, marginal = BatchConstraintMatcher(across_constraints=False).batch_propose_x0(
            [self] * tokens.shape[0], logits, tokens, mask_token_id,
            temperature=temperature, return_marginal=return_marginal,
        )
        if single:
            return x0[0], None if marginal is None else marginal[0]
        return x0, marginal

    @torch.no_grad()
    def marginal_log(self, logits, tokens, mask_token_id):
        """The exact constrained distribution at every position of the next block.

        logits, tokens and mask_token_id as in :meth:`propose_x0` (one block).
        Returns (L, V_model) fp32 log p(x_t = v | model x constraint): the
        distribution ``propose_x0`` draws each position's token from,
        marginalized over the rest of the block (-inf on tokens the constraint
        never allows there).
        """
        return BatchConstraintMatcher.batch_marginal_log(
            [self], logits.unsqueeze(0), tokens.unsqueeze(0), mask_token_id)[0]

    def accept_tokens(self, tokens):
        """Advance past a finished block. Returns False if the tokens are rejected."""
        tokens = torch.as_tensor(tokens, device=self.compiled.sampler.device).reshape(1, -1)
        node = int(self.compiled.sampler.advance(torch.tensor([self.node]), tokens)[0])
        if node < 0:
            return False
        self.node = node
        self.num_accepted += tokens.shape[1]
        return True

    def accept(self, node, num_tokens):
        """Record an accepted block that led to ``node`` (see :meth:`BatchConstraintMatcher.batch_accept_tokens`)."""
        self.node = int(node)
        self.num_accepted += num_tokens

    def is_terminated(self):
        """Whether the accepted tokens form a complete output (the rest may only pad)."""
        return bool(self.compiled.sampler.accept_node[self.node] > 0)

    def end_log(self, block_len):
        """Log end weights for the next ``block_len`` positions."""
        if self.max_new_tokens is None:
            remaining = UNREACHABLE - 1
        else:
            remaining = self.max_new_tokens - self.num_accepted - block_len
        return self.compiled.sampler.finish_within_log(remaining)

    def budget(self):
        """How many more tokens the output may have (None: unbounded)."""
        if self.max_new_tokens is None:
            return None
        return max(self.max_new_tokens - self.num_accepted, 0)


# ---- batches of requests ----


class BatchConstraintMatcher:
    """Calls on many requests' matchers at once, like xgrammar's ``BatchGrammarMatcher``.

    across_constraints: with parallel samplers and no marginals, sample every row in
        one sampler call, whatever its constraint (:func:`~mosaic.sampling.parallel.sample_many`);
        otherwise one call per constraint.
    """

    def __init__(self, across_constraints=True):
        self.across_constraints = across_constraints

    @torch.no_grad()
    def batch_propose_x0(self, matchers, logits, tokens, mask_token_id, temperature=0.0,
                         return_marginal=True, nan_counts=None):
        """:meth:`ConstraintMatcher.propose_x0` for a batch of requests: logits (bz, L, V_model),
        tokens (bz, L). Returns (x0 (bz, L), marginal (bz, L) or None).

        nan_counts: a list, on the shared path: the sampler appends its NaN-fallback
        counts (device tensors) there instead of checking them, so that the call has
        no host sync; the caller checks them (:func:`~mosaic.sampling.parallel.warn_nan_counts`).
        """
        bz, L = tokens.shape
        samplers = [m.compiled.sampler for m in matchers]
        forced = _forced_tokens(matchers, tokens, mask_token_id)
        start = [m.node for m in matchers]
        end = [m.end_log(L) for m in matchers]

        if self.across_constraints and not return_marginal and all(isinstance(s, ParallelSampler) for s in samplers):
            V = samplers[0].vocab_size
            x0 = sample_many(samplers, logits[:, :, :V].float(), forced_tokens=forced, temperature=temperature,
                             start_nodes=start, end_log=end, nan_counts=nan_counts)
            return x0, None

        x0 = torch.empty_like(tokens)
        marginal = torch.empty(bz, L, device=logits.device) if return_marginal else None
        groups = OrderedDict()  # constraint -> rows
        for i, m in enumerate(matchers):
            groups.setdefault(id(m.compiled), []).append(i)
        for rows in groups.values():
            sampler = samplers[rows[0]]
            idx = to_device_async(rows, logits.device)
            x, marg = sampler.sample(
                logits[idx, :, : sampler.vocab_size].float(), forced_tokens=forced[idx], temperature=temperature,
                return_marginal=return_marginal, start_nodes=to_device_async([start[r] for r in rows], sampler.device),
                end_log=torch.stack([end[r] for r in rows]),
            )
            x0[idx] = x.to(x0.device)
            if return_marginal:
                marginal[idx] = marg.to(marginal.dtype)
        return x0, marginal

    @staticmethod
    @torch.no_grad()
    def batch_marginal_log(matchers, logits, tokens, mask_token_id):
        """:meth:`ConstraintMatcher.marginal_log` for a batch of requests: logits (bz, L, V_model),
        tokens (bz, L). Rows that share a constraint go through one sampler call.

        Returns (bz, L, V_model) fp32 log-marginals: a point mass on a committed
        or forced token (EOS past the length bound); model tokens beyond the
        constraint's vocabulary get -inf.
        """
        bz, L = tokens.shape
        forced = _forced_tokens(matchers, tokens, mask_token_id)
        out = torch.full((bz, L, logits.shape[-1]), -torch.inf, device=logits.device)
        groups = OrderedDict()  # constraint -> rows
        for i, m in enumerate(matchers):
            groups.setdefault(id(m.compiled), []).append(i)
        for rows in groups.values():
            sampler = matchers[rows[0]].compiled.sampler
            V = sampler.vocab_size
            idx = to_device_async(rows, logits.device)
            lm_logits = torch.log_softmax(logits[idx, :, :V].float(), dim=-1)
            log_marginal = sampler._compute_marginal_log(
                lm_logits, forced[idx],
                start_nodes=to_device_async([matchers[r].node for r in rows], sampler.device),
                end_log=torch.stack([matchers[r].end_log(L) for r in rows]),
            ).to(out.device, torch.float32)
            # a forced position's token is given: a point mass (the sampler's pass leaves
            # the model's probability of it there, which a proposal never needs)
            given = forced[idx] >= 0
            if given.any():
                point = torch.full_like(log_marginal, -torch.inf)
                point.scatter_(-1, forced[idx].clamp(min=0).unsqueeze(-1), 0.0)
                log_marginal = torch.where(given.unsqueeze(-1), point, log_marginal)
            out[idx, :, :V] = log_marginal
        return out

    @staticmethod
    def batch_accept_tokens(matchers, blocks):
        """:meth:`ConstraintMatcher.accept_tokens` for several requests: one host sync for all.

        blocks: one 1-D token sequence per matcher (its finished block's new tokens),
        a tensor or a host list.
        Returns a list of bools (False: that block is rejected, its matcher unchanged).
        """
        if not matchers:
            return []
        groups = OrderedDict()  # (constraint, block length) -> rows
        for i, (m, block) in enumerate(zip(matchers, blocks)):
            groups.setdefault((id(m.compiled), block.numel() if torch.is_tensor(block) else len(block)), []).append(i)
        allowed = []
        for rows in groups.values():
            sampler = matchers[rows[0]].compiled.sampler
            tokens = torch.stack([torch.as_tensor(blocks[i], device=sampler.device).reshape(-1) for i in rows])
            allowed.append(sampler.allowed_edges(tokens))
        flat = torch.cat([a.reshape(-1) for a in allowed]).cpu().numpy()  # the one host sync
        ok, offset = [False] * len(matchers), 0
        for (key, rows), a in zip(groups.items(), allowed):
            sampler = matchers[rows[0]].compiled.sampler
            block_allowed = flat[offset:offset + a.numel()].reshape(tuple(a.shape))
            offset += a.numel()
            nodes = sampler.walk([matchers[i].node for i in rows], block_allowed)
            for i, node in zip(rows, nodes):
                if node >= 0:
                    matchers[i].accept(node, key[1])
                    ok[i] = True
        return ok


def _forced_tokens(matchers, tokens, mask_token_id):
    """Each row's committed tokens (-1: free), with EOS forced past its matcher's length bound."""
    L = tokens.shape[1]
    forced = torch.where(tokens == mask_token_id, torch.full_like(tokens, -1), tokens)
    for i, m in enumerate(matchers):  # past the length bound, only EOS padding
        budget = m.budget()
        if budget is not None and budget < L:
            assert m.compiled.eos_token_id is not None, \
                "a length bound shorter than the block needs the constraint's EOS id"
            forced[i, budget:] = m.compiled.eos_token_id
    return forced
