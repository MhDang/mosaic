# AGENTS.md

Guidance for AI agents and contributors. Pair it with the [README](README.md).

## What this is

Exact constrained sampling for diffusion LMs: at each denoising step the
proposal for all masked positions is drawn from (model's per-position
distribution) x (constraint automaton), by forward-backward over the automaton.
`mosaic/sampling/parallel.py` does it with a segment tree (O(log L) depth), `mosaic/sampling/sequential.py`
sequentially (O(L)). Every decoding loop goes through `mosaic/matcher.py`, an API
shaped like xgrammar's: a `ConstraintCompiler` per tokenizer and a
`ConstraintMatcher` per request, whose `propose_x0` proposes a block's x0 from the
automaton state the accepted blocks reached. The Dream and LLaDA loops in
`mosaic/dlm/` treat the whole response as one block; the block-diffusion ones (the
LLaDA2 loop `mosaic/dlm/llada2.py`, and SGLang via `integrations/sglang/`) go block by
block.

## Running

- Envs come from `pyproject.toml` / `uv.lock` (uv, Python 3.11): `uv sync --extra hf
  --extra sglang` makes `.venv` (torch 2.13, transformers 5.12.1: the HF models and
  SGLang); `UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --extra vllm` makes `.venv-vllm`
  (transformers 5.17, the vLLM nightly and the plugin). The extras conflict (SGLang and
  vLLM pin different versions), hence two envs. Activate one and run everything from
  the repo root; a bare `uv run` re-syncs the env without the extras.
- Dream, LLaDA and LLaDA2 run their Hub code at the commits in `REVISIONS`
  (`mosaic/dlm/models.py`), adapted to transformers 5 by `mosaic/dlm/compat.py`.
- Layout: `mosaic/` is the library; `integrations/` holds the serving engines' adapters
  (SGLang, vLLM); `benchmarks/<name>/` holds everything about a benchmark, including
  its runner per engine (`benchmarks/bfcl/run_hf.py`, `run_sglang.py`;
  `benchmarks/jevbench/run_vllm.py`).
- `benchmarks/bfcl/run_hf.py` compiles the automata (CPU workers), generates and scores;
  each model's generation settings are `MODEL_SETTINGS` there. `accelerate launch
  --num_processes=<gpus> benchmarks/bfcl/run_hf.py ...` splits the examples over GPUs.
- `python -m pytest` runs the tests.

## Gotchas

- **Host syncs.** The SGLang step queues its work behind the model's forward, so
  any host sync before its end waits for the whole forward and serializes the
  step's CPU work after it. That includes
  `torch.tensor(list, device="cuda")` and `.tolist()` / `.item()`; use
  `to_device_async`, host-side state prepared before the forward or at load
  (the blocks, `steps_to_accept`, `_edges_numpy`), and `nan_counts` lists
  checked at the end.
- **Precision.** `import mosaic.sampling` sets `torch.set_float32_matmul_precision("high")`
  (TF32). `ParallelSampler.matmul_dtype = torch.float64` (`--matmul_dtype float64`)
  computes the parallel sampler's products in fp64: slower on GPUs with weak fp64,
  but free of underflow on very peaked logits. Never remove a `nan_to_num_`, a
  max-shift or an `assert` from the samplers: they guard real numerical edge cases.
- **Automata depend on the tokenizer.** The cache path includes the tokenizer
  family (`cache/bfcl_<template>_<dream|llada|llada2>/...`). Compilation is
  deterministic (`ConstraintNFA.canonicalize`), so a rebuilt automaton has the same
  tensors. Delete a cache entry to rebuild it.
- **The models' Hub code on transformers 5.** Dream's, LLaDA's and LLaDA2's
  modeling code was written for transformers 4; `mosaic/dlm/compat.py` adapts it. One
  of its failures is silent: transformers 5 leaves the rotary frequencies
  uninitialized, and the models still produce valid (constrained) JSON with wrong
  content. After changing transformers or a `REVISIONS` commit, diff generations
  against the previous setup.
- **Dream's logits are shifted** by one position (it predicts token i from
  position i-1); `mosaic/dlm/dream.py` undoes that before anything else.
- **Block diffusion.** A matcher's proposal must start from the state its
  accepted blocks reached (`start_nodes`) and end where the output can still
  finish in the tokens left (`end_log`, and EOS forced past `max_new_tokens`).
  LLaDA2's blocks are aligned to absolute positions, so the first one also holds
  the prompt's tail: only the positions after it are proposed.
- **LLaDA2's turn ends with `<|role_end|>`** then `<|endoftext|>` padding, so its
  constraint's eot is `<|role_end|>` (`resolve_tokenizer_tokens`).

## Adding a constraint

Write a grammar with `mosaic/grammar/automata.py`'s primitives (`literal`, `any_of`,
`symbol`) and operations (`+`, `|`, `.option()`, `.kleene_star()`,
`.recursive_concat()`), finish it with `.dfa_minify().canonicalize()`, and compile it
with `ConstraintCompiler(tokenizer).compile_grammar(char_nfa, key)` (`mosaic/matcher.py`).
`mosaic/grammar/json_schema.py` is the worked example; `examples/json_schema_demo.py`
shows the whole path from a schema to a constrained generation.

## Adding a dataset

Put it in `benchmarks/<name>/`. `BFCLDataset` (`benchmarks/bfcl/data.py`) is the
pattern for one driven by the HF loops: `examples`, `build_prompt`,
`compile_constraint` (with `constraint_key`, the cache key under `--cache_dir`), `post_process`.
`benchmarks/bfcl/run_hf.py` shows how a script drives a dataset: compile, generate,
`post_process`, `summarize`.

## Verifying a change

The check that catches regressions is a before/after diff of generations: run a
few examples (`--end 8`) before and after and compare `generated_sequence`
in `generations_*.json`. `benchmarks/bfcl/summary.py` summarizes run directories.
