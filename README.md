# Mosaic: Constrained Decoding for Diffusion Language Models

Official implementation of *Constrained Decoding for Diffusion Language Models via
Efficient Inference over Finite Automata* (NeurIPS 2026).

[![arXiv](https://img.shields.io/badge/arXiv-2607.07026-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.07026)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce)](https://neurips.cc/Conferences/2026)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Autoregressive models enforce a constraint by masking out invalid next tokens. Diffusion
language models (Dream, LLaDA, LLaDA2, DiffusionGemma) instead generate many tokens at
once, in any order, so the tokens have to satisfy the constraint together.

Mosaic compiles the constraint, such as a JSON Schema or a function-call format, into a
finite automaton. At every denoising step, given the current partly denoised sequence
$x^t$, it samples the denoised output $x^0$, all tokens jointly, from the model's
prediction multiplied by the constraint:

$$
p(x^0 \mid x^t, \text{constraint}) \propto \prod_{i} p_\theta(x^0_i \mid x^t) \cdot \mathbb{1}\big[x^0 \text{ satisfies the constraint}\big]
$$

The sample $x^0$ is exact, so every output satisfies the constraint.

- **Exact**: samples from the model's own distribution restricted to the constraint.
- **Fast**: the default `parallel` sampler handles an output of $L$ tokens in $O(\log L)$ sequential steps instead of $O(L)$.
- **Ready to serve**: [SGLang](#sglang) (LLaDA2) and [vLLM](#vllm) (DiffusionGemma, with
  thinking).

## News

- **[2026/10/03]** Code released.
- **[2026/09/24]** Accepted at NeurIPS 2026.
- **[2026/07/08]** Paper on [arXiv](https://arxiv.org/abs/2607.07026).

## Setup

Linux (x86_64), an NVIDIA GPU with a CUDA 13 driver, and [uv](https://docs.astral.sh/uv/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh`). Run everything from the
repository root; models download from the Hugging Face Hub on first use.

```bash
uv sync --extra hf                                        # .venv: Dream, LLaDA, LLaDA2
uv sync --extra hf --extra sglang                         # .venv, plus SGLang (LLaDA2)
UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --extra vllm    # .venv-vllm: vLLM (DiffusionGemma)
source .venv/bin/activate                                 # or .venv-vllm
```

SGLang and vLLM pin conflicting versions, hence two environments. Activate one
rather than using `uv run`, which re-syncs it without the extras.

## Try it: any JSON Schema

In `.venv`:

```bash
python examples/json_schema_demo.py --model dream \
    --prompt "Alice is 31, lives in Paris, and likes chess and hiking. Her email is alice@example.com." \
    --schema '{"type": "object", "properties": {
        "name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"},
        "email": {"type": "string"}, "hobbies": {"type": "array", "items": {"type": "string"}}},
        "required": ["name", "age"]}'
```

```
compiled schema: 80 states, 308 edges

--- output ---
{
  "name": "Alice",
  "age": 31,
  "city": "Paris",
  "email": "alice@example.com",
  "hobbies": ["Chess", "hiking"]
}

valid: parses as JSON and satisfies the schema
```

`--schema` takes a JSON Schema, inline as above or as a file path
(`--schema my.schema.json`), with the keywords `type`, `properties`, `required`,
`items`, `minItems`, `enum`, `const` and `anyOf` / `oneOf` (properties come out in
the declared order). With `--sampler none` (plain decoding) the generation is not
valid JSON.

## Serve with SGLang or vLLM

Both integrations take a JSON Schema per request; requests without one decode as
usual, in the same batches.

### SGLang

LLaDA2.0-mini, with a drop-in dLLM algorithm, `ConstrainedLowConfidence`, in `.venv`
with the `sglang` extra (setup: [integrations/sglang](integrations/sglang/README.md)):

```bash
python -m sglang.launch_server --model-path inclusionAI/LLaDA2.0-mini --trust-remote-code \
    --dllm-algorithm ConstrainedLowConfidence --max-running-requests 16 --port 30000
```

The schema goes in the request's `custom_params`. A BFCL item (`simple_python_0`),
with BFCL's system prompt and the schema of its function calls (run from the
repository root):

```python
import json
from openai import OpenAI
from benchmarks.bfcl.data import SYSTEM_PROMPTS, tool_calls_json_schema

functions = [{"name": "calculate_triangle_area",
              "description": "Calculate the area of a triangle given its base and height.",
              "parameters": {"type": "dict", "properties": {
                  "base": {"type": "integer", "description": "The base of the triangle."},
                  "height": {"type": "integer", "description": "The height of the triangle."},
                  "unit": {"type": "string", "description": "The unit of measure (defaults to 'units' if not specified)"}},
                  "required": ["base", "height"]}}]
system = SYSTEM_PROMPTS["json"].format(functions=json.dumps(functions, indent=4))
user = "Find the area of a triangle with a base of 10 units and height of 5 units."

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")
out = client.chat.completions.create(
    model="inclusionAI/LLaDA2.0-mini", max_tokens=256, temperature=0,
    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    extra_body={"custom_params": {"json_schema": tool_calls_json_schema(functions)}},
)
print(out.choices[0].message.content)
# [{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5, "unit": "units"}}]
```

`benchmarks/bfcl/run_sglang.py` runs [BFCL](benchmarks/bfcl/README.md#through-sglang-run_sglangpy)
through SGLang's offline engine.

### vLLM

DiffusionGemma, in `.venv-vllm`, with a plugin vLLM loads by itself once installed
(setup and the server command: [integrations/vllm](integrations/vllm/README.md)). The
schema goes in the request's `vllm_xargs` (as a JSON string), and `"reasoning": "true"`
makes it a thinking request: the model thinks freely, then answers under the schema.
A [JevBench](https://github.com/fstandhartinger/jevbench) item (`hard-sol-b-judge_hard-18`),
with JevBench's system prompt (run from the repository root):

```python
import json
from openai import OpenAI
from benchmarks.jevbench.data import SYSTEM_PROMPT

user = """State:
{"request": "Return a semicolon-separated record with fields name, active, quota in that order. Exact output rules: no spaces; active is lowercase `true`; quota is the zero-padded four-digit integer 27; name is `north`. Output only the record, ending immediately after `0027` with no trailing newline or whitespace.", "response": "north;true;0027\\n"}

Does the response fully and correctly satisfy the request?

Options:
- no: The response has any substantive error or misses any explicit requirement.
- yes: The response fully and correctly satisfies every explicit request, including method, edge cases, units, formatting, and retained facts.

Output probabilities over exactly these keys: ["no", "yes"]."""
prob = {"type": "number", "x-prob": True}  # a probability with at most two decimals
schema = {"type": "object", "required": ["probabilities"], "properties": {"probabilities": {
    "type": "object", "required": ["no", "yes"], "properties": {"no": prob, "yes": prob}}}}

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
out = client.chat.completions.create(
    model="google/diffusiongemma-26B-A4B-it", max_tokens=8192,
    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True},
                "vllm_xargs": {"json_schema": json.dumps(schema), "reasoning": "true"}},
)
print(out.choices[0].message.content)  # the thinking, then {"probabilities": {"no": 1.0, "yes": 0.0}}
```

`benchmarks/jevbench/run_vllm.py` runs all of JevBench this way (see [JevBench](#jevbench)).

## Use it in your own decoding loop

`mosaic/matcher.py` is a small engine-facing API, shaped like
[xgrammar](https://github.com/mlc-ai/xgrammar)'s: compile a schema once, keep a
matcher per request, and ask it for the block's x0 at every denoising step.

```python
from mosaic.matcher import ConstraintCompiler, ConstraintMatcher

compiler = ConstraintCompiler(tokenizer)                  # one per tokenizer
compiled = compiler.compile_json_schema(schema)           # cached per schema
# or any grammar built with mosaic/grammar: compiler.compile_grammar(char_nfa, key)
matcher = ConstraintMatcher(compiled, max_new_tokens=256) # one per request

# each denoising step of a block (the positions after the accepted prefix):
x0, marginal = matcher.propose_x0(block_logits, block_tokens, mask_token_id)
# ... commit any subset of x0, e.g. positions whose confidence clears a threshold ...
# once the block is final:
matcher.accept_tokens(block_tokens)
```

`propose_x0` samples the whole block exactly from (model) x (schema), keeping
the block's committed tokens, starting where the accepted prefix left the
automaton, and ending where the output can still be finished within
`max_new_tokens`. `BatchConstraintMatcher` does both calls for a batch of requests
(like xgrammar's `BatchGrammarMatcher`).
Every loop uses this API: Dream's and LLaDA's (`mosaic/dlm/`) propose the whole response
as one block, LLaDA2's and the SGLang integration's go block by block.

## BFCL

The Berkeley Function Calling Leaderboard's eight AST splits, with BFCL's official
system prompt and checker, as JSON-array or Python-style calls; through Mosaic's own
decoding loops (Dream, LLaDA, LLaDA2) or SGLang (LLaDA2), in `.venv`:

```bash
bash benchmarks/bfcl/download.sh
python benchmarks/bfcl/run_hf.py --model dream --split simple --out outputs/bfcl/dream_simple
```

Details, options and results: [benchmarks/bfcl](benchmarks/bfcl/README.md).

## JevBench

[JevBench](https://github.com/fstandhartinger/jevbench) tests decision models: each
item gives a state and a rubric, and the answer is a probability for every option,
as JSON. It scores Intelligence (accuracy above chance, over its easy, standard and
hard tiers), Calibration (on the hard items: calibration error and distance to the
items' gold probabilities) and Capability (their mean). An answer JevBench cannot
parse, or whose probabilities do not add up to 1 (within 0.02), is a format
failure: wrong, and left out of Calibration.

`benchmarks/jevbench/run_vllm.py` sends DiffusionGemma, through vLLM, the request
JevBench sent it as djev (thinking) (thinking on, at most 8192 new tokens), in two ways:

- `base` (variant `unconstrained`): as JevBench ran it; the model writes its answer as
  free text, which JevBench's adapter parses.
- `ours` (variant `constrained`): as one Mosaic thinking request; the model thinks
  freely, then answers under the `{"probabilities": {...}}` schema.

```bash
git clone https://github.com/fstandhartinger/jevbench /path/to/jevbench
git -C /path/to/jevbench checkout 1bcc55e                     # the version these results used
CUDA_VISIBLE_DEVICES=0 python benchmarks/jevbench/run_vllm.py --jevbench /path/to/jevbench \
    --seed 0 --out outputs/jevbench
python benchmarks/jevbench/summary.py --jevbench /path/to/jevbench \
    outputs/jevbench/vllm_unconstrained_s0 outputs/jevbench/vllm_constrained_s0
```

It runs in the `.venv-vllm` env ([integrations/vllm](integrations/vllm/README.md)) on
one 48 GB GPU. All 231 public items (djev's row: its published per-item results,
scored the same way), 5 seeds, mean ± sd:

| | format failures | hard accuracy | Intelligence | Calibration | Capability |
|---|---|---|---|---|---|
| djev (thinking), [published](https://github.com/fstandhartinger/jevbench/blob/1bcc55e/RESULTS-v1.2.md) | 27 | 0.766 | 83.4 | **92.7** | 88.0 |
| `base` (DiffusionGemma) | 21.2 | 0.755 | 83.9 ± 2.5 | 89.9 ± 1.7 | 86.9 ± 2.0 |
| `ours` (DiffusionGemma + Mosaic) | **0.6** | **0.933** | **95.5 ± 1.1** | 92.2 ± 2.5 | **93.8 ± 1.7** |

- `base` (DiffusionGemma) is our reproduction of the [published djev (thinking) run](https://github.com/fstandhartinger/jevbench/blob/1bcc55e/docs/v1.2-additions-djev-thinking.md).
- `ours`' format failures are answers whose probabilities do not add up to 1
  (within 0.02): the constraint fixes each probability's form, not their sum.
- Capability is the average of Intelligence and Calibration.

Details: [benchmarks/jevbench](benchmarks/jevbench/README.md).

## Layout

```
mosaic/grammar/       automata (union, concatenation, star, ...), the JSON-Schema grammar,
                      and conversion from characters to the tokenizer's vocabulary
mosaic/sampling/      the token automaton as tensors, and the two samplers:
                      parallel.py (O(log L) depth) and sequential.py (O(L) depth)
mosaic/dlm/           model loading and the Dream / LLaDA / LLaDA2 denoising loops;
                      the constraint comes in as a ConstraintMatcher (mosaic/matcher.py)
mosaic/matcher.py     the engine-facing API: compile a schema, propose a block's x0 per request
mosaic/tokenizers.py  a tokenizer's family and end tokens; a grammar to its token automaton
integrations/         serving engines: an SGLang dLLM algorithm (LLaDA2), a vLLM plugin
                      (DiffusionGemma)
benchmarks/bfcl/      BFCL: data download, prompts, schema normalization, parsers, the official
                      AST checker; run_hf.py (the loops above), run_sglang.py, result summaries
benchmarks/jevbench/  JevBench with DiffusionGemma through vLLM: data.py, run_vllm.py, summary.py
examples/             the JSON-Schema demo
tests/                automata and grammar tests, and exact-distribution tests of the samplers
data/ cache/ outputs/ made by the runs (git-ignored): BFCL data, compiled automata, results
```

`python -m pytest` runs the tests (CPU; the token-level grammar
test uses the Dream tokenizer when it is available).

## TODO

- [ ] SGLang support for DiffusionGemma (today it is served through vLLM).
- [ ] The paper's complete experiments, including constraints beyond JSON Schema.
- [ ] Many more to come.

## Citation

```bibtex
@inproceedings{dang2026constrained,
  title     = {Constrained Decoding for Diffusion Language Models via Efficient Inference over Finite Automata},
  author    = {Dang, Meihua and Ermon, Stefano},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
}
```

The samplers build on [Mitigating Bias in Locally Constrained
Decoding via Tractable Proposals](https://arxiv.org/abs/2606.01926) (ICML 2026).

## License

MIT, see [LICENSE](LICENSE). `benchmarks/bfcl/checker.py` is adapted from the
[gorilla](https://github.com/ShishirPatil/gorilla) repository (Apache 2.0).
