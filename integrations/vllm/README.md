# vLLM: JSON-Schema-constrained DiffusionGemma

`mosaic_vllm.py` is a vLLM plugin: every DiffusionGemma request that carries a
JSON Schema is decoded under it with Mosaic. Requests without a schema decode
exactly as stock vLLM, in the same batches. Tested with vLLM
`0.29.1rc1.dev551+g1b3b88ec2` and `google/diffusiongemma-26B-A4B-it`, on one 48 GB
GPU (with `--cpu-offload-gb 16`) and on two.

## Setup

The `.venv-vllm` env installs the plugin; vLLM loads it by itself.

```bash
UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --extra vllm
source .venv-vllm/bin/activate
export MOSAIC_PATH=$PWD    # where the plugin imports mosaic from
```

## Serve

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve google/diffusiongemma-26B-A4B-it \
    --revision f7f5b7f5fa82ffc52addd066915886d497f5517b --dtype bfloat16 --tensor-parallel-size 2 \
    --diffusion-config '{"canvas_length": 128}' --kv-cache-dtype bfloat16 --attention-backend TRITON_ATTN \
    --max-num-seqs 32 --max-num-batched-tokens 2048 --enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}' --disable-custom-all-reduce
```

(`--disable-custom-all-reduce` is needed between PCIe GPUs behind one host bridge.)
The schema goes in the request's `vllm_xargs`, as a JSON string
(`SamplingParams(extra_args={"json_schema": schema})` offline). With thinking on,
`"reasoning": "true"` makes the model think freely, then answer under the schema
(`"max_thinking_tokens"`, default `max_tokens` less one canvas, closes a thinking
that runs long; `"max_answer_tokens"`, default one canvas, bounds the answer). A
[JevBench](https://github.com/fstandhartinger/jevbench) item
(`hard-sol-b-judge_hard-18`), with JevBench's system prompt (run from the
repository root):

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

Two JSON-Schema extensions help here: `"x-prob": true` (a probability with at most
two decimals) and `"x-plain": true` (a string without escapes).

Settings (environment variables):

| variable | default | |
|---|---|---|
| `MOSAIC_MATMUL_DTYPE` | `float64` | the parallel sampler's products; fp32 underflows on DiffusionGemma's peaked logits |
| `MOSAIC_CACHE_DIR` | none | compiled schemas on disk, reused across processes and runs |
| `MOSAIC_CACHE_SIZE` | 8 | compiled schemas kept on each GPU |
| `MOSAIC_ROWS_PER_CALL` | 2 | constrained canvases per step call |
| `MOSAIC_TILE_MEMORY_GB` | 6 | the free memory DiffusionGemma's sampler is told it has, the same on every GPU |
| `MOSAIC_COMMIT_RAW` | 1 | encode a committed canvas as the HF release does (0: vLLM's own) |
| `MOSAIC_ANCHOR_WINDOW` | 1 | anchor a canvas's sliding window at the canvas start, as the HF release does (0: vLLM's own) |
| `MOSAIC_EOT_TOKEN`, `MOSAIC_EOS_TOKEN` | `<turn|>`, `<pad>` for Gemma | the end-of-turn token after the value, and the padding after it |

`MOSAIC_COMMIT_RAW` and `MOSAIC_ANCHOR_WINDOW` apply to every request: they bring
vLLM's DiffusionGemma in line with the HF release.

JevBench through this plugin: [benchmarks/jevbench](../../benchmarks/jevbench/README.md).
