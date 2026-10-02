# SGLang: JSON-Schema-constrained LLaDA2

`mosaic_low_confidence.py` is a drop-in dLLM algorithm for
[SGLang](https://github.com/sgl-project/sglang): SGLang's `LowConfidence`
decoding, except that a request carrying a JSON Schema is decoded under it with
Mosaic. Requests without a schema decode exactly as `LowConfidence`, in the same
batches. Tested with SGLang 0.5.20 and `inclusionAI/LLaDA2.0-mini`.

## Setup

```bash
uv sync --extra hf --extra sglang
source .venv/bin/activate
# link the algorithm into SGLang, and tell it where mosaic is
ln -s "$PWD/integrations/sglang/mosaic_low_confidence.py" \
    "$(python -c 'import sglang, os; print(os.path.dirname(sglang.__file__))')/srt/dllm/algorithm/"
export MOSAIC_PATH=$PWD
```

SGLang compiles some kernels on first use, which needs a CUDA 13 toolkit: point
`CUDA_HOME` at one and put its `bin/` first on `PATH`.

## Serve

```bash
python -m sglang.launch_server --model-path inclusionAI/LLaDA2.0-mini --trust-remote-code \
    --dllm-algorithm ConstrainedLowConfidence --max-running-requests 16 --port 30000
```

The schema goes in `custom_params` (or in OpenAI's `response_format`). A BFCL item
(`simple_python_0`), with BFCL's system prompt and the schema of its function calls
(run from the repository root):

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

Optional `--dllm-algorithm-config <yaml>` keys:

| key | default | |
|---|---|---|
| `threshold` | 0.95 | commit every masked position above it, else the most confident one |
| `commit_by` | `model` | commit by the model's probability, or `constrained`: by the constrained marginal |
| `temperature` | 0 | for the token drawn on each edge |
| `sampler` | `parallel` | or `sequential` |
| `matmul_dtype` | `float32` | or `float64` |
| `cache_dir` | a temporary directory | where compiled schemas are saved and reused |
| `cache_size` | 8 | compiled schemas kept on the GPU |
| `compile_workers` | CPU count − 4 (at most 32) | processes compiling new schemas |

## BFCL

```bash
python benchmarks/bfcl/run_sglang.py --algorithm ConstrainedLowConfidence \
    --split simple --out outputs/sglang/simple_mosaic
python benchmarks/bfcl/run_sglang.py --algorithm LowConfidence \
    --split simple --out outputs/sglang/simple_base
```
