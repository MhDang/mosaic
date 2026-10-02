"""BFCL (JSON-array calls) through SGLang's offline engine, with or without mosaic.

    python benchmarks/bfcl/run_sglang.py --algorithm ConstrainedLowConfidence --split simple \\
        --out outputs/sglang/simple_mosaic

Each request carries its functions' call format as a JSON Schema
(:func:`~benchmarks.bfcl.data.tool_calls_json_schema`, in ``custom_params``), which
``ConstrainedLowConfidence`` enforces and ``LowConfidence`` ignores. The
schemas are first compiled into ``--cache_dir`` on CPU workers, so the engine
only loads them (``--no-precompile``: the engine compiles each on first use).
Scores with BFCL's checker like the HF ``run_hf.py``; ``constraint`` in the summary is the
fraction of outputs that validate against the schema (``jsonschema``).
"""

import argparse
import json
import os
import sys
import time

import jsonschema
import yaml
from transformers import AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.environ.setdefault("MOSAIC_PATH", REPO)  # the engine's processes import the mosaic package from here

from benchmarks.bfcl import BFCLDataset  # noqa: E402
from benchmarks.bfcl.data import tool_calls_json_schema  # noqa: E402
from benchmarks.bfcl.summary import summarize  # noqa: E402
from mosaic.matcher import ConstraintCompiler  # noqa: E402


def schema_valid(text, schema):
    try:
        jsonschema.validate(json.loads(text), schema)
        return True
    except (ValueError, jsonschema.ValidationError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="inclusionAI/LLaDA2.0-mini")
    ap.add_argument("--algorithm", default="ConstrainedLowConfidence")
    ap.add_argument("--algorithm_config", default=None, help="YAML for --dllm-algorithm-config")
    ap.add_argument("--split", default="simple")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_running_requests", type=int, default=16)
    ap.add_argument("--mem_fraction_static", type=float, default=0.75,
                    help="leave room for the compiled schemas (~250 MB each on the GPU)")
    ap.add_argument("--cache_dir", default=os.path.join(REPO, "cache/sglang"))
    ap.add_argument("--workers", type=int, default=16, help="CPU workers for precompiling")
    ap.add_argument("--precompile", action=argparse.BooleanOptionalAction, default=True,
                    help="compile the schemas on CPU workers first; with --no-precompile the "
                         "engine compiles each one (on its GPU) when a request first uses it")
    ap.add_argument("--repeat", type=int, default=1,
                    help="send every example this many times (a benchmark of repeated schemas)")
    ap.add_argument("--fdfo", action=argparse.BooleanOptionalAction, default=True,
                    help="SGLang's first-done-first-out dLLM schedule (--no-fdfo: synchronous blocks)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import sglang as sgl

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataset = BFCLDataset(tokenizer, args.split, start=args.start, end=args.end)
    examples = dataset.examples * args.repeat
    prompts = [dataset.build_prompt(ex) for ex in examples]
    schemas = [tool_calls_json_schema(ex["function"]) for ex in examples]
    params = [
        {"max_new_tokens": args.max_new_tokens, "temperature": 0.0, "custom_params": {"json_schema": s}}
        for s in schemas
    ]

    os.makedirs(args.out, exist_ok=True)
    engine_kwargs = {}
    if args.algorithm.startswith("Constrained"):
        if args.precompile:
            t = time.perf_counter()
            compiler = ConstraintCompiler(tokenizer, device="cpu", cache_dir=args.cache_dir, compile_workers=args.workers)
            for job in [compiler.precompile_json_schema(s) for s in schemas]:
                if job is not None:  # None: already in the cache
                    job.result()
            compiler.close()
            print(f"precompiled {len(schemas)} schemas in {time.perf_counter() - t:.1f}s")
        algo_cfg = {}
        if args.algorithm_config:
            with open(args.algorithm_config) as f:
                algo_cfg = yaml.safe_load(f) or {}
        algo_cfg.setdefault("cache_dir", args.cache_dir)
        engine_kwargs["dllm_algorithm_config"] = os.path.join(args.out, "algorithm_config.yaml")
        with open(engine_kwargs["dllm_algorithm_config"], "w") as f:
            yaml.safe_dump(algo_cfg, f)
    elif args.algorithm_config:
        engine_kwargs["dllm_algorithm_config"] = args.algorithm_config
    engine = sgl.Engine(
        model_path=args.model, trust_remote_code=True, dllm_algorithm=args.algorithm,
        max_running_requests=args.max_running_requests, mem_fraction_static=args.mem_fraction_static,
        log_level="warning", dllm_fdfo=args.fdfo, **engine_kwargs,
    )
    start = time.perf_counter()
    outputs = engine.generate(prompts, params)
    wall_time = time.perf_counter() - start
    engine.shutdown()

    records = []
    for ex, prompt, schema, out in zip(examples, prompts, schemas, outputs):
        text = out["text"]
        results = dataset.post_process(ex, [text])
        records.append({
            **ex, "input_prompt": prompt, "generated_sequence": [text],
            "satisfies_constraint": [schema_valid(text, schema)],
            "completion_tokens": out["meta_info"]["completion_tokens"],
            "finish_reason": out["meta_info"]["finish_reason"],
            "wall_time": wall_time / len(examples), **results,
        })

    with open(os.path.join(args.out, "generations_0.json"), "w") as f:
        json.dump(records, f, indent=2)
    print(f"{args.algorithm}: {len(examples)} examples in {wall_time:.1f}s")
    summarize([os.path.join(args.out, "generations_0.json")], file=os.path.join(args.out, "summary.txt"))


if __name__ == "__main__":
    main()
