"""Constrain a diffusion LM's output to any JSON Schema.

    python examples/json_schema_demo.py --model dream \\
        --prompt "Alice is 31 and lives in Paris." \\
        --schema '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}'

Compiles the schema (``--schema``: inline JSON, or a file path) to an automaton,
generates with the constrained sampler, and validates the result with ``jsonschema``.
Supported schema keywords: ``type``, ``properties``, ``required``, ``items``,
``minItems``, ``enum``, ``const``, ``anyOf`` and ``oneOf``; objects keep the declared
property order. Pass ``--sampler none`` to compare with unconstrained decoding.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaic import dlm  # noqa: E402
from mosaic.matcher import ConstraintCompiler, ConstraintMatcher  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="dream", help="dream | llada | llada2 | a Hub path")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--schema", required=True, help="a JSON Schema: inline JSON, or a file path")
    ap.add_argument("--sampler", default="parallel", choices=["parallel", "sequential", "none"])
    ap.add_argument("--commit_by", choices=["constrained", "model"],
                    help="default: model for LLaDA2, else constrained")
    # half the paper's BFCL settings (256 tokens, 128 steps; see benchmarks/bfcl/run_hf.py): a quick demo
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--steps", type=int, default=64, help="Dream / LLaDA")
    ap.add_argument("--block_length", type=int, default=32, help="LLaDA / LLaDA2")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.schema.lstrip().startswith("{"):
        schema = json.loads(args.schema)
    else:
        with open(args.schema) as f:
            schema = json.load(f)

    model, tokenizer, family, mask_token_id = dlm.load_model(args.model)

    matcher = None
    if args.sampler != "none":
        t0 = time.time()
        compiler = ConstraintCompiler(tokenizer, sampler=args.sampler, device=model.device)
        compiled = compiler.compile_json_schema(schema)
        matcher = ConstraintMatcher(compiled, max_new_tokens=args.max_new_tokens)
        print(f"compiled schema: {compiled.num_states} states, {compiled.num_edges} edges "
              f"({time.time() - t0:.1f}s)")

    messages = [{"role": "user", "content": (
        f"{args.prompt}\n\nRespond with only a JSON value that follows this JSON Schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    settings = {
        "max_new_tokens": args.max_new_tokens, "steps": args.steps, "temperature": args.temperature,
        "block_length": args.block_length if family in ("llada", "llada2") else None,
    }
    if family == "llada2":  # stop after the block holding the end of text
        settings["eos_id"] = tokenizer.eos_token_id
    t0 = time.time()
    with torch.inference_mode():
        out = dlm.generate(model, family, input_ids, mask_token_id, settings, matcher=matcher,
                           commit_by=args.commit_by, vocab_size=len(tokenizer))
    ids = out[:, input_ids.shape[1]:].clamp(0, len(tokenizer) - 1)
    text = tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
    print(f"\n--- output ({time.time() - t0:.1f}s) ---\n{text}\n")

    try:
        import jsonschema

        jsonschema.validate(json.loads(text), schema)
        print("valid: parses as JSON and satisfies the schema")
    except Exception as e:
        print(f"invalid: {type(e).__name__}: {str(e).splitlines()[0]}")


if __name__ == "__main__":
    main()
