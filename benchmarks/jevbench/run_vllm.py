"""JevBench public items through vLLM: djev (thinking), and the same request answered under mosaic's constraint.

    CUDA_VISIBLE_DEVICES=0 python benchmarks/jevbench/run_vllm.py \\
        --jevbench /path/to/jevbench --seed 0 --out outputs/jevbench

The variants, served by vLLM with djev-dev's server settings (128-token canvases,
the checkpoint's own sampler):

  unconstrained   JevBench's OpenAI-compatible adapter's request (thinking on, at
                  most 8192 new tokens), free text, parsed by the adapter
  constrained     the same request in one pass: the model thinks freely (only
                  ending its turn is blocked), and once it closes the thinking
                  (or reaches max_tokens less one canvas) the answer follows
                  under {"probabilities": {...}} (extra_args json_schema plus
                  reasoning, the thinking markers), parsed the same way

Writes <out>/vllm_unconstrained_s<seed> and <out>/vllm_constrained_s<seed>, with records
that summary.py (next to this file) tabulates. By default every item of a variant
is sent at once and latency_s is the request's time in that loaded engine; --serial sends
one request at a time.
"""

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.environ.setdefault("MOSAIC_PATH", REPO)
# djev-dev's runtime uses the v2 model runner (DiffusionGemma's) and batch invariance; batch
# invariance replaces torch's matmul kernels with ones without fp64, which the constrained step
# needs, so both variants run without it (it changes rounding, not the sampler)
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "0")
# ... but keep that mode's communication settings (vLLM's override_envs_for_invariance): with the
# defaults, tensor parallel over two PCIe GPUs behind one host bridge stalls at the first request.
for _key, _value in {
    "VLLM_ALLREDUCE_USE_SYMM_MEM": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NCCL_LAUNCH_MODE": "GROUP",
    "NCCL_COLLNET_ENABLE": "0", "NCCL_NVLS_ENABLE": "0", "NCCL_P2P_NET_DISABLE": "1", "NCCL_MIN_NCHANNELS": "1",
    "NCCL_MAX_NCHANNELS": "1", "NCCL_PROTO": "Simple", "NCCL_ALGO": "ring,tree;allreduce:tree",
    "NCCL_NTHREADS": "1", "NCCL_SOCKET_NTHREADS": "1", "VLLM_USE_AOT_COMPILE": "0",
}.items():
    os.environ.setdefault(_key, _value)

from benchmarks.jevbench import data as J  # noqa: E402  (the JevBench request and its parse)
from benchmarks.jevbench.summary import summarize  # noqa: E402

REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"  # djev-dev's pin
#: DiffusionGemma's thinking markers, for the constrained variant's schema
REASONING = ("<|channel>thought\n", "<channel|>")


def engine(args):
    from vllm import LLM

    # djev-dev's runtime/serve.py, text only, tensor parallel over the visible GPUs
    return LLM(
        model=J.MODEL, revision=REVISION, dtype="bfloat16", seed=args.seed,
        tensor_parallel_size=args.tp, diffusion_config={"canvas_length": args.canvas},
        cpu_offload_gb=args.cpu_offload_gb,  # one 48 GB GPU holds the bf16 weights only with some in host memory
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype="bfloat16", attention_backend="TRITON_ATTN", max_num_seqs=32,
        max_num_batched_tokens=2048, enable_prefix_caching=True,
        limit_mm_per_prompt={"image": args.images, "video": 0, "audio": 0},
        # as under batch invariance: vLLM's custom all-reduce spins forever between PCIe GPUs behind
        # one host bridge; NCCL's all-reduce works
        disable_custom_all_reduce=True,
    )


def generate(llm, prompts, params, serial):
    """(RequestOutput, seconds) per prompt."""
    from vllm.inputs import TokensPrompt

    prompts = [TokensPrompt(prompt_token_ids=p) for p in prompts]
    if serial:
        out = []
        for p, sp in zip(prompts, params):
            t0 = time.perf_counter()
            (o,) = llm.generate([p], sp, use_tqdm=False)
            out.append((o, time.perf_counter() - t0))
        return out
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    return [(o, _request_seconds(o, t0)) for o in outputs]


def _request_seconds(output, t0):
    m = getattr(output, "metrics", None)
    if m is not None and getattr(m, "finished_time", None) and getattr(m, "arrival_time", None):
        return m.finished_time - m.arrival_time
    return time.perf_counter() - t0


def run_unconstrained(llm, tokenizer, tasks, serial):
    from vllm import SamplingParams

    prompts = [J.djev_prompt(tokenizer, t) for t, _ in tasks]
    # the adapter's temperature 0 is left out: vLLM refuses a temperature for diffusion models, whose
    # sampler follows the checkpoint's own schedule
    params = [SamplingParams(max_tokens=body["max_tokens"]) for body, _ in prompts]
    records = []
    for (task, tier), (body, ids), (o, seconds) in zip(tasks, prompts, generate(llm, [ids for _, ids in prompts],
                                                                               params, serial)):
        out = o.outputs[0]
        gen = list(out.token_ids)
        res = J.adapter_parse(task, out.text, {"prompt_tokens": len(ids), "completion_tokens": len(gen)}, seconds)
        records.append({"id": task.id, "variant": "unconstrained", "tier": tier,
                        "probs": res.probs if res.ok else None, "error": res.error,
                        "finish": "length" if out.finish_reason == "length" else "stop",
                        "thought_opened": J.THOUGHT_OPEN[0] in gen, "thought_closed": J.THOUGHT_CLOSE in gen,
                        "gen_ids": gen, "raw": {"text": out.text[-2000:]}, "latency_s": seconds, "steps_run": None,
                        "output_tokens": len(gen), "text": None, "input_tokens": len(ids)})
    return records


def run_constrained(llm, tokenizer, tasks, serial, batch):
    """constrained: the unconstrained variant's request, its thinking free and its answer under the schema, in one request."""
    from vllm import SamplingParams

    prompts = [J.djev_prompt(tokenizer, t) for t, _ in tasks]
    params = [SamplingParams(max_tokens=body["max_tokens"],
                             extra_args={"json_schema": J.written_schema(t), "reasoning": list(REASONING)})
              for (t, _), (body, _) in zip(tasks, prompts)]
    # in batches: every constrained request in flight holds its compiled schema on the GPU
    outputs = []
    for b in range(0, len(prompts), batch):
        outputs += generate(llm, [ids for _, ids in prompts[b:b + batch]], params[b:b + batch], serial)
    records = []
    for (task, tier), (body, ids), (o, seconds) in zip(tasks, prompts, outputs):
        out = o.outputs[0]
        gen = list(out.token_ids)
        res = J.adapter_parse(task, out.text, {"prompt_tokens": len(ids), "completion_tokens": len(gen)}, seconds)
        records.append({"id": task.id, "variant": "constrained", "tier": tier,
                        "probs": res.probs if res.ok else None, "error": res.error,
                        "finish": "length" if out.finish_reason == "length" else "stop",
                        "thought_opened": J.THOUGHT_OPEN[0] in gen, "thought_closed": J.THOUGHT_CLOSE in gen,
                        "gen_ids": gen, "raw": {"text": out.text[-2000:]}, "latency_s": seconds, "steps_run": None,
                        "output_tokens": len(gen), "text": None, "input_tokens": len(ids)})
    return records


def write(records, out_dir, tasks, jevbench):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = summarize(records, tasks, jevbench)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jevbench", required=True, help="a checkout of fstandhartinger/jevbench")
    ap.add_argument("--out", required=True, help="the directory the two runs go in")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first N items per tier")
    ap.add_argument("--tiers", nargs="+", default=list(J.TIERS.values()))
    ap.add_argument("--variants", nargs="+", default=["unconstrained", "constrained"])
    ap.add_argument("--serial", action="store_true", help="one request at a time")
    ap.add_argument("--canvas", type=int, default=128)
    # one GPU: tensor parallel 2 gave more format failures (rounding), see integrations/vllm/README.md
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--cpu_offload_gb", type=float, default=16, help="weights kept in host memory (one 48 GB GPU)")
    ap.add_argument("--max_model_len", type=int, default=16384)
    # below djev-dev's 0.85: the compiled schemas (a dense edges x vocabulary matrix each, ~140 MiB at
    # the median here) and the constrained step's tensors are outside vLLM's memory budget
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    ap.add_argument("--constrained_batch", type=int, default=1000, help="constrained requests sent to the engine at once")
    ap.add_argument("--images", type=int, default=0, help="images per prompt the engine allows (djev-dev: 6)")
    ap.add_argument("--cache_dir", default=os.path.join(REPO, "cache/jevbench"))
    ap.add_argument("--workers", type=int, default=4, help="CPU workers precompiling the schemas (each ~3 GB)")
    args = ap.parse_args()

    sys.path.insert(0, args.jevbench)
    from jevbench.tasks import load_jsonl
    from transformers import AutoTokenizer

    tasks = []
    for fname, tier in J.TIERS.items():
        if tier in args.tiers:
            tasks += [(t, tier) for t in load_jsonl(os.path.join(args.jevbench, "datasets/public", fname))[: args.limit]]
    tokenizer = AutoTokenizer.from_pretrained(J.MODEL, revision=REVISION)
    suffix = f"_s{args.seed}" + ("_serial" if args.serial else "")
    unconstrained_dir = os.path.join(args.out, "vllm_unconstrained" + suffix)
    constrained_dir = os.path.join(args.out, "vllm_constrained" + suffix)

    # the engine loads compiled schemas from the disk cache
    os.environ["MOSAIC_CACHE_DIR"] = args.cache_dir
    if "constrained" in args.variants:
        schemas = [J.written_schema(t) for t, _ in tasks]
        t0 = time.time()
        compiler = J.make_compiler(tokenizer, args.cache_dir, "cpu", compile_workers=args.workers)
        for job in [compiler.precompile_json_schema(s) for s in schemas]:
            if job is not None:  # None: already in the cache
                job.result()
        compiler.close()
        print(f"compiled {len(schemas)} automata in {time.time() - t0:.0f}s", flush=True)

    llm = engine(args)
    if "unconstrained" in args.variants:
        write(run_unconstrained(llm, tokenizer, tasks, args.serial), unconstrained_dir, tasks, args.jevbench)
    if "constrained" in args.variants:
        write(run_constrained(llm, tokenizer, tasks, args.serial, args.constrained_batch), constrained_dir, tasks, args.jevbench)


if __name__ == "__main__":
    main()
