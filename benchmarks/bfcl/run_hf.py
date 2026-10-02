"""BFCL with the Hugging Face diffusion LMs (Dream, LLaDA, LLaDA2), with or without mosaic.

    python benchmarks/bfcl/run_hf.py --model dream --split simple --out outputs/bfcl_json/simple/dream_parallel
    python benchmarks/bfcl/run_hf.py --model dream --split simple --sampler none --out outputs/bfcl_json/simple/dream_none
    accelerate launch --num_processes=4 benchmarks/bfcl/run_hf.py --model llada --template python --split live_simple \\
        --out outputs/bfcl_python/live_simple/llada_parallel       # one process per GPU

BFCL's AST splits with its official system prompt and checker, in two call
formats: ``--template json`` (a JSON array of ``{"name", "arguments"}`` objects)
or ``python`` (BFCL's native ``[f(a=1), g(b="x")]``). Each example's call format
is compiled into an automaton first, on CPU workers, into ``--cache_dir`` (cached;
``--no-precompile``: each is compiled on its GPU when its example comes up).
Writes ``generations_<rank>.json``, ``summary.txt`` and the logs to ``--out``;
``constraint`` in the summary is the fraction of outputs the automaton accepts. The
same benchmark through SGLang: ``benchmarks/bfcl/run_sglang.py``.
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from multiprocessing import get_context

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from benchmarks.bfcl.data import DATA_DIR, BFCLDataset  # noqa: E402
from benchmarks.bfcl.summary import summarize  # noqa: E402
from mosaic import dlm  # noqa: E402
from mosaic.matcher import ConstraintCompiler, ConstraintMatcher  # noqa: E402

#: Each model's generation settings (the paper's); its checkpoint is ``dlm.MODELS[name]``.
MODEL_SETTINGS = {
    "dream": {
        "dtype": "bfloat16", "max_new_tokens": 256, "steps": 128,
        "temperature": 0.0,
        "block_length": None,  # semi-autoregressive blocks; None = one block (pure diffusion)
        "alg": "entropy",  # the unconstrained baseline's unmasking order
        "alg_temp": None, "top_p": None, "top_k": None, "eps": 1e-3,
        "commit_by": "constrained",  # commit order: the exact constrained marginal
    },
    "llada": {
        "dtype": "bfloat16", "max_new_tokens": 256, "steps": 128,
        "block_length": 32, "temperature": 0.0, "cfg_scale": 0.0,
        "remasking": "low_confidence",  # or random
        "commit_by": "constrained",
    },
    "llada2": {  # 16B MoE (1.4B active); runs each block until it is unmasked, so no steps
        "dtype": "bfloat16", "max_new_tokens": 256, "block_length": 32,
        "temperature": 0.0,
        "threshold": 0.95,  # commits every masked position whose confidence exceeds it
        "eos_id": 156892,  # <|endoftext|>: decoding stops after the block that contains it
        # the model's own probability: near-1 marginals on forced tokens would pass the
        # threshold everywhere and commit too much per step
        "commit_by": "model",
    },
}

#: Generation settings a flag may override (None: the model's own).
OVERRIDES = ("max_new_tokens", "steps", "block_length", "temperature", "alg", "remasking", "threshold",
             "commit_by")

MATMUL_DTYPES = {"float32": None, "float64": torch.float64}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="dream", choices=list(MODEL_SETTINGS))
    ap.add_argument("--template", default="json", choices=["json", "python"], help="the call format")
    ap.add_argument("--split", default="simple",
                    help="simple | multiple | parallel | parallel_multiple | live_simple | live_multiple | "
                         "live_parallel | live_parallel_multiple")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--example_ids", nargs="+", default=None, help="only these examples (e.g. simple_python_49)")
    ap.add_argument("--sampler", default="parallel", choices=["parallel", "sequential", "none"],
                    help="the constrained sampler (parallel: O(log L) depth; sequential: O(L)), or none: unconstrained")
    ap.add_argument("--commit_by", choices=["constrained", "model"],
                    help="commit positions by the proposed token's exact constrained marginal, or by the "
                         "model's probability of it (default: constrained; model for llada2)")
    ap.add_argument("--matmul_dtype", default="float32", choices=list(MATMUL_DTYPES),
                    help="the parallel sampler's matrix products (float64: slower, robust to underflow)")
    ap.add_argument("--num_samples", type=int, default=1, help="generations per prompt, decoded as one batch")
    ap.add_argument("--seed", type=int, default=42)
    g = ap.add_argument_group("generation (default: the model's settings in MODEL_SETTINGS)")
    g.add_argument("--max_new_tokens", type=int)
    g.add_argument("--steps", type=int)
    g.add_argument("--block_length", type=int)
    g.add_argument("--temperature", type=float)
    g.add_argument("--alg", choices=["entropy", "maskgit_plus", "topk_margin", "origin"], help="dream")
    g.add_argument("--remasking", choices=["low_confidence", "random"], help="llada")
    g.add_argument("--threshold", type=float, help="llada2")
    ap.add_argument("--data_dir", default=DATA_DIR, help="benchmarks/bfcl/download.sh puts the data here")
    ap.add_argument("--cache_dir", default=os.path.join(REPO, "cache"), help="the automaton cache")
    ap.add_argument("--precompile", action=argparse.BooleanOptionalAction, default=True,
                    help="compile the automata on CPU workers first")
    ap.add_argument("--workers", type=int, default=16, help="CPU workers for precompiling")
    ap.add_argument("--resume", action="store_true", help="skip the examples already in --out")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    device, rank, world = accelerator.device, accelerator.process_index, accelerator.num_processes
    os.makedirs(args.out, exist_ok=True)
    setup_logging(args.out)
    settings = {"path": dlm.MODELS[args.model], **MODEL_SETTINGS[args.model]}
    settings.update({k: getattr(args, k) for k in OVERRIDES if getattr(args, k) is not None})
    dataset_args = {"split": args.split, "template": args.template, "data_dir": args.data_dir,
                    "start": args.start, "end": args.end, "example_ids": args.example_ids}
    if accelerator.is_main_process:
        with open(os.path.join(args.out, "args.json"), "w") as f:
            json.dump({"args": vars(args), "model_settings": settings}, f, indent=2)
        if args.sampler != "none" and args.precompile:
            precompile(settings["path"], dataset_args, args.cache_dir, args.workers)
    accelerator.wait_for_everyone()

    init_seed(args.seed + rank)
    model, tokenizer, family, mask_token_id = dlm.load_model(settings["path"], settings["dtype"], device)
    dataset = BFCLDataset(tokenizer, **dataset_args)
    subset = dataset.examples[rank::world]

    compiler = None
    if args.sampler != "none":  # one example at a time on the GPU: no memory cache
        compiler = ConstraintCompiler(tokenizer, sampler=args.sampler, device=device, cache_size=0,
                                      matmul_dtype=MATMUL_DTYPES[args.matmul_dtype], cache_dir=args.cache_dir)

    out_file = os.path.join(args.out, f"generations_{rank}.json")
    records, done = [], set()
    if args.resume and os.path.exists(out_file):
        with open(out_file) as f:
            records = json.load(f)
        done = {r["example_idx"] for r in records}
        logging.info(f"[rank={rank}] resuming: {len(done)} done")

    matcher = None
    for idx, example in enumerate(tqdm(subset, desc="Sampling", disable=rank != 0)):
        if example["example_idx"] in done:
            continue
        if compiler is not None:
            matcher = free(matcher)
            compiled = dataset.compile_constraint(compiler, example)
            matcher = ConstraintMatcher(compiled, max_new_tokens=settings["max_new_tokens"])

        prompt = dataset.build_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        logging.info(
            f"[rank={rank}] id={example['example_idx']} sampler={args.sampler} "
            f"nodes={matcher and matcher.compiled.num_states} edges={matcher and matcher.compiled.num_edges}"
        )
        try:
            with accelerator.autocast(), torch.inference_mode():
                generated = generate_one(args, settings, model, tokenizer, family, mask_token_id, matcher, inputs)
        except torch.cuda.OutOfMemoryError as e:
            # Very large automata can exceed GPU memory; record an empty
            # completion so the sweep finishes and the rest can be scored.
            logging.warning(f"[rank={rank}] id={example['example_idx']} OOM, skipping: {e}")
            torch.cuda.empty_cache()
            generated = {"generated_sequence": [""], "satisfies_constraint": [False],
                         "satisfy_count": 0, "wall_time": 0.0, "skipped_oom": True}

        results = dataset.post_process(example, generated["generated_sequence"])
        records.append({**example, "input_prompt": prompt, **generated, **results})
        if idx % 10 == 0:
            save(records, out_file)

    save(records, out_file)
    open(os.path.join(args.out, f"generations_{rank}.done"), "w").close()
    if all(os.path.exists(os.path.join(args.out, f"generations_{r}.done")) for r in range(world)):
        try:  # exactly one rank prints the summary
            os.close(os.open(os.path.join(args.out, "summary.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            summarize(sorted(os.path.join(args.out, f"generations_{r}.json") for r in range(world)),
                      file=os.path.join(args.out, "summary.txt"))
        except FileExistsError:
            pass


def generate_one(args, settings, model, tokenizer, family, mask_token_id, matcher, inputs):
    """Generate ``num_samples`` completions for one prompt; decode and check them."""
    input_ids = inputs["input_ids"]
    if args.num_samples > 1:
        input_ids = input_ids.expand(args.num_samples, -1).contiguous()
    prompt_len = input_ids.shape[1]

    torch.cuda.synchronize()
    start = time.perf_counter()
    output = dlm.generate(
        model, family, input_ids, mask_token_id, settings,
        matcher=matcher, commit_by=settings["commit_by"], vocab_size=len(tokenizer),
    )
    torch.cuda.synchronize()
    wall_time = time.perf_counter() - start

    generated_ids = output[:, prompt_len:]
    satisfied = matcher.compiled.sampler.accepts(generated_ids).tolist() if matcher is not None else None
    # Sampling at T > 0 can pick ids in the model's padded vocabulary tail,
    # which do not decode; clamp them into range first.
    generated_ids = generated_ids.clamp(0, len(tokenizer) - 1)
    return {
        "generated_sequence": tokenizer.batch_decode(generated_ids, skip_special_tokens=True),
        "satisfies_constraint": satisfied,
        "satisfy_count": sum(satisfied) if satisfied is not None else None,
        "wall_time": wall_time,
    }


# ---- compiling the automata ahead ----


def precompile(model_path, dataset_args, cache_dir, workers):
    """Compile the automata missing from the cache, on CPU worker processes."""
    tokenizer = dlm.load_tokenizer(model_path)
    dataset = BFCLDataset(tokenizer, **dataset_args)
    todo = [i for i, ex in enumerate(dataset.examples)
            if not os.path.isfile(os.path.join(cache_dir, dataset.constraint_key(ex), "config.json"))]
    if not todo:
        return
    t = time.perf_counter()
    workers = max(1, min(workers, len(todo)))
    with get_context("spawn").Pool(workers) as pool:  # spawn: the parent may hold a CUDA context
        pool.map(_compile_examples, [(model_path, dataset_args, cache_dir, todo[i::workers]) for i in range(workers)])
    logging.info(f"compiled {len(todo)} automata in {time.perf_counter() - t:.1f}s")


def _compile_examples(job):
    """Worker: compile these examples' automata into the cache."""
    model_path, dataset_args, cache_dir, indices = job
    torch.set_num_threads(1)
    tokenizer = dlm.load_tokenizer(model_path)
    dataset = BFCLDataset(tokenizer, **dataset_args)
    compiler = ConstraintCompiler(tokenizer, device="cpu", cache_dir=cache_dir, cache_size=0)
    for i in indices:
        dataset.compile_constraint(compiler, dataset.examples[i])


# ---- helpers ----


def setup_logging(out):
    """INFO to the console and ``info.log``, warnings also to ``warnings.log``."""
    fmt = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler, level in ((logging.StreamHandler(sys.stdout), logging.INFO),
                           (logging.FileHandler(os.path.join(out, "info.log")), logging.INFO),
                           (logging.FileHandler(os.path.join(out, "warnings.log")), logging.WARNING)):
        handler.setLevel(level)
        handler.setFormatter(fmt)
        root.addHandler(handler)


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def free(matcher):
    """Release the previous example's automaton (its emission is E x V on the GPU)."""
    if matcher is not None:
        matcher.compiled.sampler.cpu()
        del matcher
        gc.collect()
        torch.cuda.empty_cache()
    return None


def save(records, path):
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=_json_default)


def _json_default(o):
    """Sets and tuples (e.g. from Python-literal parsing) are saved as lists."""
    if isinstance(o, set):
        return sorted(o, key=repr)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
