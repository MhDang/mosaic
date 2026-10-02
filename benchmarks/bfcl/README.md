# BFCL

The [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)'s
eight AST splits (`simple`, `multiple`, `parallel`, `parallel_multiple` and their
`live_` versions), with BFCL's official system prompt and AST checker, in two call
formats: `--template json` (a JSON array of `{"name", "arguments"}` objects) or
`--template python` (BFCL's native `[f(a=1), g(b="x")]`). Each example's functions
become its constraint. Run everything from the repository root.

## Data

```bash
bash benchmarks/bfcl/download.sh    # into data/bfcl/
```

## One run: `run_hf.py`

```bash
python benchmarks/bfcl/run_hf.py --model dream --split simple --out outputs/bfcl/dream_simple
```

The accuracy goes to `<out>/summary.txt`. `--model dream | llada | llada2`,
`--template json | python`, `--sampler none` for the unconstrained baseline;
`--help` lists the rest.

## The paper's runs: `run_paper.sh`

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash benchmarks/bfcl/run_paper.sh     # into outputs/bfcl_paper
python benchmarks/bfcl/summary.py --table outputs/bfcl_paper       # the table again, at any time
```

Dream and LLaDA, temperature 0 and 1, both formats, `base` (unconstrained) and
`mosaic`, all eight splits. Finished runs are skipped, so the script can be
restarted. The largest automata need an 80 GB GPU: on a smaller one, the examples
that run out of memory are recorded as empty answers.

## Through SGLang: `run_sglang.py`

LLaDA2.0-mini (setup: [integrations/sglang](../../integrations/sglang/README.md)):

```bash
python benchmarks/bfcl/run_sglang.py --algorithm ConstrainedLowConfidence \
    --split simple --out outputs/sglang/simple_mosaic
python benchmarks/bfcl/run_sglang.py --algorithm LowConfidence \
    --split simple --out outputs/sglang/simple_base
```

`checker.py` is adapted from [gorilla](https://github.com/ShishirPatil/gorilla) (Apache 2.0).
