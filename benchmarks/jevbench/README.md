# JevBench

[JevBench](https://github.com/fstandhartinger/jevbench) asks decision models for a
probability for every option, as JSON (231 public items: 48 easy, 72 standard, 111
hard). Here DiffusionGemma answers through vLLM with Mosaic's plugin (setup:
[integrations/vllm](../../integrations/vllm/README.md), in `.venv-vllm`). Run
everything from the repository root.

Two variants of the same request (JevBench's own, thinking on, at most 8192 tokens):

- `unconstrained` (`base`): the answer is free text, parsed by JevBench.
- `constrained` (`ours`): the model thinks freely, then answers under the schema
  `{"probabilities": {<option>: <probability>, ...}}`.

## Running

```bash
git clone https://github.com/fstandhartinger/jevbench /path/to/jevbench
git -C /path/to/jevbench checkout 1bcc55e
CUDA_VISIBLE_DEVICES=0 python benchmarks/jevbench/run_vllm.py --jevbench /path/to/jevbench \
    --seed 0 --out outputs/jevbench
python benchmarks/jevbench/summary.py --jevbench /path/to/jevbench \
    outputs/jevbench/vllm_unconstrained_s0 outputs/jevbench/vllm_constrained_s0
```

The defaults fit one 48 GB GPU. `--variants` runs one variant and `--tiers` /
`--limit` a subset of the items; `--help` lists the rest. `summary.py` scores the
runs by JevBench's rules (v1.2).

## Results

All 231 public items (djev's row: its published per-item results, scored the same
way), 5 seeds, mean ± sd:

| | format failures | hard accuracy | Intelligence | Calibration | Capability |
|---|---|---|---|---|---|
| djev (thinking), [published](https://github.com/fstandhartinger/jevbench/blob/1bcc55e/RESULTS-v1.2.md) | 27 | 0.766 | 83.4 | **92.7** | 88.0 |
| `base` (DiffusionGemma) | 21.2 | 0.755 | 83.9 ± 2.5 | 89.9 ± 1.7 | 86.9 ± 2.0 |
| `ours` (DiffusionGemma + Mosaic) | **0.6** | **0.933** | **95.5 ± 1.1** | 92.2 ± 2.5 | **93.8 ± 1.7** |

- `base` (DiffusionGemma) is our reproduction of the [published djev (thinking) run](https://github.com/fstandhartinger/jevbench/blob/1bcc55e/docs/v1.2-additions-djev-thinking.md).
- `ours`' format failures are answers whose probabilities do not add up to 1
  (within 0.02): the constraint fixes each probability's form, not their sum.
- Capability is the average of Intelligence and Calibration.

Accuracy per tier (a format failure counts as wrong):

| | easy (48) | standard (72) | hard (111) |
|---|---|---|---|
| djev (thinking), published | 0.958 | 0.986 | 0.766 |
| `base` (DiffusionGemma) | 0.983 ± 0.017 | **0.994 ± 0.008** | 0.755 ± 0.034 |
| `ours` (DiffusionGemma + Mosaic) | **1.000 ± 0.000** | **0.994 ± 0.008** | **0.933 ± 0.023** |
