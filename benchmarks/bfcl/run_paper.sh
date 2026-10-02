#!/bin/bash
# The paper's BFCL runs: Dream and LLaDA, temperature 0 and 1, both call formats, unconstrained and
# Mosaic (parallel sampler), on all eight AST splits; then one table of the results.
#
#   bash benchmarks/bfcl/download.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash benchmarks/bfcl/run_paper.sh [out_dir]   # default outputs/bfcl_paper
#
# Each run writes to <out_dir>/<template>/<model>_t<temperature>_<method>/<split>, with method base
# (unconstrained) or mosaic (the parallel sampler). A run with a summary.txt is skipped and an
# unfinished one resumed, so the script can simply be started again after an interruption.
# Several GPUs split each run's examples (accelerate). A smaller grid:
#
#   MODELS=dream TEMPLATES=json SPLITS="simple multiple" bash benchmarks/bfcl/run_paper.sh
set -u
cd "$(dirname "$0")/../.." || exit 1

OUT=${1:-outputs/bfcl_paper}
MODELS=${MODELS:-"dream llada"}
TEMPERATURES=${TEMPERATURES:-"0 1"}
TEMPLATES=${TEMPLATES:-"json python"}
METHODS=${METHODS:-"base mosaic"}
SPLITS=${SPLITS:-"simple multiple parallel parallel_multiple live_simple live_multiple live_parallel live_parallel_multiple"}

GPUS=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
if (( NUM_GPUS > 1 )); then
    LAUNCH=(accelerate launch --num_processes="$NUM_GPUS" --main_process_port="${PORT:-29511}")
else
    LAUNCH=(python)
fi

failed=()
for template in $TEMPLATES; do
for model in $MODELS; do
for temperature in $TEMPERATURES; do
for method in $METHODS; do
for split in $SPLITS; do
    dir=$OUT/$template/${model}_t${temperature}_${method}/$split
    sampler=$([[ $method == base ]] && echo none || echo parallel)
    [[ -f $dir/summary.txt ]] && continue
    echo "=== $dir"
    CUDA_VISIBLE_DEVICES=$GPUS "${LAUNCH[@]}" benchmarks/bfcl/run_hf.py --model "$model" --template "$template" \
        --temperature "$temperature" --sampler "$sampler" --split "$split" --resume --out "$dir" || failed+=("$dir")
done; done; done; done; done

python benchmarks/bfcl/summary.py --table "$OUT"
if (( ${#failed[@]} > 0 )); then
    printf 'failed: %s\n' "${failed[@]}"
    exit 1
fi
