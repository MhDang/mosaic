#!/usr/bin/env bash
# Download the BFCL v4 AST splits and their answers into data/bfcl/.
#
# Pinned to a gorilla commit so the eval set does not drift under us.
# Usage: bash benchmarks/bfcl/download.sh [out_dir]
set -euo pipefail

COMMIT=f7cf7359b7ac615a0b294831c5ba2bc95ee4a000
BASE=https://raw.githubusercontent.com/ShishirPatil/gorilla/$COMMIT/berkeley-function-call-leaderboard/bfcl_eval/data
OUT=${1:-data/bfcl}

FILES=(
    simple_python multiple parallel parallel_multiple
    live_simple live_multiple live_parallel live_parallel_multiple
)

mkdir -p "$OUT/possible_answer"
for name in "${FILES[@]}"; do
    f=BFCL_v4_${name}.json
    curl -fsSL "$BASE/$f" -o "$OUT/$f"
    curl -fsSL "$BASE/possible_answer/$f" -o "$OUT/possible_answer/$f"
    echo "  $f"
done
echo "BFCL data @ ${COMMIT:0:12} -> $OUT"
