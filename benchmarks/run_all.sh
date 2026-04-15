#!/usr/bin/env bash
#
# Benchmark sweep: run all kernels across a range of N, save CSV.
#
set -euo pipefail

OUT=results/benchmark.csv
mkdir -p results
echo "kernel,n_paths,kernel_ms,price,abs_error" > "$OUT"

KERNELS=("./build/mc_cpu" "./build/mc_naive" "./build/mc_shared" "./build/mc_antithetic")
SIZES=(10000 100000 1000000 10000000 100000000)

for K in "${KERNELS[@]}"; do
    name=$(basename "$K")
    for N in "${SIZES[@]}"; do
        # warmup
        "$K" --paths "$N" > /dev/null 2>&1 || continue
        # 5 runs, take median
        for i in 1 2 3 4 5; do
            out=$("$K" --paths "$N" --seed "$i" 2>&1)
            ms=$(echo "$out" | awk '/kernel time|time *=/ {gsub(/[^0-9.]/,"",$NF); print $NF; exit}')
            price=$(echo "$out" | awk '/estimate/ {print $3; exit}')
            err=$(echo "$out" | awk '/abs error/ {print $NF; exit}')
            echo "$name,$N,$ms,$price,$err" >> "$OUT"
        done
        echo "  $name @ N=$N done"
    done
done

echo "Wrote $OUT"
