#!/usr/bin/env bash
#
# Benchmark sweep: run all kernels across a range of N, save CSV.
#
set -euo pipefail

OUT=results/benchmark.csv
mkdir -p results
echo "kernel,n_paths,kernel_ms,price,abs_error" > "$OUT"

# Auto-detect .exe suffix (Windows vs Linux)
EXE=""
[ -f "./build/mc_cpu.exe" ] && EXE=".exe"

KERNELS=("./build/mc_cpu${EXE}" "./build/mc_naive${EXE}" "./build/mc_shared${EXE}" "./build/mc_antithetic${EXE}")
SIZES=(10000 100000 1000000 10000000 25000000 50000000)

for K in "${KERNELS[@]}"; do
    name=$(basename "$K")
    for N in "${SIZES[@]}"; do
        # warmup
        "$K" --paths "$N" > /dev/null 2>&1 || continue
        # 5 runs, take median
        for i in 1 2 3 4 5; do
            out=$("$K" --paths "$N" --seed "$i" 2>&1)
            # Prefer "kernel time" (GPU), fall back to "time" (CPU)
            ms=$(echo "$out" | awk '/kernel time/ {for(i=1;i<=NF;i++){if($i~/^[0-9.]+$/){print $i; exit}}}')
            [ -z "$ms" ] && ms=$(echo "$out" | awk '/^  time/ {for(i=1;i<=NF;i++){if($i~/^[0-9.]+$/){print $i; exit}}}')
            price=$(echo "$out" | awk '/estimate/ {print $3; exit}')
            err=$(echo "$out" | awk '/abs error/ {print $NF; exit}')
            echo "$name,$N,$ms,$price,$err" >> "$OUT"
        done
        echo "  $name @ N=$N done"
    done
done

echo "Wrote $OUT"
