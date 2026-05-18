# Makefile for GPU Monte Carlo Option Pricing
#
#   make            - build everything
#   make cpu        - just the CPU baseline
#   make gpu        - all GPU versions
#   make clean      - remove build/
#   make bench      - run the benchmark sweep
#
# Override architecture for A100: make ARCH=sm_80 all
ARCH    ?= sm_75
NVCC    ?= nvcc

NVCCFLAGS = -O3 -arch=$(ARCH) -lineinfo -std=c++14

SRC_CPU            = src/cpu/mc_cpu.c
SRC_CPU_ASIAN      = src/cpu/mc_cpu_asian.c
SRC_GPU_NAIVE      = src/gpu/mc_naive.cu
SRC_GPU_SHARED     = src/gpu/mc_shared.cu
SRC_GPU_ANTITHETIC = src/gpu/mc_antithetic.cu
SRC_GPU_ASIAN      = src/gpu/mc_asian.cu

BUILD = build

# On Windows without gcc, compile CPU code through nvcc (-x cu treats .c as CUDA/C++)
# On Linux/macOS, use gcc as usual
CC      ?= gcc
CFLAGS    = -O3 -march=native -std=c11 -Wall

all: cpu gpu

cpu: $(BUILD)/mc_cpu $(BUILD)/mc_cpu_asian

gpu: $(BUILD)/mc_naive $(BUILD)/mc_shared $(BUILD)/mc_antithetic $(BUILD)/mc_asian

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/mc_cpu: $(SRC_CPU) | $(BUILD)
	$(NVCC) -O3 -x cu -std=c++14 -o $@ $<

$(BUILD)/mc_cpu_asian: $(SRC_CPU_ASIAN) | $(BUILD)
	$(NVCC) -O3 -x cu -std=c++14 -o $@ $<

$(BUILD)/mc_naive: $(SRC_GPU_NAIVE) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BUILD)/mc_shared: $(SRC_GPU_SHARED) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BUILD)/mc_antithetic: $(SRC_GPU_ANTITHETIC) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BUILD)/mc_asian: $(SRC_GPU_ASIAN) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

bench: all
	bash benchmarks/run_all.sh

validate: all
	python tests/validate.py

clean:
	rm -rf $(BUILD)

.PHONY: all cpu gpu bench validate clean
