/*
 * Shared host-side harness for all three GPU Monte Carlo variants.
 *
 * Each .cu file (mc_naive / mc_shared / mc_antithetic) launches its own kernel
 * but uses this header for everything else: arg parsing, device allocation,
 * cuRAND state init, timing, result reporting, teardown.
 *
 * Design notes:
 *  - Grid size is fixed at MC_BLOCKS_PER_SM * (device SM count). This decouples
 *    cuRAND state count from n_paths: each thread runs a grid-stride loop over
 *    n_paths / total_threads samples, so memory cost is O(total_threads), not O(N).
 *  - Each kernel calls mc_setup() once, launches its kernel, then calls
 *    mc_report() with the divisor for the mean (n_paths for naive/shared,
 *    n_pairs for antithetic).
 */
#ifndef MC_RUNNER_CUH
#define MC_RUNNER_CUH

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../common/params.h"
#include "../common/timer.h"
#include "../common/black_scholes.h"

#define MC_BLOCK_SIZE     256
#define MC_BLOCKS_PER_SM    4

typedef struct {
    mc_params_t  p;
    float        drift, diffuse, discount;
    int          grid_blocks, block_size, total_threads;
    curandState *d_states;
    float       *d_sum, *d_sum_sq;
    cuda_timer_t t_init, t_kernel;
    float        init_ms, kernel_ms;
} mc_run_t;

__global__ void mc_init_rng_kernel(curandState *states, unsigned long seed, int n_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;
    curand_init(seed, tid, 0, &states[tid]);
}

static inline void mc_setup(mc_run_t *r, int argc, char **argv) {
    r->p = mc_parse_args(argc, argv);
    mc_print_params(&r->p);

    r->drift    = (float)((r->p.r - 0.5 * r->p.sigma * r->p.sigma) * r->p.T);
    r->diffuse  = (float)(r->p.sigma * sqrt(r->p.T));
    r->discount = (float)exp(-r->p.r * r->p.T);

    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    r->block_size    = MC_BLOCK_SIZE;
    r->grid_blocks   = sm_count * MC_BLOCKS_PER_SM;
    r->total_threads = r->grid_blocks * r->block_size;

    CUDA_CHECK(cudaMalloc(&r->d_states, (size_t)r->total_threads * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&r->d_sum,    sizeof(float)));
    CUDA_CHECK(cudaMalloc(&r->d_sum_sq, sizeof(float)));
    CUDA_CHECK(cudaMemset(r->d_sum,    0, sizeof(float)));
    CUDA_CHECK(cudaMemset(r->d_sum_sq, 0, sizeof(float)));

    cuda_timer_init(&r->t_init);
    cuda_timer_init(&r->t_kernel);

    cuda_timer_start(&r->t_init);
    mc_init_rng_kernel<<<r->grid_blocks, r->block_size>>>(r->d_states, r->p.seed, r->total_threads);
    CUDA_CHECK(cudaGetLastError());
    r->init_ms = cuda_timer_stop(&r->t_init);
}

/* n_samples = divisor for the empirical mean.
 *   - naive / shared: n_paths
 *   - antithetic   : n_pairs (each pair contributes one effective sample)
 */
static inline void mc_report(mc_run_t *r, const char *kernel_label, long n_samples) {
    float h_sum = 0.0f, h_sum_sq = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum,    r->d_sum,    sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_sum_sq, r->d_sum_sq, sizeof(float), cudaMemcpyDeviceToHost));

    double mean    = (double)h_sum    / (double)n_samples;
    double mean_sq = (double)h_sum_sq / (double)n_samples;
    double var     = mean_sq - mean * mean;
    double std_err = sqrt(var / (double)n_samples);
    double price   = r->discount * mean;
    double price_err = r->discount * std_err;

    double analytical = r->p.is_call
        ? bs_call(r->p.S0, r->p.K, r->p.r, r->p.sigma, r->p.T)
        : bs_put (r->p.S0, r->p.K, r->p.r, r->p.sigma, r->p.T);

    printf("GPU (%s) Monte Carlo result:\n", kernel_label);
    printf("  estimate     = %.6f  (95%% CI: %.6f +/- %.6f)\n",
           price, price, 1.96 * price_err);
    printf("  analytical   = %.6f\n", analytical);
    printf("  abs error    = %.6f\n", fabs(price - analytical));
    printf("  init time    = %.2f ms\n", r->init_ms);
    printf("  kernel time  = %.2f ms\n", r->kernel_ms);
    printf("  throughput   = %.2f M paths/sec\n", r->p.n_paths / r->kernel_ms / 1000.0);
    printf("  grid         = %d blocks x %d threads (%d total)\n",
           r->grid_blocks, r->block_size, r->total_threads);
}

static inline void mc_teardown(mc_run_t *r) {
    cuda_timer_destroy(&r->t_init);
    cuda_timer_destroy(&r->t_kernel);
    cudaFree(r->d_states);
    cudaFree(r->d_sum);
    cudaFree(r->d_sum_sq);
}

#endif /* MC_RUNNER_CUH */
