/*
 * Naive GPU Monte Carlo: one thread per path, global atomicAdd reduction.
 *
 * This is the v1 baseline. Intentionally NOT optimized — we want to measure
 * the "low hanging fruit" headroom for shared-memory reduction (mc_shared.cu).
 *
 * Build:  nvcc -O3 -arch=sm_75 -lineinfo -o build/mc_naive src/gpu/mc_naive.cu
 * Usage:  ./build/mc_naive --paths 1000000
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../common/params.h"
#include "../common/timer.h"
#include "../common/black_scholes.h"

#define BLOCK_SIZE 256

/* Initialize one cuRAND state per thread */
__global__ void init_rng_kernel(curandState *states, unsigned long seed, long n) {
    long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curand_init(seed, tid, 0, &states[tid]);
}

/* One thread = one path. atomicAdd into a single global accumulator. */
__global__ void mc_naive_kernel(
    curandState *states,
    long n_paths,
    float S0, float K, float drift, float diffuse, int is_call,
    float *d_sum, float *d_sum_sq)
{
    long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (tid >= n_paths) return;

    curandState local_state = states[tid];
    float Z = curand_normal(&local_state);
    states[tid] = local_state;  /* persist for next call */

    float ST = S0 * expf(drift + diffuse * Z);
    float payoff = is_call ? fmaxf(ST - K, 0.0f) : fmaxf(K - ST, 0.0f);

    /* Naive global atomic — slow but simple */
    atomicAdd(d_sum,    payoff);
    atomicAdd(d_sum_sq, payoff * payoff);
}

int main(int argc, char **argv) {
    mc_params_t p = mc_parse_args(argc, argv);
    mc_print_params(&p);

    /* GBM constants (single big step for European) */
    float drift   = (float)((p.r - 0.5 * p.sigma * p.sigma) * p.T);
    float diffuse = (float)(p.sigma * sqrt(p.T));
    float discount = (float)exp(-p.r * p.T);

    /* Allocate device memory */
    curandState *d_states;
    float *d_sum, *d_sum_sq;
    CUDA_CHECK(cudaMalloc(&d_states, p.n_paths * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_sum,    sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_sq, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum,    0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum_sq, 0, sizeof(float)));

    long grid = (p.n_paths + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* Initialize RNG states */
    cuda_timer_t t_init, t_kernel;
    cuda_timer_init(&t_init);
    cuda_timer_init(&t_kernel);

    cuda_timer_start(&t_init);
    init_rng_kernel<<<grid, BLOCK_SIZE>>>(d_states, p.seed, p.n_paths);
    CUDA_CHECK(cudaGetLastError());
    float init_ms = cuda_timer_stop(&t_init);

    /* Run Monte Carlo */
    cuda_timer_start(&t_kernel);
    mc_naive_kernel<<<grid, BLOCK_SIZE>>>(
        d_states, p.n_paths,
        (float)p.S0, (float)p.K, drift, diffuse, p.is_call,
        d_sum, d_sum_sq);
    CUDA_CHECK(cudaGetLastError());
    float kernel_ms = cuda_timer_stop(&t_kernel);

    /* Copy back */
    float h_sum = 0.0f, h_sum_sq = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum,    d_sum,    sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(float), cudaMemcpyDeviceToHost));

    double mean    = (double)h_sum    / (double)p.n_paths;
    double mean_sq = (double)h_sum_sq / (double)p.n_paths;
    double var     = mean_sq - mean * mean;
    double std_err = sqrt(var / (double)p.n_paths);
    double price   = discount * mean;
    double price_err = discount * std_err;

    double analytical = p.is_call
        ? bs_call(p.S0, p.K, p.r, p.sigma, p.T)
        : bs_put (p.S0, p.K, p.r, p.sigma, p.T);

    printf("GPU (naive) Monte Carlo result:\n");
    printf("  estimate     = %.6f  (95%% CI: %.6f +/- %.6f)\n",
           price, price, 1.96 * price_err);
    printf("  analytical   = %.6f\n", analytical);
    printf("  abs error    = %.6f\n", fabs(price - analytical));
    printf("  init time    = %.2f ms\n", init_ms);
    printf("  kernel time  = %.2f ms\n", kernel_ms);
    printf("  throughput   = %.2f M paths/sec\n", p.n_paths / kernel_ms / 1000.0);

    cuda_timer_destroy(&t_init);
    cuda_timer_destroy(&t_kernel);
    cudaFree(d_states);
    cudaFree(d_sum);
    cudaFree(d_sum_sq);
    return 0;
}
