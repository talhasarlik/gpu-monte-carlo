/*
 * v2: Shared-memory + warp-shuffle reduction.
 * Each block reduces its threads' payoffs locally, then one atomicAdd per block.
 *
 * Compared to mc_naive.cu, this should reduce global atomic contention from
 * O(n_paths) atomics to O(n_paths / blockDim.x).
 *
 * Build:  nvcc -O3 -arch=sm_75 -lineinfo -o build/mc_shared src/gpu/mc_shared.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../common/params.h"
#include "../common/timer.h"
#include "../common/black_scholes.h"
#include "reduction.cuh"

#define BLOCK_SIZE 256

__global__ void init_rng_kernel(curandState *states, unsigned long seed, long n) {
    long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curand_init(seed, tid, 0, &states[tid]);
}

__global__ void mc_shared_kernel(
    curandState *states,
    long n_paths,
    float S0, float K, float drift, float diffuse, int is_call,
    float *d_sum, float *d_sum_sq)
{
    __shared__ float ssum_warps[32];
    __shared__ float ssumsq_warps[32];

    long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    float payoff = 0.0f;
    if (tid < n_paths) {
        curandState st = states[tid];
        float Z = curand_normal(&st);
        states[tid] = st;
        float ST = S0 * expf(drift + diffuse * Z);
        payoff = is_call ? fmaxf(ST - K, 0.0f) : fmaxf(K - ST, 0.0f);
    }

    float bs   = block_reduce_sum(payoff,         ssum_warps);
    float bsq  = block_reduce_sum(payoff*payoff,  ssumsq_warps);

    if (threadIdx.x == 0) {
        atomicAdd(d_sum,    bs);
        atomicAdd(d_sum_sq, bsq);
    }
}

int main(int argc, char **argv) {
    mc_params_t p = mc_parse_args(argc, argv);
    mc_print_params(&p);

    float drift   = (float)((p.r - 0.5 * p.sigma * p.sigma) * p.T);
    float diffuse = (float)(p.sigma * sqrt(p.T));
    float discount = (float)exp(-p.r * p.T);

    curandState *d_states;
    float *d_sum, *d_sum_sq;
    CUDA_CHECK(cudaMalloc(&d_states, p.n_paths * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_sum,    sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_sq, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum,    0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum_sq, 0, sizeof(float)));

    long grid = (p.n_paths + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cuda_timer_t t_init, t_kernel;
    cuda_timer_init(&t_init);
    cuda_timer_init(&t_kernel);

    cuda_timer_start(&t_init);
    init_rng_kernel<<<grid, BLOCK_SIZE>>>(d_states, p.seed, p.n_paths);
    CUDA_CHECK(cudaGetLastError());
    float init_ms = cuda_timer_stop(&t_init);

    cuda_timer_start(&t_kernel);
    mc_shared_kernel<<<grid, BLOCK_SIZE>>>(
        d_states, p.n_paths,
        (float)p.S0, (float)p.K, drift, diffuse, p.is_call,
        d_sum, d_sum_sq);
    CUDA_CHECK(cudaGetLastError());
    float kernel_ms = cuda_timer_stop(&t_kernel);

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

    printf("GPU (shared) Monte Carlo result:\n");
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
