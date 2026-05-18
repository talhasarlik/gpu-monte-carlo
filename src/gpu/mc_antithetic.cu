/*
 * v3: Grid-stride loop + antithetic variates.
 *
 * For each Z drawn, evaluate the payoff under +Z and -Z and use the average as
 * a single effective sample. n_pairs = n_paths / 2; each pair = one sample.
 *
 * Register-accumulated, block-reduced — same memory profile as mc_shared but
 * with two payoff evaluations per RNG state read.
 *
 * Build:  nvcc -O3 -arch=sm_75 -lineinfo -o build/mc_antithetic src/gpu/mc_antithetic.cu
 */
#include "mc_runner.cuh"
#include "reduction.cuh"

__global__ void mc_antithetic_kernel(
    curandState *states,
    long n_pairs,
    float S0, float K, float drift, float diffuse, int is_call,
    float *d_sum, float *d_sum_sq)
{
    __shared__ float ssum_warps[32];
    __shared__ float ssumsq_warps[32];

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;

    curandState local_state = states[tid];
    float my_sum = 0.0f, my_sum_sq = 0.0f;

    for (long i = tid; i < n_pairs; i += stride) {
        float Z  = curand_normal(&local_state);
        float ST_p = S0 * expf(drift + diffuse * Z);
        float ST_m = S0 * expf(drift - diffuse * Z);
        float p_p  = is_call ? fmaxf(ST_p - K, 0.0f) : fmaxf(K - ST_p, 0.0f);
        float p_m  = is_call ? fmaxf(ST_m - K, 0.0f) : fmaxf(K - ST_m, 0.0f);
        float paired = 0.5f * (p_p + p_m);
        my_sum    += paired;
        my_sum_sq += paired * paired;
    }
    states[tid] = local_state;

    float bs  = block_reduce_sum(my_sum,    ssum_warps);
    float bsq = block_reduce_sum(my_sum_sq, ssumsq_warps);

    if (threadIdx.x == 0) {
        atomicAdd(d_sum,    bs);
        atomicAdd(d_sum_sq, bsq);
    }
}

int main(int argc, char **argv) {
    mc_run_t r;
    mc_setup(&r, argc, argv);

    long n_pairs = r.p.n_paths / 2;

    cuda_timer_start(&r.t_kernel);
    mc_antithetic_kernel<<<r.grid_blocks, r.block_size>>>(
        r.d_states, n_pairs,
        (float)r.p.S0, (float)r.p.K, r.drift, r.diffuse, r.p.is_call,
        r.d_sum, r.d_sum_sq);
    CUDA_CHECK(cudaGetLastError());
    r.kernel_ms = cuda_timer_stop(&r.t_kernel);

    mc_report(&r, "antithetic", n_pairs, /*has_analytical=*/1);
    mc_teardown(&r);
    return 0;
}
