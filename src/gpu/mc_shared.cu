/*
 * v2: Grid-stride loop + register accumulation + block-level reduction.
 *
 * Each thread accumulates a personal (sum, sum_sq) over its slice of n_paths in
 * registers, then a single block_reduce_sum collapses the block and emits one
 * atomicAdd per block per quantity.
 *
 * Compared to mc_naive: O(n_blocks) global atomics instead of O(n_paths).
 *
 * Build:  nvcc -O3 -arch=sm_75 -lineinfo -o build/mc_shared src/gpu/mc_shared.cu
 */
#include "mc_runner.cuh"
#include "reduction.cuh"

__global__ void mc_shared_kernel(
    curandState *states,
    long n_paths,
    float S0, float K, float drift, float diffuse, int is_call,
    float *d_sum, float *d_sum_sq)
{
    __shared__ float ssum_warps[32];
    __shared__ float ssumsq_warps[32];

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;

    curandState local_state = states[tid];
    float my_sum = 0.0f, my_sum_sq = 0.0f;

    for (long i = tid; i < n_paths; i += stride) {
        float Z  = curand_normal(&local_state);
        float ST = S0 * expf(drift + diffuse * Z);
        float payoff = is_call ? fmaxf(ST - K, 0.0f) : fmaxf(K - ST, 0.0f);
        my_sum    += payoff;
        my_sum_sq += payoff * payoff;
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

    cuda_timer_start(&r.t_kernel);
    mc_shared_kernel<<<r.grid_blocks, r.block_size>>>(
        r.d_states, r.p.n_paths,
        (float)r.p.S0, (float)r.p.K, r.drift, r.diffuse, r.p.is_call,
        r.d_sum, r.d_sum_sq);
    CUDA_CHECK(cudaGetLastError());
    r.kernel_ms = cuda_timer_stop(&r.t_kernel);

    mc_report(&r, "shared", r.p.n_paths, /*has_analytical=*/1);
    mc_teardown(&r);
    return 0;
}
