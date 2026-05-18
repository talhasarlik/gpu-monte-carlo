/*
 * Naive GPU Monte Carlo: grid-stride loop, global atomicAdd per path.
 *
 * v1 baseline. The reduction is intentionally bad — one global atomic per path,
 * to measure the headroom for shared-memory reduction in mc_shared.cu.
 *
 * Build:  nvcc -O3 -arch=sm_75 -lineinfo -o build/mc_naive src/gpu/mc_naive.cu
 * Usage:  ./build/mc_naive --paths 1000000
 */
#include "mc_runner.cuh"

__global__ void mc_naive_kernel(
    curandState *states,
    long n_paths,
    float S0, float K, float drift, float diffuse, int is_call,
    float *d_sum, float *d_sum_sq)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;

    curandState local_state = states[tid];
    for (long i = tid; i < n_paths; i += stride) {
        float Z = curand_normal(&local_state);
        float ST = S0 * expf(drift + diffuse * Z);
        float payoff = is_call ? fmaxf(ST - K, 0.0f) : fmaxf(K - ST, 0.0f);
        /* Naive global atomic per path — slow but simple. */
        atomicAdd(d_sum,    payoff);
        atomicAdd(d_sum_sq, payoff * payoff);
    }
    states[tid] = local_state;
}

int main(int argc, char **argv) {
    mc_run_t r;
    mc_setup(&r, argc, argv);

    cuda_timer_start(&r.t_kernel);
    mc_naive_kernel<<<r.grid_blocks, r.block_size>>>(
        r.d_states, r.p.n_paths,
        (float)r.p.S0, (float)r.p.K, r.drift, r.diffuse, r.p.is_call,
        r.d_sum, r.d_sum_sq);
    CUDA_CHECK(cudaGetLastError());
    r.kernel_ms = cuda_timer_stop(&r.t_kernel);

    mc_report(&r, "naive", r.p.n_paths, /*has_analytical=*/1);
    mc_teardown(&r);
    return 0;
}
