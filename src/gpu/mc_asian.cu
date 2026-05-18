/*
 * Arithmetic-average Asian option pricing via Monte Carlo.
 *
 * Path-dependent payoff:  max( mean(S_1, S_2, ..., S_n) - K, 0 )  for a call,
 * where S_i = exact-GBM step from S_{i-1} using dt = T / n_steps.
 *
 * Built on the same grid-stride harness as mc_shared.cu, but with an inner
 * loop over time steps per path. drift_step / diffuse_step are precomputed
 * for one dt step in mc_runner.cuh.
 *
 * Sanity check: at --steps 1, this kernel reduces exactly to the European
 * shared kernel — mean(S) = S_T and dt = T, so the per-step constants equal
 * the single-big-step constants. Useful for validating against bs_call.
 *
 * Build:  nvcc -O3 -arch=sm_75 -lineinfo -o build/mc_asian src/gpu/mc_asian.cu
 * Usage:  ./build/mc_asian --paths 1000000 --steps 252
 */
#include "mc_runner.cuh"
#include "reduction.cuh"

__global__ void mc_asian_kernel(
    curandState *states,
    long n_paths, int n_steps,
    float S0, float K,
    float drift_step, float diffuse_step,
    int is_call,
    float *d_sum, float *d_sum_sq)
{
    __shared__ float ssum_warps[32];
    __shared__ float ssumsq_warps[32];

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;

    curandState local_state = states[tid];
    float my_sum = 0.0f, my_sum_sq = 0.0f;
    float inv_steps = 1.0f / (float)n_steps;

    for (long i = tid; i < n_paths; i += stride) {
        float S     = S0;
        float sum_S = 0.0f;
        for (int step = 0; step < n_steps; step++) {
            float Z = curand_normal(&local_state);
            S = S * expf(drift_step + diffuse_step * Z);
            sum_S += S;
        }
        float mean_S = sum_S * inv_steps;
        float payoff = is_call ? fmaxf(mean_S - K, 0.0f) : fmaxf(K - mean_S, 0.0f);
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
    mc_asian_kernel<<<r.grid_blocks, r.block_size>>>(
        r.d_states, r.p.n_paths, r.p.n_steps,
        (float)r.p.S0, (float)r.p.K,
        r.drift_step, r.diffuse_step,
        r.p.is_call,
        r.d_sum, r.d_sum_sq);
    CUDA_CHECK(cudaGetLastError());
    r.kernel_ms = cuda_timer_stop(&r.t_kernel);

    /* The European Black-Scholes formula is the correct reference only when
     * n_steps == 1 (mean(S) collapses to S_T). For multi-step Asians there is
     * no closed-form ground truth; suppress that line so the output is honest. */
    mc_report(&r, "asian", r.p.n_paths, /*has_analytical=*/(r.p.n_steps == 1));
    mc_teardown(&r);
    return 0;
}
