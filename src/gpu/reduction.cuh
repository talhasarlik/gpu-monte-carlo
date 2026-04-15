/*
 * Reusable parallel reduction primitives for Monte Carlo sum/sum-of-squares.
 *
 * Used by mc_shared.cu and mc_antithetic.cu.
 */
#ifndef REDUCTION_CUH
#define REDUCTION_CUH

#include <cuda_runtime.h>

/* Warp-level reduction using shuffle instructions.
 * Assumes blockDim.x is a multiple of 32. Returns valid sum on lane 0 of each warp.
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/* Block-level reduction in shared memory.
 * Combines warp shuffle + a final shared-memory step across warps.
 * Returns valid sum on thread 0 of the block.
 *
 * Caller must allocate shared memory: __shared__ float sdata[32];
 */
__device__ __forceinline__ float block_reduce_sum(float val, float *sdata) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) sdata[wid] = val;
    __syncthreads();

    int n_warps = blockDim.x >> 5;
    val = (threadIdx.x < n_warps) ? sdata[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

#endif /* REDUCTION_CUH */
