/*
 * Wall-clock and CUDA event timers + CUDA error-checking macro.
 */
#ifndef TIMER_H
#define TIMER_H

#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  /* Wall-clock timer (host) — Windows */
  static inline double wall_time_sec(void) {
      LARGE_INTEGER freq, cnt;
      QueryPerformanceFrequency(&freq);
      QueryPerformanceCounter(&cnt);
      return (double)cnt.QuadPart / (double)freq.QuadPart;
  }
#else
  #include <time.h>
  /* Wall-clock timer (host) — POSIX */
  static inline double wall_time_sec(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
  }
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>

/* Standard error-check wrapper. Use on EVERY CUDA API call. */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_err));             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* Convenience: time a kernel via CUDA events. Returns ms. */
typedef struct {
    cudaEvent_t start, stop;
} cuda_timer_t;

static inline void cuda_timer_init(cuda_timer_t *t) {
    CUDA_CHECK(cudaEventCreate(&t->start));
    CUDA_CHECK(cudaEventCreate(&t->stop));
}
static inline void cuda_timer_start(cuda_timer_t *t) {
    CUDA_CHECK(cudaEventRecord(t->start, 0));
}
static inline float cuda_timer_stop(cuda_timer_t *t) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(t->stop, 0));
    CUDA_CHECK(cudaEventSynchronize(t->stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t->start, t->stop));
    return ms;
}
static inline void cuda_timer_destroy(cuda_timer_t *t) {
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
}

#endif /* __CUDACC__ */

#endif /* TIMER_H */
