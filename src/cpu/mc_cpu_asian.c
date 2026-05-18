/*
 * CPU baseline: single-threaded Monte Carlo for ARITHMETIC ASIAN options.
 *
 * Each path generates n_steps standard normals (one per dt = T/n_steps step),
 * advances S via exact GBM, accumulates the running mean of S, and the payoff
 * is max(mean(S) - K, 0) for a call.
 *
 * No closed-form ground truth for arithmetic Asians, so this binary serves as
 * the CPU reference that the GPU kernel must agree with (within MC error).
 * At --steps 1 the result equals the European Black-Scholes price.
 *
 * Build:  nvcc -O3 -x cu -std=c++14 -o build/mc_cpu_asian src/cpu/mc_cpu_asian.c
 * Usage:  ./build/mc_cpu_asian --paths 1000000 --steps 252
 */
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../common/params.h"
#include "../common/timer.h"
#include "../common/black_scholes.h"

static void box_muller(double u1, double u2, double *z1, double *z2) {
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    *z1 = r * cos(theta);
    *z2 = r * sin(theta);
}

static unsigned long long lcg_state;
static inline double uniform01(void) {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((lcg_state >> 11) & 0x1FFFFFFFFFFFFFULL) / (double)(1ULL << 53);
}

/* Pull one N(0,1) from a 2-deep Box-Muller cache. */
static double cached_z;
static int    cache_full = 0;
static double next_normal(void) {
    if (cache_full) { cache_full = 0; return cached_z; }
    double u1, u2, z1, z2;
    do { u1 = uniform01(); } while (u1 <= 0.0);
    u2 = uniform01();
    box_muller(u1, u2, &z1, &z2);
    cached_z = z2;
    cache_full = 1;
    return z1;
}

int main(int argc, char **argv) {
    mc_params_t p = mc_parse_args(argc, argv);
    if (p.n_steps < 1) p.n_steps = 1;
    mc_print_params(&p);
    lcg_state = p.seed;

    const double dt = p.T / (double)p.n_steps;
    const double drift_step   = (p.r - 0.5 * p.sigma * p.sigma) * dt;
    const double diffuse_step = p.sigma * sqrt(dt);
    const double discount     = exp(-p.r * p.T);
    const double inv_steps    = 1.0 / (double)p.n_steps;

    double t0 = wall_time_sec();

    double sum = 0.0, sum_sq = 0.0;
    for (long i = 0; i < p.n_paths; i++) {
        double S = p.S0, sum_S = 0.0;
        for (int step = 0; step < p.n_steps; step++) {
            double Z = next_normal();
            S = S * exp(drift_step + diffuse_step * Z);
            sum_S += S;
        }
        double mean_S = sum_S * inv_steps;
        double payoff = p.is_call ? fmax(mean_S - p.K, 0.0) : fmax(p.K - mean_S, 0.0);
        sum    += payoff;
        sum_sq += payoff * payoff;
    }

    double mean      = sum / (double)p.n_paths;
    double mean_sq   = sum_sq / (double)p.n_paths;
    double var       = mean_sq - mean * mean;
    double std_err   = sqrt(var / (double)p.n_paths);
    double price     = discount * mean;
    double price_err = discount * std_err;

    double t1 = wall_time_sec();
    double elapsed_ms = (t1 - t0) * 1000.0;

    /* When n_steps == 1, arithmetic mean(S) = S_T, so this reduces to the
     * European Black-Scholes price. Print the analytical reference for that
     * special case as a sanity check; otherwise we just print the estimate. */
    double analytical = (p.n_steps == 1)
        ? (p.is_call ? bs_call(p.S0, p.K, p.r, p.sigma, p.T)
                     : bs_put (p.S0, p.K, p.r, p.sigma, p.T))
        : 0.0;

    printf("CPU Asian Monte Carlo result:\n");
    printf("  estimate     = %.6f  (95%% CI: %.6f +/- %.6f)\n",
           price, price, 1.96 * price_err);
    if (p.n_steps == 1)
        printf("  analytical   = %.6f   (n_steps=1 reduces to European)\n", analytical);
    if (p.n_steps == 1)
        printf("  abs error    = %.6f\n", fabs(price - analytical));
    printf("  steps/path   = %d\n", p.n_steps);
    printf("  time         = %.2f ms\n", elapsed_ms);
    printf("  throughput   = %.2f M paths/sec\n", p.n_paths / elapsed_ms / 1000.0);
    return 0;
}
