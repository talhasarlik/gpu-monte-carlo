/*
 * CPU baseline: single-threaded Monte Carlo European option pricing.
 * Reference implementation used to validate GPU kernels.
 *
 * Build:  gcc -O3 -march=native -lm -o build/mc_cpu src/cpu/mc_cpu.c
 * Usage:  ./build/mc_cpu --paths 1000000 --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0
 */
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../common/params.h"
#include "../common/timer.h"
#include "../common/black_scholes.h"

/* Box-Muller transform: two uniform [0,1) -> two N(0,1) samples */
static void box_muller(double u1, double u2, double *z1, double *z2) {
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    *z1 = r * cos(theta);
    *z2 = r * sin(theta);
}

/* Simple LCG -> double in [0,1). Replace with Mersenne Twister later if needed. */
static unsigned long long lcg_state;
static inline double uniform01(void) {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((lcg_state >> 11) & 0x1FFFFFFFFFFFFFULL) / (double)(1ULL << 53);
}

int main(int argc, char **argv) {
    mc_params_t p = mc_parse_args(argc, argv);
    mc_print_params(&p);
    lcg_state = p.seed;

    /* Pre-compute exact-GBM constants */
    const double drift   = (p.r - 0.5 * p.sigma * p.sigma) * p.T;
    const double diffuse = p.sigma * sqrt(p.T);
    const double discount = exp(-p.r * p.T);

    double t0 = wall_time_sec();

    double sum = 0.0, sum_sq = 0.0;
    /* Process pairs to use both Box-Muller outputs */
    long n_pairs = p.n_paths / 2;
    for (long i = 0; i < n_pairs; i++) {
        double u1 = uniform01();
        double u2 = uniform01();
        double z1, z2;
        box_muller(u1, u2, &z1, &z2);

        double S1 = p.S0 * exp(drift + diffuse * z1);
        double S2 = p.S0 * exp(drift + diffuse * z2);

        double payoff1 = p.is_call ? fmax(S1 - p.K, 0.0) : fmax(p.K - S1, 0.0);
        double payoff2 = p.is_call ? fmax(S2 - p.K, 0.0) : fmax(p.K - S2, 0.0);

        sum    += payoff1 + payoff2;
        sum_sq += payoff1 * payoff1 + payoff2 * payoff2;
    }
    long n = n_pairs * 2;

    double mean    = sum / (double)n;
    double var     = (sum_sq / (double)n) - mean * mean;
    double std_err = sqrt(var / (double)n);
    double price   = discount * mean;
    double price_err = discount * std_err;

    double t1 = wall_time_sec();
    double elapsed_ms = (t1 - t0) * 1000.0;

    /* Analytical reference */
    double analytical = p.is_call
        ? bs_call(p.S0, p.K, p.r, p.sigma, p.T)
        : bs_put (p.S0, p.K, p.r, p.sigma, p.T);

    printf("CPU Monte Carlo result:\n");
    printf("  estimate     = %.6f  (95%% CI: %.6f +/- %.6f)\n",
           price, price, 1.96 * price_err);
    printf("  analytical   = %.6f\n", analytical);
    printf("  abs error    = %.6f\n", fabs(price - analytical));
    printf("  time         = %.2f ms\n", elapsed_ms);
    printf("  throughput   = %.2f M paths/sec\n", n / elapsed_ms / 1000.0);
    return 0;
}
