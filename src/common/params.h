/*
 * Shared option parameter struct + CLI parsing helpers.
 * Used by both CPU and GPU implementations so they accept the same flags.
 */
#ifndef PARAMS_H
#define PARAMS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    long   n_paths;     /* number of Monte Carlo paths */
    double S0;          /* spot price */
    double K;           /* strike price */
    double r;           /* risk-free rate */
    double sigma;       /* volatility */
    double T;           /* time to maturity (years) */
    int    n_steps;     /* time steps per path (1 for European, >1 for path-dependent) */
    unsigned long seed; /* RNG seed */
    int    is_call;     /* 1 = call, 0 = put */
} mc_params_t;

static inline mc_params_t mc_default_params(void) {
    mc_params_t p;
    p.n_paths = 1000000;
    p.S0      = 100.0;
    p.K       = 100.0;
    p.r       = 0.05;
    p.sigma   = 0.20;
    p.T       = 1.0;
    p.n_steps = 1;
    p.seed    = 42UL;
    p.is_call = 1;
    return p;
}

static inline void mc_print_params(const mc_params_t *p) {
    printf("Parameters: N=%ld  S0=%.2f  K=%.2f  r=%.4f  sigma=%.4f  T=%.2f  steps=%d  seed=%lu  type=%s\n",
           p->n_paths, p->S0, p->K, p->r, p->sigma, p->T, p->n_steps, p->seed,
           p->is_call ? "call" : "put");
}

/* Minimal CLI parser. Recognized flags:
 *   --paths N  --S0 X  --K X  --r X  --sigma X  --T X  --steps N  --seed N  --put  --call
 */
static inline mc_params_t mc_parse_args(int argc, char **argv) {
    mc_params_t p = mc_default_params();
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--paths") && i+1 < argc) p.n_paths = atol(argv[++i]);
        else if (!strcmp(argv[i], "--S0")    && i+1 < argc) p.S0      = atof(argv[++i]);
        else if (!strcmp(argv[i], "--K")     && i+1 < argc) p.K       = atof(argv[++i]);
        else if (!strcmp(argv[i], "--r")     && i+1 < argc) p.r       = atof(argv[++i]);
        else if (!strcmp(argv[i], "--sigma") && i+1 < argc) p.sigma   = atof(argv[++i]);
        else if (!strcmp(argv[i], "--T")     && i+1 < argc) p.T       = atof(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i+1 < argc) p.n_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")  && i+1 < argc) p.seed    = strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--put"))  p.is_call = 0;
        else if (!strcmp(argv[i], "--call")) p.is_call = 1;
    }
    return p;
}

#endif /* PARAMS_H */
