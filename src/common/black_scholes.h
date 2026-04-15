/*
 * Analytical Black-Scholes formula for European call/put options.
 * Used as ground truth for Monte Carlo validation.
 *
 * Reference: Black & Scholes (1973), "The Pricing of Options and Corporate Liabilities".
 */
#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H

#define _USE_MATH_DEFINES
#include <math.h>

/* Standard normal CDF using erf (accurate to ~1e-7) */
static inline double bs_norm_cdf(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

/* European call option price under Black-Scholes.
 *   S0    : spot price
 *   K     : strike price
 *   r     : risk-free rate (annualized, continuous compounding)
 *   sigma : volatility (annualized)
 *   T     : time to maturity (years)
 */
static inline double bs_call(double S0, double K, double r, double sigma, double T) {
    double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return S0 * bs_norm_cdf(d1) - K * exp(-r * T) * bs_norm_cdf(d2);
}

/* European put option price */
static inline double bs_put(double S0, double K, double r, double sigma, double T) {
    double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return K * exp(-r * T) * bs_norm_cdf(-d2) - S0 * bs_norm_cdf(-d1);
}

#endif /* BLACK_SCHOLES_H */
