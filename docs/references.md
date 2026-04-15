# References

## Books

1. **P. Glasserman (2003).** *Monte Carlo Methods in Financial Engineering.* Springer.
   The reference for Monte Carlo in finance. Variance reduction (Ch. 4), GBM (Ch. 3).

2. **J. C. Hull (2017).** *Options, Futures, and Other Derivatives.* Pearson.
   General options reference; useful for understanding what we're pricing and why.

3. **W. Hwu, D. Kirk, I. El Hajj (2022).** *Programming Massively Parallel Processors,* 4th Ed.
   Course textbook. Reduction patterns (Ch. 8), parallel algorithms.

## Papers

1. **F. Black, M. Scholes (1973).** *The Pricing of Options and Corporate Liabilities.*
   Journal of Political Economy 81(3). The original.

2. **J. K. Salmon, M. A. Moraes, R. O. Dror, D. E. Shaw (2011).** *Parallel Random Numbers:
   As Easy as 1, 2, 3.* SC '11. Introduces Philox — what cuRAND uses for its fastest generator.

3. **L. Howes, D. Thomas (2007).** *Efficient Random Number Generation and Application
   Using CUDA.* GPU Gems 3, Ch. 37.

## Documentation

1. **NVIDIA cuRAND Library Programming Guide** —
   https://docs.nvidia.com/cuda/curand/index.html
2. **NVIDIA CUDA C Best Practices Guide** —
   https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
3. **NVIDIA Nsight Compute Documentation** —
   https://docs.nvidia.com/nsight-compute/
4. **NVIDIA Nsight Systems Documentation** —
   https://docs.nvidia.com/nsight-systems/

## Course Materials

- BLG 562E lecture notes (in course folder).
- M. Hrywniak slides on CUDA performance optimization (in course folder).
- Grama, Gupta, Karypis, Kumar (2003), *Introduction to Parallel Computing,* 2nd Ed., Ch. 3.

## Related Open-Source Projects (we are NOT copying from these — for inspiration only)

- NVIDIA CUDA Samples — `MonteCarloMultiGPU` and `binomialOptions` examples.
- QuantLib — CPU reference for option pricing, useful to cross-check our analytical formulas.

## Notes on Attribution

If we use any external code (even a single function), record it here:
- (none so far)
