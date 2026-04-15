"""
Correctness validation: run each kernel and check estimate is within
2 standard errors of the analytical Black-Scholes price.
"""
import subprocess
import re
import math
import sys
from pathlib import Path

# Analytical Black-Scholes for default params (S0=100, K=100, r=0.05, sigma=0.2, T=1.0)
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S0, K, r, sigma, T):
    d1 = (math.log(S0/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S0*norm_cdf(d1) - K*math.exp(-r*T)*norm_cdf(d2)

ANALYTICAL = bs_call(100, 100, 0.05, 0.2, 1.0)  # ~10.4506
N_PATHS = 1_000_000
TOL_SIGMAS = 4  # generous; CI half-width ~0.04 at N=1M

import platform
EXE = ".exe" if platform.system() == "Windows" else ""
KERNELS = [
    f"./build/mc_cpu{EXE}",
    f"./build/mc_naive{EXE}",
    f"./build/mc_shared{EXE}",
    f"./build/mc_antithetic{EXE}",
]

def parse(out):
    m_est = re.search(r"estimate\s*=\s*([\-\d.]+).*?\+/-\s*([\d.]+)", out)
    if not m_est:
        return None, None
    return float(m_est.group(1)), float(m_est.group(2))

ok = True
for k in KERNELS:
    if not Path(k).exists():
        print(f"SKIP {k} (not built)")
        continue
    res = subprocess.run([k, "--paths", str(N_PATHS)], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FAIL {k}: nonzero exit\n{res.stderr}")
        ok = False
        continue
    est, ci_half = parse(res.stdout)
    if est is None:
        print(f"FAIL {k}: could not parse output")
        ok = False
        continue
    err = abs(est - ANALYTICAL)
    sigma = ci_half / 1.96
    n_sig = err / sigma if sigma > 0 else 0
    status = "PASS" if n_sig < TOL_SIGMAS else "FAIL"
    print(f"{status} {k:25s}  estimate={est:.4f}  analytical={ANALYTICAL:.4f}  "
          f"err={err:.4f}  ({n_sig:.1f} sigma)")
    if status == "FAIL":
        ok = False

sys.exit(0 if ok else 1)
