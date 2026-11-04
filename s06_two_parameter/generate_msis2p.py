# =====================================================================
# SI Section S6: Two-parameter families of dynamics
#
# This script: single-process driver for MSIS2P
# Feeds      : Figs. S7 and S8, via compute_2d45.py
# Produces   : equilibrium CSVs in equilibria_revision/ (input data)
# Runtime    : days
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""MSIS on an 8x8 grid of (beta, D_I).

The grid is a product: the beta range does NOT depend on D_I, so every row of
the grid uses the same eight beta values.  That is legitimate here because
the epidemic threshold moves monotonically with D_I -- verified from
lambda_max[diag(beta k/kbar) - mu I + D_I(A diag(1/k) - I)] = 0 -- so the
largest D_I in the range is the hardest, and a beta range that works there
works for every smaller D_I.

  beta   linspace(0, beta_max, 8), with beta_max from the paper's own
         widening rule (Methods sec:autotrim): start at 1/(0.15 alpha_max)
         and double until the mean rho_I at 0.5*beta_max reaches 0.1.  The
         rule is applied AT D_I = 1.0, the hardest end.
  D_I    linspace(0.2, 1.0, 8).  The endpoints are exactly the paper's
         MSIS2 (D_I = 0.2) and MSIS1 (D_I = 1.0), so those two dynamics are
         the first and last columns of the grid.

Within each D_I row the beta sweep is REVERSE WARM, from beta_max down to 0,
which is the paper's protocol for MSIS2.  Rows are independent.

Writes <tag>_msis2p_equilibrium_data.csv with columns beta,D_I,x_1..x_N
(64 rows) to equilibria_revision/ of the input data (lib/paths.py), plus
summary.json and one summary per network in equilibria_revision/_summary/.

Usage:  python3 generate_msis2p.py [tag ...]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib")); import paths  # noqa: E402
EQ = paths.EQ_REVISION        # equilibrium CSVs; see ../README.md
sys.path.insert(0, str(HERE))
import msis2p as m                                            # noqa: E402

NETWORKS = ["dolphin", "prox", "metabolic", "gkk", "lfr", "er",
            "ba", "hk", "euroroad", "email", "flybi",
            "reactome", "route_views", "spanish"]
NB, ND = 8, 8
D_LO, D_HI = 0.2, 1.0
OUT = EQ
OUT.mkdir(exist_ok=True)


def find_beta_max(net, D_ref=1.0, verbose=False):
    """The paper's widening rule, applied at the hardest D_I."""
    y0 = m.uniform_ic(net["N"])
    bm = 1.0 / (0.15 * net["alpha_max"])
    for trial in range(20):
        y, ok = m.equilibrate(net, 0.5 * bm, D_ref, y0)
        obs = y[net["N"]:].mean()
        if verbose:
            print(f"      widen {trial}: beta_max={bm:.4f} "
                  f"mean rho_I={obs:.4f}", flush=True)
        if obs >= m.THRESHOLD:
            return float(bm), trial
        bm *= 2.0
    return float(bm), 20


def main():
    todo = sys.argv[1:] or NETWORKS
    sfile = OUT / "summary.json"
    shard = OUT / "_summary"
    shard.mkdir(exist_ok=True)
    for tag in todo:
        print(f"\n========== {tag} ==========", flush=True)
        t0 = time.time()
        net = m.load_network(tag)
        N = net["N"]
        beta_max, ntrial = find_beta_max(net, verbose=True)
        betas = np.linspace(0.0, beta_max, NB)
        D_Is = np.linspace(D_LO, D_HI, ND)
        print(f"  N={N}, alpha_max={net['alpha_max']:.3f}, "
              f"beta_max={beta_max:.4f} ({ntrial} doublings)", flush=True)

        Y = np.empty((NB * ND, N))
        P = np.empty((NB * ND, 2))
        okall = True
        for di, D_I in enumerate(D_Is):
            y_prev = m.uniform_ic(N)           # reverse warm: start at the top
            for bi in range(NB - 1, -1, -1):
                y, ok = m.equilibrate(net, betas[bi], D_I, y_prev)
                okall = okall and ok
                y_prev = y.copy()
                row = di * NB + bi
                Y[row] = y[N:]                 # record rho_I, as the paper does
                P[row] = (betas[bi], D_I)
            print(f"    D_I={D_I:.4f}: rho_I mean at beta_max "
                  f"{Y[di*NB + NB-1].mean():.4f}, at beta=0 "
                  f"{Y[di*NB].mean():.2e}  ({time.time()-t0:.0f}s)", flush=True)

        hdr = "beta,D_I," + ",".join(f"x_{i+1}" for i in range(N))
        np.savetxt(OUT / f"{tag}_msis2p_equilibrium_data.csv",
                   np.column_stack([P, Y]), delimiter=",", header=hdr,
                   comments="", fmt=["%.8g", "%.8g"] + ["%.8g"] * N)
        rec = dict(tag=tag, N=N, alpha_max=net["alpha_max"],
                   kbar=net["kbar"], beta_max=beta_max, n_doublings=ntrial,
                   betas=betas.tolist(), D_Is=D_Is.tolist(),
                   ymin=float(Y.min()), ymax=float(Y.max()),
                   all_ok=bool(okall), seconds=time.time() - t0)
        (shard / f"{tag}.json").write_text(json.dumps(rec, indent=2))
        merged = [json.loads((shard / f"{t}.json").read_text())
                  for t in NETWORKS if (shard / f"{t}.json").exists()]
        sfile.write_text(json.dumps(merged, indent=2))
        print(f"  done in {time.time()-t0:.0f}s, "
              f"rho_I in [{Y.min():.4g}, {Y.max():.4g}]", flush=True)


if __name__ == "__main__":
    main()
