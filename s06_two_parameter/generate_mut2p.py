# =====================================================================
# SI Section S6: Two-parameter families of dynamics
#
# This script: single-process driver for MUT2P
# Feeds      : Figs. S7 and S8, via compute_2d45.py
# Produces   : equilibrium CSVs in equilibria_revision/ (input data)
# Runtime    : days
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""MUT two-parameter equilibrium data: 8 D x 8 B product grid.

D grid  = linspace(0, 10.258/kbar, 8)   -- the paper's MUT rule, verbatim.
B grid  = linspace(0.08, 0.15, 8)       -- step 0.01, contains the paper's
                                           B = 0.1 exactly (index 2).

Why this B range.  The saddle-node D_c falls steeply with B (a factor ~75
across B in [0.025,0.2]), so a product grid needs a B window narrow enough
that D_c stays inside the single D range for every network.  [0.08,0.15]
is the widest such window that keeps every transition inside: at B = 0.07
`er` has D_c = 1.13 * D_max, i.e. outside.  Every B here is < the local
saddle-node B_sn = 0.2257, so the Allee bistability that gives MUT its
character is preserved on the whole grid, and every B > 0 keeps x_i
strictly positive.

Protocol is the paper's published MUT (run_all_12_dynamics_v9.py):
FORWARD WARM START in D within each B row -- the chain is seeded with
x = 0.001 at D = 0 and each later D starts from the previous
equilibrium -- BDF with jac_sparsity = A+I at rtol 1e-4 / atol 1e-6.
Unlike the paper, which stops at t = 200, each solve is extended (x10 each
time) until max_i |dx_i/dt| < 1e-6 (mut2p.equilibrate_settled), so every
cell of the grid is an equilibrium.  B rows are independent of one
another; the warm chain runs only along D.

CSV columns: D,B,x_1..x_N; row = b_idx*ND + d_idx.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
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
import mut2p as m                                            # noqa: E402

NETWORKS = ["dolphin", "prox", "metabolic", "gkk", "lfr", "er",
            "ba", "hk", "euroroad", "email", "flybi",
            "reactome", "route_views", "spanish"]
ND, NBB = 8, 8
B_LO, B_HI = 0.08, 0.15
HI_BRANCH = 2.0                       # mean x above this = upper branch
OUT = EQ                              # where compute_2d45.py reads it
OUT.mkdir(exist_ok=True)


def main():
    todo = sys.argv[1:] or NETWORKS
    shard = OUT / "_summary"
    shard.mkdir(exist_ok=True)
    for tag in todo:
        t0 = time.time()
        net = m.load_network(tag)
        N = net["N"]
        D_max = m.MUT_CONSTANT / net["kbar"]
        Ds = np.linspace(0.0, D_max, ND)
        Bs = np.linspace(B_LO, B_HI, NBB)
        print(f"\n===== {tag} =====  N={N} kbar={net['kbar']:.3f} "
              f"D_max={D_max:.4f}", flush=True)

        Y = np.empty((ND * NBB, N))
        P = np.empty((ND * NBB, 2))
        okall = True
        worst_res = 0.0
        worst_t = 0.0
        jac = m.jac_sparsity(net)
        for bi, B in enumerate(Bs):
            x_prev = np.full(N, m.X0_LOW)     # forward warm chain, per B row
            for di, D in enumerate(Ds):
                y, ok, t_used, res = m.equilibrate_settled(
                    net, D, B, x0=x_prev, jac=jac)
                worst_res = max(worst_res, res)
                worst_t = max(worst_t, t_used)
                okall = okall and ok
                x_prev = y.copy()
                r = bi * ND + di
                Y[r] = y
                P[r] = (D, B)
            blk = Y[bi * ND:(bi + 1) * ND]
            n_up = int((blk.mean(axis=1) > HI_BRANCH).sum())
            print(f"   B={B:.4f}: mean x {blk[0].mean():.4f} -> "
                  f"{blk[-1].mean():.4f}, {n_up}/{ND} on upper branch, "
                  f"min x={blk.min():.6f}  ({time.time()-t0:.0f}s)",
                  flush=True)

        hdr = "D,B," + ",".join(f"x_{i+1}" for i in range(N))
        np.savetxt(OUT / f"{tag}_mut2p_equilibrium_data.csv",
                   np.column_stack([P, Y]), delimiter=",", header=hdr,
                   comments="", fmt=["%.8g", "%.8g"] + ["%.8g"] * N)
        rec = dict(tag=tag, N=N, kbar=net["kbar"], D_max=D_max,
                   Ds=Ds.tolist(), Bs=Bs.tolist(),
                   ymin=float(Y.min()), ymax=float(Y.max()),
                   n_upper=int((Y.mean(axis=1) > HI_BRANCH).sum()),
                   all_ok=bool(okall),
                   worst_residual=float(worst_res), worst_t=float(worst_t),
                   seconds=time.time() - t0)
        (shard / f"{tag}_mut2p.json").write_text(json.dumps(rec, indent=2))
        print(f"  {tag} done in {time.time()-t0:.0f}s, "
              f"x in [{Y.min():.6g}, {Y.max():.6g}], "
              f"upper-branch cells {rec['n_upper']}/64, "
              f"worst residual {worst_res:.2e} at t={worst_t:.0f}", flush=True)


if __name__ == "__main__":
    main()
