# =====================================================================
# SI Section S5: Coupled double-well started from the upper state
#
# This script: equilibria of DW from the upper initial state
# Feeds      : Figs. S5 and S6, via compute_dw45.py
# Produces   : <tag>_doublewell_upper_equilibrium_data.csv in equilibria_revision/doublewell_upper/ (input data)
# Runtime    : hours
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""DW started from the UPPER well instead of the lower one.

The published DW sweeps for the first 12 networks come from
equilibria/original16/regenerate_sis_dw_v3.py, NOT from run_all_12_dynamics_v9.py
(the two differ: v3 uses LSODA at rtol 1e-7 / atol 1e-9 with t_final = 200
and D_max = 2.5645/<k>, whereas v9 uses BDF at 1e-4/1e-6 with t_final = 50
and the rounded constant 2.564).  We mirror v3 exactly and change ONE thing,
the initial condition:

    dx_i/dt = -(x_i - 1)(x_i - 3)(x_i - 5) + D * sum_j A_ij x_j

    published ("lower"):  cold start at x_i(0) = 1, the lower stable root
    this run   ("upper"):  cold start at x_i(0) = 5, the upper stable root

The control-parameter grid is read from the published CSV itself, so the two
branches are compared at exactly the same 51 values of D.  Node ordering is
nodelist=sorted(G.nodes()), as the paper uses.

Usage
  python3 generate_dwup.py                     # upper branch, first 10 nets
  python3 generate_dwup.py --bdf flybi ...     # upper branch, large nets
  python3 generate_dwup.py --verify            # re-run the LOWER branch and
                                               # compare against published

Writes <tag>_doublewell_upper_equilibrium_data.csv to
equilibria_revision/doublewell_upper/ of the input data (lib/paths.py).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib")); import paths  # noqa: E402
EQ = paths.EQ_REVISION        # equilibrium CSVs; see ../README.md
PUBDIR = paths.EQ_SUBMITTED
OUT = EQ / "doublewell_upper"
OUT.mkdir(exist_ok=True)

RTOL, ATOL = 1e-7, 1e-9          # regenerate_sis_dw_v3.py
T_FINAL = 200.0                  # regenerate_sis_dw_v3.py
X_LOWER, X_UPPER = 1.0, 5.0
NETS = ["dolphin", "prox", "metabolic", "gkk", "lfr", "er", "ba", "hk",
        "euroroad", "email"]
BIG = ["flybi", "reactome", "route_views", "spanish"]

# For the four largest networks LSODA at rtol 1e-7 costs hours per sweep,
# so we integrate them with BDF and a sparse Jacobian at rtol 1e-4 / atol
# 1e-6 instead (--bdf).  On the UPPER branch the two agree to machine
# precision: at the midpoint of the sweep, max|LSODA - BDF| is 1.1e-14 on
# reactome and 2.8e-13 on spanish, because the upper equilibrium is unique
# and smooth.
BDF_RTOL, BDF_ATOL = 1e-4, 1e-6


def load_A(tag):
    G = nx.read_edgelist(paths.network_file(tag), nodetype=int)
    nodes = sorted(G.nodes())
    return nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float,
                                    format="csr")


def published(tag):
    arr = np.loadtxt(PUBDIR / f"{tag}_doublewell_equilibrium_data.csv",
                     delimiter=",", skiprows=1)
    return arr[:, 0], arr[:, 1:]


def make_dw(A):
    def rhs(t, x, D):
        return -(x - 1.0) * (x - 3.0) * (x - 5.0) + D * (A @ x)
    return rhs


def sweep(tag, x_init, bdf=False):
    A = load_A(tag)
    N = A.shape[0]
    grid, _ = published(tag)
    rhs = make_dw(A)
    eq = np.empty((len(grid), N))
    kw = (dict(method="BDF", rtol=BDF_RTOL, atol=BDF_ATOL,
               jac_sparsity=A + sp.eye(N)) if bdf
          else dict(method="LSODA", rtol=RTOL, atol=ATOL))
    t0 = time.time()
    for k, D in enumerate(grid):
        sol = solve_ivp(rhs, (0.0, T_FINAL), np.full(N, x_init),
                        args=(D,), **kw)
        if not sol.success:
            raise RuntimeError(f"{tag}: failed at D={D}: {sol.message}")
        eq[k] = sol.y[:, -1]
    return grid, eq, N, time.time() - t0


def write_csv(path, ps, Y):
    arr = np.column_stack([ps, Y])
    np.savetxt(path, arr, delimiter=",",
               header="p," + ",".join(f"x_{i+1}" for i in range(Y.shape[1])),
               comments="", fmt=["%.6f"] + ["%.6f"] * Y.shape[1])


def main():
    verify = "--verify" in sys.argv
    bdf = "--bdf" in sys.argv
    tags = [a for a in sys.argv[1:] if not a.startswith("--")] or NETS
    for tag in tags:
        if verify:
            grid, eq, N, dt = sweep(tag, X_LOWER, bdf)
            _, pub = published(tag)
            err = np.abs(eq - pub).max()
            print(f"  {tag}: N={N} max|mine-published| = {err:.3e} "
                  f"({dt:.1f}s)", flush=True)
        else:
            grid, eq, N, dt = sweep(tag, X_UPPER, bdf)
            write_csv(OUT / f"{tag}_doublewell_upper_equilibrium_data.csv",
                      grid, eq)
            print(f"  {tag}: N={N} upper branch written ({dt:.1f}s)",
                  flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
