"""Regenerate WC equilibrium data for all 14 networks under D_max
scaling by <k> (per the v88->v89 round).

Old (v3): D_max * alpha_max = 6.0 * 7.194 = 43.166 (fixed across nets)
New (v89): D_max * <k> = 6.0 * 5.129 = 30.774 (calibrated to dolphin
            where the equilibrium response shows a marked increase
            around D ~ 1 and D_max = 6 places this region at ~17% of
            the sweep)

Per-network D_max^{net} = 30.774 / <k>^{net}.

Overwrites {tag}_wc_equilibrium_data.csv for all 14 networks.
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import networkx as nx
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent

NETWORKS = ["dolphin", "prox", "metabolic", "gkk", "lfr", "er", "ba",
            "hk", "euroroad", "email", "flybi", "reactome",
            "route_views", "spanish"]

KBAR_DOLPHIN = 159.0 * 2.0 / 62.0       # 5.1290
WC_CONSTANT = 6.0 * KBAR_DOLPHIN         # 30.774

NSWEEP = 51
LARGE_NETS = {"reactome", "route_views", "spanish"}   # use BDF + sparse


def load_network(tag: str):
    G = nx.read_edgelist(HERE / f"{tag}_network.edgelist", nodetype=int)
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")
    N = A.shape[0]; M = A.nnz // 2
    kbar = 2.0 * M / N
    return N, M, kbar, A


def make_wc_rhs(A):
    TAU, MU = 1.0, 3.0
    def rhs(t, x, D):
        sig = 1.0 / (1.0 + np.exp(-TAU * (x - MU)))
        return -x + D * (A @ sig)
    return rhs


def equilibrate(rhs, p, x0, A_plus_I, large, t_final=200.0):
    if large:
        sol = solve_ivp(rhs, (0.0, t_final), x0, method="BDF",
                        args=(p,), rtol=1e-4, atol=1e-6,
                        jac_sparsity=A_plus_I)
    else:
        sol = solve_ivp(rhs, (0.0, t_final), x0, method="LSODA",
                        args=(p,), rtol=1e-7, atol=1e-9)
    if not sol.success:
        raise RuntimeError(f"failed at p={p}: {sol.message}")
    return sol.y[:, -1]


def write_csv(path: Path, ps: np.ndarray, Y: np.ndarray):
    arr = np.column_stack([ps, Y])
    N = Y.shape[1]
    np.savetxt(path, arr, delimiter=",",
               header="p," + ",".join(f"x_{i+1}" for i in range(N)),
               comments="", fmt=["%.6f"] + ["%.6f"] * N)


def main():
    summary = []
    for tag in NETWORKS:
        print(f"========== {tag} ==========", flush=True)
        N, M, kbar, A = load_network(tag)
        wc_max = WC_CONSTANT / kbar
        large = tag in LARGE_NETS
        A_plus_I = A + sp.eye(N) if large else None
        print(f"  N={N}, M={M}, <k>={kbar:.3f}, D_max={wc_max:.4f}")
        wc_grid = np.linspace(0.0, wc_max, NSWEEP)
        t0 = time.time()
        eq = np.empty((NSWEEP, N))
        x0 = np.full(N, 0.1)
        for k, p in enumerate(wc_grid):
            eq[k] = equilibrate(make_wc_rhs(A), p, x0.copy(),
                                A_plus_I, large, t_final=200.0)
        dt = time.time() - t0
        print(f"  WC: {NSWEEP} pts in {dt:.1f}s, "
              f"y-range [{eq.min():.4f}, {eq.max():.4f}]", flush=True)
        write_csv(HERE / f"{tag}_wc_equilibrium_data.csv", wc_grid, eq)
        summary.append((tag, N, kbar, wc_max,
                        float(eq.min()), float(eq.max()), dt))

    print("\n========== summary ==========")
    print(f"{'tag':>14s} {'N':>6s} {'<k>':>7s} {'D_max':>8s} "
          f"{'y_min':>8s} {'y_max':>9s} {'time':>7s}")
    for r in summary:
        print(f"{r[0]:>14s} {r[1]:>6d} {r[2]:>7.2f} {r[3]:>8.4f} "
              f"{r[4]:>8.4f} {r[5]:>9.4f} {r[6]:>7.1f}")


if __name__ == "__main__":
    main()
