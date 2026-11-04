"""Regenerate SIS and DW equilibrium data for the first 12 networks under
the per-network rescaled control ranges adopted in v3 of the paper draft:

  SIS: lambda in [0, lambda_max],  lambda_max = 1 / (0.15 * alpha_max)
       so that the epidemic threshold lambda_c = 1/alpha_max sits at the
       15th percentile of the sweep.

  DW:  D in [0, D_max],  D_max = D_max_dolphin * <k>_dolphin / <k>_net
       so that the average coupling input D * <k> is constant across
       networks. With D_max_dolphin = 0.5 (selected so that the saddle
       node, located at D ~ 0.19 in dolphin, lies near the 38th percentile
       of the sweep), the constant is 0.5 * 5.129... = 2.5645.

We use LSODA with rtol=1e-7, atol=1e-9 for all 12 networks (under the new
ranges DW is no longer in the strongly-coupled stiff regime). 51 sweep
points per (network, dynamics).

Existing CSVs for SIS and DW are overwritten. The bifurcation_ranges.csv
of each network is rewritten with the updated SIS and DW rows; all other
rows are preserved verbatim.
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

# 12 networks ordered by N (smallest first).
NETWORKS = [
    "dolphin", "prox", "metabolic", "gkk", "lfr", "er",
    "ba", "hk", "euroroad", "email", "flybi", "reactome",
]

# Anchor: dolphin is the smallest network. Its D_bif ~ 0.19 sits at 38% of
# the chosen sweep [0, 0.5]. The product D_max * <k> stays constant.
D_MAX_DOLPHIN = 0.5
KBAR_DOLPHIN  = 159.0 * 2.0 / 62.0   # = 5.1290..., from network_verification.csv
COUPLING_CONSTANT = D_MAX_DOLPHIN * KBAR_DOLPHIN  # 2.5645

# SIS: lambda_max so that lambda_c = 1/alpha_max is at the 15th percentile.
# lambda_c / lambda_max = 0.15 => lambda_max = (1/alpha_max) / 0.15.
SIS_PERCENTILE = 0.15

NSWEEP = 51
RTOL, ATOL = 1e-7, 1e-9


def load_network(tag: str):
    G = nx.read_edgelist(HERE / f"{tag}_network.edgelist", nodetype=int)
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")
    N = A.shape[0]
    M = A.nnz // 2
    kbar = 2.0 * M / N
    if N < 200:
        alpha_max = float(np.linalg.eigvalsh(A.toarray()).max())
    else:
        alpha_max = float(eigsh(A, k=1, which="LA",
                                return_eigenvectors=False)[0])
    return N, M, kbar, alpha_max, A


def make_sis_rhs(A):
    def rhs(t, x, lam):
        return -x + lam * (1.0 - x) * (A @ x)
    return rhs


def make_dw_rhs(A):
    R1, R2, R3 = 1.0, 3.0, 5.0
    def rhs(t, x, D):
        return -(x - R1) * (x - R2) * (x - R3) + D * (A @ x)
    return rhs


def equilibrate(rhs, p, x0, t_final=200.0):
    sol = solve_ivp(rhs, (0.0, t_final), x0, method="LSODA",
                    args=(p,), rtol=RTOL, atol=ATOL)
    if not sol.success:
        raise RuntimeError(f"ODE failed at p={p}: {sol.message}")
    return sol.y[:, -1]


def sweep(rhs, ps, x0_func, label, N, t_final=200.0):
    t0 = time.time()
    eq = np.empty((len(ps), N))
    for k, p in enumerate(ps):
        x0 = x0_func(p) if callable(x0_func) else x0_func
        eq[k] = equilibrate(rhs, p, x0.copy(), t_final=t_final)
    dt = time.time() - t0
    print(f"    {label}: {len(ps)} points in {dt:.1f}s, "
          f"y-range [{eq.min():.4f}, {eq.max():.4f}]")
    return eq, dt


def write_csv(path: Path, ps: np.ndarray, Y: np.ndarray):
    arr = np.column_stack([ps, Y])
    N = Y.shape[1]
    header = ["p"] + [f"x_{i+1}" for i in range(N)]
    np.savetxt(path, arr, delimiter=",", header=",".join(header),
               comments="", fmt=["%.6f"] + ["%.6f"] * N)


def update_ranges_csv(tag: str, new_rows: dict):
    """Rewrite <tag>_bifurcation_ranges.csv, replacing rows for the keys
    in new_rows (e.g. 'SIS', 'DW') with the new (p_min, p_max, n_sweep,
    y_min, y_max) tuples."""
    fn = HERE / f"{tag}_bifurcation_ranges.csv"
    rows = []
    if fn.exists():
        with fn.open() as f:
            header = f.readline().rstrip("\n")
            for line in f:
                parts = line.rstrip("\n").split(",")
                if not parts:
                    continue
                rows.append(parts)
    else:
        header = "dynamics,p_min,p_max,n_sweep,y_min,y_max"
    by_key = {r[0]: r for r in rows}
    for k, v in new_rows.items():
        by_key[k] = [k] + [f"{v[0]:.6f}", f"{v[1]:.6f}",
                           f"{v[2]:d}", f"{v[3]:.6f}", f"{v[4]:.6f}"]
    KEYS = ["SIS", "DW", "MUT", "GEN", "WC", "gLV", "SIR", "NM",
            "EUT", "VEG", "MSIS", "MSIR"]
    out = [header]
    for k in KEYS:
        if k in by_key:
            out.append(",".join(by_key[k]))
    with fn.open("w") as f:
        f.write("\n".join(out) + "\n")


def main():
    summary = []
    for tag in NETWORKS:
        print(f"\n========== {tag} ==========")
        N, M, kbar, alpha_max, A = load_network(tag)
        lam_max = 1.0 / (SIS_PERCENTILE * alpha_max)
        D_max   = COUPLING_CONSTANT / kbar
        print(f"  N={N}, M={M}, <k>={kbar:.3f}, alpha_max={alpha_max:.3f}")
        print(f"  SIS: lambda in [0, {lam_max:.4f}]  (lambda_c = "
              f"{1.0/alpha_max:.4f} at percentile "
              f"{(1.0/alpha_max)/lam_max*100:.1f}%)")
        print(f"  DW:  D in [0, {D_max:.4f}]  (D*<k> = "
              f"{D_max*kbar:.4f}, constant)")

        # SIS
        sis_grid = np.linspace(0.0, lam_max, NSWEEP)
        sis_rhs  = make_sis_rhs(A)
        x0_sis   = np.full(N, 0.9)
        sis_Y, sis_dt = sweep(sis_rhs, sis_grid, x0_sis, "SIS", N,
                              t_final=200.0)
        write_csv(HERE / f"{tag}_sis_equilibrium_data.csv", sis_grid, sis_Y)

        # DW
        dw_grid = np.linspace(0.0, D_max, NSWEEP)
        dw_rhs  = make_dw_rhs(A)
        x0_dw   = np.ones(N)
        dw_Y, dw_dt = sweep(dw_rhs, dw_grid, x0_dw, "DW", N, t_final=200.0)
        write_csv(HERE / f"{tag}_doublewell_equilibrium_data.csv",
                  dw_grid, dw_Y)

        update_ranges_csv(tag, {
            "SIS": (0.0, lam_max, NSWEEP, float(sis_Y.min()), float(sis_Y.max())),
            "DW":  (0.0, D_max,   NSWEEP, float(dw_Y.min()),  float(dw_Y.max())),
        })
        summary.append((tag, N, kbar, alpha_max, lam_max, D_max,
                        sis_dt, dw_dt))

    # Summary
    print("\n========== summary ==========")
    print(f"{'tag':>14s} {'N':>6s} {'<k>':>7s} {'a_max':>8s} "
          f"{'lam_max':>9s} {'D_max':>9s} {'SIS s':>7s} {'DW s':>7s}")
    for r in summary:
        print(f"{r[0]:>14s} {r[1]:>6d} {r[2]:>7.2f} {r[3]:>8.2f} "
              f"{r[4]:>9.4f} {r[5]:>9.4f} {r[6]:>7.1f} {r[7]:>7.1f}")
    total_sis = sum(r[6] for r in summary)
    total_dw  = sum(r[7] for r in summary)
    print(f"{'TOTAL':>14s} {' ':>6s} {' ':>7s} {' ':>8s} {' ':>9s} {' ':>9s} "
          f"{total_sis:>7.1f} {total_dw:>7.1f}")


if __name__ == "__main__":
    main()
