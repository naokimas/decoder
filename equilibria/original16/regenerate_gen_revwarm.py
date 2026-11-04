"""Regenerate the GEN equilibrium data on every network with the
reverse-warm-start protocol newly announced in v40:

  - x_i(0) = 10 at D = D_max for every network
  - descending sweep, each subsequent equilibrium warm-started from the
    previous one
  - data reordered to ascending D before saving

The D-grid (size 51) and D_max are taken from the existing CSV so that
the column axis is preserved.

The original (cold-start) CSVs are backed up to
{tag}_genereg_equilibrium_data.csv.coldstart-backup before being
overwritten.

A summary of new-vs-old differences (per network) is printed.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent

NETWORKS = [
    "dolphin", "prox", "metabolic", "euroroad", "email", "er", "ba",
    "hk", "gkk", "lfr", "flybi", "reactome", "route_views", "spanish",
]

# Solver settings: mirror generate_data.py.
RTOL_TIGHT, ATOL_TIGHT = 1e-7, 1e-9
RTOL_LOOSE, ATOL_LOOSE = 1e-4, 1e-6
T_FINAL = 200.0
B, F_, H_ = 1.0, 1.0, 2.0
X_INIT_REVWARM = 10.0


def make_gen_rhs(A_sparse):
    def rhs(t, x, D):
        x_pos = np.maximum(x, 0.0)
        hill = (x_pos ** H_) / (1.0 + x_pos ** H_)
        return -B * (x_pos ** F_) + D * (A_sparse @ hill)
    return rhs


def per_node_jac_sparsity(A_sparse):
    from scipy.sparse import eye as sp_eye
    N = A_sparse.shape[0]
    return ((A_sparse != 0).astype(np.int8)
            + sp_eye(N, dtype=np.int8) > 0).astype(np.int8).tocsr()


def regen_one(tag):
    csv_path = HERE / f"{tag}_genereg_equilibrium_data.csv"
    if not csv_path.exists():
        print(f"  [{tag}] CSV missing; skip"); return None
    arr_old = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    grid = arr_old[:, 0]
    Y_old = arr_old[:, 1:]
    N = Y_old.shape[1]
    M = grid.size
    D_max = float(grid[-1])

    G = nx.read_edgelist(HERE / f"{tag}_network.edgelist", nodetype=int)
    nodes = sorted(G.nodes())
    A_sparse = nx.to_scipy_sparse_array(
        G, nodelist=nodes, dtype=float, format="csr")
    rhs = make_gen_rhs(A_sparse)

    # Solver dispatch: tight LSODA for the 11 small networks, BDF+sparse
    # Jacobian for flybi (per per_node_pat criterion in generate_data.py),
    # loose tol + BDF for reactome / route_views / spanish.
    if N >= 2000:
        jac_pat = per_node_jac_sparsity(A_sparse)
        method = "BDF"
        if tag in {"reactome", "route_views", "spanish"}:
            rtol, atol = RTOL_LOOSE, ATOL_LOOSE
        else:
            rtol, atol = RTOL_TIGHT, ATOL_TIGHT
        extra = {"jac_sparsity": jac_pat}
    else:
        method = "LSODA"
        rtol, atol = RTOL_TIGHT, ATOL_TIGHT
        extra = {}

    # Reverse warm start. Iterate descending grid.
    Y_new = np.empty((M, N))
    x_prev = np.full(N, X_INIT_REVWARM)
    t0 = time.time()
    for k in range(M - 1, -1, -1):
        D = grid[k]
        sol = solve_ivp(rhs, (0.0, T_FINAL), x_prev, method=method,
                        args=(D,), rtol=rtol, atol=atol, **extra)
        if not sol.success:
            raise RuntimeError(
                f"[{tag}] solver failed at D={D}: {sol.message}")
        Y_new[k] = sol.y[:, -1]
        x_prev = Y_new[k].copy()
        if k % 10 == 0 or k == M - 1:
            print(f"    [{tag}] reverse-warm k={k:2d} D={D:.4f}  "
                  f"y_mean={Y_new[k].mean():.4f}  "
                  f"y_max={Y_new[k].max():.4f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    # Compare new vs old.
    diff = Y_new - Y_old
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    rng = float(Y_old.max() - Y_old.min())
    rmse_rel = rmse / rng if rng > 0 else float("nan")
    max_abs_diff = float(np.max(np.abs(diff)))
    mean_abs_diff = float(np.mean(np.abs(diff)))
    print(f"  [{tag}] n-RMSE(new vs old) = {rmse_rel*100:.3f}%, "
          f"max|Δ| = {max_abs_diff:.4f}, mean|Δ| = {mean_abs_diff:.4f}")

    # Back up the cold-start CSV and write the new one.
    backup_path = csv_path.with_suffix(".csv.coldstart-backup")
    if not backup_path.exists():
        shutil.copyfile(csv_path, backup_path)
    arr_new = np.column_stack([grid, Y_new])
    header = "p," + ",".join(f"x_{i+1}" for i in range(N))
    np.savetxt(csv_path, arr_new, delimiter=",",
               header=header, comments="",
               fmt=["%.6f"] + ["%.6f"] * N)
    print(f"  [{tag}] wrote new CSV; backup at {backup_path.name}")
    return dict(tag=tag, N=N, D_max=D_max,
                rmse_rel=rmse_rel, max_abs_diff=max_abs_diff,
                mean_abs_diff=mean_abs_diff)


def main():
    results = []
    for tag in NETWORKS:
        print(f"\n=== {tag.upper()} ===")
        r = regen_one(tag)
        if r is not None:
            results.append(r)
    print("\n=== summary (sorted by n-RMSE(new vs old)) ===")
    results.sort(key=lambda r: -r["rmse_rel"])
    print(f"{'network':<12s} {'N':>6s} {'D_max':>9s} "
          f"{'n-RMSE %':>10s} {'max|Δ|':>10s} {'mean|Δ|':>10s}")
    for r in results:
        print(f"{r['tag']:<12s} {r['N']:>6d} {r['D_max']:>9.4f} "
              f"{r['rmse_rel']*100:>10.3f} {r['max_abs_diff']:>10.4f} "
              f"{r['mean_abs_diff']:>10.4f}")


if __name__ == "__main__":
    main()
