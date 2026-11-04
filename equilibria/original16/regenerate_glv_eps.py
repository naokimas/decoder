"""Regenerate gLV equilibria with the fixed leading-eigenvalue
proximity rule: D_max = D_c - eps/alpha_max (eps = 0.5), D_min =
0.05 * D_c. Closed-form solve. Reports per-network blow-up
diagnostics (mean, max, hub localization, conditioning) so we can
confirm no pathological blow-up.

Usage:
  python3 regenerate_glv_eps.py            # all 14 nets, save
  python3 regenerate_glv_eps.py --report   # compute + report, no save
  python3 regenerate_glv_eps.py --skip spanish   # save all but spanish
"""
from __future__ import annotations
import sys, shutil
from pathlib import Path
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, spsolve

HERE = Path(__file__).resolve().parent
NETWORKS = ["dolphin", "prox", "metabolic", "euroroad", "email", "er",
            "ba", "hk", "gkk", "lfr", "flybi", "reactome",
            "route_views", "spanish"]
NSWEEP = 51
EPS = 0.5
LAMBDA_BAR = 1.0


def regen(tag, save=True):
    G = nx.read_edgelist(HERE / f"{tag}_network.edgelist", nodetype=int)
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float,
                                  format="csr")
    N = A.shape[0]
    if N <= 200:
        amax = float(np.linalg.eigvalsh(A.toarray()).max())
    else:
        amax = float(eigsh(A, k=1, which="LA",
                           return_eigenvectors=False)[0])
    Dc = (amax + 1.0) / amax
    c = amax + 1.0
    D_min = 0.05 * Dc
    D_max = Dc - EPS / amax       # lead-eig = -EPS at D_max
    grid = np.linspace(D_min, D_max, NSWEEP)
    ones = np.ones(N)
    eq = np.empty((NSWEEP, N))
    for k, D in enumerate(grid):
        if N <= 5000:
            M = D * A.toarray() - c * np.eye(N)
            eq[k] = -LAMBDA_BAR * np.linalg.solve(M, ones)
        else:
            M = (D * A - c * sp.eye(N)).tocsc()
            eq[k] = -LAMBDA_BAR * spsolve(M, ones)
    # Diagnostics.
    top = eq[-1]
    neg_frac = float((eq < 0).mean())
    hub_ratio = float(top.max() / max(top.mean(), 1e-12))
    print(f"  [{tag}] amax={amax:.2f} Dc={Dc:.5f} "
          f"D=[{D_min:.4f},{D_max:.4f}]  "
          f"top: mean={top.mean():.3f} max={top.max():.3f} "
          f"min={top.min():.4f}  hub_ratio={hub_ratio:.1f}  "
          f"neg_frac={neg_frac*100:.2f}%", flush=True)
    if save:
        csv_path = HERE / f"{tag}_glv_equilibrium_data.csv"
        backup = csv_path.with_suffix(".csv.v45-glv-backup")
        if csv_path.exists() and not backup.exists():
            shutil.copyfile(csv_path, backup)
        out = np.column_stack([grid, eq])
        header = "p," + ",".join(f"x_{i+1}" for i in range(N))
        np.savetxt(csv_path, out, delimiter=",", header=header,
                   comments="", fmt=["%.6f"] + ["%.6f"] * N)
    return dict(tag=tag, amax=amax, D_max=D_max,
                top_mean=float(top.mean()), top_max=float(top.max()),
                top_min=float(top.min()), neg_frac=neg_frac)


def main():
    args = sys.argv[1:]
    report_only = "--report" in args
    skip = []
    if "--skip" in args:
        skip = args[args.index("--skip") + 1].split(",")
    print(f"gLV regen, eps={EPS}, report_only={report_only}, skip={skip}")
    for tag in NETWORKS:
        save = (not report_only) and (tag not in skip)
        regen(tag, save=save)


if __name__ == "__main__":
    main()
