"""Regenerate MUT, GEN, WC equilibrium data for the first 12 networks
under the per-network rescaled control ranges adopted in v3 of the paper
draft.

Anchors (chosen from the existing dolphin data, dolphin = smallest network):

  MUT (saddle-node-like, scaled by <k>):
      D_bif^{dolphin} ~ 0.78. Set D_max^{dolphin} = 2.0 so the bifurcation
      lands at 39% of the sweep, well inside the 30-70% band.
      Constant: D_max * <k>_dolphin = 2.0 * 5.129 = 10.258.
      => D_max^{net} = 10.258 / <k>_net.

  GEN (epidemic-like, scaled by alpha_max):
      D_thresh^{dolphin} ~ 0.28. Set D_max^{dolphin} = 1.6 so the threshold
      lands at 17.5% of the sweep (the user-requested 15-20% percentile).
      Constant: D_max * alpha_max_dolphin = 1.6 * 7.194 = 11.510.
      => D_max^{net} = 11.510 / alpha_max_net.

  WC (epidemic-like, scaled by alpha_max):
      D_thresh^{dolphin} ~ 1.0. Set D_max^{dolphin} = 6.0 so the threshold
      lands at 17% of the sweep.
      Constant: D_max * alpha_max_dolphin = 6.0 * 7.194 = 43.166.
      => D_max^{net} = 43.166 / alpha_max_net.

For reactome (N=5973), DW already showed that LSODA stalls and BDF +
sparse Jacobian is needed. We use the same dispatch for MUT, GEN, WC
on reactome (BDF + sparse Jacobian, loose tolerance rtol=1e-4 atol=1e-6,
t_final=50 for DW-like dynamics, t_final=200 for sigmoidal dynamics).

For the 11 smaller networks, LSODA with tight tolerance is used.

Existing CSVs for MUT, GEN, WC are overwritten. The bifurcation_ranges.csv
of each network is rewritten with the updated rows for MUT, GEN, WC; all
other rows are preserved verbatim.
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

NETWORKS = ["dolphin", "prox", "metabolic", "gkk", "lfr", "er",
            "ba", "hk", "euroroad", "email", "flybi", "reactome"]

# Anchors derived from dolphin pilot.
KBAR_DOLPHIN  = 159.0 * 2.0 / 62.0       # 5.1290
ALPHA_DOLPHIN = 7.1942                    # eigsh result
MUT_CONSTANT = 2.0 * KBAR_DOLPHIN          # 10.258
GEN_CONSTANT = 1.6 * ALPHA_DOLPHIN         # 11.510
WC_CONSTANT  = 6.0 * ALPHA_DOLPHIN         # 43.166

NSWEEP = 51

# Per-network solver tier. Reactome uses loose-tier with BDF + sparse Jac
# for the per-node dynamics other than gLV / SIS / DW (which were treated
# separately).
LARGE_NETS = {"reactome"}


def load_network(tag: str):
    G = nx.read_edgelist(HERE / f"{tag}_network.edgelist", nodetype=int)
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")
    N = A.shape[0]; M = A.nnz // 2
    kbar = 2.0 * M / N
    if N < 200:
        alpha_max = float(np.linalg.eigvalsh(A.toarray()).max())
    else:
        alpha_max = float(eigsh(A, k=1, which="LA",
                                return_eigenvectors=False)[0])
    es = []; ed = []
    A_coo = A.tocoo()
    for i, j in zip(A_coo.row, A_coo.col):
        if i < j:
            es.append(int(i)); ed.append(int(j))
    es = np.array(es); ed = np.array(ed)
    return N, M, kbar, alpha_max, A, es, ed


def make_mut_rhs(es, ed):
    B, K_, C = 0.1, 5.0, 1.0
    DTILDE, E_, H_ = 5.0, 0.9, 0.1
    def rhs(t, x, D):
        self_term = B + x * (1.0 - x / K_) * (x / C - 1.0)
        xs = x[es]; xd = x[ed]
        c_sd = xs * xd / (DTILDE + E_ * xs + H_ * xd)
        c_ds = xd * xs / (DTILDE + E_ * xd + H_ * xs)
        c = np.zeros_like(x)
        np.add.at(c, es, c_sd)
        np.add.at(c, ed, c_ds)
        return self_term + D * c
    return rhs


def make_gen_rhs(A):
    B, F_, H_ = 1.0, 1.0, 2.0
    def rhs(t, x, D):
        x_pos = np.maximum(x, 0.0)
        hill = (x_pos ** H_) / (1.0 + x_pos ** H_)
        return -B * (x_pos ** F_) + D * (A @ hill)
    return rhs


def make_wc_rhs(A):
    TAU, MU = 1.0, 3.0
    def rhs(t, x, D):
        sig = 1.0 / (1.0 + np.exp(-TAU * (x - MU)))
        return -x + D * (A @ sig)
    return rhs


def equilibrate(rhs, p, x0, A_plus_I, large, t_final=200.0):
    if large:
        method = "BDF"
        sol = solve_ivp(rhs, (0.0, t_final), x0, method=method,
                        args=(p,), rtol=1e-4, atol=1e-6,
                        jac_sparsity=A_plus_I)
    else:
        method = "LSODA"
        sol = solve_ivp(rhs, (0.0, t_final), x0, method=method,
                        args=(p,), rtol=1e-7, atol=1e-9)
    if not sol.success:
        raise RuntimeError(f"{method} failed at p={p}: {sol.message}")
    return sol.y[:, -1]


def sweep(rhs, ps, x0_func, label, N, A_plus_I, large, t_final=200.0):
    t0 = time.time()
    eq = np.empty((len(ps), N))
    for k, p in enumerate(ps):
        x0 = x0_func(p) if callable(x0_func) else x0_func
        eq[k] = equilibrate(rhs, p, x0.copy(), A_plus_I, large,
                            t_final=t_final)
    dt = time.time() - t0
    print(f"    {label}: {len(ps)} pts in {dt:.1f}s, "
          f"y-range [{eq.min():.4f}, {eq.max():.4f}]")
    return eq, dt


def write_csv(path: Path, ps: np.ndarray, Y: np.ndarray):
    arr = np.column_stack([ps, Y])
    N = Y.shape[1]
    np.savetxt(path, arr, delimiter=",",
               header="p," + ",".join(f"x_{i+1}" for i in range(N)),
               comments="", fmt=["%.6f"] + ["%.6f"] * N)


def update_ranges_csv(tag: str, new_rows: dict):
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
        N, M, kbar, alpha_max, A, es, ed = load_network(tag)
        large = tag in LARGE_NETS
        A_plus_I = A + sp.eye(N) if large else None
        mut_max = MUT_CONSTANT / kbar
        gen_max = GEN_CONSTANT / alpha_max
        wc_max  = WC_CONSTANT  / alpha_max
        print(f"  N={N}, M={M}, <k>={kbar:.3f}, alpha_max={alpha_max:.3f}, "
              f"large={large}")
        print(f"  MUT: D in [0, {mut_max:.4f}]  (D*<k> = {mut_max*kbar:.3f})")
        print(f"  GEN: D in [0, {gen_max:.4f}]  (D*alpha_max = {gen_max*alpha_max:.3f})")
        print(f"  WC:  D in [0, {wc_max:.4f}]   (D*alpha_max = {wc_max*alpha_max:.3f})")

        mut_grid = np.linspace(0.0, mut_max, NSWEEP)
        gen_grid = np.linspace(0.0, gen_max, NSWEEP)
        wc_grid  = np.linspace(0.0, wc_max,  NSWEEP)

        # MUT (low Allee branch initial)
        mut_Y, _ = sweep(make_mut_rhs(es, ed), mut_grid,
                         np.full(N, 0.001), "MUT", N, A_plus_I, large,
                         t_final=200.0)
        write_csv(HERE / f"{tag}_mutualistic_equilibrium_data.csv",
                  mut_grid, mut_Y)

        # GEN (initial 2)
        gen_Y, _ = sweep(make_gen_rhs(A), gen_grid,
                         np.full(N, 2.0), "GEN", N, A_plus_I, large,
                         t_final=200.0)
        write_csv(HERE / f"{tag}_genereg_equilibrium_data.csv",
                  gen_grid, gen_Y)

        # WC (initial 0.1)
        wc_Y, _ = sweep(make_wc_rhs(A), wc_grid,
                        np.full(N, 0.1), "WC", N, A_plus_I, large,
                        t_final=200.0)
        write_csv(HERE / f"{tag}_wc_equilibrium_data.csv",
                  wc_grid, wc_Y)

        update_ranges_csv(tag, {
            "MUT": (0.0, mut_max, NSWEEP, float(mut_Y.min()), float(mut_Y.max())),
            "GEN": (0.0, gen_max, NSWEEP, float(gen_Y.min()), float(gen_Y.max())),
            "WC":  (0.0, wc_max,  NSWEEP, float(wc_Y.min()),  float(wc_Y.max())),
        })
        summary.append((tag, N, kbar, alpha_max, mut_max, gen_max, wc_max))

    print("\n========== summary ==========")
    print(f"{'tag':>14s} {'N':>6s} {'<k>':>7s} {'a_max':>8s} "
          f"{'MUTmax':>8s} {'GENmax':>8s} {'WCmax':>8s}")
    for r in summary:
        print(f"{r[0]:>14s} {r[1]:>6d} {r[2]:>7.2f} {r[3]:>8.2f} "
              f"{r[4]:>8.4f} {r[5]:>8.4f} {r[6]:>8.4f}")


if __name__ == "__main__":
    main()
