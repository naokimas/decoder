# =====================================================================
# SI Section S11: Cross-transfer classes with four additional dynamics
#
# This script: equilibria of AEUT1
# Feeds      : Fig. S22 and Table S14
# Produces   : equilibrium CSVs in equilibria_revision/ (input data)
# Runtime    : days -- do not re-run
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""AEUT1: the adjacency-coupled eutrophication model.

The paper's EUT1 dynamics (decoder_v100_main.tex) is diffusively (Laplacian)
coupled,

    dx_i/dt = alpha - b_i x_i + r x_i^p/(x_i^p + 1)
              + D sum_j A_ij (x_j - x_i),

with r = 1, p = 8, D = 0.05, per-node loss rates b_i ~ U[0.8,1.2] fixed for
each network, and the external loading alpha swept over [0.1,0.9].  Here we
replace the diffusive coupling by adjacency coupling:

    dx_i/dt = alpha - b_i x_i + x_i^p/(x_i^p + 1)
              + D sum_j A_ij x_j.                                  (AEUT1)

Reusing the ORIGINAL b_i.  The paper's EUT loss rates are generated in
equilibria/original16/generate_data.py by

    rng_b = np.random.default_rng(11);  b_vec = 0.8 + 0.4 * rng_b.random(N)

which is deterministic, so we regenerate exactly the same vector.  Re-running
the Laplacian EUT1 with this vector reproduces the stored
<tag>_eutroph_equilibrium_data.csv, which was checked when the data were
produced.

DIVERGENCE.  Unlike ANM, this model has no mechanism that caps the state:
the only negative feedback is the LINEAR loss -b_i x_i, and the Hill term
saturates at 1.  So the large-x behavior is governed entirely by the linear
operator

    J_inf = -diag(b) + D A,

and the dynamics is bounded if and only if lambda_max(J_inf) < 0, i.e.
roughly D alpha_max(A) < min_i b_i.  With the paper's D = 0.05 kept fixed,
that fails on every network whose alpha_max exceeds about 16, and the state
runs away to +infinity.  We therefore choose D so that D alpha_max(A) is
constant across networks, anchored at the paper's D = 0.05 on the dolphin
network, i.e. D alpha_max = 0.05 * 7.1936 = 0.35968.  This is the convention
the paper already uses for its adjacency-coupled GEN dynamics ("we keep
D_max alpha_max constant across networks"), and it makes
lambda_max(J_inf) < 0 on every network by construction.  A terminal blow-up
event stops the integration at |x| = 1e6 as a safeguard.

Everything is written into this folder; nothing outside it is touched.

Usage:  python3 generate_aeut.py [tag ...]
"""
from __future__ import annotations

# LSODA/BDF call dense LAPACK on matrices of only a few hundred to a few
# thousand rows, thousands of times per sweep point.  Multi-threaded BLAS
# pays its thread-pool synchronization cost on every one of those calls; on
# this machine that made a single sweep point 245x slower (63.7 s vs 0.26 s
# on prox).  Pin the BLAS to one thread before numpy is imported.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time
from pathlib import Path

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigsh

HERE = Path(__file__).resolve().parent
# Input and output folders are set in lib/paths.py.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib")); import paths  # noqa: E402
EQDIR = paths.EQ_SUBMITTED
OUT = paths.EQ_REVISION              # all 1-D equilibria live together
OUT.mkdir(exist_ok=True)

NETWORKS = ["dolphin", "prox", "metabolic", "gkk", "lfr", "er",
            "ba", "hk", "euroroad", "email", "flybi", "reactome",
            "route_views", "spanish"]

P_HILL = 8.0                   # EUT1 Hill exponent, as in the paper
R_RECYC = 1.0
D_PAPER = 0.05                 # the paper's EUT dispersal rate
AMAX_DOLPHIN = 7.193614015378684
COUPLING_CONSTANT = D_PAPER * AMAX_DOLPHIN     # 0.35968
A_MIN, A_MAX = 0.1, 0.9        # the paper's alpha sweep; see README
NSWEEP = 51
X0 = 0.5                       # the paper's EUT cold start
T_FINAL = 200.0
RTOL, ATOL = 1e-7, 1e-9
# Settling criterion.  As for ANM, a node sitting exactly on its fold --- and
# the p = 8 Hill switch makes that fold very sharp --- relaxes algebraically,
# so no finite horizon reaches an arbitrarily small residual.  Stop at
# max_i |dx_i/dt| < RES_TOL or at T_CAP, whichever comes first, and report
# both the residual and how many points hit the cap.
RES_TOL = 1e-6
T_CAP = 2.0e4
BLOWUP = 1e6
BIG_N = 1500
SEED_B = 11


def loss_rates(N):
    """The paper's EUT per-node loss rates, regenerated bit-for-bit."""
    return 0.8 + 0.4 * np.random.default_rng(SEED_B).random(N)


def load_network(tag):
    G = nx.read_edgelist(paths.network_file(tag), nodetype=int)
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")
    N = A.shape[0]
    M = A.nnz // 2
    k = np.asarray(A.sum(axis=1)).ravel()
    if N < 1500:
        amax = float(np.linalg.eigvalsh(A.toarray()).max())
    else:
        amax = float(eigsh(A, k=1, which="LA", return_eigenvectors=False)[0])
    return dict(tag=tag, N=N, M=M, kbar=2.0 * M / N, A=A, k=k, amax=amax)


def lam_max_linear(A, b, D):
    """Leading eigenvalue of -diag(b) + D A: the exact boundedness test."""
    J = (-sp.diags(b) + D * A).tocsr()
    if A.shape[0] < 1500:
        return float(np.linalg.eigvalsh(J.toarray()).max())
    return float(eigsh(J, k=1, which="LA", return_eigenvectors=False)[0])


def rhs_factory(A, b, D, alpha):
    def rhs(t, x):
        xp = np.maximum(x, 0.0) ** P_HILL
        return alpha - b * x + R_RECYC * xp / (xp + 1.0) + D * (A @ x)
    return rhs


def jac_factory(A, b, D):
    DA = (D * A).tocsr()

    def jac(t, x):
        xc = np.maximum(x, 0.0)
        xp = xc ** P_HILL
        dhill = P_HILL * xc ** (P_HILL - 1.0) / (xp + 1.0) ** 2
        return sp.diags(-b + dhill).tocsr() + DA
    return jac


def equilibrate(A, b, D, alpha, x0, t_first=T_FINAL, t_cap=T_CAP):
    """Integrate until settled or divergent.

    Returns (state, residual, t_used, ok, diverged)."""
    rhs = rhs_factory(A, b, D, alpha)
    N = A.shape[0]
    kw = (dict(method="BDF", jac=jac_factory(A, b, D)) if N >= BIG_N
          else dict(method="LSODA"))

    def blow(t, x):
        return BLOWUP - np.max(np.abs(x))
    blow.terminal = True
    blow.direction = -1

    y = x0.copy()
    t_done, t_target, ok = 0.0, t_first, True
    while True:
        with np.errstate(over="ignore", invalid="ignore"):
            sol = solve_ivp(rhs, (0.0, t_target - t_done), y,
                            rtol=RTOL, atol=ATOL, events=blow, **kw)
        diverged = (sol.status == 1) or not np.all(np.isfinite(sol.y[:, -1]))
        if diverged:
            finite = np.all(np.isfinite(sol.y), axis=0)
            y = (sol.y[:, np.flatnonzero(finite)[-1]] if finite.any()
                 else np.full_like(x0, np.nan))
            return y, np.inf, t_done, False, True
        ok = ok and bool(sol.success)
        y = sol.y[:, -1]
        t_done = t_target
        res = float(np.abs(rhs(0.0, y)).max())
        if res < RES_TOL or t_done >= t_cap:
            return y, res, t_done, ok, False
        t_target = min(t_done * 10.0, t_cap)


def write_csv(path, ps, Y):
    header = ["p"] + [f"x_{i+1}" for i in range(Y.shape[1])]
    np.savetxt(path, np.column_stack([ps, Y]), delimiter=",",
               header=",".join(header), comments="",
               fmt=["%.8g"] + ["%.8g"] * Y.shape[1])


def sweep(A, b, D, grid, label, N):
    Y = np.empty((len(grid), N))
    res = np.empty(len(grid))
    tused = np.empty(len(grid))
    div = np.zeros(len(grid), dtype=bool)
    oks = []
    t0 = time.time()
    for j, a in enumerate(grid):
        Y[j], res[j], tused[j], ok, div[j] = equilibrate(
            A, b, D, a, np.full(N, X0))
        oks.append(ok)
        if N >= BIG_N and (j + 1) % 5 == 0:
            print(f"      {label}: {j+1}/{len(grid)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    fin = np.isfinite(Y)
    print(f"    {label}: {len(grid)} pts in {time.time()-t0:6.1f}s, "
          f"x in [{Y[fin].min():.4g}, {Y[fin].max():.4g}], "
          f"{int(div.sum())}/{len(grid)} DIVERGED, "
          f"max residual {np.max(res[~div]) if (~div).any() else float('nan'):.2e}, "
          f"{int((tused > T_FINAL).sum())} extended, "
          f"{int((tused >= T_CAP).sum())} hit cap", flush=True)
    return Y, res, tused, div


def main():
    import sys
    todo = sys.argv[1:] or NETWORKS
    sfile = OUT / "summary.json"
    by_tag = {}
    if sfile.exists():
        by_tag = {rr["tag"]: rr for rr in json.load(sfile.open())}

    def flush():
        with sfile.open("w") as f:
            json.dump([by_tag[t] for t in NETWORKS if t in by_tag], f,
                      indent=2)

    grid = np.linspace(A_MIN, A_MAX, NSWEEP)
    for tag in todo:
        print(f"\n========== {tag} ==========", flush=True)
        net = load_network(tag)
        N, A = net["N"], net["A"]
        b = loss_rates(N)
        np.save(OUT / f"{tag}_aeut_b.npy", b)
        D_scaled = COUPLING_CONSTANT / net["amax"]
        lam_s = lam_max_linear(A, b, D_scaled)
        lam_f = lam_max_linear(A, b, D_PAPER)
        lam_k = lam_max_linear(A, b, 0.05 * (159.0 * 2 / 62) / net["kbar"])
        print(f"  N={N}, <k>={net['kbar']:.3f}, alpha_max={net['amax']:.3f}; "
              f"D(scaled)={D_scaled:.5f}", flush=True)
        print(f"  lambda_max(-diag(b)+D A):  scaled {lam_s:+.4f}   "
              f"fixed(0.05) {lam_f:+.4f}   <k>-scaled {lam_k:+.4f}",
              flush=True)

        rec = dict(tag=tag, N=N, M=net["M"], kbar=net["kbar"],
                   amax=net["amax"], D_scaled=float(D_scaled),
                   D_fixed=D_PAPER, lam_scaled=lam_s, lam_fixed=lam_f,
                   lam_kbar=lam_k)
        variants = [("aeut", D_scaled)]
        for stem, D in variants:
            Y, res, tused, div = sweep(A, b, D, grid, f"{stem} (D={D:.5f})", N)
            write_csv(OUT / f"{tag}_{stem}_equilibrium_data.csv", grid, Y)
            np.save(OUT / f"{tag}_{stem}_diverged.npy", div)
            fin = np.isfinite(Y)
            rec[stem] = dict(
                D=float(D), n_diverged=int(div.sum()),
                ymin=float(Y[fin].min()), ymax=float(Y[fin].max()),
                max_residual=(float(np.max(res[~div])) if (~div).any()
                              else None),
                n_extended=int((tused > T_FINAL).sum()),
                max_horizon=float(tused.max()),
                n_hit_cap=int((tused >= T_CAP).sum()),
                max_residual_finite=(float(np.max(res[~div]))
                                     if (~div).any() else None),
                frac_high_at_amin=float((Y[0] > 1.0).mean())
                if not div[0] else None,
                frac_high_at_amax=float((Y[-1] > 1.0).mean())
                if not div[-1] else None)
        by_tag[tag] = rec
        flush()

    print("\n===== summary written to equilibria_revision/summary.json =====", flush=True)


if __name__ == "__main__":
    main()
