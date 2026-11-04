# =====================================================================
# SI Section S11: Cross-transfer classes with four additional dynamics
#
# This script: equilibria of AVEG1
# Feeds      : Fig. S22 and Table S14
# Produces   : equilibrium CSVs in equilibria_revision/ (input data)
# Runtime    : days -- do not re-run
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""AVEG1: the adjacency-coupled vegetation--turbidity dynamics.

The paper's VEG1 (decoder_v100_main.tex) is diffusively coupled,

    dx_i/dt = r_V x_i [ 1 - x_i (h_i^p + E_i^p) / h_i^p ]
              + D sum_j A_ij (x_j - x_i),      E_i = E_o h_V / (h_V + x_i),

with r_V = 0.5, h_V = 0.2, p = 4, D = 0.05, per-node half-saturation
constants h_i ~ U[1,3] fixed for each network, and the background turbidity
E_o swept over [2,12].  Here we replace the diffusive coupling by adjacency
coupling:

    dx_i/dt = r_V x_i [ 1 - x_i (h_i^p + E_i^p) / h_i^p ]
              + D sum_j A_ij x_j .                                (AVEG1)

Reusing the ORIGINAL h_i.  generate_data.py builds them as
    rng_h = np.random.default_rng(13);  hE_vec = 1.0 + 2.0 * rng_h.random(N)
which is deterministic, so we regenerate the identical vector, which was
checked against the stored <tag>_vegturb_equilibrium_data.csv when the data
were produced.

DOES IT DIVERGE?  No, and unlike AEUT1 the reason is structural rather than
a matter of choosing D.  As x grows, E_i -> 0 and the bracket -> 1 - x, so
the local term behaves as -r_V x^2: the damping is QUADRATIC and always
beats the linear coupling input D sum_j A_ij x_j.  Concretely, at the node
holding the maximum X,

    dX/dt <= r_V X (1 - X) + D k_i X = X [ r_V (1 - X) + D k_i ] ,

which is negative once X > 1 + D k_max / r_V.  So the state is bounded for
every D and every network -- there is nothing to suppress.  (Contrast
AEUT1, whose only negative feedback is the LINEAR loss -b_i x_i, so
boundedness there needs lambda_max(-diag(b) + D A) < 0.)

D is chosen, as for ANM, so that D <k> is constant across networks,
anchored at the paper's D = 0.05 on dolphin, D <k> = 0.2565.

Everything else is unchanged from the paper's VEG1: r_V, h_V, p = 4, the
cold start x_i(0) = 0.6, t_max = 200, M = 51 sweep points and E_o in [2,12].

Usage:  python3 generate_aveg.py [tag ...]
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

P_HILL = 4.0                   # VEG1 Hill exponent, as in the paper
R_V, H_V = 0.5, 0.2
D_PAPER = 0.05                 # the paper's VEG dispersal rate
KBAR_DOLPHIN = 159.0 * 2.0 / 62.0
COUPLING_CONSTANT = D_PAPER * KBAR_DOLPHIN     # 0.25645
A_MIN, A_MAX = 2.0, 12.0       # the paper's E_o sweep, unchanged
NSWEEP = 51
X0 = 0.6                       # the paper's VEG cold start
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
SEED_B = 13


def half_saturations(N):
    """The paper's VEG per-node half-saturation constants, bit-for-bit."""
    return 1.0 + 2.0 * np.random.default_rng(SEED_B).random(N)


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


def bound(D, kmax):
    """The a-priori cap X <= 1 + D k_max / r_V derived in the docstring."""
    return 1.0 + D * kmax / R_V


def rhs_factory(A, h, D, Eo):
    hp = h ** P_HILL

    def rhs(t, x):
        xp = np.maximum(x, 0.0)
        E = Eo * H_V / (H_V + xp)
        return R_V * xp * (1.0 - xp * (hp + E ** P_HILL) / hp) + D * (A @ x)
    return rhs


def jac_factory(A, h, D, Eo):
    """Analytic Jacobian.  With G(x) = 1 + (E/h)^p and E = Eo H_V/(H_V+x),
    the local term is r_V (x - x^2 G) and
    d/dx = r_V [1 - 2 x G + x^2 p E^p / ((H_V + x) h^p)]."""
    DA = (D * A).tocsr()
    hp = h ** P_HILL

    def jac(t, x):
        xc = np.maximum(x, 0.0)
        E = Eo * H_V / (H_V + xc)
        Ep = E ** P_HILL
        G = 1.0 + Ep / hp
        d = R_V * (1.0 - 2.0 * xc * G
                   + xc * xc * P_HILL * Ep / ((H_V + xc) * hp))
        return sp.diags(d).tocsr() + DA
    return jac


def equilibrate(A, b, D, alpha, x0, t_first=T_FINAL, t_cap=T_CAP):
    """Integrate until settled or divergent.

    Returns (state, residual, t_used, ok, diverged)."""
    rhs = rhs_factory(A, b, D, alpha)
    N = A.shape[0]
    kw = (dict(method="BDF", jac=jac_factory(A, b, D, alpha)) if N >= BIG_N
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
        b = half_saturations(N)
        np.save(OUT / f"{tag}_aveg_b.npy", b)
        D_scaled = COUPLING_CONSTANT / net["kbar"]
        kmax = float(net["k"].max())
        print(f"  N={N}, <k>={net['kbar']:.3f}, k_max={kmax:.0f}; "
              f"D(scaled)={D_scaled:.5f}", flush=True)
        print(f"  a-priori cap 1 + D k_max / r_V:  scaled "
              f"{bound(D_scaled, kmax):.2f}   fixed(0.05) "
              f"{bound(D_PAPER, kmax):.2f}", flush=True)

        rec = dict(tag=tag, N=N, M=net["M"], kbar=net["kbar"],
                   amax=net["amax"], kmax=kmax,
                   D_scaled=float(D_scaled), D_fixed=D_PAPER,
                   bound_scaled=bound(D_scaled, kmax),
                   bound_fixed=bound(D_PAPER, kmax))
        variants = [("aveg", D_scaled)]
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
