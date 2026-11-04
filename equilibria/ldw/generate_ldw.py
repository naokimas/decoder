# =====================================================================
# Equilibrium generation: LDW, the Laplacian-coupled double well
#
# This script: equilibria of LDW on the 14 networks
# Feeds      : LDW is the 17th dynamics of the main text, so its equilibria
#              are used throughout the main text and the SI
# Produces   : equilibrium CSVs in equilibria_revision/ (input data)
# Runtime    : days -- do not re-run
# See ../../README.md for the run order and for where the
# equilibrium data goes.
# =====================================================================
"""LDW: the Laplacian (diffusively coupled) double-well dynamics.

The paper's DW dynamics (decoder_v100_main.tex) is adjacency-coupled,

    dx_i/dt = -(x_i - r1)(x_i - r2)(x_i - r3) + D sum_j A_ij x_j,

with r1=1, r2=3, r3=5.  Here we replace the coupling  sum_j A_ij x_j  by the
usual (negative-Laplacian, i.e. diffusive) coupling  sum_j (-L_ij) x_j:

    dx_i/dt = -(x_i-1)(x_i-3)(x_i-5) + D sum_j A_ij (x_j - x_i).      (LDW)

Two structural facts drive the design of this script.

  * The box [1,5]^N is forward invariant: at x_i = 1 the reaction term
    vanishes and the coupling term is >= 0; at x_i = 5 it vanishes and the
    coupling term is <= 0.  So every recorded state lies in [1,5].

  * The paper's DW initial condition, the uniform cold start x_i(0) = 1, is
    an EXACT equilibrium of LDW for every D (the reaction term vanishes at
    the well and the Laplacian annihilates any uniform vector).  It carries
    no information.  We therefore cold-start from a heterogeneous state
    x_i(0) ~ U[1,5], drawn once per network and reused at every D.

Because the double well is symmetric about x = 3 (with u = x-3 the reaction
term is -u^3 + 4u), the coupling exerts no net drive; LDW is the symmetric
Allen--Cahn / Nagumo dynamics on a network.  Increasing D therefore does not
push nodes up a branch as in DW: it coarsens a frozen pattern of nodes in
the two wells until, at a network-dependent D_c, the pattern depins entirely
and the state collapses onto a single uniform well.  D_c has no closed form,
so we locate it numerically per network (doubling search + bisection) and
set D_max = D_c / 0.9, placing the collapse at the 90% position of the
sweep.  This is the same adaptive spirit as the paper's beta_max widening
rule for MSIS/MSIR (Methods, sec:autotrim).

The equilibrium CSVs are written to equilibria_revision/ of the input data
(lib/paths.py), which is also where the network edge lists are looked up.

Usage:  python3 generate_ldw.py
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib")); import paths  # noqa: E402
EQDIR = paths.EQ_SUBMITTED
OUT = paths.EQ_REVISION              # all 1-D equilibria live together
OUT.mkdir(exist_ok=True)

# The 12 smallest of the 14 networks used in the paper, ordered by N.
NETWORKS = ["dolphin", "prox", "metabolic", "gkk", "lfr", "er",
            "ba", "hk", "euroroad", "email", "flybi", "reactome",
            "route_views", "spanish"]

# Above this size we switch from LSODA (dense numerical Jacobian) to BDF with
# the analytic sparse Jacobian, as the paper's pipeline does for its large
# networks.
BIG_N = 1500

R1, R2, R3 = 1.0, 3.0, 5.0     # double-well roots, as in the paper
NSWEEP = 51                    # M = 51 sweep points, as in the paper
T_FINAL = 200.0                # t_max = 200, as for DW in the paper
RTOL, ATOL = 1e-7, 1e-9
RES_TOL = 1e-6                 # settle until max_i |dx_i/dt| < RES_TOL
RTOL_SEARCH, ATOL_SEARCH = 1e-4, 1e-6   # cheap tier inside the D_c search
DC_SEED_NUM = 0.5              # seed the search at D = DC_SEED_NUM/lambda_2
T_CAP = 2.0e5                  # ... but never integrate past this
T_CAP_SEARCH = 2.0e4           # cheaper cap inside the D_c search
COLLAPSE_TOL = 1e-6            # max_i x_i - min_i x_i below this = uniform
DC_PERCENTILE = 0.9            # place D_c at the 90% position of the sweep
D_SEARCH_CAP = 1.0e4           # give up looking for a collapse beyond this
SEED_IC = 20260829


def load_network(tag: str):
    G = nx.read_edgelist(paths.network_file(tag), nodetype=int)
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")
    N = A.shape[0]
    M = A.nnz // 2
    k = np.asarray(A.sum(axis=1)).ravel()
    L = (sp.diags(k) - A).tocsr()
    if N < 1500:
        Ld = L.toarray()
        w = np.linalg.eigvalsh(Ld)
        lam2, lam_max = float(w[1]), float(w[-1])
    else:
        lam2 = float(eigsh(L, k=2, which="SM",
                           return_eigenvectors=False)[0])
        lam_max = float(eigsh(L, k=1, which="LA",
                              return_eigenvectors=False)[0])
    return dict(tag=tag, N=N, M=M, kbar=2.0 * M / N, L=L, k=k,
                lam2=lam2, lam_max=lam_max)


def rhs_factory(L, D):
    def rhs(t, x):
        return -(x - R1) * (x - R2) * (x - R3) - D * (L @ x)
    return rhs


def jac_factory(L, D, N):
    """Analytic Jacobian  diag(f'(x)) - D L  with
    f(x) = -(x-1)(x-3)(x-5),  f'(x) = -3x^2 + 18x - 23."""
    DL = (D * L).tocsr()

    def jac(t, x):
        return sp.diags(-3.0 * x * x + 18.0 * x - 23.0).tocsr() - DL
    return jac


def equilibrate(L, D, x0, t_first=T_FINAL, t_cap=T_CAP,
                rtol=RTOL, atol=ATOL):
    """Integrate until the state has settled; return (state, res, t_used, ok).

    Coarsening in a bistable diffusive system can be far slower than the
    t_max = 200 the paper uses for DW: a handful of sweep points near the
    depinning transition are still moving at t = 200.  We therefore keep
    integrating in blocks, each 10x longer than the last, until
    max_i |dx_i/dt| < RES_TOL or the horizon reaches t_cap."""
    rhs = rhs_factory(L, D)
    N = L.shape[0]
    if N >= BIG_N:
        kw = dict(method="BDF", jac=jac_factory(L, D, N))
    else:
        kw = dict(method="LSODA")
    y = x0.copy()
    t_done = 0.0
    t_target = t_first
    ok = True
    while True:
        sol = solve_ivp(rhs, (0.0, t_target - t_done), y,
                        rtol=rtol, atol=atol, **kw)
        ok = ok and bool(sol.success)
        y = sol.y[:, -1]
        t_done = t_target
        res = float(np.abs(rhs(0.0, y)).max())
        if res < RES_TOL or t_done >= t_cap:
            return y, res, t_done, ok
        t_target = min(t_done * 10.0, t_cap)


def find_Dc(L, x0, lam2=None, verbose=False):
    """Smallest D at which the equilibrium becomes spatially uniform.

    Bracketing search followed by bisection.  Empirically D_c lambda_2(L)
    lies in roughly [0.2, 1.0] across networks, so when lambda_2 is known we
    seed the search at D = DC_SEED_NUM/lambda_2 and expand outwards, which
    replaces ~15 doublings from D = 0.01 by a handful of steps.  Search
    evaluations use the cheap tolerance tier (the criterion is only a
    yes/no on spatial uniformity); the final sweep uses the tight one.

    Returns (D_c, n_evals), or (None, n_evals) if no collapse is found
    below D_SEARCH_CAP."""
    n = 0

    def uniform(D):
        nonlocal n
        n += 1
        t0 = time.time()
        y, _, _, _ = equilibrate(L, D, x0, t_cap=T_CAP_SEARCH,
                                 rtol=RTOL_SEARCH, atol=ATOL_SEARCH)
        u = (y.max() - y.min()) < COLLAPSE_TOL
        if verbose:
            print(f"      probe {n:3d}: D={D:12.5g} -> "
                  f"{'uniform' if u else 'patterned'} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        return u

    seed = (DC_SEED_NUM / lam2) if lam2 and lam2 > 0 else 0.01
    if uniform(seed):
        # bracket downwards
        hi = seed
        lo = seed / 1.6
        while uniform(lo):
            hi = lo
            lo /= 1.6
            if lo < 1e-8:
                lo = 0.0
                break
    else:
        # bracket upwards
        lo = seed
        hi = seed * 1.6
        while not uniform(hi):
            lo, hi = hi, hi * 1.6
            if hi > D_SEARCH_CAP:
                return None, n
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if uniform(mid):
            hi = mid
        else:
            lo = mid
    return hi, n


def write_csv(path: Path, ps, Y):
    header = ["p"] + [f"x_{i+1}" for i in range(Y.shape[1])]
    np.savetxt(path, np.column_stack([ps, Y]), delimiter=",",
               header=",".join(header), comments="",
               fmt=["%.8g"] + ["%.8g"] * Y.shape[1])


def sweep(L, grid, x0, label, N):
    Y = np.empty((len(grid), N))
    res = np.empty(len(grid))
    tused = np.empty(len(grid))
    oks = []
    t0 = time.time()
    for j, D in enumerate(grid):
        Y[j], res[j], tused[j], ok = equilibrate(L, D, x0)
        oks.append(ok)
    print(f"    {label}: {len(grid)} pts in {time.time()-t0:6.1f}s, "
          f"x in [{Y.min():.4f}, {Y.max():.4f}], "
          f"max residual {res.max():.2e}, "
          f"max horizon {tused.max():.0f}, "
          f"{int((tused > T_FINAL).sum())} pts needed extension, "
          f"{sum(oks)}/{len(oks)} ok", flush=True)
    return Y, res, tused


def main():
    """--dc VALUE skips the D_c search and uses the supplied value.

    Used for spanish, where the bisection had already pinned
    D_c in (3.6213, 3.6447) but each further step cost ~12 h: right at the
    depinning point the state neither survives nor collapses, so the
    settling loop runs to its cap.  We take the upper end, which is a
    MEASURED collapsed state, so D_max = D_c/0.9 is guaranteed to reach
    past the transition; the residual 0.6% uncertainty in D_c shifts each
    sweep point by under a fifth of the spacing to its neighbor."""
    import sys
    argv = sys.argv[1:]
    forced_dc = None
    if argv and argv[0] == "--dc":
        forced_dc = float(argv[1]); argv = argv[2:]
    todo = argv or NETWORKS
    sfile = OUT / "summary.json"
    by_tag = {}
    if sfile.exists():
        by_tag = {r["tag"]: r for r in json.load(sfile.open())}

    def flush():
        ordered = [by_tag[t] for t in NETWORKS if t in by_tag]
        with sfile.open("w") as f:
            json.dump(ordered, f, indent=2)

    for tag in todo:
        print(f"\n========== {tag} ==========", flush=True)
        net = load_network(tag)
        N, L = net["N"], net["L"]
        print(f"  N={N}, M={net['M']}, <k>={net['kbar']:.3f}, "
              f"lambda_2(L)={net['lam2']:.5f}, "
              f"lambda_max(L)={net['lam_max']:.3f}", flush=True)

        rng = np.random.default_rng(SEED_IC)
        x0 = rng.uniform(R1, R3, N)
        np.save(OUT / f"{tag}_ldw_x0.npy", x0)

        t0 = time.time()
        if forced_dc is not None:
            Dc, nev = forced_dc, 0
            print(f"  D_c supplied on the command line: {Dc:.5f}", flush=True)
        else:
            Dc, nev = find_Dc(L, x0, lam2=net["lam2"], verbose=(N >= BIG_N))
        if Dc is None:
            D_max = D_SEARCH_CAP
            print(f"  no collapse below D={D_SEARCH_CAP:g} after {nev} "
                  f"evals ({time.time()-t0:.0f}s); using D_max="
                  f"{D_max:g}", flush=True)
        else:
            D_max = Dc / DC_PERCENTILE
            print(f"  D_c={Dc:.5f} ({nev} evals, {time.time()-t0:.0f}s)"
                  f" -> D_max={D_max:.5f}, D_max*<k>="
                  f"{D_max*net['kbar']:.3f}", flush=True)

        grid = np.linspace(0.0, D_max, NSWEEP)
        Y, res, tused = sweep(L, grid, x0, "LDW/het", N)
        write_csv(OUT / f"{tag}_ldw_equilibrium_data.csv", grid, Y)
        np.save(OUT / f"{tag}_ldw_residuals.npy", res)
        np.save(OUT / f"{tag}_ldw_horizons.npy", tused)

        by_tag[tag] = dict(
            tag=tag, N=N, M=net["M"], kbar=net["kbar"],
            lam2=net["lam2"], lam_max=net["lam_max"],
            Dc=(None if Dc is None else float(Dc)), D_max=float(D_max),
            ymin=float(Y.min()), ymax=float(Y.max()),
            max_residual=float(res.max()),
            max_horizon=float(tused.max()),
            n_extended=int((tused > T_FINAL).sum()),
            std_at_0=float(Y[0].std()), std_at_max=float(Y[-1].std()),
            n_uniform=int(((Y.max(axis=1) - Y.min(axis=1))
                           < COLLAPSE_TOL).sum()))
        flush()

    print("\n===== summary written to equilibria_revision/summary.json =====", flush=True)


if __name__ == "__main__":
    main()
