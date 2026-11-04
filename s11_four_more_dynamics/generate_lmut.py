# =====================================================================
# SI Section S11: Cross-transfer classes with four additional dynamics
#
# This script: equilibria of LMUT
# Feeds      : Fig. S22 and Table S14
# Produces   : equilibrium CSVs in equilibria_revision/ (input data)
# Runtime    : days -- do not re-run
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""LMUT: the diffusively coupled mutualistic dynamics.

The paper's MUT dynamics (decoder_v100_main.tex) is adjacency-coupled through
a saturating mutualistic kernel,

    dx_i/dt = B + x_i (1 - x_i/K)(x_i/C - 1) + D sum_j A_ij g(x_i, x_j),
    g(a, b) = a b / (Dt + E a + H b),

with B = 0.1, K = 5, C = 1, Dt = 5, E = 0.9, H = 0.1.  Making the coupling
diffusive means replacing A_ij by -L_ij INSIDE the sum, and because
sum_j A_ij = k_i that is exactly a kernel difference:

    D sum_j (-L_ij) g(x_i,x_j) == D sum_j A_ij [ g(x_i,x_j) - g(x_i,x_i) ]

so

    dx_i/dt = B + x_i (1 - x_i/K)(x_i/C - 1)
              + D sum_j A_ij [ g(x_i,x_j) - g(x_i,x_i) ].            (LMUT)

The coupling vanishes on any uniform state, as a diffusive coupling must.

WHY THIS READING.  An earlier version of this file used the other reading,
-D (L x)_i, i.e. the coupling operator applied linearly to the state.  That
is plain diffusion and makes LMUT nearly a relabeling of LDW, so the pair
tested the coupling-operator claim only once.  Keeping MUT's own kernel and
differencing it is the faithful substitution and is a genuinely different
object.

Why it works where the other adjacency-coupled dynamics do not: the local
term f(x) = B + x(1-x/K)(x/C-1) is BISTABLE.  Its roots are x = 0.11577
(stable), 0.85950 (unstable) and 5.02472 (stable), so LMUT is an
Allen-Cahn/Nagumo system like LDW: diffusion cannot synchronize it, it
coarsens a frozen pattern instead.  The box [x_lo, x_hi] between the two
stable roots is forward invariant: f vanishes at both ends, and g is
increasing in its second argument, so at x_i = x_lo every bracket
g(x_lo,x_j) - g(x_lo,x_lo) >= 0 and at x_i = x_hi every bracket <= 0.

Everything else follows LDW: the paper's uniform cold start sits on an
invariant uniform manifold, so we cold-start from x_i(0) ~ U[x_lo, x_hi]
drawn once per network; D_max is set adaptively from the measured depinning
point D_c; and the horizon is settling-aware.

Usage:  python3 generate_lmut.py [tag ...]
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

# The 12 smallest of the 14 networks used in the paper, ordered by N.
NETWORKS = ["dolphin", "prox", "metabolic", "gkk", "lfr", "er",
            "ba", "hk", "euroroad", "email", "flybi", "reactome",
            "route_views", "spanish"]

# Above this size we switch from LSODA (dense numerical Jacobian) to BDF with
# the analytic sparse Jacobian, as the paper's pipeline does for its large
# networks.
BIG_N = 1500

B_IMM, K_CAP, C_ALLEE = 0.1, 5.0, 1.0
D_SAT, E_SAT, H_SAT = 5.0, 0.9, 0.1   # MUT's kernel, unchanged      # MUT local parameters
# roots of f(x) = B + x(1-x/K)(x/C-1), computed once below
_c = np.polynomial.polynomial.polyroots(
    [B_IMM, -1.0, 1.0 + 1.0 / K_CAP, -1.0 / (K_CAP * C_ALLEE)])
_r = np.sort(_c[np.abs(_c.imag) < 1e-9].real)
R1, R2, R3 = float(_r[0]), float(_r[1]), float(_r[2])   # lo, unstable, hi
NSWEEP = 51                    # M = 51 sweep points, as in the paper
T_FINAL = 200.0                # t_max = 200, as for DW in the paper
RTOL, ATOL = 1e-7, 1e-9
RES_TOL = 1e-6                 # settle until max_i |dx_i/dt| < RES_TOL
RTOL_SEARCH, ATOL_SEARCH = 1e-4, 1e-6   # cheap tier inside the D_c search
DC_SCAN_START = 1.0e-4         # upward scan for D_c starts here ...
DC_SCAN_STEP = 1.3             # ... and multiplies by this each step
T_CAP = 2.0e5                  # ... but never integrate past this
# The D_c search must use the SAME settling horizon as the sweep: MUT's
# barrier is shallow, so a pattern that is merely coarsening slowly still
# looks "patterned" under a shorter horizon, and the search then reports a
# D_c above the value at which the sweep actually collapses.
T_CAP_SEARCH = T_CAP
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
    return dict(tag=tag, N=N, M=M, kbar=2.0 * M / N, L=L, A=A.tocsr(), k=k,
                lam2=lam2, lam_max=lam_max)


def _f(x):
    """MUT local term: B + x (1 - x/K)(x/C - 1)."""
    return B_IMM + x * (1.0 - x / K_CAP) * (x / C_ALLEE - 1.0)


def _fprime(x):
    return (1.0 - x / K_CAP) * (x / C_ALLEE - 1.0) \
        + x * (-1.0 / K_CAP) * (x / C_ALLEE - 1.0) \
        + x * (1.0 - x / K_CAP) / C_ALLEE


def _g(a, b):
    """MUT's saturating mutualistic kernel."""
    return a * b / (D_SAT + E_SAT * a + H_SAT * b)


def _dg(a, b):
    """(dg/da, dg/db) with S = Dt + E a + H b."""
    S = D_SAT + E_SAT * a + H_SAT * b
    return b * (D_SAT + H_SAT * b) / S**2, a * (D_SAT + E_SAT * a) / S**2


def rhs_factory(net, D):
    """dx_i/dt = f(x_i) + D sum_j A_ij [g(x_i,x_j) - g(x_i,x_i)].

    Vectorized over A's nonzeros: `rows` repeats each row index once per
    neighbor, `cols` are the neighbors, so the coupling is a segment sum.
    """
    A = net["A"]
    k = net["k"]
    cols = A.indices
    rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
    N = A.shape[0]

    def rhs(t, x):
        xi, xj = x[rows], x[cols]
        contrib = np.bincount(rows, weights=_g(xi, xj), minlength=N)
        return _f(x) + D * (contrib - k * _g(x, x))
    return rhs


def jac_factory(net, D, N=None):
    """Analytic sparse Jacobian of the above.

    off-diagonal (j != i):  D A_ij  dg/db (x_i, x_j)
    diagonal:               f'(x_i)
                            + D sum_j A_ij dg/da (x_i, x_j)
                            - D k_i [dg/da + dg/db](x_i, x_i)

    NOTE: an earlier version of this file passed LDW's cubic derivative here
    by copy-paste.  BDF still converged (it solves the implicit equation
    against the true rhs), but Newton convergence was poor and the large
    networks were correspondingly slow.  Keep this in step with _f.
    """
    A = net["A"]
    k = net["k"]
    cols = A.indices
    rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
    n = A.shape[0]

    def jac(t, x):
        xi, xj = x[rows], x[cols]
        ga, gb = _dg(xi, xj)
        off = sp.csr_matrix((D * gb, (rows, cols)), shape=(n, n))
        saa, sab = _dg(x, x)
        d = (_fprime(x)
             + D * np.bincount(rows, weights=ga, minlength=n)
             - D * k * (saa + sab))
        return (off + sp.diags(d)).tocsr()
    return jac


def equilibrate(net, D, x0, t_first=T_FINAL, t_cap=T_CAP,
                rtol=RTOL, atol=ATOL):
    """Integrate until the state has settled; return (state, res, t_used, ok).

    Coarsening in a bistable diffusive system can be far slower than the
    t_max = 200 the paper uses for DW: a handful of sweep points near the
    depinning transition are still moving at t = 200.  We therefore keep
    integrating in blocks, each 10x longer than the last, until
    max_i |dx_i/dt| < RES_TOL or the horizon reaches t_cap."""
    rhs = rhs_factory(net, D)
    N = net['A'].shape[0]
    if N >= BIG_N:
        kw = dict(method="BDF", jac=jac_factory(net, D))
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


# Optional pre-established bracket [lo, hi] for D_c, set from the command
# line as --bracket lo,hi.  Both ends must be FACTS, not guesses: lo a D at
# which the full criterion found a pattern, hi a D at which the state is
# already collapsed to machine precision at t = 200 (and the uniform state
# is absorbing, so it stays collapsed under any longer horizon).  Given such
# a bracket the upward scan can be skipped entirely, which on the large
# graphs saves hours of probing through the slow small-D region.
BRACKET = None
N_BISECT = 18        # bisection steps; --bisect N to override


def find_Dc(net, x0, lam2=None, verbose=False):
    """Smallest D at which the equilibrium becomes spatially uniform.

    Upward scan followed by bisection.  We march up from DC_SCAN_START in
    steps of DC_SCAN_STEP until the first collapse, then bisect 18 times.
    A seeded two-sided bracket (as in generate_ldw.py) is faster but UNSAFE here:
    uniformity is not monotone in D, and a downward bracket can latch onto
    the top edge of a re-entrant patterned window and return a D_c far too
    large -- which is what happened on flybi.  Marching up from below always
    returns the FIRST collapse, which is the one we want.

    Probes use the cheap tolerance tier (the criterion is only a yes/no on
    spatial uniformity) but the FULL sweep horizon T_CAP_SEARCH = T_CAP:
    MUT's barrier is shallow enough that a shorter probe horizon reports
    "patterned" where the sweep collapses.  The final sweep uses the tight
    tolerance tier.  lam2 is accepted for signature compatibility with
    generate_ldw.find_Dc and is not used.

    Returns (D_c, n_evals), or (None, n_evals) if no collapse is found
    below D_SEARCH_CAP."""
    n = 0

    def uniform(D):
        nonlocal n
        n += 1
        t0 = time.time()
        y, _, _, _ = equilibrate(net, D, x0, t_cap=T_CAP_SEARCH,
                                 rtol=RTOL_SEARCH, atol=ATOL_SEARCH)
        u = (y.max() - y.min()) < COLLAPSE_TOL
        if verbose:
            print(f"      probe {n:3d}: D={D:12.5g} -> "
                  f"{'uniform' if u else 'patterned'} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        return u

    # Scan UPWARD from a small D.  A seeded two-sided bracket is faster but
    # unsafe: uniformity is not monotone in D (on flybi there is a narrow
    # re-entrant patterned window well above the true collapse), and a
    # downward bracket can latch onto the top edge of such a window and
    # return a D_c an order of magnitude too large.  Marching up from below
    # always returns the FIRST collapse, which is the one we want.
    if BRACKET is not None:
        lo, hi = BRACKET
        if verbose:
            print(f"      using supplied bracket [{lo:g}, {hi:g}]",
                  flush=True)
    else:
        lo, hi = 0.0, DC_SCAN_START
        while not uniform(hi):
            lo, hi = hi, hi * DC_SCAN_STEP
            if hi > D_SEARCH_CAP:
                return None, n
    for _ in range(N_BISECT):
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


def sweep(net, grid, x0, label, N):
    Y = np.empty((len(grid), N))
    res = np.empty(len(grid))
    tused = np.empty(len(grid))
    oks = []
    t0 = time.time()
    for j, D in enumerate(grid):
        Y[j], res[j], tused[j], ok = equilibrate(net, D, x0)
        oks.append(ok)
    print(f"    {label}: {len(grid)} pts in {time.time()-t0:6.1f}s, "
          f"x in [{Y.min():.4f}, {Y.max():.4f}], "
          f"max residual {res.max():.2e}, "
          f"max horizon {tused.max():.0f}, "
          f"{int((tused > T_FINAL).sum())} pts needed extension, "
          f"{sum(oks)}/{len(oks)} ok", flush=True)
    return Y, res, tused


def main():
    global BRACKET, N_BISECT
    import sys
    argv = sys.argv[1:]
    while argv and argv[0] in ("--bracket", "--bisect"):
        if argv[0] == "--bracket":
            BRACKET = tuple(float(v) for v in argv[1].split(","))
        else:
            # Fewer steps are enough when the bracket is already tight: the
            # sweep spacing is D_max/50, so resolving D_c to ~1% places the
            # collapse within half a sweep point of the 90% position.
            N_BISECT = int(argv[1])
        argv = argv[2:]
    todo = argv or NETWORKS
    sfile = OUT / "summary.json"
    by_tag = {}
    # by_tag holds only the tags THIS process computed; everything else is
    # merged back in from the shards written by other processes.

    # Per-tag shards, then a merge.  Several tags are run as separate
    # processes (the networks are independent), and the previous
    # load-once/rewrite-whole-file scheme let two processes clobber each
    # other's entries.  Each process now owns exactly one file per tag.
    shard_dir = HERE / "data" / "_summary"
    shard_dir.mkdir(parents=True, exist_ok=True)

    def flush():
        for t, rec in by_tag.items():
            (shard_dir / f"{t}.json").write_text(json.dumps(rec, indent=2))
        merged = []
        for t in NETWORKS:
            f = shard_dir / f"{t}.json"
            if f.exists():
                merged.append(json.loads(f.read_text()))
        with sfile.open("w") as f:
            json.dump(merged, f, indent=2)

    for tag in todo:
        print(f"\n========== {tag} ==========", flush=True)
        net = load_network(tag)
        N, L = net["N"], net["L"]
        print(f"  N={N}, M={net['M']}, <k>={net['kbar']:.3f}, "
              f"lambda_2(L)={net['lam2']:.5f}, "
              f"lambda_max(L)={net['lam_max']:.3f}", flush=True)

        rng = np.random.default_rng(SEED_IC)
        x0 = rng.uniform(R1, R3, N)
        np.save(OUT / f"{tag}_lmut_x0.npy", x0)

        t0 = time.time()
        Dc, nev = find_Dc(net, x0, lam2=net["lam2"], verbose=True)
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
        Y, res, tused = sweep(net, grid, x0, "LMUT/het", N)
        write_csv(OUT / f"{tag}_lmut_equilibrium_data.csv", grid, Y)
        np.save(OUT / f"{tag}_lmut_residuals.npy", res)
        np.save(OUT / f"{tag}_lmut_horizons.npy", tused)

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
