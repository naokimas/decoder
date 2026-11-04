# =====================================================================
# SI Section S11: Cross-transfer classes with four additional dynamics
#
# This script: equilibria of ANM
# Feeds      : Fig. S22 and Table S14
# Produces   : equilibrium CSVs in equilibria_revision/ (input data)
# Runtime    : days -- do not re-run
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""ANM: the adjacency-coupled Noy--Meir grazing dynamics.

The paper's NM dynamics (decoder_v100_main.tex) is diffusively (Laplacian)
coupled,

    dx_i/dt = r_i x_i (1 - x_i/K) - c x_i^2/(x_i^2+1)
              + D sum_j A_ij (x_j - x_i),

with K = 10, D = 0.05, r_i ~ U[0.6,1] fixed per network, and the grazing
pressure c swept over [1,3].  Here we replace the diffusive coupling by
adjacency coupling, i.e. sum_j A_ij (x_j - x_i) -> sum_j A_ij x_j:

    dx_i/dt = r_i x_i (1 - x_i/K) - c x_i^2/(x_i^2+1)
              + D sum_j A_ij x_j.                                   (ANM)

Reusing the ORIGINAL r_i.  The paper's NM growth rates are generated in
equilibria/original16/generate_data.py by

    rng_r = np.random.default_rng(7);  r_vec = 0.6 + 0.4 * rng_r.random(N)

which is deterministic, so we regenerate exactly the same vector rather than
re-drawing.  Re-running the Laplacian NM with this vector reproduces the
stored <tag>_noymeir_equilibrium_data.csv, which was checked when the data
were produced.

Convergence.  x = 0 is an equilibrium but is unstable (the grazing term is
O(x^2), so the Jacobian there is diag(r) + D A, which has positive
eigenvalues), and the logistic term keeps the state bounded: dropping the
(negative) grazing term, dx_i/dt <= x_i [ r_i (1 - x_i/K) + D k_i ], so
x_i cannot exceed K (1 + D k_i / r_i).  Adjacency coupling therefore pushes
equilibria well above the carrying capacity K = 10 at high-degree nodes, but
it cannot make them diverge.

We set D = C / <k>  with  C = 0.05 * <k>_dolphin = 0.25645, i.e. D <k>
held constant across networks, anchored so that the dolphin network keeps
the paper's D = 0.05.  This is the convention the paper already uses for its
adjacency-coupled DW and WC dynamics.

Everything is written into this folder; nothing outside it is touched.

Usage:  python3 generate_anm.py [tag ...]
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

K_CAP = 10.0                   # carrying capacity, as in the paper
D_PAPER = 0.05                 # the paper's NM dispersal rate
KBAR_DOLPHIN = 159.0 * 2.0 / 62.0          # 5.1290..., as in the paper
COUPLING_CONSTANT = D_PAPER * KBAR_DOLPHIN  # 0.25645
# Grazing-pressure sweep.  The paper's NM uses the fixed range c in [1,3].
# Under adjacency coupling the collapse becomes a node-by-node cascade whose
# width depends strongly on the network: it is complete by c ~ 4 on er but
# needs c ~ 12 on reactome, whose high-degree core sustains itself.  No
# single fixed range therefore covers the whole transition everywhere, so we
# set c_max per network: c_full is the smallest c at which EVERY node has
# reached the grazed state (max_i x_i < GRAZED_LEVEL), found by bisection,
# and c_max = c_full / C_FULL_POSITION puts that completion near the top of
# the sweep.  This is the same adaptive spirit as the paper's beta_max
# widening rule for MSIS/MSIR, and as LDW's D_c rule in Part I.
C_MIN = 1.0
GRAZED_LEVEL = 2.0        # no node above this = all grazed
C_FULL_POSITION = 0.95    # c_full sits at 95% of [C_MIN, c_max]
C_SEARCH_CAP = 100.0
NSWEEP = 51
X0 = 8.0                       # the paper's NM cold start
T_FINAL = 200.0                # the paper's t_max for NM
RTOL, ATOL = 1e-7, 1e-9        # tighter than the paper's 1e-4/1e-6
RTOL_BIG, ATOL_BIG = 1e-5, 1e-7   # N >= BIG_N; still far tighter than the
                                  # paper's large-network tier (1e-4/1e-6),
                                  # and the settling test below is what
                                  # actually certifies the equilibrium
# Settling criterion.  Right at a node's saddle node the relaxation is
# algebraically slow, so no finite horizon reaches an arbitrarily small
# residual; we therefore stop at max_i |dx_i/dt| < RES_TOL or at T_CAP,
# whichever comes first, and report both the residual and the number of
# points that hit the cap.  The actual drift of the recorded states, which
# is the meaningful check, was measured separately.
RES_TOL = 1e-5
T_CAP = 2.0e4
BIG_N = 1500
SEED_R = 7                     # the seed used by generate_data.py


def growth_rates(N):
    """The paper's NM per-node growth rates, regenerated bit-for-bit."""
    return 0.6 + 0.4 * np.random.default_rng(SEED_R).random(N)


def load_network(tag):
    G = nx.read_edgelist(paths.network_file(tag), nodetype=int)
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")
    N = A.shape[0]
    M = A.nnz // 2
    k = np.asarray(A.sum(axis=1)).ravel()
    return dict(tag=tag, N=N, M=M, kbar=2.0 * M / N, A=A, k=k)


def rhs_factory(A, r, D, c):
    def rhs(t, x):
        return (r * x * (1.0 - x / K_CAP) - c * x * x / (x * x + 1.0)
                + D * (A @ x))
    return rhs


def jac_factory(A, r, D, c):
    DA = (D * A).tocsr()

    def jac(t, x):
        x2 = x * x
        d = r * (1.0 - 2.0 * x / K_CAP) - c * 2.0 * x / (x2 + 1.0) ** 2
        return sp.diags(d).tocsr() + DA
    return jac


def equilibrate(A, r, D, c, x0, t_first=T_FINAL, t_cap=T_CAP):
    """Integrate until settled; return (state, residual, t_used, ok)."""
    rhs = rhs_factory(A, r, D, c)
    N = A.shape[0]
    if N >= BIG_N:
        kw = dict(method="BDF", jac=jac_factory(A, r, D, c))
        rtol, atol = RTOL_BIG, ATOL_BIG
    else:
        kw = dict(method="LSODA")
        rtol, atol = RTOL, ATOL
    y = x0.copy()
    t_done, t_target, ok = 0.0, t_first, True
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


def find_c_full(A, r, D, N, verbose=False):
    """Smallest c at which every node has collapsed to the grazed state.

    Expands upward from c = 4 in steps of x1.5 until no node is left above
    GRAZED_LEVEL, then bisects 12 times.  Unlike the D_c searches of
    generate_ldw.py / generate_lmut.py, the probes run at the SWEEP's own tolerances and
    horizon (equilibrate() is called with its defaults), so c_full is
    measured under exactly the conditions the recorded data are produced
    under."""
    n = 0

    def all_grazed(c):
        nonlocal n
        n += 1
        t0 = time.time()
        y, _, _, _ = equilibrate(A, r, D, c, np.full(N, X0))
        ok = bool(np.max(y) < GRAZED_LEVEL)
        if verbose:
            print(f"      probe {n:2d}: c={c:8.4f} -> "
                  f"{'all grazed' if ok else 'some vegetated'} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        return ok

    lo, hi = C_MIN, 4.0
    while not all_grazed(hi):
        lo, hi = hi, hi * 1.5
        if hi > C_SEARCH_CAP:
            return None, n
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if all_grazed(mid):
            hi = mid
        else:
            lo = mid
    return hi, n


def write_csv(path, ps, Y):
    header = ["p"] + [f"x_{i+1}" for i in range(Y.shape[1])]
    np.savetxt(path, np.column_stack([ps, Y]), delimiter=",",
               header=",".join(header), comments="",
               fmt=["%.8g"] + ["%.8g"] * Y.shape[1])


def sweep(A, r, D, grid, label, N):
    Y = np.empty((len(grid), N))
    res = np.empty(len(grid))
    tused = np.empty(len(grid))
    oks = []
    t0 = time.time()
    for j, c in enumerate(grid):
        Y[j], res[j], tused[j], ok = equilibrate(A, r, D, c, np.full(N, X0))
        oks.append(ok)
        if N >= BIG_N and (j + 1) % 5 == 0:
            print(f"      {label}: {j+1}/{len(grid)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"    {label}: {len(grid)} pts in {time.time()-t0:6.1f}s, "
          f"x in [{Y.min():.4f}, {Y.max():.4f}], "
          f"max residual {res.max():.2e}, max horizon {tused.max():.0f}, "
          f"{int((tused > T_FINAL).sum())} extended, "
          f"{int((tused >= T_CAP).sum())} hit cap, "
          f"{sum(oks)}/{len(oks)} ok", flush=True)
    return Y, res, tused


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

    for tag in todo:
        print(f"\n========== {tag} ==========", flush=True)
        net = load_network(tag)
        N, A, k = net["N"], net["A"], net["k"]
        r = growth_rates(N)
        np.save(OUT / f"{tag}_anm_r.npy", r)
        D_scaled = COUPLING_CONSTANT / net["kbar"]
        print(f"  N={N}, M={net['M']}, <k>={net['kbar']:.3f}, "
              f"k_max={k.max():.0f}; D(scaled)={D_scaled:.5f}, "
              f"D(fixed)={D_PAPER}", flush=True)

        t0 = time.time()
        c_full, nev = find_c_full(A, r, D_scaled, N, verbose=(N >= BIG_N))
        if c_full is None:
            c_max = C_SEARCH_CAP
            print(f"  no full collapse below c={C_SEARCH_CAP:g} after "
                  f"{nev} probes; using c_max={c_max:g}", flush=True)
        else:
            c_max = c_full / C_FULL_POSITION
            print(f"  c_full={c_full:.4f} ({nev} probes, "
                  f"{time.time()-t0:.0f}s) -> c_max={c_max:.4f}", flush=True)
        grid = np.linspace(C_MIN, c_max, NSWEEP)

        rec = dict(tag=tag, N=N, M=net["M"], kbar=net["kbar"],
                   kmax=float(k.max()), D_scaled=float(D_scaled),
                   D_fixed=D_PAPER,
                   c_full=(None if c_full is None else float(c_full)),
                   c_max=float(c_max))
        variants = [("anm", D_scaled)]
        for stem, D in variants:
            Y, res, tused = sweep(A, r, D, grid, f"{stem} (D={D:.5f})", N)
            write_csv(OUT / f"{tag}_{stem}_equilibrium_data.csv", grid, Y)
            np.save(OUT / f"{tag}_{stem}_residuals.npy", res)
            np.save(OUT / f"{tag}_{stem}_horizons.npy", tused)
            rec[stem] = dict(
                D=float(D), ymin=float(Y.min()), ymax=float(Y.max()),
                max_residual=float(res.max()),
                max_horizon=float(tused.max()),
                n_extended=int((tused > T_FINAL).sum()),
                mean_at_cmin=float(Y[0].mean()),
                mean_at_cmax=float(Y[-1].mean()),
                min_at_cmax=float(Y[-1].min()))
        by_tag[tag] = rec
        flush()

    print("\n===== summary written to equilibria_revision/summary.json =====", flush=True)


if __name__ == "__main__":
    main()
