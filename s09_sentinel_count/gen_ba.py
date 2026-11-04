# =====================================================================
# SI Section S9: Robustness to the number of sentinel nodes
#
# This script: Barabasi-Albert networks of six sizes and all 17 dynamics on them
# Feeds      : Figs. S14 to S18, via compute_scale.py
# Produces   : equilibrium CSVs in equilibria_revision/ba_size_sweep/ (input data)
# Runtime    : days -- do not re-run
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Equilibrium data for Barabasi-Albert networks spanning a range of sizes.

Section S9 studies the reconstruction error as a function of both the
network size N and the sentinel count n, which needs networks whose size
varies while everything else is held fixed.  This script builds BA networks seeded with a triangle
and with m = 3 edges per new node, so the mean degree stays near 6 for every
N, and then produces the 17 dynamics of the main text on each of them.

The numerical protocol is unchanged, with the one exception noted below.
Each dynamics is produced by the script that produced the published CSVs for
the twelve smaller networks, per the table at the end of
equilibria/original16/README.txt:

    sis, doublewell           regenerate_sis_dw_v3.py
    sir                       regenerate_sir_v3.py
    msis                      regenerate_msis_v3.py
    msir                      regenerate_msir_v3.py
    mutualistic, genereg      regenerate_mut_gen_wc_v3.py
    wc                        regenerate_wc_kbar_v89.py
    glv                       regenerate_glv_eps.py
    noymeir, eutroph, vegturb generate_data.py
    msis2, msir2              generate_msis2_msir2.py
    eutroph2, vegturb2        generate_eut2_veg2.py
    ldw                       equilibria/ldw/generate_ldw.py

The exception is GEN.  regenerate_mut_gen_wc_v3.py writes an untrimmed grid,
so trim_gen_low_end() applies the low-end auto-trim afterwards and
re-integrates the trimmed grid with BDF at rtol 1e-4, atol 1e-6 for every
size, whereas the producer uses LSODA at rtol 1e-7, atol 1e-9 for networks
off its large-network tier.  GEN on N = 100 to 10^4 (the sizes that trigger
the trim) therefore comes from the looser solver.  Re-solving the stored
grids with LSODA at rtol 1e-7 changes no entry by more than 1e-6 of the GEN
range at any size, which is the size of the difference between two LSODA
solves at that tolerance, so the data were left as they are.

Those modules are imported unmodified and their module-level HERE (which is
both where they read the edgelist and where they write the CSV) is repointed
at the working folder WORK.  equilibria/original16/ is never written to.

    python3 gen_ba.py --build                 write the edgelists
    python3 gen_ba.py --verify                re-make the published ba network
    python3 gen_ba.py <tag> [<tag> ...]       run the full pipeline
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib")); import paths  # noqa: E402
EQ = paths.ORIGINAL16
PUB = paths.EQ_SUBMITTED
# The sweep writes its edge lists, bifurcation ranges and equilibrium CSVs
# into equilibria_revision/ba_size_sweep/ of the input data, which is where
# compute_scale.py reads the CSVs from (copies of the six edge lists are in
# networks/ba_size_sweep/ of this repository).  BASIZE_WORK overrides the
# folder for a side-by-side rerun.
WORK = (HERE / os.environ["BASIZE_WORK"] if "BASIZE_WORK" in os.environ
        else HERE / "verify" if "--verify" in sys.argv
        else paths.EQ_REVISION / "ba_size_sweep")
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(EQ))

M_EDGES = 3                      # edges brought in by each new node
SEED_GRAPH = 0                   # RNG seed for the BA growth
EXPONENTS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def tag_for(exponent):
    return f"ba{str(exponent).replace('.', 'p')}"


def n_for(exponent):
    return int(round(10 ** exponent))


def build_networks(exponents=None):
    """Triangle-seeded BA graphs, one per size, written as edgelists."""
    rows = []
    for e in (exponents or EXPONENTS):
        N = n_for(e)
        seed = nx.complete_graph(3)                 # the connected triangle
        G = nx.barabasi_albert_graph(N, M_EDGES, seed=SEED_GRAPH,
                                     initial_graph=seed)
        assert nx.is_connected(G), f"{tag_for(e)} is not connected"
        tag = tag_for(e)
        nx.write_edgelist(G, WORK / f"{tag}_network.edgelist", data=False)
        k = np.array([d for _, d in G.degree()], dtype=float)
        A = nx.to_scipy_sparse_array(G, nodelist=sorted(G.nodes()),
                                     dtype=float, format="csr")
        from scipy.sparse.linalg import eigsh
        amax = (float(np.linalg.eigvalsh(A.toarray()).max()) if N < 200
                else float(eigsh(A, k=1, which="LA",
                                 return_eigenvectors=False)[0]))
        rows.append((tag, N, G.number_of_edges(), k.mean(), int(k.max()),
                     amax, int(np.floor(np.log(N)))))
        print(f"  {tag:8s} N={N:6d} M={G.number_of_edges():7d} "
              f"<k>={k.mean():6.3f} k_max={int(k.max()):5d} "
              f"alpha_max={amax:8.3f} n=floor(ln N)={int(np.floor(np.log(N)))}",
              flush=True)
    np.savetxt(WORK / "_networks.csv",
               np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows]),
               delimiter=",", header="N,M,kbar,kmax,alpha_max,n_sentinel",
               comments="", fmt="%.6f")
    return rows


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# The published CSVs were produced with the dolphin reference values ROUNDED
# to two decimals, which is what the main text states: <k> = 5.13 and
# alpha_max = 7.19, giving D_max<k> = 2.565 for DW, 10.26 for MUT and
# D_max alpha_max = 1.6 x 7.19 = 11.504 for GEN.  The scripts in the repo
# carry more precise dolphin values (5.129032, 7.1942) and therefore do not
# reproduce the published grids.  We restore the documented constants.
CONSTANTS = {
    "regenerate_sis_dw_v3":     {"COUPLING_CONSTANT": 0.5 * 5.13},
    "regenerate_mut_gen_wc_v3": {"MUT_CONSTANT": 2.0 * 5.13,
                                 "GEN_CONSTANT": 1.6 * 7.19},
}


LARGE = False     # set from argv in main(); see run_one()


def _repoint(mod, tags):
    """Send a producer's reads and writes to WORK, and limit it to `tags`."""
    mod.HERE = WORK
    if LARGE and hasattr(mod, "LARGE_NETS"):
        # The producers decide between a dense LSODA solve at rtol=1e-7 and a
        # sparse BDF solve at rtol=1e-4 from a hard-coded set of network
        # NAMES, not from N.  reactome (N=5973) is in that set; the BA
        # networks obviously are not, so at N=10^4 they take the dense path,
        # whose Jacobian is (2N)^2 = 3.2 GB for the two-block dynamics.  This
        # puts the new tags on the large-network tier instead.
        mod.LARGE_NETS = set(mod.LARGE_NETS) | set(tags)
    if hasattr(mod, "NETWORKS"):
        mod.NETWORKS = list(tags)
    for k, v in CONSTANTS.get(getattr(mod, "__name__", ""), {}).items():
        setattr(mod, k, v)
    if hasattr(mod, "NETWORK_FILES"):
        mod.NETWORK_FILES = dict(mod.NETWORK_FILES)
        for t in tags:
            mod.NETWORK_FILES[t] = f"{t}_network.edgelist"
    return mod


STAGES = [
    # (module file, what it writes, how to invoke it)
    # generate_data.py is the live producer of NM, EUT and VEG only; its
    # other nine stems are overwritten by the v3 / wc89 / glv passes below,
    # so we ask it for just those three.  This changes no protocol, it only
    # skips work whose output is discarded, and it matters at large N where
    # the discarded MSIS and MSIR sweeps are among the most expensive.
    ("generate_data.py",            "noymeir, eutroph, vegturb",
     lambda m, tags: [(setattr(m, "SYSTEMS", ["NM", "EUT", "VEG"]),
                       m.main(t))[1] for t in tags]),
    ("regenerate_sis_dw_v3.py",     "sis, doublewell",      lambda m, t: m.main()),
    ("regenerate_sir_v3.py",        "sir",                  lambda m, t: m.main()),
    ("regenerate_msis_v3.py",       "msis",                 lambda m, t: m.main()),
    ("regenerate_msir_v3.py",       "msir",                 lambda m, t: m.main()),
    ("regenerate_mut_gen_wc_v3.py", "mutualistic, genereg", lambda m, t: m.main()),
    ("regenerate_wc_kbar_v89.py",   "wc",                   lambda m, t: m.main()),
    ("regenerate_glv_eps.py",       "glv",                  lambda m, t: m.main()),
    ("generate_msis2_msir2.py",     "msis2, msir2",         lambda m, t: m.main()),
    ("generate_eut2_veg2.py",       "eutroph2, vegturb2",   lambda m, t: m.main()),
]


def trim_gen_low_end(tags):
    """Apply the main text's low-end auto-trim to the GEN sweep.

    Methods, sec:autotrim: probe at the 20% position of the interval; if the
    node-averaged state there is below 0.02, replace p_min by the probe point
    and repeat.  The v3 producer writes an untrimmed grid, whereas the
    published GEN CSVs are trimmed once, so we apply the documented rule.

    The re-integration uses BDF at rtol 1e-4 for every size rather than the
    producer's LSODA at rtol 1e-7; see the module docstring for why this is
    harmless.
    """
    import networkx as nx
    from scipy.integrate import solve_ivp
    import scipy.sparse as sp
    for tag in tags:
        f = WORK / f"{tag}_genereg_equilibrium_data.csv"
        arr = np.loadtxt(f, delimiter=",", skiprows=1)
        G = nx.read_edgelist(WORK / f"{tag}_network.edgelist", nodetype=int)
        A = nx.to_scipy_sparse_array(G, nodelist=sorted(G.nodes()),
                                     dtype=float, format="csr")
        N = A.shape[0]
        jac = A + sp.eye(N)

        def rhs(t, x, D):
            x = np.maximum(x, 0.0)            # GEN clipping, Methods
            return -x + D * (A @ (x ** 2 / (1.0 + x ** 2)))

        pmin, pmax = float(arr[0, 0]), float(arr[-1, 0])
        n_trim = 0
        while True:
            pq = pmin + 0.2 * (pmax - pmin)
            sol = solve_ivp(rhs, (0.0, 200.0), np.full(N, 2.0), method="BDF",
                            args=(pq,), rtol=1e-4, atol=1e-6, jac_sparsity=jac)
            xbar = float(np.maximum(sol.y[:, -1], 0.0).mean())
            if xbar >= 0.02:
                break
            pmin = pq
            n_trim += 1
            if n_trim > 6:
                break
        if n_trim == 0:
            print(f"    {tag}: GEN not trimmed", flush=True)
            continue
        grid = np.linspace(pmin, pmax, arr.shape[0])
        Y = np.empty((grid.size, N))
        for k, D in enumerate(grid):
            sol = solve_ivp(rhs, (0.0, 200.0), np.full(N, 2.0), method="BDF",
                            args=(D,), rtol=1e-4, atol=1e-6, jac_sparsity=jac)
            Y[k] = np.maximum(sol.y[:, -1], 0.0)
        out = np.column_stack([grid, Y])
        np.savetxt(f, out, delimiter=",",
                   header="p," + ",".join(f"x_{i+1}" for i in range(N)),
                   comments="", fmt=["%.6f"] + ["%.6f"] * N)
        print(f"    {tag}: GEN trimmed {n_trim}x -> "
              f"[{pmin:.5f}, {pmax:.5f}]", flush=True)


def run_ldw(tags, dc=None):
    """LDW comes from equilibria/ldw/generate_ldw.py, not from equilibria/original16/.

    `dc` is forwarded as generate_ldw.py's --dc option, which skips the D_c
    bisection and uses the supplied value.  The sweep itself is unchanged.
    """
    import io, contextlib
    mod = _load("generate_ldw",
                HERE.parent / "equilibria" / "ldw" / "generate_ldw.py")
    mod.OUT = WORK                        # write beside the other CSVs
    mod.NETWORKS = list(tags)
    extra = ["--dc", str(dc)] if dc is not None else []
    saved, sys.argv = sys.argv, ["generate_ldw"] + extra + list(tags)
    try:
        mod.main()
    finally:
        sys.argv = saved


def run_pipeline(tags):
    for fname, what, call in STAGES:
        t0 = time.time()
        print(f"\n=== {fname}  ->  {what} ===", flush=True)
        mod = _repoint(_load(Path(fname).stem, EQ / fname), tags)
        # the helper modules these import also resolve paths through HERE
        for helper in ("v9", "gd"):
            if hasattr(mod, helper):
                _repoint(getattr(mod, helper), tags)
        call(mod, tags)
        print(f"    [{time.time()-t0:.0f}s]", flush=True)
    t0 = time.time()
    print("\n=== auto-trim  ->  genereg ===", flush=True)
    trim_gen_low_end(tags)
    print(f"    [{time.time()-t0:.0f}s]", flush=True)
    t0 = time.time()
    print("\n=== generate_ldw.py  ->  ldw ===", flush=True)
    run_ldw(tags)
    print(f"    [{time.time()-t0:.0f}s]", flush=True)


STAGE_BY_NAME = {Path(f).stem: (f, what, call) for f, what, call in STAGES}


def use_symmetric_lu_ordering():
    """Give SuperLU a fill-reducing ordering suited to a symmetric pattern.

    The BDF solver factorizes I - h*c*J, whose sparsity is that of A + I and
    is therefore structurally symmetric.  SciPy calls splu() with its default
    column ordering, COLAMD, which targets unsymmetric problems.  On a
    scale-free network the hubs make COLAMD fill in catastrophically: at
    N = 10^4 one factorization costs 39.6 s and 46.3 M nonzeros (fill x661),
    against 1.6 s and 3.8 M nonzeros (fill x55) under MMD_AT_PLUS_A, the
    minimum-degree ordering for the pattern of A + A^T.

    The ordering is a permutation chosen before any arithmetic: it changes
    which entries fill in, not the linear system being solved.  Results
    therefore agree to solver tolerance rather than bit for bit, because the
    floating-point operations are reassociated.
    """
    from scipy.sparse.linalg import splu as _splu
    import scipy.integrate._ivp.bdf as _bdf

    def splu_symmetric(A):
        return _splu(A, permc_spec="MMD_AT_PLUS_A")

    _bdf.splu = splu_symmetric
    print("    [LU ordering: MMD_AT_PLUS_A]", flush=True)


def run_one(step, tags, systems=None, dc=None, fastlu=False):
    """Run a single pipeline step, for use by the parallel driver run_par.py.

    The numerics are untouched: this calls exactly the same producer with
    exactly the same arguments as run_pipeline() would.  It only lets the
    caller pick which step, and which systems inside the generate_data step,
    to run in this process.
    """
    if fastlu:
        use_symmetric_lu_ordering()
    t0 = time.time()
    if step == "trim":
        print("\n=== auto-trim  ->  genereg ===", flush=True)
        trim_gen_low_end(tags)
    elif step == "ldw":
        print("\n=== generate_ldw.py  ->  ldw ===", flush=True)
        run_ldw(tags, dc=dc)
    else:
        fname, what, call = STAGE_BY_NAME[step]
        label = what if systems is None else ", ".join(systems)
        print(f"\n=== {fname}  ->  {label} ===", flush=True)
        mod = _repoint(_load(Path(fname).stem, EQ / fname), tags)
        for helper in ("v9", "gd"):
            if hasattr(mod, helper):
                _repoint(getattr(mod, helper), tags)
        if systems is not None:
            # generate_data.py only: restrict this process to these systems
            for t in tags:
                mod.SYSTEMS = list(systems)
                mod.main(t)
        else:
            call(mod, tags)
    print(f"    [{time.time()-t0:.0f}s]", flush=True)


def main():
    args = sys.argv[1:]
    if "--build" in args:
        build_networks()
        return
    step = systems = dc = None
    fastlu = "--fastlu" in args
    global LARGE
    LARGE = "--large" in args
    for a in args:
        if a.startswith("--step="):
            step = a.split("=", 1)[1]
        elif a.startswith("--systems="):
            systems = a.split("=", 1)[1].split(",")
        elif a.startswith("--dc="):
            dc = float(a.split("=", 1)[1])
    tags = [a for a in args if not a.startswith("--")]
    if not tags:
        tags = [tag_for(e) for e in EXPONENTS]
    if step:
        run_one(step, tags, systems, dc, fastlu)
        return
    run_pipeline(tags)


if __name__ == "__main__":
    main()
