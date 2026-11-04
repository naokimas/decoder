# =====================================================================
# SI Section S7: Variability over networks and random factors
#
# This script: re-draws the heterogeneous node parameters
# Feeds      : Figs. S9 and S10, Tables S9 and S10, via compute_repeat.py
# Produces   : equilibrium CSVs in equilibria_revision/hetero_replicates/ (input data)
# Runtime    : hours
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Regenerate the five dynamics with heterogeneous node parameters.

NM draws r_i, EUT1/EUT2 share b_i, and VEG1/VEG2 share hE_i.  The manuscript
uses one realization of each, from default_rng(7), (11), and (13).  Here the
same three draws are repeated with other seeds so that the spread they induce
can be measured.

Model right-hand sides, control-parameter grids, initial conditions,
integrators, tolerances and horizons are taken from the paper's own
generators (run_all_12_dynamics_v9.py and generate_eut2_veg2.py) and are not
touched; only the seed of the parameter draw changes.

  python3 gen_hetero.py --verify            reproduce the published data
  python3 gen_hetero.py <rep> [tags...]     write replicate <rep>

Writes equilibria_revision/hetero_replicates/rep<NN>/<tag>_<stem>_equilibrium_data.csv
(--verify writes rep00).  Without network names it runs all 14 networks,
but Section S7 uses only the 11 smallest (HET_NETS in compute_repeat.py), so
pass those names to avoid days of unused computation on reactome,
route_views and spanish.  compute_repeat.py uses the published draw as its
first replicate and rep01 to rep19 as the other 19, so 19 replicates are
needed for R = 20.
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
EQ = paths.EQ_REVISION        # equilibrium CSVs; see ../README.md
EQDIR = paths.ORIGINAL16
PUB = paths.EQ_SUBMITTED
sys.path.insert(0, str(EQDIR))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


v9 = _load("v9", EQDIR / "run_all_12_dynamics_v9.py")
e2 = _load("e2", EQDIR / "generate_eut2_veg2.py")
# NM, EUT1 and VEG1 in the published data come from generate_data.py, which
# cold starts every control-parameter value; only EUT2 and VEG2 come from the
# forward warm start of generate_eut2_veg2.py.  Verified by reproduction.
gd = _load("gd", EQDIR / "generate_data.py")

NETS = ["dolphin", "prox", "metabolic", "gkk", "lfr", "er", "ba", "hk",
        "euroroad", "email", "flybi", "reactome", "route_views", "spanish"]
# (stem, seed used by the manuscript, draw, model factory, grid, x0, solver)
STEMS = {"NM": "noymeir", "EUT": "eutroph", "EUT2": "eutroph2",
         "VEG": "vegturb", "VEG2": "vegturb2"}
BASE_SEED = {"r": 7, "b": 11, "h": 13}


def load_network(tag):
    """v9.load_network, but reading the edgelists from their actual folder."""
    G = nx.read_edgelist(paths.network_file(tag), nodetype=int)
    A = nx.to_scipy_sparse_array(G, nodelist=sorted(G.nodes()), dtype=float,
                                 format="csr")
    return {"A": A, "N": A.shape[0],
            "k_vec": np.asarray(A.sum(axis=1)).ravel()}


def draws(N, rep):
    """The three parameter vectors for replicate `rep`; rep 0 is the paper's."""
    off = 0 if rep == 0 else 1000 * rep
    r = 0.6 + 0.4 * np.random.default_rng(BASE_SEED["r"] + off).random(N)
    b = 0.8 + 0.4 * np.random.default_rng(BASE_SEED["b"] + off).random(N)
    h = 1.0 + 2.0 * np.random.default_rng(BASE_SEED["h"] + off).random(N)
    return r, b, h


def published_grid(tag, dyn):
    """The control-parameter values actually used in the paper.

    VEG1's range was auto-trimmed per network, so the nominal linspace is
    wrong there; reading the grid from the published file also guarantees
    that every replicate is evaluated at exactly the same conditions.
    """
    arr = np.loadtxt(PUB / f"{tag}_{STEMS[dyn]}_equilibrium_data.csv",
                     delimiter=",", skiprows=1, usecols=0)
    return arr


def sweep_one(dyn, net, rep, tag):
    A, N, k_vec = net["A"], net["N"], net["k_vec"]
    r_vec, b_vec, h_vec = draws(N, rep)
    jac = v9.per_node_jac(A)
    bdf = lambda rhs, p, x, t_final: v9.integrate_bdf(rhs, p, x, jac, t_final)
    lsoda = v9.integrate_lsoda
    if dyn == "NM":                       # cold start
        rhs = gd.make_noymeir_rhs(A, k_vec, r_vec)
        grid = published_grid(tag, "NM")
        # The published NM sweeps use LSODA with a dense Jacobian, whose cost
        # grows as N^2 and which needs 88 minutes per sweep on Reactome.  On
        # the two largest networks we therefore integrate NM with BDF and a
        # sparse Jacobian instead.  The two agree: over five control-parameter
        # values the largest difference is 6.5e-7 on the Erdos-Renyi network
        # and 2.5e-3 on flybi, against equilibrium values of order 1-8.
        if N > 2000:
            return grid, gd.sweep_ode(rhs, grid, np.full(N, 8.0), "NM", N,
                                      jac_sparsity=gd.per_node_jac_sparsity(A))
        return grid, gd.sweep_ode(rhs, grid, np.full(N, 8.0), "NM", N)
    elif dyn == "EUT":                    # cold start, BDF
        rhs = gd.make_eutroph_rhs(A, k_vec, b_vec)
        grid = published_grid(tag, "EUT")
        return grid, gd.sweep_ode(rhs, grid, np.full(N, 0.5), "EUT", N,
                                  jac_sparsity=gd.per_node_jac_sparsity(A))
    elif dyn == "VEG":                    # cold start, BDF
        rhs = gd.make_vegturb_rhs(A, k_vec, hE_vec := h_vec)
        grid = published_grid(tag, "VEG")
        return grid, gd.sweep_ode(rhs, grid, np.full(N, 0.6), "VEG", N,
                                  jac_sparsity=gd.per_node_jac_sparsity(A))
    elif dyn == "EUT2":
        rhs = e2.make_eut2(A, k_vec, b_vec)
        grid, x0, integ = published_grid(tag, "EUT2"), 0.5, bdf
    elif dyn == "VEG2":
        rhs = e2.make_veg2(A, k_vec, h_vec)
        grid, x0, integ = published_grid(tag, "VEG2"), 0.6, bdf
    else:
        raise ValueError(dyn)
    log = HERE / "data" / "_progress.log"
    return grid, v9.sweep_forward_warm(rhs, grid, np.full(N, x0), integ,
                                       200.0, dyn, log)


def main():
    verify = "--verify" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rep = 0 if verify else int(args[0])
    tags = args[1:] if len(args) > 1 else NETS
    out = EQ / "hetero_replicates" / f"rep{rep:02d}"
    out.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        net = load_network(tag)
        for dyn, stem in STEMS.items():
            t0 = time.time()
            grid, eq = sweep_one(dyn, net, rep, tag)
            if verify:
                pub = np.loadtxt(PUB / f"{tag}_{stem}_equilibrium_data.csv",
                                 delimiter=",", skiprows=1)[:, 1:]
                err = np.abs(eq - pub).max()
                print(f"  {tag:12s} {dyn:5s} max|mine-published| = {err:.3e}"
                      f"  ({time.time()-t0:.1f}s)", flush=True)
            else:
                v9.write_csv(out / f"{tag}_{stem}_equilibrium_data.csv",
                             grid, eq)
                print(f"  {tag:12s} {dyn:5s} rep{rep:02d} written "
                      f"({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
