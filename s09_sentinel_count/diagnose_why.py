# =====================================================================
# SI Section S9: Robustness to the number of sentinel nodes
#
# This script: why the n-RMSE falls with N at fixed n
# Feeds      : Figs. S17 and S18, via plot_scale.py
# Produces   : data/why.npz
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Why does the n-RMSE fall as N grows at fixed n?  Three candidate causes.

  (A) normalization: the n-RMSE denominator is the target's GLOBAL range,
      which on a BA network is set by the hub.  k_max grows with N, so the
      denominator inflates even if reconstruction has not improved.
  (B) intrinsic dimension: if the effective rank of the equilibrium matrix
      does not grow with N, a fixed n keeps spanning it.
  (C) mean-field concentration: if x_i becomes more tightly a function of
      k_i alone as N grows, the problem genuinely gets easier.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib")); import paths  # noqa: E402
BA = paths.EQ_REVISION / "ba_size_sweep"   # see ../README.md
import build_v23_manuscript as bv23                              # noqa: E402

DYN = ["SIS", "SIR", "MSIS", "MSIS2", "MSIR", "MSIR2", "DW", "MUT", "GEN",
       "WC", "gLV", "NM", "EUT", "EUT2", "VEG", "VEG2", "LDW"]
ADJ11 = DYN[:11]
CSV = dict(bv23.CSV_NAMES); CSV["LDW"] = "ldw"
TAGS = [("ba1p5", 32), ("ba2p0", 100), ("ba2p5", 316),
        ("ba3p0", 1000), ("ba3p5", 3162), ("ba4p0", 10000)]


def load(tag, s):
    a = np.loadtxt(BA / f"{tag}_{CSV[s]}_equilibrium_data.csv",
                   delimiter=",", skiprows=1)
    return a[:, 1:]                       # (51, N)


def ranks(X):
    """X is (M, N); the SI uses (N, M).  Singular values are the same."""
    sv = np.linalg.svd(X, compute_uv=False)
    p = sv ** 2 / (sv ** 2).sum()
    r99 = int(np.searchsorted(np.cumsum(p), 0.99) + 1)
    ent = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    stab = float((sv ** 2).sum() / sv[0] ** 2)
    return r99, ent, stab


out = {}
print(f"  {'dyn':6s} {'tag':7s} {'N':>6s} {'kmax':>5s} {'range':>10s}"
      f" {'r99':>4s} {'r99c':>5s} {'ent_c':>6s} {'stab_c':>7s} {'degR2':>6s}")
for s in ADJ11:
    for tag, N in TAGS:
        G = nx.read_edgelist(paths.network_file(tag), nodetype=int)
        deg = np.array([G.degree(i) for i in range(N)], float)
        X = load(tag, s)
        rng = float(X.max() - X.min())
        r99, _, _ = ranks(X)
        Xc = X - X.mean(axis=0, keepdims=True)        # center each node
        r99c, entc, stabc = ranks(Xc)
        # (C) how much of the across-node variance is explained by degree
        #     alone, averaged over the control-parameter values
        num = den = 0.0
        for k in range(X.shape[0]):
            v = X[k]
            if v.std() == 0:
                continue
            m = np.zeros_like(v)
            for dv in np.unique(deg):
                sel = deg == dv
                m[sel] = v[sel].mean()
            num += ((m - v.mean()) ** 2).sum(); den += ((v - v.mean()) ** 2).sum()
        degR2 = num / den if den > 0 else np.nan
        out[(s, tag)] = dict(N=N, kmax=int(deg.max()), rng=rng, r99=r99,
                             r99c=r99c, entc=entc, stabc=stabc, degR2=degR2)
        if s in ("SIS", "MUT", "GEN"):
            print(f"  {s:6s} {tag:7s} {N:6d} {int(deg.max()):5d} {rng:10.4g}"
                  f" {r99:4d} {r99c:5d} {entc:6.2f} {stabc:7.2f} {degR2:6.3f}")
np.savez(HERE / "data" / "why.npz",
         keys=np.array([f"{a}|{b}" for a, b in out], dtype=object),
         vals=np.array([[v["N"], v["kmax"], v["rng"], v["r99"], v["r99c"],
                         v["entc"], v["stabc"], v["degR2"]]
                        for v in out.values()]))
print(f"\n  wrote data/why.npz ({len(out)} dynamics-size pairs)")
