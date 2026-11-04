# =====================================================================
# SI Section S9: Robustness to the number of sentinel nodes
#
# This script: effective ranks of the 14 networks, reused from S2
# Feeds      : Fig. S13
# Produces   : data/ranks14.npz
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Effective ranks of the equilibrium matrices for the 14 networks.

The four effective ranks of SI Section S2 (energy ranks at 0.99 and 0.999,
entropy rank, stable rank), computed on the centered matrix, then averaged
over the 11 adjacency-coupled dynamics.  Section S2 reports them for all 17
dynamics; Fig. S13 uses only the 11 adjacency-coupled ones.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import importlib.util
import numpy as np

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "lib"          # shared modules
sys.path.insert(0, str(HERE.parent / "lib")); import paths  # noqa: E402; sys.path.insert(0, str(LIB))
_s = importlib.util.spec_from_file_location("leq", LIB / "load_equilibria.py")
leq = importlib.util.module_from_spec(_s); _s.loader.exec_module(leq)

DYN = ["SIS", "SIR", "MSIS", "MSIS2", "MSIR", "MSIR2", "DW", "MUT", "GEN",
       "WC", "gLV"]
NSIZE = {"dolphin": 62, "prox": 410, "metabolic": 453, "gkk": 949,
         "lfr": 998, "er": 1000, "ba": 1000, "hk": 1000, "euroroad": 1039,
         "email": 1133, "flybi": 2705, "reactome": 5973,
         "route_views": 6474, "spanish": 11558}


def ranks(X):
    """The four effective ranks of SI Section S2."""
    sv = np.linalg.svd(X, compute_uv=False)
    e = sv ** 2 / (sv ** 2).sum()          # energy, for the two thresholds
    c = np.cumsum(e)
    q = sv / sv.sum()                      # Roy--Vetterli, as in Section S2
    q = q[q > 0]
    return (float(np.searchsorted(c, 0.99) + 1),
            float(np.searchsorted(c, 0.999) + 1),
            float(np.exp(-(q * np.log(q)).sum())),
            float((sv ** 2).sum() / sv[0] ** 2))


rows = []
for tag in leq.NETWORKS:
    per = []
    for s in DYN:
        X = leq.load_eq(tag, s)                       # (51, N)
        per.append(ranks(X - X.mean(axis=0, keepdims=True)))
    m = np.mean(per, axis=0); sd = np.std(per, axis=0)
    rows.append((tag, NSIZE[tag], *m, *sd))
    print(f"  {tag:12s} N={NSIZE[tag]:6d}  r99={m[0]:.2f}  r999={m[1]:.2f}  "
          f"ent={m[2]:.3f}  stab={m[3]:.3f}")
rows.sort(key=lambda r: r[1])
np.savez(HERE / "data" / "ranks14.npz",
         tags=np.array([r[0] for r in rows], dtype=object),
         N=np.array([r[1] for r in rows]),
         mean=np.array([r[2:6] for r in rows]),
         sd=np.array([r[6:10] for r in rows]),
         dynamics=np.array(DYN, dtype=object))
print("  wrote data/ranks14.npz")
