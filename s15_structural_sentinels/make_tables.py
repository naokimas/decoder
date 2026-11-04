# =====================================================================
# SI Section S15: Sentinel sets from network structure alone
#
# This script: Tables S22 and S23
# Produces   : tables/tab_sentsel_z.tex, tables/tab_sentsel_struct.tex
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""LaTeX table bodies for SI Section S15.

  tables/tab_sentsel_z.tex       Table S22: how unusual the MinCorr and QR
                                 sentinel sets look as sets of network
                                 nodes, as z-scores against 10^3 uniformly
                                 random sets of the same size.  Each z-score
                                 is one of the 154 (network, dynamics)
                                 pairs, and the table summarizes them by
                                 mean, median and the percentage with
                                 |z| > 2.
  tables/tab_sentsel_struct.tex  Table S23: what the purely structural
                                 sentinel rules, largest degree and largest
                                 betweenness, cost in reconstruction
                                 accuracy, against the three selection
                                 methods used everywhere else.

Both tables cover the 11 adjacency-coupled dynamics only, so the leading
11 x 11 block of each n-RMSE matrix is the one to take.  The MinCorr, QR and
random columns of Table S23 come from the main-text matrices in
../shared_data/; only the degree and betweenness rules are computed in this
section, by compute_struct.py.

Run after compute_sentsel.py and compute_struct.py.  Takes under a second.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "tables"; OUT.mkdir(exist_ok=True)
NADJ = 11

# ----------------------------------------------------------------------
# Table S22
# ----------------------------------------------------------------------
Z = np.load(HERE / "data" / "sentsel.npz", allow_pickle=True)

# (key in the npz, row label as the SI writes it).  The overlap row is the
# odd one out: it has its own pair of arrays, Zov_minc and Zov_qr.
ZROWS = [("mean_degree", "mean degree"),
         ("mean_betweenness", "mean betweenness"),
         ("std_degree", "standard deviation of the degrees"),
         ("mean_distance", "mean pairwise distance"),
         (None, "MinCorr--QR overlap")]


def zcells(z):
    """mean, median and the percentage of |z| > 2, over all 154 pairs."""
    z = z.ravel()
    return [f"{np.mean(z):.2f}", f"{np.median(z):.2f}",
            f"{100 * np.mean(np.abs(z) > 2):.0f}\\%"]


def math_minus(cells):
    """A negative number goes in math mode so LaTeX sets a proper minus."""
    return [f"${c}$" if c.startswith("-") else c for c in cells]


rows = []
for key, label in ZROWS:
    cells = []
    for meth in ("minc", "qr"):
        cells += zcells(Z[f"Zov_{meth}"] if key is None else Z[f"Z_{meth}_{key}"])
    rows.append(f"{label} & " + " & ".join(math_minus(cells)) + r"\\")
(OUT / "tab_sentsel_z.tex").write_text("\n".join(rows) + "\n")
print("  wrote tables/tab_sentsel_z.tex")
for r in rows:
    print("   ", r)

# ----------------------------------------------------------------------
# Table S23
# ----------------------------------------------------------------------
S = np.load(HERE / "data" / "struct.npz", allow_pickle=True)
F = np.load(HERE.parent / "shared_data" / "fig45_matrices.npz",
            allow_pickle=True)
EYE = np.eye(NADJ, dtype=bool)
DECS = [("nn", "shallow NN decoder"), ("lin", "linear ridge decoder")]
SENT = [("minc", "MinCorr"), ("qr", "QR"), ("random", "random"),
        ("degree", "degree"), ("betweenness", "betweenness")]

rows = []
for di, (dec, dname) in enumerate(DECS):
    for si, (sent, sname) in enumerate(SENT):
        src = F if sent in ("minc", "qr", "random") else S
        M = src[f"M_{dec}_{sent}"][:NADJ, :NADJ] * 100
        lead = (r"\multirow{5}{*}{" + dname + "}") if si == 0 else ""
        rows.append(f"{lead} & {sname} & "
                    f"{np.median(M[EYE]):.2f} & {np.median(M[~EYE]):.2f} & "
                    f"{np.mean(M[EYE]):.2f} & {np.mean(M[~EYE]):.2f}" + r"\\")
    if di < len(DECS) - 1:
        rows.append(r"\addlinespace")
(OUT / "tab_sentsel_struct.tex").write_text("\n".join(rows) + "\n")
print("  wrote tables/tab_sentsel_struct.tex")
for r in rows:
    print("   ", r)
