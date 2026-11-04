# =====================================================================
# SI Section S12: Why the transferable dynamics transfer
#
# This script: Table S15
# Produces   : tables/tab_why.tex
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Does equilibrium similarity predict cross-transfer accuracy?

Fig. S23 shows, cell by cell, how similar the equilibria of two dynamics
are: panel (a) uses, for each dynamics, the control-parameter value at
which the sample variance of the equilibrium state over the nodes is
largest, and panel (b) the value whose sample variance is the median.
Table S15 turns that picture into one number per panel by correlating the
equilibrium-state correlation against the two cross-transfer performance
measures, across the 110 ordered pairs of distinct adjacency-coupled
dynamics.  The 11 adjacency-coupled dynamics come first in both files, so
the leading 11 x 11 block is the one to take.

A negative correlation means that dynamics whose equilibria look alike
transfer into each other more accurately, which is the claim the section
makes.  Run after compute_corr.py.  Takes under a second.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "tables"; OUT.mkdir(exist_ok=True)
C = np.load(HERE / "data" / "corr.npz", allow_pickle=True)
F = np.load(HERE.parent / "shared_data" / "fig45_matrices.npz",
            allow_pickle=True)

NADJ = 11
OFF = ~np.eye(NADJ, dtype=bool)          # the 110 ordered pairs, S != T
PANELS = [("mean_max", "Fig. S23(a)"), ("mean_med", "Fig. S23(b)")]
ROWS = [("nn", "NN"), ("lin", "linear")]
SENT = [("minc", "MinCorr"), ("qr", "QR"), ("random", "random")]


def pearson(x, y):
    """Pearson correlation over the cells where both sides are finite."""
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1])


rows = []
for dec, dname in ROWS:
    for sent, sname in SENT:
        M = F[f"M_{dec}_{sent}"][:NADJ, :NADJ]
        B = F[f"B_{dec}_{sent}"][:NADJ, :NADJ]
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(np.isfinite(B) & (B > 0), M / B, np.nan)
        # column order of Table S15: n-RMSE against each panel, then
        # r(S -> T) against each panel
        cells = [f"${pearson(C[key][:NADJ, :NADJ][OFF], perf[OFF]):.3f}$"
                 for perf in (M, R) for key, _ in PANELS]
        rows.append(f"{dname}, {sname} & " + " & ".join(cells) + r"\\")
(OUT / "tab_why.tex").write_text("\n".join(rows) + "\n")
print("  wrote tables/tab_why.tex")
for r in rows:
    print("   ", r)
