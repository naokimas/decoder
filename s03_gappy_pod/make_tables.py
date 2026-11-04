# =====================================================================
# SI Section S3: Gappy POD
#
# This script: Tables S3 and S4
# Produces   : tables/tab_gappy_summary.tex, tables/tab_gappy_ratio.tex
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""LaTeX table bodies for SI Section S3 (Gappy POD).

Reads data/gappy_matrices.npz, which compute_gappy.py writes, and emits the
two table bodies that appear in Section S3:

  tables/tab_gappy_summary.tex   Table S3: median n-RMSE (%) over the
                                 diagonal and the off-diagonal cells of the
                                 17 x 17 source-target matrix, for the four
                                 gappy POD variants and the two decoders.
  tables/tab_gappy_ratio.tex     Table S4: the median, over the
                                 off-diagonal cells, of the decoder's n-RMSE
                                 divided by that of gappy POD, for each of
                                 the four gappy POD configurations used as
                                 the denominator.

The medians are taken over cells of the matrix that is already averaged over
the 14 networks, which is the convention stated in the captions.

Run after compute_gappy.py.  Takes under a second.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "tables"; OUT.mkdir(exist_ok=True)
d = np.load(HERE / "data" / "gappy_matrices.npz", allow_pickle=True)
F = np.load(HERE.parent / "shared_data" / "fig45_matrices.npz",
            allow_pickle=True)
N = len(d["dynamics"])
EYE = np.eye(N, dtype=bool)

# (key in the npz, label shown in the table)
VARIANTS = [("minmax", "rn",   r"gappy POD, uncentered, $r=n$"),
            ("minmax", "e999", r"gappy POD, uncentered, $99.9\%$"),
            ("center", "rn",   r"gappy POD, centered, $r=n$"),
            ("center", "e999", r"gappy POD, centered, $99.9\%$")]
SENT = [("minc", "MinCorr"), ("qr", "QR"), ("random", "random")]
DECS = [("nn", "shallow NN decoder"), ("lin", "linear ridge decoder")]

rows = []
for vi, (cen, rk, label) in enumerate(VARIANTS):
    for si, (s, sname) in enumerate(SENT):
        M = d[f"G_{cen}_{s}_{rk}"] * 100
        lead = label if si == 0 else ""
        rows.append(f"{lead} & {sname} & {np.median(M[EYE]):.2f} & "
                    f"{np.median(M[~EYE]):.2f}" + r"\\")
    rows.append(r"\addlinespace")
for dec, dname in DECS:
    for si, (s, sname) in enumerate(SENT):
        M = F[f"M_{dec}_{s}"] * 100
        lead = dname if si == 0 else ""
        rows.append(f"{lead} & {sname} & {np.median(M[EYE]):.2f} & "
                    f"{np.median(M[~EYE]):.2f}" + r"\\")
    if dec != DECS[-1][0]:
        rows.append(r"\addlinespace")
(OUT / "tab_gappy_summary.tex").write_text("\n".join(rows) + "\n")
print("  wrote tables/tab_gappy_summary.tex")
for r in rows:
    if "&" in r:
        print("   ", r)


# --------------------------------------------------------------------------
# Table S4: r_gappy(S -> T), the decoder's n-RMSE over gappy POD's.
#
# The denominator always uses the QR-pivoted sentinels -- the only sentinel
# set that keeps (U_r)_S well conditioned -- but runs over all four gappy POD
# configurations, one per column, as the caption of Table S4 states.  The
# numerator varies over the decoder and over the decoder's own sentinel
# method, one per row.  The decoder matrices come from the main-text figure
# data in ../shared_data/.  Fig. S4 shows the same ratio cell by cell, but
# only for the uncentered r = n denominator.
# --------------------------------------------------------------------------
# Table S4 abbreviates the decoder names that Table S3 spells out.
DECS_S4 = [("nn", "NN"), ("lin", "linear")]

rows = []
for di, (dec, dname) in enumerate(DECS_S4):
    for s, sname in SENT:
        cells = []
        for cen, rk, _ in VARIANTS:
            Gq = d[f"G_{cen}_qr_{rk}"]
            with np.errstate(divide="ignore", invalid="ignore"):
                R = np.where(np.isfinite(Gq) & (Gq > 0),
                             F[f"M_{dec}_{s}"] / Gq, np.nan)
            cells.append(f"${np.nanmedian(R[~EYE]):.3f}$")
        rows.append(f"{dname} & {sname} & " + " & ".join(cells) + r" \\")
    if di < len(DECS_S4) - 1:
        rows.append(r"\addlinespace")
(OUT / "tab_gappy_ratio.tex").write_text("\n".join(rows) + "\n")
print("  wrote tables/tab_gappy_ratio.tex")
for r in rows:
    if "&" in r:
        print("   ", r)
