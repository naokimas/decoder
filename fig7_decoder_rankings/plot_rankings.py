# =====================================================================
# Main text Fig. 7: source-decoder rankings
#
# This script: the scatter of the four leading source decoders
# Feeds      : Fig. 7
# Produces   : figs/scatter_topfour_rev.png
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Main-text Fig. 7: the leading source decoders.

Marker shape codes the source decoder and fill color codes the (decoder,
sentinel) combination, with darker hues for the NN decoder.  The
all-dynamics pool is C17, and the double-triangle points average over all 17
target dynamics.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path as MPath

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
# The panel list and the rank.npz reader live with Section S13's tables, which
# rank the same decoders; importing them keeps the two in step.
sys.path.insert(0, str(HERE.parent / "s13_decoder_rankings"))
import make_tables as mt                                 # noqa: E402
PANELS = mt.PANELS
KEEP = ["SIS", "SIR", "C17", "C11"]
MARKER = {"SIS": "o", "SIR": "s", "C17": "^", "C11": "v"}
COLOR = {("nn", "minc"): "#1f4d8b", ("nn", "qr"): "#1f6f2e",
         ("nn", "random"): "#8b1f1f",
         ("lin", "minc"): "#5b8fbe", ("lin", "qr"): "#a5d6a7",
         ("lin", "random"): "#c87878"}
ALPHA = {p: (0.60 if p == ("lin", "qr") else 0.78) for p in PANELS}
METH = {"minc": "MinCorr", "qr": "QR", "random": "random"}
DEC = {"nn": "NN", "lin": "linear"}


def double_triangle():
    out = [(0.0, 0.7), (-0.6, -0.4), (0.6, -0.4), (0.0, 0.7)]
    inn = [(0.0, 0.35), (0.3, -0.2), (-0.3, -0.2), (0.0, 0.35)]
    codes = ([MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY] * 2)
    return MarkerStyle(MPath(out + inn, codes))


fig, ax = plt.subplots(figsize=(9.0, 7.2))
for s in KEEP:
    for p in PANELS:
        rm, rt = mt.scores(p, "t11")[s]
        ax.scatter(rm * 100, rt, marker=MARKER[s], s=160, c=COLOR[p],
                   edgecolor="black", linewidth=0.9, alpha=ALPHA[p], zorder=3)
dbl = double_triangle()
for p in PANELS:
    rm, rt = mt.scores(p, "all")["C17"]
    ax.scatter(rm * 100, rt, marker=dbl, s=260, c=COLOR[p],
               edgecolor="black", linewidth=0.9, alpha=ALPHA[p], zorder=4)
for p in PANELS:
    rm, rt = mt.scores(p, "t11")["C11L1"]
    ax.scatter(rm * 100, rt, marker="D", s=115, c=COLOR[p],
               edgecolor="black", linewidth=0.9, alpha=ALPHA[p], zorder=4)

ax.set_xlabel("n-RMSE (%)", fontsize=18)
ax.set_ylabel(r"$r(\mathrm{S} \to \mathrm{T})$", fontsize=18)
ax.set_xlim(0, 8.0); ax.set_ylim(0, 0.6)
ax.grid(alpha=0.4, linewidth=0.5, zorder=0); ax.tick_params(labelsize=15)
fig.canvas.draw()
ax.set_yticklabels(["0" if abs(v) < 1e-9 else t.get_text()
                    for v, t in zip(ax.get_yticks(), ax.get_yticklabels())])

h = [Line2D([0], [0], marker=MARKER[s], color="w", markerfacecolor="lightgray",
            markeredgecolor="black", markersize=11, alpha=0.85,
            label=f"{s}, 11 target dynamics") for s in KEEP]
h.append(Line2D([0], [0], marker="D", color="w", markerfacecolor="lightgray",
                markeredgecolor="black", markersize=9, alpha=0.85,
                label="C11L1, 11 target dynamics"))
h.append(Line2D([0], [0], marker=dbl, color="w", markerfacecolor="lightgray",
                markeredgecolor="black", markersize=13, alpha=0.85,
                label="C17, 17 target dynamics"))
h.append(Line2D([0], [0], marker="None", linestyle="None", label=" "))
h += [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR[p],
             markeredgecolor="black", markersize=11, alpha=ALPHA[p],
             label=f"{DEC[p[0]]}, {METH[p[1]]}") for p in PANELS]
ax.legend(handles=h, loc="upper left", fontsize=13, frameon=False,
          handletextpad=0.4)
fig.tight_layout()
fig.savefig(FIGS / "scatter_topfour_rev.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  wrote figs/scatter_topfour_rev.png")
