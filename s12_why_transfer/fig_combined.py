# =====================================================================
# SI Section S12: Why the transferable dynamics transfer
#
# This script: Fig. S23
# Produces   : corrmatrix_pair.pdf
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Fig. S23: the two Pearson matrices as one two-panel figure, Section S12.

Font sizes follow Fig. 4 of the main text (plot_fig4_fig5.py: CELL_FS 8.4,
TICK_FS 14, LABEL_FS 30 at figsize 22 x 19 rendered across \\textwidth), so
that the two figures look alike on the page.  One color bar serves both
panels and stands to the right of panel (b).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from cellcolor import text_color_fn

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
d = np.load(HERE / "data" / "corr.npz", allow_pickle=True)
LABP = [str(x) for x in d["labels"]]
DISP = [l.replace("degree", "$k$").replace("log $k$", r"$\ln k$")
        for l in LABP]
DISP[LABP.index("log degree")] = r"$\ln k$"
n = len(LABP)
NADJ, NREF = 11, 2

TITLE_FS, LABEL_FS, TICK_FS, CELL_FS = 32, 30, 15.5, 9.2
cmap = plt.get_cmap("RdBu_r")
tc = text_color_fn(cmap, -1, 1)

fig = plt.figure(figsize=(19.6, 10.0))
gs = fig.add_gridspec(1, 2, wspace=0.16)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

for ax, key, lett in [
        (axes[0], "mean_max", "a"),
        (axes[1], "mean_med", "b")]:
    R = d[key]
    im = ax.imshow(R, cmap=cmap, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            v = R[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-.", "−."),
                    ha="center", va="center", fontsize=CELL_FS, color=tc(v))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(DISP, rotation=90, fontsize=TICK_FS)
    ax.set_yticklabels(DISP, fontsize=TICK_FS)
    for b in (NADJ - 0.5, NADJ + NREF - 0.5):
        ax.axhline(b, color="k", lw=2.0); ax.axvline(b, color="k", lw=2.0)
    ax.set_title(f"({lett})", fontsize=TITLE_FS, loc="left", pad=12)

# one color bar for both panels, standing to the right of panel (b)
fig.canvas.draw()
pb = axes[1].get_position()
cax = fig.add_axes([pb.x1 + 0.030 * pb.width, pb.y0,
                    0.028 * pb.width, pb.height])
cb = fig.colorbar(im, cax=cax)
cb.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
cb.set_ticklabels(["\u22121", "\u22120.5", "0", "0.5", "1"])
cb.set_label("mean Pearson correlation", fontsize=LABEL_FS, labelpad=7)
cb.ax.tick_params(labelsize=TICK_FS)
cax.yaxis.set_ticks_position("right")
cax.yaxis.set_label_position("right")

fig.savefig(FIGS / "corrmatrix_pair.pdf", bbox_inches="tight")
print("  wrote corrmatrix_pair.pdf")
