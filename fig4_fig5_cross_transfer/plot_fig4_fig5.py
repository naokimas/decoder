# =====================================================================
# Main text Figs. 4 and 5: the cross-transfer matrices
#
# This script: the two 6-panel matrix figures
# Feeds      : Figs. 4 and 5
# Produces   : figs/heatmap_rev.png, figs/ratio_rev.png
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Render Figs. 4 and 5 from ../shared_data/fig45_matrices.npz.

Each figure has six panels, one per combination of decoder architecture
(NN, linear) and sentinel-selection method (MinCorr, QR, random), each
showing a 17 x 17 source-by-target matrix.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from cellcolor import text_color_fn          # noqa: E402

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "shared_data"   # the matrices many sections read
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
PANELS = [("a", "NN, MinCorr", "nn", "minc"), ("b", "NN, QR", "nn", "qr"),
          ("c", "NN, random", "nn", "random"),
          ("d", "linear, MinCorr", "lin", "minc"),
          ("e", "linear, QR", "lin", "qr"),
          ("f", "linear, random", "lin", "random")]
TITLE_FS = 30; LABEL_FS = 30; TICK_FS = 14; CELL_FS = 8.4; TITLE_LIFT_PAD = 8


def six_panel(mats, labels, fname, vmin, vmax, cmap, fmt, cbar_label,
              textcol, cbar_ticks=None):
    n = len(labels)
    fig = plt.figure(figsize=(22.0, 19.0))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.5 / n, hspace=0.28,
                           left=0.07, right=0.95, top=0.95, bottom=0.06)
    last = None
    for k, (lett, title, dec, meth) in enumerate(PANELS):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        M = mats[(dec, meth)]
        last = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, fontsize=TICK_FS)
        ax.set_yticks(range(n))
        if k % 3 == 0:
            ax.set_yticklabels(labels, fontsize=TICK_FS)
            ax.set_ylabel("source decoder", fontsize=LABEL_FS)
        else:
            ax.set_yticklabels([""] * n)
        ax.set_xlabel("target dynamics", fontsize=LABEL_FS)
        ax.set_title(f"({lett}) {title}", fontsize=TITLE_FS, loc="left",
                     pad=TITLE_LIFT_PAD)
        for i in range(n):
            for j in range(n):
                v = M[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "--", ha="center", va="center",
                            fontsize=CELL_FS, color="gray"); continue
                ax.text(j, i, fmt(v), ha="center", va="center",
                        fontsize=CELL_FS, color=textcol(v))
    cax = fig.add_axes([0.965, 0.10, 0.010, 0.80])
    cb = fig.colorbar(last, cax=cax)
    cb.set_label(cbar_label, fontsize=LABEL_FS)
    cb.ax.tick_params(labelsize=TICK_FS)
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)
    fig.savefig(FIGS / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fname}")


def main():
    d = np.load(SHARED / "fig45_matrices.npz", allow_pickle=True)
    labels = [str(x) for x in d["dynamics"]]
    heat, ratio = {}, {}
    for _, _, dec, meth in PANELS:
        M = d[f"M_{dec}_{meth}"]; B = d[f"B_{dec}_{meth}"]
        heat[(dec, meth)] = M * 100.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio[(dec, meth)] = np.where(np.isfinite(B) & (B > 0),
                                          M / B, np.nan)
    # viridis is dark at the low end, so white text only there
    six_panel(heat, labels, "heatmap_rev.png", 0.0, 50.0, "viridis",
              lambda v: f"{v:.1f}", r"n-RMSE (%)",
              text_color_fn("viridis", 0.0, 50.0))
    # RdBu_r is pale around r = 1, where white text would be invisible,
    # so only the saturated ends take white text
    six_panel(ratio, labels, "ratio_rev.png", 0.0, 2.0, "RdBu_r",
              lambda v: f"{v:.2f}", r"$r(\mathrm{S}\to\mathrm{T})$",
              text_color_fn("RdBu_r", 0.0, 2.0),
              cbar_ticks=[0.0, 0.5, 1.0, 1.5, 2.0])


if __name__ == "__main__":
    main()
