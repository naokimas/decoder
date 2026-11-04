# =====================================================================
# SI Section S6: Two-parameter families of dynamics
#
# This script: Figs. S7 and S8
# Produces   : figs/twod_heatmap.pdf, figs/twod_ratio.pdf
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""The Figs. 4 and 5 equivalents, restricted to the two two-parameter families.

The full matrix is 19 x 19 (the manuscript's 17 dynamics plus MSIS2P and
MUT2P).  Only the two two-parameter rows and columns are shown, with the
columns transposed so that all four appear as rows:

    row 1   MSIS2P as the source decoder, against every target
    row 2   MUT2P as the source decoder, against every target
            (blank separator)
    row 3   every source decoder, against MSIS2P as the target
    row 4   every source decoder, against MUT2P as the target

figs/twod_heatmap.pdf   n-RMSE, the Fig. 4 equivalent
figs/twod_ratio.pdf     r(S -> T), the Fig. 5 equivalent
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
FIG = HERE / "figs"
FIG.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# column-major so that the letters keep the manuscript's order while the
# sentinel methods line up in rows
PANELS = [("a", "NN, MinCorr", "nn", "minc"), ("b", "NN, QR", "nn", "qr"),
          ("c", "NN, random", "nn", "random"),
          ("d", "linear, MinCorr", "lin", "minc"),
          ("e", "linear, QR", "lin", "qr"),
          ("f", "linear, random", "lin", "random")]
ROWLAB = [r"MSIS2P $\to$ T", r"MUT2P $\to$ T", "",
          r"S $\to$ MSIS2P", r"S $\to$ MUT2P"]
TITLE_FS = 28; LABEL_FS = 22; TICK_FS = 18; CELL_FS = 13


def four_rows(M, lo, up):
    """The two rows, a blank, then the two columns transposed."""
    out = np.full((5, M.shape[1]), np.nan)
    out[0] = M[lo, :]
    out[1] = M[up, :]
    out[3] = M[:, lo]
    out[4] = M[:, up]
    return out


def six_panel(mats, labels, lo, up, fname, vmin, vmax, cmap, fmt,
              cbar_label, textcol, cbar_ticks=None):
    n = len(labels)
    cm = matplotlib.colormaps[cmap].copy()
    cm.set_bad("white")
    fig = plt.figure(figsize=(26.0, 12.0))
    gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.045, hspace=0.55,
                           left=0.055, right=0.94, top=0.93, bottom=0.05)
    last = None
    for k, (lett, title, dec, meth) in enumerate(PANELS):
        ax = fig.add_subplot(gs[k % 3, k // 3])
        A = four_rows(mats[(dec, meth)], lo, up)
        last = ax.imshow(A, vmin=vmin, vmax=vmax, cmap=cm, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, fontsize=TICK_FS,
                           ha="right")
        ax.set_yticks(range(5))
        ax.set_yticklabels(ROWLAB if k // 3 == 0 else [""] * 5,
                           fontsize=TICK_FS)
        if k % 3 == 2:
            ax.set_xlabel("S or T", fontsize=LABEL_FS)
        ax.set_title(f"({lett}) {title}", fontsize=TITLE_FS, loc="left",
                     pad=6)
        for i in range(5):
            for j in range(n):
                v = A[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(j, i, fmt(v), ha="center", va="center",
                        fontsize=CELL_FS, color=textcol(v))
    cax = fig.add_axes([0.955, 0.08, 0.009, 0.84])
    cb = fig.colorbar(last, cax=cax)
    cb.set_label(cbar_label, fontsize=LABEL_FS)
    cb.ax.tick_params(labelsize=TICK_FS)
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)
    fig.savefig(FIG / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figs/{fname}")


def main():
    d = np.load(HERE / "data" / "twod45_matrices.npz", allow_pickle=True)
    labels = [str(x) for x in d["dynamics"]]
    lo, up = labels.index("MSIS2P"), labels.index("MUT2P")
    heat, ratio = {}, {}
    for _, _, dec, meth in PANELS:
        M = d[f"M_{dec}_{meth}"]; B = d[f"B_{dec}_{meth}"]
        heat[(dec, meth)] = M * 100.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio[(dec, meth)] = np.where(np.isfinite(B) & (B > 0), M / B,
                                          np.nan)
    six_panel(heat, labels, lo, up, "twod_heatmap.pdf", 0.0, 50.0, "viridis",
              lambda v: f"{v:.1f}", r"n-RMSE (%)",
              text_color_fn("viridis", 0.0, 50.0))
    # RdBu_r is pale near r = 1, so only the saturated ends take white text
    six_panel(ratio, labels, lo, up, "twod_ratio.pdf", 0.0, 2.0, "RdBu_r",
              lambda v: f"{v:.2f}", r"$r(\mathrm{S}\to\mathrm{T})$",
              text_color_fn("RdBu_r", 0.0, 2.0),
              cbar_ticks=[0.0, 0.5, 1.0, 1.5, 2.0])


if __name__ == "__main__":
    main()
