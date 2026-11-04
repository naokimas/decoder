# =====================================================================
# SI Section S7: Variability over networks and random factors
#
# This script: Figs. S9 and S10
# Produces   : figs/cv_nrmse.pdf, figs/frac_beats.pdf
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Compare the dispersion contributed by networks with that of each other
source of randomness, cell by cell.

For every cell of the Fig. 4 and Fig. 5 matrices we have
  base    14 values, one per network, everything else at the paper's seeds
  <mode>  14 x R values, the same but with one extra source of randomness on
and we compare the standard deviation of the two samples.

Writes the two figures of Section S7: figs/cv_nrmse.pdf (Fig. S9), the
coefficient of variation of the n-RMSE over the 14 networks, and
figs/frac_beats.pdf (Fig. S10), the percentage of networks on which the
decoder beats the mean-prediction baseline.
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
FIG = HERE / "figs"; FIG.mkdir(exist_ok=True)
PANELS = [("a", "NN, MinCorr", "nn", "minc"), ("b", "NN, QR", "nn", "qr"),
          ("c", "NN, random", "nn", "random"),
          ("d", "linear, MinCorr", "lin", "minc"),
          ("e", "linear, QR", "lin", "qr"),
          ("f", "linear, random", "lin", "random")]
MODES = [("nn", "NN initialization", "#1f77b4"),
         ("split", "train/test splits", "#d62728"),
         ("sent", "random sentinel set", "#2ca02c"),
         ("hetero", "node parameters", "#9467bd")]
HETERO = {"NM", "EUT1", "EUT2", "VEG1", "VEG2"}


def load(mode):
    f = HERE / "data" / f"rep_{mode}.npz"
    return np.load(f, allow_pickle=True) if f.exists() else None


def main():
    base = load("base")
    labels = [str(x) for x in base["dynamics"]]
    n = len(labels)
    hidx = np.array([i for i, l in enumerate(labels) if l in HETERO])
    hmask = np.zeros((n, n), bool); hmask[hidx, :] = True; hmask[:, hidx] = True
    data = {m: load(m) for m, _, _ in MODES}

    rows = []
    # ---- coefficient of variation across the networks ------------------
    # Layout, fonts and color-bar geometry follow the main text's Fig. 4
    # (fig4_fig5_cross_transfer/plot_fig4_fig5.py).
    TITLE_FS, LABEL_FS, TICK_FS, CELL_FS = 30, 30, 14, 8.4

    def six_panel(mat_of, fname, vmin, vmax, cmap, fmt, cbar_label, textcol,
                  cbar_ticks=None):
        fig = plt.figure(figsize=(22.0, 19.0))
        gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.5 / n, hspace=0.28,
                               left=0.07, right=0.95, top=0.95, bottom=0.06)
        last = None
        for k, (lett, title, dec, sent) in enumerate(PANELS):
            ax = fig.add_subplot(gs[k // 3, k % 3])
            M = mat_of(dec, sent)
            last = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap,
                             aspect="auto")
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=45, fontsize=TICK_FS)
            if k % 3 == 0:
                ax.set_yticklabels(labels, fontsize=TICK_FS)
                ax.set_ylabel("source decoder", fontsize=LABEL_FS)
            else:
                ax.set_yticklabels([""] * n)
            ax.set_xlabel("target dynamics", fontsize=LABEL_FS)
            ax.set_title(f"({lett}) {title}", fontsize=TITLE_FS, loc="left",
                         pad=8)
            for i in range(n):
                for j in range(n):
                    v = M[i, j]
                    if not np.isfinite(v):
                        continue
                    ax.text(j, i, fmt(v), ha="center", va="center",
                            fontsize=CELL_FS, color=textcol(v))
        cax = fig.add_axes([0.965, 0.10, 0.010, 0.80])
        cb = fig.colorbar(last, cax=cax)
        cb.set_label(cbar_label, fontsize=LABEL_FS)
        cb.ax.tick_params(labelsize=TICK_FS)
        if cbar_ticks is not None:
            cb.set_ticks(cbar_ticks)
        fig.savefig(FIG / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote figs/{fname}")

    def cv_of(dec, sent):
        X = base[f"M_{dec}_{sent}"][0] * 100.0
        return np.nanstd(X, axis=0, ddof=1) / np.nanmean(X, axis=0)

    def frac_of(dec, sent):
        M = base[f"M_{dec}_{sent}"][0]; B = base[f"B_{dec}_{sent}"][0]
        return np.nanmean(np.where(B > 0, M / B, np.nan) < 1.0, axis=0) * 100

    # RdBu_r as in Fig. 5 of the main text, rescaled so that white sits at
    # 0.5; blue is less network-dependent than typical and red more so
    six_panel(cv_of, "cv_nrmse.pdf", 0.0, 1.0, "RdBu_r",
              lambda v: f"{v:.2f}",
              "CV of the n-RMSE across the networks",
              text_color_fn("RdBu_r", 0.0, 1.0),
              cbar_ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    six_panel(frac_of, "frac_beats.pdf", 0.0, 100.0, "RdYlBu",
              lambda v: f"{v:.0f}",
              r"networks with $r(\mathrm{S}\to\mathrm{T})<1$ (%)",
              lambda v: "black",
              cbar_ticks=[0, 25, 50, 75, 100])


if __name__ == "__main__":
    main()
