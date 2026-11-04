# =====================================================================
# SI Section S14: Robustness to NN bias terms and ridge strength
#
# This script: Figs. S24 and S25
# Produces   : heatmap_nn_bias_rev.png, heatmap_lin_alpha0p1_rev.png
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""SI Section S14 figures: the two decoder-specification robustness heatmaps.

Three panels each, in the style of Figs. 4 and 5 of the main text, on the
17 dynamics and with the out-of-sample target calibration.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HERE = Path(__file__).resolve().parent
# the rank and budget sweeps are Section S13's, shared with this section
DATA = HERE.parent / "s13_decoder_rankings" / "data"
sys.path.insert(0, str(HERE.parent / "lib"))
from cellcolor import text_color_fn                             # noqa: E402

FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
TITLE_FS = 30; LABEL_FS = 30; TICK_FS = 14; CELL_FS = 8.4


def three_panel(mats, labels, fname, titles):
    n = len(labels)
    fig = plt.figure(figsize=(22.0, 9.8))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.5 / n,
                           left=0.07, right=0.95, top=0.92, bottom=0.12)
    tc = text_color_fn("viridis", 0.0, 50.0)
    last = None
    for k, (lett, title, key) in enumerate(titles):
        ax = fig.add_subplot(gs[0, k])
        M = mats[key]
        last = ax.imshow(M, vmin=0.0, vmax=50.0, cmap="viridis",
                         aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, fontsize=TICK_FS)
        ax.set_yticks(range(n))
        if k == 0:
            ax.set_yticklabels(labels, fontsize=TICK_FS)
            ax.set_ylabel("source decoder", fontsize=LABEL_FS)
        else:
            ax.set_yticklabels([""] * n)
        ax.set_xlabel("target dynamics", fontsize=LABEL_FS)
        ax.set_title(f"({lett}) {title}", fontsize=TITLE_FS, loc="left",
                     pad=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        fontsize=CELL_FS, color=tc(M[i, j]))
    cax = fig.add_axes([0.965, 0.14, 0.010, 0.74])
    cb = fig.colorbar(last, cax=cax)
    cb.set_label(r"n-RMSE (%)", fontsize=LABEL_FS)
    cb.ax.tick_params(labelsize=TICK_FS)
    fig.savefig(FIGS / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fname}")


d0 = np.load(DATA / "rank.npz", allow_pickle=True)
labels = [str(x) for x in d0["dynamics"]]

db = np.load(DATA / "rank_bias.npz", allow_pickle=True)
three_panel({s: db[f"M_nn_{s}_single"] * 100 for s in
             ("minc", "qr", "random")},
            labels, "heatmap_nn_bias_rev.png",
            [("a", "MinCorr", "minc"), ("b", "QR", "qr"),
             ("c", "random", "random")])

da = np.load(DATA / "rank_alpha01.npz", allow_pickle=True)
three_panel({s: da[f"M_lin_{s}_single"] * 100 for s in
             ("minc", "qr", "random")},
            labels, "heatmap_lin_alpha0p1_rev.png",
            [("a", "MinCorr", "minc"), ("b", "QR", "qr"),
             ("c", "random", "random")])

# ---- the shift statistics quoted in the text ----
print("\nshifts against the main-text specification (percentage points):")
for tag, dv, dec in (("bias-restored NN", db, "nn"),
                     ("linear, alpha=1e-1", da, "lin")):
    for s in ("minc", "qr", "random"):
        A = d0[f"M_{dec}_{s}_single"] * 100
        Bm = dv[f"M_{dec}_{s}_single"] * 100
        D = Bm - A
        off = ~np.eye(len(labels), dtype=bool)
        print(f"  {tag:20s} {s:7s}: diag mean {np.diag(D).mean():+.2f}   "
              f"off-diag mean {D[off].mean():+.2f}   "
              f"|shift|<1pp {100*np.mean(np.abs(D) < 1):.0f}%   "
              f"<2pp {100*np.mean(np.abs(D) < 2):.0f}%")
    A = np.concatenate([d0[f"M_{dec}_{s}_single"].ravel() for s in ("minc", "qr")]) * 100
    Bm = np.concatenate([dv[f"M_{dec}_{s}_single"].ravel() for s in ("minc", "qr")]) * 100
    D = np.abs(Bm - A)
    print(f"  {tag:20s} non-random pooled: |shift|<1pp {100*np.mean(D<1):.0f}%"
          f"   <2pp {100*np.mean(D<2):.0f}%")
