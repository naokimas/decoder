# =====================================================================
# SI Section S3: Gappy POD
#
# This script: Figs. S3 and S4
# Produces   : figs/gappy_nrmse.pdf, figs/gappy_ratio.pdf
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Figures for the gappy-POD baseline.

  figs/gappy_nrmse.pdf        Fig. S3: 3 panels, n-RMSE of gappy POD at r = n
  figs/gappy_ratio.pdf        Fig. S4: 6 panels, decoder n-RMSE divided by
                              gappy POD's, at the uncentered r = n denominator
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
(HERE / "figs").mkdir(exist_ok=True)
G = np.load(HERE / "data" / "gappy_matrices.npz", allow_pickle=True)
F = np.load(HERE.parent / "shared_data" /
            "fig45_matrices.npz", allow_pickle=True)
L = [str(x) for x in G["dynamics"]]
NN = len(L)
TITLE_FS, LABEL_FS, TICK_FS, CELL_FS = 26, 26, 12, 7.2


def panels(mats, titles, fname, vmin, vmax, cmap, fmt, cbar_label,
           ncol, cbar_ticks=None, textcol=None):
    nrow = int(np.ceil(len(mats) / ncol))
    fig = plt.figure(figsize=(7.4 * ncol, 7.2 * nrow))
    gs = gridspec.GridSpec(nrow, ncol, figure=fig, wspace=0.5 / NN,
                           hspace=0.30, left=0.08, right=0.94,
                           top=0.94, bottom=0.08)
    last = None
    for k, (M, ttl) in enumerate(zip(mats, titles)):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        last = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_xticks(range(NN)); ax.set_yticks(range(NN))
        ax.set_xticklabels(L, rotation=45, fontsize=TICK_FS)
        if k % ncol == 0:
            ax.set_yticklabels(L, fontsize=TICK_FS)
            ax.set_ylabel("source dynamics", fontsize=LABEL_FS)
        else:
            ax.set_yticklabels([""] * NN)
        ax.set_xlabel("target dynamics", fontsize=LABEL_FS)
        ax.set_title(ttl, fontsize=TITLE_FS, loc="left", pad=8)
        for i in range(NN):
            for j in range(NN):
                v = M[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "--", ha="center", va="center",
                            fontsize=CELL_FS, color="gray"); continue
                col = (textcol(v) if textcol is not None
                       else "white" if v < (vmin + vmax) / 2.0 else "black")
                ax.text(j, i, fmt(v), ha="center", va="center",
                        fontsize=CELL_FS, color=col)
    cax = fig.add_axes([0.955, 0.12, 0.010, 0.74])
    cb = fig.colorbar(last, cax=cax)
    cb.set_label(cbar_label, fontsize=LABEL_FS)
    cb.ax.tick_params(labelsize=TICK_FS)
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)
    fig.savefig(HERE / "figs" / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figs/{fname}")


SENT = [("minc", "MinCorr"), ("qr", "QR"), ("random", "random")]
# Fig. S3: the uncentered r = n variant, one panel per sentinel method.  The
# energy-truncated variant appears in Table S3, not as a figure.
mats = [G[f"G_minmax_{s}_rn"] * 100 for s, _ in SENT]
ttls = [f"({c}) gappy POD, {d}" for c, (_, d) in zip("abc", SENT)]
panels(mats, ttls, "gappy_nrmse.pdf", 0.0, 50.0, "viridis",
       lambda v: f"{v:.1f}", r"n-RMSE (%)", 3,
       textcol=text_color_fn("viridis", 0.0, 50.0))

# The denominator is always gappy POD with QR sentinels, because that is the
# only configuration in which gappy POD is a well-posed baseline.  The
# numerator varies over the decoder and its own sentinel method.
mats, ttls = [], []
Gq = G["G_minmax_qr_rn"]
for dec, dn in (("nn", "NN"), ("lin", "linear")):
    for s, d in SENT:
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(np.isfinite(Gq) & (Gq > 0), F[f"M_{dec}_{s}"] / Gq,
                         np.nan)
        mats.append(R); ttls.append(f"{dn}, {d}")
ttls = [f"({c}) {t}" for c, t in zip("abcdef", ttls)]
panels(mats, ttls, "gappy_ratio.pdf", 0.0, 2.0, "RdBu_r",
       lambda v: f"{v:.2f}",
       r"$r_{\rm gappy}(\mathrm{S}\to\mathrm{T})$", 3,
       cbar_ticks=[0.0, 0.5, 1.0, 1.5, 2.0],
       textcol=text_color_fn("RdBu_r", 0.0, 2.0))
