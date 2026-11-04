# =====================================================================
# SI Section S8: Reconstruction under measurement noise
#
# This script: Figs. S11 and S12
# Produces   : figs/si_noise_nrmse.pdf, figs/si_noise_ratio.pdf
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Figs. S11 and S12 of SI Section S8: reconstruction under measurement noise.

Only the fully observed-data protocol is plotted: the sentinel selection,
the decoder training, the calibration and the mean-prediction baseline all
see nothing but the noisy measurements.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
d = np.load(HERE / "data" / "noise.npz", allow_pickle=True)
eps = d["eps"]; CASES = [str(c) for c in d["cases"]]
LAB = {c: str(d[f"{c}__label"]) for c in CASES}
LET = "abcd"
COL = "#4C72B0"                      # a mid blue, for line and band
FS_AX, FS_TICK, FS_TITLE = 22.5, 13, 21
SEL = (eps > 0) & (eps <= 1.0)       # 1 is already a huge error


def _pow10(v, _pos):
    """Decade labels, with 10^0 written as 1."""
    if v <= 0:
        return ""
    e = np.log10(v)
    if abs(e) < 1e-9:
        return "1"
    if abs(e - round(e)) < 1e-9:
        return rf"$10^{{{int(round(e))}}}$"
    return ""


def panels(get, ylab, fname, hline=None):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8))
    for k, (ax, c) in enumerate(zip(axes.ravel(), CASES)):
        A = get(c)
        ax.fill_between(eps[SEL], np.nanpercentile(A[SEL], 10, axis=1),
                        np.nanpercentile(A[SEL], 90, axis=1), color=COL,
                        alpha=0.22, lw=0)
        ax.plot(eps[SEL], np.nanmean(A[SEL], axis=1), "-o", color=COL,
                ms=4, lw=1.8)
        ax.axhline(np.nanmean(A[0]), ls="--", color="#444444", lw=1.1)
        if hline is not None:
            ax.axhline(hline, ls="-", color="black", lw=1.4)
        ax.set_xscale("log")                     # y stays linear
        ax.xaxis.set_major_formatter(FuncFormatter(_pow10))
        ax.set_xlim(eps[SEL].min() * 0.85, 1.15)
        ax.set_xlabel(r"$\epsilon$", fontsize=FS_AX)
        ax.set_ylabel(ylab, fontsize=FS_AX)
        ax.set_title(f"({LET[k]}) {LAB[c]}", fontsize=FS_TITLE, loc="left",
                     pad=8)
        ax.grid(alpha=0.3, lw=0.4, which="both")
        ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
    fig.tight_layout()
    fig.savefig(FIGS / fname, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {fname}")


panels(lambda c: d[f"{c}__M"] * 100, "n-RMSE (%)",
       "si_noise_nrmse.pdf")          # Fig. S11 of the SI
panels(lambda c: d[f"{c}__M"] / d[f"{c}__B"],
       r"$r(\mathrm{S}\to\mathrm{T})$",
       "si_noise_ratio.pdf", hline=1.0)               # Fig. S12 of the SI
