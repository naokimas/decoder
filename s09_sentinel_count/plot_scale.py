# =====================================================================
# SI Section S9: Robustness to the number of sentinel nodes
#
# This script: Figs. S13 to S18
# Produces   : figs/si_ranks14.pdf, figs/si_scale_*.pdf
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Figs. S13 to S18 of SI Section S9: effective rank and sentinel count against network size."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
d = np.load(HERE / "data" / "scale.npz", allow_pickle=True)
M, B = d["M"], d["B"]
M = M[:, :, :, :, :11, :11]; B = B[:, :, :, :, :11, :11]   # adjacency block
R = M / B
ns = list(d["ns"]); sizes = list(d["sizes"])
TAGS = ["ba1p5", "ba2p0", "ba2p5", "ba3p0", "ba3p5", "ba4p0"]
DYN11 = ["SIS", "SIR", "MSIS", "MSIS2", "MSIR", "MSIR2", "DW", "MUT", "GEN",
         "WC", "gLV"]
COMBOS = [(0, 0, "NN, MinCorr"), (0, 1, "NN, QR"),
          (1, 0, "linear, MinCorr"), (1, 1, "linear, QR")]
RULE = [int(np.floor(np.log(N))) for N in sizes]

# target ranges, used both to undo the normalization and to pick the subset
w = np.load(HERE / "data" / "why.npz", allow_pickle=True)
RNG = np.array([[{str(k): w["vals"][i, 2] for i, k in enumerate(w["keys"])}
                 [f"{s}|{t}"] for t in TAGS] for s in DYN11])   # (11, 6)
GROWTH = RNG[:, -1] / RNG[:, 0]
SUB = [j for j in range(11) if GROWTH[j] < 3.0]     # range stays bounded
print("  bounded-range subset: "
      + ", ".join(f"{DYN11[j]} ({GROWTH[j]:.2f})" for j in SUB))

# size-invariant error: RMSE divided by the target's range on the SMALLEST
# network, so the normalizer no longer depends on N
_fac = (RNG / RNG[:, [0]]).T                      # (6 sizes, 11 targets)
SI = M * _fac[:, None, None, None, None, :]


def block(X, di, si, diag, idx, scale=1.0):
    Y = X[:, :, di, si][:, :, idx][:, :, :, idx]
    e = np.eye(len(idx), dtype=bool)
    return np.nanmedian(Y[..., e if diag else ~e], axis=-1) * scale




def _sizelab(N):
    e = np.log10(N)
    return f"$10^{{{e:.0f}}}$" if abs(e - round(e)) < 1e-9 else \
        f"$10^{{{e:.1f}}}$"


def heatmaps(X, scale, label, fname, idx, fmt="{:.2f}", note=""):
    """Two rows of four panels, one shared color scale per row."""
    FS_AX, FS_TICK, FS_TITLE, FS_CELL = 17, 14, 18, 11
    fig, axes = plt.subplots(2, 4, figsize=(21.0, 10.0),
                             constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.012, wspace=0.015)
    LET = "abcdefgh"
    rows = ((True, "within dynamics"), (False, "cross transfer"))
    V = {(r, c): block(X, COMBOS[c][0], COMBOS[c][1], rows[r][0], idx,
                       scale).T for r in (0, 1) for c in range(4)}
    for r in (0, 1):                       # one color range per row
        lo = min(np.nanmin(V[(r, c)]) for c in range(4))
        hi = max(np.nanmax(V[(r, c)]) for c in range(4))
        for c in range(4):
            ax = axes[r, c]; A = V[(r, c)]
            im = ax.imshow(A, origin="lower", aspect="auto",
                           cmap="viridis_r", vmin=lo, vmax=hi)
            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels([_sizelab(N) for N in sizes], fontsize=FS_TICK)
            ax.set_yticks(range(len(ns)))
            # only the leftmost panel of each row carries the y axis
            ax.set_yticklabels(ns if c == 0 else [], fontsize=FS_TICK)
            mid = 0.5 * (lo + hi)
            for ti in range(len(sizes)):
                for ni in range(len(ns)):
                    ax.text(ti, ni, fmt.format(A[ni, ti]), ha="center",
                            va="center", fontsize=FS_CELL,
                            color="white" if A[ni, ti] > mid else "black")
                if RULE[ti] in ns:
                    ax.add_patch(Rectangle((ti - .5, ns.index(RULE[ti]) - .5),
                                           1, 1, fill=False, edgecolor="red",
                                           lw=2.2))
            ax.set_title(f"({LET[4 * r + c]}) {COMBOS[c][2]}, {rows[r][1]}",
                         fontsize=FS_TITLE, loc="left", pad=7)
            if c == 0:
                ax.set_ylabel("$n$", fontsize=FS_AX)
            ax.set_xlabel("$N$", fontsize=FS_AX)
        cb = fig.colorbar(im, ax=list(axes[r, :]), fraction=0.022, pad=0.012)
        cb.ax.tick_params(labelsize=FS_TICK)
    fig.savefig(FIGS / fname, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {fname}")




def ranks14():
    r = np.load(HERE / "data" / "ranks14.npz", allow_pickle=True)
    N = r["N"].astype(float); mean = r["mean"]; sd = r["sd"]
    names = ["$r_{99\\%}$", "$r_{99.9\\%}$",
             "$r_{\\mathrm{ent}}$", "$r_{\\mathrm{stab}}$"]
    FS_AX, FS_TICK, FS_TITLE, FS_LEG = 15, 12, 14, 11
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for i, ax in enumerate(axes.ravel()):
        ax.errorbar(N, mean[:, i], yerr=sd[:, i], fmt="o", ms=7, capsize=3,
                    color="magenta", ecolor="magenta", elinewidth=1.1,
                    markerfacecolor="magenta", lw=1.2, zorder=3)
        s, b = np.polyfit(np.log(N), mean[:, i], 1)
        r2 = np.corrcoef(np.log(N), mean[:, i])[0, 1] ** 2
        xs = np.logspace(np.log10(N.min()), np.log10(N.max()), 50)
        sign = "+" if b >= 0 else "-"
        fit = f"${s:.2f}\\ln N {sign} {abs(b):.2f}$"
        r2s = f"($R^2 = {r2:.2f}$)"
        # Panel (d) only: the one-line label runs into the magenta error
        # bars, so the fit alone goes in the legend there and the R^2 is
        # drawn on its own line underneath (see after tight_layout below).
        lab = fit if i == 3 else f"{fit}  {r2s}"
        ax.plot(xs, s * np.log(xs) + b, "-", color="#08519c", lw=2,
                label=lab)
        ax.set_xscale("log")
        ax.set_xlabel("$N$", fontsize=FS_AX)
        ax.set_ylabel(names[i], fontsize=FS_AX)
        ax.set_title(f"({'abcd'[i]})", fontsize=FS_TITLE, loc="left", pad=6)
        ax.grid(alpha=0.3, lw=0.4); ax.tick_params(labelsize=FS_TICK)
        leg = ax.legend(fontsize=FS_LEG, frameon=False,
                        loc="upper left")
        if i == 3:
            leg_d, r2s_d = leg, r2s
    fig.tight_layout()
    # Panel (d)'s second legend line.  Placed after tight_layout, so that the
    # axes has stopped moving and the measured position still holds, and with
    # the left edge taken from the legend label so the two line up.
    fig.canvas.draw()
    ax = axes.ravel()[3]
    bb = (leg_d.get_texts()[0].get_window_extent()
          .transformed(ax.transAxes.inverted()))
    ax.text(bb.x0, bb.y0, r2s_d, transform=ax.transAxes, fontsize=FS_LEG,
            ha="left", va="top")
    fig.savefig(FIGS / "si_ranks14.pdf", bbox_inches="tight"); plt.close(fig)
    print("  wrote si_ranks14.pdf")


ALL = list(range(11))
heatmaps(M, 100.0, "n-RMSE (%)", "si_scale_nrmse.pdf", ALL)
heatmaps(R, 1.0, "$r(\\mathrm{S}\\to\\mathrm{T})$", "si_scale_ratio.pdf", ALL,
         "{:.3f}", note=", $11$ adjacency-coupled dynamics")
heatmaps(SI, 100.0, "RMSE / (range at $N=32$) (%)", "si_scale_absolute.pdf", ALL,
         note=", $11$ adjacency-coupled dynamics")
sub = ", ".join(DYN11[j] for j in SUB)
heatmaps(M, 100.0, "n-RMSE (%)", "si_scale_nrmse_sub.pdf", SUB,
         note=f", bounded-range dynamics ({sub})")
heatmaps(R, 1.0, "$r(\\mathrm{S}\\to\\mathrm{T})$", "si_scale_ratio_sub.pdf", SUB,
         "{:.3f}", note=f", bounded-range dynamics ({sub})")
ranks14()
