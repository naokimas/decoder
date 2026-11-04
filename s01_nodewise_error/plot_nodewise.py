# =====================================================================
# SI Section S1: Node-wise reconstruction error
#
# This script: Figs. S1 and S2
# Produces   : figs/fig_nodewise_4panel.pdf, figs/fig_ownrange.pdf
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Figs. S1 and S2 of SI Section S1: node-wise reconstruction error.

Two four-panel figures, both laid out (a) (b) / (c) (d) over the same four
cases, so that they can be read against each other:

  fig_nodewise_4panel.pdf  node-wise n-RMSE against degree
  fig_ownrange.pdf         the same error divided by each node's own range

Conventions, following the SI captions: the legend appears in panel (a)
only, the Spearman rank correlation coefficient is printed inside each
plotting box near its top right corner, and the y label is shared by the
left column and drawn once, midway between the two left-hand panels.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
D = np.load(HERE / "data" / "nodewise.npz")

CASES = [
    ("dolphin_SIS_SIS",   "dolphin, SIS $\\to$ SIS"),
    ("dolphin_SIS_WC",    "dolphin, SIS $\\to$ WC"),
    ("email_MSIR1_MSIR1", "email, MSIR1 $\\to$ MSIR1"),
    ("email_MSIR1_GEN",   "email, MSIR1 $\\to$ GEN"),
]
LETTER = ["(a)", "(b)", "(c)", "(d)"]
SENT_FACE, SENT_EDGE = "#FFF59D", "#B8860B"
FS_LAB, FS_TICK, FS_LEG, FS_TITLE, FS_RHO = 17, 13, 12, 22.5, 14
YLAB_GAP = 0.0258          # half of the gap the shared y label used to have
TOP_FRAC = 0.855           # axes fraction at which the top symbol sits


def panel(ax, key, title, values, letter, ref=None):
    deg = D[f"{key}__degree"]; sent = D[f"{key}__sentinel"]
    mask = np.zeros(len(deg), dtype=bool); mask[sent] = True
    rho = spearmanr(deg, values).correlation
    ax.scatter(deg[~mask], values[~mask] * 100, s=22, facecolor="#4C72B0",
               edgecolor="none", alpha=0.55, label="non-sentinel node")
    ax.scatter(deg[mask], values[mask] * 100, s=85, facecolor=SENT_FACE,
               edgecolor=SENT_EDGE, linewidth=1.3, zorder=3,
               label="sentinel node")
    if ref is not None:
        # no legend entry: the dashed line is explained in the caption
        ax.axhline(ref * 100, ls="--", color="#444444", lw=1.2, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("degree, $k_i$", fontsize=FS_LAB)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.3, linewidth=0.4, which="both")
    ax.set_title(f"{letter} {title}", fontsize=FS_TITLE, pad=8,
                 loc="left")
    # rho inside the box, near the top right corner, no parentheses and no
    # plus sign; placed after the head room below so it clears the symbols.
    ax.text(0.965, 0.935, f"$\\rho = {rho:.2f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=FS_RHO)


MIN_GAP_PT = 6.0           # required clearance around the rho text / legend


def _min_gap_pt(fig, ax, renderer):
    """Smallest gap, in points, between the annotations and any symbol."""
    boxes = [txt.get_window_extent(renderer=renderer) for txt in ax.texts]
    leg = ax.get_legend()
    if leg is not None:
        boxes.append(leg.get_window_extent(renderer=renderer))
    if not boxes:
        return float("inf")
    gaps = []
    for coll in ax.collections:
        pts = ax.transData.transform(np.asarray(coll.get_offsets()))
        rad = np.sqrt(coll.get_sizes()[0]) / 2 * fig.dpi / 72 + 1
        for bb in boxes:
            dx = np.maximum(bb.x0 - rad - pts[:, 0], pts[:, 0] - bb.x1 - rad)
            dy = np.maximum(bb.y0 - rad - pts[:, 1], pts[:, 1] - bb.y1 - rad)
            gaps.append(np.maximum(dx, dy).min())
    return min(gaps) / fig.dpi * 72


def relieve_overlaps(fig, axes):
    """Open more head room, panel by panel, until nothing collides."""
    for ax, letter in zip(axes.ravel(), LETTER):
        for _ in range(40):
            fig.canvas.draw()
            gap = _min_gap_pt(fig, ax, fig.canvas.get_renderer())
            if gap >= MIN_GAP_PT:
                break
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi * 10 ** 0.04)
        print(f"    {letter} clearance {gap:5.1f} pt"
              f"{'  <-- STILL COLLIDING' if gap < MIN_GAP_PT else ''}")


def shared_ylabel(fig, axes, ylab):
    """Place one y label for the left column, close to its tick labels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    xs = [lab.get_window_extent(renderer=r).transformed(inv).x0
          for ax in (axes[0, 0], axes[1, 0]) for lab in ax.get_yticklabels()
          if lab.get_visible() and lab.get_text()]
    fig.text(min(xs) - YLAB_GAP, 0.5, ylab, rotation=90,
             ha="center", va="center", fontsize=FS_LAB)


def four_panel(out, values_of, ylab, with_ref):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2))
    for j, (ax, letter, (key, title)) in enumerate(
            zip(axes.ravel(), LETTER, CASES)):
        agg = float(D[f"{key}__agg"])
        panel(ax, key, title, values_of(key), letter,
              ref=agg if with_ref else None)
        lo, _ = ax.get_ylim()
        if not with_ref and j == 0:
            lo /= 10 ** 0.45      # (a) puts its legend at the bottom left
        # Put the highest plotted symbol at TOP_FRAC of the axes height, so
        # the rho text and the legend always sit in clear space.
        top = np.nanmax(values_of(key)) * 100
        ax.set_ylim(lo, 10 ** (np.log10(lo)
                               + (np.log10(top) - np.log10(lo)) / TOP_FRAC))
    axes[0, 0].legend(fontsize=FS_LEG, frameon=False,
                      loc="upper left" if with_ref else "lower left",
                      borderaxespad=0.5, handletextpad=0.5)
    fig.tight_layout(rect=(0.045, 0, 1, 1))
    relieve_overlaps(fig, axes)
    shared_ylabel(fig, axes, ylab)
    fig.savefig(FIGS / out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    four_panel("fig_nodewise_4panel.pdf",
               lambda k: D[f"{k}__nodewise"],
               "node-wise n-RMSE (%)", with_ref=True)

    def own(k):
        nw = D[f"{k}__nodewise"]; nr = D[f"{k}__node_range"]
        gr = float(D[f"{k}__global_range"])
        return nw * gr / np.where(nr > 0, nr, np.nan)
    four_panel("fig_ownrange.pdf", own,
               "node-wise RMSE / node's own range (%)", with_ref=False)


if __name__ == "__main__":
    main()
