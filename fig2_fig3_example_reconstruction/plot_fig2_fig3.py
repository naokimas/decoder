# =====================================================================
# Main text Figs. 2 and 3: reconstruction on the dolphin network
#
# This script: Figs. 2 and 3, ten representative nodes each
# Feeds      : Figs. 2 and 3
# Produces   : figs/sis_sis_dolphin_10node.png, figs/sis_wc_dolphin_10node.png
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Figs. 2 and 3 of the main text: reconstruction at 10 representative nodes.

Each figure shows 10 nodes of the dolphin network in a 2 x 5 grid:

  columns 1-2 : the 4 MinCorr sentinel nodes, in ascending node ID,
                filled down each column and then across
  column  3   : the 2 nodes with the smallest node-wise n-RMSE
  column  4   : the 2 nodes closest to the median node-wise n-RMSE
  column  5   : the 2 nodes with the largest node-wise n-RMSE

Columns 3-5 are selected among NON-sentinel nodes.  Without that
restriction the SIS -> WC figure would repeat two sentinel panels, because
there the two best-reconstructed nodes of the network are sentinels.

For Fig. 3, which shows two decoders, the ranking uses the SIS decoder, the
one drawn as "SIS decoder" in the legend.

Node-wise n-RMSE is the definition of SI Section S1: the RMSE over the 10
evaluation control-parameter values, normalized by the target's global range.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
SENT_BG = "#FFF59D"
FS_TICK, FS_TITLE, FS_LEG, FS_LAB = 12, 12, 18, 20



def trim(path):
    """Crop the uniform white border matplotlib leaves around the canvas.

    bbox_inches="tight" still keeps a hairline margin, and the figures as
    they appear in the manuscript have it removed.  Doing it here keeps the
    files this script writes identical to the ones the manuscript includes.
    """
    from PIL import Image, ImageChops
    im = Image.open(path).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if box and box != (0, 0, *im.size):
        im.crop(box).save(path)

def node_nrmse(Y_pred, Y_true, ev):
    rng_ = float(Y_true.max() - Y_true.min())
    return np.sqrt(((Y_pred[ev] - Y_true[ev]) ** 2).mean(axis=0)) / rng_


def pick(v, sent):
    """The 10 nodes to show, as (node, role) in panel order."""
    sent = sorted(int(s) for s in sent)
    free = np.array([i for i in range(len(v)) if i not in sent])
    by_err = free[np.argsort(v[free])]
    med = np.median(v)                       # median over ALL nodes
    by_med = free[np.argsort(np.abs(v[free] - med))]
    return ([(sent[0], "sentinel"), (sent[1], "sentinel"),
             (sent[2], "sentinel"), (sent[3], "sentinel")]
            + [(int(by_err[0]), "smallest"), (int(by_err[1]), "smallest")]
            + [(int(by_med[0]), "median"), (int(by_med[1]), "median")]
            + [(int(by_err[-1]), "largest"), (int(by_err[-2]), "largest")])


def _shift_legend(fig, ax, leg, n_e, n_t=0.0):
    """Shift the legend by n_e widths of 'e' and n_t heights of 't'."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    fp = FontProperties(size=FS_LEG)
    w_pt = TextPath((0, 0), "e", prop=fp).get_extents().width
    h_pt = TextPath((0, 0), "t", prop=fp).get_extents().height
    fig.canvas.draw()
    bb = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    dx = n_e * w_pt * fig.dpi / 72.0 / bb.width            # axes fraction
    dy = n_t * h_pt * fig.dpi / 72.0 / bb.height
    leg.set_bbox_to_anchor((dx, dy, 1.0, 1.0), transform=ax.transAxes)
    print(f"      legend shifted by {n_e} x width('e') = {n_e*w_pt:.2f} pt "
          f"({dx:+.4f} of panel width)"
          + (f", and {n_t} x height('t') = {n_t*h_pt:.2f} pt "
             f"({dy:+.4f} of panel height)" if n_t else ""))


def _tighten_xlabel(fig, axes, sup):
    """Halve the gap between the shared x label and the tick labels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    ys = [lab.get_window_extent(renderer=r).transformed(inv).y0
          for ax in axes[-1, :] for lab in ax.get_xticklabels()
          if lab.get_visible() and lab.get_text()]
    tick_y0 = min(ys)
    lab_y1 = sup.get_window_extent(renderer=r).transformed(inv).y1
    gap = tick_y0 - lab_y1
    sup.set_y(sup.get_position()[1] + gap / 2)
    fig.canvas.draw()
    new = tick_y0 - sup.get_window_extent(
        renderer=fig.canvas.get_renderer()).transformed(inv).y1
    print(f"      x label gap: {gap:.4f} -> {new:.4f} "
          f"(ratio {new/gap:.2f})")


def draw(out_png, ps, Y_true, curves, sent, v, xlabel, xlim,
         leg_loc="upper left", leg_shift=0.0, leg_up=0.0):
    """curves: list of (Y_pred, style kwargs, label); the first ranks nodes."""
    chosen = pick(v, sent)
    fig, axes = plt.subplots(2, 5, figsize=(19.0, 7.4),
                             sharex=True, sharey=True)
    ylim = (float(Y_true.min()), float(Y_true.max()))
    for k, (node, role) in enumerate(chosen):
        ax = axes[k % 2, k // 2]             # fill down each column
        if role == "sentinel":
            ax.set_facecolor(SENT_BG)
        ax.plot(ps, Y_true[:, node], "-", color="black", lw=1.3, label="true")
        for Yp, kw, lab in curves:
            ax.plot(ps, Yp[:, node], label=lab, **kw)
        ax.set_ylim(ylim); ax.set_xlim(*xlim)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.3, linewidth=0.4)
        if k // 2 == 0:
            ax.set_ylabel(r"$x_i$", fontsize=FS_LAB)
    leg = axes[0, 0].legend(fontsize=FS_LEG, frameon=False, loc=leg_loc,
                            borderaxespad=1.1, handlelength=1.6,
                            handletextpad=0.5)
    if leg_shift or leg_up:
        _shift_legend(fig, axes[0, 0], leg, leg_shift, leg_up)
    sup = fig.supxlabel(xlabel, fontsize=FS_LAB, y=0.01)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _tighten_xlabel(fig, axes, sup)
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _lb = leg.get_window_extent(renderer=_r)
    _ab = axes[0, 0].get_window_extent(renderer=_r)
    print(f"      legend spans x in "
          f"[{(_lb.x0-_ab.x0)/_ab.width:.3f}, {(_lb.x1-_ab.x0)/_ab.width:.3f}]"
          f" of the panel width")
    fig.savefig(FIGS / out_png, dpi=200, bbox_inches="tight")
    trim(FIGS / out_png)
    plt.close(fig)
    print(f"  wrote {out_png}")
    for node, role in chosen:
        print(f"      node {node:3d}  {role:9s} n-RMSE_i = {v[node]*100:6.3f}%")


def main():
    # ---- Fig. 2: dolphin, SIS -> SIS --------------------------------
    d = np.load(HERE / "data" / "fig2_dolphin_sis_sis.npz")
    ev, sent = d["eval_rows"], d["sentinel"]
    v = node_nrmse(d["Y_pred"], d["Y_true"], ev)
    draw("sis_sis_dolphin_10node.png", d["ps"], d["Y_true"],
         [(d["Y_pred"], dict(ls="--", color="red", lw=1.3), "SIS decoder")],
         sent, v, r"infection rate, $\beta$", (0.0, 1.0),
         leg_loc="lower right", leg_shift=+1.0)

    # ---- Fig. 3: dolphin, SIS -> WC ---------------------------------
    d = np.load(HERE / "data" / "fig3_dolphin_sis_wc.npz")
    ev, sent = d["eval_rows"], d["sis_sentinel"]
    v = node_nrmse(d["Y_pred_sis_wc"], d["Y_true"], ev)   # ranks on SIS
    draw("sis_wc_dolphin_10node.png", d["ps"], d["Y_true"],
         [(d["Y_pred_sis_wc"], dict(ls="--", color="red", lw=1.3),
           "SIS decoder"),
          (d["Y_pred_coo_wc"], dict(ls=":", color="#1f77b4", lw=1.6),
           "C11L1")],
         sent, v, r"coupling strength, $D$", (0.0, 6.0),
         leg_loc="upper left", leg_shift=-1.0, leg_up=+1.0)


if __name__ == "__main__":
    main()
