# =====================================================================
# Main text Fig. 1: the conceptual overview
#
# This script: the schematic of network, sentinels and decoder
# Feeds      : Fig. 1
# Produces   : figs/concept_fig.png
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Fig. 1 of the main text: conceptual overview of the method.

The figure shows the network, the sentinel nodes, the training of the
decoder and its use on a different target dynamics.  Everything drawn is
computed from real data: the dolphin network, its MinCorr sentinels, and the
SIS (source) and WC (target) equilibrium sweeps that Fig. 3 also uses.

Reading order, left to right and then down:

  the network with its sentinel nodes, and the source equations
      -> the equilibria of every node across the control-parameter sweep,
         of which only the sentinel curves are observed
      -> the decoder, which maps the n sentinel readings to all N nodes
      -> deployed on a different target dynamics, where again only the
         sentinel curves are available, to recover all N node states
"""
import os, sys, importlib.util
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
LIB = HERE.parent / "lib"
sys.path.insert(0, str(HERE.parent / "lib")); import paths  # noqa: E402
sys.path.insert(0, str(LIB))
_s = importlib.util.spec_from_file_location("leq", LIB / "load_equilibria.py")
leq = importlib.util.module_from_spec(_s); _s.loader.exec_module(leq)
import pipeline_unified as pu                                    # noqa: E402
import splits                                                    # noqa: E402

# ---------------------------------------------------------------- the data
TAG = "dolphin"
G = nx.read_edgelist(leq.network_file(TAG), nodetype=int)
NODES = sorted(G.nodes())
N = len(NODES)
K = int(np.floor(np.log(N)))
YS = leq.load_eq(TAG, "SIS")          # source dynamics
YT = leq.load_eq(TAG, "WC")           # target dynamics
tr, _ = splits.rows_random()
SENT = list(pu.min_corr_sensors(YS[tr], K))

ORANGE = "#ef8a17"       # sentinel nodes and sentinel curves
GREEN = "#8fc78f"        # every other node
GRAY = "#8c8c8c"         # decoder internals
FS_T, FS_L, FS_S = 19, 16, 15
NODE_S = 170.0          # network node marker area, pt^2
DEC_S = 140.0           # decoder circle marker area, pt^2
FIG_W, FIG_H = 16.0, 9.0
ARROW = dict(arrowstyle="-|>", mutation_scale=32, lw=10.0, color="0.45",
             capstyle="butt", joinstyle="miter")

fig = plt.figure(figsize=(FIG_W, FIG_H))


def arrow(x0, y0, x1, y1, **kw):
    a = FancyArrowPatch((x0, y0), (x1, y1),
                        transform=fig.transFigure, **{**ARROW, **kw})
    fig.patches.append(a)


# ------------------------------------------------- network with sentinels
pos = nx.spring_layout(G, seed=7)
# A handful of nodes sit at the end of long near-vertical edges and set the
# height of the panel on their own.  Pull just those back towards the body;
# every other node keeps its position.  The panel box is then shorter than
# it was, which is what reduces the height of the drawing.
_ys = np.array([pos[v][1] for v in NODES])
_mid = float(np.median(_ys))
_order = np.argsort(_ys)
for v in ([NODES[i] for i in _order[:2]] + [NODES[i] for i in _order[-4:]]):
    x, y = pos[v]
    pos[v] = (x, _mid + 0.70 * (y - _mid))
NET_RECT = [0.005, 0.625, 0.245, 0.355]
ax = fig.add_axes(NET_RECT)
# the "sentinel" label is written to the left of the top-right sentinel;
# halve the edge of whichever green node sits under it
_tr = max(SENT, key=lambda i: pos[NODES[i]][0] + pos[NODES[i]][1])
_tx, _ty = pos[NODES[_tr]]
_cand = [v for v in NODES if NODES.index(v) not in SENT and pos[v][0] < _tx]
_near = min(_cand, key=lambda v: (pos[v][0] - (_tx - 0.22)) ** 2
            + (pos[v][1] - _ty) ** 2)
_nb = list(G.neighbors(_near))
if _nb:
    _anchor = min(_nb, key=lambda u: (pos[u][0] - pos[_near][0]) ** 2
                  + (pos[u][1] - pos[_near][1]) ** 2)
    pos[_near] = (0.5 * (pos[_near][0] + pos[_anchor][0]),
                  0.5 * (pos[_near][1] + pos[_anchor][1]))

# separate any nodes that touch: every pair at least 1.2 diameters apart,
# measured in points so that the unequal x and y scales are handled properly
_xs = np.array([pos[v][0] for v in NODES]); _ys = np.array([pos[v][1] for v in NODES])
_mx, _my = 0.06 * np.ptp(_xs), 0.10 * np.ptp(_ys)
_XL0 = (_xs.min() - _mx, _xs.max() + _mx)
_YL0 = (_ys.min() - _my, _ys.max() + _my)
_sx = NET_RECT[2] * FIG_W * 72.0 / (_XL0[1] - _XL0[0])
_sy = NET_RECT[3] * FIG_H * 72.0 / (_YL0[1] - _YL0[0])
_d = 2.0 * np.sqrt(NODE_S / np.pi)
_minsep = 1.2 * _d
P = np.array([[pos[v][0] * _sx, pos[v][1] * _sy] for v in NODES])
for _ in range(400):
    D = P[:, None, :] - P[None, :, :]
    dist = np.hypot(D[..., 0], D[..., 1])
    np.fill_diagonal(dist, np.inf)
    bad = dist < _minsep
    if not bad.any():
        break
    push = np.zeros_like(P)
    ii, jj = np.where(np.triu(bad))
    for a, b in zip(ii, jj):
        v = P[a] - P[b]
        r = np.hypot(*v) or 1e-9
        step = 0.55 * (_minsep - r) * v / r
        push[a] += step; push[b] -= step
    P += push
# The blob is tilted, so turning it by +10 degrees lowers its bounding box
# (height -7.9%, width unchanged).  P is already in points, so the box is
# then sized directly from the rotated content and the drawing scale is
# preserved in both directions.
_th = np.radians(10.0)
_R = np.array([[np.cos(_th), -np.sin(_th)], [np.sin(_th), np.cos(_th)]])
_c = P.mean(axis=0)
P = (P - _c) @ _R.T + _c
_m = 0.6 * _d
_XL = (P[:, 0].min() - _m, P[:, 0].max() + _m)
_YL = (P[:, 1].min() - _m, P[:, 1].max() + _m)
NET_RECT[2] = (_XL[1] - _XL[0]) / (FIG_W * 72.0)
NET_RECT[3] = (_YL[1] - _YL[0]) / (FIG_H * 72.0)
ax.set_position(NET_RECT)
pos = {v: (P[k, 0], P[k, 1]) for k, v in enumerate(NODES)}
ax.set_xlim(*_XL); ax.set_ylim(*_YL)

others = [NODES[i] for i in range(N) if i not in SENT]
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="0.82", width=1.0)
nx.draw_networkx_nodes(G, pos, nodelist=others, node_color=GREEN,
                       node_size=170, linewidths=0, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=[NODES[i] for i in SENT],
                       node_color=ORANGE, node_size=170, linewidths=0, ax=ax)
ax.axis("off")
top_right = max(SENT, key=lambda i: pos[NODES[i]][0] + pos[NODES[i]][1])
sx, sy = pos[NODES[top_right]]
ax.annotate("sentinel", xy=(sx, sy), xytext=(-12, 2),
            textcoords="offset points", color=ORANGE, fontsize=FS_L,
            ha="right", va="center")

# --------------------------------------------------- the source equations
fig.text(0.13453, 0.585,
         r"$\dot{x}_i = -\mu x_i + \beta\,(1-x_i)\sum_{j=1}^{N}A_{ij}x_j$",
         ha="center", va="center", fontsize=FS_T)

# ------------------------------------------------------ the sweep, source
ax = fig.add_axes([0.375, 0.630, 0.205, 0.235])
p = np.arange(YS.shape[0])
for i in range(N):
    if i not in SENT:
        ax.plot(p, YS[:, i], color=GREEN, lw=1.0, alpha=0.85, zorder=1)
for i in SENT:
    ax.plot(p, YS[:, i], color=ORANGE, lw=2.6, zorder=3)
ax.set_xlabel("control parameter", fontsize=FS_L)
ax.set_ylabel("equilibrium $x_i$", fontsize=FS_L)
ax.set_xticks([])
ax.tick_params(axis="y", labelsize=FS_S)
ax.tick_params(axis="x", length=0)

# ------------------------------------------------------------ the decoder
ax = fig.add_axes([0.65341, 0.545, 0.230, 0.420])
ax.set_xlim(1.2, 3.7); ax.set_ylim(0, 6.8); ax.axis("off")
ys_in = np.linspace(2.0, 4.4, K)
ys_h = np.linspace(1.2, 5.2, 7)
_dd = 2.0 * np.sqrt(DEC_S / np.pi)              # circle diameter, pt
_pt_per_unit = 0.420 * FIG_H * 72.0 / 6.8       # decoder axes: h and ylim
_gap = 1.4 * _dd / _pt_per_unit                 # centers, with a 0.4d gap
ys_out = 3.2 + (np.arange(11) - 5) * _gap
for y in ys_in:
    for yh in ys_h:
        ax.plot([1.7, 2.45], [y, yh], color="0.88", lw=0.7, zorder=1)
for yh in ys_h:
    for yo in ys_out:
        ax.plot([2.45, 3.2], [yh, yo], color="0.88", lw=0.7, zorder=1)
ax.scatter(np.full(K, 1.7), ys_in, s=140, c=ORANGE, zorder=3)
ax.scatter(np.full(7, 2.45), ys_h, s=140, c=GRAY, zorder=3)
ax.scatter(np.full(11, 3.2), ys_out, s=140, c=GREEN, zorder=3)
ax.text(1.7, 4.75, "$n$ sentinel\nreadings", ha="center", va="bottom",
        fontsize=FS_S, color=ORANGE)
ax.text(3.1846, ys_out.max() + 0.30, "all $N$ node\nstates",
        ha="center", va="bottom", fontsize=FS_S,
        color="#5a9a5a")

# -------------------------------------------------- cross-dynamics transfer
fig.text(0.52888, 0.48236, "cross-dynamics\ntransfer", fontsize=FS_T,
         ha="center", va="center")


def target_panel(rect, with_green):
    a = fig.add_axes(rect)
    pp = np.arange(YT.shape[0])
    if with_green:
        for i in range(N):
            if i not in SENT:
                a.plot(pp, YT[:, i], color=GREEN, lw=1.0, alpha=0.85,
                       zorder=1)
    for i in SENT:
        a.plot(pp, YT[:, i], color=ORANGE, lw=2.6, zorder=3)
    a.set_xlabel("control parameter", fontsize=FS_L)
    a.set_ylabel("equilibrium $x_i$", fontsize=FS_L)
    a.set_xticks([])
    a.set_ylim(-0.05 * YT.max(), 1.05 * YT.max())
    a.tick_params(axis="y", labelsize=FS_S)
    a.tick_params(axis="x", length=0)
    a.patch.set_alpha(0.0)
    return a


target_panel([0.188, 0.21036, 0.205, 0.235], with_green=False)
target_panel([0.660, 0.21036, 0.205, 0.235], with_green=True)
fig.text(0.2905, 0.45736, "sentinel readings from\nthe target dynamics",
         ha="center", va="bottom", fontsize=FS_S, color=ORANGE)
fig.text(0.7625, 0.45736, "reconstructed equilibria\nof the target dynamics",
         ha="center", va="bottom", fontsize=FS_S, color="#5a9a5a")

# ---------------------------------------------------------------- arrows
arrow(0.258, 0.748, 0.322, 0.748)          # network -> sweep
arrow(0.258, 0.605, 0.318, 0.652)          # equations -> sweep
arrow(0.59806, 0.748, 0.67906, 0.748)          # sweep -> decoder
arrow(0.68925, 0.56874, 0.512, 0.35336)          # decoder -> the transfer arrow
arrow(0.411, 0.32536, 0.60966, 0.32536)          # target sentinels -> reconstruction
fig.text(0.63459, 0.71029, "train\ndecoder", fontsize=FS_T, ha="center",
         va="top", color="black")

fig.savefig(FIGS / "concept_fig.png", dpi=200, bbox_inches="tight")
print("  wrote figs/concept_fig.png")
