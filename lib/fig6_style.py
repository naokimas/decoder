"""The cross-transfer network drawing style, shared with main-text Fig. 6.

SI Fig. S20 (s10_transfer_network/plot_net.py) draws the same kind of graph
for all six decoder x sentinel combinations, so the graph construction, the
circular layout and the arrow drawing live here rather than in the main-text
script.  The style follows the network figure of the first submission:
black circles, node labels
placed outside the circle, bidirectional pairs as one double-headed arrow and
one-directional edges as a single-headed arrow.  The only change is that the
circular layout now carries 17 rather than 16 dynamics, LDW being appended at
the end of the ordering.
"""
import os
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
d = np.load(HERE.parent / "shared_data" / "fig45_matrices.npz",
            allow_pickle=True)
LAB = [str(x) for x in d["dynamics"]]

# SIS sits at 12 o'clock and the dynamics run clockwise in this order.
# It coincides with the order of the dynamics in fig45_matrices.npz (LAB), so
# no reordering takes place; LDW is appended at the end.
ORDER = ["SIS", "SIR", "MSIS1", "MSIS2", "MSIR1", "MSIR2",
         "DW", "MUT", "GEN", "WC", "gLV",
         "NM", "EUT1", "EUT2", "VEG1", "VEG2", "LDW"]
assert sorted(ORDER) == sorted(LAB)

# Horizontal label nudges, in units of the label font size, for labels that
# would otherwise touch a node circle: the width of "M" (0.863 em in DejaVu
# Sans) to the right, the width of a digit (0.636 em) to the left.
LABEL_SHIFT = {"MSIS2": 0.863, "MSIR1": 0.863, "MSIR2": 0.863,
               "EUT1": -0.636, "EUT2": -0.636, "VEG1": -0.636}


def build_graph(key):
    M = d[f"M_{key}"]; B = d[f"B_{key}"]
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(np.isfinite(B) & (B > 0), M / B, np.nan)
    G = nx.DiGraph(); G.add_nodes_from(ORDER)
    for i, u in enumerate(LAB):
        for j, v in enumerate(LAB):
            if i != j and M[i, j] < 0.15 and R[i, j] < 0.8:
                G.add_edge(u, v)
    return G


def layout(order):
    pos = {}
    n = len(order)
    for k, name in enumerate(order):
        ang = np.pi / 2 - 2 * np.pi * k / n     # clockwise from 12 o'clock
        pos[name] = (np.cos(ang), np.sin(ang))
    return pos


def draw(G, pos, ax, node_size=620, label_fs=19, lw_bi=1.2, lw_uni=0.9,
         ms_bi=14, ms_uni=12, shrink=11, label_r=1.18):
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#000000",
                           node_size=node_size, edgecolors="black",
                           linewidths=1.0)
    edge_set = set(G.edges())
    bidir, uni = set(), []
    for (u, v) in G.edges():
        if (v, u) in edge_set:
            bidir.add(tuple(sorted([u, v])))
        else:
            uni.append((u, v))
    for u, v in sorted(bidir):
        (x1, y1), (x2, y2) = pos[u], pos[v]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="<|-|>",
                                     mutation_scale=ms_bi, color="#444444",
                                     lw=lw_bi, shrinkA=shrink, shrinkB=shrink))
    for u, v in uni:
        (x1, y1), (x2, y2) = pos[u], pos[v]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=ms_uni, color="#444444",
                                     lw=lw_uni, shrinkA=shrink - 1,
                                     shrinkB=shrink + 1))
    for name, (x, y) in pos.items():
        ax.annotate(name, (label_r * x, label_r * y),
                    xytext=(LABEL_SHIFT.get(name, 0.0) * label_fs, 0),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=label_fs, color="#000000")
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal"); ax.axis("off")
    return len(bidir), len(uni)
