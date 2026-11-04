# =====================================================================
# SI Section S10: Robustness of the cross-transfer network structure
#
# This script: Fig. S20, the six cross-transfer networks
# Produces   : transfer_networks.pdf
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Fig. 6 equivalent: the cross-transfer network for all six panels.

The drawing style is that of Fig. 6 of the main text (see ../lib/fig6_style.py):
black circles, labels outside the circle, one double-headed arrow per
reciprocated pair and a single-headed arrow per one-directional edge.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import fig6_style as f6          # the Fig. 6 drawing style, shared

HERE = Path(__file__).resolve().parent
FIG = HERE / "figs"; FIG.mkdir(exist_ok=True)
PANELS = [("a", "NN, MinCorr", "nn_minc"), ("b", "NN, QR", "nn_qr"),
          ("c", "NN, random", "nn_random"),
          ("d", "linear, MinCorr", "lin_minc"),
          ("e", "linear, QR", "lin_qr"),
          ("f", "linear, random", "lin_random")]

pos = f6.layout(f6.ORDER)
fig, axes = plt.subplots(2, 3, figsize=(19.5, 14.0))
for ax, (lett, title, key) in zip(axes.ravel(), PANELS):
    G = f6.build_graph(key)
    f6.draw(G, pos, ax, node_size=330, label_fs=15, lw_bi=1.0, lw_uni=0.8,
            ms_bi=11, ms_uni=9, shrink=8, label_r=1.20)
    # The drawing only reaches |y| = 1.2 (labels sit at radius 1.20), so a
    # tighter vertical range brings the panel title down onto the graph.
    ax.set_xlim(-1.62, 1.62); ax.set_ylim(-1.34, 1.34)
    ax.set_title(f"({lett}) {title}", fontsize=24, loc="left", pad=4)
fig.tight_layout(h_pad=2.0, w_pad=1.0)
fig.savefig(FIG / "transfer_networks.pdf", bbox_inches="tight")
print("  wrote figs/transfer_networks.pdf")
