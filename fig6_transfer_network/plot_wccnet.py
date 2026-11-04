# =====================================================================
# Main text Fig. 6: the cross-transfer network
#
# This script: the single-panel graph, NN decoder with MinCorr sentinels
# Feeds      : Fig. 6
# Produces   : figs/fig_wccnet_rev.png
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Main-text Fig. 6: the cross-transfer network over the 17 dynamics.

An edge S -> T means that the decoder trained on S reconstructs T well
enough to pass both thresholds, n-RMSE < 0.15 and r(S -> T) < 0.8.  The
11 adjacency-coupled dynamics form the large mutually connected core; the
six Laplacian-coupled ones sit outside it.

The graph construction, the circular layout and the arrow drawing are in
../lib/fig6_style.py, because SI Fig. S20 draws the same picture for all
six decoder x sentinel combinations and must look identical.  This file is
just the driver for the one panel the main text shows.

Reads ../shared_data/fig45_matrices.npz, which compute_matrices.py in
fig4_fig5_cross_transfer/ writes.  Takes a second.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parent / "lib"))
import fig6_style as f6                                  # noqa: E402

G = f6.build_graph("nn_minc")
pos = f6.layout(f6.ORDER)
fig, ax = plt.subplots(figsize=(8.6, 8.6))
n_bi, n_uni = f6.draw(G, pos, ax)
fig.tight_layout()
fig.savefig(FIGS / "fig_wccnet_rev.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"  wrote figs/fig_wccnet_rev.png ({G.number_of_edges()} edges, "
      f"{n_bi} bidirectional pairs, {n_uni} one-directional)")
