# =====================================================================
# SI Section S10: Robustness of the cross-transfer network structure
#
# This script: Fig. S21, the dendrograms
# Produces   : dendrograms.pdf
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Clustering of the continuous transfer errors, with no threshold on n-RMSE.

The partition into transfer classes rests on a thresholded graph.  Here the
n-RMSE is never thresholded.  The only binary ingredient is the one the
main text already uses to say that a transfer beats the baseline,
r(S->T) < 0.8, which decides whether an ordered pair carries an edge at all.

Each ordered pair that carries an edge is weighted by its n-RMSE, smaller
meaning a tighter link.  We symmetrize by averaging the two directions,
giving a pair with no edge in a direction the worst observed error in that
panel, and then cluster the resulting distance matrix two ways:

  * average-linkage hierarchical clustering, which needs no number of
    clusters and yields a dendrogram.
  * greedy modularity maximization on the graph that keeps only the pairs
    linked in at least one direction, weighted by s = exp(-d / median(d)),
    an entirely different objective that also chooses its own number of
    clusters.

Agreement between the two, and with the weakly connected components, is the
evidence that the partition is not an artifact of the threshold.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import dendro_style as ds

HERE = Path(__file__).resolve().parent
FIG = HERE / "figs"; FIG.mkdir(exist_ok=True)
d = np.load(HERE.parent / "shared_data" / "fig45_matrices.npz",
            allow_pickle=True)
LAB = [str(x) for x in d["dynamics"]]
n = len(LAB)
PANELS = [("a", "NN, MinCorr", "nn", "minc"), ("b", "NN, QR", "nn", "qr"),
          ("c", "NN, random", "nn", "random"),
          ("d", "linear, MinCorr", "lin", "minc"),
          ("e", "linear, QR", "lin", "qr"),
          ("f", "linear, random", "lin", "random")]
ADJ = {"SIS", "SIR", "MSIS1", "MSIS2", "MSIR1", "MSIR2", "DW", "MUT", "GEN",
       "WC", "gLV"}
RATIO_T = 0.8
REFERENCE = {"adjacency-11": sorted(ADJ), "EUT": ["EUT1", "EUT2"],
             "VEG": ["VEG1", "VEG2"], "NM": ["NM"], "LDW": ["LDW"]}


def distance_matrix(dec, sent):
    M = d[f"M_{dec}_{sent}"] * 100.0            # n-RMSE in percent
    B = d[f"B_{dec}_{sent}"] * 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(np.isfinite(B) & (B > 0), M / B, np.nan)
    has = (R < RATIO_T) & ~np.eye(n, dtype=bool)
    cap = np.nanmax(M[has]) if has.any() else np.nanmax(M)
    W = np.where(has, M, cap)                   # no edge -> worst observed
    np.fill_diagonal(W, 0.0)
    D = (W + W.T) / 2.0                         # symmetrize the two directions
    np.fill_diagonal(D, 0.0)
    return D, has


def partition_of(labels_arr):
    out = {}
    for lab, name in zip(labels_arr, LAB):
        out.setdefault(int(lab), []).append(name)
    return sorted((sorted(v, key=LAB.index) for v in out.values()),
                  key=lambda c: (-len(c), LAB.index(c[0])))


def same_as_reference(part):
    ref = sorted((sorted(v, key=LAB.index) for v in REFERENCE.values()),
                 key=lambda c: (-len(c), LAB.index(c[0])))
    return part == ref


def main():
    print("CLUSTERING ON CONTINUOUS n-RMSE (edge iff r < 0.8; n-RMSE never "
          "thresholded)\n")
    # Layout, fonts and font sizes follow Figs. 4 and 5 of the main text.
    TITLE_FS, LABEL_FS, TICK_FS, LEAF_FS = 30, 30, 14, 16
    fig, axes = plt.subplots(2, 3, figsize=(22.0, 13.0))
    for k, (lett, title, dec, sent) in enumerate(PANELS):
        D, has = distance_matrix(dec, sent)
        Z = linkage(squareform(D, checks=False), method="average")
        # cut at the number of weakly connected components of the same graph
        G = nx.DiGraph(); G.add_nodes_from(range(n))
        G.add_edges_from([(i, j) for i in range(n) for j in range(n)
                          if has[i, j]])
        n_wcc = nx.number_weakly_connected_components(G)
        hier = partition_of(fcluster(Z, t=n_wcc, criterion="maxclust"))
        # modularity on the similarity graph, no k required
        # only pairs that carry an edge in at least one direction, which is
        # the r < 0.8 condition; the weight is the continuous n-RMSE
        S = np.exp(-D / np.median(D[D > 0]))
        H = nx.Graph()
        H.add_nodes_from(LAB)
        for i in range(n):
            for j in range(i + 1, n):
                if has[i, j] or has[j, i]:
                    H.add_edge(LAB[i], LAB[j], weight=float(S[i, j]))
        comm = nx.community.greedy_modularity_communities(H, weight="weight")
        mod = sorted((sorted(c, key=LAB.index) for c in comm),
                     key=lambda c: (-len(c), LAB.index(c[0])))
        print(f"=== ({lett}) {title}")
        print(f"  hierarchical (k = {n_wcc}, the number of components):"
              f"  {'MATCHES the component partition' if same_as_reference(hier) else 'differs'}")
        for c in hier:
            print(f"      {len(c):2d}  {' '.join(c)}")
        print(f"  modularity (k chosen by the method = {len(mod)}):"
              f"  {'MATCHES' if same_as_reference(mod) else 'differs'}")
        for c in mod:
            print(f"      {len(c):2d}  {' '.join(c)}")
        print()
        ax = axes[k // 3, k % 3]
        R = dendrogram(Z, labels=LAB, ax=ax, leaf_font_size=LEAF_FS,
                       link_color_func=ds.link_color_func(Z, LAB, ADJ))
        ds.orient_left(ax, R["ivl"], ADJ)
        ax.set_title(f"({lett}) {title}", fontsize=TITLE_FS, loc="left",
                     pad=10)
        ax.set_ylabel("")
        ds.yticks_by_five(ax, Z)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.tick_params(axis="x", rotation=90, labelsize=LEAF_FS)
    fig.supylabel("average-linkage distance (n-RMSE, %)", fontsize=LABEL_FS,
                  x=0.004)
    fig.tight_layout(rect=[0.018, 0, 1, 1], h_pad=5.0)
    fig.savefig(FIG / "dendrograms.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote figs/dendrograms.pdf")


if __name__ == "__main__":
    main()
