# =====================================================================
# SI Section S11: Cross-transfer classes with four additional dynamics
#
# This script: Fig. S22 and Table S14
# Produces   : figs/dendrograms21.pdf, tables/tab_21.tex
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""The same component and clustering analysis over all 21 dynamics.

SI Section S11 defines four further dynamics, each of which swaps the coupling
operator of one of the existing ones: ANM, AEUT1 and AVEG1 couple NM, EUT1
and VEG1 through the adjacency matrix, and LMUT couples MUT through the graph
Laplacian.  Together with LDW, which does the same to DW, they give five
matched pairs that differ only in the coupling.  Adding them makes 21
dynamics, 14 adjacency coupled and 7 Laplacian coupled.
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
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                        / "s10_transfer_network"))
from analyze import reciprocity          # shared with Section S10

HERE = Path(__file__).resolve().parent
EX = HERE.parent / "s04_interp_vs_extrap" / "data"
FIG = HERE / "figs"; FIG.mkdir(exist_ok=True)
REN = {"MSIS": "MSIS1", "MSIR": "MSIR1", "EUT": "EUT1", "VEG": "VEG1"}
PANELS = [("a", "NN, MinCorr", "nn", "minc"), ("b", "NN, QR", "nn", "qr"),
          ("c", "NN, random", "nn", "random"),
          ("d", "linear, MinCorr", "lin", "minc"),
          ("e", "linear, QR", "lin", "qr"),
          ("f", "linear, random", "lin", "random")]
# adjacency coupled: the original 11 plus the three new A-coupled variants
ADJ = {"SIS", "SIR", "MSIS1", "MSIS2", "MSIR1", "MSIR2", "DW", "MUT", "GEN",
       "WC", "gLV", "ANM", "AEUT1", "AVEG1"}
PAIRS = [("DW", "LDW"), ("MUT", "LMUT"), ("ANM", "NM"),
         ("AEUT1", "EUT1"), ("AVEG1", "VEG1")]
NRMSE_T, RATIO_T = 0.15, 0.8

_r = np.load(HERE / "data" / "rand21_matrices.npz", allow_pickle=True)
LAB = [REN.get(str(x), str(x)) for x in _r["dynamics"]]
n = len(LAB)


def matrices(dec, sent):
    if sent == "random":
        M, B = _r[f"M_{dec}_random"], _r[f"B_{dec}_random"]
    else:
        M = np.nanmean(np.load(EX / f"grid9_random_random_{dec}_{sent}.npz",
                               allow_pickle=True)["rel"], axis=0)
        B = np.nanmean(np.load(EX / f"bgrid_random_random_{sent}.npz",
                               allow_pickle=True)["rel"], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(np.isfinite(B) & (B > 0), M / B, np.nan)
    return M, R


def build(M, R):
    G = nx.DiGraph(); G.add_nodes_from(LAB)
    for i in range(n):
        for j in range(n):
            if i != j and M[i, j] < NRMSE_T and R[i, j] < RATIO_T:
                G.add_edge(LAB[i], LAB[j])
    return G


def dist(M, R):
    has = (R < RATIO_T) & ~np.eye(n, dtype=bool)
    cap = np.nanmax(M[has] * 100) if has.any() else np.nanmax(M * 100)
    W = np.where(has, M * 100, cap); np.fill_diagonal(W, 0.0)
    D = (W + W.T) / 2.0; np.fill_diagonal(D, 0.0)
    return D


print(f"  {n} dynamics: {len(ADJ)} adjacency coupled, {n-len(ADJ)} Laplacian "
      f"coupled\n")
# Layout, fonts and font sizes follow Fig. S11.
TITLE_FS, LABEL_FS, TICK_FS, LEAF_FS = 30, 30, 14, 16
TAB = HERE / "tables"; TAB.mkdir(exist_ok=True)
trows = []          # Table S14, filled in as the panels are built


def tex(x, fmt):
    """LaTeX cell; a negative number goes in math mode for the proper minus."""
    s = f"{x:{fmt}}"
    return f"${s}$" if x < 0 else s
fig, axes = plt.subplots(2, 3, figsize=(22.0, 13.0))
for k, (lett, title, dec, sent) in enumerate(PANELS):
    M, R = matrices(dec, sent)
    G = build(M, R)
    comps = sorted(nx.weakly_connected_components(G),
                   key=lambda c: (-len(c), sorted(c)))
    big = comps[0]
    H = G.subgraph(big)
    dens = H.number_of_edges() / (len(big) * (len(big) - 1)) if len(big) > 1 else float("nan")
    rec = reciprocity(H)    # Garlaschelli and Loffredo (2004)
    nscc = nx.number_strongly_connected_components(H)
    trows.append(
        f"{title} & {G.number_of_edges() / (n * (n - 1)):.3f} & "
        f"{tex(reciprocity(G), '.3f')} & {len(big)} & {dens:.3f} & "
        f"{tex(rec, '.3f')} & {nscc} & "
        f"{max(len(c) for c in nx.strongly_connected_components(H))}"
        + r"\\")
    print(f"=== ({lett}) {title}: {G.number_of_edges()} edges, "
          f"{len(comps)} components")
    print(f"    largest: {len(big)} nodes, density {dens:.3f}, "
          f"reciprocity {rec:.3f}, {nscc} SCC(s)")
    print(f"      = all 14 adjacency? {big == ADJ}"
          f"   adjacency in it: {len(big & ADJ)}/14"
          f"   Laplacian in it: {len(big - ADJ)}")
    print(f"      members: {' '.join(sorted(big, key=LAB.index))}")
    for c in comps[1:]:
        print(f"    other: {' '.join(sorted(c, key=LAB.index))}")
    print("    matched pairs (A-coupled / L-coupled, same local dynamics):")
    for a, b in PAIRS:
        ca = next(i for i, c in enumerate(comps) if a in c)
        cb = next(i for i, c in enumerate(comps) if b in c)
        print(f"      {a:6s} in comp {ca}   {b:6s} in comp {cb}"
              f"   {'SAME' if ca == cb else 'separated'}")
    D = dist(M, R)
    Z = linkage(squareform(D, checks=False), method="average")
    hier = {}
    for lab, name in zip(fcluster(Z, t=len(comps), criterion="maxclust"), LAB):
        hier.setdefault(int(lab), []).append(name)
    hp = sorted((sorted(v, key=LAB.index) for v in hier.values()),
                key=lambda c: (-len(c), LAB.index(c[0])))
    cp = sorted((sorted(c, key=LAB.index) for c in comps),
                key=lambda c: (-len(c), LAB.index(c[0])))
    print(f"    hierarchical clustering (k = {len(comps)}): "
          f"{'MATCHES the components' if hp == cp else 'differs'}")
    if hp != cp:
        for c in hp:
            print(f"        {len(c):2d}  {' '.join(c)}")
    print()
    ax = axes[k // 3, k % 3]
    Zd = ds.group_first(Z, LAB, ADJ)     # green subtree to the left
    dendrogram(Zd, labels=LAB, ax=ax, leaf_font_size=LEAF_FS,
               link_color_func=ds.link_color_func(Zd, LAB, ADJ))
    ax.set_title(f"({lett}) {title}", fontsize=TITLE_FS, loc="left", pad=10)
    ax.set_ylabel("")
    ds.yticks_by_five(ax, Z)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.tick_params(axis="x", rotation=90, labelsize=LEAF_FS)
fig.supylabel("average-linkage distance (n-RMSE, %)", fontsize=LABEL_FS,
              x=0.004)
fig.tight_layout(rect=[0.018, 0, 1, 1], h_pad=5.0)
fig.savefig(FIG / "dendrograms21.pdf", bbox_inches="tight")
print("  wrote figs/dendrograms21.pdf")

# Table S14: the same seven structural columns as Table S11, now for the 21
# dynamics.  The rows were filled in above, one per decoder x sentinel panel.
(TAB / "tab_21.tex").write_text("\n".join(trows) + "\n")
print("  wrote tables/tab_21.tex")
for _r in trows:
    print("   ", _r)
