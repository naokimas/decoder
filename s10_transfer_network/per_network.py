# =====================================================================
# SI Section S10: Robustness of the cross-transfer network structure
#
# This script: Table S13, the component structure network by network
# Produces   : tables/tab_pernet.tex
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""The same component analysis, network by network rather than on the average.

An edge defined from an error averaged over the 14 networks need not
represent transfer that works on individual networks.  Here the edge rule is
applied separately to each network, using that network's own n-RMSE and
r(S->T), and we ask the one question that matters for the paper's claim:
does the largest weakly connected component still consist of the 11
adjacency-coupled dynamics?
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
d = np.load(HERE.parent / "s07_variability" / "data" / "rep_base.npz", allow_pickle=True)
LAB = [str(x) for x in d["dynamics"]]
NETS = [str(x) for x in d["networks"]]
n = len(LAB)
PANELS = [("NN, MinCorr", "nn", "minc"), ("NN, QR", "nn", "qr"),
          ("NN, random", "nn", "random"), ("linear, MinCorr", "lin", "minc"),
          ("linear, QR", "lin", "qr"), ("linear, random", "lin", "random")]
ADJ = ["SIS", "SIR", "MSIS1", "MSIS2", "MSIR1", "MSIR2", "DW", "MUT", "GEN",
       "WC", "gLV"]
ADJS = set(ADJ)
NRMSE_T, RATIO_T = 0.15, 0.8


def graph_for(dec, sent, ni):
    M = d[f"M_{dec}_{sent}"][0, ni]
    B = d[f"B_{dec}_{sent}"][0, ni]
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(np.isfinite(B) & (B > 0), M / B, np.nan)
    G = nx.DiGraph(); G.add_nodes_from(LAB)
    for i in range(n):
        for j in range(n):
            if i != j and M[i, j] < NRMSE_T and R[i, j] < RATIO_T:
                G.add_edge(LAB[i], LAB[j])
    return G


# Short display names, as the rows of Table S13 are labeled.
SHORT = {"dolphin": "Dolphin", "prox": "Proximity", "metabolic": "Metabolic",
         "gkk": "GKK", "lfr": "LFR", "er": "ER", "ba": "BA", "hk": "HK",
         "euroroad": "Euroroad", "email": "Email", "flybi": "Flybi",
         "reactome": "Reactome", "route_views": "Route views",
         "spanish": "Spanish"}

rows = []
print(f"  edge rule: n-RMSE < {NRMSE_T} and r < {RATIO_T}, applied to each "
      f"network's own values\n")
hdr = f"  {'network':<12s}" + "".join(f"{p[0]:>18s}" for p in PANELS)
print(hdr); print("  " + "-" * (len(hdr) - 2))
for ni, tag in enumerate(NETS):
    cells = []
    for title, dec, sent in PANELS:
        G = graph_for(dec, sent, ni)
        big = max(nx.weakly_connected_components(G), key=len)
        n_adj_in = len(big & ADJS)
        exact = (big == ADJS)
        cells.append((len(big), n_adj_in, exact))
    rows.append((tag, cells))
    print(f"  {tag:<12s}" + "".join(
        f"{('%d' % c[0]) + ('=11 adj' if c[2] else ' (%d adj)' % c[1]):>18s}"
        for c in cells))

# Table S13: one row per network, one column per decoder x sentinel
# combination, each cell the size of that network's own LWCC.
OUT = HERE / "tables"; OUT.mkdir(exist_ok=True)
body = [f"{SHORT[tag]} & " + " & ".join(str(c[0]) for c in cells) + r" \\"
        for tag, cells in rows]
(OUT / "tab_pernet.tex").write_text("\n".join(body) + "\n")
print("\n  wrote tables/tab_pernet.tex")

print()
for k, (title, dec, sent) in enumerate(PANELS):
    ex = sum(1 for _, cells in rows if cells[k][2])
    sizes = [cells[k][0] for _, cells in rows]
    adjin = [cells[k][1] for _, cells in rows]
    print(f"  {title:<16s} largest component is exactly the 11 adjacency "
          f"dynamics on {ex}/14 networks; "
          f"size {min(sizes)}-{max(sizes)}, "
          f"contains {min(adjin)}-{max(adjin)} of the 11")

# how often does the largest component contain a diffusive dynamics?
print()
for k, (title, dec, sent) in enumerate(PANELS):
    bad = []
    for ni, tag in enumerate(NETS):
        G = graph_for(dec, sent, ni)
        big = max(nx.weakly_connected_components(G), key=len)
        intr = sorted(big - ADJS, key=LAB.index)
        miss = sorted(ADJS - big, key=LAB.index)
        if intr or miss:
            bad.append((tag, intr, miss))
    if bad:
        print(f"  {title}:")
        for tag, intr, miss in bad:
            s = []
            if intr:
                s.append("extra " + ",".join(intr))
            if miss:
                s.append("missing " + ",".join(miss))
            print(f"      {tag:<12s} {'; '.join(s)}")
