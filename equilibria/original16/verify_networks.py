"""Verify the 20 undirected unweighted networks listed in supplement-final.tex S1.

Loads each .rds file from networks-from-Neil/, extracts the largest connected
component, and prints N, M, <k> for sanity check against expected values.
Also writes <tag>_network.edgelist for each (used by generate_data.py).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import rdata

HERE = Path(__file__).resolve().parent
RDS_DIR = HERE / "networks-from-Neil"

# (tag, rds-file, expected N from supplement S1).
# tags chosen to match the names in the supplement (lowercase, underscored).
NETWORKS = [
    # small (N < 2000)
    ("dolphin",      "dolphin.rds",      62),
    ("prox",         "proximity.rds",    410),
    ("metabolic",    "celegans.rds",     453),
    ("euroroad",     "euroroad.rds",     1039),
    ("email",        "email_uni.rds",    1133),
    ("er",           "er.rds",           1000),
    ("ba",           "ba.rds",           1000),
    ("hk",           "hk.rds",           1000),
    ("gkk",          "gkk.rds",          949),
    ("lfr",          "lfr.rds",          998),
    # medium (2k - 25k)
    ("flybi",        "drosophila.rds",   2705),
    ("reactome",     "reactome.rds",     5973),
    ("route_views",  "route_views.rds",  6474),
    ("spanish",      "spanish.rds",      11558),
    ("foldoc",       "foldoc.rds",       13356),
    ("tree_of_life", "tree_of_life.rds", 16415),
    ("english",      "word_assoc.rds",   23132),
    # large (>25k)
    ("enron",        "enron.rds",        33696),
    ("marker_cafe",  "marker_cafe.rds",  69317),
    ("prosper",      "prosper.rds",      89171),
]


def load_rds_to_lcc(fname: str) -> nx.Graph:
    """Read .rds, build undirected graph (drop self-loops + duplicates), keep LCC,
    relabel 0..N-1 by sorted order."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = rdata.read_rds(RDS_DIR / fname)
    n_nodes_raw = int(parsed[0][0])
    to_arr = parsed[2].astype(int)
    from_arr = parsed[3].astype(int)
    edges = set()
    for u, v in zip(from_arr.tolist(), to_arr.tolist()):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))
    G = nx.Graph(); G.add_nodes_from(range(n_nodes_raw)); G.add_edges_from(edges)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering="sorted")
    return G


def write_edgelist(G: nx.Graph, tag: str) -> Path:
    out = HERE / f"{tag}_network.edgelist"
    nx.write_edgelist(G, out, data=False)
    return out


def main():
    print(f"Verifying {len(NETWORKS)} networks against supplement S1.\n")
    print(f"{'tag':>14s}  {'rds file':>22s}  {'N_exp':>6s}  "
          f"{'N_LCC':>6s}  {'M':>9s}  {'<k>':>6s}  {'max k':>6s}  {'K=floor(ln N)':>14s}  match?")
    print("-" * 110)
    summary = []
    for tag, fname, n_exp in NETWORKS:
        try:
            G = load_rds_to_lcc(fname)
            N = G.number_of_nodes()
            M = G.number_of_edges()
            kbar = 2*M / N if N > 0 else 0
            max_k = max(d for _, d in G.degree()) if N > 0 else 0
            K = int(np.floor(np.log(N))) if N > 0 else 0
            ok = "YES" if N == n_exp else "(diff)"
            print(f"{tag:>14s}  {fname:>22s}  {n_exp:>6d}  "
                  f"{N:>6d}  {M:>9d}  {kbar:>6.2f}  {max_k:>6d}  {K:>14d}  {ok}")
            write_edgelist(G, tag)
            summary.append((tag, fname, n_exp, N, M, kbar, max_k, K))
        except Exception as e:
            print(f"{tag:>14s}  {fname:>22s}  {n_exp:>6d}  ERROR: {e}")
            summary.append((tag, fname, n_exp, None, None, None, None, None))

    # Save the table to disk for the paper
    out = HERE / "network_verification.csv"
    with open(out, "w") as f:
        f.write("tag,rds_file,N_expected,N_LCC,M,mean_k,max_k,K_floor_lnN\n")
        for row in summary:
            f.write(",".join(str(x if x is not None else "") for x in row) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
