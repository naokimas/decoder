# =====================================================================
# SI Section S17: Networks
#
# This script: Table S24, the statistics of the 14 networks
# Produces   : tables/tab_networks.tex
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Regenerate the network-statistics table of SI Section S17.

Every number in the table is read off the edgelists themselves, so running
this script is also a check that the table matches the networks actually
used.  The display name and the citation key of each network cannot be
derived from the data and are listed here in the order in which the table
presents them.

Run with --check to compare the regenerated numbers against the values
currently printed in the SI instead of writing the table body.
"""
import sys
from pathlib import Path
import networkx as nx

HERE = Path(__file__).resolve().parent
OUT = HERE / "tables"; OUT.mkdir(exist_ok=True)

# (edgelist tag, display name, citation key), in the order of the SI table
NETS = [
 ("dolphin",     r"Dolphin social network",                       "Lusseau2003BES"),
 ("prox",        r"Human face-to-face proximity",                 "Isella2011JTheorBiol"),
 ("metabolic",   r"\emph{C.\,elegans} metabolic network",         "Jeong2000Nature"),
 ("gkk",         r"Goh--Kahng--Kim model",                        "Goh2001PRL"),
 ("lfr",         r"Lancichinetti--Fortunato--Radicchi benchmark", "Lancichinetti2008PRE"),
 ("er",          r"Erd\H{o}s--R\'enyi random graph",              "Erdos1960Publ"),
 ("ba",          r"Barab\'asi--Albert model",                     "BarabasiAlbert1999Science"),
 ("hk",          r"Holme--Kim model",                             "HolmeKim2002PRE"),
 ("euroroad",    r"European road network",                        "Subelj2011EurPhysJB"),
 ("email",       r"University email",                             "Guimera2003PRE"),
 ("flybi",       r"\emph{Drosophila} protein--protein binary",    "Tang2023NatCommun"),
 ("reactome",    r"Reactome biochemical-reactions network",       "JoshiTope2005NAR"),
 ("route_views", r"Internet AS-level (RouteViews)",               "Leskovec2005KDD"),
 ("spanish",     r"Spanish-language word-association network",    "Batagelj2002LangTech"),
]
# the values printed in the SI, for --check
SI = {"dolphin": (62, 159, 5.13, 12, 4), "prox": (410, 2765, 13.49, 50, 6),
      "metabolic": (453, 2025, 8.94, 237, 6), "gkk": (949, 2496, 5.26, 26, 6),
      "lfr": (998, 1988, 3.98, 36, 6), "er": (1000, 25132, 50.26, 76, 6),
      "ba": (1000, 1996, 3.99, 81, 6), "hk": (1000, 1996, 3.99, 102, 6),
      "euroroad": (1039, 1305, 2.51, 10, 6), "email": (1133, 5451, 9.62, 71, 7),
      "flybi": (2705, 8458, 6.25, 129, 7),
      "reactome": (5973, 145778, 48.81, 855, 8),
      "route_views": (6474, 12572, 3.88, 1458, 8),
      "spanish": (11558, 43050, 7.45, 2986, 9)}


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib")); import paths  # noqa: E402
def thousands(x):
    return f"{x:,}".replace(",", "{,}")


def stats(tag):
    import numpy as np
    G = nx.read_edgelist(paths.network_file(tag), nodetype=int)
    N, E = G.number_of_nodes(), G.number_of_edges()
    deg = [d for _, d in G.degree()]
    return N, E, 2 * E / N, max(deg), int(np.floor(np.log(N)))


if "--check" in sys.argv:
    bad = 0
    for tag, _, _ in NETS:
        got = stats(tag); want = SI[tag]
        ok = (got[0] == want[0] and got[1] == want[1]
              and abs(got[2] - want[2]) < 0.005 and got[3] == want[3]
              and got[4] == want[4])
        bad += not ok
        print(f"  {tag:12s} computed {got}  SI {want}  {'ok' if ok else 'MISMATCH'}")
    print(f"\n  rows matching the SI: {len(NETS) - bad} of {len(NETS)}")
    sys.exit(0)

rows = []
for tag, name, cite in NETS:
    N, E, kbar, kmax, n = stats(tag)
    rows.append(f"{name:44s} & {thousands(N):8s} & {thousands(E):9s} & "
                f"{kbar:5.2f} & {thousands(kmax):7s} & {n} & "
                f"\\cite{{{cite}}} \\\\")
(OUT / "tab_networks.tex").write_text("\n".join(rows) + "\n")
print("  wrote tables/tab_networks.tex")
