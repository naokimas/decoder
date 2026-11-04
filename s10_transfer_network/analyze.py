# =====================================================================
# SI Section S10: Robustness of the cross-transfer network structure
#
# This script: Table S11: component structure, edge density and reciprocity
# Produces   : tables/tab_wcc.tex, data/wcc_stats.json
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Structure of the cross-transfer network, beyond weak connectivity.

Weak connectivity is a permissive criterion for a transfer class.  This
script recomputes the transfer network from the current Fig. 4 and Fig. 5
matrices (17 dynamics, LDW included, out-of-sample target calibration) and
reports, for every one of the six decoder x sentinel combinations,

  * the weakly connected components and their membership,
  * the edge density and the reciprocity of each component, the latter in
    the density-corrected sense of Garlaschelli and Loffredo, Phys. Rev.
    Lett. 93, 268701 (2004), rho = (r - a) / (1 - a), where r is the
    fraction of edges that are reciprocated and a is the edge density,
  * the strongly connected components inside each weak component, and
  * a clustering of the continuous n-RMSE values that never thresholds the
    error, using only r(S->T) < 0.8 to decide which ordered pairs carry an
    edge.

Edge rule of the main text: S -> T iff n-RMSE < 0.15 and r < 0.8.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
from pathlib import Path

import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
FIG45 = HERE.parent / "shared_data" / "fig45_matrices.npz"
PANELS = [("a", "NN, MinCorr", "nn", "minc"), ("b", "NN, QR", "nn", "qr"),
          ("c", "NN, random", "nn", "random"),
          ("d", "linear, MinCorr", "lin", "minc"),
          ("e", "linear, QR", "lin", "qr"),
          ("f", "linear, random", "lin", "random")]
ADJ = {"SIS", "SIR", "MSIS1", "MSIS2", "MSIR1", "MSIR2", "DW", "MUT", "GEN",
       "WC", "gLV"}
NRMSE_T, RATIO_T = 0.15, 0.8

d = np.load(FIG45, allow_pickle=True)
LAB = [str(x) for x in d["dynamics"]]
n = len(LAB)


def matrices(dec, sent):
    M = d[f"M_{dec}_{sent}"]                      # n-RMSE, fraction of range
    B = d[f"B_{dec}_{sent}"]
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(np.isfinite(B) & (B > 0), M / B, np.nan)
    return M, R


def build_graph(M, R, use_nrmse=True):
    G = nx.DiGraph()
    G.add_nodes_from(LAB)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ok = R[i, j] < RATIO_T
            if use_nrmse:
                ok = ok and M[i, j] < NRMSE_T
            if ok:
                G.add_edge(LAB[i], LAB[j], nrmse=float(M[i, j]),
                           ratio=float(R[i, j]))
    return G


def reciprocity(G):
    """Density-corrected reciprocity of Garlaschelli and Loffredo (2004).

    rho = (r - a) / (1 - a), with r the fraction of directed edges whose
    reverse edge is also present and a = L / (N (N - 1)) the edge density.
    rho = 0 for a network whose mutual edges are as frequent as chance,
    rho > 0 when mutual edges are over-represented, rho < 0 when they are
    under-represented.  Self-loops are absent by construction.
    """
    k = G.number_of_nodes()
    e = G.number_of_edges()
    if k < 2 or e == 0:
        return float("nan")
    r = sum(1 for u, v in G.edges() if G.has_edge(v, u)) / e
    a = e / (k * (k - 1))
    return (r - a) / (1 - a) if a < 1 else float("nan")


def component_stats(G, nodes):
    """Directed edge density and reciprocity on the induced subgraph."""
    H = G.subgraph(nodes)
    k = H.number_of_nodes()
    e = H.number_of_edges()
    density = e / (k * (k - 1)) if k > 1 else float("nan")
    recip = reciprocity(H)
    sccs = [sorted(c) for c in nx.strongly_connected_components(H)]
    sccs.sort(key=lambda c: (-len(c), c))
    return dict(size=k, edges=e, density=density, reciprocity=recip,
                sccs=sccs, largest_scc=len(sccs[0]) if sccs else 0)


def wcc_report(G):
    out = []
    for comp in sorted(nx.weakly_connected_components(G),
                       key=lambda c: (-len(c), sorted(c))):
        nodes = sorted(comp, key=LAB.index)
        st = component_stats(G, nodes)
        st["members"] = nodes
        st["all_adjacency"] = set(nodes) <= ADJ
        st["all_diffusive"] = not (set(nodes) & ADJ)
        out.append(st)
    return out


def tex(x, fmt):
    """LaTeX cell; a negative number goes in math mode for the proper minus."""
    s = f"{x:{fmt}}"
    return f"${s}$" if x < 0 else s


def write_table(results):
    """Table S11: the structure of each cross-transfer network and its LWCC.

    The components of each entry are sorted by decreasing size, so the first
    one is the largest weakly connected component.
    """
    out = HERE / "tables"; out.mkdir(exist_ok=True)
    rows = []
    for _, title, _, _ in PANELS:
        r = results[title]
        big = r["components"][0]
        rows.append(
            f"{title} & {r['density']:.3f} & {tex(r['reciprocity'], '.3f')} & "
            f"{big['size']} & {big['density']:.3f} & "
            f"{tex(big['reciprocity'], '.3f')} & "
            f"{len(big['sccs'])} & {big['largest_scc']}" + r"\\")
    (out / "tab_wcc.tex").write_text("\n".join(rows) + "\n")
    print("  wrote tables/tab_wcc.tex")
    for r in rows:
        print("   ", r)
    print()


def main():
    results = {}
    for lett, title, dec, sent in PANELS:
        M, R = matrices(dec, sent)
        G = build_graph(M, R)
        results[title] = dict(
            n_edges=G.number_of_edges(),
            density=G.number_of_edges() / (n * (n - 1)),
            reciprocity=reciprocity(G),
            components=wcc_report(G))
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "wcc_stats.json").write_text(json.dumps(results, indent=1))
    write_table(results)

    print("EDGE RULE: n-RMSE < 0.15 and r < 0.8    "
          f"({n} dynamics, {n*(n-1)} ordered pairs)\n")
    for lett, title, dec, sent in PANELS:
        r = results[title]
        print(f"=== ({lett}) {title}: {r['n_edges']} edges, "
              f"density {r['density']:.3f}, reciprocity {r['reciprocity']:.3f}")
        for c in r["components"]:
            kind = ("all adjacency" if c["all_adjacency"] else
                    "all diffusive" if c["all_diffusive"] else "MIXED")
            if c["size"] == 1:
                print(f"    singleton {c['members'][0]}")
                continue
            print(f"    size {c['size']:2d} [{kind}] density {c['density']:.3f} "
                  f"reciprocity {c['reciprocity']:.3f} "
                  f"largest SCC {c['largest_scc']}/{c['size']}")
            print(f"      members: {' '.join(c['members'])}")
            if len(c["sccs"]) > 1:
                nontriv = [s for s in c["sccs"] if len(s) > 1]
                triv = [s[0] for s in c["sccs"] if len(s) == 1]
                for s in nontriv:
                    print(f"      SCC({len(s)}): {' '.join(s)}")
                if triv:
                    print(f"      SCC(1) x{len(triv)}: {' '.join(triv)}")
        print()


if __name__ == "__main__":
    main()
