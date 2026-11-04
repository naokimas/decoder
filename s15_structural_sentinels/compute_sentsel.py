# =====================================================================
# SI Section S15: Sentinel sets from network structure alone
#
# This script: z-scores of five structural statistics of the MinCorr and QR sets
# Feeds      : Table S22
# Produces   : data/sentsel.npz
# Runtime    : about ten minutes
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Are the MinCorr and QR sentinel sets special as SETS OF NETWORK NODES?

Why do particular sentinel sets work, and can their quality be judged in
advance?  The cheapest thing a practitioner could do
is judge a sentinel set from the network alone, without running any dynamics.
This script tests whether that is possible, by asking whether the sets the two
heuristics actually pick are distinguishable from uniformly random sets of the
same size on five purely structural statistics:

  mean degree, mean betweenness centrality, the standard deviation of the
  degrees within the set, the mean shortest-path distance between members of
  the set, and the overlap between the MinCorr set and the QR set.

Each is turned into a z-score against a null of R uniformly random n-node
sets drawn from all N nodes of the same network; the same R sets serve all
four statistics.  The overlap is scored against the overlap
of the same heuristic set with a random set of the same size.

Scope: the 11 adjacency-coupled dynamics and the 14 networks, so 154
(network, dynamics) pairs per heuristic.  This is about sentinel selection,
not about cross transfer, so each dynamics is treated on its own.

Writes data/sentsel.npz.
"""
import os, sys, importlib.util
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.sparse.csgraph import shortest_path

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "lib"          # shared modules
sys.path.insert(0, str(HERE.parent / "lib")); import paths  # noqa: E402
sys.path.insert(0, str(LIB))
_s = importlib.util.spec_from_file_location("leq", LIB / "load_equilibria.py")
leq = importlib.util.module_from_spec(_s); _s.loader.exec_module(leq)
import pipeline_unified as pu                                    # noqa: E402
import splits                                                    # noqa: E402

DYN = ["SIS", "SIR", "MSIS", "MSIS2", "MSIR", "MSIR2", "DW", "MUT", "GEN",
       "WC", "gLV"]                      # the 11 adjacency-coupled dynamics
DISP = {"MSIS": "MSIS1", "MSIR": "MSIR1"}
STATS = ["mean degree", "mean betweenness", "std degree", "mean distance"]
R = 1000
(HERE / "data").mkdir(exist_ok=True)


def z(actual, null):
    sd = float(np.std(null, ddof=1))
    return (float(actual) - float(np.mean(null))) / sd if sd > 0 else np.nan


def main():
    nets, out = leq.NETWORKS, {}
    tr, _ = splits.rows_random()
    Z = {m: {s: np.full((len(nets), len(DYN)), np.nan)
             for s in STATS} for m in ("minc", "qr")}
    Zov = {m: np.full((len(nets), len(DYN)), np.nan) for m in ("minc", "qr")}
    OV = np.full((len(nets), len(DYN)), np.nan)
    t0 = time.time()
    for ni, tag in enumerate(nets):
        G = nx.read_edgelist(leq.network_file(tag),
                             nodetype=int)
        nodes = sorted(G.nodes())
        idx = {v: k for k, v in enumerate(nodes)}
        N = len(nodes)
        n = int(np.floor(np.log(N)))
        deg = np.array([G.degree(v) for v in nodes], dtype=float)
        # a few minutes on the largest networks
        bt = nx.betweenness_centrality(G, normalized=True)
        btw = np.array([bt[v] for v in nodes], dtype=float)
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr",
                                     dtype=np.float64)
        # scipy's csgraph wants 32-bit index arrays
        A.indices = A.indices.astype(np.int32)
        A.indptr = A.indptr.astype(np.int32)

        rng = np.random.default_rng(4242 + ni)
        sets = {}
        for dj, sname in enumerate(DYN):
            Y = leq.load_eq(tag, sname)[tr]
            sets[("minc", dj)] = np.sort(pu.min_corr_sensors(Y, n))
            sets[("qr", dj)] = np.sort(pu.qr_pivot_sensors(Y, n))

        # null distributions: R uniformly random n-node sets drawn from all N
        # nodes, shared by the four statistics
        draws = np.array([rng.choice(N, size=n, replace=False)
                          for _ in range(R)])

        # distances: BFS from every node that appears in a heuristic set or
        # in a null set, in chunks, kept in single precision (hop counts)
        srcs = np.unique(np.concatenate([draws.ravel()]
                                        + [v for v in sets.values()]))
        Dsrc = np.empty((srcs.size, N), dtype=np.float32)
        for c in range(0, srcs.size, 500):
            Dsrc[c:c + 500] = shortest_path(
                A, method="D", unweighted=True,
                indices=srcs[c:c + 500].astype(np.int32))
        srow = {int(v): k for k, v in enumerate(srcs)}

        def mean_pair_dist(S):
            r = [Dsrc[srow[int(a)], int(b)] for i, a in enumerate(S)
                 for b in S[i + 1:]]
            return float(np.mean(r))

        null = {"mean degree": deg[draws].mean(axis=1),
                "mean betweenness": btw[draws].mean(axis=1),
                "std degree": deg[draws].std(axis=1, ddof=1),
                "mean distance": np.array([mean_pair_dist(d) for d in draws])}

        for dj in range(len(DYN)):
            for m in ("minc", "qr"):
                S = sets[(m, dj)]
                act = {"mean degree": deg[S].mean(),
                       "mean betweenness": btw[S].mean(),
                       "std degree": deg[S].std(ddof=1),
                       "mean distance": mean_pair_dist(S)}
                for st in STATS:
                    Z[m][st][ni, dj] = z(act[st], null[st])
            a, b = sets[("minc", dj)], sets[("qr", dj)]
            ov = len(set(a.tolist()) & set(b.tolist()))
            OV[ni, dj] = ov
            for m, S in (("minc", a), ("qr", b)):
                Sset = set(S.tolist())
                nullov = np.array([len(Sset & set(d.tolist())) for d in draws],
                                  dtype=float)
                Zov[m][ni, dj] = z(ov, nullov)
        print(f"  {tag}: N={N}, n={n}  [{time.time()-t0:.0f}s]", flush=True)

    np.savez(HERE / "data" / "sentsel.npz",
             networks=np.array(nets, dtype=object),
             dynamics=np.array([DISP.get(d, d) for d in DYN], dtype=object),
             stats=np.array(STATS, dtype=object), overlap=OV,
             **{f"Z_{m}_{s.replace(' ', '_')}": Z[m][s]
                for m in ("minc", "qr") for s in STATS},
             **{f"Zov_{m}": Zov[m] for m in ("minc", "qr")})
    print(f"  wrote data/sentsel.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
