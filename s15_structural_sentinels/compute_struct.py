# =====================================================================
# SI Section S15: Sentinel sets from network structure alone
#
# This script: decoders driven by degree and betweenness sentinels
# Feeds      : Table S23
# Produces   : data/struct.npz
# Runtime    : about ten minutes
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Sentinel sets chosen from the network structure alone.

Can the quality of a sentinel set be judged a priori?  The z-score analysis
of compute_sentsel.py shows that MinCorr and QR pick high-degree,
high-betweenness nodes.  The obvious follow-up is to select on those
properties directly, without looking at any dynamics:

  degree        the n nodes of largest degree
  betweenness   the n nodes of largest betweenness centrality

Ties are broken uniformly at random.  Both sets depend on the network only,
so a single set serves all 17 dynamics on that network, in contrast to
MinCorr and QR, which are refitted for every source dynamics.

Everything else follows fig4_fig5_cross_transfer/compute_matrices.py
exactly: the same 17 dynamics, the same interspersed 41/10 split on both
sides, the same out-of-sample target calibration.  The output is the
17 x 17 n-RMSE matrix averaged over the 14 networks, for each of the two
decoder architectures; make_tables.py takes its 11 x 11 adjacency-coupled
block and puts the median and mean of its diagonal and off-diagonal cells
beside those of the MinCorr, QR and random sentinels in SI Table S23.

Writes data/struct.npz.
"""
import os, sys, importlib.util
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
from pathlib import Path

import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "lib"          # shared modules
sys.path.insert(0, str(HERE.parent / "lib")); import paths  # noqa: E402
sys.path.insert(0, str(LIB))
_s = importlib.util.spec_from_file_location("leq", LIB / "load_equilibria.py")
leq = importlib.util.module_from_spec(_s); _s.loader.exec_module(leq)
import splits                                                    # noqa: E402

DYN = ["SIS", "SIR", "MSIS", "MSIS2", "MSIR", "MSIR2", "DW", "MUT", "GEN",
       "WC", "gLV", "NM", "EUT", "EUT2", "VEG", "VEG2", "LDW"]
DISP = {"MSIS": "MSIS1", "MSIR": "MSIR1", "EUT": "EUT1", "VEG": "VEG1"}
REG = {"nn": 0.0, "lin": 0.01}
(HERE / "data").mkdir(exist_ok=True)


def nrmse(pred, true, full_range):
    r = float(np.sqrt(np.mean((pred - true) ** 2)))
    return r / full_range if full_range > 0 else np.nan


def top_n(score, n, rng):
    """The n largest entries, ties broken uniformly at random."""
    jitter = rng.random(score.size)                # random within ties
    order = np.lexsort((jitter, -score))
    return np.sort(order[:n])


def main():
    n_d = len(DYN)
    keys = [(d, s) for d in ("nn", "lin") for s in ("degree", "betweenness")]
    acc = {k: np.zeros((n_d, n_d)) for k in keys}
    accb = {k: np.zeros((n_d, n_d)) for k in keys}
    tr, ev = splits.rows_random()
    t0 = time.time()
    for ni, tag in enumerate(leq.NETWORKS):
        G = nx.read_edgelist(leq.network_file(tag),
                             nodetype=int)
        nodes = sorted(G.nodes()); N = len(nodes)
        K = int(np.floor(np.log(N)))
        rng = np.random.default_rng(909 + ni)
        deg = np.array([G.degree(v) for v in nodes], dtype=float)
        # a few minutes on the largest networks
        bt = nx.betweenness_centrality(G, normalized=True)
        btw = np.array([bt[v] for v in nodes], dtype=float)
        SENT = {"degree": top_n(deg, K, rng),
                "betweenness": top_n(btw, K, rng)}
        eq = {s: leq.load_eq(tag, s) for s in DYN}
        rngf = {s: float(eq[s].max() - eq[s].min()) for s in DYN}
        for dec in ("nn", "lin"):
            for sname, idx in SENT.items():
                for i, src in enumerate(DYN):
                    ck = (splits.train_nn(eq[src], idx, REG[dec], tr)
                          if dec == "nn"
                          else splits.fit_linear(eq[src], idx, REG[dec], tr))
                    for j, tgt in enumerate(DYN):
                        Yt = eq[tgt]
                        P = splits.predict_calibrated(ck, Yt, dec, tr)
                        acc[(dec, sname)][i, j] += nrmse(P[ev], Yt[ev],
                                                         rngf[tgt])
                        per = Yt[ev][:, idx].mean(axis=1)
                        B = np.broadcast_to(per[:, None], Yt[ev].shape)
                        accb[(dec, sname)][i, j] += nrmse(B, Yt[ev],
                                                          rngf[tgt])
        print(f"  {tag}: N={N}, n={K}  [{time.time()-t0:.0f}s]", flush=True)
    out = {}
    nn = len(leq.NETWORKS)
    for (dec, sname) in keys:
        out[f"M_{dec}_{sname}"] = acc[(dec, sname)] / nn
        out[f"B_{dec}_{sname}"] = accb[(dec, sname)] / nn
    np.savez(HERE / "data" / "struct.npz",
             dynamics=np.array([DISP.get(d, d) for d in DYN], dtype=object),
             **out)
    print(f"  wrote data/struct.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
