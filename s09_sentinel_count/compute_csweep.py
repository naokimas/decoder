# =====================================================================
# SI Section S9: Robustness to the number of sentinel nodes
#
# This script: the c-sweep on the 14 networks; run once per shard
# Feeds      : Fig. S19, via merge_plot.py
# Produces   : data/part_*.npz
# Runtime    : about two hours with seven shards
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Sentinel-count sweep for SI Section S9 (Fig. S19).

On the target side, the input map and the affine (a, b) are fitted on the 41
interspersed calibration rows, and every cell is scored on the remaining 10.

For each network we set n = max(1, floor(c floor(ln N))) with c in
[0.25, 2], train the five leading source decoders (SIS, SIR, C17, C11,
C11L1) under each decoder architecture and each non-random
sentinel-selection method, and evaluate on the 11 transferable target
dynamics.  Values are averaged over the 11 targets and then over the 14
networks.
"""
import os, sys, time, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util
from pathlib import Path
import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "lib"          # shared modules
sys.path.insert(0, str(HERE.parent / "lib")); import paths  # noqa: E402; sys.path.insert(0, str(LIB))
_s = importlib.util.spec_from_file_location("leq", LIB / "load_equilibria.py")
leq = importlib.util.module_from_spec(_s); _s.loader.exec_module(leq)
_r = importlib.util.spec_from_file_location(
    "rk", HERE.parent / "s13_decoder_rankings" / "compute_rank.py")
rk = importlib.util.module_from_spec(_r); _r.loader.exec_module(rk)

import splits                                                   # noqa: E402

DYN, TRANS = rk.DYN, rk.TRANS
SOURCES = ["SIS", "SIR", "C17", "C11", "C11L1"]
PANELS = [("nn", "minc"), ("nn", "qr"), ("lin", "minc"), ("lin", "qr")]
NETS = leq.NETWORKS
(HERE / "data").mkdir(exist_ok=True)


def enumerate_c(Ks, cmin=0.25, cmax=2.0):
    cs = set()
    for K in Ks:
        for m in range(max(1, int(np.ceil(cmin * K - 1e-9))),
                       int(np.floor(cmax * K + 1e-9)) + 1):
            c = m / K
            if cmin - 1e-9 <= c <= cmax + 1e-9:
                cs.add(round(c, 10))
    return sorted(cs)


def one_cell(eq, rngf, tr, ev, n, dec, sent, RAND):
    """Mean n-RMSE and baseline over the 11 transferable targets."""
    out = {}
    for src in SOURCES:
        if src == "C11L1":
            ms, bs = [], []
            for tgt in TRANS:
                Yp = rk.pooled_matrix([s for s in TRANS if s != tgt], eq, tr)
                idx = rk.sentinels(Yp, sent, n, RAND)
                ck = rk.train(Yp, idx, dec, np.arange(Yp.shape[0]))
                m, b = rk.score(ck, eq[tgt], dec, tr, ev, rngf[tgt])
                ms.append(m); bs.append(b)
            out[src] = (float(np.mean(ms)), float(np.mean(bs)))
            continue
        if src in ("C11", "C17"):
            members = TRANS if src == "C11" else DYN
            Yp = rk.pooled_matrix(members, eq, tr)
            idx = rk.sentinels(Yp, sent, n, RAND)
            ck = rk.train(Yp, idx, dec, np.arange(Yp.shape[0]))
        else:
            idx = rk.sentinels(eq[src][tr], sent, n, RAND)
            ck = rk.train(eq[src], idx, dec, tr)
        ms, bs = [], []
        for tgt in TRANS:
            m, b = rk.score(ck, eq[tgt], dec, tr, ev, rngf[tgt])
            ms.append(m); bs.append(b)
        out[src] = (float(np.mean(ms)), float(np.mean(bs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--nets", default="")
    ap.add_argument("--out", default="csweep.npz")
    a = ap.parse_args()
    nets = [t for t in NETS if not a.nets or t in a.nets.split(",")]
    tr, ev = splits.rows_random()

    Ks = {}
    for tag in NETS:
        G = nx.read_edgelist(leq.network_file(tag),
                             nodetype=int)
        Ks[tag] = (len(G.nodes()), int(np.floor(np.log(len(G.nodes())))))
    CS = enumerate_c({K for _, K in Ks.values()})
    print(f"  {len(CS)} values of c, {len(nets)} networks", flush=True)

    if a.probe:
        tag = "spanish"; N, K = Ks[tag]
        eq = {s: leq.load_eq(tag, s) for s in DYN}
        rngf = {s: float(eq[s].max() - eq[s].min()) for s in DYN}
        RAND = np.sort(np.random.default_rng(2026).choice(N, K, replace=False))
        for dec, sent in (("nn", "minc"), ("lin", "minc")):
            t0 = time.time()
            one_cell(eq, rngf, tr, ev, K, dec, sent, RAND)
            print(f"  {tag} (N={N}) one cell, {dec}: {time.time()-t0:.1f}s")
        return

    M = np.full((len(CS), len(SOURCES), len(PANELS)), 0.0)
    Bm = np.zeros_like(M); CNT = np.zeros_like(M)
    NDOL = np.zeros(len(CS))
    t0 = time.time()
    for tag in nets:
        N, K = Ks[tag]
        eq = {s: leq.load_eq(tag, s) for s in DYN}
        rngf = {s: float(eq[s].max() - eq[s].min()) for s in DYN}
        RAND = np.sort(np.random.default_rng(2026).choice(N, K, replace=False))
        for ci, c in enumerate(CS):
            n = min(max(1, int(np.floor(c * K + 1e-9))), N)
            if tag == "dolphin":
                NDOL[ci] = n
            for pi, (dec, sent) in enumerate(PANELS):
                res = one_cell(eq, rngf, tr, ev, n, dec, sent, RAND)
                for si, s in enumerate(SOURCES):
                    M[ci, si, pi] += res[s][0]
                    Bm[ci, si, pi] += res[s][1]
                    CNT[ci, si, pi] += 1
        print(f"  {tag}: {time.time()-t0:.0f}s", flush=True)
    np.savez(HERE / "data" / a.out, M=M, B=Bm, CNT=CNT,
             cs=np.array(CS), sources=np.array(SOURCES, dtype=object),
             panels=np.array([f"{d}_{s}" for d, s in PANELS], dtype=object),
             n_dolphin=NDOL, cal_rows=tr, eval_rows=ev)
    print(f"  wrote data/{a.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
