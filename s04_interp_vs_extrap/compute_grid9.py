# =====================================================================
# SI Section S4: Interpolation versus extrapolation
#
# This script: the 3 x 3 grid of source and target splits
# Feeds      : Table S7; the random/random cell also feeds Fig. S22 and Table S14
# Produces   : data/grid9_*.npz
# Runtime    : a few hours
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""All nine (training split) x (target-calibration split) combinations.

The source decoder is trained on 41 control values chosen one of three ways
(random / first / last).  Independently, the target range mapping and the
affine (a, b) of Eqs. (27)-(30) are estimated from 41 control values chosen
one of the same three ways.  That gives nine cases for cross-transfer.

Each case is scored on the 10 control values NOT in its calibration set --
the only rows that are out of sample for the target side -- and the RMSE is
divided by the target's full range over all 51 rows, which is the
manuscript's n-RMSE.  Note that the evaluation rows therefore follow the
CALIBRATION split, so a comparison between two cases with different
calibration splits also changes the evaluation window; the diagnostic figure
says so.

Only the four decoder x sentinel panels the manuscript relies on are
computed (MinCorr and QR, both decoders); random sentinels are omitted.

Writes data/grid9_<train>_<cal>_<dec>_<sent>.npz, each (n_net, 21, 21).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "lib"          # shared modules
sys.path.insert(0, str(HERE.parent / "lib")); import paths  # noqa: E402
sys.path.insert(0, str(LIB))
sys.path.insert(0, str(HERE))

_spec = importlib.util.spec_from_file_location(
    "leq", LIB / "load_equilibria.py")
leq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(leq)

import pipeline_unified as pu                                  # noqa: E402
import splits                                                  # noqa: E402

DYN, NETS = leq.DYNAMICS, leq.NETWORKS
PANELS = [("nn", 0.0, "minc"), ("nn", 0.0, "qr"),
          ("lin", 0.01, "minc"), ("lin", 0.01, "qr")]
WAYS = ("random", "first", "last")
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)


def process_network(tag):
    G = nx.read_edgelist(leq.network_file(tag), nodetype=int)
    N = len(G.nodes())
    K = int(np.floor(np.log(N)))
    eq = {s: leq.load_eq(tag, s) for s in DYN}
    rng_full = {s: float(eq[s].max() - eq[s].min()) for s in DYN}

    out = {}
    for tr_way in WAYS:
        tr, _ = splits.SPLITS[tr_way]()
        sent_idx = {}
        for s in DYN:
            Ytr = eq[s][tr]
            sent_idx[("qr", s)] = pu.qr_pivot_sensors(Ytr, K)
            sent_idx[("minc", s)] = pu.min_corr_sensors(Ytr, K)
        for dec, reg, sent in PANELS:
            ck = {}
            for s in DYN:
                idx = sent_idx[(sent, s)]
                ck[s] = (splits.train_nn(eq[s], idx, reg, tr) if dec == "nn"
                         else splits.fit_linear(eq[s], idx, reg, tr))
            for cal_way in WAYS:
                cal, ev = splits.SPLITS[cal_way]()
                M = np.full((len(DYN), len(DYN)), np.nan)
                for i, src in enumerate(DYN):
                    for j, tgt in enumerate(DYN):
                        Yt = eq[tgt]
                        P = splits.predict_calibrated(ck[src], Yt, dec, cal)
                        rmse = float(np.sqrt(np.mean((P[ev] - Yt[ev]) ** 2)))
                        M[i, j] = rmse / rng_full[tgt]
                out[(tr_way, cal_way, dec, sent)] = M
    return out


def main():
    keys, cubes = None, {}
    t0 = time.time()
    for n, tag in enumerate(NETS):
        t1 = time.time()
        res = process_network(tag)
        if keys is None:
            keys = list(res)
            cubes = {k: np.full((len(NETS), len(DYN), len(DYN)), np.nan)
                     for k in keys}
        for k, M in res.items():
            cubes[k][n] = M
        print(f"  {tag}: {time.time()-t1:6.1f}s "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)
    for (trw, calw, dec, sent), c in cubes.items():
        np.savez(OUT / f"grid9_{trw}_{calw}_{dec}_{sent}.npz", rel=c,
                 dynamics=np.array(DYN), networks=np.array(NETS))
    print(f"wrote {len(cubes)} NPZ files")


if __name__ == "__main__":
    main()
