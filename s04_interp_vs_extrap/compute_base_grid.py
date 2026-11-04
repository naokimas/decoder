# =====================================================================
# SI Section S4: Interpolation versus extrapolation
#
# This script: the matching baselines for the 3 x 3 grid
# Feeds      : the baselines of Table S7, and of Fig. S22 and Table S14
# Produces   : data/bgrid_*.npz
# Runtime    : under an hour
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Mean-prediction baseline for every (training, calibration, sentinel) case.

r(S->T) of the manuscript's Fig. 5 is the decoder's n-RMSE divided by the
n-RMSE of predicting every node by the mean of the SOURCE decoder's sentinel
readings on the target.  That baseline depends on

  the sentinel set   -> chosen on the TRAINING rows, so it follows the
                        training split and the sentinel method;
  the evaluation rows -> the 10 outside the CALIBRATION set.

so it must be recomputed for each (train, cal, sentinel) combination.  No
decoder is involved.  n-RMSE uses the target's full range over all 51 rows,
as everywhere else in this repository.

Writes data/bgrid_<train>_<cal>_<sent>.npz.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
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
leq = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(leq)
import pipeline_unified as pu                                   # noqa: E402
import splits                                                   # noqa: E402

DYN, NETS = leq.DYNAMICS, leq.NETWORKS
WAYS = ("random", "first", "last")
SENTS = ("minc", "qr")
OUT = HERE / "data"

cubes = {}
t0 = time.time()
for ni, tag in enumerate(NETS):
    G = nx.read_edgelist(leq.network_file(tag), nodetype=int)
    N = len(G.nodes()); K = int(np.floor(np.log(N)))
    eq = {s: leq.load_eq(tag, s) for s in DYN}
    rng_full = {s: float(eq[s].max() - eq[s].min()) for s in DYN}
    for trw in WAYS:
        tr, _ = splits.SPLITS[trw]()
        sidx = {}
        for s in DYN:
            Ytr = eq[s][tr]
            sidx[("qr", s)] = pu.qr_pivot_sensors(Ytr, K)
            sidx[("minc", s)] = pu.min_corr_sensors(Ytr, K)
        for calw in WAYS:
            _, ev = splits.SPLITS[calw]()
            for sent in SENTS:
                key = (trw, calw, sent)
                if key not in cubes:
                    cubes[key] = np.full((len(NETS), len(DYN), len(DYN)),
                                         np.nan)
                for i, src in enumerate(DYN):
                    s_idx = sidx[(sent, src)]
                    for j, tgt in enumerate(DYN):
                        Y = eq[tgt][ev]
                        per_row = Y[:, s_idx].mean(axis=1)
                        Yh = np.broadcast_to(per_row[:, None], Y.shape)
                        rmse = float(np.sqrt(np.mean((Yh - Y) ** 2)))
                        cubes[key][ni, i, j] = rmse / rng_full[tgt]
    print(f"  {tag}: elapsed {time.time()-t0:.0f}s", flush=True)
for (trw, calw, sent), c in cubes.items():
    np.savez(OUT / f"bgrid_{trw}_{calw}_{sent}.npz", rel=c,
             dynamics=np.array(DYN), networks=np.array(NETS))
print(f"wrote {len(cubes)} baseline cubes")
