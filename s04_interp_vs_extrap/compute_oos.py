# =====================================================================
# SI Section S4: Interpolation versus extrapolation
#
# This script: the within-dynamics reconstruction behind Tables S5 and S6
# Produces   : data/oos_*.npz, data/oosbase_*.npz
# Runtime    : about an hour
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Fully out-of-sample cross-transfer: source split x target calibration.

The affine calibration is never fitted on the evaluation rows, and the RMSE
is normalized by the target's FULL range over all M conditions, as in the
manuscript.  (At the ends of a sweep the range of the 10 evaluation rows can
be a few percent of the full range, so normalizing by it would inflate the
n-RMSE many times over.)

For each source split (which fixes 41 training and 10 evaluation
conditions):

  source side   decoder trained on the 41 training rows; sentinels chosen on
                those rows; the normalization extrema of the manuscript's
                Eqs. (20)-(23) taken from those rows only.
  target side   the range mapping (Eqs. 27-28) and the affine (a, b)
                (Eqs. 29-30) taken from the target's CALIBRATION rows:
                  mode "oos"          the same 41 indices  -> out of sample
  scoring       RMSE over the 10 evaluation rows, divided by the target's
                full range over all 51 rows (the manuscript's n-RMSE).

Writes data/oos_<split>_oos_<dec>_<sent>.npz and
data/oosbase_<split>_<sent>.npz, which make_tables17.py turns into Tables S5
and S6 and into the within-dynamics numbers quoted in the text of Section
S4.  The "oos" tag is kept in the file name because the stored results carry
it.
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
# Tables S5 and S6 report the two non-random sentinel-selection methods
# only; Section S4 states that it omits random sentinels.
SENTINELS = ["minc", "qr"]
CELLS = [("nn", 0.0), ("lin", 0.01)]
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)


def nrmse(pred_rows, true_rows, full_range):
    """Manuscript n-RMSE: RMSE on the evaluated rows over the FULL range."""
    rmse = float(np.sqrt(np.mean((pred_rows - true_rows) ** 2)))
    return rmse / full_range if full_range > 0 else np.nan


def process_network(tag):
    G = nx.read_edgelist(leq.network_file(tag), nodetype=int)
    N = len(G.nodes())
    K = int(np.floor(np.log(N)))
    eq = {s: leq.load_eq(tag, s) for s in DYN}
    rng_full = {s: float(eq[s].max() - eq[s].min()) for s in DYN}

    out = {}
    for split in ("random", "first", "last"):
        tr, ev = splits.SPLITS[split]()
        sent_idx = {}
        for s in DYN:
            Ytr = eq[s][tr]
            sent_idx[("qr", s)] = pu.qr_pivot_sensors(Ytr, K)
            sent_idx[("minc", s)] = pu.min_corr_sensors(Ytr, K)

        for sent in SENTINELS:                       # mean-of-sentinels base
            B = np.full((len(DYN), len(DYN)), np.nan)
            for i, src in enumerate(DYN):
                s_idx = sent_idx[(sent, src)]
                for j, tgt in enumerate(DYN):
                    Y = eq[tgt]
                    per_row = Y[ev][:, s_idx].mean(axis=1)
                    Yhat = np.broadcast_to(per_row[:, None], Y[ev].shape)
                    B[i, j] = nrmse(Yhat, Y[ev], rng_full[tgt])
            out[("base", split, sent)] = B

        for dec, reg in CELLS:
            for sent in SENTINELS:
                ck = {}
                for s in DYN:
                    idx = sent_idx[(sent, s)]
                    ck[s] = (splits.train_nn(eq[s], idx, reg, tr)
                             if dec == "nn"
                             else splits.fit_linear(eq[s], idx, reg, tr))
                # the target is calibrated on its own 41 training rows, so
                # the 10 scored rows are out of sample on both sides
                M = np.full((len(DYN), len(DYN)), np.nan)
                for i, src in enumerate(DYN):
                    for j, tgt in enumerate(DYN):
                        Yt = eq[tgt]
                        P = splits.predict_calibrated(ck[src], Yt, dec, tr)
                        M[i, j] = nrmse(P[ev], Yt[ev], rng_full[tgt])
                out[(dec, split, "oos", sent)] = M
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
    for k, c in cubes.items():
        stem = (f"oosbase_{k[1]}_{k[2]}" if k[0] == "base"
                else f"oos_{k[1]}_{k[2]}_{k[0]}_{k[3]}")
        np.savez(OUT / f"{stem}.npz", rel=c, dynamics=np.array(DYN),
                 networks=np.array(NETS))
    print(f"wrote {len(cubes)} NPZ files")


if __name__ == "__main__":
    main()
