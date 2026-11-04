# =====================================================================
# SI Section S4: Interpolation versus extrapolation
#
# This script: sweep of the target calibration fraction
# Feeds      : Table S8
# Produces   : data/tfrac_*.npz, data/tfracbase_*.npz
# Runtime    : a few hours
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""How cross-transfer depends on the SIZE of the target calibration set.

Table S7 varies the target calibration set among interspersed / first-41 /
last-41 at a fixed size of 41.  Here the target calibration is always
interspersed -- the two contiguous choices perform too poorly to be worth
sweeping -- and its SIZE varies instead, over the fractions in FRACS.

The source side keeps the three choices of Table S7 (interspersed, first-41,
last-41), all at 41 training values.

Sizes follow the manuscript's rule: with M = 51 control-parameter values and
a target calibration fraction p, the evaluation set holds
floor((1 - p) * M) values and the calibration set the remaining ones.  At
p = 0.8 this gives floor(0.2 * 51) = 10 and 41, the manuscript's own split.

The interspersed sets are drawn ONCE, with the manuscript's seed, by
shuffling the 51 indices and taking the first floor((1 - p) * M) as the
evaluation set.  So (i) p = 0.8 reproduces the paper's split exactly, and
(ii) the evaluation sets are nested: the 5 rows evaluated at p = 0.9 are a
subset of the 10 evaluated at p = 0.8, and so on.

Each case is scored on the rows outside its own calibration set, with the
RMSE divided by the target's full range over all 51 rows, exactly as in
compute_grid9.py, and the output calibration is the manuscript's
least-squares (a, b) of Eq. (30).

Writes data/tfrac_<srcway>_<pct>_<dec>_<sent>.npz    (decoder n-RMSE)
   and data/tfracbase_<srcway>_<pct>_<sent>.npz      (mean-prediction base)
each (n_net, 21, 21).
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
SRC_WAYS = ("random", "first", "last")
# The calibration-set sizes of Table S8, as percentages of the 51 rows.
FRACS = (50, 60, 70, 80, 90)
M_TOTAL = 51
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)


def rows_frac(pct):
    """Interspersed calibration/evaluation split at a given percentage.

    Same shuffle as splits.rows_random, so pct = 80 returns exactly the
    manuscript's 41/10 split and the evaluation sets are nested in pct.
    """
    n_eval = int(np.floor((1.0 - pct / 100.0) * M_TOTAL))
    rng = np.random.default_rng(pu.SEED)
    idx = np.arange(M_TOTAL)
    rng.shuffle(idx)
    return np.sort(idx[n_eval:]), np.sort(idx[:n_eval])


def process_network(tag):
    G = nx.read_edgelist(leq.network_file(tag), nodetype=int)
    N = len(G.nodes())
    K = int(np.floor(np.log(N)))
    eq = {s: leq.load_eq(tag, s) for s in DYN}
    rng_full = {s: float(eq[s].max() - eq[s].min()) for s in DYN}
    splits_frac = {p: rows_frac(p) for p in FRACS}

    out = {}
    for tr_way in SRC_WAYS:
        tr, _ = splits.SPLITS[tr_way]()
        sent_idx = {}
        for s in DYN:
            Ytr = eq[s][tr]
            sent_idx[("qr", s)] = pu.qr_pivot_sensors(Ytr, K)
            sent_idx[("minc", s)] = pu.min_corr_sensors(Ytr, K)

        # mean-prediction baseline: no decoder, but the sentinel set depends
        # on the source dynamics and the source split, and the evaluation
        # rows depend on the fraction.
        for sent in ("minc", "qr"):
            for p in FRACS:
                _, ev = splits_frac[p]
                B = np.full((len(DYN), len(DYN)), np.nan)
                for i, src in enumerate(DYN):
                    s_idx = sent_idx[(sent, src)]
                    for j, tgt in enumerate(DYN):
                        Y = eq[tgt][ev]
                        per_row = Y[:, s_idx].mean(axis=1)
                        Yh = np.broadcast_to(per_row[:, None], Y.shape)
                        rmse = float(np.sqrt(np.mean((Yh - Y) ** 2)))
                        B[i, j] = rmse / rng_full[tgt]
                out[("base", tr_way, p, None, sent)] = B

        for dec, reg, sent in PANELS:
            ck = {}
            for s in DYN:
                idx = sent_idx[(sent, s)]
                ck[s] = (splits.train_nn(eq[s], idx, reg, tr) if dec == "nn"
                         else splits.fit_linear(eq[s], idx, reg, tr))
            for p in FRACS:
                cal, ev = splits_frac[p]
                M = np.full((len(DYN), len(DYN)), np.nan)
                for i, src in enumerate(DYN):
                    for j, tgt in enumerate(DYN):
                        Yt = eq[tgt]
                        P = splits.predict_calibrated(ck[src], Yt, dec, cal)
                        rmse = float(np.sqrt(np.mean((P[ev] - Yt[ev]) ** 2)))
                        M[i, j] = rmse / rng_full[tgt]
                out[("dec", tr_way, p, dec, sent)] = M
    return out


def main():
    for p in FRACS:
        cal, ev = rows_frac(p)
        print(f"  {p}%: {len(cal)} calibration, {len(ev)} evaluation values")
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
    for (kind, trw, p, dec, sent), c in cubes.items():
        name = (f"tfrac_{trw}_{p}_{dec}_{sent}.npz" if kind == "dec"
                else f"tfracbase_{trw}_{p}_{sent}.npz")
        np.savez(OUT / name, rel=c, dynamics=np.array(DYN),
                 networks=np.array(NETS))
    print(f"wrote {len(cubes)} NPZ files")


if __name__ == "__main__":
    main()
