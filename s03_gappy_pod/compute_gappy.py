# =====================================================================
# SI Section S3: Gappy POD
#
# This script: the 17 x 17 matrices for the four gappy variants
# Feeds      : Figs. S3 and S4, Tables S3 and S4
# Produces   : data/gappy_matrices.npz
# Runtime    : about half an hour
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Cross-transfer matrices for gappy POD, alongside the NN and linear ones.

Protocol is identical to the revised Figs. 4 and 5: 17 dynamics with LDW
last, the source decoder fitted on 41 interspersed control-parameter values,
the target input map and (a, b) calibrated on the same 41 interspersed values
of the target, and every cell scored on the remaining 10.  Only the
reconstruction step changes.

Writes data/gappy_matrices.npz.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util, time
from pathlib import Path
import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "lib"          # shared modules
sys.path.insert(0, str(HERE.parent / "lib")); import paths  # noqa: E402
sys.path.insert(0, str(LIB))
sys.path.insert(0, str(HERE))
_s = importlib.util.spec_from_file_location(
    "leq", LIB / "load_equilibria.py")
leq = importlib.util.module_from_spec(_s); _s.loader.exec_module(leq)

import pipeline_unified as pu                                   # noqa: E402
import splits                                                   # noqa: E402
import gappy                                                    # noqa: E402

DYN = ["SIS", "SIR", "MSIS", "MSIS2", "MSIR", "MSIR2", "DW", "MUT", "GEN",
       "WC", "gLV", "NM", "EUT", "EUT2", "VEG", "VEG2", "LDW"]
DISP = {"MSIS": "MSIS1", "MSIR": "MSIR1", "EUT": "EUT1", "VEG": "VEG1"}
SENTS = ["minc", "qr", "random"]
MODES = ["minmax", "center"]
# rank rules: n (the Q-DEIM form) and an energy-truncated variant
RANKS = {"rn": None, "e999": 0.999}
NETS = leq.NETWORKS


def nrmse(pred, true, full_range):
    return float(np.sqrt(np.mean((pred - true) ** 2))) / full_range \
        if full_range > 0 else np.nan


def main():
    n = len(DYN)
    acc = {(m, s, rk): np.zeros((n, n))
           for m in MODES for s in SENTS for rk in RANKS}
    cond = {(m, s, rk): [] for m in MODES for s in SENTS for rk in RANKS}
    tr, ev = splits.rows_random()
    t0 = time.time()
    for tag in NETS:
        G = nx.read_edgelist(leq.network_file(tag),
                             nodetype=int)
        N = len(G.nodes()); K = int(np.floor(np.log(N)))
        eq = {s: leq.load_eq(tag, s) for s in DYN}
        rngf = {s: float(eq[s].max() - eq[s].min()) for s in DYN}
        rnd = np.random.default_rng(2026)
        RAND = np.sort(rnd.choice(N, size=K, replace=False))
        for sent in SENTS:
            sidx = {}
            for s in DYN:
                Ytr = eq[s][tr]
                sidx[s] = (pu.qr_pivot_sensors(Ytr, K) if sent == "qr" else
                           pu.min_corr_sensors(Ytr, K) if sent == "minc"
                           else RAND.copy())
            for mode in MODES:
                for rk, rv in RANKS.items():
                    for i, src in enumerate(DYN):
                        ck = gappy.fit_gappy(eq[src], sidx[src], tr,
                                             rank=rv, mode=mode)
                        cond[(mode, sent, rk)].append(
                            float(np.linalg.cond(ck["basis"][sidx[src]])))
                        for j, tgt in enumerate(DYN):
                            Yt = eq[tgt]
                            P = gappy.predict_calibrated_gappy(ck, Yt, tr)
                            acc[(mode, sent, rk)][i, j] += nrmse(
                                P[ev], Yt[ev], rngf[tgt])
        print(f"  {tag}: {time.time()-t0:.0f}s", flush=True)
    out = {f"G_{m}_{s}_{rk}": acc[(m, s, rk)] / len(NETS)
           for m in MODES for s in SENTS for rk in RANKS}
    out.update({f"C_{m}_{s}_{rk}": np.array(cond[(m, s, rk)])
                for m in MODES for s in SENTS for rk in RANKS})
    np.savez(HERE / "data" / "gappy_matrices.npz",
             dynamics=np.array([DISP.get(d, d) for d in DYN], dtype=object),
             cal_rows=tr, eval_rows=ev, **out)
    print(f"  wrote data/gappy_matrices.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
