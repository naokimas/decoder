# =====================================================================
# SI Section S6: Two-parameter families of dynamics
#
# This script: cross-transfer matrices including MSIS2P and MUT2P
# Feeds      : Figs. S7 and S8
# Produces   : data/twod45_matrices.npz
# Runtime    : about an hour
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Cross-transfer matrices with the two two-parameter families as extra dynamics.

The columns are the 17 dynamics of the main text plus MSIS2P and MUT2P, so
the matrix is 19 x 19.  Only the interspersed split is used, on both sides:

  one-parameter dynamics  41 of the 51 control-parameter values for training
                          or calibration, scored on the remaining 10
  two-parameter families  51 of the 64 grid cells, scored on the remaining 13

Each cell is the n-RMSE of a decoder trained on the source and deployed on
the target, scored on the target's own held-out values and normalized by the
target's full range, together with the mean-prediction baseline computed on
the same rows with the same sentinel set.  This is the protocol of the
revision's Figs. 4 and 5, extended to targets whose sweep is a grid.

Writes data/twod45_matrices.npz, averaged over the 14 networks.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util
import time
from pathlib import Path
import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "lib"          # shared modules
sys.path.insert(0, str(HERE.parent / "lib")); import paths  # noqa: E402
sys.path.insert(0, str(LIB))
_s = importlib.util.spec_from_file_location(
    "leq", LIB / "load_equilibria.py")
leq = importlib.util.module_from_spec(_s); _s.loader.exec_module(leq)

import pipeline_unified as pu                                   # noqa: E402
import splits                                                   # noqa: E402
import splits2d as s2                                           # noqa: E402

DYN1 = ["SIS", "SIR", "MSIS", "MSIS2", "MSIR", "MSIR2", "DW", "MUT", "GEN",
        "WC", "gLV", "NM", "EUT", "EUT2", "VEG", "VEG2", "LDW"]
DYN2 = ["MSIS2P", "MUT2P"]
DYN = DYN1 + DYN2
DISP = {"MSIS": "MSIS1", "MSIR": "MSIR1", "EUT": "EUT1", "VEG": "VEG1"}
PANELS = [("nn", "minc"), ("nn", "qr"), ("nn", "random"),
          ("lin", "minc"), ("lin", "qr"), ("lin", "random")]
REG = {"nn": 0.0, "lin": 0.01}
NETS = leq.NETWORKS


def nrmse(pred, true, full_range):
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return rmse / full_range if full_range > 0 else np.nan


def main():
    n = len(DYN)
    acc = {p: np.zeros((n, n)) for p in PANELS}
    accb = {p: np.zeros((n, n)) for p in PANELS}
    tr1, ev1 = splits.rows_random()          # 41 / 10, interspersed
    tr2, ev2 = s2.rows_random()              # 51 / 13, interspersed
    ROWS = {d: (tr2, ev2) if d in DYN2 else (tr1, ev1) for d in DYN}
    t0 = time.time()
    for tag in NETS:
        G = nx.read_edgelist(leq.network_file(tag),
                             nodetype=int)
        N = len(G.nodes()); K = int(np.floor(np.log(N)))
        eq = {}
        for d in DYN:
            eq[d] = s2.load_eq2(d, tag) if d in DYN2 else leq.load_eq(tag, d)
            assert eq[d].shape[1] == N, f"{tag}/{d}: {eq[d].shape}, N={N}"
        rngf = {d: float(eq[d].max() - eq[d].min()) for d in DYN}
        rnd = np.random.default_rng(2026)
        RAND = np.sort(rnd.choice(N, size=K, replace=False))
        for dec, sent in PANELS:
            sidx = {}
            for d in DYN:
                Ytr = eq[d][ROWS[d][0]]
                sidx[d] = (pu.qr_pivot_sensors(Ytr, K) if sent == "qr" else
                           pu.min_corr_sensors(Ytr, K) if sent == "minc"
                           else RAND.copy())
            for i, src in enumerate(DYN):
                idx = sidx[src]
                trs = ROWS[src][0]
                ck = (splits.train_nn(eq[src], idx, REG[dec], trs)
                      if dec == "nn"
                      else splits.fit_linear(eq[src], idx, REG[dec], trs))
                for j, tgt in enumerate(DYN):
                    Yt = eq[tgt]
                    cal, ev = ROWS[tgt]
                    P = splits.predict_calibrated(ck, Yt, dec, cal)
                    acc[(dec, sent)][i, j] += nrmse(P[ev], Yt[ev], rngf[tgt])
                    per = Yt[ev][:, idx].mean(axis=1)
                    B = np.broadcast_to(per[:, None], Yt[ev].shape)
                    accb[(dec, sent)][i, j] += nrmse(B, Yt[ev], rngf[tgt])
        print(f"  {tag}: {time.time()-t0:.0f}s", flush=True)
    out = {}
    for dec, sent in PANELS:
        out[f"M_{dec}_{sent}"] = acc[(dec, sent)] / len(NETS)
        out[f"B_{dec}_{sent}"] = accb[(dec, sent)] / len(NETS)
    np.savez(HERE / "data" / "twod45_matrices.npz",
             dynamics=np.array([DISP.get(d, d) for d in DYN], dtype=object),
             networks=np.array(NETS), **out)
    print(f"  wrote data/twod45_matrices.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
