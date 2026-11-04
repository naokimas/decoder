# =====================================================================
# Main text Figs. 4 and 5: the cross-transfer matrices
#
# This script: the 17 x 17 n-RMSE and baseline matrices, averaged over 14 networks
# Feeds      : Figs. 4 and 5, and eight SI scripts (see ../shared_data/README.md)
# Produces   : ../shared_data/fig45_matrices.npz
# Runtime    : hours
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""The cross-transfer matrices behind Figs. 4 and 5 of the main text.

Every source decoder is applied to every target dynamics over the 17
dynamics (LDW last, after VEG2), so the matrices are 17 x 17.  The source
decoder is trained on 41 of the 51 control-parameter values (the
interspersed split of lib/splits.py).  On the target side, the input map and
the affine constants (a, b) are fitted on the same 41 values, and every cell
is scored on the remaining 10.

No numerical integration is performed; the script reads the equilibrium
CSVs.

Writes ../shared_data/fig45_matrices.npz with, for each of the six (decoder,
sentinel) panels, the 17 x 17 n-RMSE matrix and the matching
mean-prediction baseline matrix, each averaged over the 14 networks.
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
SHARED = HERE.parent / "shared_data"   # the matrices many sections read
LIB = HERE.parent / "lib"
sys.path.insert(0, str(HERE.parent / "lib")); import paths  # noqa: E402
sys.path.insert(0, str(LIB))
_s = importlib.util.spec_from_file_location(
    "leq", LIB / "load_equilibria.py")
leq = importlib.util.module_from_spec(_s); _s.loader.exec_module(leq)

import pipeline_unified as pu                                   # noqa: E402
import splits                                                   # noqa: E402

# the paper's 16, in the paper's display order, with LDW appended
DYN = ["SIS", "SIR", "MSIS", "MSIS2", "MSIR", "MSIR2", "DW", "MUT", "GEN",
       "WC", "gLV", "NM", "EUT", "EUT2", "VEG", "VEG2", "LDW"]
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
    cnt = np.zeros((n, n))
    tr, ev = splits.rows_random()          # 41 / 10, interspersed, both sides
    t0 = time.time()
    for tag in NETS:
        G = nx.read_edgelist(leq.network_file(tag),
                             nodetype=int)
        N = len(G.nodes()); K = int(np.floor(np.log(N)))
        eq = {s: leq.load_eq(tag, s) for s in DYN}
        rngf = {s: float(eq[s].max() - eq[s].min()) for s in DYN}
        rnd = np.random.default_rng(2026)
        RAND = np.sort(rnd.choice(N, size=K, replace=False))
        for dec, sent in PANELS:
            sidx = {}
            for s in DYN:
                Ytr = eq[s][tr]
                sidx[s] = (pu.qr_pivot_sensors(Ytr, K) if sent == "qr" else
                           pu.min_corr_sensors(Ytr, K) if sent == "minc"
                           else RAND.copy())
            for i, src in enumerate(DYN):
                idx = sidx[src]
                ck = (splits.train_nn(eq[src], idx, REG[dec], tr)
                      if dec == "nn"
                      else splits.fit_linear(eq[src], idx, REG[dec], tr))
                for j, tgt in enumerate(DYN):
                    Yt = eq[tgt]
                    P = splits.predict_calibrated(ck, Yt, dec, tr)
                    acc[(dec, sent)][i, j] += nrmse(P[ev], Yt[ev], rngf[tgt])
                    per = Yt[ev][:, idx].mean(axis=1)
                    B = np.broadcast_to(per[:, None], Yt[ev].shape)
                    accb[(dec, sent)][i, j] += nrmse(B, Yt[ev], rngf[tgt])
        cnt += 1
        print(f"  {tag}: {time.time()-t0:.0f}s", flush=True)
    out = {}
    for dec, sent in PANELS:
        out[f"M_{dec}_{sent}"] = acc[(dec, sent)] / len(NETS)
        out[f"B_{dec}_{sent}"] = accb[(dec, sent)] / len(NETS)
    np.savez(SHARED / "fig45_matrices.npz",
             dynamics=np.array([DISP.get(d, d) for d in DYN], dtype=object),
             eval_rows=ev, cal_rows=tr, **out)
    print(f"  wrote ../shared_data/fig45_matrices.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
