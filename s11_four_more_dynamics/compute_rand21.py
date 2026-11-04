# =====================================================================
# SI Section S11: Cross-transfer classes with four additional dynamics
#
# This script: the 21 x 21 cross-transfer matrices
# Feeds      : Fig. S22 and Table S14, for the random-sentinel panels
# Produces   : rand21_matrices.npz
# Runtime    : about an hour
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""The two uniformly random sentinel panels over all 21 dynamics.

The MinCorr and QR panels over the 21 dynamics are the interspersed /
interspersed cell of s04_interp_vs_extrap/compute_grid9.py, which
analyze21.py reads.  This script adds the two random-sentinel panels under
exactly the same protocol: interspersed 41/10 split on both the source and
the target side, the manuscript's least-squares output calibration, the
n-RMSE normalized by the target's full range, and the mean-prediction
baseline on the same rows with the same sentinel set.  The random sentinel
set is drawn once per network from seed 2026, as in the Fig. 4 and Fig. 5
computation.
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
_s = importlib.util.spec_from_file_location("leq", LIB / "load_equilibria.py")
leq = importlib.util.module_from_spec(_s); _s.loader.exec_module(leq)
import pipeline_unified as pu                                   # noqa: E402
import splits                                                   # noqa: E402

DYN, NETS = leq.DYNAMICS, leq.NETWORKS
REG = {"nn": 0.0, "lin": 0.01}
tr, ev = splits.rows_random()


def main():
    n = len(DYN)
    acc = {d: np.zeros((n, n)) for d in REG}
    accb = {d: np.zeros((n, n)) for d in REG}
    t0 = time.time()
    for tag in NETS:
        G = nx.read_edgelist(leq.network_file(tag),
                             nodetype=int)
        N = len(G.nodes()); K = int(np.floor(np.log(N)))
        eq = {s: leq.load_eq(tag, s) for s in DYN}
        rngf = {s: float(eq[s].max() - eq[s].min()) for s in DYN}
        RAND = np.sort(np.random.default_rng(2026).choice(N, size=K,
                                                          replace=False))
        for dec in REG:
            for i, src in enumerate(DYN):
                ck = (splits.train_nn(eq[src], RAND, REG[dec], tr)
                      if dec == "nn"
                      else splits.fit_linear(eq[src], RAND, REG[dec], tr))
                for j, tgt in enumerate(DYN):
                    Yt = eq[tgt]
                    P = splits.predict_calibrated(ck, Yt, dec, tr)
                    r = float(np.sqrt(np.mean((P[ev] - Yt[ev]) ** 2)))
                    acc[dec][i, j] += r / rngf[tgt]
                    per = Yt[ev][:, RAND].mean(axis=1)
                    Bh = np.broadcast_to(per[:, None], Yt[ev].shape)
                    rb = float(np.sqrt(np.mean((Bh - Yt[ev]) ** 2)))
                    accb[dec][i, j] += rb / rngf[tgt]
        print(f"  {tag}: {time.time()-t0:.0f}s", flush=True)
    out = {}
    for dec in REG:
        out[f"M_{dec}_random"] = acc[dec] / len(NETS)
        out[f"B_{dec}_random"] = accb[dec] / len(NETS)
    np.savez(HERE / "data" / "rand21_matrices.npz",
             dynamics=np.array(DYN, dtype=object),
             networks=np.array(NETS), **out)
    print(f"  wrote rand21_matrices.npz ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
