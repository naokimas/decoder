# =====================================================================
# SI Section S9: Robustness to the number of sentinel nodes
#
# This script: the n x N sweep on the BA networks
# Feeds      : Figs. S14 to S18
# Produces   : data/scale.npz
# Runtime    : about three minutes
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""How many sentinels does a larger network need?

For the BA size sweep (N = 32 ... 10000) we repeat the cross-transfer
experiment of Figs. 4 and 5 while varying the number of sentinels n from
NMIN to NMAX, and ask how n must grow with N to hold the error fixed.  The
paper's rule is n = floor(ln N), so the question is whether the iso-error
contour in the (N, n) plane is logarithmic.

Everything except n and the network follows the main text: the 41/10
interspersed split on both sides, the affine cross-transfer map fitted on
the 41 calibration values, and the error scored on the 10 held out.  The
LDW set used here is the heterogeneous-initial-condition one; the
uniform-initial-condition branch is not part of the 17 dynamics.

Writes data/scale.npz with M[size, n, dec, sent, 17, 17] (n-RMSE) and the
matching mean-prediction baseline B, from which r(S->T) = M / B.
"""
import os, sys, time, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path
import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib")); import paths  # noqa: E402
EQ = paths.EQ_REVISION        # equilibrium CSVs; see ../README.md
LIB = HERE.parent / "lib"          # shared modules
# The equilibrium CSVs of the Barabasi-Albert sweep are in
# equilibria_revision/ba_size_sweep/ of the input data (see ../README.md);
# its edge lists are in networks/ba_size_sweep/ of this repository.
BAEQ = EQ / "ba_size_sweep"
sys.path.insert(0, str(LIB))

import pipeline_unified as pu                                   # noqa: E402
import splits                                                   # noqa: E402
import build_v23_manuscript as bv23                             # noqa: E402

DYN = ["SIS", "SIR", "MSIS", "MSIS2", "MSIR", "MSIR2", "DW", "MUT", "GEN",
       "WC", "gLV", "NM", "EUT", "EUT2", "VEG", "VEG2", "LDW"]
DISP = {"MSIS": "MSIS1", "MSIR": "MSIR1", "EUT": "EUT1", "VEG": "VEG1"}
CSV = dict(bv23.CSV_NAMES); CSV["LDW"] = "ldw"
TAGS = [("ba1p5", 32), ("ba2p0", 100), ("ba2p5", 316),
        ("ba3p0", 1000), ("ba3p5", 3162), ("ba4p0", 10000)]
DECS = ["nn", "lin"]
SENTS = ["minc", "qr"]                 # non-random only, as requested
REG = {"nn": 0.0, "lin": 0.01}
NMIN, NMAX = 2, 10
NS = list(range(NMIN, NMAX + 1))


def load_eq(tag, s):
    a = np.loadtxt(BAEQ / f"{tag}_{CSV[s]}_equilibrium_data.csv",
                   delimiter=",", skiprows=1)
    return a[:, 1:]


def nrmse(pred, true, rng_):
    return float(np.sqrt(np.mean((pred - true) ** 2))) / rng_ if rng_ > 0 \
        else np.nan


def one_block(eq, rngf, tr, ev, n, dec, sent):
    """17 x 17 n-RMSE and baseline for one (n, decoder, sentinel) cell."""
    nd = len(DYN)
    M = np.zeros((nd, nd)); B = np.zeros((nd, nd))
    sidx = {s: (pu.qr_pivot_sensors(eq[s][tr], n) if sent == "qr"
                else pu.min_corr_sensors(eq[s][tr], n)) for s in DYN}
    for i, src in enumerate(DYN):
        idx = sidx[src]
        ck = (splits.train_nn(eq[src], idx, REG[dec], tr) if dec == "nn"
              else splits.fit_linear(eq[src], idx, REG[dec], tr))
        for j, tgt in enumerate(DYN):
            Yt = eq[tgt]
            P = splits.predict_calibrated(ck, Yt, dec, tr)
            M[i, j] = nrmse(P[ev], Yt[ev], rngf[tgt])
            per = Yt[ev][:, idx].mean(axis=1)
            B[i, j] = nrmse(np.broadcast_to(per[:, None], Yt[ev].shape),
                            Yt[ev], rngf[tgt])
    return M, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="time one block at the largest size and stop")
    ap.add_argument("--tags", default="")
    ap.add_argument("--out", default="scale.npz")
    a = ap.parse_args()

    tags = [t for t in TAGS if not a.tags or t[0] in a.tags.split(",")]
    tr, ev = splits.rows_random()

    if a.probe:
        tag, N = TAGS[-1]
        t0 = time.time()
        eq = {s: load_eq(tag, s) for s in DYN}
        rngf = {s: float(eq[s].max() - eq[s].min()) for s in DYN}
        print(f"  load {tag} (N={N}): {time.time()-t0:.1f}s")
        for dec in DECS:
            t1 = time.time()
            one_block(eq, rngf, tr, ev, 9, dec, "minc")
            print(f"  one block, {tag}, dec={dec}: {time.time()-t1:.1f}s")
        return

    nd = len(DYN)
    M = np.full((len(tags), len(NS), len(DECS), len(SENTS), nd, nd), np.nan)
    B = np.full_like(M, np.nan)
    t0 = time.time()
    for ti, (tag, N) in enumerate(tags):
        eq = {s: load_eq(tag, s) for s in DYN}
        rngf = {s: float(eq[s].max() - eq[s].min()) for s in DYN}
        for ni, n in enumerate(NS):
            if n > N:
                continue
            for di, dec in enumerate(DECS):
                for si, sent in enumerate(SENTS):
                    M[ti, ni, di, si], B[ti, ni, di, si] = one_block(
                        eq, rngf, tr, ev, n, dec, sent)
            print(f"    {tag} n={n}: {time.time()-t0:.0f}s", flush=True)
        print(f"  {tag} done: {time.time()-t0:.0f}s", flush=True)
    np.savez(HERE / "data" / a.out, M=M, B=B,
             tags=np.array([t for t, _ in tags], dtype=object),
             sizes=np.array([N for _, N in tags]),
             ns=np.array(NS), decs=np.array(DECS, dtype=object),
             sents=np.array(SENTS, dtype=object),
             dynamics=np.array([DISP.get(d, d) for d in DYN], dtype=object),
             cal_rows=tr, eval_rows=ev)
    print(f"  wrote data/{a.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
