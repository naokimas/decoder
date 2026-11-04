# =====================================================================
# SI Section S8: Reconstruction under measurement noise
#
# This script: the noise sweep for the four cases of S1
# Feeds      : Figs. S11 and S12
# Produces   : data/noise.npz
# Runtime    : about a minute
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Measurement noise on the equilibrium data.

The practical monitoring motivation calls for a test with noisy readings.
We repeat the four cases of SI Section S1 -- two within-dynamics and two
cross-dynamics -- with independent Gaussian noise added to every recorded
equilibrium value.

Noise model.  For a dynamics d, let range_i(d) be node i's dynamic range
over the 51 control-parameter values, and let R(d) be the median of
range_i(d) over the N nodes.  At relative noise level eps we add
N(0, (eps R(d))^2) independently to every (node, control-parameter value)
entry of that dynamics.  The standard deviation is therefore the same for
every node, as requested, and eps is dimensionless, which lets the source
and the target dynamics be perturbed comparably even when their ranges
differ by orders of magnitude.  eps is the horizontal axis of every figure.

What sees the noise.  Everything an experimenter would actually measure:
the source equilibria used to pick the sentinels and to train the decoder,
and the target sentinel readings used to calibrate and to drive the
cross-transfer map.  The error is scored against the CLEAN target equilibria
and normalized by the clean target's global range, so the reported n-RMSE
measures how well the true state is recovered from noisy observations rather
than how closely the noise is reproduced.  The mean-prediction baseline is
computed from the same noisy sentinel readings, so r(S->T) compares like
with like.
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path
import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "lib"          # shared modules
BA = HERE.parent / "s09_sentinel_count" / "data"
sys.path.insert(0, str(HERE.parent / "lib")); import paths  # noqa: E402
sys.path.insert(0, str(LIB))

import pipeline_v21 as p21
p21.HERE = paths.EQ_SUBMITTED
import pipeline_unified as pu                                   # noqa: E402
import pipeline_v22 as p22                                      # noqa: E402
import splits                                                   # noqa: E402

CASES = [("dolphin", "SIS",  "SIS",  "dolphin, SIS $\\to$ SIS"),
         ("dolphin", "SIS",  "WC",   "dolphin, SIS $\\to$ WC"),
         ("email",   "MSIR", "MSIR", "email, MSIR1 $\\to$ MSIR1"),
         ("email",   "MSIR", "GEN",  "email, MSIR1 $\\to$ GEN")]
EPS = [0.0] + list(np.logspace(-3, 2, 16))
NDRAW = 20


def median_range(Y):
    return float(np.median(Y.max(axis=0) - Y.min(axis=0)))


def nrmse(pred, true, rng_):
    return float(np.sqrt(np.mean((pred - true) ** 2))) / rng_


def main():
    out = {}
    t0 = time.time()
    for tag, skey, tkey, label in CASES:
        tr, ev = splits.rows_random()
        G = nx.read_edgelist(paths.network_file(tag), nodetype=int)
        N = G.number_of_nodes(); K = int(np.floor(np.log(N)))
        Y_S = p22.load_eq(tag, skey)
        Y_T = Y_S if tkey == skey else p22.load_eq(tag, tkey)
        R_S, R_T = median_range(Y_S), median_range(Y_T)
        rng_T = float(Y_T.max() - Y_T.min())          # clean global range
        key = f"{tag}_{skey}_{tkey}"
        M = np.full((len(EPS), NDRAW), np.nan)
        B = np.full_like(M, np.nan)
        # second variant: the sentinel set is chosen from clean data, as if
        # the monitoring sites had been fixed by a noise-free pilot study
        Mc = np.full_like(M, np.nan); Bc = np.full_like(M, np.nan)
        sent_clean = pu.min_corr_sensors(Y_S[tr], K)
        for ei, eps in enumerate(EPS):
            for d in range(NDRAW if eps > 0 else 1):
                rs = np.random.default_rng(1000 * ei + d)
                Ys = Y_S + rs.normal(0.0, eps * R_S, Y_S.shape)
                Yt = (Ys if tkey == skey
                      else Y_T + rs.normal(0.0, eps * R_T, Y_T.shape))
                for sent, MM, BB in ((pu.min_corr_sensors(Ys[tr], K), M, B),
                                     (sent_clean, Mc, Bc)):
                    ck = splits.train_nn(Ys, sent, 0.0, tr)
                    P = splits.predict_calibrated(ck, Yt, "nn", tr)
                    MM[ei, d] = nrmse(P[ev], Y_T[ev], rng_T)
                    per = Yt[ev][:, sent].mean(axis=1)
                    BB[ei, d] = nrmse(
                        np.broadcast_to(per[:, None], Y_T[ev].shape),
                        Y_T[ev], rng_T)
                    if np.array_equal(np.sort(sent), np.sort(sent_clean)):
                        break          # the two variants coincide
            v = M[ei, ~np.isnan(M[ei])]
            print(f"  {key:20s} eps={eps:<6g} n-RMSE {v.mean()*100:7.3f}%  "
                  f"r {np.nanmean(M[ei]/B[ei]):6.3f}  [{time.time()-t0:.0f}s]",
                  flush=True)
        Mc[np.isnan(Mc)] = M[np.isnan(Mc)]
        Bc[np.isnan(Bc)] = B[np.isnan(Bc)]
        out[key] = dict(M=M, B=B, Mc=Mc, Bc=Bc, R_S=R_S, R_T=R_T, rng_T=rng_T, N=N, n=K,
                        label=label)
        print(f"    {key}: median node range / global range = "
              f"{R_T / rng_T:.4f}", flush=True)
    np.savez(HERE / "data" / "noise.npz", eps=np.array(EPS),
             cases=np.array([f"{a}_{b}_{c}" for a, b, c, _ in CASES],
                            dtype=object),
             **{f"{k}__{f}": v for k, d in out.items() for f, v in d.items()})
    print(f"  wrote data/noise.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
