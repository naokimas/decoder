# =====================================================================
# SI Section S1: Node-wise reconstruction error
#
# This script: node-wise n-RMSE for the four cases
# Feeds      : Figs. S1 and S2
# Produces   : data/nodewise.npz
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Node-wise n-RMSE for the four cases of SI Section S1.

The decoder, the sentinel-selection method, the training split and the
calibration protocol are all exactly those behind Figs. 2 and 3 of the main
text: a bias-free shallow NN decoder, MinCorr sentinels, the paper's own
41/10 interspersed split, and an affine cross-transfer map fitted on the 41
calibration rows.  Nothing here is new numerics; only the error is
disaggregated.

Node-wise n-RMSE, for node i, is Eq. (n-RMSE) of the Methods with the average
over nodes removed and the average over the 10 evaluation control-parameter
values kept:

    n-RMSE_i = sqrt( (1/|K_eval|) sum_{k in K_eval} (xhat_ik - x_ik)^2 )
               / ( max_{i,k} x_ik - min_{i,k} x_ik ).

The denominator is the target's global range, unchanged from the Methods, so
the node-wise values are on the same scale as the reported aggregate and
satisfy  n-RMSE = sqrt( mean_i n-RMSE_i^2 ).
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path
import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "lib"          # shared modules
sys.path.insert(0, str(HERE.parent / "lib")); import paths  # noqa: E402
sys.path.insert(0, str(LIB))

import pipeline_v21 as p21
p21.HERE = paths.EQ_SUBMITTED
import pipeline_unified as pu                                   # noqa: E402
import pipeline_v22 as p22                                      # noqa: E402
import splits                                                   # noqa: E402


# (net tag, source key, target key, display names)
CASES = [
    ("dolphin", "SIS",  "SIS",  "SIS",   "SIS"),
    ("dolphin", "SIS",  "WC",   "SIS",   "WC"),
    ("email",   "MSIR", "MSIR", "MSIR1", "MSIR1"),
    ("email",   "MSIR", "GEN",  "MSIR1", "GEN"),
]


def node_nrmse(Y_pred, Y_true, ev):
    rng_ = float(Y_true.max() - Y_true.min())
    se = (Y_pred[ev] - Y_true[ev]) ** 2          # (|K_eval|, N)
    return np.sqrt(se.mean(axis=0)) / rng_       # (N,)


def baseline_node_nrmse(Y_true, sent_idx, ev):
    """Mean-prediction baseline, disaggregated the same way."""
    rng_ = float(Y_true.max() - Y_true.min())
    per_row = Y_true[ev][:, sent_idx].mean(axis=1)
    base = np.broadcast_to(per_row[:, None], Y_true[ev].shape)
    return np.sqrt(((base - Y_true[ev]) ** 2).mean(axis=0)) / rng_


def main():
    out = {}
    for tag, skey, tkey, sname, tname in CASES:
        tr, ev = splits.rows_random()
        G = nx.read_edgelist(paths.network_file(tag), nodetype=int)
        N = G.number_of_nodes()
        deg = np.array([G.degree(i) for i in range(N)], dtype=float)
        K = int(np.floor(np.log(N)))

        Y_S = p22.load_eq(tag, skey)
        sent = pu.min_corr_sensors(Y_S[tr], K)
        ckpt = splits.train_nn(Y_S, sent, 0.0, tr)

        Y_T = Y_S if tkey == skey else p22.load_eq(tag, tkey)
        Y_pred = splits.predict_calibrated(ckpt, Y_T, "nn", tr)

        nw = node_nrmse(Y_pred, Y_T, ev)
        node_rng = Y_T.max(axis=0) - Y_T.min(axis=0)   # each node's own range
        bw = baseline_node_nrmse(Y_T, sent, ev)
        agg = float(np.sqrt((nw ** 2).mean()))
        agg_b = float(np.sqrt((bw ** 2).mean()))

        key = f"{tag}_{sname}_{tname}"
        out[key] = dict(nodewise=nw, baseline=bw, degree=deg,
                        node_range=node_rng,
                        global_range=float(Y_T.max() - Y_T.min()),
                        sentinel=np.asarray(sent, dtype=int),
                        eval_rows=ev, agg=agg, ratio=agg / agg_b, N=N)
        print(f"  {tag:8s} {sname:6s} -> {tname:6s}  N={N:5d}  n={K}  "
              f"n-RMSE={agg*100:7.3f}%  r={agg/agg_b:5.3f}  "
              f"node-wise range [{nw.min()*100:.3f}%, {nw.max()*100:.3f}%]")

    np.savez(HERE / "data" / "nodewise.npz",
             **{f"{k}__{f}": v for k, d in out.items() for f, v in d.items()})
    print(f"  wrote {HERE/'data'/'nodewise.npz'}")


if __name__ == "__main__":
    main()
