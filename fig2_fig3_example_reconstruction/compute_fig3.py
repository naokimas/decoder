# =====================================================================
# Main text Figs. 2 and 3: reconstruction on the dolphin network
#
# This script: the SIS -> WC cross-transfer on dolphin, out of sample
# Feeds      : Fig. 3
# Produces   : data/fig3_dolphin_sis_wc.npz
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Data for Fig. 3 of the main text: SIS -> WC on the dolphin network, with
the SIS decoder and the pooled C11L1 decoder.

On the SOURCE side, both decoders use the 41/10 interspersed split of
splits.rows_random().  The C11L1 decoder is built exactly as in
s13_decoder_rankings/compute_rank.py, which gives the C11L1 numbers of
Fig. 7: each of the 10 member dynamics is min-max scaled to [0, 1] on its
own 41 training rows, the blocks are stacked, the MinCorr sentinels are
picked from that pooled matrix, and the decoder is trained on the same
pooled matrix.

On the TARGET side, the target sweep is split in the same way as the source
sweep: 41 control-parameter values,
drawn uniformly at random and therefore interspersed, are used to fit the
input map and the affine constants (a, b), and the error is reported on the
10 untouched values.

No numerical integration is performed; the script reads the equilibrium
CSVs.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path
import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "lib"
sys.path.insert(0, str(HERE.parent / "lib")); import paths  # noqa: E402
sys.path.insert(0, str(LIB))

import pipeline_v21 as p21
p21.HERE = paths.EQ_SUBMITTED
import pipeline_unified as pu                                   # noqa: E402
import pipeline_v22 as p22                                      # noqa: E402
import splits                                                   # noqa: E402

TAG = "dolphin"
HELD_OUT = "WC"
TRANSFERABLE_11 = ["SIS", "SIR", "MSIS", "MSIS2", "MSIR", "MSIR2",
                   "DW", "MUT", "GEN", "WC", "gLV"]
OUT = HERE / "data" / "fig3_dolphin_sis_wc.npz"


def metrics(Y_pred, Y_true, sent_idx, ev):
    """n-RMSE on the evaluation rows, normalized by the target's FULL range.

    The full range is taken over all M rows and all N nodes, which is the
    manuscript's definition; only the rows entering the RMSE change.
    """
    rng_ = float(Y_true.max() - Y_true.min())
    rmse = float(np.sqrt(np.mean((Y_pred[ev] - Y_true[ev]) ** 2)))
    nrmse = rmse / rng_ if rng_ > 0 else float("nan")
    per_row = Y_true[ev][:, sent_idx].mean(axis=1)
    base = np.broadcast_to(per_row[:, None], Y_true[ev].shape)
    rmse_b = float(np.sqrt(np.mean((base - Y_true[ev]) ** 2)))
    nrmse_b = rmse_b / rng_ if rng_ > 0 else float("nan")
    return nrmse, (nrmse / nrmse_b if nrmse_b > 0 else float("nan"))


def main():
    tr, ev = splits.rows_random()             # 41 / 10, the paper's own split
    G = nx.read_edgelist(paths.network_file(TAG), nodetype=int)
    K = int(np.floor(np.log(len(G.nodes()))))

    # ---- source decoder 1: per-dynamics SIS, exactly as in Fig. 2 --------
    Y_sis = p22.load_eq(TAG, "SIS")
    sis_sent = pu.min_corr_sensors(Y_sis[tr], K)
    sis_ckpt = splits.train_nn(Y_sis, sis_sent, 0.0, tr)

    # ---- source decoder 2: C11L1, pooled over the 10 transferable --------
    # as in s13_decoder_rankings/compute_rank.py: each member scaled to
    # [0, 1] on its own 41 training rows, stacked, sentinels picked from
    # and decoder trained on that same pooled matrix
    pool = [s for s in TRANSFERABLE_11 if s != HELD_OUT]
    blocks = []
    for s in pool:
        Y = p22.load_eq(TAG, s)[tr]
        lo, hi = float(Y.min()), float(Y.max())
        blocks.append((Y - lo) / max(hi - lo, 1e-12))
    Y_pool = np.vstack(blocks)
    c11l1_sent = pu.min_corr_sensors(Y_pool, K)
    c11l1_ckpt = splits.train_nn(Y_pool, c11l1_sent, 0.0,
                                 np.arange(Y_pool.shape[0]))

    # ---- deploy both on dolphin WC, calibrating on the 41 only -----------
    arr = np.loadtxt(p21.HERE / f"{TAG}_wc_equilibrium_data.csv",
                     delimiter=",", skiprows=1)
    ps, Y_true = arr[:, 0], arr[:, 1:]
    Y_sis_pred = splits.predict_calibrated(sis_ckpt, Y_true, "nn", tr)
    Y_c11_pred = splits.predict_calibrated(c11l1_ckpt, Y_true, "nn", tr)

    sis_n, sis_r = metrics(Y_sis_pred, Y_true, sis_sent, ev)
    c11_n, c11_r = metrics(Y_c11_pred, Y_true, c11l1_sent, ev)
    print(f"  scored on the {len(ev)} held-out control-parameter values "
          f"{list(ev)}")
    print(f"  SIS decoder : n-RMSE = {sis_n*100:.3f}% , r(S->T) = {sis_r:.3f}")
    print(f"  C11L1       : n-RMSE = {c11_n*100:.3f}% , r(S->T) = {c11_r:.3f}")

    OUT.parent.mkdir(exist_ok=True)
    np.savez(OUT, ps=ps, Y_true=Y_true,
             Y_pred_sis_wc=Y_sis_pred, Y_pred_coo_wc=Y_c11_pred,
             sis_sentinel=np.asarray(sis_sent, dtype=int),
             coo_sentinel=np.asarray(c11l1_sent, dtype=int),
             cal_rows=tr, eval_rows=ev,
             sis_nrmse=sis_n, sis_ratio=sis_r,
             coo_nrmse=c11_n, coo_ratio=c11_r)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
