# =====================================================================
# Main text Figs. 2 and 3: reconstruction on the dolphin network
#
# This script: the SIS -> SIS reconstruction on dolphin, out of sample
# Feeds      : Fig. 2
# Produces   : data/fig2_dolphin_sis_sis.npz
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Data for Fig. 2 of the main text: SIS -> SIS on the dolphin network.

The decoder is trained on 41 of the 51 control-parameter values (the
interspersed split of lib/splits.py), the sentinels are picked from those 41
values, and the affine calibration constants are fitted on the same 41
values.  The error is reported on the remaining 10 values, which are used
neither for training nor for calibration, as for Fig. 3 (compute_fig3.py).

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
OUT = HERE / "data" / "fig2_dolphin_sis_sis.npz"


def metrics(Y_pred, Y_true, sent_idx, ev):
    """n-RMSE on the evaluation rows, normalized by the target's FULL range."""
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

    Y_sis = p22.load_eq(TAG, "SIS")
    sis_sent = pu.min_corr_sensors(Y_sis[tr], K)
    ckpt = splits.train_nn(Y_sis, sis_sent, 0.0, tr)

    arr = np.loadtxt(p21.HERE / f"{TAG}_sis_equilibrium_data.csv",
                     delimiter=",", skiprows=1)
    ps, Y_true = arr[:, 0], arr[:, 1:]
    Y_pred = splits.predict_calibrated(ckpt, Y_true, "nn", tr)

    nrmse, ratio = metrics(Y_pred, Y_true, sis_sent, ev)
    print(f"  sentinels (MinCorr, n={K}): {sorted(int(s) for s in sis_sent)}")
    print(f"  scored on the {len(ev)} held-out control-parameter values "
          f"{list(ev)}")
    print(f"  SIS decoder : n-RMSE = {nrmse*100:.3f}% , "
          f"r(S->T) = {ratio:.3f}")

    OUT.parent.mkdir(exist_ok=True)
    np.savez(OUT, ps=ps, Y_true=Y_true, Y_pred=Y_pred,
             sentinel=np.asarray(sis_sent, dtype=int),
             cal_rows=tr, eval_rows=ev, nrmse=nrmse, ratio=ratio)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
