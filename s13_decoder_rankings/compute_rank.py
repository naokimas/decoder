# =====================================================================
# SI Section S13: Source-decoder rankings
#
# This script: all source decoders, including the pooled ones
# Feeds      : Tables S16 to S19; the bias and alpha01 variants feed Figs. S24 and S25
#              and Tables S20 and S21
# Produces   : data/rank.npz, and, with the 'bias' and 'alpha01'
#              arguments that Section S14 needs, data/rank_bias.npz
#              and data/rank_alpha01.npz
# Runtime    : a few hours
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Source-decoder rankings (SI Section S13, and main-text Fig. 7).

As in fig4_fig5_cross_transfer/compute_matrices.py, there are 17 dynamics,
so the pooled decoder over all of them is C17, and the target side is out of
sample: the input map and the affine (a, b) are fitted on the 41
interspersed calibration rows, and every cell is scored on the remaining 10.

Sources evaluated, for each of the six (decoder, sentinel) combinations and
each of the 14 networks:

  * the 17 per-dynamics decoders;
  * C11, trained on the pooled data of the 11 transferable dynamics;
  * C17, trained on the pooled data of all 17;
  * C11L1, the leave-one-dynamics-out pool over the 11: for target t the
    decoder is trained on the other 10;
  * C17L1, the same over all 17.

A pooled decoder normalizes each member dynamics to [0, 1] on its own
training rows and stacks the blocks, so the pooled source has y_min = 0 and
y_max = 1 by construction, exactly as in the paper's COMBINED decoder.
Sentinels for a pooled source are picked from the pooled training matrix.

Three decoder specifications are supported, selected by a command-line
argument, so that SI Section S14 can compare them on the same footing:

  main      the specification of the main text: bias-free NN, ridge 1e-2
  bias      the NN with its bias terms restored (the nn panels only)
  alpha01   the linear ridge decoder at alpha = 1e-1 (the lin panels only)

Writes data/rank.npz, data/rank_bias.npz or data/rank_alpha01.npz.
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
_s = importlib.util.spec_from_file_location("leq", LIB / "load_equilibria.py")
leq = importlib.util.module_from_spec(_s); _s.loader.exec_module(leq)

import pipeline_unified as pu                                   # noqa: E402
import splits                                                   # noqa: E402

DYN = ["SIS", "SIR", "MSIS", "MSIS2", "MSIR", "MSIR2", "DW", "MUT", "GEN",
       "WC", "gLV", "NM", "EUT", "EUT2", "VEG", "VEG2", "LDW"]
DISP = {"MSIS": "MSIS1", "MSIR": "MSIR1", "EUT": "EUT1", "VEG": "VEG1"}
TRANS = DYN[:11]                       # the 11 transferable dynamics
PANELS = [("nn", "minc"), ("nn", "qr"), ("nn", "random"),
          ("lin", "minc"), ("lin", "qr"), ("lin", "random")]
REG = {"nn": 0.0, "lin": 0.01}
NETS = leq.NETWORKS

VARIANT = "main"          # set from argv in main()
VARIANTS = {"main": {"panels": PANELS, "bias": False, "alpha": 0.01},
            "bias": {"panels": PANELS[:3], "bias": True, "alpha": 0.01},
            "alpha01": {"panels": PANELS[3:], "bias": False, "alpha": 0.1}}
(HERE / "data").mkdir(exist_ok=True)


def nrmse(pred, true, full_range):
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return rmse / full_range if full_range > 0 else np.nan


def pooled_matrix(members, eq, tr):
    """Stack the member dynamics, each scaled to [0, 1] on its own rows."""
    blocks = []
    for s in members:
        Y = eq[s][tr]
        lo = float(Y.min()); hi = float(Y.max())
        blocks.append((Y - lo) / max(hi - lo, 1e-12))
    return np.vstack(blocks)


def sentinels(Y, sent, K, rand_idx):
    if sent == "qr":
        return pu.qr_pivot_sensors(Y, K)
    if sent == "minc":
        return pu.min_corr_sensors(Y, K)
    return rand_idx.copy()


def train_nn_bias(Y_full, sensor_idx, weight_decay, train_rows):
    """splits.train_nn with the paper's bias-carrying ShallowDecoder.

    Identical to splits.train_nn except for the module class and for the
    absence of the _no_bias flag, which is what makes sweep's _decode_fn
    rebuild the biased network at deployment.
    """
    import torch
    torch.manual_seed(pu.SEED)
    Y_train = Y_full[train_rows]
    N = Y_full.shape[1]
    y_min = float(Y_train.min()); y_max = float(Y_train.max())
    y_range = max(y_max - y_min, 1e-12)
    Xtr = (Y_train[:, sensor_idx] - y_min) / y_range
    Ytr = (Y_train - y_min) / y_range
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
    model = pu.ShallowDecoder(len(sensor_idx), pu.HIDDEN, N)
    opt = torch.optim.Adam(model.parameters(), lr=pu.LR,
                           weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    for _ in range(pu.EPOCHS):
        opt.zero_grad(); loss_fn(model(Xtr_t), Ytr_t).backward(); opt.step()
    model.eval()
    return {"model_state_dict": model.state_dict(),
            "k_sentinels": len(sensor_idx), "hidden": pu.HIDDEN, "n_nodes": N,
            "sensor_idx": sensor_idx,
            "y_min": y_min, "y_max": y_max, "y_range": y_range,
            "src_sent_min": float(Y_train[:, sensor_idx].min()),
            "src_sent_max": float(Y_train[:, sensor_idx].max())}


def train(Y, idx, dec, train_rows):
    cfg = VARIANTS[VARIANT]
    if dec == "nn":
        fn = train_nn_bias if cfg["bias"] else splits.train_nn
        return fn(Y, idx, REG["nn"], train_rows)
    return splits.fit_linear(Y, idx, cfg["alpha"], train_rows)


def score(ck, Yt, dec, tr, ev, rngf_t):
    P = splits.predict_calibrated(ck, Yt, dec, tr)
    m = nrmse(P[ev], Yt[ev], rngf_t)
    per = Yt[ev][:, ck["sensor_idx"]].mean(axis=1)
    B = np.broadcast_to(per[:, None], Yt[ev].shape)
    return m, nrmse(B, Yt[ev], rngf_t)


def main():
    global VARIANT
    if len(sys.argv) > 1:
        VARIANT = sys.argv[1]
    panels = VARIANTS[VARIANT]["panels"]
    print(f"variant {VARIANT}: panels {panels}", flush=True)
    n = len(DYN); nt = len(TRANS)
    keys = ["single", "c11", "c17", "c11l1", "c17l1"]
    shape = {"single": (n, n), "c11": (n,), "c17": (n,),
             "c11l1": (nt,), "c17l1": (n,)}
    M = {p: {k: np.zeros(shape[k]) for k in keys} for p in panels}
    B = {p: {k: np.zeros(shape[k]) for k in keys} for p in panels}
    tr, ev = splits.rows_random()
    t0 = time.time()
    for tag in NETS:
        G = nx.read_edgelist(leq.network_file(tag),
                             nodetype=int)
        N = len(G.nodes()); K = int(np.floor(np.log(N)))
        eq = {s: leq.load_eq(tag, s) for s in DYN}
        rngf = {s: float(eq[s].max() - eq[s].min()) for s in DYN}
        RAND = np.sort(np.random.default_rng(2026).choice(N, size=K,
                                                          replace=False))
        for dec, sent in panels:
            p = (dec, sent)
            # ---- the 17 per-dynamics sources ----
            for i, src in enumerate(DYN):
                idx = sentinels(eq[src][tr], sent, K, RAND)
                ck = train(eq[src], idx, dec, tr)
                for j, tgt in enumerate(DYN):
                    m, b = score(ck, eq[tgt], dec, tr, ev, rngf[tgt])
                    M[p]["single"][i, j] += m
                    B[p]["single"][i, j] += b
            # ---- the two full pools ----
            for key, members in (("c11", TRANS), ("c17", DYN)):
                Yp = pooled_matrix(members, eq, tr)
                rows = np.arange(Yp.shape[0])
                idx = sentinels(Yp, sent, K, RAND)
                ck = train(Yp, idx, dec, rows)
                for j, tgt in enumerate(DYN):
                    m, b = score(ck, eq[tgt], dec, tr, ev, rngf[tgt])
                    M[p][key][j] += m
                    B[p][key][j] += b
            # ---- the two leave-one-dynamics-out pools ----
            for key, pool_set in (("c11l1", TRANS), ("c17l1", DYN)):
                for j, tgt in enumerate(pool_set):
                    members = [s for s in pool_set if s != tgt]
                    Yp = pooled_matrix(members, eq, tr)
                    rows = np.arange(Yp.shape[0])
                    idx = sentinels(Yp, sent, K, RAND)
                    ck = train(Yp, idx, dec, rows)
                    m, b = score(ck, eq[tgt], dec, tr, ev, rngf[tgt])
                    M[p][key][j] += m
                    B[p][key][j] += b
        print(f"  {tag}: {time.time()-t0:.0f}s", flush=True)
    out = {}
    for dec, sent in panels:
        for k in keys:
            out[f"M_{dec}_{sent}_{k}"] = M[(dec, sent)][k] / len(NETS)
            out[f"B_{dec}_{sent}_{k}"] = B[(dec, sent)][k] / len(NETS)
    fn = "rank.npz" if VARIANT == "main" else f"rank_{VARIANT}.npz"
    np.savez(HERE / "data" / fn,
             dynamics=np.array([DISP.get(d, d) for d in DYN], dtype=object),
             transferable=np.array([DISP.get(d, d) for d in TRANS],
                                   dtype=object),
             eval_rows=ev, cal_rows=tr, **out)
    print(f"  wrote data/{fn}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
