# =====================================================================
# SI Section S7: Variability over networks and random factors
#
# This script: repeats the cross-transfer experiment under each random factor
# Feeds      : Figs. S9 and S10, Tables S9 and S10; rep_base.npz also feeds Section S10
# Produces   : data/rep_*.npz
# Runtime    : many hours
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Dispersion of the Fig. 4 and Fig. 5 cells: networks versus other randomness.

Every cell of Figs. 4 and 5 is an average over the 14 networks of a single
seeded realization of everything else.  This script re-runs those matrices
many times, turning on one source of randomness at a time, so that the
standard deviation contributed by each can be compared with the standard
deviation across networks.

Modes
  base    the paper's own seeds, one replicate.  14 samples per cell.
  nn      R replicates, the NN initialization seed varying.  NN panels only.
  split   R replicates, the interspersed split redrawn with a new seed each
          time.  As in the main text, the same split serves both sides: the
          source decoder is trained on the 41 rows, the target maps are
          calibrated on the same 41, and both are scored on the other 10.
          (An earlier version drew the target split independently, which
          let evaluation rows of a within-dynamics cell enter training.)
  sent    R replicates, the uniformly random sentinel set redrawn.  The two
          random-sentinel panels only.
  hetero  R replicates, the heterogeneous node parameters of NM, EUT1, EUT2,
          VEG1 and VEG2 redrawn (see gen_hetero.py).  Replicate 0 is the
          published draw and replicates 1 to R-1 read rep01 to rep<R-1>, so
          the R samples are R different parameter draws.

Writes data/rep_<mode>.npz with, for each panel, arrays of shape
(R, n_net, n_dyn, n_dyn) holding the n-RMSE and the mean-prediction baseline.
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
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib")); import paths  # noqa: E402
EQ = paths.EQ_REVISION        # equilibrium CSVs; see ../README.md
LIB = HERE.parent / "lib"          # shared modules
sys.path.insert(0, str(LIB))
_s = importlib.util.spec_from_file_location("leq", LIB / "load_equilibria.py")
leq = importlib.util.module_from_spec(_s); _s.loader.exec_module(leq)

import pipeline_unified as pu                                   # noqa: E402
import splits                                                   # noqa: E402
import test_zero_anchor as tz                                   # noqa: E402

DYN = ["SIS", "SIR", "MSIS", "MSIS2", "MSIR", "MSIR2", "DW", "MUT", "GEN",
       "WC", "gLV", "NM", "EUT", "EUT2", "VEG", "VEG2", "LDW"]
DISP = {"MSIS": "MSIS1", "MSIR": "MSIR1", "EUT": "EUT1", "VEG": "VEG1"}
HETERO = {"NM", "EUT", "EUT2", "VEG", "VEG2"}       # heterogeneous node params
STEMS = {"NM": "noymeir", "EUT": "eutroph", "EUT2": "eutroph2",
         "VEG": "vegturb", "VEG2": "vegturb2"}
ALL_PANELS = [("nn", "minc"), ("nn", "qr"), ("nn", "random"),
              ("lin", "minc"), ("lin", "qr"), ("lin", "random")]
REG = {"nn": 0.0, "lin": 0.01}
NETS = leq.NETWORKS
# The heterogeneous-parameter replicates cover the eleven smallest networks.
# Reactome, the twelfth, is out of reach: NM there costs about 2.5 hours per
# sweep even with a sparse Jacobian, i.e. a day of compute for R = 10.  The
# comparison for this factor is made on these eleven, against a network-only
# baseline on the same eleven.
HET_NETS = NETS[:11]
M_TOTAL, N_TEST = 51, 10


def panels_for(mode):
    if mode == "nn":
        return [p for p in ALL_PANELS if p[0] == "nn"]
    if mode == "sent":
        return [p for p in ALL_PANELS if p[1] == "random"]
    return ALL_PANELS


def rows_seeded(seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(M_TOTAL); rng.shuffle(idx)
    return np.sort(idx[N_TEST:]), np.sort(idx[:N_TEST])


def train_nn_seeded(Y_full, sensor_idx, wd, train_rows, seed):
    """splits.train_nn with the initialization seed exposed."""
    torch.manual_seed(seed)
    Y_train = Y_full[train_rows]
    N = Y_full.shape[1]
    y_min = float(Y_train.min()); y_max = float(Y_train.max())
    y_range = max(y_max - y_min, 1e-12)
    Xtr = torch.tensor((Y_train[:, sensor_idx] - y_min) / y_range,
                       dtype=torch.float32)
    Ytr = torch.tensor((Y_train - y_min) / y_range, dtype=torch.float32)
    model = tz.ShallowDecoderNoBias(len(sensor_idx), pu.HIDDEN, N)
    opt = torch.optim.Adam(model.parameters(), lr=pu.LR, weight_decay=wd)
    loss_fn = torch.nn.MSELoss()
    for _ in range(pu.EPOCHS):
        opt.zero_grad(); loss_fn(model(Xtr), Ytr).backward(); opt.step()
    model.eval()
    return {"model_state_dict": model.state_dict(), "sensor_idx": sensor_idx,
            "y_min": y_min, "y_max": y_max, "y_range": y_range,
            "src_sent_min": float(Y_train[:, sensor_idx].min()),
            "src_sent_max": float(Y_train[:, sensor_idx].max()),
            "k_sentinels": len(sensor_idx), "hidden": pu.HIDDEN,
            "n_nodes": N, "_no_bias": True}


def load_eq_rep(tag, dyn, mode, rep):
    if mode == "hetero" and dyn in HETERO and rep > 0:
        f = (EQ / "hetero_replicates" / f"rep{rep:02d}" /
             f"{tag}_{STEMS[dyn]}_equilibrium_data.csv")
        return np.loadtxt(f, delimiter=",", skiprows=1)[:, 1:]
    return leq.load_eq(tag, dyn)


def nrmse(pred, true, rng_full):
    return float(np.sqrt(np.mean((pred - true) ** 2))) / rng_full


def one_replicate(mode, rep, panels):
    """(n_net, n_dyn, n_dyn) n-RMSE and baseline arrays for one replicate."""
    n = len(DYN)
    nets = HET_NETS if mode == "hetero" else NETS
    M = {p: np.full((len(nets), n, n), np.nan) for p in panels}
    B = {p: np.full((len(nets), n, n), np.nan) for p in panels}
    # which seeds move in this mode
    nn_seed = (1000 + rep) if mode == "nn" else pu.SEED
    # one split for both sides, so no evaluation row is ever a training row
    src_rows, src_ev = (rows_seeded(2_000 + rep) if mode == "split"
                        else splits.rows_random())
    tgt_cal, tgt_ev = src_rows, src_ev
    sent_seed = (3_000 + rep) if mode == "sent" else 2026

    for ni, tag in enumerate(nets):
        G = nx.read_edgelist(leq.network_file(tag),
                             nodetype=int)
        N = len(G.nodes()); K = int(np.floor(np.log(N)))
        eq = {s: load_eq_rep(tag, s, mode, rep) for s in DYN}
        rngf = {s: float(eq[s].max() - eq[s].min()) for s in DYN}
        RAND = np.sort(np.random.default_rng(sent_seed).choice(
            N, size=K, replace=False))
        for dec, sent in panels:
            sidx = {}
            for s in DYN:
                Ytr = eq[s][src_rows]
                sidx[s] = (pu.qr_pivot_sensors(Ytr, K) if sent == "qr" else
                           pu.min_corr_sensors(Ytr, K) if sent == "minc"
                           else RAND.copy())
            for i, src in enumerate(DYN):
                idx = sidx[src]
                ck = (train_nn_seeded(eq[src], idx, REG[dec], src_rows,
                                      nn_seed) if dec == "nn"
                      else splits.fit_linear(eq[src], idx, REG[dec], src_rows))
                for j, tgt in enumerate(DYN):
                    Yt = eq[tgt]
                    P = splits.predict_calibrated(ck, Yt, dec, tgt_cal)
                    M[(dec, sent)][ni, i, j] = nrmse(P[tgt_ev], Yt[tgt_ev],
                                                     rngf[tgt])
                    per = Yt[tgt_ev][:, idx].mean(axis=1)
                    Bh = np.broadcast_to(per[:, None], Yt[tgt_ev].shape)
                    B[(dec, sent)][ni, i, j] = nrmse(Bh, Yt[tgt_ev],
                                                     rngf[tgt])
    return M, B


def main():
    mode = sys.argv[1]
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if mode == "base":
        R = 1
    panels = panels_for(mode)
    accM = {p: [] for p in panels}
    accB = {p: [] for p in panels}
    t0 = time.time()
    for rep in range(R):
        M, B = one_replicate(mode, rep, panels)
        for p in panels:
            accM[p].append(M[p]); accB[p].append(B[p])
        print(f"  {mode} replicate {rep+1}/{R} done "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
    out = {}
    for dec, sent in panels:
        out[f"M_{dec}_{sent}"] = np.array(accM[(dec, sent)])
        out[f"B_{dec}_{sent}"] = np.array(accB[(dec, sent)])
    np.savez_compressed(HERE / "data" / f"rep_{mode}.npz",
                        dynamics=np.array([DISP.get(d, d) for d in DYN],
                                          dtype=object),
                        networks=np.array(HET_NETS if mode == "hetero"
                                          else NETS), **out)
    print(f"  wrote data/rep_{mode}.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
