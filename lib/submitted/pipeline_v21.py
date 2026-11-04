"""v21 cross-matrix pipeline: same as pipeline_v19m (16 dynamics x 14
networks, partial coverage, global RMSE/range metric) BUT trained
under global min-max scaling instead of per-row z-score.

Source training: compute (y_min, y_max) as scalars over the entire
training matrix Y (N rows = nodes, M_train cols = swept parameter
values). Normalize Y -> (Y - y_min) / (y_max - y_min). Train decoder
on normalized data.

Target deployment (fair rule): compute (y_min^T, y_max^T) from the
target's sentinel rows alone, normalize sentinel readings, pass
through decoder, un-normalize by SOURCE's (y_min, y_max), apply the
single global (a, b) affine fit on sentinel rows.

Outputs:
  <tag>_v21_cross_matrix_data_<method>.npz        (NN)
  <tag>_v21_linear_cross_matrix_data_<method>.npz (linear)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import networkx as nx
import torch
import torch.nn as nn

import pipeline_unified as pu
import pipeline_linear as pl

HERE = Path(__file__).resolve().parent

NETWORKS = ["dolphin", "prox", "metabolic", "gkk", "lfr", "er", "ba", "hk",
            "euroroad", "email", "flybi", "reactome", "route_views",
            "spanish"]
SYSTEMS = ["SIS", "SIR", "MSIS", "MSIR", "MSIS2", "MSIR2",
           "DW", "MUT", "GEN", "WC", "gLV", "NM",
           "EUT", "VEG", "EUT2", "VEG2"]
CSV_NAMES = {"SIS": "sis", "DW": "doublewell", "MUT": "mutualistic",
             "GEN": "genereg", "WC": "wc", "gLV": "glv",
             "SIR": "sir", "NM": "noymeir",
             "EUT": "eutroph", "VEG": "vegturb",
             "MSIS": "msis", "MSIR": "msir",
             "MSIS2": "msis2", "MSIR2": "msir2",
             "EUT2": "eutroph2", "VEG2": "vegturb2"}

RIDGE_ALPHA = 1e-2


def available_systems(tag):
    return [s for s in SYSTEMS
            if (HERE / f"{tag}_{CSV_NAMES[s]}_equilibrium_data.csv").exists()]


def load_eq(tag, s):
    arr = np.loadtxt(HERE / f"{tag}_{CSV_NAMES[s]}_equilibrium_data.csv",
                     delimiter=",", skiprows=1)
    return arr[:, 1:]


# ---------- Min-max NN ----------
def train_nn_minmax(Y_full, sensor_idx, epochs=pu.EPOCHS, lr=pu.LR):
    rng = np.random.default_rng(pu.SEED); torch.manual_seed(pu.SEED)
    M, N = Y_full.shape
    idx = np.arange(M); rng.shuffle(idx)
    n_test = int(round(pu.TEST_FRACTION * M))
    Y_test = Y_full[np.sort(idx[:n_test])]
    Y_train = Y_full[np.sort(idx[n_test:])]
    y_min = float(Y_train.min()); y_max = float(Y_train.max())
    y_range = max(y_max - y_min, 1e-12)
    Xtr = (Y_train[:, sensor_idx] - y_min) / y_range
    Ytr = (Y_train - y_min) / y_range
    Xte = (Y_test[:, sensor_idx] - y_min) / y_range
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    model = pu.ShallowDecoder(len(sensor_idx), pu.HIDDEN, N)
    opt = torch.optim.Adam(model.parameters(), lr=lr); loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); loss_fn(model(Xtr_t), Ytr_t).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Yte_pred = model(Xte_t).numpy() * y_range + y_min
    return {
        "model_state_dict": model.state_dict(),
        "k_sentinels": len(sensor_idx), "hidden": pu.HIDDEN, "n_nodes": N,
        "sensor_idx": sensor_idx,
        "y_min": y_min, "y_max": y_max, "y_range": y_range,
        "test_mse": float(np.mean((Yte_pred - Y_test) ** 2)),
    }


def fair_predict_nn_minmax(ckpt, target_full):
    sensor_idx = ckpt["sensor_idx"]
    S = target_full[:, sensor_idx]
    y_min_T = float(S.min()); y_max_T = float(S.max())
    y_range_T = max(y_max_T - y_min_T, 1e-12)
    Sn = (S - y_min_T) / y_range_T
    model = pu.ShallowDecoder(ckpt["k_sentinels"], ckpt["hidden"],
                              ckpt["n_nodes"])
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    with torch.no_grad():
        Yn = model(torch.tensor(Sn, dtype=torch.float32)).numpy()
    Y_src = Yn * ckpt["y_range"] + ckpt["y_min"]
    pred_sent = Y_src[:, sensor_idx].flatten()
    true_sent = S.flatten()
    A_d = np.column_stack([pred_sent, np.ones_like(pred_sent)])
    (a, b), *_ = np.linalg.lstsq(A_d, true_sent, rcond=None)
    return a * Y_src + b, float(a), float(b)


def train_nn_combined_minmax(Y_train, Y_test, test_labels, sensor_idx,
                              epochs=400, lr=5e-3, sys_tags=()):
    torch.manual_seed(pu.SEED)
    N = Y_train.shape[1]
    y_min = float(Y_train.min()); y_max = float(Y_train.max())
    y_range = max(y_max - y_min, 1e-12)
    Xtr = torch.tensor((Y_train[:, sensor_idx] - y_min) / y_range,
                       dtype=torch.float32)
    Ytr = torch.tensor((Y_train - y_min) / y_range, dtype=torch.float32)
    model = pu.ShallowDecoder(len(sensor_idx), pu.HIDDEN, N)
    opt = torch.optim.Adam(model.parameters(), lr=lr); loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); loss_fn(model(Xtr), Ytr).backward(); opt.step()
    return {
        "model_state_dict": model.state_dict(),
        "k_sentinels": len(sensor_idx), "hidden": pu.HIDDEN, "n_nodes": N,
        "sensor_idx": sensor_idx,
        "y_min": y_min, "y_max": y_max, "y_range": y_range,
    }


# ---------- Min-max linear ----------
def fit_linear_minmax(Y_full, sensor_idx, alpha=RIDGE_ALPHA):
    rng = np.random.default_rng(pl.SEED)
    M, N = Y_full.shape
    idx = np.arange(M); rng.shuffle(idx)
    n_test = int(round(pl.TEST_FRACTION * M))
    Y_train = Y_full[np.sort(idx[n_test:])]
    Y_test  = Y_full[np.sort(idx[:n_test])]
    y_min = float(Y_train.min()); y_max = float(Y_train.max())
    y_range = max(y_max - y_min, 1e-12)
    X_train = Y_train[:, sensor_idx]
    Xn = (X_train - y_min) / y_range
    Yn = (Y_train - y_min) / y_range
    A_des = np.column_stack([Xn, np.ones(Xn.shape[0])])
    Wb = pl.ridge_solve(A_des, Yn, alpha)
    W = Wb[:-1, :]; b = Wb[-1, :]
    Xte_n = (Y_test[:, sensor_idx] - y_min) / y_range
    Yte_pred = (Xte_n @ W + b) * y_range + y_min
    return {"W": W, "b": b, "k_sentinels": len(sensor_idx), "n_nodes": N,
            "sensor_idx": sensor_idx,
            "y_min": y_min, "y_max": y_max, "y_range": y_range,
            "test_mse": float(np.mean((Yte_pred - Y_test) ** 2))}


def fair_predict_linear_minmax(ckpt, target_full):
    sensor_idx = ckpt["sensor_idx"]
    S = target_full[:, sensor_idx]
    y_min_T = float(S.min()); y_max_T = float(S.max())
    y_range_T = max(y_max_T - y_min_T, 1e-12)
    Sn = (S - y_min_T) / y_range_T
    Y_src = (Sn @ ckpt["W"] + ckpt["b"]) * ckpt["y_range"] + ckpt["y_min"]
    pred_sent = Y_src[:, sensor_idx].flatten()
    true_sent = S.flatten()
    A_d = np.column_stack([pred_sent, np.ones_like(pred_sent)])
    (a, b), *_ = np.linalg.lstsq(A_d, true_sent, rcond=None)
    return a * Y_src + b, float(a), float(b)


def fit_linear_combined_minmax(eq_dict, sensor_idx, sys_tags,
                                alpha=RIDGE_ALPHA):
    rng = np.random.default_rng(pl.SEED)
    train_blocks = []
    for s in sys_tags:
        Y = eq_dict[s]; M = Y.shape[0]
        idx = np.arange(M); rng.shuffle(idx)
        n_test = int(round(pl.TEST_FRACTION * M))
        train_blocks.append(Y[np.sort(idx[n_test:])])
    Y_train = np.vstack(train_blocks)
    y_min = float(Y_train.min()); y_max = float(Y_train.max())
    y_range = max(y_max - y_min, 1e-12)
    X_train = Y_train[:, sensor_idx]
    Xn = (X_train - y_min) / y_range
    Yn = (Y_train - y_min) / y_range
    A_des = np.column_stack([Xn, np.ones(Xn.shape[0])])
    Wb = pl.ridge_solve(A_des, Yn, alpha)
    W = Wb[:-1, :]; b = Wb[-1, :]
    return {"W": W, "b": b, "k_sentinels": len(sensor_idx),
            "n_nodes": Y_train.shape[1], "sensor_idx": sensor_idx,
            "y_min": y_min, "y_max": y_max, "y_range": y_range}


# ---------- Metric: global RMSE/range (v19 metric) ----------
def global_rel(Y_pred, Y_true):
    rmse = float(np.sqrt(np.mean((Y_pred - Y_true) ** 2)))
    rng = float(Y_true.max() - Y_true.min())
    return rmse / rng if rng > 0 else float("nan")


def cross_metrics(eq, ckpts, sources_full, targets_full, fp_fn, avail):
    nS, nT = len(sources_full), len(targets_full)
    rel = np.full((nS, nT), np.nan)
    for i, src in enumerate(sources_full):
        if src not in ckpts:
            continue
        ckpt = ckpts[src]
        for j, tgt in enumerate(targets_full):
            if tgt not in avail:
                continue
            Y_true = eq[tgt]
            Y_pred, a, b = fp_fn(ckpt, Y_true)
            rel[i, j] = global_rel(Y_pred, Y_true)
    return rel


def run_one(tag):
    avail = available_systems(tag)
    print(f"\n===== v21 on {tag.upper()}: {len(avail)}/{len(SYSTEMS)} "
          f"=====", flush=True)
    if len(avail) < 2:
        return
    G = nx.read_edgelist(HERE / f"{tag}_network.edgelist", nodetype=int)
    N = len(G.nodes()); K = int(np.floor(np.log(N)))
    eq = {s: load_eq(tag, s) for s in avail}

    rng_rand = np.random.default_rng(2026)
    RAND = np.sort(rng_rand.choice(N, size=K, replace=False))

    nn_qr, nn_rd, nn_mc = {}, {}, {}
    li_qr, li_rd, li_mc = {}, {}, {}
    for s in avail:
        Y = eq[s]; M = Y.shape[0]
        rng = np.random.default_rng(pu.SEED); idx = np.arange(M); rng.shuffle(idx)
        n_test = int(round(pu.TEST_FRACTION * M))
        Y_train_pick = Y[np.sort(idx[n_test:])]
        qr_idx = pu.qr_pivot_sensors(Y_train_pick, K)
        mc_idx = pu.min_corr_sensors(Y_train_pick, K)
        nn_qr[s] = train_nn_minmax(Y, qr_idx)
        nn_rd[s] = train_nn_minmax(Y, RAND.copy())
        nn_mc[s] = train_nn_minmax(Y, mc_idx)
        li_qr[s] = fit_linear_minmax(Y, qr_idx)
        li_rd[s] = fit_linear_minmax(Y, RAND.copy())
        li_mc[s] = fit_linear_minmax(Y, mc_idx)
    # COMBINED on union of available
    rng = np.random.default_rng(pu.SEED)
    train_blocks, test_blocks, test_labels = [], [], []
    for s in avail:
        Y = eq[s]; M = Y.shape[0]; idx = np.arange(M); rng.shuffle(idx)
        n_test = int(round(pu.TEST_FRACTION * M))
        train_blocks.append(Y[np.sort(idx[n_test:])])
        test_blocks.append(Y[np.sort(idx[:n_test])])
        test_labels.extend([s] * n_test)
    Yc_tr = np.vstack(train_blocks); Yc_te = np.vstack(test_blocks)
    test_labels = np.array(test_labels)
    cqr = pu.qr_pivot_sensors(Yc_tr, K)
    cmc = pu.min_corr_sensors(Yc_tr, K)
    nn_qr["COMBINED"] = train_nn_combined_minmax(Yc_tr, Yc_te, test_labels,
                                                  cqr, sys_tags=tuple(avail))
    nn_rd["COMBINED"] = train_nn_combined_minmax(Yc_tr, Yc_te, test_labels,
                                                  RAND.copy(),
                                                  sys_tags=tuple(avail))
    nn_mc["COMBINED"] = train_nn_combined_minmax(Yc_tr, Yc_te, test_labels,
                                                  cmc, sys_tags=tuple(avail))
    li_qr["COMBINED"] = fit_linear_combined_minmax(eq, cqr, sys_tags=tuple(avail))
    li_rd["COMBINED"] = fit_linear_combined_minmax(eq, RAND.copy(),
                                                    sys_tags=tuple(avail))
    li_mc["COMBINED"] = fit_linear_combined_minmax(eq, cmc, sys_tags=tuple(avail))

    SOURCES_FULL = SYSTEMS + ["COMBINED"]
    TARGETS = SYSTEMS

    for variant, ckpts in [("qr", nn_qr), ("random", nn_rd), ("minc", nn_mc)]:
        rel = cross_metrics(eq, ckpts, SOURCES_FULL, TARGETS,
                            fair_predict_nn_minmax, avail)
        np.savez(HERE / f"{tag}_v21_cross_matrix_data_{variant}.npz",
                 sources=np.array(SOURCES_FULL, dtype="U16"),
                 targets=np.array(TARGETS, dtype="U16"),
                 N=N, K=K, rel=rel)
    for variant, ckpts in [("qr", li_qr), ("random", li_rd), ("minc", li_mc)]:
        rel = cross_metrics(eq, ckpts, SOURCES_FULL, TARGETS,
                            fair_predict_linear_minmax, avail)
        np.savez(HERE / f"{tag}_v21_linear_cross_matrix_data_{variant}.npz",
                 sources=np.array(SOURCES_FULL, dtype="U16"),
                 targets=np.array(TARGETS, dtype="U16"),
                 N=N, K=K, rel=rel)
    print(f"  wrote 6 v21 npz files for {tag}", flush=True)


def main():
    for tag in NETWORKS:
        run_one(tag)


if __name__ == "__main__":
    main()
