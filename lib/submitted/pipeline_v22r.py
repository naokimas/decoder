"""v22r (v22 + regularization sweep) cross-matrix pipeline.

Same training and cross-transfer procedure as pipeline_v22 (global
min-max scaling, source-to-source sentinel-range mapping in Step 2(a),
fair-affine fit on normalized output in Step 3), but with knobs for:

  alpha          ridge regularizer for the linear decoder (default 0.01).
  weight_decay   Adam weight_decay for the NN decoder (default 0).
  suffix         file suffix for output npz, e.g. "v22r_a01" -> writes
                 <tag>_v22r_a01_cross_matrix_data_<method>.npz.

Usage:  python3 pipeline_v22r.py <suffix> <alpha> <weight_decay>
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import networkx as nx
import torch
import torch.nn as nn

import pipeline_unified as pu
import pipeline_linear as pl
import pipeline_v22 as p22

HERE = Path(__file__).resolve().parent

NETWORKS = p22.NETWORKS
SYSTEMS = p22.SYSTEMS
CSV_NAMES = p22.CSV_NAMES


def available_systems(tag):
    return p22.available_systems(tag)


def load_eq(tag, s):
    return p22.load_eq(tag, s)


def train_nn_wd(Y_full, sensor_idx, weight_decay,
                epochs=pu.EPOCHS, lr=pu.LR):
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
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); loss_fn(model(Xtr_t), Ytr_t).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Yte_pred = model(Xte_t).numpy() * y_range + y_min
    X_train_src = Y_train[:, sensor_idx]
    return {
        "model_state_dict": model.state_dict(),
        "k_sentinels": len(sensor_idx), "hidden": pu.HIDDEN, "n_nodes": N,
        "sensor_idx": sensor_idx,
        "y_min": y_min, "y_max": y_max, "y_range": y_range,
        "src_sent_min": float(X_train_src.min()),
        "src_sent_max": float(X_train_src.max()),
        "test_mse": float(np.mean((Yte_pred - Y_test) ** 2)),
    }


def train_nn_combined_wd(Y_train, Y_test, test_labels, sensor_idx,
                          weight_decay, epochs=400, lr=5e-3, sys_tags=()):
    torch.manual_seed(pu.SEED)
    N = Y_train.shape[1]
    y_min = float(Y_train.min()); y_max = float(Y_train.max())
    y_range = max(y_max - y_min, 1e-12)
    Xtr = torch.tensor((Y_train[:, sensor_idx] - y_min) / y_range,
                       dtype=torch.float32)
    Ytr = torch.tensor((Y_train - y_min) / y_range, dtype=torch.float32)
    model = pu.ShallowDecoder(len(sensor_idx), pu.HIDDEN, N)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); loss_fn(model(Xtr), Ytr).backward(); opt.step()
    return {
        "model_state_dict": model.state_dict(),
        "k_sentinels": len(sensor_idx), "hidden": pu.HIDDEN, "n_nodes": N,
        "sensor_idx": sensor_idx,
        "y_min": y_min, "y_max": y_max, "y_range": y_range,
        "src_sent_min": float(Y_train[:, sensor_idx].min()),
        "src_sent_max": float(Y_train[:, sensor_idx].max()),
    }


def run_one(tag, alpha, weight_decay, suffix):
    avail = available_systems(tag)
    print(f"\n===== {suffix} on {tag.upper()}: {len(avail)}/{len(SYSTEMS)} "
          f" (alpha={alpha}, wd={weight_decay}) =====", flush=True)
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
        nn_qr[s] = train_nn_wd(Y, qr_idx, weight_decay)
        nn_rd[s] = train_nn_wd(Y, RAND.copy(), weight_decay)
        nn_mc[s] = train_nn_wd(Y, mc_idx, weight_decay)
        li_qr[s] = p22.fit_linear(Y, qr_idx, alpha=alpha)
        li_rd[s] = p22.fit_linear(Y, RAND.copy(), alpha=alpha)
        li_mc[s] = p22.fit_linear(Y, mc_idx, alpha=alpha)
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
    nn_qr["COMBINED"] = train_nn_combined_wd(Yc_tr, Yc_te, test_labels,
                                              cqr, weight_decay,
                                              sys_tags=tuple(avail))
    nn_rd["COMBINED"] = train_nn_combined_wd(Yc_tr, Yc_te, test_labels,
                                              RAND.copy(), weight_decay,
                                              sys_tags=tuple(avail))
    nn_mc["COMBINED"] = train_nn_combined_wd(Yc_tr, Yc_te, test_labels,
                                              cmc, weight_decay,
                                              sys_tags=tuple(avail))
    li_qr["COMBINED"] = p22.fit_linear_combined(eq, cqr,
                                                  sys_tags=tuple(avail),
                                                  alpha=alpha)
    li_rd["COMBINED"] = p22.fit_linear_combined(eq, RAND.copy(),
                                                  sys_tags=tuple(avail),
                                                  alpha=alpha)
    li_mc["COMBINED"] = p22.fit_linear_combined(eq, cmc,
                                                  sys_tags=tuple(avail),
                                                  alpha=alpha)

    SOURCES_FULL = SYSTEMS + ["COMBINED"]
    TARGETS = SYSTEMS

    for variant, ckpts in [("qr", nn_qr), ("random", nn_rd), ("minc", nn_mc)]:
        rel = p22.cross_metrics(eq, ckpts, SOURCES_FULL, TARGETS,
                                 p22.fair_predict_nn, avail)
        np.savez(HERE / f"{tag}_{suffix}_cross_matrix_data_{variant}.npz",
                 sources=np.array(SOURCES_FULL, dtype="U16"),
                 targets=np.array(TARGETS, dtype="U16"),
                 N=N, K=K, rel=rel)
    for variant, ckpts in [("qr", li_qr), ("random", li_rd), ("minc", li_mc)]:
        rel = p22.cross_metrics(eq, ckpts, SOURCES_FULL, TARGETS,
                                 p22.fair_predict_linear, avail)
        np.savez(HERE / f"{tag}_{suffix}_linear_cross_matrix_data_{variant}.npz",
                 sources=np.array(SOURCES_FULL, dtype="U16"),
                 targets=np.array(TARGETS, dtype="U16"),
                 N=N, K=K, rel=rel)
    print(f"  wrote 6 {suffix} npz files for {tag}", flush=True)


def main(suffix, alpha, weight_decay):
    for tag in NETWORKS:
        run_one(tag, alpha, weight_decay, suffix)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: python3 pipeline_v22r.py <suffix> <alpha> <weight_decay>")
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
