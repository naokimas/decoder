"""v22 cross-matrix pipeline.

Training (Sections 5.2-5.4): unchanged from v21 (global min-max scaling).
Cross-transfer (Section 5.5): corrected.
  - Step 2(a): map the target sentinel raw range to the source sentinel
    raw range (using the source's per-sentinel min/max recorded at
    training), then apply the source's global min-max. The decoder
    operates in-distribution on its input side.
  - Step 2(c): DROPPED. The output un-scaling is absorbed analytically
    into the Step 3 fair-affine fit.
  - Step 3: fit (a, b) directly on the NORMALIZED decoder output
    against the target's raw sentinel readings.

Outputs:
  <tag>_v22_cross_matrix_data_<method>.npz        (NN)
  <tag>_v22_linear_cross_matrix_data_<method>.npz (linear)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import networkx as nx
import torch
import torch.nn as nn

import pipeline_unified as pu
import pipeline_linear as pl
import pipeline_v21 as p21

HERE = Path(__file__).resolve().parent

NETWORKS = p21.NETWORKS
SYSTEMS = p21.SYSTEMS
CSV_NAMES = p21.CSV_NAMES
RIDGE_ALPHA = p21.RIDGE_ALPHA


def available_systems(tag):
    return p21.available_systems(tag)


def load_eq(tag, s):
    return p21.load_eq(tag, s)


# ----- Training (same as v21, but record source sentinel min/max). -----
def train_nn(Y_full, sensor_idx, epochs=pu.EPOCHS, lr=pu.LR):
    ckpt = p21.train_nn_minmax(Y_full, sensor_idx, epochs=epochs, lr=lr)
    # Record source sentinel-only min/max from the training split.
    rng = np.random.default_rng(pu.SEED); idx = np.arange(Y_full.shape[0])
    rng.shuffle(idx)
    n_test = int(round(pu.TEST_FRACTION * Y_full.shape[0]))
    Y_train = Y_full[np.sort(idx[n_test:])]
    X_train_src = Y_train[:, sensor_idx]
    ckpt["src_sent_min"] = float(X_train_src.min())
    ckpt["src_sent_max"] = float(X_train_src.max())
    return ckpt


def fit_linear(Y_full, sensor_idx, alpha=RIDGE_ALPHA):
    ckpt = p21.fit_linear_minmax(Y_full, sensor_idx, alpha=alpha)
    rng = np.random.default_rng(pl.SEED); idx = np.arange(Y_full.shape[0])
    rng.shuffle(idx)
    n_test = int(round(pl.TEST_FRACTION * Y_full.shape[0]))
    Y_train = Y_full[np.sort(idx[n_test:])]
    X_train_src = Y_train[:, sensor_idx]
    ckpt["src_sent_min"] = float(X_train_src.min())
    ckpt["src_sent_max"] = float(X_train_src.max())
    return ckpt


def train_nn_combined(Y_train, Y_test, test_labels, sensor_idx,
                      epochs=400, lr=5e-3, sys_tags=()):
    ckpt = p21.train_nn_combined_minmax(Y_train, Y_test, test_labels,
                                          sensor_idx, epochs=epochs, lr=lr,
                                          sys_tags=sys_tags)
    X_train_src = Y_train[:, sensor_idx]
    ckpt["src_sent_min"] = float(X_train_src.min())
    ckpt["src_sent_max"] = float(X_train_src.max())
    return ckpt


def fit_linear_combined(eq_dict, sensor_idx, sys_tags, alpha=RIDGE_ALPHA):
    ckpt = p21.fit_linear_combined_minmax(eq_dict, sensor_idx,
                                            sys_tags=sys_tags, alpha=alpha)
    rng = np.random.default_rng(pl.SEED)
    train_blocks = []
    for s in sys_tags:
        Y = eq_dict[s]; M = Y.shape[0]
        idx = np.arange(M); rng.shuffle(idx)
        n_test = int(round(pl.TEST_FRACTION * M))
        train_blocks.append(Y[np.sort(idx[n_test:])])
    Y_train = np.vstack(train_blocks)
    X_train_src = Y_train[:, sensor_idx]
    ckpt["src_sent_min"] = float(X_train_src.min())
    ckpt["src_sent_max"] = float(X_train_src.max())
    return ckpt


# ----- Cross-transfer: corrected v22 procedure ------
def _fair_v22_core(ckpt, target_full, decode_fn):
    """decode_fn maps source-normalized n x M input to source-normalized
    N x M output."""
    sensor_idx = ckpt["sensor_idx"]
    S = target_full[:, sensor_idx]
    y_min_T = float(S.min()); y_max_T = float(S.max())
    y_range_T = max(y_max_T - y_min_T, 1e-12)
    y_min_S_s = ckpt["src_sent_min"]; y_max_S_s = ckpt["src_sent_max"]
    y_range_S_s = max(y_max_S_s - y_min_S_s, 1e-12)
    # Step 2(a): map target sentinel raw -> source sentinel raw, then
    # apply source's global min-max.
    tilde_S_raw = y_min_S_s + (S - y_min_T) * y_range_S_s / y_range_T
    tilde_S_norm = (tilde_S_raw - ckpt["y_min"]) / ckpt["y_range"]
    # Step 2(b): pass through decoder.
    Yhat_norm = decode_fn(tilde_S_norm)
    # Step 3: fit (a, b) on sentinel rows against target raw sentinel
    # readings; no separate Step 2(c) un-scale.
    pred_sent = Yhat_norm[:, sensor_idx].flatten()
    true_sent = S.flatten()
    A_d = np.column_stack([pred_sent, np.ones_like(pred_sent)])
    (a, b), *_ = np.linalg.lstsq(A_d, true_sent, rcond=None)
    return a * Yhat_norm + b, float(a), float(b)


def fair_predict_nn(ckpt, target_full):
    model = pu.ShallowDecoder(ckpt["k_sentinels"], ckpt["hidden"],
                              ckpt["n_nodes"])
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    def decode(Xn):
        with torch.no_grad():
            return model(torch.tensor(Xn, dtype=torch.float32)).numpy()
    return _fair_v22_core(ckpt, target_full, decode)


def fair_predict_linear(ckpt, target_full):
    W = ckpt["W"]; bvec = ckpt["b"]
    def decode(Xn):
        return Xn @ W + bvec
    return _fair_v22_core(ckpt, target_full, decode)


# ----- Metric: global RMSE/range -----
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
    print(f"\n===== v22 on {tag.upper()}: {len(avail)}/{len(SYSTEMS)} "
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
        nn_qr[s] = train_nn(Y, qr_idx)
        nn_rd[s] = train_nn(Y, RAND.copy())
        nn_mc[s] = train_nn(Y, mc_idx)
        li_qr[s] = fit_linear(Y, qr_idx)
        li_rd[s] = fit_linear(Y, RAND.copy())
        li_mc[s] = fit_linear(Y, mc_idx)
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
    nn_qr["COMBINED"] = train_nn_combined(Yc_tr, Yc_te, test_labels, cqr,
                                          sys_tags=tuple(avail))
    nn_rd["COMBINED"] = train_nn_combined(Yc_tr, Yc_te, test_labels,
                                          RAND.copy(), sys_tags=tuple(avail))
    nn_mc["COMBINED"] = train_nn_combined(Yc_tr, Yc_te, test_labels, cmc,
                                          sys_tags=tuple(avail))
    li_qr["COMBINED"] = fit_linear_combined(eq, cqr, sys_tags=tuple(avail))
    li_rd["COMBINED"] = fit_linear_combined(eq, RAND.copy(),
                                            sys_tags=tuple(avail))
    li_mc["COMBINED"] = fit_linear_combined(eq, cmc, sys_tags=tuple(avail))

    SOURCES_FULL = SYSTEMS + ["COMBINED"]
    TARGETS = SYSTEMS

    for variant, ckpts in [("qr", nn_qr), ("random", nn_rd), ("minc", nn_mc)]:
        rel = cross_metrics(eq, ckpts, SOURCES_FULL, TARGETS,
                            fair_predict_nn, avail)
        np.savez(HERE / f"{tag}_v22_cross_matrix_data_{variant}.npz",
                 sources=np.array(SOURCES_FULL, dtype="U16"),
                 targets=np.array(TARGETS, dtype="U16"),
                 N=N, K=K, rel=rel)
    for variant, ckpts in [("qr", li_qr), ("random", li_rd), ("minc", li_mc)]:
        rel = cross_metrics(eq, ckpts, SOURCES_FULL, TARGETS,
                            fair_predict_linear, avail)
        np.savez(HERE / f"{tag}_v22_linear_cross_matrix_data_{variant}.npz",
                 sources=np.array(SOURCES_FULL, dtype="U16"),
                 targets=np.array(TARGETS, dtype="U16"),
                 N=N, K=K, rel=rel)
    print(f"  wrote 6 v22 npz files for {tag}", flush=True)


def main():
    for tag in NETWORKS:
        run_one(tag)


if __name__ == "__main__":
    main()
