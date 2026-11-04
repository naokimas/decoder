"""Factorial sweep across (decoder, reg, sentinel, bias-treatment).

For each (decoder, reg, sentinel, treatment), build a 17 x 16 cross-matrix
of relative RMSE (RMSE / target dynamic range), aggregated via nanmean
across the 14 networks, then compute:
  - mean on-diagonal RMSE/range (16 cells, source == target),
  - mean off-diagonal RMSE/range (the 16x16 square minus the diagonal,
    plus separately the COMBINED row),
  - min-cover tau at |C|=3 and |C|=2.

Cells:
  decoder x reg:
    NN with weight_decay in {0.0, 0.01, 0.1}, Linear ridge with alpha in
    {0.01, 0.1}.  Total 5 (decoder, reg) cells.
  sentinel: QR, random, MinCorr.  3 cells.
  treatment: baseline (= v22), A (zero-anchor at deployment),
             B (retrained no-bias decoder, v22 fair-affine),
             A+B (no-bias decoder AND zero-anchor at deployment).
             4 cells.
Total 5 * 3 * 4 = 60 (decoder, reg, sentinel, treatment) cells.

Outputs:
  --per-cell-csv:  one row per (network, dec, reg, sent, treat, src, tgt).
  --out:           one row per (dec, reg, sent, treat) summary.

Usage:
  python3 sweep_v22r_factorial.py [--networks dolphin]
  python3 sweep_v22r_factorial.py  # all 14 networks
"""
from __future__ import annotations
from itertools import combinations
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import networkx as nx
import torch
import torch.nn as nn

import pipeline_unified as pu
import pipeline_linear as pl
import pipeline_v22 as p22
import pipeline_v22r as p22r
import test_zero_anchor as tz

HERE = Path(__file__).resolve().parent

NETWORKS = p22.NETWORKS
SYSTEMS = p22.SYSTEMS
SOURCES = SYSTEMS + ["COMBINED"]
TARGETS = SYSTEMS

CELL_REGS = [
    ("nn", 0.0),
    ("nn", 0.01),
    ("nn", 0.1),
    ("lin", 0.01),
    ("lin", 0.1),
]
SENTINELS = ["qr", "random", "minc"]
TREATMENTS = ["baseline", "A", "B", "ApB"]


# ---------- No-bias combined trainers ----------
def train_nn_combined_no_bias(Y_train, sensor_idx, weight_decay,
                              epochs=400, lr=5e-3):
    torch.manual_seed(pu.SEED)
    N = Y_train.shape[1]
    y_min = float(Y_train.min()); y_max = float(Y_train.max())
    y_range = max(y_max - y_min, 1e-12)
    Xtr = torch.tensor((Y_train[:, sensor_idx] - y_min) / y_range,
                       dtype=torch.float32)
    Ytr = torch.tensor((Y_train - y_min) / y_range, dtype=torch.float32)
    model = tz.ShallowDecoderNoBias(len(sensor_idx), pu.HIDDEN, N)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); loss_fn(model(Xtr), Ytr).backward(); opt.step()
    return {"model_state_dict": model.state_dict(),
            "k_sentinels": len(sensor_idx), "hidden": pu.HIDDEN, "n_nodes": N,
            "sensor_idx": sensor_idx,
            "y_min": y_min, "y_max": y_max, "y_range": y_range,
            "src_sent_min": float(Y_train[:, sensor_idx].min()),
            "src_sent_max": float(Y_train[:, sensor_idx].max()),
            "_no_bias": True}


def fit_linear_combined_no_bias(eq_dict, sensor_idx, sys_tags, alpha):
    rng = np.random.default_rng(pl.SEED)
    train_blocks = []
    for s in sys_tags:
        Y = eq_dict[s]; M = Y.shape[0]
        idx = np.arange(M); rng.shuffle(idx)
        n_test = int(round(pl.TEST_FRACTION * M))
        train_blocks.append(Y[np.sort(idx[n_test:])])
    Y_train = np.vstack(train_blocks)
    N = Y_train.shape[1]
    y_min = float(Y_train.min()); y_max = float(Y_train.max())
    y_range = max(y_max - y_min, 1e-12)
    Xn = (Y_train[:, sensor_idx] - y_min) / y_range
    Yn = (Y_train - y_min) / y_range
    ATA = Xn.T @ Xn + alpha * np.eye(Xn.shape[1])
    W = np.linalg.solve(ATA, Xn.T @ Yn)
    return {"W": W, "b": np.zeros(N), "sensor_idx": sensor_idx,
            "y_min": y_min, "y_max": y_max, "y_range": y_range,
            "src_sent_min": float(Y_train[:, sensor_idx].min()),
            "src_sent_max": float(Y_train[:, sensor_idx].max()),
            "k_sentinels": len(sensor_idx), "n_nodes": N,
            "_no_bias": True}


# ---------- Prediction under four treatments ----------
def _decode_fn(ckpt, dec_type):
    if dec_type == "nn":
        if ckpt.get("_no_bias", False):
            model = tz.ShallowDecoderNoBias(ckpt["k_sentinels"],
                                            ckpt["hidden"], ckpt["n_nodes"])
        else:
            model = pu.ShallowDecoder(ckpt["k_sentinels"],
                                      ckpt["hidden"], ckpt["n_nodes"])
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        def f(Xn):
            with torch.no_grad():
                return model(torch.tensor(Xn, dtype=torch.float32)).numpy()
        return f
    W = ckpt["W"]; bvec = ckpt["b"]
    def g(Xn):
        return Xn @ W + bvec
    return g


def _tilde_norm(ckpt, target_full):
    sensor_idx = ckpt["sensor_idx"]
    S = target_full[:, sensor_idx]
    y_min_T = float(S.min()); y_max_T = float(S.max())
    y_range_T = max(y_max_T - y_min_T, 1e-12)
    y_min_S_s = ckpt["src_sent_min"]; y_max_S_s = ckpt["src_sent_max"]
    y_range_S_s = max(y_max_S_s - y_min_S_s, 1e-12)
    tilde_S_raw = y_min_S_s + (S - y_min_T) * y_range_S_s / y_range_T
    tilde_S_norm = (tilde_S_raw - ckpt["y_min"]) / ckpt["y_range"]
    return tilde_S_norm, S


def predict_baseline(ckpt, target_full, dec_type):
    decode = _decode_fn(ckpt, dec_type)
    tilde_S_norm, S = _tilde_norm(ckpt, target_full)
    Yhat_norm = decode(tilde_S_norm)
    sensor_idx = ckpt["sensor_idx"]
    pred_sent = Yhat_norm[:, sensor_idx].flatten()
    true_sent = S.flatten()
    A_d = np.column_stack([pred_sent, np.ones_like(pred_sent)])
    (a, b), *_ = np.linalg.lstsq(A_d, true_sent, rcond=None)
    return a * Yhat_norm + b


def predict_optA(ckpt, target_full, dec_type):
    decode = _decode_fn(ckpt, dec_type)
    tilde_S_norm, S = _tilde_norm(ckpt, target_full)
    Yhat_norm = decode(tilde_S_norm)
    K = ckpt["k_sentinels"]
    Y0 = decode(np.zeros((1, K), dtype=tilde_S_norm.dtype))[0]
    Yhat_centered = Yhat_norm - Y0[None, :]
    sensor_idx = ckpt["sensor_idx"]
    pred_sent = Yhat_centered[:, sensor_idx].flatten()
    true_sent = S.flatten()
    denom = float(pred_sent @ pred_sent)
    a = float(pred_sent @ true_sent) / max(denom, 1e-12)
    return a * Yhat_centered


def rmse_over_range(Y_pred, Y_true):
    rmse = float(np.sqrt(np.mean((Y_pred - Y_true) ** 2)))
    rng = float(Y_true.max() - Y_true.min())
    return rmse / rng if rng > 0 else float("nan")


# ---------- Cover-chain ----------
def min_cover_chain(M):
    n_src, n_tgt = M.shape
    finite_vals = M[np.isfinite(M)]
    if not finite_vals.size:
        return []
    taus = sorted(set(finite_vals.tolist()))
    coverable = np.where(np.any(np.isfinite(M), axis=0))[0]
    tgt_full = 0
    for j in coverable:
        tgt_full |= (1 << j)
    last_size = None; chain = []
    for tau in taus:
        adj = (M <= tau) & np.isfinite(M)
        if not adj[:, coverable].any(axis=0).all():
            continue
        sets_per = np.zeros(n_src, dtype=np.int64)
        for s in range(n_src):
            bits = 0
            for t in range(n_tgt):
                if adj[s, t]:
                    bits |= (1 << t)
            sets_per[s] = bits
        min_size, min_combo = None, None
        for k in range(1, n_src + 1):
            for combo in combinations(range(n_src), k):
                covered = 0
                for s in combo:
                    covered |= int(sets_per[s])
                if (covered & tgt_full) == tgt_full:
                    min_size, min_combo = k, combo; break
            if min_size is not None: break
        if last_size is None or min_size < last_size:
            chain.append((tau, min_size, list(min_combo)))
            last_size = min_size
        if min_size == 1: break
    return chain


def diag_offdiag_stats(M):
    """For a 17 x 16 cross-matrix M (16 systems + COMBINED as sources).

    Returns:
      diag_mean         -- mean of finite M[i, i] for i in 0..15
      offdiag_mean      -- mean of finite M[i, j] for i, j in 0..15, i != j
      combined_row_mean -- mean of finite M[16, j] for j in 0..15
    """
    n_sys = len(SYSTEMS)
    diag = [M[i, i] for i in range(n_sys) if np.isfinite(M[i, i])]
    off = []
    for i in range(n_sys):
        for j in range(n_sys):
            if i == j: continue
            v = M[i, j]
            if np.isfinite(v): off.append(v)
    comb = [v for v in M[n_sys, :] if np.isfinite(v)]
    return (float(np.mean(diag)) if diag else float("nan"),
            float(np.mean(off)) if off else float("nan"),
            float(np.mean(comb)) if comb else float("nan"))


# ---------- Per-network processing ----------
def process_network(tag, all_results, per_cell_writer):
    avail = p22.available_systems(tag)
    if len(avail) < 2:
        print(f"  {tag}: only {len(avail)} systems, skipping")
        return
    G = nx.read_edgelist(HERE / f"{tag}_network.edgelist", nodetype=int)
    N = len(G.nodes()); K = int(np.floor(np.log(N)))
    eq = {s: p22.load_eq(tag, s) for s in avail}
    rng_rand = np.random.default_rng(2026)
    RAND = np.sort(rng_rand.choice(N, size=K, replace=False))

    sentinel_idx = {}
    for s in avail:
        Y = eq[s]; M = Y.shape[0]
        rng = np.random.default_rng(pu.SEED); idx = np.arange(M); rng.shuffle(idx)
        n_test = int(round(pu.TEST_FRACTION * M))
        Y_train_pick = Y[np.sort(idx[n_test:])]
        sentinel_idx[("qr", s)] = pu.qr_pivot_sensors(Y_train_pick, K)
        sentinel_idx[("random", s)] = RAND.copy()
        sentinel_idx[("minc", s)] = pu.min_corr_sensors(Y_train_pick, K)

    rng = np.random.default_rng(pu.SEED)
    train_blocks = []
    for s in avail:
        Y = eq[s]; M = Y.shape[0]; idx = np.arange(M); rng.shuffle(idx)
        n_test = int(round(pu.TEST_FRACTION * M))
        train_blocks.append(Y[np.sort(idx[n_test:])])
    Yc_tr = np.vstack(train_blocks)
    cqr = pu.qr_pivot_sensors(Yc_tr, K)
    cmc = pu.min_corr_sensors(Yc_tr, K)
    sentinel_idx[("qr", "COMBINED")] = cqr
    sentinel_idx[("random", "COMBINED")] = RAND.copy()
    sentinel_idx[("minc", "COMBINED")] = cmc

    for dec_type, reg in CELL_REGS:
        for sent in SENTINELS:
            ckpts_b, ckpts_nb = {}, {}
            for s in avail:
                idx = sentinel_idx[(sent, s)]
                if dec_type == "nn":
                    ckpts_b[s] = p22r.train_nn_wd(eq[s], idx,
                                                  weight_decay=reg)
                    ckpts_nb[s] = tz.train_nn_no_bias(eq[s], idx,
                                                      weight_decay=reg)
                else:
                    ckpts_b[s] = p22.fit_linear(eq[s], idx, alpha=reg)
                    ckpts_nb[s] = tz.fit_linear_no_bias(eq[s], idx,
                                                        alpha=reg)
            cidx = sentinel_idx[(sent, "COMBINED")]
            if dec_type == "nn":
                ckpts_b["COMBINED"] = p22r.train_nn_combined_wd(
                    Yc_tr, None, None, cidx, weight_decay=reg,
                    sys_tags=tuple(avail))
                ckpts_nb["COMBINED"] = train_nn_combined_no_bias(
                    Yc_tr, cidx, weight_decay=reg)
            else:
                ckpts_b["COMBINED"] = p22.fit_linear_combined(
                    eq, cidx, sys_tags=tuple(avail), alpha=reg)
                ckpts_nb["COMBINED"] = fit_linear_combined_no_bias(
                    eq, cidx, sys_tags=tuple(avail), alpha=reg)

            for treatment in TREATMENTS:
                ckpts = ckpts_b if treatment in ("baseline", "A") else ckpts_nb
                fn = predict_baseline if treatment in ("baseline", "B") \
                                       else predict_optA
                for src in SOURCES:
                    if src not in ckpts:
                        continue
                    ckpt = ckpts[src]
                    for tgt in TARGETS:
                        if tgt not in avail:
                            continue
                        Y_true = eq[tgt]
                        Y_pred = fn(ckpt, Y_true, dec_type)
                        rel = rmse_over_range(Y_pred, Y_true)
                        row = {"network": tag, "dec_type": dec_type,
                               "reg": reg, "sent": sent,
                               "treatment": treatment, "source": src,
                               "target": tgt, "rel": rel}
                        all_results.append(row)
                        per_cell_writer.writerow(row)


# ---------- Aggregation ----------
def aggregate_and_summarize(all_results, out_csv):
    summary = []
    for dec_type, reg in CELL_REGS:
        for sent in SENTINELS:
            for treatment in TREATMENTS:
                Ms = {}
                for r in all_results:
                    if (r["dec_type"] == dec_type and r["reg"] == reg
                            and r["sent"] == sent
                            and r["treatment"] == treatment):
                        net = r["network"]
                        if net not in Ms:
                            Ms[net] = np.full((len(SOURCES), len(TARGETS)),
                                              np.nan)
                        si = SOURCES.index(r["source"])
                        ti = TARGETS.index(r["target"])
                        Ms[net][si, ti] = r["rel"]
                if not Ms:
                    continue
                M_mean = np.nanmean(np.stack(list(Ms.values()), axis=0),
                                    axis=0)
                diag, off, comb = diag_offdiag_stats(M_mean)
                chain = min_cover_chain(M_mean)
                tau_C3 = next((tau for tau, k, si in chain if k == 3),
                              float("nan"))
                tau_C2 = next((tau for tau, k, si in chain if k == 2),
                              float("nan"))
                summary.append({"dec_type": dec_type, "reg": reg,
                                "sent": sent, "treatment": treatment,
                                "diag_mean": diag, "offdiag_mean": off,
                                "combined_row_mean": comb,
                                "tau_C3": tau_C3, "tau_C2": tau_C2,
                                "n_networks": len(Ms)})
    with open(out_csv, "w", newline="") as f:
        fieldnames = ["dec_type", "reg", "sent", "treatment",
                      "diag_mean", "offdiag_mean", "combined_row_mean",
                      "tau_C3", "tau_C2", "n_networks"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    print(f"Wrote {out_csv} with {len(summary)} summary rows.")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="*", default=None,
                    help="Subset of networks (default: all 14)")
    ap.add_argument("--out", default="sweep_v22r_factorial.csv")
    ap.add_argument("--per-cell-csv", default="sweep_v22r_per_cell.csv")
    args = ap.parse_args()

    networks = args.networks if args.networks else NETWORKS
    print(f"Sweep over {len(networks)} networks: {networks}")
    print(f"Cells: {len(CELL_REGS)} (dec,reg) x {len(SENTINELS)} sent x "
          f"{len(TREATMENTS)} treat = "
          f"{len(CELL_REGS) * len(SENTINELS) * len(TREATMENTS)}", flush=True)

    t0 = time.time()
    all_results = []
    with open(args.per_cell_csv, "w", newline="") as f:
        fieldnames = ["network", "dec_type", "reg", "sent", "treatment",
                      "source", "target", "rel"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tag in networks:
            t1 = time.time()
            process_network(tag, all_results, writer)
            f.flush()
            print(f"  {tag} done in {time.time()-t1:.1f}s (elapsed "
                  f"{time.time()-t0:.1f}s, {len(all_results)} rows)",
                  flush=True)
    print(f"\nWrote {args.per_cell_csv} with {len(all_results)} rows.")
    aggregate_and_summarize(all_results, args.out)
    print(f"Total wall-clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
