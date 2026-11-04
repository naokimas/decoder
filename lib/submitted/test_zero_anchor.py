"""Test the bias-mitigation options A / B / A+B on three decoders for
the SIS-source -> WC-target diagnostic on dolphin (uniformly random
sentinels). Manuscript draft v22 is not touched.

Three decoders:
  D1: shallow NN with weight_decay = 0.1
  D2: linear ridge with alpha = 0.01 (v22 baseline linear)
  D3: linear ridge with alpha = 0.1

Four configurations per decoder:
  baseline: v22 cross-transfer (fit a, b on sentinel rows)
  A:        zero-anchor at deployment (subtract decoder(input=0), fit
            only slope a)
  B:        retrain decoder without bias terms, baseline (a, b) fit
  A+B:      retrain without bias terms AND zero-anchor at deployment
            (final prediction has prediction(input=0) == 0 by construction)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import pipeline_unified as pu
import pipeline_linear as pl
import pipeline_v22 as p22
import pipeline_v22r as p22r

HERE = Path(__file__).resolve().parent
TAG = "dolphin"
TARGET = "WC"


# ---------- Linear decoder, no bias ----------
def fit_linear_no_bias(Y_full, sensor_idx, alpha):
    rng = np.random.default_rng(pl.SEED)
    M, N = Y_full.shape
    idx = np.arange(M); rng.shuffle(idx)
    n_test = int(round(pl.TEST_FRACTION * M))
    Y_train = Y_full[np.sort(idx[n_test:])]
    Y_test = Y_full[np.sort(idx[:n_test])]
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
            "_has_bias": False}


# ---------- NN decoder, no bias ----------
class ShallowDecoderNoBias(nn.Module):
    def __init__(self, k_in, hidden, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k_in, hidden, bias=False), nn.ReLU(),
            nn.Linear(hidden, n_out, bias=False),
        )
    def forward(self, x): return self.net(x)


def train_nn_no_bias(Y_full, sensor_idx, weight_decay,
                     epochs=pu.EPOCHS, lr=pu.LR):
    rng = np.random.default_rng(pu.SEED); torch.manual_seed(pu.SEED)
    M, N = Y_full.shape
    idx = np.arange(M); rng.shuffle(idx)
    n_test = int(round(pu.TEST_FRACTION * M))
    Y_train = Y_full[np.sort(idx[n_test:])]
    y_min = float(Y_train.min()); y_max = float(Y_train.max())
    y_range = max(y_max - y_min, 1e-12)
    Xtr = (Y_train[:, sensor_idx] - y_min) / y_range
    Ytr = (Y_train - y_min) / y_range
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
    model = ShallowDecoderNoBias(len(sensor_idx), pu.HIDDEN, N)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); loss_fn(model(Xtr_t), Ytr_t).backward(); opt.step()
    model.eval()
    return {"model_state_dict": model.state_dict(),
            "k_sentinels": len(sensor_idx), "hidden": pu.HIDDEN, "n_nodes": N,
            "sensor_idx": sensor_idx,
            "y_min": y_min, "y_max": y_max, "y_range": y_range,
            "src_sent_min": float(Y_train[:, sensor_idx].min()),
            "src_sent_max": float(Y_train[:, sensor_idx].max()),
            "_no_bias": True}


# ---------- Decoder dispatch ----------
def make_linear_decoder_fn(ckpt):
    W = ckpt["W"]; bvec = ckpt["b"]
    def decode(Xn):
        return Xn @ W + bvec
    return decode


def make_nn_decoder_fn(ckpt):
    if ckpt.get("_no_bias", False):
        model = ShallowDecoderNoBias(ckpt["k_sentinels"], ckpt["hidden"],
                                     ckpt["n_nodes"])
    else:
        model = pu.ShallowDecoder(ckpt["k_sentinels"], ckpt["hidden"],
                                  ckpt["n_nodes"])
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    def decode(Xn):
        with torch.no_grad():
            return model(torch.tensor(Xn, dtype=torch.float32)).numpy()
    return decode


# ---------- Cross-transfer: v22 (baseline) vs v22+A (zero-anchored) ----------
def cross_xfer_v22(ckpt, target_full, decode_fn):
    """Standard v22 cross-transfer (fit a, b on sentinel rows)."""
    sensor_idx = ckpt["sensor_idx"]
    S = target_full[:, sensor_idx]
    y_min_T = float(S.min()); y_max_T = float(S.max())
    y_range_T = max(y_max_T - y_min_T, 1e-12)
    y_min_S_s = ckpt["src_sent_min"]; y_max_S_s = ckpt["src_sent_max"]
    y_range_S_s = max(y_max_S_s - y_min_S_s, 1e-12)
    tilde_S_raw = y_min_S_s + (S - y_min_T) * y_range_S_s / y_range_T
    tilde_S_norm = (tilde_S_raw - ckpt["y_min"]) / ckpt["y_range"]
    Yhat_norm = decode_fn(tilde_S_norm)
    pred_sent = Yhat_norm[:, sensor_idx].flatten()
    true_sent = S.flatten()
    A_d = np.column_stack([pred_sent, np.ones_like(pred_sent)])
    (a, b), *_ = np.linalg.lstsq(A_d, true_sent, rcond=None)
    return a * Yhat_norm + b, float(a), float(b)


def cross_xfer_optA(ckpt, target_full, decode_fn):
    """v22 + Option A: zero-anchor at deployment.  Compute
    Yhat_norm(input=0), subtract it, fit only slope a (no intercept).
    Final prediction satisfies pred(input=0) == 0 for every node."""
    sensor_idx = ckpt["sensor_idx"]
    S = target_full[:, sensor_idx]
    y_min_T = float(S.min()); y_max_T = float(S.max())
    y_range_T = max(y_max_T - y_min_T, 1e-12)
    y_min_S_s = ckpt["src_sent_min"]; y_max_S_s = ckpt["src_sent_max"]
    y_range_S_s = max(y_max_S_s - y_min_S_s, 1e-12)
    tilde_S_raw = y_min_S_s + (S - y_min_T) * y_range_S_s / y_range_T
    tilde_S_norm = (tilde_S_raw - ckpt["y_min"]) / ckpt["y_range"]
    Yhat_norm = decode_fn(tilde_S_norm)
    K = ckpt["k_sentinels"]
    # Decoder evaluated at the all-zero normalized input.
    Y0 = decode_fn(np.zeros((1, K), dtype=tilde_S_norm.dtype))[0]  # shape (N,)
    Yhat_centered = Yhat_norm - Y0[None, :]
    pred_sent = Yhat_centered[:, sensor_idx].flatten()
    true_sent = S.flatten()
    # Slope-only fit: a = (pred · true) / (pred · pred).
    denom = float(pred_sent @ pred_sent)
    a = float(pred_sent @ true_sent) / max(denom, 1e-12)
    return a * Yhat_centered, a, 0.0


def rmse_over_range(Y_pred, Y_true):
    return float(np.sqrt(np.mean((Y_pred - Y_true) ** 2)) /
                 (Y_true.max() - Y_true.min()))


# ---------- Plot per-node panel grid ----------
def plot_per_node(Y_true, Y_pred, ps, N, out_png, title):
    cols = 9
    rows = int(np.ceil(N / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 1.6 * rows),
                             squeeze=False)
    for n in range(rows * cols):
        r, c = n // cols, n % cols
        ax = axes[r, c]
        if n >= N:
            ax.axis("off"); continue
        ax.plot(ps, Y_true[:, n], "-", color="black", lw=1.0, label="true")
        ax.plot(ps, Y_pred[:, n], "--", color="red", lw=1.0, label="pred")
        ax.set_title(f"node {n+1}", fontsize=7)
        ax.tick_params(labelsize=5)
        ax.grid(alpha=0.3, linewidth=0.4)
        if n == 0:
            ax.legend(loc="best", fontsize=5)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    G = nx.read_edgelist(HERE / f"{TAG}_network.edgelist", nodetype=int)
    N = len(G.nodes()); K = int(np.floor(np.log(N)))
    rng_rand = np.random.default_rng(2026)
    RAND = np.sort(rng_rand.choice(N, size=K, replace=False))
    Y_sis = np.loadtxt(HERE / f"{TAG}_sis_equilibrium_data.csv",
                       delimiter=",", skiprows=1)[:, 1:]
    arr_wc = np.loadtxt(HERE / f"{TAG}_wc_equilibrium_data.csv",
                        delimiter=",", skiprows=1)
    ps = arr_wc[:, 0]; Y_wc = arr_wc[:, 1:]

    DECODERS = [
        ("D0_nn_wd0p01", "shallow NN, weight_decay=0.01, random sentinels",
            "nn", 0.01),
        ("D1_nn_wd0p1",  "shallow NN, weight_decay=0.1, random sentinels",
            "nn", 0.1),
        ("D2_lin_a0p01", "linear ridge, alpha=0.01, random sentinels",
            "lin", 0.01),
        ("D3_lin_a0p1",  "linear ridge, alpha=0.1, random sentinels",
            "lin", 0.1),
    ]

    print(f"{'decoder':<28} {'config':<10}  {'RMSE/range':>10}  "
          f"{'pred@D=0 min':>14}  {'pred@D=0 max':>14}")
    summary = []
    for tag, label, kind, reg in DECODERS:
        # --- Baseline decoder (with bias) ---
        if kind == "nn":
            ckpt_b = p22r.train_nn_wd(Y_sis, RAND.copy(), weight_decay=reg)
            decode_b = make_nn_decoder_fn(ckpt_b)
        else:
            ckpt_b = p22.fit_linear(Y_sis, RAND.copy(), alpha=reg)
            decode_b = make_linear_decoder_fn(ckpt_b)

        # --- No-bias decoder ---
        if kind == "nn":
            ckpt_nb = train_nn_no_bias(Y_sis, RAND.copy(), weight_decay=reg)
            decode_nb = make_nn_decoder_fn(ckpt_nb)
        else:
            ckpt_nb = fit_linear_no_bias(Y_sis, RAND.copy(), alpha=reg)
            decode_nb = make_linear_decoder_fn(ckpt_nb)

        for cfg, ckpt, decode, fn in [
            ("baseline", ckpt_b,  decode_b,  cross_xfer_v22),
            ("A",        ckpt_b,  decode_b,  cross_xfer_optA),
            ("B",        ckpt_nb, decode_nb, cross_xfer_v22),
            ("A+B",      ckpt_nb, decode_nb, cross_xfer_optA),
        ]:
            Y_pred, *_ = fn(ckpt, Y_wc, decode)
            rel = rmse_over_range(Y_pred, Y_wc)
            p0min = float(Y_pred[0].min()); p0max = float(Y_pred[0].max())
            print(f"{tag:<28} {cfg:<10}  {rel*100:>9.2f}%  "
                  f"{p0min:>14.3f}  {p0max:>14.3f}")
            plot_per_node(Y_wc, Y_pred, ps, N,
                          f"sis_wc_dolphin_{tag}_{cfg.replace('+', 'p')}.png",
                          f"{label} -- {cfg}.  RMSE/range = {rel*100:.2f}\\%")
            summary.append((tag, cfg, rel, p0min, p0max))
        print()
    return summary


if __name__ == "__main__":
    main()
