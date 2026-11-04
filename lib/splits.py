"""Decoder training with an explicit choice of training rows.

The paper's own trainers (test_zero_anchor.fit_linear_no_bias and
train_nn_no_bias) hard-code the split: shuffle the 51 sweep indices with
default_rng(0) and hold out the first 10.  We need the same decoders trained
on other row subsets -- the first 41 and the last 41 -- so the functions
below take the training rows as an argument.

Everything else is copied verbatim from the paper's versions: the same
normalization by the training min/max, the same ridge solve, the same
network (64 hidden ReLU units, no biases), the same Adam settings, the same
seeds.  verify_against_paper() checks that with the paper's own row set they
return bit-identical checkpoints, so any difference in the results below is
due to the split and to nothing else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths                     # noqa: E402  (puts lib/submitted/ on the path)
import pipeline_unified as pu    # noqa: E402
import test_zero_anchor as tz    # noqa: E402

N_TEST = 10          # round(0.20 * 51), the paper's held-out count


def rows_random():
    """The paper's split: shuffle with seed 0, hold out the first 10."""
    rng = np.random.default_rng(pu.SEED)
    idx = np.arange(51)
    rng.shuffle(idx)
    return np.sort(idx[N_TEST:]), np.sort(idx[:N_TEST])


def rows_first():
    """Train on the first 41 control values, test on the last 10."""
    return np.arange(41), np.arange(41, 51)


def rows_last():
    """Train on the last 41 control values, test on the first 10."""
    return np.arange(10, 51), np.arange(10)


SPLITS = {"random": rows_random, "first": rows_first, "last": rows_last}


def fit_linear(Y_full, sensor_idx, alpha, train_rows):
    Y_train = Y_full[train_rows]
    y_min = float(Y_train.min()); y_max = float(Y_train.max())
    y_range = max(y_max - y_min, 1e-12)
    Xn = (Y_train[:, sensor_idx] - y_min) / y_range
    Yn = (Y_train - y_min) / y_range
    ATA = Xn.T @ Xn + alpha * np.eye(Xn.shape[1])
    W = np.linalg.solve(ATA, Xn.T @ Yn)
    return {"W": W, "b": np.zeros(Y_full.shape[1]), "sensor_idx": sensor_idx,
            "y_min": y_min, "y_max": y_max, "y_range": y_range,
            "src_sent_min": float(Y_train[:, sensor_idx].min()),
            "src_sent_max": float(Y_train[:, sensor_idx].max()),
            "k_sentinels": len(sensor_idx), "n_nodes": Y_full.shape[1],
            "_has_bias": False}


def train_nn(Y_full, sensor_idx, weight_decay, train_rows):
    torch.manual_seed(pu.SEED)
    Y_train = Y_full[train_rows]
    N = Y_full.shape[1]
    y_min = float(Y_train.min()); y_max = float(Y_train.max())
    y_range = max(y_max - y_min, 1e-12)
    Xtr = (Y_train[:, sensor_idx] - y_min) / y_range
    Ytr = (Y_train - y_min) / y_range
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
    model = tz.ShallowDecoderNoBias(len(sensor_idx), pu.HIDDEN, N)
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
            "src_sent_max": float(Y_train[:, sensor_idx].max()),
            "_no_bias": True}


def verify_against_paper(Y, sensor_idx):
    """With the paper's own rows, do we reproduce the paper's decoders?"""
    tr, _ = rows_random()
    a = fit_linear(Y, sensor_idx, 0.01, tr)
    b = tz.fit_linear_no_bias(Y, sensor_idx, 0.01)
    lin_ok = np.array_equal(a["W"], b["W"]) and a["y_min"] == b["y_min"]
    c = train_nn(Y, sensor_idx, 0.0, tr)
    d = tz.train_nn_no_bias(Y, sensor_idx, weight_decay=0.0)
    nn_ok = all(torch.equal(c["model_state_dict"][k], d["model_state_dict"][k])
                for k in c["model_state_dict"])
    return lin_ok, nn_ok


# ---------------------------------------------------------------- target side
def predict_calibrated(ckpt, target_full, dec_type, cal_rows, mode="lsq"):
    """The paper's fair-affine deployment, but calibrated out of sample.

    sweep_v22r_factorial._tilde_norm and .predict_baseline take the target
    sentinel range and the affine (a, b) from ALL M target rows, so the error
    for a target absent from training is still reported on rows that entered
    the calibration.  Here both are computed from `cal_rows` only; the caller
    scores on the complement.  With cal_rows = all rows this reduces exactly
    to the paper's function, which test_reduces_to_paper() checks.
    """
    sensor_idx = ckpt["sensor_idx"]
    S_all = target_full[:, sensor_idx]
    S_cal = S_all[cal_rows]

    # target sentinel extrema from the calibration rows (main text
    # Eqs. 28-29), then the input map of Eq. 30
    y_min_T = float(S_cal.min()); y_max_T = float(S_cal.max())
    y_range_T = max(y_max_T - y_min_T, 1e-12)
    y_min_S_s = ckpt["src_sent_min"]; y_max_S_s = ckpt["src_sent_max"]
    y_range_S_s = max(y_max_S_s - y_min_S_s, 1e-12)
    tilde_raw = y_min_S_s + (S_all - y_min_T) * y_range_S_s / y_range_T
    tilde_norm = (tilde_raw - ckpt["y_min"]) / ckpt["y_range"]

    import sweep_v22r_factorial as sw
    Yhat = sw._decode_fn(ckpt, dec_type)(tilde_norm)

    if mode == "lsq":
        # the output map: (a, b) fitted by least squares on the calibration
        # rows' sentinel entries (the unnumbered display just before main
        # text Eq. 31), then applied to every node as in Eq. 31.
        pred = Yhat[np.ix_(cal_rows, sensor_idx)].flatten()
        true = S_cal.flatten()
        A = np.column_stack([pred, np.ones_like(pred)])
        (a, b), *_ = np.linalg.lstsq(A, true, rcond=None)
        return a * Yhat + b

    if mode == "inv":
        # the alternative: simply invert the input map of Eq. 30 and apply
        # it to every node.  No free parameters -- the two constants follow
        # from the source and target sentinel extrema already used on the
        # input side.  Undo the source normalization, then undo Eq. 30:
        #   u  = zhat * y_range_S + y_min_S                (source raw units)
        #   x^T = y_min_T + (u - src_sent_min) * y_range_T / y_range_S_s
        u = Yhat * ckpt["y_range"] + ckpt["y_min"]
        return y_min_T + (u - y_min_S_s) * y_range_T / y_range_S_s

    raise ValueError("mode must be 'lsq' or 'inv'")


def test_reduces_to_paper(ckpt, target_full, dec_type):
    import sweep_v22r_factorial as sw
    mine = predict_calibrated(ckpt, target_full, dec_type,
                              np.arange(target_full.shape[0]))
    theirs = sw.predict_baseline(ckpt, target_full, dec_type)
    return float(np.abs(mine - theirs).max())
