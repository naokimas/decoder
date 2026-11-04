# =====================================================================
# SI Section S3: Gappy POD
#
# This script: the gappy POD estimator itself, imported by the other two
# Feeds      : Figs. S3 and S4, Tables S3 and S4, as a module
# Produces   : (module)
# Runtime    : -
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Gappy POD as a drop-in replacement for the decoder.

The reconstruction is the standard one (Everson & Sirovich 1995; Willcox
2006).  Take the truncated SVD of the source training snapshots, assume the
state lies in the span of the leading r left singular vectors, and solve for
the expansion coefficients from the sentinel entries:

    xhat = U_r (U_r)_S^+ s                      (uncentered)
    xhat = mu + U_r (U_r)_S^+ (s - mu_S)        (centered)

Our equilibrium matrices are stored as Y (M x N), one row per
control-parameter value, so the snapshot matrix of SI Section S3 is Y^T and
the basis we need is the set of leading RIGHT singular vectors of Y.

Two normalizations are implemented, because the choice is not obvious:

  "minmax"  the basis is built in the same coordinates the decoders use,
            i.e. after subtracting the scalar training minimum and dividing
            by the scalar training range.  This makes gappy POD differ from
            the linear ridge decoder only in the truncation and the ridge
            penalty.
  "center"  the textbook form: subtract each node's mean over the training
            columns, take the SVD of what is left, and add the mean back.
            The mean is a property of the training data, so using it is
            admissible at deployment.

Both produce a checkpoint with the same keys the NN and linear checkpoints
carry, so the cross-transfer calibration code can be reused unchanged.
"""
import numpy as np


def fit_gappy(Y_full, sensor_idx, train_rows, rank=None, mode="minmax",
              rcond=1e-10):
    Y_train = Y_full[train_rows]
    y_min = float(Y_train.min()); y_max = float(Y_train.max())
    y_range = max(y_max - y_min, 1e-12)
    A = (Y_train - y_min) / y_range                  # M_train x N, normalized
    mu = A.mean(axis=0) if mode == "center" else np.zeros(A.shape[1])
    _, sv, Vt = np.linalg.svd(A - mu, full_matrices=False)
    n = len(sensor_idx)
    if rank is None:
        r = n
    elif isinstance(rank, float):
        # energy criterion: the smallest r holding a fraction `rank` of the
        # squared spectrum, capped at n because n sentinels cannot identify
        # more than n coefficients.  Fixed before any test error is seen.
        e = np.cumsum(sv ** 2) / max(np.sum(sv ** 2), 1e-300)
        r = int(np.searchsorted(e, rank) + 1)
    else:
        r = int(rank)
    r = max(1, min(r, n, Vt.shape[0]))
    basis = Vt[:r].T                                  # N x r
    return {"basis": basis, "mu": mu, "rank": r, "mode": mode,
            "rcond": rcond,
            "sensor_idx": sensor_idx,
            "y_min": y_min, "y_max": y_max, "y_range": y_range,
            "src_sent_min": float(Y_train[:, sensor_idx].min()),
            "src_sent_max": float(Y_train[:, sensor_idx].max()),
            "k_sentinels": n, "n_nodes": Y_full.shape[1]}


def decode_gappy(ckpt):
    """Map normalized sentinel rows (M x n) to normalized full states (M x N)."""
    basis, mu, idx = ckpt["basis"], ckpt["mu"], ckpt["sensor_idx"]
    Bs = basis[idx]                                   # n x r
    # A truncated pseudo-inverse is the regularized form of the coefficient
    # solve.  It matters: with MinCorr or random sentinels the submatrix Bs
    # can have a condition number of 1e19, because only QR pivoting selects
    # sentinels so as to keep it well conditioned (Drmac & Gugercin 2016).
    Bs_pinv = np.linalg.pinv(Bs, rcond=ckpt.get("rcond", 1e-10))
    mu_s = mu[idx]

    def f(Sn):
        coef = (Sn - mu_s) @ Bs_pinv.T                # M x r
        return coef @ basis.T + mu                    # M x N

    return f


def predict_calibrated_gappy(ckpt, target_full, cal_rows):
    """The manuscript's fair-affine deployment, with gappy POD as the decoder.

    Identical to splits.predict_calibrated except for the decode step, so the
    input map of Eq. (29) and the least-squares (a, b) of Eq. (30) are fitted
    on cal_rows exactly as they are for the NN and linear decoders.
    """
    idx = ckpt["sensor_idx"]
    S_all = target_full[:, idx]
    S_cal = S_all[cal_rows]
    y_min_T = float(S_cal.min()); y_max_T = float(S_cal.max())
    y_range_T = max(y_max_T - y_min_T, 1e-12)
    y_min_S_s = ckpt["src_sent_min"]; y_max_S_s = ckpt["src_sent_max"]
    y_range_S_s = max(y_max_S_s - y_min_S_s, 1e-12)
    tilde_raw = y_min_S_s + (S_all - y_min_T) * y_range_S_s / y_range_T
    tilde_norm = (tilde_raw - ckpt["y_min"]) / ckpt["y_range"]
    Yhat = decode_gappy(ckpt)(tilde_norm)
    pred = Yhat[np.ix_(cal_rows, idx)].ravel()
    true = S_cal.ravel()
    Amat = np.column_stack([pred, np.ones_like(pred)])
    (a, b), *_ = np.linalg.lstsq(Amat, true, rcond=None)
    return a * Yhat + b
