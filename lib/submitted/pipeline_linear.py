"""Linear-decoder baseline.

Replaces the K -> 64 -> N shallow NN with a single linear map K -> N (trained
in z-scored space, matching the existing pipeline's normalization). Same
sentinels (QR or random), same train/test split, same fair B-fair-affine
deployment for cross-transfer.

Parameter counts (per (network, K)):
    Linear: K*N + N = (K+1)*N
    NN:     K*64 + 64 + 64*N + N = 65*(K + N) + 64

Per network the NN has roughly 10-14x more parameters than the linear model.

Usage:  python3 pipeline_linear.py            (runs all 5 networks)
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import scipy.linalg as sla

HERE = Path(__file__).resolve().parent

NETWORK_FILES = {
    # 14 paper-scale networks (12 dynamics each on disk).
    "dolphin":     "dolphin_network.edgelist",
    "prox":        "prox_network.edgelist",
    "metabolic":   "metabolic_network.edgelist",
    "euroroad":    "euroroad_network.edgelist",
    "email":       "email_network.edgelist",
    "er":          "er_network.edgelist",
    "ba":          "ba_network.edgelist",
    "hk":          "hk_network.edgelist",
    "gkk":         "gkk_network.edgelist",
    "lfr":         "lfr_network.edgelist",
    "flybi":       "flybi_network.edgelist",
    "route_views": "route_views_network.edgelist",
    "spanish":     "spanish_network.edgelist",
    "reactome":    "reactome_network.edgelist",
}
SYSTEMS = ["SIS", "DW", "MUT", "GEN", "WC", "gLV", "SIR", "NM",
           "EUT", "VEG", "MSIS", "MSIR"]
CSV_NAMES = {"SIS": "sis", "DW": "doublewell", "MUT": "mutualistic",
             "GEN": "genereg", "WC": "wc", "gLV": "glv",
             "SIR": "sir", "NM": "noymeir",
             "EUT": "eutroph", "VEG": "vegturb", "MSIS": "msis",
             "MSIR": "msir"}

TEST_FRACTION = 0.20
SEED = 0
RIDGE_ALPHA = 1e-2     # mild ridge regularization in z-scored space; needed for
                       # random sentinels that can be near-colinear and blow up
                       # the unregularized lstsq weights.


def ridge_solve(A, Y, alpha):
    """Solve (A^T A + alpha I) W = A^T Y."""
    ATA = A.T @ A + alpha * np.eye(A.shape[1])
    return np.linalg.solve(ATA, A.T @ Y)


def qr_pivot_sensors(X_train, k):
    _, _, Vt = np.linalg.svd(X_train, full_matrices=False)
    Psi = Vt[:k, :]
    _, _, piv = sla.qr(Psi, pivoting=True, mode="economic")
    return np.sort(piv[:k])


def min_corr_sensors(Y_train, k):
    """Greedy max-min |Pearson correlation| sentinel selection.
    Iteratively add the node whose largest abs correlation with already-selected
    nodes is smallest. O(k * N) per step, total O(k^2 * N).
    """
    Y_c = Y_train - Y_train.mean(axis=0)
    norms = np.linalg.norm(Y_c, axis=0)
    norms = np.where(norms < 1e-12, 1.0, norms)
    Y_n = Y_c / norms
    C = Y_n.T @ Y_n     # N x N correlation matrix (NaN-free, since norms clipped)
    np.fill_diagonal(C, 0.0)
    # Seed with the highest-variance column (it carries the most signal).
    selected = [int(np.argmax(norms))]
    candidates = set(range(Y_train.shape[1])) - set(selected)
    for _ in range(k - 1):
        cand_arr = np.fromiter(candidates, dtype=int)
        max_abs_corr = np.max(np.abs(C[np.array(selected)][:, cand_arr]), axis=0)
        next_idx = int(cand_arr[int(np.argmin(max_abs_corr))])
        selected.append(next_idx)
        candidates.remove(next_idx)
    return np.sort(np.array(selected))


def fit_linear(Y_full, sensor_idx, seed=SEED):
    rng = np.random.default_rng(seed)
    M, N = Y_full.shape
    idx = np.arange(M); rng.shuffle(idx)
    n_test = int(round(TEST_FRACTION * M))
    Y_train = Y_full[np.sort(idx[n_test:])]
    Y_test  = Y_full[np.sort(idx[:n_test])]
    X_train = Y_train[:, sensor_idx]
    x_mean = X_train.mean(axis=0); x_std = np.where(X_train.std(axis=0) < 1e-12, 1.0, X_train.std(axis=0))
    y_mean = Y_train.mean(axis=0); y_std = np.where(Y_train.std(axis=0) < 1e-12, 1.0, Y_train.std(axis=0))
    Xz = (X_train - x_mean) / x_std
    Yz = (Y_train - y_mean) / y_std
    A = np.column_stack([Xz, np.ones(Xz.shape[0])])
    Wb = ridge_solve(A, Yz, RIDGE_ALPHA)
    W = Wb[:-1, :]; b = Wb[-1, :]
    Xte = (Y_test[:, sensor_idx] - x_mean) / x_std
    Yte_pred = (Xte @ W + b) * y_std + y_mean
    test_mse = float(np.mean((Yte_pred - Y_test) ** 2))
    return {"W": W, "b": b, "k_sentinels": len(sensor_idx), "n_nodes": N,
            "sensor_idx": sensor_idx,
            "x_mean": x_mean, "x_std": x_std,
            "y_mean": y_mean, "y_std": y_std,
            "test_mse": test_mse}


def fit_combined(eq_dict, sensor_idx, sys_tags=tuple(SYSTEMS), seed=SEED):
    rng = np.random.default_rng(seed)
    train_blocks, test_blocks, test_labels = [], [], []
    for sys_tag in sys_tags:
        Y = eq_dict[sys_tag]
        M = Y.shape[0]; idx = np.arange(M); rng.shuffle(idx)
        n_test = int(round(TEST_FRACTION * M))
        train_blocks.append(Y[np.sort(idx[n_test:])])
        test_blocks.append(Y[np.sort(idx[:n_test])])
        test_labels.extend([sys_tag] * n_test)
    Y_train = np.vstack(train_blocks); Y_test = np.vstack(test_blocks)
    test_labels = np.array(test_labels)
    X_train = Y_train[:, sensor_idx]
    x_mean = X_train.mean(axis=0); x_std = np.where(X_train.std(axis=0) < 1e-12, 1.0, X_train.std(axis=0))
    y_mean = Y_train.mean(axis=0); y_std = np.where(Y_train.std(axis=0) < 1e-12, 1.0, Y_train.std(axis=0))
    Xz = (X_train - x_mean) / x_std
    Yz = (Y_train - y_mean) / y_std
    A = np.column_stack([Xz, np.ones(Xz.shape[0])])
    Wb = ridge_solve(A, Yz, RIDGE_ALPHA)
    W = Wb[:-1, :]; b = Wb[-1, :]
    Xte = (Y_test[:, sensor_idx] - x_mean) / x_std
    Yte_pred = (Xte @ W + b) * y_std + y_mean
    test_mse = float(np.mean((Yte_pred - Y_test) ** 2))
    per_sys = {tag: float(np.mean((Yte_pred[test_labels == tag]
                                    - Y_test[test_labels == tag]) ** 2))
               for tag in sys_tags}
    return {"W": W, "b": b, "k_sentinels": len(sensor_idx), "n_nodes": Y_train.shape[1],
            "sensor_idx": sensor_idx,
            "x_mean": x_mean, "x_std": x_std,
            "y_mean": y_mean, "y_std": y_std,
            "test_mse": test_mse, "per_system_test_mse": per_sys}


def fair_predict(ckpt, target_full):
    sensor_idx = ckpt["sensor_idx"]
    S = target_full[:, sensor_idx]
    mu_x = S.mean(axis=0); sd_x = S.std(axis=0); sd_x = np.where(sd_x < 1e-12, 1.0, sd_x)
    Sn = (S - mu_x) / sd_x
    Y_src = (Sn @ ckpt["W"] + ckpt["b"]) * ckpt["y_std"] + ckpt["y_mean"]
    pred_sent = Y_src[:, sensor_idx].flatten()
    true_sent = S.flatten()
    A_design = np.column_stack([pred_sent, np.ones_like(pred_sent)])
    (a, b), *_ = np.linalg.lstsq(A_design, true_sent, rcond=None)
    return a * Y_src + b, float(a), float(b)


def n_params_linear(K, N):
    return K * N + N            # W: K*N, b: N

def n_params_nn(K, N, hidden=64):
    return K * hidden + hidden + hidden * N + N


def run_one(tag):
    G = nx.read_edgelist(HERE / NETWORK_FILES[tag], nodetype=int)
    N = len(G.nodes())
    K = int(np.floor(np.log(N)))
    print(f"\n===== {tag.upper()}: N={N}, K={K} =====")
    print(f"  parameters per decoder:  linear={n_params_linear(K, N):>6d}, "
          f"NN={n_params_nn(K, N):>6d}, ratio NN/linear={n_params_nn(K, N)/n_params_linear(K, N):.2f}")

    # Load equilibrium data
    eq = {}
    for s in SYSTEMS:
        arr = np.loadtxt(HERE / f"{tag}_{CSV_NAMES[s]}_equilibrium_data.csv",
                         delimiter=",", skiprows=1)
        eq[s] = arr[:, 1:]

    # Per-system QR + random sentinels
    rng_random = np.random.default_rng(2026)
    RANDOM_SENTINELS = np.sort(rng_random.choice(N, size=K, replace=False))

    ckpts_qr = {}; ckpts_rd = {}; ckpts_mc = {}
    for s in SYSTEMS:
        Y = eq[s]; M = Y.shape[0]
        rng = np.random.default_rng(SEED); idx = np.arange(M); rng.shuffle(idx)
        n_test = int(round(TEST_FRACTION * M))
        Y_train_for_pick = Y[np.sort(idx[n_test:])]
        qr_idx = qr_pivot_sensors(Y_train_for_pick, K)
        mc_idx = min_corr_sensors(Y_train_for_pick, K)
        ckpts_qr[s] = fit_linear(Y, qr_idx)
        ckpts_rd[s] = fit_linear(Y, RANDOM_SENTINELS.copy())
        ckpts_mc[s] = fit_linear(Y, mc_idx)

    # COMBINED: QR + random + min-corr
    rng = np.random.default_rng(SEED)
    train_blocks = []
    for s in SYSTEMS:
        Y = eq[s]; M = Y.shape[0]; idx = np.arange(M); rng.shuffle(idx)
        n_test = int(round(TEST_FRACTION * M))
        train_blocks.append(Y[np.sort(idx[n_test:])])
    Y_train_c = np.vstack(train_blocks)
    comb_qr_idx = qr_pivot_sensors(Y_train_c, K)
    comb_mc_idx = min_corr_sensors(Y_train_c, K)
    ckpts_qr["COMBINED"] = fit_combined(eq, comb_qr_idx)
    ckpts_rd["COMBINED"] = fit_combined(eq, RANDOM_SENTINELS.copy())
    ckpts_mc["COMBINED"] = fit_combined(eq, comb_mc_idx)

    # Cross matrix
    SOURCES = SYSTEMS + ["COMBINED"]
    TARGETS = SYSTEMS

    def run_matrix(ckpts, variant: str):
        rows = ["src -> tgt    MSE         RMSE      RMSE/range   relRMSE_v12   a       b   "]
        mse_grid = np.full((len(SOURCES), len(TARGETS)), np.nan)
        rel_grid = np.full((len(SOURCES), len(TARGETS)), np.nan)
        rel_v12_grid = np.full((len(SOURCES), len(TARGETS)), np.nan)
        rel_v13_grid = np.full((len(SOURCES), len(TARGETS)), np.nan)
        rel_v14_grid = np.full((len(SOURCES), len(TARGETS)), np.nan)
        a_grid = np.full((len(SOURCES), len(TARGETS)), np.nan)
        b_grid = np.full((len(SOURCES), len(TARGETS)), np.nan)
        for i, src in enumerate(SOURCES):
            sensor_idx = ckpts[src]["sensor_idx"]
            for j, tgt in enumerate(TARGETS):
                Y_true = eq[tgt]
                Y_pred, a, b = fair_predict(ckpts[src], Y_true)
                mse = float(np.mean((Y_pred - Y_true) ** 2))
                rmse = float(np.sqrt(mse))
                dyn = float(Y_true.max() - Y_true.min())
                rel = rmse / dyn if dyn > 0 else float("nan")
                row_range = Y_true.max(axis=1) - Y_true.min(axis=1)
                keep = row_range >= 1e-3
                if keep.any():
                    Yk = Y_true[keep]; Yp = Y_pred[keep]
                    base = Yk[:, sensor_idx].mean(axis=1, keepdims=True)
                    num = float(np.sum((Yp - Yk) ** 2))
                    den = float(np.sum((base - Yk) ** 2))
                    rel_v12 = float(np.sqrt(num / den)) if den > 0 else float("nan")
                    num_per_row = np.sum((Yp - Yk) ** 2, axis=1)
                    den_per_row = np.sum((base - Yk) ** 2, axis=1)
                    valid = den_per_row > 0
                    if valid.any():
                        ratios = num_per_row[valid] / den_per_row[valid]
                        rel_v13 = float(np.sqrt(ratios.mean()))
                    else:
                        rel_v13 = float("nan")
                else:
                    rel_v12 = float("nan")
                    rel_v13 = float("nan")
                row_maxabs = np.max(np.abs(Y_true), axis=1)
                keep14 = row_maxabs > 1e-2
                if keep14.any():
                    Yk14 = Y_true[keep14]; Yp14 = Y_pred[keep14]
                    base14 = Yk14[:, sensor_idx].mean(axis=1, keepdims=True)
                    num14 = np.sum((Yp14 - Yk14) ** 2, axis=1)
                    den14 = np.sum((base14 - Yk14) ** 2, axis=1)
                    valid14 = den14 > 0
                    if valid14.any():
                        rel_v14 = float(np.sqrt((num14[valid14] / den14[valid14]).mean()))
                    else:
                        rel_v14 = float("nan")
                else:
                    rel_v14 = float("nan")
                mse_grid[i, j] = mse; rel_grid[i, j] = rel
                rel_v12_grid[i, j] = rel_v12
                rel_v13_grid[i, j] = rel_v13
                rel_v14_grid[i, j] = rel_v14
                a_grid[i, j] = a; b_grid[i, j] = b
                rows.append(f"  {src:>8s} -> {tgt:<3s}  {mse:>10.4f}  "
                            f"{rmse:>8.4f}  {rel:>9.3f}   {rel_v12:>9.3f}   "
                            f"{a:>6.3f}  {b:>6.3f}")
        out_txt = HERE / f"{tag}_linear_cross_matrix_summary_{variant}.txt"
        out_txt.write_text("\n".join(rows) + "\n")
        np.savez(HERE / f"{tag}_linear_cross_matrix_data_{variant}.npz",
                 mse=mse_grid, rel=rel_grid,
                 rel_v12=rel_v12_grid, rel_v13=rel_v13_grid,
                 rel_v14=rel_v14_grid,
                 a=a_grid, b=b_grid,
                 sources=np.array(SOURCES, dtype="U16"),
                 targets=np.array(TARGETS, dtype="U16"),
                 N=N, K=K)

    run_matrix(ckpts_qr, "qr")
    run_matrix(ckpts_rd, "random")
    run_matrix(ckpts_mc, "minc")

    # Off-diag mean comparison
    for variant in ["qr", "random", "minc"]:
        d_lin = np.load(HERE / f"{tag}_linear_cross_matrix_data_{variant}.npz")
        rel_lin = d_lin["rel"]
        s = list(d_lin["sources"]); t = list(d_lin["targets"])
        off_lin = np.mean([rel_lin[i, j] for i, si in enumerate(s)
                           for j, tj in enumerate(t) if si != tj])
        if variant != "minc":
            d_nn = np.load(HERE / f"{tag}_cross_matrix_data_{variant}.npz")
            rel_nn = d_nn["rel"]
            off_nn = np.mean([rel_nn[i, j] for i, si in enumerate(s)
                              for j, tj in enumerate(t) if si != tj])
            print(f"  {variant:>6s}: linear={off_lin*100:6.2f}%, "
                  f"NN={off_nn*100:6.2f}%, diff={(off_lin-off_nn)*100:+6.2f}pp")
        else:
            print(f"  {variant:>6s}: linear={off_lin*100:6.2f}%  (no NN counterpart)")


def main():
    print("LINEAR DECODER BASELINE\n")
    for tag in NETWORK_FILES:
        run_one(tag)


if __name__ == "__main__":
    main()
