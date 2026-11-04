"""Unified pipeline: one network -> all 6 dynamics + 7 decoders + 2 cross-matrices.

Usage:  python3 pipeline_unified.py <NETWORK_TAG> [KMULT]
        NETWORK_TAG in {ba, lfr, dolphin, gkk, prox}
        KMULT (optional, default 1) -- sentinel count = KMULT * floor(ln N).

For each network, we:
  1. Load the edgelist; compute K = KMULT * floor(ln N) and alpha_max.
  2. Generate equilibrium sweeps for SIS, DW, MUT, GEN, WC, gLV (51 points each).
     If the equilibrium CSVs already exist (KMULT-independent), reuse them.
  3. Train 6 per-system decoders + 1 COMBINED decoder, twice each
     (QR-pivoted sentinels and uniformly-random sentinels). 7 x 2 = 14 decoders.
  4. Build the 7 x 6 fair B-fair-affine cross-transfer matrix for each variant.
  5. Save (with file prefix `<tag>` if KMULT==1, else `<tag>_k{KMULT}x`):
        <prefix>_<sys>_equilibrium_data.csv       (6 files; KMULT==1 only)
        <prefix>_<sys>_qr_decoder_checkpoint.pt   (6 files)
        <prefix>_<sys>_random_decoder_checkpoint.pt (6 files)
        <prefix>_combined_qr_decoder_checkpoint.pt   (1)
        <prefix>_combined_random_decoder_checkpoint.pt (1)
        <prefix>_cross_matrix_summary_qr.txt
        <prefix>_cross_matrix_summary_random.txt
        <prefix>_bifurcations_overview.png        (6-panel diagnostic; KMULT==1)
        <prefix>_meta.txt

WC dynamics (Cowan-Wilson, Laurence et al. PRX 2019, Eq. 2a):
    dx_i/dt = -x_i + D * sum_j w_ij / (1 + exp(-tau (x_j - mu)))
with tau=1, mu=3 (Laurence et al. defaults).

gLV dynamics (Masuda et al. spectral-nonleading; matches Laurence Eq. 2b after
self-stabilization with w_ii = -(alpha_max + 1) / D):
    x* = -lam_bar (D W - c I)^{-1} 1,    c = alpha_max + 1, W has zero diagonal.
We sweep D in [0.05, 0.95] * D_c where D_c = (alpha_max + 1) / alpha_max.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
from scipy.integrate import solve_ivp
from torch import nn

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
    # Legacy tag retained to keep older cross-matrix outputs reachable.
    "email_uni":   "email_uni_network.edgelist",
}


def load_network(tag: str):
    G = nx.read_edgelist(HERE / NETWORK_FILES[tag], nodetype=int)
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")
    N = A.shape[0]
    K = int(np.floor(np.log(N)))
    if N < 200:
        alpha_max = float(np.linalg.eigvalsh(A.toarray()).max())
    else:
        alpha_max = float(eigsh(A, k=1, which="LA",
                                return_eigenvectors=False)[0])
    edges_uv = [(min(u, v), max(u, v)) for u, v in G.edges()]
    es = np.array([u for u, _ in edges_uv]); ed = np.array([v for _, v in edges_uv])
    return N, K, alpha_max, A, es, ed


# --- Dynamics RHS factories (closed over (A, es, ed)) -------------------
def make_sis_rhs(A):
    MU = 1.0
    def rhs(t, x, lam): return -MU * x + lam * (1.0 - x) * (A @ x)
    return rhs

def make_dw_rhs(A):
    R1, R2, R3 = 1.0, 3.0, 5.0
    def rhs(t, x, D): return -(x - R1) * (x - R2) * (x - R3) + D * (A @ x)
    return rhs

def make_mut_rhs(es, ed):
    B, K_, C = 0.1, 5.0, 1.0
    DTILDE, E_, H_ = 5.0, 0.9, 0.1
    def rhs(t, x, D):
        self_term = B + x * (1.0 - x / K_) * (x / C - 1.0)
        xs = x[es]; xd = x[ed]
        c_sd = xs * xd / (DTILDE + E_ * xs + H_ * xd)
        c_ds = xd * xs / (DTILDE + E_ * xd + H_ * xs)
        c = np.zeros_like(x)
        np.add.at(c, es, c_sd)
        np.add.at(c, ed, c_ds)
        return self_term + D * c
    return rhs

def make_gen_rhs(A):
    B, F_, H_ = 1.0, 1.0, 2.0
    def rhs(t, x, D):
        x_pos = np.maximum(x, 0.0)
        hill = (x_pos ** H_) / (1.0 + x_pos ** H_)
        return -B * (x_pos ** F_) + D * (A @ hill)
    return rhs

def make_wc_rhs(A):
    TAU, MU = 1.0, 3.0
    def rhs(t, x, D):
        sig = 1.0 / (1.0 + np.exp(-TAU * (x - MU)))
        return -x + D * (A @ sig)
    return rhs


def make_sir_rhs(A):
    """Mean-field SIR per node.  state = [S, I] (R = 1 - S - I implicit)."""
    GAMMA = 1.0
    def rhs(t, state, beta):
        N = state.size // 2
        S = state[:N]; I = state[N:]
        AI = A @ I
        dS = -beta * S * AI
        dI =  beta * S * AI - GAMMA * I
        return np.concatenate([dS, dI])
    return rhs


def make_noymeir_rhs(A, degrees, r_vec):
    """Noy-Meir / May 1977 grazing model with diffusive network coupling.
    dX_i/dt = r_i*X_i*(1 - X_i/K) - c*X_i^2/(X_i^2+1) + D*sum_j A_ij*(X_j - X_i).
    Per-node growth rate r_i in [0.6, 1.0] gives spatial heterogeneity (Dakos).
    """
    K_, D_DISP = 10.0, 0.05
    def rhs(t, x, c):
        return (r_vec * x * (1.0 - x / K_)
                - c * x * x / (x * x + 1.0)
                + D_DISP * (A @ x - degrees * x))
    return rhs


def make_eutroph_rhs(A, degrees, b_vec):
    """Eutrophication model (Carpenter 1999; Dakos 2010 Table 1, model 2).
    dX_i/dt = alpha - b_i*X_i + r*X_i^p/(X_i^p+1) + D*sum_j A_ij*(X_j - X_i).
    Per-node loss rate b_i in [0.8, 1.2] gives spatial heterogeneity.
    Control parameter alpha in [0.1, 0.9].  r=1, p=8.
    """
    R_RECYC, P_HILL, D_DISP = 1.0, 8.0, 0.05
    def rhs(t, x, alpha):
        x_pos = np.maximum(x, 0.0)
        x_p   = x_pos ** P_HILL
        return (alpha - b_vec * x
                + R_RECYC * x_p / (x_p + 1.0)
                + D_DISP * (A @ x - degrees * x))
    return rhs


def make_vegturb_rhs(A, degrees, hE_vec):
    """Vegetation-turbidity model (Scheffer 1998; Dakos 2010 Table 1, model 3).
    dV_i/dt = rV*V_i*(1 - V_i*(hE_i^p + E_i^p)/hE_i^p) + D*sum_j A_ij*(V_j - V_i),
    E_i = Eo * hV / (hV + V_i),
    rV=0.5, hV=0.2, p=4.  Control parameter Eo in [2, 12].
    Per-node hE_i in [1, 3] gives spatial heterogeneity.
    """
    RV, HV, P_HILL, D_DISP = 0.5, 0.2, 4.0, 0.05
    hE_p = hE_vec ** P_HILL
    def rhs(t, V, Eo):
        V_pos = np.maximum(V, 0.0)
        E = Eo * HV / (HV + V_pos)
        E_p = E ** P_HILL
        return (RV * V_pos * (1.0 - V_pos * (hE_p + E_p) / hE_p)
                + D_DISP * (A @ V - degrees * V))
    return rhs


def make_metapop_sis_rhs(A, k_vec):
    """Metapopulation SIS (metapop.tex Eqs 1, 2). state = [rho_S, rho_I].
    Random-walk diffusion with rate D_S, D_I; recovery rate mu=1.
    """
    MU, D_S, D_I = 1.0, 1.0, 1.0
    def rhs(t, state, beta):
        N = state.size // 2
        rS = state[:N]; rI = state[N:]
        # column-normalized incoming flux: M @ rho where M[i,j]=A[i,j]/k_j
        diff_S = D_S * (A @ (rS / k_vec) - rS)
        diff_I = D_I * (A @ (rI / k_vec) - rI)
        infect = beta * rS * rI
        recov  = MU * rI
        return np.concatenate([-infect + recov + diff_S,
                                infect - recov + diff_I])
    return rhs


def make_metapop_sir_rhs(A, k_vec):
    """Metapopulation SIR (assumed: recovered individuals never move, D_R = 0).
    state = [rho_S, rho_I, rho_R].
    """
    MU, D_S, D_I = 1.0, 1.0, 1.0
    def rhs(t, state, beta):
        N = state.size // 3
        rS = state[:N]; rI = state[N:2*N]    # rR = state[2*N:] not needed in RHS
        diff_S = D_S * (A @ (rS / k_vec) - rS)
        diff_I = D_I * (A @ (rI / k_vec) - rI)
        infect = beta * rS * rI
        recov  = MU * rI
        return np.concatenate([-infect + diff_S,
                                infect - recov + diff_I,
                                recov])
    return rhs


def equilibrate(rhs, p, x0, t_final=200.0):
    sol = solve_ivp(rhs, (0.0, t_final), x0, method="LSODA",
                    args=(p,), rtol=1e-7, atol=1e-9)
    if not sol.success:
        raise RuntimeError(f"ODE failed at p={p}: {sol.message}")
    return sol.y[:, -1]


def sweep_ode(rhs, ps, x0_func, label, N):
    t0 = time.time()
    eq = np.empty((len(ps), N))
    for k, p in enumerate(ps):
        x0 = x0_func(p) if callable(x0_func) else x0_func
        eq[k] = equilibrate(rhs, p, x0.copy())
    dt = time.time() - t0
    print(f"    {label}: {len(ps)} points in {dt:.1f}s, "
          f"y-range [{eq.min():.3f}, {eq.max():.3f}]")
    return eq


def sweep_sir(rhs, betas, N, t_final=400.0, i0=0.01):
    """SIR sweep: integrate (S, I) until I -> 0; record final R = 1 - S."""
    t0 = time.time()
    eq = np.empty((len(betas), N))
    for k, beta in enumerate(betas):
        S0 = np.full(N, 1.0 - i0); I0 = np.full(N, i0)
        state0 = np.concatenate([S0, I0])
        sol = solve_ivp(rhs, (0.0, t_final), state0, method="LSODA",
                        args=(beta,), rtol=1e-7, atol=1e-9)
        if not sol.success:
            raise RuntimeError(f"SIR ODE failed at beta={beta}: {sol.message}")
        S_final = sol.y[:N, -1]
        eq[k] = 1.0 - S_final
    print(f"    SIR: {len(betas)} points in {time.time()-t0:.1f}s, "
          f"y-range [{eq.min():.3f}, {eq.max():.3f}]")
    return eq


def sweep_metapop_sis(rhs, betas, N, t_final=400.0, i0=0.01):
    """Metapop SIS sweep: integrate (rho_S, rho_I) to equilibrium; record rho_I."""
    t0 = time.time()
    eq = np.empty((len(betas), N))
    for k, beta in enumerate(betas):
        S0 = np.full(N, 1.0 - i0); I0 = np.full(N, i0)
        state0 = np.concatenate([S0, I0])
        sol = solve_ivp(rhs, (0.0, t_final), state0, method="LSODA",
                        args=(beta,), rtol=1e-7, atol=1e-9)
        if not sol.success:
            raise RuntimeError(f"MSIS ODE failed at beta={beta}: {sol.message}")
        eq[k] = sol.y[N:2*N, -1]
    print(f"    MSIS: {len(betas)} points in {time.time()-t0:.1f}s, "
          f"y-range [{eq.min():.3f}, {eq.max():.3f}]")
    return eq


def sweep_metapop_sir(rhs, betas, N, t_final=400.0, i0=0.01):
    """Metapop SIR sweep: integrate (rho_S, rho_I, rho_R) to extinction; record rho_R."""
    t0 = time.time()
    eq = np.empty((len(betas), N))
    for k, beta in enumerate(betas):
        S0 = np.full(N, 1.0 - i0); I0 = np.full(N, i0); R0 = np.zeros(N)
        state0 = np.concatenate([S0, I0, R0])
        sol = solve_ivp(rhs, (0.0, t_final), state0, method="LSODA",
                        args=(beta,), rtol=1e-7, atol=1e-9)
        if not sol.success:
            raise RuntimeError(f"MSIR ODE failed at beta={beta}: {sol.message}")
        eq[k] = sol.y[2*N:3*N, -1]
    print(f"    MSIR: {len(betas)} points in {time.time()-t0:.1f}s, "
          f"y-range [{eq.min():.3f}, {eq.max():.3f}]")
    return eq


def trim_grid(grid, Y, threshold=0.05, max_flat_frac=0.20):
    """Return a new grid that bounds the flat region (mean<threshold) to
    <= max_flat_frac of the swept range, or None if no trim is needed.
    Detects flat at low or high end."""
    means = Y.mean(axis=1)
    flat_mask = means < threshold
    n = len(grid); flat_count = int(flat_mask.sum())
    if flat_count == 0 or flat_count <= max_flat_frac * n:
        return None
    if flat_mask[0] and not flat_mask[-1]:
        first_nonflat = int(np.where(~flat_mask)[0][0])
        p_t = grid[first_nonflat]
        new_p_min = max(0.0,
                        (p_t - max_flat_frac * grid[-1]) / (1 - max_flat_frac))
        if new_p_min >= grid[-1]:
            return None
        return np.linspace(new_p_min, grid[-1], n)
    if flat_mask[-1] and not flat_mask[0]:
        last_nonflat = int(np.where(~flat_mask)[0][-1])
        p_t = grid[last_nonflat]
        new_p_max = (p_t - max_flat_frac * grid[0]) / (1 - max_flat_frac)
        if new_p_max <= grid[0]:
            return None
        return np.linspace(grid[0], new_p_max, n)
    return None


def glv_sweep(A_dense, alpha_max, D_grid, lam_bar=1.0):
    """Closed-form gLV sweep. W has zero diagonal."""
    N = A_dense.shape[0]
    c = alpha_max + 1.0
    eq = np.empty((len(D_grid), N))
    eye = np.eye(N)
    for k, D in enumerate(D_grid):
        M = D * A_dense - c * eye
        x = -lam_bar * np.linalg.solve(M, np.ones(N))
        eq[k] = x
    print(f"    gLV: {len(D_grid)} points (closed-form), "
          f"y-range [{eq.min():.3f}, {eq.max():.3f}]")
    return eq


# --- Decoder ------------------------------------------------------------
class ShallowDecoder(nn.Module):
    def __init__(self, k_in, hidden, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out),
        )
    def forward(self, x): return self.net(x)


HIDDEN = 64
EPOCHS = 200
LR = 1e-2
TEST_FRACTION = 0.20
SEED = 0


def qr_pivot_sensors(X_train, k):
    _, _, Vt = np.linalg.svd(X_train, full_matrices=False)
    Psi = Vt[:k, :]
    _, _, piv = sla.qr(Psi, pivoting=True, mode="economic")
    return np.sort(piv[:k])


def min_corr_sensors(Y_train, k):
    """Greedy max-min |Pearson correlation| sentinel selection. O(k^2 N)."""
    Y_c = Y_train - Y_train.mean(axis=0)
    norms = np.linalg.norm(Y_c, axis=0)
    norms = np.where(norms < 1e-12, 1.0, norms)
    Y_n = Y_c / norms
    C = Y_n.T @ Y_n
    np.fill_diagonal(C, 0.0)
    selected = [int(np.argmax(norms))]
    candidates = set(range(Y_train.shape[1])) - set(selected)
    for _ in range(k - 1):
        cand_arr = np.fromiter(candidates, dtype=int)
        max_abs = np.max(np.abs(C[np.array(selected)][:, cand_arr]), axis=0)
        nxt = int(cand_arr[int(np.argmin(max_abs))])
        selected.append(nxt); candidates.remove(nxt)
    return np.sort(np.array(selected))


def train_one(Y_full, sensor_idx, epochs=EPOCHS, lr=LR):
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    M, N = Y_full.shape
    idx = np.arange(M); rng.shuffle(idx)
    n_test = int(round(TEST_FRACTION * M))
    Y_test = Y_full[np.sort(idx[:n_test])]
    Y_train = Y_full[np.sort(idx[n_test:])]
    X_train = Y_train[:, sensor_idx]; X_test = Y_test[:, sensor_idx]
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    y_mean, y_std = Y_train.mean(axis=0), Y_train.std(axis=0)
    x_std = np.where(x_std < 1e-12, 1.0, x_std)
    y_std = np.where(y_std < 1e-12, 1.0, y_std)
    Xtr = torch.tensor((X_train - x_mean) / x_std, dtype=torch.float32)
    Ytr = torch.tensor((Y_train - y_mean) / y_std, dtype=torch.float32)
    Xte = torch.tensor((X_test - x_mean) / x_std, dtype=torch.float32)
    model = ShallowDecoder(len(sensor_idx), HIDDEN, N)
    opt = torch.optim.Adam(model.parameters(), lr=lr); loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); loss_fn(model(Xtr), Ytr).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Yte_pred = model(Xte).numpy() * y_std + y_mean
    return {
        "model_state_dict": model.state_dict(),
        "k_sentinels": len(sensor_idx), "hidden": HIDDEN, "n_nodes": N,
        "sensor_idx": sensor_idx,
        "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
        "test_mse": float(np.mean((Yte_pred - Y_test) ** 2)),
    }


def train_combined(Y_train, Y_test, test_labels, sensor_idx,
                   epochs=400, lr=5e-3, sys_tags=("SIS","DW","MUT","GEN","WC","gLV")):
    torch.manual_seed(SEED)
    N = Y_train.shape[1]
    X_train = Y_train[:, sensor_idx]; X_test = Y_test[:, sensor_idx]
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    y_mean, y_std = Y_train.mean(axis=0), Y_train.std(axis=0)
    x_std = np.where(x_std < 1e-12, 1.0, x_std)
    y_std = np.where(y_std < 1e-12, 1.0, y_std)
    Xtr = torch.tensor((X_train - x_mean) / x_std, dtype=torch.float32)
    Ytr = torch.tensor((Y_train - y_mean) / y_std, dtype=torch.float32)
    Xte = torch.tensor((X_test  - x_mean) / x_std, dtype=torch.float32)
    model = ShallowDecoder(len(sensor_idx), HIDDEN, N)
    opt = torch.optim.Adam(model.parameters(), lr=lr); loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); loss_fn(model(Xtr), Ytr).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Yte_pred = model(Xte).numpy() * y_std + y_mean
    per_sys = {tag: float(np.mean((Yte_pred[test_labels == tag]
                                   - Y_test[test_labels == tag]) ** 2))
               for tag in sys_tags}
    return {
        "model_state_dict": model.state_dict(),
        "k_sentinels": len(sensor_idx), "hidden": HIDDEN, "n_nodes": N,
        "sensor_idx": sensor_idx,
        "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
        "test_mse": float(np.mean((Yte_pred - Y_test) ** 2)),
        "per_system_test_mse": per_sys,
    }


def fair_predict(ckpt, target_full):
    sensor_idx = ckpt["sensor_idx"]
    S = target_full[:, sensor_idx]
    mu_x = S.mean(axis=0); sd_x = S.std(axis=0)
    sd_x = np.where(sd_x < 1e-12, 1.0, sd_x)
    Sn = (S - mu_x) / sd_x
    model = ShallowDecoder(ckpt["k_sentinels"], ckpt["hidden"], ckpt["n_nodes"])
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    with torch.no_grad():
        Yn = model(torch.tensor(Sn, dtype=torch.float32)).numpy()
    Y_src = Yn * ckpt["y_std"] + ckpt["y_mean"]
    pred_sent = Y_src[:, sensor_idx].flatten()
    true_sent = S.flatten()
    A_design = np.column_stack([pred_sent, np.ones_like(pred_sent)])
    (a, b), *_ = np.linalg.lstsq(A_design, true_sent, rcond=None)
    return a * Y_src + b, float(a), float(b)


def kmult_label(kmult: float) -> str:
    """Output-file label for a given kmult. Empty for kmult==1."""
    if kmult == 1:
        return ""
    if float(kmult).is_integer():
        return f"_k{int(kmult)}x"
    return f"_k{str(kmult).replace('.', 'p')}x"


# --- Main ---------------------------------------------------------------
def main(tag: str, kmult: float = 1.0):
    label = kmult_label(kmult)
    prefix = tag if kmult == 1 else f"{tag}{label}"
    print(f"\n======= pipeline_unified: {prefix.upper()} (kmult={kmult}) =======")
    N, K_base, alpha_max, A_sparse, es, ed = load_network(tag)
    K = max(1, int(round(kmult * K_base)))
    if K > N:
        sys.exit(f"K={K} exceeds N={N}; pick a smaller kmult")
    A_dense = A_sparse.toarray()
    print(f"  N={N}, K={K} (round({kmult} * floor ln N = {kmult}*{K_base})), "
          f"alpha_max={alpha_max:.3f}")

    rng_rand = np.random.default_rng(2026)
    RANDOM_SENTINELS = np.sort(rng_rand.choice(N, size=K, replace=False))
    print(f"  random sentinels (1-based): {(RANDOM_SENTINELS + 1).tolist()}")

    # --- Bifurcation parameter ranges (per-network calibration) ---
    csv_names = {"SIS": "sis", "DW": "doublewell", "MUT": "mutualistic",
                 "GEN": "genereg", "WC": "wc", "gLV": "glv",
                 "SIR": "sir", "NM": "noymeir",
                 "EUT": "eutroph", "VEG": "vegturb", "MSIS": "msis",
                 "MSIR": "msir"}
    NSWEEP = 51
    sis_grid  = np.linspace(0.0, 1.0, NSWEEP)
    dw_grid   = np.linspace(0.0, 1.0, NSWEEP)
    mut_grid  = np.linspace(0.0, 3.0, NSWEEP)
    gen_grid  = np.linspace(0.0, 1.0, NSWEEP)
    wc_dmax   = float(max(3.0, min(50.0 / alpha_max, 8.0)))
    wc_grid   = np.linspace(0.0, wc_dmax, NSWEEP)
    Dc_glv    = (alpha_max + 1.0) / alpha_max
    glv_grid  = np.linspace(0.05 * Dc_glv, 0.95 * Dc_glv, NSWEEP)
    sir_grid  = np.linspace(0.02, 1.0, NSWEEP)
    nm_grid   = np.linspace(1.0, 3.0, NSWEEP)
    eut_grid  = np.linspace(0.1, 0.9, NSWEEP)
    veg_grid  = np.linspace(2.0, 12.0, NSWEEP)
    msis_grid = np.linspace(0.05, 1.5, NSWEEP)
    msir_grid = np.linspace(0.05, 1.5, NSWEEP)
    grids = {"SIS": sis_grid, "DW": dw_grid, "MUT": mut_grid,
             "GEN": gen_grid, "WC": wc_grid, "gLV": glv_grid,
             "SIR": sir_grid, "NM": nm_grid,
             "EUT": eut_grid, "VEG": veg_grid, "MSIS": msis_grid,
             "MSIR": msir_grid}

    SYSTEMS = ["SIS", "DW", "MUT", "GEN", "WC", "gLV", "SIR", "NM",
               "EUT", "VEG", "MSIS", "MSIR"]

    # --- Per-system equilibrium CSV reuse ---
    csv_paths = {sys_tag: HERE / f"{tag}_{csv_names[sys_tag]}_equilibrium_data.csv"
                 for sys_tag in SYSTEMS}
    eq = {}
    missing = [s for s in SYSTEMS if not csv_paths[s].exists()]
    if not missing:
        print("  reusing all existing equilibrium CSVs")
    else:
        print(f"  generating sweeps for: {missing}")
    for sys_tag in SYSTEMS:
        if sys_tag not in missing:
            arr = np.loadtxt(csv_paths[sys_tag], delimiter=",", skiprows=1)
            eq[sys_tag] = arr[:, 1:]

    if missing:
        SIS_RHS = make_sis_rhs(A_sparse)
        DW_RHS  = make_dw_rhs(A_sparse)
        MUT_RHS = make_mut_rhs(es, ed)
        GEN_RHS = make_gen_rhs(A_sparse)
        WC_RHS  = make_wc_rhs(A_sparse)
        SIR_RHS = make_sir_rhs(A_sparse)
        degrees = np.asarray(A_sparse.sum(axis=1)).ravel()
        rng_r = np.random.default_rng(7)
        r_vec = 0.6 + 0.4 * rng_r.random(N)
        NM_RHS  = make_noymeir_rhs(A_sparse, degrees, r_vec)
        rng_b = np.random.default_rng(11)
        b_vec = 0.8 + 0.4 * rng_b.random(N)
        EUT_RHS = make_eutroph_rhs(A_sparse, degrees, b_vec)
        rng_h = np.random.default_rng(13)
        hE_vec = 1.0 + 2.0 * rng_h.random(N)
        VEG_RHS = make_vegturb_rhs(A_sparse, degrees, hE_vec)
        MSIS_RHS = make_metapop_sis_rhs(A_sparse, degrees)
        MSIR_RHS = make_metapop_sir_rhs(A_sparse, degrees)

        # Per-system sweep with auto-trim (flat region <= 20% of swept range).
        def do_sweep(sys_tag, grid, sweep_kind):
            for _ in range(2):     # at most one auto-trim re-sweep
                if sweep_kind == "ode":
                    rhs_fn, x0_fn = SWEEP_FNS[sys_tag]
                    Y = sweep_ode(rhs_fn, grid, x0_fn(N), sys_tag, N)
                elif sweep_kind == "sir":
                    Y = sweep_sir(SIR_RHS, grid, N)
                elif sweep_kind == "msis":
                    Y = sweep_metapop_sis(MSIS_RHS, grid, N)
                elif sweep_kind == "msir":
                    Y = sweep_metapop_sir(MSIR_RHS, grid, N)
                elif sweep_kind == "glv":
                    Y = glv_sweep(A_dense, alpha_max, grid, lam_bar=1.0)
                trimmed = trim_grid(grid, Y)
                if trimmed is None:
                    return grid, Y
                print(f"    {sys_tag}: flat>20%, retrimming to "
                      f"[{trimmed[0]:.3f}, {trimmed[-1]:.3f}]")
                grid = trimmed
            return grid, Y

        SWEEP_FNS = {
            "SIS": (SIS_RHS, lambda N: np.full(N, 0.9)),
            "DW":  (DW_RHS,  lambda N: np.ones(N)),
            "MUT": (MUT_RHS, lambda N: np.full(N, 0.001)),
            "GEN": (GEN_RHS, lambda N: np.full(N, 2.0)),
            "WC":  (WC_RHS,  lambda N: np.full(N, 0.1)),
            "NM":  (NM_RHS,  lambda N: np.full(N, 8.0)),
            "EUT": (EUT_RHS, lambda N: np.full(N, 0.5)),
            "VEG": (VEG_RHS, lambda N: np.full(N, 0.6)),
        }
        for sys_tag in missing:
            if sys_tag == "gLV":
                ps, eq[sys_tag] = do_sweep(sys_tag, glv_grid, "glv")
                grids[sys_tag] = ps
            elif sys_tag == "SIR":
                ps, eq[sys_tag] = do_sweep(sys_tag, sir_grid, "sir")
                grids[sys_tag] = ps
            elif sys_tag == "MSIS":
                ps, eq[sys_tag] = do_sweep(sys_tag, msis_grid, "msis")
                grids[sys_tag] = ps
            elif sys_tag == "MSIR":
                ps, eq[sys_tag] = do_sweep(sys_tag, msir_grid, "msir")
                grids[sys_tag] = ps
            else:
                ps, eq[sys_tag] = do_sweep(sys_tag, grids[sys_tag], "ode")
                grids[sys_tag] = ps
            arr = np.column_stack([ps, eq[sys_tag]])
            header = ["p"] + [f"x_{i+1}" for i in range(N)]
            np.savetxt(csv_paths[sys_tag], arr, delimiter=",",
                       header=",".join(header), comments="",
                       fmt=["%.6f"] + ["%.6f"] * N)

        # Diagnostic 12-panel bifurcation overview (KMULT==1 path)
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        for ax, sys_tag in zip(axes.ravel(), SYSTEMS):
            Y = eq[sys_tag]; ps = grids[sys_tag]
            for i in range(N):
                ax.plot(ps, Y[:, i], color="steelblue", alpha=0.18, linewidth=0.4)
            ax.plot(ps, Y.mean(axis=1), color="black", linewidth=1.8)
            ax.set_title(f"{sys_tag}  (range {Y.min():.2f},{Y.max():.2f})",
                         fontsize=10)
            ax.set_xlabel("control param."); ax.set_ylabel(r"$x_i^*$")
        fig.suptitle(f"{tag.upper()}: 12 dynamics on N={N}, "
                     f"alpha_max={alpha_max:.2f}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(HERE / f"{tag}_bifurcations_overview.png", dpi=130)
        plt.close(fig)

    # --- Train per-system decoders (qr, random, minc) ---
    ckpts_qr = {}; ckpts_rd = {}; ckpts_mc = {}
    print("  per-system decoder training:")
    for sys_tag in SYSTEMS:
        Y = eq[sys_tag]; M = Y.shape[0]
        rng = np.random.default_rng(SEED); idx = np.arange(M); rng.shuffle(idx)
        n_test = int(round(TEST_FRACTION * M))
        Y_train_for_pick = Y[np.sort(idx[n_test:])]
        qr_idx = qr_pivot_sensors(Y_train_for_pick, K)
        mc_idx = min_corr_sensors(Y_train_for_pick, K)
        ck_qr = train_one(Y, qr_idx)
        ck_rd = train_one(Y, RANDOM_SENTINELS.copy())
        ck_mc = train_one(Y, mc_idx)
        torch.save(ck_qr, HERE / f"{prefix}_{sys_tag.lower()}_qr_decoder_checkpoint.pt")
        torch.save(ck_rd, HERE / f"{prefix}_{sys_tag.lower()}_random_decoder_checkpoint.pt")
        torch.save(ck_mc, HERE / f"{prefix}_{sys_tag.lower()}_minc_decoder_checkpoint.pt")
        ckpts_qr[sys_tag] = ck_qr; ckpts_rd[sys_tag] = ck_rd; ckpts_mc[sys_tag] = ck_mc
        print(f"    {sys_tag}: QR={(qr_idx+1).tolist()}, MC={(mc_idx+1).tolist()}, "
              f"mse_qr={ck_qr['test_mse']:.3e}, mse_rd={ck_rd['test_mse']:.3e}, "
              f"mse_mc={ck_mc['test_mse']:.3e}")

    # --- COMBINED ---
    print("  COMBINED decoder training:")
    rng = np.random.default_rng(SEED)
    train_blocks, test_blocks, test_labels = [], [], []
    for sys_tag in SYSTEMS:
        Y = eq[sys_tag]; M = Y.shape[0]
        idx = np.arange(M); rng.shuffle(idx)
        n_test = int(round(TEST_FRACTION * M))
        train_blocks.append(Y[np.sort(idx[n_test:])])
        test_blocks.append(Y[np.sort(idx[:n_test])])
        test_labels.extend([sys_tag] * n_test)
    Y_train_c = np.vstack(train_blocks); Y_test_c = np.vstack(test_blocks)
    test_labels = np.array(test_labels)
    comb_qr_idx = qr_pivot_sensors(Y_train_c, K)
    comb_mc_idx = min_corr_sensors(Y_train_c, K)
    ck_comb_qr = train_combined(Y_train_c, Y_test_c, test_labels,
                                comb_qr_idx, sys_tags=tuple(SYSTEMS))
    ck_comb_rd = train_combined(Y_train_c, Y_test_c, test_labels,
                                RANDOM_SENTINELS.copy(), sys_tags=tuple(SYSTEMS))
    ck_comb_mc = train_combined(Y_train_c, Y_test_c, test_labels,
                                comb_mc_idx, sys_tags=tuple(SYSTEMS))
    torch.save(ck_comb_qr, HERE / f"{prefix}_combined_qr_decoder_checkpoint.pt")
    torch.save(ck_comb_rd, HERE / f"{prefix}_combined_random_decoder_checkpoint.pt")
    torch.save(ck_comb_mc, HERE / f"{prefix}_combined_minc_decoder_checkpoint.pt")
    ckpts_qr["COMBINED"] = ck_comb_qr; ckpts_rd["COMBINED"] = ck_comb_rd
    ckpts_mc["COMBINED"] = ck_comb_mc
    print(f"    QR={(comb_qr_idx+1).tolist()} mse={ck_comb_qr['test_mse']:.3e}")
    print(f"    rd={(RANDOM_SENTINELS+1).tolist()} mse={ck_comb_rd['test_mse']:.3e}")
    print(f"    MC={(comb_mc_idx+1).tolist()} mse={ck_comb_mc['test_mse']:.3e}")

    # --- Cross matrix (7x6) ---
    SOURCES = SYSTEMS + ["COMBINED"]
    TARGETS = SYSTEMS

    def run_matrix(ckpts, variant: str):
        rows = ["src -> tgt    MSE         RMSE      RMSE/range   relRMSE_v12   a       b   "]
        mse_grid = np.full((len(SOURCES), len(TARGETS)), np.nan)
        rel_grid = np.full((len(SOURCES), len(TARGETS)), np.nan)
        rel_v12_grid = np.full((len(SOURCES), len(TARGETS)), np.nan)
        rel_v13_grid = np.full((len(SOURCES), len(TARGETS)), np.nan)
        rel_v14_grid = np.full((len(SOURCES), len(TARGETS)), np.nan)
        a_grid   = np.full((len(SOURCES), len(TARGETS)), np.nan)
        b_grid   = np.full((len(SOURCES), len(TARGETS)), np.nan)
        for i, src in enumerate(SOURCES):
            sensor_idx = ckpts[src]["sensor_idx"]
            for j, tgt in enumerate(TARGETS):
                Y_true = eq[tgt]
                Y_pred, a, b = fair_predict(ckpts[src], Y_true)
                mse = float(np.mean((Y_pred - Y_true) ** 2))
                rmse = float(np.sqrt(mse))
                dyn = float(Y_true.max() - Y_true.min())
                rel = rmse / dyn if dyn > 0 else float("nan")
                # v12 relRMSE: drop sweep rows with max-min < 1e-3, then
                # divide by the RMSE of the constant-prediction baseline
                # (per-row mean of the source decoder's sentinel readings).
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
                # v14: per-column ratio averaged across columns whose
                # max |Y| > 1e-2 (new validity criterion).
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
        out_txt = HERE / f"{prefix}_cross_matrix_summary_{variant}.txt"
        out_txt.write_text("\n".join(rows) + "\n")
        np.savez(HERE / f"{prefix}_cross_matrix_data_{variant}.npz",
                 mse=mse_grid, rel=rel_grid,
                 rel_v12=rel_v12_grid, rel_v13=rel_v13_grid,
                 rel_v14=rel_v14_grid,
                 a=a_grid, b=b_grid,
                 sources=np.array(SOURCES, dtype="U16"),
                 targets=np.array(TARGETS, dtype="U16"),
                 N=N, K=K, alpha_max=alpha_max)
        print(f"  --- {variant} matrix ---")
        for r in rows: print(r)

    run_matrix(ckpts_qr, "qr")
    run_matrix(ckpts_rd, "random")
    run_matrix(ckpts_mc, "minc")

    meta = HERE / f"{prefix}_meta.txt"
    meta.write_text(
        f"network_tag: {tag}\nkmult: {kmult}\nN: {N}\nK: {K} "
        f"({kmult} * floor(ln N))\nalpha_max: {alpha_max:.4f}\n"
        f"random_sentinels_0based: {RANDOM_SENTINELS.tolist()}\n"
        f"random_sentinels_1based: {(RANDOM_SENTINELS+1).tolist()}\n"
        f"D_ranges:\n"
        f"  SIS: [0,1]\n  DW: [0,1]\n  MUT: [0,3]\n  GEN: [0,1]\n"
        f"  WC: [0,{wc_dmax:.3f}]  (D_max = max(3, min(50/alpha_max, 8)))\n"
        f"  gLV: [{0.05*Dc_glv:.3f},{0.95*Dc_glv:.3f}]  "
        f"(D_c=(alpha_max+1)/alpha_max={Dc_glv:.3f})\n"
    )
    print(f"\n  wrote {prefix}_cross_matrix_summary_qr.txt, "
          f"{prefix}_cross_matrix_summary_random.txt, {prefix}_meta.txt\n")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3 or sys.argv[1] not in NETWORK_FILES:
        sys.exit(f"usage: python3 pipeline_unified.py "
                 f"<{'|'.join(NETWORK_FILES)}> [KMULT]")
    kmult = float(sys.argv[2]) if len(sys.argv) == 3 else 1.0
    main(sys.argv[1], kmult)
