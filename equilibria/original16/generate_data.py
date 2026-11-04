"""Equilibrium-data generation for one network and 12 dynamical systems.

Standalone (no neural-network or decoder code). For each (network, dynamics)
pair this script:
  1. Sweeps the bifurcation parameter at 51 points.
  2. Auto-trims the parameter range so the "flat" region (mean(x_i*) < 0.05)
     is at most 20% of the swept range.
  3. Saves the equilibrium matrix as a CSV.
  4. Records the final [D_min, D_max] in <tag>_bifurcation_ranges.csv.

Usage:  python3 generate_data.py <NETWORK_TAG>
   where NETWORK_TAG is one of the tags in verify_networks.py NETWORKS list.

Dynamics:
  SIS  -- susceptible-infected-susceptible (mean field)
  DW   -- coupled double-well (Gao et al.)
  MUT  -- mutualistic species (Gao et al.)
  GEN  -- gene-regulatory (Gao et al.)
  WC   -- Cowan-Wilson on networks (Laurence 2019)
  gLV  -- generalized Lotka-Volterra (closed-form equilibrium)
  SIR  -- susceptible-infected-recovered (mean field per node)
  NM   -- Noy-Meir / May 1977 grazing (Dakos 2010 model 1)
  EUT  -- eutrophication (Carpenter 1999 / Dakos 2010 model 2)
  VEG  -- vegetation-turbidity (Scheffer 1998 / Dakos 2010 model 3)
  MSIS -- metapopulation SIS (metapop.tex Eqs 1-2)
  MSIR -- metapopulation SIR (assumed: D_R = 0)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigsh

HERE = Path(__file__).resolve().parent

# Edgelist files produced by verify_networks.py.
NETWORK_FILES = {
    "dolphin":      "dolphin_network.edgelist",
    "prox":         "prox_network.edgelist",
    "metabolic":    "metabolic_network.edgelist",
    "euroroad":     "euroroad_network.edgelist",
    "email":        "email_network.edgelist",
    "er":           "er_network.edgelist",
    "ba":           "ba_network.edgelist",
    "hk":           "hk_network.edgelist",
    "gkk":          "gkk_network.edgelist",
    "lfr":          "lfr_network.edgelist",
    "flybi":        "flybi_network.edgelist",
    "reactome":     "reactome_network.edgelist",
    "route_views":  "route_views_network.edgelist",
    "spanish":      "spanish_network.edgelist",
    "foldoc":       "foldoc_network.edgelist",
    "tree_of_life": "tree_of_life_network.edgelist",
    "english":      "english_network.edgelist",
    "enron":        "enron_network.edgelist",
    "marker_cafe":  "marker_cafe_network.edgelist",
    "prosper":      "prosper_network.edgelist",
}

CSV_NAMES = {"SIS": "sis", "DW": "doublewell", "MUT": "mutualistic",
             "GEN": "genereg", "WC": "wc", "gLV": "glv",
             "SIR": "sir", "NM": "noymeir",
             "EUT": "eutroph", "VEG": "vegturb", "MSIS": "msis",
             "MSIR": "msir"}
SYSTEMS = list(CSV_NAMES.keys())


def load_network(tag):
    G = nx.read_edgelist(HERE / NETWORK_FILES[tag], nodetype=int)
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")
    N = A.shape[0]
    if N <= 200:
        alpha_max = float(np.linalg.eigvalsh(A.toarray()).max())
    else:
        alpha_max = float(eigsh(A, k=1, which="LA",
                                return_eigenvectors=False)[0])
    edges_uv = [(min(u, v), max(u, v)) for u, v in G.edges()]
    es = np.array([u for u, _ in edges_uv]); ed = np.array([v for _, v in edges_uv])
    return N, alpha_max, A, es, ed


# ------------------ Dynamics RHS factories ------------------

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
    GAMMA = 1.0
    def rhs(t, state, beta):
        N = state.size // 2
        S = state[:N]; I = state[N:]
        AI = A @ I
        dS = -beta * S * AI
        dI =  beta * S * AI - GAMMA * I
        return np.concatenate([dS, dI])
    return rhs


def sir_jac_sparsity(A_sparse):
    """2N x 2N Jacobian sparsity for network SIR. Conservative super-set:
    block structure [[I, A+I], [I, A+I]] (J_SS is diagonal, J_SI=A,
    J_IS is diagonal, J_II=A+I; we use A+I in the right column for safety)."""
    from scipy.sparse import (csr_matrix, vstack, hstack, eye as sp_eye)
    N = A_sparse.shape[0]
    A_plus_I = ((A_sparse != 0).astype(np.int8) + sp_eye(N, dtype=np.int8) > 0).astype(np.int8)
    Id = sp_eye(N, format="csr", dtype=np.int8)
    return vstack([
        hstack([Id, A_plus_I]),
        hstack([Id, A_plus_I]),
    ]).tocsr()

def make_noymeir_rhs(A, degrees, r_vec):
    K_, D_DISP = 10.0, 0.05
    def rhs(t, x, c):
        return (r_vec * x * (1.0 - x / K_)
                - c * x * x / (x * x + 1.0)
                + D_DISP * (A @ x - degrees * x))
    return rhs

def make_eutroph_rhs(A, degrees, b_vec):
    R_RECYC, P_HILL, D_DISP = 1.0, 8.0, 0.05
    def rhs(t, x, alpha):
        x_pos = np.maximum(x, 0.0)
        x_p = x_pos ** P_HILL
        return (alpha - b_vec * x
                + R_RECYC * x_p / (x_p + 1.0)
                + D_DISP * (A @ x - degrees * x))
    return rhs

def make_vegturb_rhs(A, degrees, hE_vec):
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
    MU, D_S, D_I = 1.0, 1.0, 1.0
    def rhs(t, state, beta):
        N = state.size // 2
        rS = state[:N]; rI = state[N:]
        diff_S = D_S * (A @ (rS / k_vec) - rS)
        diff_I = D_I * (A @ (rI / k_vec) - rI)
        infect = beta * rS * rI
        recov  = MU * rI
        return np.concatenate([-infect + recov + diff_S,
                                infect - recov + diff_I])
    return rhs


def metapop_sis_jac_sparsity(A_sparse):
    """2N x 2N Jacobian sparsity for metapop SIS, block structure [[A+I,I],[I,A+I]]."""
    from scipy.sparse import (csr_matrix, vstack, hstack, eye as sp_eye)
    N = A_sparse.shape[0]
    A_plus_I = ((A_sparse != 0).astype(np.int8) + sp_eye(N, dtype=np.int8) > 0).astype(np.int8)
    Id = sp_eye(N, format="csr", dtype=np.int8)
    return vstack([
        hstack([A_plus_I, Id]),
        hstack([Id, A_plus_I]),
    ]).tocsr()

def make_metapop_sir_rhs(A, k_vec):
    MU, D_S, D_I = 1.0, 1.0, 1.0
    def rhs(t, state, beta):
        N = state.size // 3
        rS = state[:N]; rI = state[N:2*N]
        diff_S = D_S * (A @ (rS / k_vec) - rS)
        diff_I = D_I * (A @ (rI / k_vec) - rI)
        infect = beta * rS * rI
        recov  = MU * rI
        return np.concatenate([-infect + diff_S,
                                infect - recov + diff_I,
                                recov])
    return rhs


def metapop_sir_jac_sparsity(A_sparse):
    """3N x 3N Jacobian sparsity for metapop SIR. Non-zero structure:
        [[ A+I , I    , 0  ],
         [ I   , A+I  , 0  ],
         [ 0   , I    , 0  ]]
    where A is the network adjacency (sparsity preserved under D_S * A diag-normalized
    diffusion) and I is the identity."""
    from scipy.sparse import (csr_matrix, vstack, hstack, eye as sp_eye)
    N = A_sparse.shape[0]
    A_plus_I = ((A_sparse != 0).astype(np.int8) + sp_eye(N, dtype=np.int8) > 0).astype(np.int8)
    Id = sp_eye(N, format="csr", dtype=np.int8)
    Z = csr_matrix((N, N), dtype=np.int8)
    return vstack([
        hstack([A_plus_I, Id, Z]),
        hstack([Id, A_plus_I, Z]),
        hstack([Z, Id, Z]),
    ]).tocsr()


# ------------------ Sweep helpers ------------------

def equilibrate(rhs, p, x0, t_final=200.0, rtol=1e-4, atol=1e-6,
                jac_sparsity=None):
    method = "BDF" if jac_sparsity is not None else "LSODA"
    extra = {"jac_sparsity": jac_sparsity} if jac_sparsity is not None else {}
    sol = solve_ivp(rhs, (0.0, t_final), x0, method=method,
                    args=(p,), rtol=rtol, atol=atol, **extra)
    if not sol.success:
        raise RuntimeError(f"ODE failed at p={p}: {sol.message}")
    return sol.y[:, -1]

def sweep_ode(rhs, ps, x0_func, label, N, jac_sparsity=None):
    t0 = time.time()
    eq = np.empty((len(ps), N))
    method = "BDF" if jac_sparsity is not None else "LSODA"
    for k, p in enumerate(ps):
        x0 = x0_func(p) if callable(x0_func) else x0_func
        eq[k] = equilibrate(rhs, p, x0.copy(), jac_sparsity=jac_sparsity)
    print(f"    {label} ({method}): {len(ps)} points in {time.time()-t0:.1f}s, "
          f"y-range [{eq.min():.3f}, {eq.max():.3f}]")
    return eq


def per_node_jac_sparsity(A_sparse):
    """N x N sparsity = A + I.  Used for the stiff single-variable dynamics
    (SIS, DW, MUT, GEN, WC, NM, EUT, VEG) on hub-heavy networks."""
    from scipy.sparse import eye as sp_eye
    N = A_sparse.shape[0]
    return ((A_sparse != 0).astype(np.int8) + sp_eye(N, dtype=np.int8) > 0).astype(np.int8).tocsr()

def sweep_sir(rhs, betas, N, t_final=400.0, i0=0.01, jac_sparsity=None):
    t0 = time.time()
    eq = np.empty((len(betas), N))
    method = "BDF" if jac_sparsity is not None else "LSODA"
    extra = {"jac_sparsity": jac_sparsity} if jac_sparsity is not None else {}
    for k, beta in enumerate(betas):
        S0 = np.full(N, 1.0 - i0); I0 = np.full(N, i0)
        sol = solve_ivp(rhs, (0.0, t_final), np.concatenate([S0, I0]),
                        method=method, args=(beta,),
                        rtol=1e-4, atol=1e-6, **extra)
        if not sol.success:
            raise RuntimeError(f"SIR ODE failed at beta={beta}: {sol.message}")
        eq[k] = 1.0 - sol.y[:N, -1]
    print(f"    SIR ({method}): {len(betas)} points in {time.time()-t0:.1f}s, "
          f"y-range [{eq.min():.3f}, {eq.max():.3f}]")
    return eq

def sweep_metapop_sis(rhs, betas, N, t_final=100.0, i0=0.01,
                      jac_sparsity=None):
    """When jac_sparsity is provided, switch to BDF with sparse-Jacobian
    structure (avoids LSODA's expensive numerical Jacobian estimation on the
    2N-state ODE for hub-heavy networks)."""
    t0 = time.time()
    eq = np.empty((len(betas), N))
    method = "BDF" if jac_sparsity is not None else "LSODA"
    extra = {"jac_sparsity": jac_sparsity} if jac_sparsity is not None else {}
    for k, beta in enumerate(betas):
        S0 = np.full(N, 1.0 - i0); I0 = np.full(N, i0)
        sol = solve_ivp(rhs, (0.0, t_final), np.concatenate([S0, I0]),
                        method=method, args=(beta,),
                        rtol=1e-4, atol=1e-6, **extra)
        if not sol.success:
            raise RuntimeError(f"MSIS ODE failed at beta={beta}: {sol.message}")
        eq[k] = sol.y[N:2*N, -1]
    print(f"    MSIS ({method}): {len(betas)} points in {time.time()-t0:.1f}s, "
          f"y-range [{eq.min():.3f}, {eq.max():.3f}]")
    return eq

def sweep_metapop_sir(rhs, betas, N, t_final=100.0, i0=0.01,
                      jac_sparsity=None):
    """When jac_sparsity is provided we use BDF (a stiff solver) and tell it where
    the Jacobian non-zeros live; this avoids LSODA's O(state-size) numerical
    Jacobian estimation, which is the bottleneck on hub-heavy networks."""
    t0 = time.time()
    eq = np.empty((len(betas), N))
    method = "BDF" if jac_sparsity is not None else "LSODA"
    extra = {"jac_sparsity": jac_sparsity} if jac_sparsity is not None else {}
    for k, beta in enumerate(betas):
        S0 = np.full(N, 1.0 - i0); I0 = np.full(N, i0); R0 = np.zeros(N)
        sol = solve_ivp(rhs, (0.0, t_final), np.concatenate([S0, I0, R0]),
                        method=method, args=(beta,),
                        rtol=1e-4, atol=1e-6, **extra)
        if not sol.success:
            raise RuntimeError(f"MSIR ODE failed at beta={beta}: {sol.message}")
        eq[k] = sol.y[2*N:3*N, -1]
    print(f"    MSIR ({method}): {len(betas)} points in {time.time()-t0:.1f}s, "
          f"y-range [{eq.min():.3f}, {eq.max():.3f}]")
    return eq

def glv_sweep(A_dense, alpha_max, D_grid, lam_bar=1.0):
    N = A_dense.shape[0]
    c = alpha_max + 1.0
    eq = np.empty((len(D_grid), N))
    eye = np.eye(N)
    for k, D in enumerate(D_grid):
        eq[k] = -lam_bar * np.linalg.solve(D * A_dense - c * eye, np.ones(N))
    print(f"    gLV: {len(D_grid)} points (closed-form, dense), "
          f"y-range [{eq.min():.3f}, {eq.max():.3f}]")
    return eq


def glv_sweep_sparse(A_sparse, alpha_max, D_grid, lam_bar=1.0):
    """Sparse spsolve version for large N where the dense (N x N) form
    would not fit in memory or be too slow."""
    from scipy.sparse import eye as sp_eye
    from scipy.sparse.linalg import spsolve
    N = A_sparse.shape[0]
    c = alpha_max + 1.0
    eq = np.empty((len(D_grid), N))
    one = np.ones(N)
    A_csc = A_sparse.tocsc()
    eye_csc = sp_eye(N, format="csc")
    t0 = time.time()
    for k, D in enumerate(D_grid):
        M = (D * A_csc - c * eye_csc).tocsc()
        eq[k] = -lam_bar * spsolve(M, one)
    print(f"    gLV: {len(D_grid)} points (closed-form, sparse) "
          f"in {time.time()-t0:.1f}s, "
          f"y-range [{eq.min():.3f}, {eq.max():.3f}]")
    return eq


# ------------------ Auto-trim ------------------

def trim_grid(grid, Y, threshold=0.05, max_flat_frac=0.20):
    """Return a new 51-point grid that bounds the flat region (mean<threshold)
    to at most max_flat_frac of the swept range, or None if no trim needed."""
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


# ------------------ Main ------------------

def main(tag):
    print(f"\n======= generate_data: {tag.upper()} =======")
    N, alpha_max, A_sparse, es, ed = load_network(tag)
    A_dense = A_sparse.toarray() if N <= 5000 else None
    K_log = int(np.floor(np.log(N)))
    print(f"  N={N}, alpha_max={alpha_max:.3f}, K=floor(ln N)={K_log}")

    # Initial grids; auto-trim adjusts.
    NSWEEP = 51
    init_grids = {
        "SIS":  np.linspace(0.0, 1.0, NSWEEP),
        "DW":   np.linspace(0.0, 1.0, NSWEEP),
        "MUT":  np.linspace(0.0, 3.0, NSWEEP),
        "GEN":  np.linspace(0.0, 1.0, NSWEEP),
        "WC":   np.linspace(0.0, float(max(3.0, min(50.0/alpha_max, 8.0))), NSWEEP),
        "gLV":  np.linspace(0.05*(alpha_max+1)/alpha_max,
                            0.95*(alpha_max+1)/alpha_max, NSWEEP),
        "SIR":  np.linspace(0.02, 1.0, NSWEEP),
        "NM":   np.linspace(1.0, 3.0, NSWEEP),
        "EUT":  np.linspace(0.1, 0.9, NSWEEP),
        "VEG":  np.linspace(2.0, 12.0, NSWEEP),
        "MSIS": np.linspace(0.05, 1.5, NSWEEP),
        "MSIR": np.linspace(0.05, 1.5, NSWEEP),
    }

    # Build RHS functions.
    SIS_RHS  = make_sis_rhs(A_sparse)
    DW_RHS   = make_dw_rhs(A_sparse)
    MUT_RHS  = make_mut_rhs(es, ed)
    GEN_RHS  = make_gen_rhs(A_sparse)
    WC_RHS   = make_wc_rhs(A_sparse)
    SIR_RHS  = make_sir_rhs(A_sparse)
    degrees  = np.asarray(A_sparse.sum(axis=1)).ravel()
    rng_r    = np.random.default_rng(7)
    r_vec    = 0.6 + 0.4 * rng_r.random(N)
    NM_RHS   = make_noymeir_rhs(A_sparse, degrees, r_vec)
    rng_b    = np.random.default_rng(11)
    b_vec    = 0.8 + 0.4 * rng_b.random(N)
    EUT_RHS  = make_eutroph_rhs(A_sparse, degrees, b_vec)
    rng_h    = np.random.default_rng(13)
    hE_vec   = 1.0 + 2.0 * rng_h.random(N)
    VEG_RHS  = make_vegturb_rhs(A_sparse, degrees, hE_vec)
    MSIS_RHS = make_metapop_sis_rhs(A_sparse, degrees)
    MSIR_RHS = make_metapop_sir_rhs(A_sparse, degrees)

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

    # BDF + sparse-Jacobian only for the per-node dynamics that empirically
    # stall under LSODA at hub-heavy networks (DW, MUT, VEG).  The others
    # (SIS, GEN, WC, NM, EUT) stay on LSODA, which uses dense numerical
    # Jacobian and benefits from multi-core BLAS via OMP_NUM_THREADS=4.
    BDF_DYNAMICS = {"DW", "MUT", "VEG", "EUT"}
    per_node_pat = per_node_jac_sparsity(A_sparse) if N >= 2000 else None

    def run_sweep(sys_tag, grid):
        kind = ("ode" if sys_tag in SWEEP_FNS
                else {"gLV": "glv", "SIR": "sir",
                      "MSIS": "msis", "MSIR": "msir"}[sys_tag])
        if kind == "ode":
            rhs_fn, x0_fn = SWEEP_FNS[sys_tag]
            jac_pat = per_node_pat if sys_tag in BDF_DYNAMICS else None
            return sweep_ode(rhs_fn, grid, x0_fn(N), sys_tag, N,
                             jac_sparsity=jac_pat)
        if kind == "sir":
            jac_pat = sir_jac_sparsity(A_sparse) if N >= 2000 else None
            return sweep_sir(SIR_RHS, grid, N, jac_sparsity=jac_pat)
        if kind == "msis":
            jac_pat = metapop_sis_jac_sparsity(A_sparse) if N >= 2000 else None
            return sweep_metapop_sis(MSIS_RHS, grid, N, jac_sparsity=jac_pat)
        if kind == "msir":
            jac_pat = metapop_sir_jac_sparsity(A_sparse) if N >= 2000 else None
            return sweep_metapop_sir(MSIR_RHS, grid, N, jac_sparsity=jac_pat)
        if kind == "glv":
            if A_dense is not None:
                return glv_sweep(A_dense, alpha_max, grid, lam_bar=1.0)
            return glv_sweep_sparse(A_sparse, alpha_max, grid, lam_bar=1.0)

    # Per-system sweep with auto-trim. Reuse existing CSVs if present.
    csv_paths = {s: HERE / f"{tag}_{CSV_NAMES[s]}_equilibrium_data.csv"
                 for s in SYSTEMS}
    eq = {}; final_grids = {}
    for sys_tag in SYSTEMS:
        if csv_paths[sys_tag].exists():
            arr = np.loadtxt(csv_paths[sys_tag], delimiter=",", skiprows=1)
            final_grids[sys_tag] = arr[:, 0]
            eq[sys_tag] = arr[:, 1:]
            print(f"    {sys_tag}: reusing existing CSV "
                  f"(range [{arr[0,0]:.4f}, {arr[-1,0]:.4f}])")
            continue
        grid = init_grids[sys_tag]
        Y = run_sweep(sys_tag, grid)
        for _ in range(2):
            trimmed = trim_grid(grid, Y)
            if trimmed is None:
                break
            print(f"    {sys_tag}: flat>20%, retrimming to "
                  f"[{trimmed[0]:.4f}, {trimmed[-1]:.4f}]")
            grid = trimmed
            Y = run_sweep(sys_tag, grid)
        final_grids[sys_tag] = grid; eq[sys_tag] = Y
        # Save CSV
        arr = np.column_stack([grid, Y])
        header = ["p"] + [f"x_{i+1}" for i in range(N)]
        np.savetxt(csv_paths[sys_tag], arr, delimiter=",",
                   header=",".join(header), comments="",
                   fmt=["%.6f"] + ["%.6f"] * N)

    # Save bifurcation ranges as a CSV.
    rng_path = HERE / f"{tag}_bifurcation_ranges.csv"
    with open(rng_path, "w") as f:
        f.write("dynamics,p_min,p_max,n_sweep,y_min,y_max\n")
        for s in SYSTEMS:
            grid = final_grids[s]; Y = eq[s]
            f.write(f"{s},{grid[0]:.6f},{grid[-1]:.6f},{len(grid)},"
                    f"{Y.min():.6f},{Y.max():.6f}\n")
    print(f"  wrote {rng_path.name}")

    # Diagnostic 12-panel bifurcation overview.
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, sys_tag in zip(axes.ravel(), SYSTEMS):
        Y = eq[sys_tag]; ps = final_grids[sys_tag]
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
    print(f"  wrote {tag}_bifurcations_overview.png\n")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in NETWORK_FILES:
        sys.exit(f"usage: python3 generate_data.py "
                 f"<{'|'.join(NETWORK_FILES)}>")
    main(sys.argv[1])
