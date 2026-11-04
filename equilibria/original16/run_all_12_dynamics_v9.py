"""Run all 12 dynamics for one larger network under the v9 protocols.

Usage:  python3 run_all_12_dynamics_v9.py <tag>

Larger-network protocol summary (per decoder_v9.tex):
  SIS  : reverse warm-start, beta_max = 1/(0.15*alpha_max)
         (no widening; SIS-anchored beta_max is well-tuned)
         IC at beta_max: x_i(0) = 0.9
  SIR  : cold start, beta_max = 1/(0.15*alpha_max)
         IC at every beta: S_i = 0.999, I_i = 0.001
  MSIS : reverse warm-start + adaptive beta_max widening
         IC at beta_max: rho_S = 0.9, rho_I = 0.1
  MSIR : cold start + adaptive beta_max widening
         IC at every beta: rho_S = 0.999, rho_I = 0.001, rho_R = 0
  DW   : cold start, D_max * <k> = 2.564
         IC at every D: x_i = 1
  MUT  : forward warm-start, D_max * <k> = 10.258
         IC at D=0: x_i = 0.001
  GEN  : reverse warm-start (because larger networks are heterogeneous)
         D_max * alpha_max = 11.510
         IC at D_max: x_i = 10
  WC   : forward warm-start, D_max * alpha_max = 43.166
         IC at D=0: x_i = 0.1
  gLV  : closed form, D in [0.05*D_c, 0.95*D_c], D_c = (alpha_max+1)/alpha_max
  NM   : forward warm-start, c in [1, 3]
         IC at c=1: x_i = 8
  EUT  : forward warm-start, alpha in [0.1, 0.9]
         IC at alpha=0.1: x_i = 0.5
  VEG  : forward warm-start, E_o in [2, 12]
         IC at E_o=2: x_i = 0.6

Solver tier: rtol=1e-4, atol=1e-6 (loose, for the larger networks).
BDF + sparse Jacobian for the stiff dynamics:
  DW, MUT, VEG, EUT, SIR, MSIS, MSIR, GEN.
LSODA for the rest (SIS, WC, NM).

Outputs (saved into the same directory as this script):
  <tag>_<dyn>_equilibrium_data.csv  for each of the 12 dynamics
  <tag>_bifurcation_ranges.csv      summary of (p_min, p_max, y_min, y_max)
  <tag>_v9_progress.log             one-line-per-dynamics progress log
                                    (for monitoring progress)

The progress log writes a "DONE: <dyn>" line at the end of each dynamics
and a final "ALL 12 DYNAMICS DONE for <tag>" marker.
"""
from __future__ import annotations
import sys
import time
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, spsolve
import networkx as nx
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
NSWEEP = 51
RTOL, ATOL = 1e-4, 1e-6
THRESHOLD_META = 0.02


def _log(progress_path: Path, msg: str):
    with progress_path.open("a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg, flush=True)


# -----------------------------------------------------------------
# Network loading and stiff Jacobian patterns
# -----------------------------------------------------------------
def load_network(tag: str):
    G = nx.read_edgelist(HERE / f"{tag}_network.edgelist", nodetype=int)
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")
    N = A.shape[0]
    M = A.nnz // 2
    k_vec = np.asarray(A.sum(axis=1)).ravel()
    kbar = float(k_vec.mean())
    if N < 200:
        alpha_max = float(np.linalg.eigvalsh(A.toarray()).max())
    else:
        alpha_max = float(eigsh(A, k=1, which="LA",
                                return_eigenvectors=False)[0])
    A_coo = A.tocoo()
    es = []; ed = []
    for i, j in zip(A_coo.row, A_coo.col):
        if i < j:
            es.append(int(i)); ed.append(int(j))
    return dict(N=N, M=M, A=A, k_vec=k_vec, kbar=kbar, alpha_max=alpha_max,
                es=np.array(es), ed=np.array(ed))


def per_node_jac(A):
    return A + sp.eye(A.shape[0])


def two_block_jac(A):
    N = A.shape[0]
    AI = A + sp.eye(N)
    return sp.bmat([[AI, AI], [AI, AI]], format="csr")


def msir_jac(A):
    N = A.shape[0]
    AI = A + sp.eye(N)
    Z = sp.csr_matrix((N, N))
    return sp.bmat([[AI, AI, Z],
                    [AI, AI, Z],
                    [Z, sp.eye(N), Z]], format="csr")


# -----------------------------------------------------------------
# RHS factories (v9)
# -----------------------------------------------------------------
def make_sis(A):
    def rhs(t, x, beta):
        return -x + beta * (1.0 - x) * (A @ x)
    return rhs

def make_dw(A):
    def rhs(t, x, D):
        return -(x - 1.0) * (x - 3.0) * (x - 5.0) + D * (A @ x)
    return rhs

def make_mut(es, ed):
    B, K_, C = 0.1, 5.0, 1.0; DT, E_, H_ = 5.0, 0.9, 0.1
    def rhs(t, x, D):
        self_term = B + x * (1.0 - x / K_) * (x / C - 1.0)
        xs = x[es]; xd = x[ed]
        c_sd = xs * xd / (DT + E_ * xs + H_ * xd)
        c_ds = xd * xs / (DT + E_ * xd + H_ * xs)
        c = np.zeros_like(x)
        np.add.at(c, es, c_sd); np.add.at(c, ed, c_ds)
        return self_term + D * c
    return rhs

def make_gen(A):
    def rhs(t, x, D):
        x_pos = np.maximum(x, 0.0)
        h = (x_pos ** 2) / (1.0 + x_pos ** 2)
        return -x_pos + D * (A @ h)
    return rhs

def make_wc(A):
    TAU, MUS = 1.0, 3.0
    def rhs(t, x, D):
        sig = 1.0 / (1.0 + np.exp(-TAU * (x - MUS)))
        return -x + D * (A @ sig)
    return rhs

def make_sir(A):
    def rhs(t, state, beta):
        N = state.size // 2
        S = state[:N]; I = state[N:]
        AI = A @ I
        return np.concatenate([-beta * S * AI,
                                beta * S * AI - I])
    return rhs

def make_nm(A, k_vec, r_vec):
    K_, D_ = 10.0, 0.05
    def rhs(t, x, c):
        return (r_vec * x * (1.0 - x / K_)
                - c * x * x / (x * x + 1.0)
                + D_ * (A @ x - k_vec * x))
    return rhs

def make_eut(A, k_vec, b_vec):
    R_, P_, D_ = 1.0, 8.0, 0.05
    def rhs(t, x, alpha):
        x_pos = np.maximum(x, 0.0)
        x_p = x_pos ** P_
        return (alpha - b_vec * x
                + R_ * x_p / (x_p + 1.0)
                + D_ * (A @ x - k_vec * x))
    return rhs

def make_veg(A, k_vec, hE_vec):
    RV, HV, P_, D_ = 0.5, 0.2, 4.0, 0.05
    hE_p = hE_vec ** P_
    def rhs(t, x, Eo):
        x_pos = np.maximum(x, 0.0)
        E = Eo * HV / (HV + x_pos)
        E_p = E ** P_
        return (RV * x_pos * (1.0 - x_pos * (hE_p + E_p) / hE_p)
                + D_ * (A @ x - k_vec * x))
    return rhs

def make_msis(A, k_vec):
    def rhs(t, state, beta):
        N = state.size // 2
        rS = state[:N]; rI = state[N:]
        diff_S = (A @ (rS / k_vec) - rS)
        diff_I = (A @ (rI / k_vec) - rI)
        infect = beta * rS * rI
        return np.concatenate([-infect + rI + diff_S,
                                infect - rI + diff_I])
    return rhs

def make_msir(A, k_vec):
    def rhs(t, state, beta):
        N = state.size // 3
        rS = state[:N]; rI = state[N:2*N]
        diff_S = (A @ (rS / k_vec) - rS)
        diff_I = (A @ (rI / k_vec) - rI)
        infect = beta * rS * rI
        return np.concatenate([-infect + diff_S,
                                infect - rI + diff_I,
                                rI])
    return rhs


# -----------------------------------------------------------------
# Integrators (loose tolerance, BDF + sparse Jac for stiff dynamics)
# -----------------------------------------------------------------
def integrate_bdf(rhs, p, x0, jac, t_final):
    sol = solve_ivp(rhs, (0.0, t_final), x0, method="BDF",
                    args=(p,), rtol=RTOL, atol=ATOL, jac_sparsity=jac)
    if not sol.success:
        raise RuntimeError(f"BDF failed at p={p}: {sol.message}")
    return sol.y[:, -1]

def integrate_lsoda(rhs, p, x0, t_final):
    sol = solve_ivp(rhs, (0.0, t_final), x0, method="LSODA",
                    args=(p,), rtol=RTOL, atol=ATOL)
    if not sol.success:
        raise RuntimeError(f"LSODA failed at p={p}: {sol.message}")
    return sol.y[:, -1]


# -----------------------------------------------------------------
# Sweep helpers (cold / forward warm / reverse warm) and savers
# -----------------------------------------------------------------
def sweep_cold(rhs, ps, x0_func, integrator, t_final, dyn_label, prog_path):
    eq = np.empty((len(ps), x0_func(0.0).size))
    t0 = time.time()
    for k, p in enumerate(ps):
        eq[k] = integrator(rhs, p, x0_func(p), t_final=t_final)
        if k % 10 == 0 or k == len(ps) - 1:
            _log(prog_path, f"  {dyn_label} cold k={k}/{len(ps)-1} "
                            f"p={p:.4f} y_mean={eq[k][:eq[k].size].mean():.3f} "
                            f"({time.time()-t0:.1f}s)")
    return eq

def sweep_forward_warm(rhs, ps, x0_init, integrator, t_final, dyn_label,
                       prog_path):
    eq = np.empty((len(ps), x0_init.size))
    x_prev = x0_init.copy()
    t0 = time.time()
    for k, p in enumerate(ps):
        eq[k] = integrator(rhs, p, x_prev, t_final=t_final)
        x_prev = eq[k].copy()
        if k % 10 == 0 or k == len(ps) - 1:
            _log(prog_path, f"  {dyn_label} fwd-warm k={k}/{len(ps)-1} "
                            f"p={p:.4f} y_mean={eq[k].mean():.3f} "
                            f"({time.time()-t0:.1f}s)")
    return eq

def sweep_reverse_warm(rhs, ps, x0_init_top, integrator, t_final, dyn_label,
                       prog_path):
    """ps is ascending; x0_init_top is the IC at ps[-1]."""
    eq = np.empty((len(ps), x0_init_top.size))
    x_prev = x0_init_top.copy()
    t0 = time.time()
    for k in range(len(ps) - 1, -1, -1):
        eq[k] = integrator(rhs, ps[k], x_prev, t_final=t_final)
        x_prev = eq[k].copy()
        if k % 10 == 0 or k == len(ps) - 1:
            _log(prog_path, f"  {dyn_label} rev-warm k={k}/{len(ps)-1} "
                            f"p={ps[k]:.4f} y_mean={eq[k].mean():.3f} "
                            f"({time.time()-t0:.1f}s)")
    return eq

def write_csv(path, ps, Y):
    arr = np.column_stack([ps, Y])
    N = Y.shape[1]
    np.savetxt(path, arr, delimiter=",",
               header="p," + ",".join(f"x_{i+1}" for i in range(N)),
               comments="", fmt=["%.6f"] + ["%.6f"] * N)

def update_ranges(tag, dyn_key, p_min, p_max, y_min, y_max):
    fn = HERE / f"{tag}_bifurcation_ranges.csv"
    rows = []
    if fn.exists():
        with fn.open() as f:
            header = f.readline().rstrip("\n")
            for line in f:
                parts = line.rstrip("\n").split(",")
                if parts and parts[0]:
                    rows.append(parts)
    else:
        header = "dynamics,p_min,p_max,n_sweep,y_min,y_max"
    by_key = {r[0]: r for r in rows}
    by_key[dyn_key] = [dyn_key, f"{p_min:.6f}", f"{p_max:.6f}",
                       f"{NSWEEP:d}", f"{y_min:.6f}", f"{y_max:.6f}"]
    KEYS = ["SIS", "DW", "MUT", "GEN", "WC", "gLV", "SIR", "NM",
            "EUT", "VEG", "MSIS", "MSIR"]
    out = [header]
    for k in KEYS:
        if k in by_key:
            out.append(",".join(by_key[k]))
    fn.write_text("\n".join(out) + "\n")


# -----------------------------------------------------------------
# Per-dynamics drivers
# -----------------------------------------------------------------
def run_sis(net, tag, prog_path):
    A = net["A"]; N = net["N"]; alpha_max = net["alpha_max"]
    beta_max = 1.0 / (0.15 * alpha_max)
    grid = np.linspace(0.0, beta_max, NSWEEP)
    rhs = make_sis(A)
    integ = lambda r, p, x, t_final: integrate_lsoda(r, p, x, t_final)
    eq = sweep_reverse_warm(rhs, grid, np.full(N, 0.9), integ, 200.0,
                            "SIS", prog_path)
    write_csv(HERE / f"{tag}_sis_equilibrium_data.csv", grid, eq)
    update_ranges(tag, "SIS", 0.0, beta_max, float(eq.min()), float(eq.max()))

def run_sir(net, tag, prog_path):
    A = net["A"]; N = net["N"]; alpha_max = net["alpha_max"]
    beta_max = 1.0 / (0.15 * alpha_max)
    grid = np.linspace(0.0, beta_max, NSWEEP)
    rhs = make_sir(A)
    jac = two_block_jac(A)
    eq = np.empty((NSWEEP, N))
    t0 = time.time()
    for k, beta in enumerate(grid):
        S0 = np.full(N, 0.999); I0 = np.full(N, 0.001)
        state0 = np.concatenate([S0, I0])
        state_eq = integrate_bdf(rhs, beta, state0, jac, 400.0)
        S = state_eq[:N]; I = state_eq[N:]
        eq[k] = 1.0 - S - I
        if k % 10 == 0 or k == NSWEEP - 1:
            _log(prog_path, f"  SIR cold k={k}/{NSWEEP-1} "
                            f"beta={beta:.4f} R_mean={eq[k].mean():.3f} "
                            f"({time.time()-t0:.1f}s)")
    write_csv(HERE / f"{tag}_sir_equilibrium_data.csv", grid, eq)
    update_ranges(tag, "SIR", 0.0, beta_max, float(eq.min()), float(eq.max()))

def _meta_widen(net, dyn_kind, prog_path):
    """Returns the widened beta_max."""
    A = net["A"]; N = net["N"]; alpha_max = net["alpha_max"]
    k_vec = net["k_vec"]
    beta_max = 1.0 / (0.15 * alpha_max)
    if dyn_kind == "MSIS":
        rhs = make_msis(A, k_vec); jac = two_block_jac(A)
        rS = np.full(N, 0.9); rI = np.full(N, 0.1)
        ic = np.concatenate([rS, rI])
    else:
        rhs = make_msir(A, k_vec); jac = msir_jac(A)
        rS = np.full(N, 0.999); rI = np.full(N, 0.001); rR = np.zeros(N)
        ic = np.concatenate([rS, rI, rR])
    for trial in range(20):
        beta_test = 0.5 * beta_max
        state_eq = integrate_bdf(rhs, beta_test, ic, jac, 100.0)
        if dyn_kind == "MSIS":
            obs = state_eq[N:].mean()
        else:
            obs = state_eq[2*N:].mean()
        _log(prog_path, f"  {dyn_kind} widen trial={trial} "
                        f"beta_max={beta_max:.4f} y={obs:.4f}")
        if obs >= THRESHOLD_META:
            return beta_max, rhs, jac, ic
        beta_max *= 2.0
    return beta_max, rhs, jac, ic

def run_msis(net, tag, prog_path):
    N = net["N"]
    beta_max, rhs, jac, ic = _meta_widen(net, "MSIS", prog_path)
    grid = np.linspace(0.0, beta_max, NSWEEP)
    eq = np.empty((NSWEEP, N))
    state_prev = ic.copy()
    t0 = time.time()
    for k in range(NSWEEP - 1, -1, -1):
        state_eq = integrate_bdf(rhs, grid[k], state_prev, jac, 100.0)
        state_prev = state_eq.copy()
        eq[k] = state_eq[N:]
        if k % 10 == 0 or k == NSWEEP - 1:
            _log(prog_path, f"  MSIS rev-warm k={k}/{NSWEEP-1} "
                            f"beta={grid[k]:.4f} rho_I_mean={eq[k].mean():.3f} "
                            f"({time.time()-t0:.1f}s)")
    write_csv(HERE / f"{tag}_msis_equilibrium_data.csv", grid, eq)
    update_ranges(tag, "MSIS", 0.0, beta_max, float(eq.min()), float(eq.max()))

def run_msir(net, tag, prog_path):
    N = net["N"]
    beta_max, rhs, jac, ic = _meta_widen(net, "MSIR", prog_path)
    grid = np.linspace(0.0, beta_max, NSWEEP)
    eq = np.empty((NSWEEP, N))
    t0 = time.time()
    for k, beta in enumerate(grid):
        state_eq = integrate_bdf(rhs, beta, ic, jac, 100.0)
        eq[k] = state_eq[2*N:]
        if k % 10 == 0 or k == NSWEEP - 1:
            _log(prog_path, f"  MSIR cold k={k}/{NSWEEP-1} "
                            f"beta={beta:.4f} rho_R_mean={eq[k].mean():.3f} "
                            f"({time.time()-t0:.1f}s)")
    write_csv(HERE / f"{tag}_msir_equilibrium_data.csv", grid, eq)
    update_ranges(tag, "MSIR", 0.0, beta_max, float(eq.min()), float(eq.max()))

def run_dw(net, tag, prog_path):
    A = net["A"]; N = net["N"]; kbar = net["kbar"]
    D_max = 2.564 / kbar
    grid = np.linspace(0.0, D_max, NSWEEP)
    rhs = make_dw(A); jac = per_node_jac(A)
    integ = lambda r, p, x, t_final: integrate_bdf(r, p, x, jac, t_final)
    eq = sweep_cold(rhs, grid, lambda p: np.ones(N), integ, 50.0,
                    "DW", prog_path)
    write_csv(HERE / f"{tag}_doublewell_equilibrium_data.csv", grid, eq)
    update_ranges(tag, "DW", 0.0, D_max, float(eq.min()), float(eq.max()))

def run_mut(net, tag, prog_path):
    N = net["N"]; kbar = net["kbar"]; es = net["es"]; ed = net["ed"]
    A = net["A"]
    D_max = 10.258 / kbar
    grid = np.linspace(0.0, D_max, NSWEEP)
    rhs = make_mut(es, ed); jac = per_node_jac(A)
    integ = lambda r, p, x, t_final: integrate_bdf(r, p, x, jac, t_final)
    eq = sweep_forward_warm(rhs, grid, np.full(N, 0.001), integ, 200.0,
                            "MUT", prog_path)
    write_csv(HERE / f"{tag}_mutualistic_equilibrium_data.csv", grid, eq)
    update_ranges(tag, "MUT", 0.0, D_max, float(eq.min()), float(eq.max()))

def run_gen(net, tag, prog_path):
    A = net["A"]; N = net["N"]; alpha_max = net["alpha_max"]
    D_max = 11.510 / alpha_max
    grid = np.linspace(0.0, D_max, NSWEEP)
    rhs = make_gen(A); jac = per_node_jac(A)
    integ = lambda r, p, x, t_final: integrate_bdf(r, p, x, jac, t_final)
    # reverse warm-start, IC at top = x=10
    eq = sweep_reverse_warm(rhs, grid, np.full(N, 10.0), integ, 200.0,
                            "GEN", prog_path)
    write_csv(HERE / f"{tag}_genereg_equilibrium_data.csv", grid, eq)
    update_ranges(tag, "GEN", 0.0, D_max, float(eq.min()), float(eq.max()))

def run_wc(net, tag, prog_path):
    A = net["A"]; N = net["N"]; alpha_max = net["alpha_max"]
    D_max = 43.166 / alpha_max
    grid = np.linspace(0.0, D_max, NSWEEP)
    rhs = make_wc(A)
    integ = lambda r, p, x, t_final: integrate_lsoda(r, p, x, t_final)
    eq = sweep_forward_warm(rhs, grid, np.full(N, 0.1), integ, 200.0,
                            "WC", prog_path)
    write_csv(HERE / f"{tag}_wc_equilibrium_data.csv", grid, eq)
    update_ranges(tag, "WC", 0.0, D_max, float(eq.min()), float(eq.max()))

def run_glv(net, tag, prog_path):
    A = net["A"]; N = net["N"]; alpha_max = net["alpha_max"]
    Dc = (alpha_max + 1.0) / alpha_max
    grid = np.linspace(0.05 * Dc, 0.95 * Dc, NSWEEP)
    eq = np.empty((NSWEEP, N))
    c = alpha_max + 1.0
    A_csc = A.tocsc()
    for k, D in enumerate(grid):
        # sparse spsolve since N is large
        M = D * A_csc - c * sp.eye(N, format="csc")
        eq[k] = -spsolve(M, np.ones(N))
    _log(prog_path, f"  gLV closed-form 51 pts done, "
                    f"y-range [{eq.min():.3f}, {eq.max():.3f}]")
    write_csv(HERE / f"{tag}_glv_equilibrium_data.csv", grid, eq)
    update_ranges(tag, "gLV", grid[0], grid[-1],
                  float(eq.min()), float(eq.max()))

def run_nm(net, tag, prog_path):
    A = net["A"]; N = net["N"]; k_vec = net["k_vec"]
    rng_r = np.random.default_rng(7)
    r_vec = 0.6 + 0.4 * rng_r.random(N)
    rhs = make_nm(A, k_vec, r_vec); jac = per_node_jac(A)
    grid = np.linspace(1.0, 3.0, NSWEEP)
    integ = lambda r, p, x, t_final: integrate_lsoda(r, p, x, t_final)
    eq = sweep_forward_warm(rhs, grid, np.full(N, 8.0), integ, 200.0,
                            "NM", prog_path)
    write_csv(HERE / f"{tag}_noymeir_equilibrium_data.csv", grid, eq)
    update_ranges(tag, "NM", 1.0, 3.0, float(eq.min()), float(eq.max()))

def run_eut(net, tag, prog_path):
    A = net["A"]; N = net["N"]; k_vec = net["k_vec"]
    rng_b = np.random.default_rng(11)
    b_vec = 0.8 + 0.4 * rng_b.random(N)
    rhs = make_eut(A, k_vec, b_vec); jac = per_node_jac(A)
    grid = np.linspace(0.1, 0.9, NSWEEP)
    integ = lambda r, p, x, t_final: integrate_bdf(r, p, x, jac, t_final)
    eq = sweep_forward_warm(rhs, grid, np.full(N, 0.5), integ, 200.0,
                            "EUT", prog_path)
    write_csv(HERE / f"{tag}_eutroph_equilibrium_data.csv", grid, eq)
    update_ranges(tag, "EUT", 0.1, 0.9, float(eq.min()), float(eq.max()))

def run_veg(net, tag, prog_path):
    A = net["A"]; N = net["N"]; k_vec = net["k_vec"]
    rng_h = np.random.default_rng(13)
    hE_vec = 1.0 + 2.0 * rng_h.random(N)
    rhs = make_veg(A, k_vec, hE_vec); jac = per_node_jac(A)
    grid = np.linspace(2.0, 12.0, NSWEEP)
    integ = lambda r, p, x, t_final: integrate_bdf(r, p, x, jac, t_final)
    eq = sweep_forward_warm(rhs, grid, np.full(N, 0.6), integ, 200.0,
                            "VEG", prog_path)
    write_csv(HERE / f"{tag}_vegturb_equilibrium_data.csv", grid, eq)
    update_ranges(tag, "VEG", 2.0, 12.0, float(eq.min()), float(eq.max()))


# -----------------------------------------------------------------
# Master driver
# -----------------------------------------------------------------
DYNAMICS = [
    ("SIS",  run_sis),
    ("SIR",  run_sir),
    ("MSIS", run_msis),
    ("MSIR", run_msir),
    ("DW",   run_dw),
    ("MUT",  run_mut),
    ("GEN",  run_gen),
    ("WC",   run_wc),
    ("gLV",  run_glv),
    ("NM",   run_nm),
    ("EUT",  run_eut),
    ("VEG",  run_veg),
]


def main():
    if len(sys.argv) != 2:
        print(f"usage: python3 {sys.argv[0]} <network_tag>")
        sys.exit(1)
    tag = sys.argv[1]
    edge_file = HERE / f"{tag}_network.edgelist"
    if not edge_file.exists():
        print(f"ERROR: {edge_file} not found")
        sys.exit(1)

    prog_path = HERE / f"{tag}_v9_progress.log"
    _log(prog_path, f"=== START {tag} (all 12 dynamics, v9 protocols) ===")
    net = load_network(tag)
    _log(prog_path, f"  N={net['N']}, M={net['M']}, "
                    f"<k>={net['kbar']:.3f}, alpha_max={net['alpha_max']:.3f}")

    for dyn_name, runner in DYNAMICS:
        # Skip if CSV already exists (resume support)
        csv_map = {"SIS": "sis", "DW": "doublewell", "MUT": "mutualistic",
                   "GEN": "genereg", "WC": "wc", "gLV": "glv",
                   "SIR": "sir", "NM": "noymeir", "EUT": "eutroph",
                   "VEG": "vegturb", "MSIS": "msis", "MSIR": "msir"}
        csv = HERE / f"{tag}_{csv_map[dyn_name]}_equilibrium_data.csv"
        if csv.exists():
            _log(prog_path, f"SKIP: {dyn_name} (CSV already on disk)")
            continue
        _log(prog_path, f"BEGIN: {dyn_name}")
        t0 = time.time()
        try:
            runner(net, tag, prog_path)
            _log(prog_path, f"DONE: {dyn_name} in {time.time()-t0:.1f}s")
        except Exception as e:
            _log(prog_path, f"FAIL: {dyn_name} -- {e}")
            raise

    _log(prog_path, f"ALL 12 DYNAMICS DONE for {tag}")


if __name__ == "__main__":
    main()
