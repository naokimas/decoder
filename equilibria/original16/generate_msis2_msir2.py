"""Generate MSIS2 / MSIR2 equilibrium data for the 13 small networks.

MSIS2 / MSIR2 are identical to MSIS / MSIR except the infected-compartment
diffusion rate is D_I = 0.2 (instead of D_I = 1.0). Susceptible diffusion
is unchanged (D_S = 1.0); recovery rate mu = 1.0 unchanged. Same
beta-sweep procedure as MSIS / MSIR (adaptive beta_max widening; MSIS2
reverse-warm; MSIR2 cold-start), same NSWEEP = 51, same loose
tolerances (rtol=1e-4, atol=1e-6).

Outputs:
  <tag>_msis2_equilibrium_data.csv
  <tag>_msir2_equilibrium_data.csv
  <tag>_bifurcation_ranges_msis2_msir2.csv  (separate from the existing
                                             <tag>_bifurcation_ranges.csv
                                             so the manuscript pipeline
                                             is not perturbed)
Progress log: msis2_msir2_progress.log
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the v9 helpers verbatim.
import run_all_12_dynamics_v9 as v9

HERE = Path(__file__).resolve().parent
NSWEEP = 51
RTOL, ATOL = 1e-4, 1e-6
THRESHOLD_META = 0.1    # adaptive beta_max widening acceptance (Methods sec:autotrim)
D_I_NEW = 0.2
D_S_NEW = 1.0

NETWORKS = ["dolphin", "prox", "metabolic", "gkk", "lfr", "er", "ba", "hk",
            "euroroad", "email", "flybi", "reactome", "route_views"]


def _log(prog_path: Path, msg: str):
    with prog_path.open("a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg, flush=True)


def make_msis2(A, k_vec, D_S=D_S_NEW, D_I=D_I_NEW):
    def rhs(t, state, beta):
        N = state.size // 2
        rS = state[:N]; rI = state[N:]
        diff_S = D_S * (A @ (rS / k_vec) - rS)
        diff_I = D_I * (A @ (rI / k_vec) - rI)
        infect = beta * rS * rI
        return np.concatenate([-infect + rI + diff_S,
                                infect - rI + diff_I])
    return rhs


def make_msir2(A, k_vec, D_S=D_S_NEW, D_I=D_I_NEW):
    def rhs(t, state, beta):
        N = state.size // 3
        rS = state[:N]; rI = state[N:2*N]
        diff_S = D_S * (A @ (rS / k_vec) - rS)
        diff_I = D_I * (A @ (rI / k_vec) - rI)
        infect = beta * rS * rI
        return np.concatenate([-infect + diff_S,
                                infect - rI + diff_I,
                                rI])
    return rhs


def widen_beta_max(net, dyn_kind, prog_path):
    A = net["A"]; N = net["N"]; alpha_max = net["alpha_max"]
    k_vec = net["k_vec"]
    beta_max = 1.0 / (0.15 * alpha_max)
    if dyn_kind == "MSIS2":
        rhs = make_msis2(A, k_vec); jac = v9.two_block_jac(A)
        rS = np.full(N, 0.9); rI = np.full(N, 0.1)
        ic = np.concatenate([rS, rI])
    else:
        rhs = make_msir2(A, k_vec); jac = v9.msir_jac(A)
        rS = np.full(N, 0.999); rI = np.full(N, 0.001); rR = np.zeros(N)
        ic = np.concatenate([rS, rI, rR])
    for trial in range(20):
        beta_test = 0.5 * beta_max
        state_eq = v9.integrate_bdf(rhs, beta_test, ic, jac, 100.0)
        if dyn_kind == "MSIS2":
            obs = state_eq[N:].mean()
        else:
            obs = state_eq[2*N:].mean()
        _log(prog_path, f"  {dyn_kind} widen trial={trial} "
                        f"beta_max={beta_max:.4f} y={obs:.4f}")
        if obs >= THRESHOLD_META:
            return beta_max, rhs, jac, ic
        beta_max *= 2.0
    return beta_max, rhs, jac, ic


def run_msis2(net, tag, prog_path):
    out_csv = HERE / f"{tag}_msis2_equilibrium_data.csv"
    if out_csv.exists():
        _log(prog_path, f"SKIP: {tag}_msis2 (CSV already on disk)")
        return None
    _log(prog_path, f"BEGIN: {tag} MSIS2")
    N = net["N"]
    beta_max, rhs, jac, ic = widen_beta_max(net, "MSIS2", prog_path)
    grid = np.linspace(0.0, beta_max, NSWEEP)
    eq = np.empty((NSWEEP, N))
    state_prev = ic.copy()
    t0 = time.time()
    for k in range(NSWEEP - 1, -1, -1):
        state_eq = v9.integrate_bdf(rhs, grid[k], state_prev, jac, 100.0)
        state_prev = state_eq.copy()
        eq[k] = state_eq[N:]
        if k % 10 == 0 or k == NSWEEP - 1:
            _log(prog_path, f"  MSIS2 rev-warm k={k}/{NSWEEP-1} "
                            f"beta={grid[k]:.4f} rho_I_mean={eq[k].mean():.3f} "
                            f"({time.time()-t0:.1f}s)")
    v9.write_csv(out_csv, grid, eq)
    info = ("MSIS2", 0.0, float(beta_max), float(eq.min()), float(eq.max()))
    _log(prog_path, f"DONE: {tag} MSIS2 in {time.time()-t0:.1f}s")
    return info


def run_msir2(net, tag, prog_path):
    out_csv = HERE / f"{tag}_msir2_equilibrium_data.csv"
    if out_csv.exists():
        _log(prog_path, f"SKIP: {tag}_msir2 (CSV already on disk)")
        return None
    _log(prog_path, f"BEGIN: {tag} MSIR2")
    N = net["N"]
    beta_max, rhs, jac, ic = widen_beta_max(net, "MSIR2", prog_path)
    grid = np.linspace(0.0, beta_max, NSWEEP)
    eq = np.empty((NSWEEP, N))
    t0 = time.time()
    for k, beta in enumerate(grid):
        state_eq = v9.integrate_bdf(rhs, beta, ic, jac, 100.0)
        eq[k] = state_eq[2*N:]
        if k % 10 == 0 or k == NSWEEP - 1:
            _log(prog_path, f"  MSIR2 cold k={k}/{NSWEEP-1} "
                            f"beta={beta:.4f} rho_R_mean={eq[k].mean():.3f} "
                            f"({time.time()-t0:.1f}s)")
    v9.write_csv(out_csv, grid, eq)
    info = ("MSIR2", 0.0, float(beta_max), float(eq.min()), float(eq.max()))
    _log(prog_path, f"DONE: {tag} MSIR2 in {time.time()-t0:.1f}s")
    return info


def update_ranges_extra(tag, infos):
    fn = HERE / f"{tag}_bifurcation_ranges_msis2_msir2.csv"
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
    for info in infos:
        if info is None:
            continue
        key, p_min, p_max, y_min, y_max = info
        by_key[key] = [key, f"{p_min:.6f}", f"{p_max:.6f}",
                       f"{NSWEEP:d}", f"{y_min:.6f}", f"{y_max:.6f}"]
    out_lines = [header]
    for key in ["MSIS2", "MSIR2"]:
        if key in by_key:
            out_lines.append(",".join(by_key[key]))
    fn.write_text("\n".join(out_lines) + "\n")


def main():
    prog_path = HERE / "msis2_msir2_progress.log"
    _log(prog_path, "=== MSIS2 / MSIR2 generation: 13 small networks ===")
    _log(prog_path, f"  D_S = {D_S_NEW}, D_I = {D_I_NEW}, mu = 1.0")
    for tag in NETWORKS:
        edgefile = HERE / f"{tag}_network.edgelist"
        if not edgefile.exists():
            _log(prog_path, f"MISSING: {edgefile.name}; skip {tag}")
            continue
        _log(prog_path, f"--- {tag} ---")
        net = v9.load_network(tag)
        _log(prog_path, f"  N={net['N']}, M={net['M']}, "
                        f"<k>={net['kbar']:.3f}, alpha_max={net['alpha_max']:.3f}")
        infos = []
        for fn in (run_msis2, run_msir2):
            try:
                infos.append(fn(net, tag, prog_path))
            except Exception as exc:
                _log(prog_path, f"  ERROR ({fn.__name__}, {tag}): {exc}")
        update_ranges_extra(tag, infos)
    _log(prog_path, "ALL MSIS2 / MSIR2 STAGES DONE")


if __name__ == "__main__":
    main()
