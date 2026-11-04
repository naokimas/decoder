"""Generate EUT2 / VEG2 equilibrium data for the 13 small networks.

EUT2 = current EUT (eutrophication) but with Hill exponent P = 2 (was 8).
VEG2 = current VEG (vegetation--turbidity) but with Hill exponent P = 1
       (was 4).

All other parameters, b_vec / hE_vec heterogeneity (same RNG seeds),
initial conditions, parameter sweep ranges, integrator (BDF + sparse
Jacobian) match EUT / VEG. The only change is P.

Sanity guard: if at any sweep point the equilibrium has min(eq) < -1
or max(eq) > 100 across the full sweep, the dynamics is flagged
DOUBTFUL for that network and the CSV is NOT written. This keeps
unphysical/unmanageable runs out of downstream analysis.

If the MSIS2 / MSIR2 generation script is still running (its progress
log lacks the "ALL MSIS2 / MSIR2 STAGES DONE" marker), this script
waits and polls every 90 seconds.

Outputs:
  <tag>_eutroph2_equilibrium_data.csv
  <tag>_vegturb2_equilibrium_data.csv
  <tag>_bifurcation_ranges_eut2_veg2.csv
Progress log: eut2_veg2_progress.log
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np

import run_all_12_dynamics_v9 as v9

HERE = Path(__file__).resolve().parent
NSWEEP = 51
RTOL, ATOL = 1e-4, 1e-6
EUT2_P = 2.0      # EUT1 uses 8.0
VEG2_P = 1.0      # VEG1 uses 4.0
SANITY_MIN = -1.0
SANITY_MAX = 100.0

NETWORKS = ["dolphin", "prox", "metabolic", "gkk", "lfr", "er", "ba", "hk",
            "euroroad", "email", "flybi", "reactome", "route_views"]

UPSTREAM_LOG = HERE / "msis2_msir2_progress.log"
UPSTREAM_DONE = "ALL MSIS2 / MSIR2 STAGES DONE"


def _log(prog_path: Path, msg: str):
    with prog_path.open("a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg, flush=True)


def wait_for_upstream(prog_path: Path):
    if not UPSTREAM_LOG.exists():
        _log(prog_path, "upstream log missing; proceeding without wait.")
        return
    text = UPSTREAM_LOG.read_text()
    if UPSTREAM_DONE in text:
        _log(prog_path, "MSIS2/MSIR2 already done; proceeding.")
        return
    _log(prog_path, "waiting for MSIS2/MSIR2 to finish (poll 90s)...")
    while True:
        time.sleep(90)
        if UPSTREAM_DONE in UPSTREAM_LOG.read_text():
            _log(prog_path, "MSIS2/MSIR2 done; proceeding.")
            return


def make_eut2(A, k_vec, b_vec):
    R_, P_, D_ = 1.0, EUT2_P, 0.05
    def rhs(t, x, alpha):
        x_pos = np.maximum(x, 0.0)
        x_p = x_pos ** P_
        return (alpha - b_vec * x
                + R_ * x_p / (x_p + 1.0)
                + D_ * (A @ x - k_vec * x))
    return rhs


def make_veg2(A, k_vec, hE_vec):
    RV, HV, P_, D_ = 0.5, 0.2, VEG2_P, 0.05
    hE_p = hE_vec ** P_
    def rhs(t, x, Eo):
        x_pos = np.maximum(x, 0.0)
        E = Eo * HV / (HV + x_pos)
        E_p = E ** P_
        return (RV * x_pos * (1.0 - x_pos * (hE_p + E_p) / hE_p)
                + D_ * (A @ x - k_vec * x))
    return rhs


def sanity_ok(eq, prog_path, label):
    lo = float(eq.min()); hi = float(eq.max())
    if lo < SANITY_MIN or hi > SANITY_MAX:
        _log(prog_path, f"  DOUBTFUL ({label}): eq range [{lo:.3f}, {hi:.3f}] "
                        f"outside [{SANITY_MIN}, {SANITY_MAX}]; CSV not written.")
        return False
    return True


def run_eut2(net, tag, prog_path):
    out_csv = HERE / f"{tag}_eutroph2_equilibrium_data.csv"
    if out_csv.exists():
        _log(prog_path, f"SKIP: {tag}_eut2 (CSV already on disk)")
        return None
    _log(prog_path, f"BEGIN: {tag} EUT2 (P={EUT2_P})")
    A = net["A"]; N = net["N"]; k_vec = net["k_vec"]
    rng_b = np.random.default_rng(11)            # same seed as EUT1
    b_vec = 0.8 + 0.4 * rng_b.random(N)
    rhs = make_eut2(A, k_vec, b_vec); jac = v9.per_node_jac(A)
    grid = np.linspace(0.1, 0.9, NSWEEP)
    integ = lambda r, p, x, t_final: v9.integrate_bdf(r, p, x, jac, t_final)
    t0 = time.time()
    try:
        eq = v9.sweep_forward_warm(rhs, grid, np.full(N, 0.5), integ,
                                    200.0, "EUT2", prog_path)
    except Exception as exc:
        _log(prog_path, f"  ERROR (EUT2, {tag}): {exc}")
        return None
    if not sanity_ok(eq, prog_path, f"{tag} EUT2"):
        return None
    v9.write_csv(out_csv, grid, eq)
    info = ("EUT2", 0.1, 0.9, float(eq.min()), float(eq.max()))
    _log(prog_path, f"DONE: {tag} EUT2 in {time.time()-t0:.1f}s")
    return info


def run_veg2(net, tag, prog_path):
    out_csv = HERE / f"{tag}_vegturb2_equilibrium_data.csv"
    if out_csv.exists():
        _log(prog_path, f"SKIP: {tag}_veg2 (CSV already on disk)")
        return None
    _log(prog_path, f"BEGIN: {tag} VEG2 (P={VEG2_P})")
    A = net["A"]; N = net["N"]; k_vec = net["k_vec"]
    rng_h = np.random.default_rng(13)            # same seed as VEG1
    hE_vec = 1.0 + 2.0 * rng_h.random(N)
    rhs = make_veg2(A, k_vec, hE_vec); jac = v9.per_node_jac(A)
    grid = np.linspace(2.0, 12.0, NSWEEP)
    integ = lambda r, p, x, t_final: v9.integrate_bdf(r, p, x, jac, t_final)
    t0 = time.time()
    try:
        eq = v9.sweep_forward_warm(rhs, grid, np.full(N, 0.6), integ,
                                    200.0, "VEG2", prog_path)
    except Exception as exc:
        _log(prog_path, f"  ERROR (VEG2, {tag}): {exc}")
        return None
    if not sanity_ok(eq, prog_path, f"{tag} VEG2"):
        return None
    v9.write_csv(out_csv, grid, eq)
    info = ("VEG2", 2.0, 12.0, float(eq.min()), float(eq.max()))
    _log(prog_path, f"DONE: {tag} VEG2 in {time.time()-t0:.1f}s")
    return info


def update_ranges_extra(tag, infos):
    fn = HERE / f"{tag}_bifurcation_ranges_eut2_veg2.csv"
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
    for key in ["EUT2", "VEG2"]:
        if key in by_key:
            out_lines.append(",".join(by_key[key]))
    fn.write_text("\n".join(out_lines) + "\n")


def main():
    prog_path = HERE / "eut2_veg2_progress.log"
    _log(prog_path, "=== EUT2 / VEG2 generation: 13 small networks ===")
    _log(prog_path, f"  EUT2 Hill P = {EUT2_P} (EUT1 uses 8); "
                    f"VEG2 Hill P = {VEG2_P} (VEG1 uses 4)")
    _log(prog_path, f"  sanity guard: drop if eq outside "
                    f"[{SANITY_MIN}, {SANITY_MAX}]")
    wait_for_upstream(prog_path)
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
        for fn in (run_eut2, run_veg2):
            try:
                infos.append(fn(net, tag, prog_path))
            except Exception as exc:
                _log(prog_path, f"  ERROR ({fn.__name__}, {tag}): {exc}")
        update_ranges_extra(tag, infos)
    _log(prog_path, "ALL EUT2 / VEG2 STAGES DONE")


if __name__ == "__main__":
    main()
