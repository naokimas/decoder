"""Generate MSIS2 / MSIR2 / EUT2 / VEG2 equilibrium data for spanish
(the 14th network, N=11558). Reuses the per-dynamics drivers from
generate_msis2_msir2.py and generate_eut2_veg2.py; only the network
list is overridden.

Outputs (per dynamics):
  spanish_msis2_equilibrium_data.csv
  spanish_msir2_equilibrium_data.csv
  spanish_eutroph2_equilibrium_data.csv
  spanish_vegturb2_equilibrium_data.csv
  spanish_bifurcation_ranges_msis2_msir2.csv
  spanish_bifurcation_ranges_eut2_veg2.csv
Progress log: spanish_param_variants_progress.log
"""
from __future__ import annotations
import time
from pathlib import Path

import generate_msis2_msir2 as g_meta
import generate_eut2_veg2 as g_eut
import run_all_12_dynamics_v9 as v9

HERE = Path(__file__).resolve().parent
TAG = "spanish"


def _log(prog: Path, msg: str):
    with prog.open("a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg, flush=True)


def main():
    prog = HERE / "spanish_param_variants_progress.log"
    _log(prog, f"=== {TAG} parameter-variant generation ===")
    _log(prog, "  MSIS2/MSIR2: D_I=0.2 (else as MSIS1/MSIR1).")
    _log(prog, "  EUT2: Hill p=2 (else as EUT1); VEG2: Hill p=1 (else as VEG1).")
    edgefile = HERE / f"{TAG}_network.edgelist"
    if not edgefile.exists():
        _log(prog, f"MISSING: {edgefile.name}; abort")
        return
    net = v9.load_network(TAG)
    _log(prog, f"  N={net['N']}, M={net['M']}, "
               f"<k>={net['kbar']:.3f}, alpha_max={net['alpha_max']:.3f}")

    # ---- MSIS2 / MSIR2 ----
    meta_infos = []
    for fn in (g_meta.run_msis2, g_meta.run_msir2):
        try:
            meta_infos.append(fn(net, TAG, prog))
        except Exception as exc:
            _log(prog, f"  ERROR ({fn.__name__}): {exc}")
    g_meta.update_ranges_extra(TAG, meta_infos)

    # ---- EUT2 / VEG2 ----
    eut_infos = []
    for fn in (g_eut.run_eut2, g_eut.run_veg2):
        try:
            eut_infos.append(fn(net, TAG, prog))
        except Exception as exc:
            _log(prog, f"  ERROR ({fn.__name__}): {exc}")
    g_eut.update_ranges_extra(TAG, eut_infos)

    _log(prog, f"ALL SPANISH PARAM-VARIANT STAGES DONE")


if __name__ == "__main__":
    main()
