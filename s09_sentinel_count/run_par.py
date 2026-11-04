# =====================================================================
# SI Section S9: Robustness to the number of sentinel nodes
#
# This script: parallel driver for gen_ba.py
# Feeds      : Figs. S14 to S18, via compute_scale.py
# Produces   : as gen_ba.py
# Runtime    : days -- do not re-run
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Parallel driver for the BA size sweep.

gen_ba.py runs the ten producer stages one after another in a single
process.  At N = 10^4 that is far too slow, and the machine has eight idle
cores.  This driver runs exactly the same producers, with exactly the same
arguments, as separate processes wherever the data dependencies allow it.
No solver, tolerance, grid or constant is changed; only the order and the
concurrency are.

Dependencies that force the ordering:

  * every later stage reads <tag>_bifurcation_ranges.csv, which the
    generate_data step creates, so that step must finish first;
  * inside generate_data the three systems NM, EUT and VEG are independent,
    so they run concurrently and a final cheap pass (which reuses the CSVs
    just written) rewrites a consistent ranges file;
  * regenerate_sis_dw_v3 and regenerate_mut_gen_wc_v3 are the only stages
    that rewrite rows of the ranges file, so they never run at the same time
    as each other or as a stage that reads it;
  * SIR, MSIS and MSIR read the SIS row that sis_dw writes;
  * msis2_msir2 reads the MSIS and MSIR sweeps; wc_kbar overwrites the WC
    sweep written by mut_gen_wc; eut2_veg2 reads the EUT and VEG sweeps;
  * the GEN auto-trim and the LDW generator come last, as in gen_ba.py.

Usage:  python3 run_par.py <tag> [--dc VALUE] [--jobs N] [--fastlu]
                            [--skip step1,step2,...]
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
LARGE_TIER = "--large" in sys.argv

# (step name, systems restriction or None) grouped into levels; the steps
# inside a level run concurrently, levels run in order.
LEVELS = [
    [("generate_data", ["NM"]), ("generate_data", ["EUT"]),
     ("generate_data", ["VEG"])],
    [("generate_data", ["NM", "EUT", "VEG"])],      # cheap: rewrites ranges
    [("regenerate_sis_dw_v3", None)],               # writes SIS, DW rows
    [("regenerate_mut_gen_wc_v3", None)],           # writes MUT, GEN, WC rows
    [("regenerate_sir_v3", None), ("regenerate_msis_v3", None),
     ("regenerate_msir_v3", None), ("regenerate_wc_kbar_v89", None),
     ("regenerate_glv_eps", None), ("generate_eut2_veg2", None)],
    [("generate_msis2_msir2", None)],
    [("trim", None)],
    [("ldw", None)],
]


def launch(tag, step, systems, dc, logdir, fastlu=False):
    cmd = [PY, "-u", str(HERE / "gen_ba.py"), tag, f"--step={step}"]
    if LARGE_TIER:
        cmd.append("--large")
    if systems:
        cmd.append("--systems=" + ",".join(systems))
    if dc is not None and step == "ldw":
        cmd.append(f"--dc={dc}")
    if fastlu:
        cmd.append("--fastlu")
    name = step + ("-" + "_".join(systems) if systems else "")
    log = open(logdir / f"{tag}_{name}.log", "w")
    return name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tag = args[0]
    dc = None
    jobs = 2          # each ba4p0 step can hold tens of GB of anonymous
                      # memory in the sparse LU factors, so keep the
                      # concurrency low at the largest size
    fastlu = "--fastlu" in sys.argv
    skip = set()
    for i, a in enumerate(sys.argv):
        if a == "--dc":
            dc = float(sys.argv[i + 1])
        elif a.startswith("--dc="):
            dc = float(a.split("=", 1)[1])
        elif a == "--jobs":
            jobs = int(sys.argv[i + 1])
        elif a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        elif a.startswith("--skip="):
            skip = set(a.split("=", 1)[1].split(","))
    logdir = HERE / "parlogs"
    logdir.mkdir(exist_ok=True)
    t_all = time.time()
    for li, level in enumerate(LEVELS):
        print(f"\n===== level {li}: {len(level)} step(s), up to {jobs} at a "
              f"time =====", flush=True)
        failed = []
        pending = [(st, sy) for st, sy in level if st not in skip]
        if not pending:
            print("  (all steps of this level already done)", flush=True)
            continue
        running = []
        while pending or running:
            while pending and len(running) < jobs:
                step, systems = pending.pop(0)
                t0 = time.time()
                name, proc, log = launch(tag, step, systems, dc, logdir,
                                         fastlu)
                print(f"  -> {name}", flush=True)
                running.append((name, proc, log, t0))
            time.sleep(5)
            for item in list(running):
                name, proc, log, t0 = item
                if proc.poll() is not None:
                    running.remove(item)
                    log.close()
                    status = "ok" if proc.returncode == 0 else \
                        f"FAILED rc={proc.returncode}"
                    print(f"  <- {name}: {status} "
                          f"[{time.time()-t0:.0f}s]", flush=True)
                    if proc.returncode != 0:
                        # A stage killed by the OS (rc = -9) used to abort the
                        # whole run and take its siblings with it.  Let the
                        # siblings finish and retry this one alone afterwards.
                        print(f"     see parlogs/{tag}_{name}.log", flush=True)
                        failed.append(name)
        for (step, systems) in level:
            nm = step + ("-" + "_".join(systems) if systems else "")
            if nm not in failed:
                continue
            print(f"  retrying {nm} alone", flush=True)
            t0 = time.time()
            _, proc, log = launch(tag, step, systems, dc, logdir, fastlu)
            proc.wait(); log.close()
            print(f"  <- {nm}: "
                  f"{'ok' if proc.returncode == 0 else 'FAILED AGAIN'} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            if proc.returncode != 0:
                sys.exit(1)
    print(f"\n===== {tag} done in {time.time()-t_all:.0f}s =====", flush=True)


if __name__ == "__main__":
    main()
