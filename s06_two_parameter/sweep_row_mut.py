# =====================================================================
# SI Section S6: Two-parameter families of dynamics
#
# This script: one row of the MUT2P parameter sweep
# Feeds      : Figs. S7 and S8, via merge_rows_mut.py
# Produces   : rows/<tag>_<i>.npz
# Runtime    : hours per row
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Compute ONE B row of the MUT two-parameter grid, for one network.

The forward warm chain runs along D *within* a B row; B rows never touch
each other.  So the 8 rows the serial generator does back to back can be
run as 8 processes with an identical result.

    python3 sweep_row_mut.py <tag> <b_index>

Each solve is extended until settled, as in generate_mut2p.py.
Writes data/_rows_mut2p/<tag>_<b_index>.npz; merge_rows_mut.py
reassembles the same CSV and summary shard the serial path writes.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mut2p as m                                            # noqa: E402
from generate_mut2p import ND, NBB, B_LO, B_HI               # noqa: E402


def main():
    tag, bi = sys.argv[1], int(sys.argv[2])
    out = HERE / "data" / "_rows_mut2p"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    net = m.load_network(tag)
    N = net["N"]
    jac = m.jac_sparsity(net)
    D_max = m.MUT_CONSTANT / net["kbar"]
    Ds = np.linspace(0.0, D_max, ND)
    B = np.linspace(B_LO, B_HI, NBB)[bi]

    Y = np.empty((ND, N))
    ok_all = True
    worst_res = 0.0
    worst_t = 0.0
    x_prev = np.full(N, m.X0_LOW)
    for di, D in enumerate(Ds):
        y, ok, t_used, res = m.equilibrate_settled(net, D, B, x0=x_prev,
                                                   jac=jac)
        worst_res = max(worst_res, res)
        worst_t = max(worst_t, t_used)
        ok_all = ok_all and ok
        x_prev = y.copy()
        Y[di] = y
        print(f"  {tag} B-row {bi} (B={B:.4f}) D={D:.5f}: mean x={y.mean():.5f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(out / f"{tag}_{bi}.npz", Y=Y, Ds=Ds, B=B, bi=bi,
                        ok=ok_all, worst_res=worst_res, worst_t=worst_t,
                        seconds=time.time() - t0)
    print(f"{tag} B-row {bi} done in {time.time()-t0:.0f}s, ok={ok_all}, "
          f"worst residual {worst_res:.2e} at t={worst_t:.0f}", flush=True)


if __name__ == "__main__":
    main()
