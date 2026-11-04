# =====================================================================
# SI Section S6: Two-parameter families of dynamics
#
# This script: one row of the MSIS2P parameter sweep
# Feeds      : Figs. S7 and S8, via merge_rows.py
# Produces   : rows/<tag>_<i>.npz
# Runtime    : hours per row
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Compute ONE D_I row of the MSIS two-parameter grid, for one network.

The serial generator does 8 rows back to back.  Rows are independent
(the reverse-warm chain runs *within* a row, never across rows), so they
can be run as separate processes.  Usage:

    python3 sweep_row.py <tag> <beta_max> <row_index>

beta_max must be the value the serial generator already determined by the
paper's widening rule, so the grid is identical to what it would have
produced.  Each row is written to data/_rows/<tag>_<row>.npz and merged
by merge_rows.py.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import msis2p as m                                           # noqa: E402
from generate_msis2p import NB, ND, D_LO, D_HI               # noqa: E402

OUTR = HERE / "data" / "_rows"
OUTR.mkdir(parents=True, exist_ok=True)


def main():
    tag, beta_max, di = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    t0 = time.time()
    net = m.load_network(tag)
    N = net["N"]
    betas = np.linspace(0.0, beta_max, NB)
    D_I = np.linspace(D_LO, D_HI, ND)[di]

    Y = np.empty((NB, N))
    ok_all = True
    y_prev = m.uniform_ic(N)                 # reverse warm, as in the serial code
    for bi in range(NB - 1, -1, -1):
        y, ok = m.equilibrate(net, betas[bi], D_I, y_prev)
        ok_all = ok_all and ok
        y_prev = y.copy()
        Y[bi] = y[N:]
        print(f"  {tag} row {di} (D_I={D_I:.4f}) beta={betas[bi]:.5f}: "
              f"mean rho_I={Y[bi].mean():.5f}  ({time.time()-t0:.0f}s)",
              flush=True)
    np.savez_compressed(OUTR / f"{tag}_{di}.npz", Y=Y, betas=betas,
                        D_I=D_I, di=di, ok=ok_all,
                        seconds=time.time() - t0)
    print(f"{tag} row {di} done in {time.time()-t0:.0f}s, ok={ok_all}",
          flush=True)


if __name__ == "__main__":
    main()
