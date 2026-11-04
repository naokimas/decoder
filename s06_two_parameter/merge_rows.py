# =====================================================================
# SI Section S6: Two-parameter families of dynamics
#
# This script: assembles the MSIS2P rows into one CSV
# Feeds      : Figs. S7 and S8, via compute_2d45.py
# Produces   : <tag>_msis2p_equilibrium_data.csv in equilibria_revision/ (input data)
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Merge per-row npz shards into the same CSV + summary the serial path writes."""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib")); import paths  # noqa: E402
EQ = paths.EQ_REVISION        # equilibrium CSVs; see ../README.md
sys.path.insert(0, str(HERE))
import msis2p as m                                           # noqa: E402
from generate_msis2p import NB, ND                           # noqa: E402

OUT = EQ
ROWS = HERE / "data" / "_rows"     # where sweep_row.py writes the row shards


def main():
    tag = sys.argv[1]
    rows = [np.load(ROWS / f"{tag}_{d}.npz") for d in range(ND)]
    net = m.load_network(tag)
    N = net["N"]
    betas = rows[0]["betas"]
    D_Is = np.array([float(r["D_I"]) for r in rows])
    Y = np.empty((NB * ND, N))
    P = np.empty((NB * ND, 2))
    for di, r in enumerate(rows):
        for bi in range(NB):
            Y[di * NB + bi] = r["Y"][bi]
            P[di * NB + bi] = (betas[bi], D_Is[di])
    hdr = "beta,D_I," + ",".join(f"x_{i+1}" for i in range(N))
    np.savetxt(OUT / f"{tag}_msis2p_equilibrium_data.csv",
               np.column_stack([P, Y]), delimiter=",", header=hdr,
               comments="", fmt=["%.8g", "%.8g"] + ["%.8g"] * N)
    rec = dict(tag=tag, N=N, alpha_max=net["alpha_max"], kbar=net["kbar"],
               beta_max=float(betas[-1]), n_doublings=-1,
               betas=betas.tolist(), D_Is=D_Is.tolist(),
               ymin=float(Y.min()), ymax=float(Y.max()),
               all_ok=bool(all(bool(r["ok"]) for r in rows)),
               seconds=float(sum(float(r["seconds"]) for r in rows)),
               note="row-parallel; seconds = summed CPU over rows")
    (OUT / "_summary").mkdir(exist_ok=True)
    (OUT / "_summary" / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    print(f"{tag}: merged {ND} rows, all_ok={rec['all_ok']}, "
          f"rho_I in [{Y.min():.4g}, {Y.max():.4g}]")


if __name__ == "__main__":
    main()
