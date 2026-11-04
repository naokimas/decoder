# =====================================================================
# SI Section S6: Two-parameter families of dynamics
#
# This script: assembles the MUT2P rows into one CSV
# Feeds      : Figs. S7 and S8, via compute_2d45.py
# Produces   : <tag>_mut2p_equilibrium_data.csv in equilibria_revision/ (input data)
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Merge per-B-row npz shards into the CSV + summary the serial path writes."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib")); import paths  # noqa: E402
EQ = paths.EQ_REVISION        # equilibrium CSVs; see ../README.md
sys.path.insert(0, str(HERE))
import mut2p as m                                            # noqa: E402
from generate_mut2p import ND, NBB, HI_BRANCH                # noqa: E402


def main():
    tag = sys.argv[1]
    root = EQ                       # where compute_2d45.py reads it
    # the row shards are where sweep_row_mut.py writes them
    shards = HERE / "data" / "_rows_mut2p"
    rows = [np.load(shards / f"{tag}_{b}.npz") for b in range(NBB)]
    net = m.load_network(tag)
    N = net["N"]
    Ds = rows[0]["Ds"]
    Bs = np.array([float(r["B"]) for r in rows])
    Y = np.empty((ND * NBB, N))
    P = np.empty((ND * NBB, 2))
    for bi, r in enumerate(rows):
        for di in range(ND):
            Y[bi * ND + di] = r["Y"][di]
            P[bi * ND + di] = (Ds[di], Bs[bi])
    hdr = "D,B," + ",".join(f"x_{i+1}" for i in range(N))
    root.mkdir(parents=True, exist_ok=True)
    np.savetxt(root / f"{tag}_mut2p_equilibrium_data.csv",
               np.column_stack([P, Y]), delimiter=",", header=hdr,
               comments="", fmt=["%.8g", "%.8g"] + ["%.8g"] * N)
    rec = dict(tag=tag, N=N, kbar=net["kbar"],
               D_max=float(m.MUT_CONSTANT / net["kbar"]),
               Ds=Ds.tolist(), Bs=Bs.tolist(),
               ymin=float(Y.min()), ymax=float(Y.max()),
               n_upper=int((Y.mean(axis=1) > HI_BRANCH).sum()),
               all_ok=bool(all(bool(r["ok"]) for r in rows)),
               worst_residual=float(max(float(r["worst_res"]) for r in rows)),
               worst_t=float(max(float(r["worst_t"]) for r in rows)),
               seconds=float(sum(float(r["seconds"]) for r in rows)),
               note="row-parallel; seconds = summed CPU over B rows")
    (root / "_summary").mkdir(parents=True, exist_ok=True)
    (root / "_summary" / f"{tag}_mut2p.json").write_text(json.dumps(rec, indent=2))
    print(f"{tag}: merged {NBB} B rows, all_ok={rec['all_ok']}, "
          f"upper {rec['n_upper']}/64, worst residual "
          f"{rec['worst_residual']:.2e}")


if __name__ == "__main__":
    main()
