"""Train/test splits on an 8 x 8 two-parameter grid, and loaders for it.

The paper's one-parameter sweeps have M = 51 points and hold out 10 (about
20%).  A two-parameter family has M = 64 = 8 x 8; 20% of 64 is 12.8, so we
hold out 13 and train on 51 -- the same count as the paper's training set,
which makes the two directly comparable.

Row layout of both two-parameter CSVs is  row = p2_index * 8 + p1_index,
where p1 is the parameter the paper already swept (beta for MSIS, D for
MUT) and p2 is the new one (D_I for MSIS, B for MUT).

Three ways to choose the 51 training points:

  random  shuffle the 64 cells with seed 0, hold out the first 13.
          The two-dimensional analogue of the paper's own split.

  first   the LOW corner: the first 7 values of p1 crossed with the first
          7 values of p2 (49 cells), plus two more to reach 51.  The two
          are (p1,p2) = (7,0) and (0,7): among the 15 cells outside the
          7 x 7 block, those are the only two with p1+p2 = 7, the smallest
          value available, so they are the least extrapolatory additions.

  last    the HIGH corner, the mirror image: p1 >= 1 and p2 >= 1 (49 cells)
          plus the two cells outside it with the largest p1+p2, which are
          the same two, (0,7) and (7,0).

  So `first` trains low and is asked to extrapolate high; `last` trains
  high and is asked to extrapolate low; each holds out 13 cells.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

NP = 8                      # points per parameter axis
M2 = NP * NP                # 64 cells
N_TEST = 13                 # 64 - 51; round(0.2 * 64) = 13
SEED = 0

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib")); import paths  # noqa: E402
REVDATA = paths.EQ_REVISION
SRC2D = {
    "MSIS2P": (REVDATA, "msis2p"),
    # the settled dataset: every cell an equilibrium, max_i|dx_i/dt| < 1e-6
    "MUT2P":  (REVDATA, "mut2p"),
}
DYN2 = list(SRC2D)


def cell(p1, p2):
    """Row index of grid cell (p1, p2)."""
    return p2 * NP + p1


def rows_random():
    rng = np.random.default_rng(SEED)
    idx = np.arange(M2)
    rng.shuffle(idx)
    return np.sort(idx[N_TEST:]), np.sort(idx[:N_TEST])


def _corner(low: bool):
    if low:
        keep = [cell(p1, p2) for p2 in range(NP - 1) for p1 in range(NP - 1)]
    else:
        keep = [cell(p1, p2) for p2 in range(1, NP) for p1 in range(1, NP)]
    keep += [cell(NP - 1, 0), cell(0, NP - 1)]      # the two extra cells
    tr = np.sort(np.array(sorted(set(keep))))
    ev = np.sort(np.setdiff1d(np.arange(M2), tr))
    assert tr.size == 51 and ev.size == N_TEST, (tr.size, ev.size)
    return tr, ev


def rows_first():
    return _corner(low=True)


def rows_last():
    return _corner(low=False)


SPLITS = {"random": rows_random, "first": rows_first, "last": rows_last}


def load_eq2(dyn, tag):
    """(64 x N) equilibrium matrix for a two-parameter dynamics."""
    folder, stem = SRC2D[dyn]
    arr = np.loadtxt(folder / f"{tag}_{stem}_equilibrium_data.csv",
                     delimiter=",", skiprows=1)
    assert arr.shape[0] == M2, (dyn, tag, arr.shape)
    return arr[:, 2:]                     # drop the two parameter columns


def params2(dyn, tag):
    folder, stem = SRC2D[dyn]
    arr = np.loadtxt(folder / f"{tag}_{stem}_equilibrium_data.csv",
                     delimiter=",", skiprows=1)
    return arr[:, 0], arr[:, 1]
