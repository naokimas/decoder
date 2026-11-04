# =====================================================================
# SI Section S12: Why the transferable dynamics transfer
#
# This script: the AUC values quoted in the text of Section S12
# Produces   : (prints)
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""How cleanly does the correlation matrix separate the two families?

Section S12 quotes two AUC values, one per panel of Fig. S23.  Table S15 is
a different summary of the same matrices and is written by make_table.py.

The 11 x 11 block of the adjacency-coupled dynamics should hold high values
and the two rectangular cross blocks, 11 x 6 and 6 x 11, low ones.  AUC is
the probability that a value drawn at random from the first set exceeds one
drawn at random from the second, computed here as the rank-sum statistic
divided by the number of comparisons.  AUC = 1 means every within-family
cell beats every cross-family cell.  We report the AUC only; the SI makes
no claim of statistical significance, so no test is carried out.

The diagonal is excluded because it is trivially 1, and the 6 x 6 block of
the Laplacian-coupled dynamics is excluded because the question is whether
the transferable family stands apart from the rest, not whether the
Laplacian-coupled dynamics resemble one another.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
d = np.load(HERE / "data" / "corr.npz", allow_pickle=True)
LAB = [str(x) for x in d["labels"]]
ADJ, LAPL = LAB[:11], LAB[13:]          # the two degree vectors sit between
assert len(ADJ) == 11 and len(LAPL) == 6, (ADJ, LAPL)


def auc(R):
    ia = [LAB.index(l) for l in ADJ]
    il = [LAB.index(l) for l in LAPL]
    within = R[np.ix_(ia, ia)][~np.eye(11, dtype=bool)]    # 110 values
    cross = np.concatenate([R[np.ix_(ia, il)].ravel(),     # 11 x 6
                            R[np.ix_(il, ia)].ravel()])    # 6 x 11
    within = within[np.isfinite(within)]
    cross = cross[np.isfinite(cross)]
    # AUC as the normalized rank-sum of the within-family values
    r = rankdata(np.concatenate([within, cross]))[:within.size].sum()
    u = r - within.size * (within.size + 1) / 2.0
    return u / (within.size * cross.size)


for key, panel in (("mean_max", "Fig. S23(a)"), ("mean_med", "Fig. S23(b)")):
    print(f"  {panel}: AUC = {auc(d[key]):.3f}")
