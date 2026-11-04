# =====================================================================
# SI Section S2: Effective rank of the equilibrium data
#
# This script: singular spectra and four effective ranks per (dynamics, network)
# Feeds      : Tables S1 and S2, and the effective ranks quoted in Section S4
# Produces   : data/effective_rank.csv
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Singular-value spectra and effective ranks of the equilibrium matrices.

For every (dynamics, network) pair the paper's sweep gives a 51 x N matrix Y:
one row per control-parameter value, one column per node.  Since 51 < N for
every network (the smallest, dolphin, has N = 62), there are exactly 51
singular values regardless of N, so the spectra are directly comparable
across networks.

The question is whether Y has low EFFECTIVE rank: the exact rank is almost
always 51, because rounding and truncation lift every singular value off
zero, but if the spectrum decays fast then only a few directions carry the
data.  We report four standard summaries, all computed from the singular
values sigma_1 >= ... >= sigma_51:

  r99   smallest r with sum_{i<=r} sigma_i^2 >= 0.99 * sum_i sigma_i^2.
        "How many directions hold 99% of the energy."  The most direct
        reading of low-dimensionality, and the one we quote.
  r90,  the same at 90% and at 99.9%.  r999 is the more discriminating of
  r999  the three here, because sigma_1 alone often already clears 99%.
  rent  the entropy effective rank of Roy & Vetterli (2007):
        exp(-sum_i p_i log p_i) with p_i = sigma_i / sum_j sigma_j.
        A continuous measure; it does not need a threshold.
  rstab the stable rank, ||Y||_F^2 / ||Y||_2^2 = sum_i sigma_i^2 / sigma_1^2.
        Dominated by the leading direction, so it is the most conservative.

Both the raw matrix and the column-centered one are analyzed.  Centering
matters: the equilibria are all positive and share a large common level, so
the raw spectrum has a huge sigma_1 that by itself makes any data set look
low rank.  Centering removes that offset and asks how many directions the
VARIATION across the sweep needs.  The raw numbers are the ones relevant to
the decoder, which reconstructs x itself and not its deviation from a mean;
the centered ones are the honest check that the low rank is not an artifact
of the common level.

Writes data/effective_rank.csv, one row per (dynamics, network) pair.
Usage:  python3 compute_rank.py
"""

# Orientation of the data matrix (code review, S02).  SI Section S2.1, like
# the main text, writes the data matrix as X in R^{N x M}, one row per node.
# Y here is its transpose, M x N, one row per control-parameter value, as in
# the equilibrium CSVs (see code/README.md).  X and X^T have the same
# singular values, so the spectra and all four effective ranks are the same
# whichever orientation is used.
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "lib"          # shared modules
# Reuse the loader that already knows where all 21 equilibrium sets live.
_spec = importlib.util.spec_from_file_location(
    "leq", LIB / "load_equilibria.py")
leq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(leq)

DYN, NETS = leq.DYNAMICS, leq.NETWORKS
DISPLAY = leq.DISPLAY
ADJ = set(leq.ADJ_21)


def ranks(sv):
    """Effective-rank summaries from a descending singular-value vector."""
    s2 = sv ** 2
    tot = s2.sum()
    cum = np.cumsum(s2) / tot
    r99 = int(np.searchsorted(cum, 0.99) + 1)
    r90 = int(np.searchsorted(cum, 0.90) + 1)
    r999 = int(np.searchsorted(cum, 0.999) + 1)
    p = sv / sv.sum()
    p = p[p > 0]
    rent = float(np.exp(-(p * np.log(p)).sum()))
    rstab = float(tot / sv[0] ** 2)
    return r99, r90, r999, rent, rstab


def main():
    rows = ["dynamics,class,network,N,sigma1,sigma51_over_sigma1,"
            "r99,r90,r999,r_entropy,r_stable,"
            "r99_c,r90_c,r999_c,r_entropy_c,r_stable_c"]
    for d in DYN:
        for t in NETS:
            Y = leq.load_eq(t, d)
            assert Y.shape[0] == 51, (d, t, Y.shape)
            sv = np.linalg.svd(Y, compute_uv=False)
            Yc = Y - Y.mean(axis=0, keepdims=True)
            svc = np.linalg.svd(Yc, compute_uv=False)
            r = ranks(sv)
            rc = ranks(svc) if svc[0] > 0 else (0, 0, 0, 0.0, 0.0)
            rows.append(
                f"{DISPLAY.get(d, d)},{'adj' if d in ADJ else 'dif'},{t},"
                f"{Y.shape[1]},{sv[0]:.6g},{sv[-1]/sv[0]:.6g},"
                f"{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]:.4f},"
                f"{rc[0]},{rc[1]},{rc[2]},{rc[3]:.4f},{rc[4]:.4f}")
        print(f"  {DISPLAY.get(d, d):>6s}: 14 networks done", flush=True)
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "effective_rank.csv").write_text("\n".join(rows) + "\n")
    print(f"wrote data/effective_rank.csv ({len(rows)-1} rows)")


if __name__ == "__main__":
    main()
