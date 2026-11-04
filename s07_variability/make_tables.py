# =====================================================================
# SI Section S7: Variability over networks and random factors
#
# This script: Tables S9 and S10
# Produces   : tables/tab_std_nrmse.tex, tables/tab_std_ratio.tex
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Tables S9 and S10: the median per-cell standard deviation, by cell subset.

Each decoder x sentinel pair gets three rows:
  all            the 17x17 matrix
  node param.    S or T in {NM, EUT1, EUT2, VEG1, VEG2}, the dynamics whose
                 node parameters are drawn at random
  adjacency      S and T both adjacency coupled

Columns are the median over the cells of that subset of the per-cell standard
deviation, first with only the network varying and then with one further
source of randomness released.  A dash marks a combination that does not
apply.
"""
import numpy as np
from pathlib import Path

H = Path(__file__).resolve().parent / "data"
PAN = [("NN, MinCorr", "nn", "minc"), ("NN, QR", "nn", "qr"),
       ("NN, random", "nn", "random"), ("linear, MinCorr", "lin", "minc"),
       ("linear, QR", "lin", "qr"), ("linear, random", "lin", "random")]
MODES = ["nn", "split", "sent", "hetero"]
HET = {"NM", "EUT1", "EUT2", "VEG1", "VEG2"}
ADJ = {"SIS", "SIR", "MSIS1", "MSIS2", "MSIR1", "MSIR2", "DW", "MUT", "GEN",
       "WC", "gLV"}

base = np.load(H / "rep_base.npz", allow_pickle=True)
lab = [str(x) for x in base["dynamics"]]
n = len(lab)
D = {m: np.load(H / f"rep_{m}.npz", allow_pickle=True) for m in MODES}
N_HET = len(D["hetero"]["networks"])          # the ten networks it covers

hi = [i for i, l in enumerate(lab) if l in HET]
ai = [i for i, l in enumerate(lab) if l in ADJ]
M_ALL = np.ones((n, n), bool)
M_HET = np.zeros((n, n), bool); M_HET[hi, :] = True; M_HET[:, hi] = True
M_ADJ = np.zeros((n, n), bool); M_ADJ[np.ix_(ai, ai)] = True


def std_cells(d, dec, sent, what, nets=None):
    """Per-cell std over the pooled (R, n_net) samples."""
    M, B = d[f"M_{dec}_{sent}"], d[f"B_{dec}_{sent}"]
    X = M * 100.0 if what == "nrmse" else np.where(B > 0, M / B, np.nan)
    if nets is not None:
        X = X[:, nets]
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    return np.nanstd(X, axis=0, ddof=1)


def med(s, mask):
    v = s[mask]
    return np.nanmedian(v[np.isfinite(v)])


def fmt(v, what):
    return "--" if v is None else (f"{v:.2f}" if what == "nrmse"
                                   else f"{v:.3f}")


def body(what):
    out = []
    for k, (title, dec, sent) in enumerate(PAN):
        if k:
            out.append(r"\addlinespace")
        has = lambda m: f"M_{dec}_{sent}" in D[m].files          # noqa: E731
        sub10 = np.arange(N_HET)

        # --- row 1: all cells, everything but the node parameters
        s0 = std_cells(base, dec, sent, what)
        cells = [med(s0, M_ALL)]
        for m in ("nn", "split", "sent"):
            cells.append(med(std_cells(D[m], dec, sent, what), M_ALL)
                         if has(m) else None)
        cells.append(None)
        out.append(f"{title} & all & "
                   + " & ".join(fmt(c, what) for c in cells) + r" \\")

        # --- row 2: the cells a heterogeneous dynamics enters, on the same
        #            ten networks the node-parameter replicates cover
        s0h = std_cells(base, dec, sent, what, sub10)
        cells = [med(s0h, M_HET), None, None, None,
                 med(std_cells(D["hetero"], dec, sent, what), M_HET)]
        out.append(" & node param. & "
                   + " & ".join(fmt(c, what) for c in cells) + r" \\")

        # --- row 3: adjacency -> adjacency cells
        cells = [med(s0, M_ADJ)]
        for m in ("nn", "split", "sent"):
            cells.append(med(std_cells(D[m], dec, sent, what), M_ADJ)
                         if has(m) else None)
        cells.append(None)
        out.append(" & adjacency & "
                   + " & ".join(fmt(c, what) for c in cells) + r" \\")
    out[-1] = out[-1].rstrip()[:-2].rstrip()
    return "\n".join(out) + "\n"


for what, f in (("nrmse", "tab_std_nrmse.tex"), ("ratio", "tab_std_ratio.tex")):
    OUT = Path(__file__).resolve().parent / "tables"
    OUT.mkdir(exist_ok=True)
    (OUT / f).write_text(body(what))
    print(f"  wrote {f}")
print(f"  node-parameter replicates cover {N_HET} networks; "
      f"cells: all {M_ALL.sum()}, heterogeneous {M_HET.sum()}, "
      f"adjacency {M_ADJ.sum()}")
