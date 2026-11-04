# =====================================================================
# SI Section S14: Robustness to NN bias terms and ridge strength
#
# This script: Tables S20 and S21
# Produces   : tables/tab_s14_*.tex
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""LaTeX bodies for the two decoder-specification tables of SI Section S14.

15 candidate sources -- the 11 transferable per-dynamics decoders, C11 and
C11L1, and the two budget-matched pooled decoders C11B and C11L1B -- giving
the mean n-RMSE and the mean r(S->T) over the 11 transferable targets, with
the main-text specification beside the modified one.

Each table of the SI stacks two tabulars, the NN one above the linear one,
so each is filled from two of the four bodies written here.

The per-dynamics and pooled rows come from rank*.npz; the two budget-matched
rows come from budget*.npz, where the pooled training set is subsampled to
the same 41 control-parameter values that a single-source decoder sees.
Run after s13's compute_rank.py and compute_budget.py.  Takes a second.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
# the rank and budget sweeps are Section S13's, shared with this section
DATA = HERE.parent / "s13_decoder_rankings" / "data"
VARIANTS = (("main", ""), ("bias", "_bias"), ("alpha01", "_alpha01"))
D = {k: np.load(DATA / f"rank{s}.npz", allow_pickle=True)
     for k, s in VARIANTS}
BUD = {k: np.load(DATA / f"budget{s}.npz", allow_pickle=True)
       for k, s in VARIANTS}
DYN = [str(x) for x in D["main"]["dynamics"]]
TR = [str(x) for x in D["main"]["transferable"]]
TI = [DYN.index(t) for t in TR]
ROWS = TR + ["C11", "C11B", "C11L1", "C11L1B"]
# the two budget-matched rows and their keys in budget*.npz
BROWS = {"C11B": "c11b", "C11L1B": "c11l1b"}


def vals(d, dec, sent, idx):
    M = d[f"M_{dec}_{sent}_single"]; B = d[f"B_{dec}_{sent}_single"]
    out = {}
    for s in TR:
        i = DYN.index(s)
        out[s] = (M[i, TI].mean(), (M[i, TI] / B[i, TI]).mean())[idx]
    m = d[f"M_{dec}_{sent}_c11"]; b = d[f"B_{dec}_{sent}_c11"]
    out["C11"] = (m[TI].mean(), (m[TI] / b[TI]).mean())[idx]
    m = d[f"M_{dec}_{sent}_c11l1"]; b = d[f"B_{dec}_{sent}_c11l1"]
    out["C11L1"] = (m.mean(), (m / b).mean())[idx]
    return out


def bvals(d, dec, sent, idx, key):
    """The budget-matched pooled decoder, averaged over the 11 targets."""
    M = d[f"M_{dec}_{sent}_{key}"][TI]
    B = d[f"B_{dec}_{sent}_{key}"][TI]
    return (M.mean(), (M / B).mean())[idx]


def body(dec, variant, idx):
    sents = ("minc", "qr", "random")
    a = {s: vals(D["main"], dec, s, idx) for s in sents}
    b = {s: vals(D[variant], dec, s, idx) for s in sents}
    for s in sents:
        for row, key in BROWS.items():
            a[s][row] = bvals(BUD["main"], dec, s, idx, key)
            b[s][row] = bvals(BUD[variant], dec, s, idx, key)
    sc = 100 if idx == 0 else 1
    lines = []
    for r in ROWS:
        cells = []
        for s in sents:
            cells.append(f"{a[s][r]*sc:.2f}")
            cells.append(f"{b[s][r]*sc:.2f}")
        lines.append(f"{r} & " + " & ".join(cells))
    return " \\\\\n".join(lines)


if __name__ == "__main__":
    for dec, variant, idx, fn in (
            ("nn", "bias", 0, "tab_s14_nn_rmse.tex"),
            ("lin", "alpha01", 0, "tab_s14_lin_rmse.tex"),
            ("nn", "bias", 1, "tab_s14_nn_ratio.tex"),
            ("lin", "alpha01", 1, "tab_s14_lin_ratio.tex")):
        (HERE / "tables" / fn).write_text(body(dec, variant, idx) + "\n")
        print(f"  wrote {fn}")
