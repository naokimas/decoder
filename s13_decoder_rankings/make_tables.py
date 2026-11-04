# =====================================================================
# SI Section S13: Source-decoder rankings
#
# This script: Tables S16 to S19
# Produces   : tables/tab_rank*.tex
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""LaTeX bodies for the source-decoder ranking tables of SI Section S13.

Four tables: the 11-transferable-target comparison (14 candidate sources)
and the all-17-target comparison (19 candidate sources), each ranked by mean
n-RMSE and by mean r(S->T).  r is formed from the network-averaged n-RMSE
and the network-averaged mean-prediction baseline, then averaged over the
target set, as everywhere else in the revision.

The script also prints the numbers that the text of Section S13 quotes for
the budget-matched pooled decoders of compute_budget.py (data/budget.npz):
their mean n-RMSE ranges over the six panels next to those of the full
pools, and the rank each would take if it were added to the tables.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
d = np.load(HERE / "data" / "rank.npz", allow_pickle=True)
bud = np.load(HERE / "data" / "budget.npz", allow_pickle=True)
DYN = [str(x) for x in d["dynamics"]]
TR = [str(x) for x in d["transferable"]]
TI = [DYN.index(t) for t in TR]
PANELS = [("nn", "minc"), ("nn", "qr"), ("nn", "random"),
          ("lin", "minc"), ("lin", "qr"), ("lin", "random")]


def scores(panel, scope):
    """{source: (mean n-RMSE, mean r)} for one panel and one target scope."""
    dec, sent = panel
    M = d[f"M_{dec}_{sent}_single"]; B = d[f"B_{dec}_{sent}_single"]
    sel = TI if scope == "t11" else list(range(len(DYN)))
    names = TR if scope == "t11" else DYN
    out = {}
    for s in names:
        i = DYN.index(s)
        out[s] = (M[i, sel].mean(), (M[i, sel] / B[i, sel]).mean())
    if scope == "t11":
        for key, lbl in (("c17", "C17"), ("c11", "C11")):
            m = d[f"M_{dec}_{sent}_{key}"]; b = d[f"B_{dec}_{sent}_{key}"]
            out[lbl] = (m[sel].mean(), (m[sel] / b[sel]).mean())
        m = d[f"M_{dec}_{sent}_c11l1"]; b = d[f"B_{dec}_{sent}_c11l1"]
        out["C11L1"] = (m.mean(), (m / b).mean())
    else:
        for key, lbl in (("c17", "C17"), ("c17l1", "C17L1")):
            m = d[f"M_{dec}_{sent}_{key}"]; b = d[f"B_{dec}_{sent}_{key}"]
            out[lbl] = (m.mean(), (m / b).mean())
    return out


def body(scope, measure):
    idx = 0 if measure == "rmse" else 1
    cols = [sorted(scores(p, scope).items(), key=lambda kv: kv[1][idx])
            for p in PANELS]
    n = len(cols[0])
    lines = []
    for r in range(n):
        cells = []
        for c in cols:
            name, v = c[r]
            val = v[idx] * 100 if idx == 0 else v[idx]
            cells.append(f"{name} & {val:.2f}")
        lines.append(f"{r+1} & " + " & ".join(cells))
    return " \\\\\n".join(lines)


def budget_summary():
    """Print the budget-matched values quoted in the text of Section S13.

    The two budget-matched decoders of a target scope (C11B and C11L1B for
    the 11 targets, C17B and C17L1B for all 17) are ranked as if both were
    added to the corresponding table together.
    """
    for scope, pair in (("t11", (("C11B", "c11b", "C11"), ("C11L1B", "c11l1b", "C11L1"))),
                        ("all", (("C17B", "c17b", "C17"), ("C17L1B", "c17l1b", "C17L1")))):
        res = {lbl: {"v": [], "full": [], "rank_m": [], "rank_r": []}
               for lbl, _, _ in pair}
        for dec, sent in PANELS:
            sc = scores((dec, sent), scope)
            for lbl, key, _ in pair:
                m = bud[f"M_{dec}_{sent}_{key}"]; b = bud[f"B_{dec}_{sent}_{key}"]
                if key == "c11b":             # stored for all 17 targets
                    m, b = m[TI], b[TI]
                sc[lbl] = (m.mean(), (m / b).mean())
            for lbl, _, full in pair:
                v = sc[lbl]
                res[lbl]["v"].append(v[0]); res[lbl]["full"].append(sc[full][0])
                res[lbl]["rank_m"].append(1 + sum(int(x[0] < v[0]) for x in sc.values()))
                res[lbl]["rank_r"].append(1 + sum(int(x[1] < v[1]) for x in sc.values()))
        for lbl, _, full in pair:
            r = res[lbl]
            print(f"  {lbl:6s} n-RMSE {100*min(r['v']):.2f}% to {100*max(r['v']):.2f}% "
                  f"({full}: {100*min(r['full']):.2f}% to {100*max(r['full']):.2f}%); "
                  f"rank by n-RMSE {r['rank_m']}, by r {r['rank_r']}")


if __name__ == "__main__":
    for scope, measure, fn in (("t11", "rmse", "tab_rank_rmse.tex"),
                               ("t11", "ratio", "tab_rank_ratio.tex"),
                               ("all", "rmse", "tab_rank17_rmse.tex"),
                               ("all", "ratio", "tab_rank17_ratio.tex")):
        (HERE / "tables" / fn).write_text(body(scope, measure) + "\n")
        print(f"  wrote {fn}")
    budget_summary()
