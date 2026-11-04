# =====================================================================
# SI Section S2: Effective rank of the equilibrium data
#
# This script: Tables S1 and S2, restricted to the 17 dynamics
# Produces   : tables/tab_rank17*.tex
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Re-aggregate the effective ranks over the 17 dynamics of the main text.

The per-(dynamics, network) values in data/effective_rank.csv are unchanged;
only the set of dynamics summarized changes, from 21 to 17.  The four extra
dynamics of the earlier version (ANM, AEUT1, AVEG1, LMUT) are dropped.
"""
import csv
import statistics as st
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROWS = list(csv.DictReader((HERE / "data" / "effective_rank.csv").open()))
DYN17 = ["SIS", "SIR", "MSIS1", "MSIS2", "MSIR1", "MSIR2", "DW", "MUT",
         "GEN", "WC", "gLV", "NM", "EUT1", "EUT2", "VEG1", "VEG2", "LDW"]
by = {}
for r in ROWS:
    if r["dynamics"] in DYN17:
        by.setdefault(r["dynamics"], []).append(r)
assert all(len(v) == 14 for v in by.values()) and len(by) == 17

g = lambda v, k: [float(x[k]) for x in v]
(HERE / "tables").mkdir(exist_ok=True)


def body(cols, decs):
    out = []
    for d in DYN17:
        v = by[d]; cells = []
        for k, dec in zip(cols, decs):
            x = g(v, k)
            f = f"{{:.{dec}f}}"
            cells.append(f.format(st.median(x)))
            cells.append(f.format(min(x)) + "--" + f.format(max(x)))
        out.append(f"{d} & " + " & ".join(cells) + r" \\")
    return "\n".join(out) + "\n"


(HERE / "tables" / "tab_rank17.tex").write_text(
    body(["r99", "r999", "r_entropy", "r_stable"], [0, 0, 1, 2]))
(HERE / "tables" / "tab_rank17_centered.tex").write_text(
    body(["r99_c", "r999_c", "r_entropy_c", "r_stable_c"], [0, 0, 1, 2]))
print("  wrote tables/tab_rank17.tex and tables/tab_rank17_centered.tex")

# ---- the numbers quoted in the text --------------------------------------
flat = {k: np.array([float(r[k]) for d in DYN17 for r in by[d]])
        for k in ("r99", "r999", "r_entropy", "r_stable",
                  "r99_c", "r999_c", "r_entropy_c", "r_stable_c")}
n = len(flat["r99"])
print(f"\n  {n} dynamics-network pairs")
for k in ("r99", "r999"):
    q = np.percentile(flat[k], [25, 50, 75])
    print(f"    {k:10s} median {q[1]:.0f}, IQR {q[0]:.0f}-{q[2]:.0f}, "
          f"max {flat[k].max():.0f}")
print(f"    fraction with r99 <= 3: {100*(flat['r99']<=3).mean():.0f}%")
med = {k: np.array([st.median(g(by[d], k)) for d in DYN17]) for k in flat}
print("\n  Spearman between per-dynamics medians (raw):")
for a, b in (("r99", "r_entropy"), ("r99", "r_stable"),
             ("r999", "r_entropy"), ("r999", "r_stable")):
    print(f"    {a} vs {b}: {spearmanr(med[a], med[b]).correlation:.2f}")
print("\n  centered, over all pairs:")
for k in ("r99_c", "r999_c", "r_entropy_c"):
    print(f"    {k:12s} median {np.median(flat[k]):.1f}, "
          f"range {flat[k].min():.1f}-{flat[k].max():.1f}")
print("\n  Spearman between raw and centered per-dynamics medians:")
for k in ("r99", "r999", "r_entropy", "r_stable"):
    print(f"    {k:10s} {spearmanr(med[k], med[k+'_c']).correlation:.2f}")


# ---------------------------------------------------------------------------
# The two groups of dynamics discussed in the text of Section S2.
#
# Section S2 groups the dynamics, phenomenologically, by the shape of the
# transition that the control-parameter sweep passes through, and quotes the
# median of each effective rank within each group.  Those medians are what
# this block reproduces.
#
# No significance test is computed, as the SI reports none.
# ---------------------------------------------------------------------------
SMOOTH = ["SIS", "SIR", "MSIS1", "MSIR1", "MUT", "GEN", "WC", "gLV",
          "EUT2", "VEG2"]                       # single smooth global shift
CASCADE = ["DW", "MSIS2", "MSIR2", "NM", "EUT1", "VEG1", "LDW"]
assert sorted(SMOOTH + CASCADE) == sorted(DYN17)

print("\n  median effective rank within each group of dynamics")
print(f"    {'group':28s} {'r_99%':>7s} {'r_99.9%':>9s} "
      f"{'r_ent':>7s} {'r_stab':>8s}")
for name, grp in (("single smooth global shift", SMOOTH),
                  ("cascade or bistable switch", CASCADE)):
    v = [r for d in grp for r in by[d]]
    med = [st.median([float(x[k]) for x in v])
           for k in ("r99", "r999", "r_entropy", "r_stable")]
    print(f"    {name:28s} {med[0]:7.0f} {med[1]:9.0f} "
          f"{med[2]:7.1f} {med[3]:8.2f}   ({len(v)} pairs)")
