# =====================================================================
# SI Section S4: Interpolation versus extrapolation
#
# This script: Tables S5 to S8, restricted to the 17 dynamics
# Produces   : tables/tab_extrap_*17.tex, tables/tab_grid17.tex, tables/tab_tfrac_*17.tex
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""S4 tables restricted to the 17 dynamics of the main text.

The stored arrays are (14 networks, 21 sources, 21 targets); nothing is
recomputed here.  We drop the four extra dynamics ANM, AEUT1, AVEG1 and
LMUT and re-aggregate.  Run with --check to reproduce the 21-dynamics
numbers instead, as a validation of the aggregation.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE / "data"
OUT = HERE / "tables"; OUT.mkdir(exist_ok=True)
CHECK = "--check" in sys.argv

DISP = {"MSIS": "MSIS1", "MSIR": "MSIR1", "EUT": "EUT1", "VEG": "VEG1"}
DROP = set() if CHECK else {"ANM", "AEUT1", "AVEG1", "LMUT"}
ADJ21 = {"SIS", "SIR", "MSIS", "MSIS2", "MSIR", "MSIR2", "DW", "MUT", "GEN",
         "WC", "gLV", "ANM", "AEUT1", "AVEG1"}
SPLITS = [("random", "intersp."), ("first", r"first-$41$"),
          ("last", r"last-$41$")]
PAN = [("nn", "minc", "NN, MinCorr"), ("nn", "qr", "NN, QR"),
       ("lin", "minc", "linear, MinCorr"), ("lin", "qr", "linear, QR")]

_z = np.load(D / "oos_random_oos_nn_minc.npz", allow_pickle=True)
DYN_ALL = [str(x) for x in _z["dynamics"]]
KEEP = [i for i, d in enumerate(DYN_ALL) if d not in DROP]
DYN = [DYN_ALL[i] for i in KEEP]
n = len(DYN)
off = ~np.eye(n, dtype=bool)
isadj = np.array([d in ADJ21 and d not in DROP for d in DYN])
adjblk = np.outer(isadj, isadj) & off
print(f"  {n} dynamics, {int(isadj.sum())} of them adjacency-coupled")


def sub(a):
    return np.asarray(a)[np.ix_(KEEP, KEEP)]


def netmean(path):
    return np.nanmean(np.load(D / path)["rel"], axis=0)


# ---------------- Tables S5/S6: within dynamics --------------------------
def within(dec):
    rows = []
    for i, d in enumerate(DYN):
        cells = []
        for sent in ("minc", "qr"):
            v, r = [], []
            for sp, _ in SPLITS:
                M = sub(netmean(f"oos_{sp}_oos_{dec}_{sent}.npz"))
                B = sub(netmean(f"oosbase_{sp}_{sent}.npz"))
                v.append(M[i, i] * 100); r.append(M[i, i] / B[i, i])
            cells += [f"{x:.1f}" for x in v] + [f"{x:.2f}" for x in r]
        rows.append(f"{DISP.get(d, d)} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows) + "\n"


for dec, fn in (("nn", "tab_extrap_nn17.tex"), ("lin", "tab_extrap_lin17.tex")):
    (OUT / fn).write_text(within(dec))
print("  wrote the two within-dynamics table bodies")


# ---------------- Table S7: the 3 x 3 source/target grid -----------------
def stats(M, B):
    R, V = M / B, M * 100
    o, ro, a, ra = V[off], R[off], V[adjblk], R[adjblk]
    return (np.median(o), o.mean(), np.median(ro), (ro < 1).mean() * 100,
            np.median(a), a.mean(), np.median(ra), (ra < 1).mean() * 100)


def grid_rows():
    out = []
    for k, (dec, sent, lab) in enumerate(PAN):
        body = []
        for si, (sp, sl) in enumerate(SPLITS):
            for ti, (tp, tl) in enumerate(SPLITS):
                M = sub(netmean(f"grid9_{sp}_{tp}_{dec}_{sent}.npz"))
                B = sub(netmean(f"bgrid_{sp}_{tp}_{sent}.npz"))
                s = stats(M, B)
                lead = (f"\\multirow{{9}}{{*}}{{{lab}}} & " if si == 0 and ti == 0
                        else " & ")
                body.append(lead + f"{sl}/{tl} & "
                            + f"{s[0]:.1f} & {s[1]:.1f} & {s[2]:.2f} & {s[3]:.0f} & "
                            + f"{s[4]:.1f} & {s[5]:.1f} & {s[6]:.2f} & {s[7]:.0f}"
                            + r" \\")
        out.append("\n".join(body))
    return "\n\\addlinespace\n".join(out) + "\n"


(OUT / "tab_grid17.tex").write_text(grid_rows())
print("  wrote the grid table body")


# ---------------- Table S8: the target calibration fraction --------------
FR = [50, 60, 70, 80, 90]


def tfrac_rows(panels):
    out = []
    for dec, sent, lab in panels:
        body = []
        for si, (sp, sl) in enumerate(SPLITS):
            for p in FR:
                M = sub(netmean(f"tfrac_{sp}_{p}_{dec}_{sent}.npz"))
                B = sub(netmean(f"tfracbase_{sp}_{p}_{sent}.npz"))
                s = stats(M, B)
                lead = (f"\\multirow{{17}}{{*}}{{{lab}}} & " if si == 0 and p == FR[0]
                        else " & ")
                body.append(lead + f"{sl}/${p}\\%$ & "
                            + f"{s[0]:.1f} & {s[1]:.1f} & {s[2]:.2f} & {s[3]:.0f} & "
                            + f"{s[4]:.1f} & {s[5]:.1f} & {s[6]:.2f} & {s[7]:.0f}"
                            + r" \\")
            if si < 2:
                body.append(r"\addlinespace[3pt]")
        out.append("\n".join(body))
    return "\n\\addlinespace[8pt]\n".join(out) + "\n"


(OUT / "tab_tfrac_nn17.tex").write_text(tfrac_rows(PAN[:2]))
(OUT / "tab_tfrac_lin17.tex").write_text(tfrac_rows(PAN[2:]))
print("  wrote the two target-fraction table bodies")


# ---------------- the numbers quoted in the S4 text ----------------------
print("\n  ---- text numbers ----")
agg = {}
for dec, sent, lab in PAN:
    for sp, sl in SPLITS:
        M = sub(netmean(f"oos_{sp}_oos_{dec}_{sent}.npz"))
        B = sub(netmean(f"oosbase_{sp}_{sent}.npz"))
        d = np.arange(n)
        agg[(dec, sent, sp)] = (M[d, d] * 100, M[d, d] / B[d, d])
for sp, sl in SPLITS:
    v = [agg[(d, s, sp)][0].mean() for d, s, _ in PAN]
    nb = [int((agg[(d, s, sp)][1] < 1).sum()) for d, s, _ in PAN]
    print(f"    {sl:12s} mean n-RMSE {min(v):.1f}-{max(v):.1f}%   "
          f"dynamics with r<1: {min(nb)}-{max(nb)} of {n}")
print("\n    dynamics ever reaching r >= 1:")
for i, dd in enumerate(DYN):
    bad = [f"{s}/{d}-{se}" for d, se, _ in PAN for s, _ in SPLITS
           if agg[(d, se, s)][1][i] >= 1]
    if bad:
        print(f"      {DISP.get(dd, dd):6s} in {len(bad)} of 12 combinations: "
              + ", ".join(sorted({b.split('/')[0] for b in bad})))
print("\n  ---- grid table text numbers (adj->adj) ----")
for ti, (tp, tl) in enumerate(SPLITS):
    med, rr, fr = [], [], []
    for dec, sent, _ in PAN:
        M = sub(netmean(f"grid9_random_{tp}_{dec}_{sent}.npz"))
        B = sub(netmean(f"bgrid_random_{tp}_{sent}.npz"))
        s = stats(M, B); med.append(s[4]); rr.append(s[6]); fr.append(s[7])
    print(f"    target {tl:12s} median n-RMSE {min(med):.1f}-{max(med):.1f}%  "
          f"median r {min(rr):.2f}-{max(rr):.2f}  r<1 {min(fr):.0f}-{max(fr):.0f}%")
print("\n  ---- all pairs, fraction with r<1 ----")
for ti, (tp, tl) in enumerate(SPLITS):
    fr = []
    for dec, sent, _ in PAN:
        M = sub(netmean(f"grid9_random_{tp}_{dec}_{sent}.npz"))
        B = sub(netmean(f"bgrid_random_{tp}_{sent}.npz"))
        fr.append(stats(M, B)[3])
    print(f"    target {tl:12s} {min(fr):.0f}-{max(fr):.0f}%")
