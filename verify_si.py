"""Re-derive numbers that the SI reports from the stored intermediate results.

This does not re-run any experiment.  It reads the .npz and .csv files that
the compute scripts left behind, recomputes a sample of the quantities that
the SI reports, and compares them against the SI.  Some of these are printed
in the text or tables (Sections S1, S2, S3, S15 and the effective-rank fits
of Section S9); others are values plotted in a figure (Fig. S11 for Section
S8, Fig. S19 for the c-sweep of Section S9), checked against the value at
the plotted point.  It needs no equilibrium data and finishes in seconds.

Usage:  python3 verify_si.py
Exit status is 0 if every check passes and 1 otherwise.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FAIL = []


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    if not ok:
        FAIL.append(label)
    print(f"    {label:58s} {got:10.3f} vs {want:8.3f}  "
          f"{'ok' if ok else 'MISMATCH'}")


def s01():
    print("  S1  node-wise reconstruction error")
    d = np.load(HERE / "s01_nodewise_error/data/nodewise.npz", allow_pickle=True)
    from scipy.stats import spearmanr
    cases = ["dolphin_SIS_SIS", "dolphin_SIS_WC",
             "email_MSIR1_MSIR1", "email_MSIR1_GEN"]
    for c, agg, rho in zip(cases, [0.43, 5.54, 0.20, 5.16],
                           [-0.35, 0.15, 0.66, 0.14]):
        check(f"{c}: aggregate n-RMSE (%)",
              float(d[f"{c}__agg"]) * 100, agg, 0.006)
        r = spearmanr(d[f"{c}__degree"], d[f"{c}__nodewise"]).correlation
        check(f"{c}: Spearman(degree, node-wise n-RMSE)", r, rho, 0.006)


def s02():
    print("  S2  effective rank")
    import csv
    rows = list(csv.DictReader(
        (HERE / "s02_effective_rank/data/effective_rank.csv").open()))
    D17 = {"SIS", "SIR", "MSIS1", "MSIS2", "MSIR1", "MSIR2", "DW", "MUT",
           "GEN", "WC", "gLV", "NM", "EUT1", "EUT2", "VEG1", "VEG2", "LDW"}
    r99 = np.array([float(r["r99"]) for r in rows if r["dynamics"] in D17])
    r999 = np.array([float(r["r999"]) for r in rows if r["dynamics"] in D17])
    check("number of dynamics-network pairs", len(r99), 238, 0)
    check("median r_99%", np.median(r99), 2, 0)
    check("max r_99%", r99.max(), 7, 0)
    check("fraction with r_99% <= 3 (%)", (r99 <= 3).mean() * 100, 89, 0.6)
    check("median r_99.9%", np.median(r999), 2.5, 0)
    check("max r_99.9%", r999.max(), 30, 0)


def s03():
    print("  S3  gappy POD, Table S3")
    d = np.load(HERE / "s03_gappy_pod/data/gappy_matrices.npz", allow_pickle=True)
    eye = np.eye(17, dtype=bool)
    SI = {("minmax", "minc", "rn"): (2.84, 231.96),
          ("minmax", "qr", "rn"): (0.10, 12.10),
          ("minmax", "random", "rn"): (2.32, 989.80),
          ("center", "minc", "e999"): (0.45, 40.43),
          ("center", "qr", "e999"): (0.49, 19.59),
          ("center", "random", "e999"): (0.74, 116.21)}
    for (c, s, r), (sd, so) in SI.items():
        M = d[f"G_{c}_{s}_{r}"] * 100
        check(f"{c}/{s}/{r}: median diagonal", np.median(M[eye]), sd, 0.006)
        check(f"{c}/{s}/{r}: median off-diagonal",
              np.median(M[~eye]), so, max(0.006, 0.0006 * so))


def s08():
    print("  S8  measurement noise, values plotted in Fig. S11 at epsilon = 1")
    d = np.load(HERE / "s08_measurement_noise/data/noise.npz", allow_pickle=True)
    eps = d["eps"]; i1 = int(np.argmin(np.abs(eps - 1.0)))
    for c, v in zip([str(x) for x in d["cases"]],
                    [60.1, 28.3, 0.85, 13.0]):
        check(f"{c}: n-RMSE (%) at eps=1",
              np.nanmean(d[f"{c}__M"][i1]) * 100, v, 0.06)


def s09():
    print("  S9  sentinel count: fits quoted in the text, c-sweep plotted in Fig. S19")
    r = np.load(HERE / "s09_sentinel_count/data/ranks14.npz", allow_pickle=True)
    N = r["N"].astype(float)
    for i, (sl, ic, r2) in enumerate([(0.49, -0.55, 0.52), (1.01, -2.07, 0.47),
                                      (0.51, -0.57, 0.50), (0.02, 0.96, 0.35)]):
        s, b = np.polyfit(np.log(N), r["mean"][:, i], 1)
        q = np.corrcoef(np.log(N), r["mean"][:, i])[0, 1] ** 2
        check(f"effective rank {i+1}: slope per ln N", s, sl, 0.006)
        check(f"effective rank {i+1}: intercept", b, ic, 0.006)
        check(f"effective rank {i+1}: R^2", q, r2, 0.006)
    c = np.load(HERE / "s09_sentinel_count/data/csweep.npz", allow_pickle=True)
    cs = c["cs"]; src = [str(x) for x in c["sources"]]
    pan = [str(x) for x in c["panels"]].index("nn_minc")
    i1 = int(np.argmin(np.abs(cs - 1.0)))
    for s, v in zip(["SIS", "SIR", "C17", "C11", "C11L1"],
                    [7.12, 5.40, 2.61, 2.24, 2.95]):
        check(f"c-sweep at c=1, NN+MinCorr, {s}: n-RMSE (%)",
              c["M"][i1, src.index(s), pan] * 100, v, 0.006)


def s15():
    print("  S15 structural sentinels, Table S23")
    d = np.load(HERE / "s15_structural_sentinels/data/struct.npz",
                allow_pickle=True)
    eye = np.eye(11, dtype=bool)
    # the SI restricts the summary to the 11 adjacency-coupled dynamics
    SI = {("nn", "degree"): (0.50, 16.07, 3.22, 17.40),
          ("nn", "betweenness"): (0.44, 13.50, 2.76, 14.89),
          ("lin", "degree"): (0.59, 13.90, 3.17, 18.49),
          ("lin", "betweenness"): (0.35, 12.02, 2.74, 16.45)}
    for (dec, rule), (md, mo, ad, ao) in SI.items():
        M = d[f"M_{dec}_{rule}"][:11, :11] * 100
        for lab, got, want in (("median diagonal", np.median(M[eye]), md),
                               ("median off-diagonal", np.median(M[~eye]), mo),
                               ("mean diagonal", M[eye].mean(), ad),
                               ("mean off-diagonal", M[~eye].mean(), ao)):
            check(f"{dec}/{rule}: {lab}", got, want, max(0.006, 0.001 * want))


for f in (s01, s02, s03, s08, s09, s15):
    try:
        f()
    except Exception as e:
        FAIL.append(f.__name__)
        print(f"    {f.__name__}: could not run -- {e}")
print()
if FAIL:
    print(f"  {len(FAIL)} check(s) failed: {', '.join(FAIL)}")
    sys.exit(1)
print("  all checks passed")
