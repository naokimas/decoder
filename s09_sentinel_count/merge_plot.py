# =====================================================================
# SI Section S9: Robustness to the number of sentinel nodes
#
# This script: merges the shards and draws Fig. S19
# Produces   : data/csweep.npz, fig_csweep_rev.pdf
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Merge the c-sweep shards and draw Fig. S19.

Every data/part_*.npz is read; the stored results come from seven shards
(part_A, part_B, part_C and part_D1 to part_D4), and the merge checks that
together they cover each cell with exactly the 14 networks.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)

parts = sorted((HERE / "data").glob("part_*.npz"))
M = B = CNT = None
for p in parts:
    d = np.load(p, allow_pickle=True)
    M = d["M"] if M is None else M + d["M"]
    B = d["B"] if B is None else B + d["B"]
    CNT = d["CNT"] if CNT is None else CNT + d["CNT"]
    cs, sources, panels = d["cs"], d["sources"], d["panels"]
    ndol = d["n_dolphin"] if d["n_dolphin"].max() > 0 else None
print(f"  merged {len(parts)} shards; networks per cell = {CNT.max():.0f}")
assert CNT.min() == CNT.max() == 14, f"uneven coverage: {CNT.min()}-{CNT.max()}"
M = M / CNT; B = B / CNT; R = M / B
np.savez(HERE / "data" / "csweep.npz", M=M, B=B, cs=cs, sources=sources,
         panels=panels, n_dolphin=ndol if ndol is not None else np.zeros(0))

SRC = [str(s) for s in sources]; PAN = [str(s) for s in panels]
COLOR = {"SIS": "#1f77b4", "SIR": "#ff7f0e", "C17": "#2ca02c",
         "C11": "#d62728", "C11L1": "#9467bd"}
LSTY = {"minc": "-", "qr": ":"}
DEC = {"nn": "NN", "lin": "linear"}
fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True)
LAY = [(0, 0, "nn", "rmse"), (0, 1, "nn", "ratio"),
       (1, 0, "lin", "rmse"), (1, 1, "lin", "ratio")]
LETT = {(0, 0): "a", (0, 1): "b", (1, 0): "c", (1, 1): "d"}
for r, c, dec, met in LAY:
    ax = axes[r, c]
    for s in SRC:
        for sent in ("minc", "qr"):
            pi = PAN.index(f"{dec}_{sent}"); si = SRC.index(s)
            y = (M[:, si, pi] * 100) if met == "rmse" else R[:, si, pi]
            ax.plot(cs, y, LSTY[sent], color=COLOR[s], linewidth=1.6)
    ax.set_xlim(0.25, 2.0); ax.set_xticks([0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(["0.5", "1", "1.5", "2"])
    if met == "rmse":
        ax.set_ylabel("n-RMSE (%)", fontsize=12)
    else:
        ax.set_ylabel(r"$r(\mathrm{S}\to\mathrm{T})$", fontsize=12)
    if r == 1:
        ax.set_xlabel(r"$c$", fontsize=12)
    ax.set_title(f"({LETT[(r, c)]}) {DEC[dec]}, "
                 + ("n-RMSE" if met == "rmse"
                    else r"$r(\mathrm{S}\to\mathrm{T})$"),
                 fontsize=12, loc="left")
    ax.tick_params(labelsize=10); ax.grid(alpha=0.4, linewidth=0.4)
h = [Line2D([0], [0], color=COLOR[s], lw=1.8, label=s) for s in SRC]
h += [Line2D([0], [0], color="black", ls=LSTY[m], lw=1.8,
             label={"minc": "MinCorr", "qr": "QR"}[m]) for m in ("minc", "qr")]
fig.legend(handles=h, labels=[x.get_label() for x in h], loc="lower center",
           ncol=7, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(FIGS / "fig_csweep_rev.pdf", bbox_inches="tight")
plt.close(fig)
print("  wrote figs/fig_csweep_rev.pdf")

# numbers quoted in the SI text
print("\n  at c = 1 (n = floor(ln N)), NN decoder, MinCorr sentinels:")
i1 = int(np.argmin(np.abs(cs - 1.0)))
for s in SRC:
    pi = PAN.index("nn_minc"); si = SRC.index(s)
    print(f"    {s:6s} n-RMSE {M[i1,si,pi]*100:6.2f}%   r {R[i1,si,pi]:.3f}")
print("\n  n-RMSE at the extremes of c, NN + MinCorr:")
for s in SRC:
    pi = PAN.index("nn_minc"); si = SRC.index(s)
    print(f"    {s:6s} c=0.25 {M[0,si,pi]*100:6.2f}%  c=1 "
          f"{M[i1,si,pi]*100:6.2f}%  c=2 {M[-1,si,pi]*100:6.2f}%")
