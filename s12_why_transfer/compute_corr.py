# =====================================================================
# SI Section S12: Why the transferable dynamics transfer
#
# This script: Pearson correlations of the equilibrium state vectors
# Feeds      : Fig. S23, Table S15, and the block means and AUC values
#              quoted in Section S12
# Produces   : data/corr.npz
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Why do the adjacency-coupled dynamics transfer to one another?

The hypothesis tested here is that the transferable dynamics arrange their
equilibrium states across the nodes in nearly the same way, approximately
as a function of degree, so that a decoder trained on one of them carries
over to another.

For each network and each dynamics we take one state vector from the
control-parameter sweep: the one at which the sample variance of the
equilibrium state over the nodes is largest (Fig. S23(a)), or the one whose
sample variance is the median of the M = 51 values (Fig. S23(b)).  To the 17
state vectors we add the degree vector and the log-degree vector as
references, compute the Pearson correlation between every pair of these 19
vectors, and average each entry of the 19 x 19 matrix over the 14 networks.

Output: the two mean matrices, mean_max and mean_med.  The script also
prints the block means that Section S12 quotes.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np
import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib")); import paths  # noqa: E402
EQ = paths.EQ_SUBMITTED
REVDATA = paths.EQ_REVISION
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

NETWORKS = ["dolphin", "prox", "metabolic", "gkk", "lfr", "er", "ba", "hk",
            "euroroad", "email", "flybi", "reactome", "route_views",
            "spanish"]
# 11 adjacency-coupled first, then the degree and the log degree, then the
# 6 Laplacian-coupled, so that the adjacency-coupled block is contiguous.
ADJ = [("SIS", "sis"), ("SIR", "sir"), ("MSIS1", "msis"), ("MSIS2", "msis2"),
       ("MSIR1", "msir"), ("MSIR2", "msir2"), ("DW", "doublewell"),
       ("MUT", "mutualistic"), ("GEN", "genereg"), ("WC", "wc"),
       ("gLV", "glv")]
LAP = [("NM", "noymeir"), ("EUT1", "eutroph"), ("EUT2", "eutroph2"),
       ("VEG1", "vegturb"), ("VEG2", "vegturb2"), ("LDW", "ldw")]
LABELS = [n for n, _ in ADJ] + ["degree", "log degree"] + [n for n, _ in LAP]
NADJ = len(ADJ)


def load_state(net, stem):
    """Equilibrium sweep for one (network, dynamics), shape (M, N)."""
    f = EQ / f"{net}_{stem}_equilibrium_data.csv"
    if not f.exists():                       # LDW lives with the revision
        f = REVDATA / f"{net}_{stem}_equilibrium_data.csv"
    arr = np.loadtxt(f, delimiter=",", skiprows=1)
    return arr[:, 0], arr[:, 1:]


def degrees(net):
    G = nx.read_edgelist(paths.network_file(net), nodetype=int)
    nodes = sorted(G.nodes())                # the ordering the producers use
    return np.array([G.degree(v) for v in nodes], dtype=float)


def selected_state(ps, Y, how):
    """The row of the sweep at the largest, or the median, sample variance."""
    v = Y.var(axis=1, ddof=1)
    j = int(np.argmax(v)) if how == "max" else int(np.argsort(v)[len(v) // 2])
    return Y[j], float(ps[j]), float(v[j])


def build(how):
    """Mean Pearson correlation matrix over the networks.

    `how` is "max" or "median": which point of the control-parameter sweep
    supplies the state vector.
    """
    mats = []
    for net in NETWORKS:
        k = degrees(net)
        vecs, N = [], len(k)
        for name, stem in ADJ + [("degree", None)] + LAP:
            if stem is None:
                assert k.min() > 0, f"{net}: isolated node, log undefined"
                vecs += [k, np.log(k)]
                continue
            ps, Y = load_state(net, stem)
            assert Y.shape[1] == N, f"{net} {stem}: {Y.shape[1]} != {N}"
            x, _, _ = selected_state(ps, Y, how)
            vecs.append(x)
        V = np.vstack(vecs)
        C = np.corrcoef(V)
        flat = np.ptp(V, axis=1) == 0         # no correlation is defined
        if flat.any():
            print(f"  {net}: constant state vector for "
                  f"{[LABELS[i] for i in np.where(flat)[0]]}")
            C[flat, :] = np.nan; C[:, flat] = np.nan
        mats.append(C)
    return np.nanmean(np.dstack(mats), axis=2)


def main():
    out = {"mean_max": build("max"), "mean_med": build("median")}
    np.savez(OUT / "corr.npz", labels=np.array(LABELS),
             networks=np.array(NETWORKS), **out)
    print("  wrote data/corr.npz")

    # the block means quoted in Section S12 (off-diagonal entries only)
    A = list(range(NADJ)); L = list(range(NADJ + 2, len(LABELS)))
    for key, panel in (("mean_max", "Fig. S23(a)"), ("mean_med", "Fig. S23(b)")):
        M = out[key]
        def off(rows, cols):
            return np.nanmean([M[i, j] for i in rows for j in cols if i != j])
        cross = np.nanmean([M[i, j] for i in A for j in L]
                           + [M[j, i] for i in A for j in L])
        print(f"  {panel}: adjacency x adjacency {off(A, A):.3f}, "
              f"Laplacian x Laplacian {off(L, L):.3f}, "
              f"adjacency x Laplacian {cross:.3f}")


if __name__ == "__main__":
    main()
