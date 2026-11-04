# =====================================================================
# SI Section S13: Source-decoder rankings
#
# This script: the budget-matched pooled decoders C11B and C11L1B
# Feeds      : the C11B and C11L1B rows of Tables S20 and S21, and the budget-matched
#              ranges and ranks quoted in the text of Section S13, which
#              make_tables.py prints
# Produces   : data/budget*.npz
# Runtime    : about an hour
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Pooled decoders given the SAME training budget as a per-dynamics decoder.

C11 sees 11 x 41 training columns against a per-dynamics decoder's 41, and
it also sees the target dynamics itself, so part of its advantage could come
from the amount of training data alone.  C11L1 removes the target
but still carries ten times the data.

This script builds the budget-matched counterparts:

  C11B    pooled over the 11 transferable dynamics, but 41 training columns
          in TOTAL.  With 11 members, 41 = 8 x 4 + 3 x 3, so eight dynamics
          contribute four control-parameter values and three contribute
          three.  Which members get four is drawn uniformly at random.
  C11L1B  the same for the leave-one-dynamics-out pool: 10 members,
          41 = 1 x 5 + 9 x 4.
  C17B    pooled over all 17 dynamics, 41 columns: 41 = 7 x 3 + 10 x 2.
  C17L1B  its leave-one-out counterpart: 16 members, 41 = 9 x 3 + 7 x 2.

The columns contributed by a member are drawn uniformly at random, without
replacement, from that member's 41 interspersed training rows, so they stay
interspersed over the sweep.  Each member is normalized to [0, 1] on the
columns it actually contributes, since those are the only ones the decoder
sees.  Everything else -- architecture, sentinel selection on the pooled
matrix, out-of-sample target calibration -- is as in compute_rank.py.

One draw of the 41 columns is used per (network, decoder, sentinel)
combination, with a deterministic seed, so no averaging over draws is
involved; the average is over the 14 networks only.

The seed is built from the network index and from the position of the panel
in the variant's panel list, and the pools then draw from it one after
another.  The draws are therefore not shared across variants: the linear
panels sit at positions 3 to 5 in 'main' but 0 to 2 in 'alpha01', and
'main' draws C17B before the leave-one-out pools whereas 'bias' and
'alpha01' do not.  Only C11B under 'bias' uses the same columns as under
'main'.  The comparisons in Tables S20 and S21 for C11B and C11L1B thus also
include a change of draw.

Writes data/budget.npz.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
from pathlib import Path
import numpy as np
import networkx as nx

import compute_rank as cr

HERE = Path(__file__).resolve().parent
R_DRAWS = 1
BUDGET = 41


def split_budget(n_members, total, rng):
    """`total` columns over `n_members` members, as equal as possible.

    The members that get the extra column are drawn uniformly at random.
    """
    base, extra = divmod(total, n_members)
    counts = np.full(n_members, base)
    counts[rng.choice(n_members, size=extra, replace=False)] += 1
    return counts


def budget_pool(members, eq, tr, rng):
    """Stack the members using `BUDGET` columns in total."""
    counts = split_budget(len(members), BUDGET, rng)
    blocks = []
    for s, c in zip(members, counts):
        rows = rng.choice(tr, size=int(c), replace=False)
        Y = eq[s][np.sort(rows)]
        lo = float(Y.min()); hi = float(Y.max())
        blocks.append((Y - lo) / max(hi - lo, 1e-12))
    return np.vstack(blocks), counts


def main():
    global R_DRAWS
    variant = sys.argv[1] if len(sys.argv) > 1 else "main"
    cr.VARIANT = variant
    panels = cr.VARIANTS[variant]["panels"]
    full = variant == "main"      # the 17-dynamics pools only for the main spec
    print(f"variant {variant}: panels {panels}, 17-pools {full}", flush=True)
    n = len(cr.DYN); nt = len(cr.TRANS)
    keys = ["c11b", "c11l1b"] + (["c17b", "c17l1b"] if full else [])
    shape = {"c11b": (n,), "c11l1b": (nt,), "c17b": (n,), "c17l1b": (n,)}
    M = {p: {k: np.zeros(shape[k]) for k in keys} for p in panels}
    B = {p: {k: np.zeros(shape[k]) for k in keys} for p in panels}
    S2 = {p: {k: np.zeros(shape[k]) for k in keys} for p in panels}
    tr, ev = cr.splits.rows_random()
    t0 = time.time()
    for ni, tag in enumerate(cr.NETS):
        G = nx.read_edgelist(cr.leq.network_file(tag),
                             nodetype=int)
        N = len(G.nodes()); K = int(np.floor(np.log(N)))
        eq = {s: cr.leq.load_eq(tag, s) for s in cr.DYN}
        rngf = {s: float(eq[s].max() - eq[s].min()) for s in cr.DYN}
        RAND = np.sort(np.random.default_rng(2026).choice(N, size=K,
                                                          replace=False))
        for pi, (dec, sent) in enumerate(panels):
            p = (dec, sent)
            acc = {k: np.zeros(shape[k]) for k in keys}
            accb = {k: np.zeros(shape[k]) for k in keys}
            acc2 = {k: np.zeros(shape[k]) for k in keys}
            for r in range(R_DRAWS):
                rng = np.random.default_rng((ni * 97 + pi) * 1009 + r)
                pools = [("c11b", cr.TRANS)] + ([("c17b", cr.DYN)] if full
                                                else [])
                for key, members in pools:
                    Yp, _ = budget_pool(members, eq, tr, rng)
                    idx = cr.sentinels(Yp, sent, K, RAND)
                    ck = cr.train(Yp, idx, dec, np.arange(Yp.shape[0]))
                    for j, tgt in enumerate(cr.DYN):
                        m, b = cr.score(ck, eq[tgt], dec, tr, ev, rngf[tgt])
                        acc[key][j] += m; accb[key][j] += b
                        acc2[key][j] += m * m
                loo = [("c11l1b", cr.TRANS)] + ([("c17l1b", cr.DYN)] if full
                                                else [])
                for key, pool_set in loo:
                    for j, tgt in enumerate(pool_set):
                        members = [s for s in pool_set if s != tgt]
                        Yp, _ = budget_pool(members, eq, tr, rng)
                        idx = cr.sentinels(Yp, sent, K, RAND)
                        ck = cr.train(Yp, idx, dec, np.arange(Yp.shape[0]))
                        m, b = cr.score(ck, eq[tgt], dec, tr, ev, rngf[tgt])
                        acc[key][j] += m; accb[key][j] += b
                        acc2[key][j] += m * m
            for k in keys:
                M[p][k] += acc[k] / R_DRAWS
                B[p][k] += accb[k] / R_DRAWS
                S2[p][k] += acc2[k] / R_DRAWS
        print(f"  {tag}: {time.time()-t0:.0f}s", flush=True)
    out = {}
    for dec, sent in panels:
        for k in keys:
            out[f"M_{dec}_{sent}_{k}"] = M[(dec, sent)][k] / len(cr.NETS)
            out[f"B_{dec}_{sent}_{k}"] = B[(dec, sent)][k] / len(cr.NETS)
            var = S2[(dec, sent)][k] / len(cr.NETS) - out[f"M_{dec}_{sent}_{k}"] ** 2
            out[f"S_{dec}_{sent}_{k}"] = np.sqrt(np.maximum(var, 0.0))
    fn = "budget.npz" if variant == "main" else f"budget_{variant}.npz"
    np.savez(HERE / "data" / fn,
             dynamics=np.array(
                 [cr.DISP.get(d, d) for d in cr.DYN], dtype=object),
             transferable=np.array(
                 [cr.DISP.get(d, d) for d in cr.TRANS], dtype=object),
             draws=R_DRAWS, budget=BUDGET, **out)
    print(f"  wrote data/{fn}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
