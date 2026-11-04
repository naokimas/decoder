# =====================================================================
# SI Section S10: Robustness of the cross-transfer network structure
#
# This script: shared dendrogram styling, imported by cluster.py
# Produces   : (module)
# Runtime    : -
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""Shared styling for the dendrogram figures of SI Sections S10 and S11.

The subtree spanned by the adjacency-coupled dynamics is drawn in green and
every other link of the dendrogram in black.
"""
import numpy as np

GREEN = "#2f9e44"     # mid green: visible against white, unlike "lightgreen"
BLACK = "#000000"


def leaf_sets(Z, n):
    """Set of leaf indices below each node of the linkage matrix Z."""
    sets = [frozenset([i]) for i in range(n)]
    for row in Z:
        sets.append(sets[int(row[0])] | sets[int(row[1])])
    return sets


def link_color_func(Z, labels, group):
    """Color a link green iff all leaves below it belong to `group`."""
    n = len(labels)
    sets = leaf_sets(Z, n)
    idx = frozenset(i for i, name in enumerate(labels) if name in group)

    def f(k):
        return GREEN if sets[k] <= idx else BLACK
    return f


def group_first(Z, labels, group):
    """Reorder the children of every node so that `group` ends up on the left.

    scipy draws the two children of a node in the order in which they appear
    in the row of the linkage matrix, so swapping the two entries mirrors that
    subtree without changing the tree it encodes.  At each node we put the
    child with the larger fraction of `group` leaves first, which walks the
    subtree spanned by `group` to the left-hand edge of the panel.
    """
    n = len(labels)
    sets = leaf_sets(Z, n)
    idx = frozenset(i for i, name in enumerate(labels) if name in group)
    share = [len(sets[k] & idx) / len(sets[k]) for k in range(len(sets))]
    out = Z.copy()
    for i, row in enumerate(out):
        a, b = int(row[0]), int(row[1])
        if share[b] > share[a]:
            out[i, 0], out[i, 1] = b, a
    return out


def orient_left(ax, order, group):
    """Mirror the dendrogram so that the `group` block sits on the left.

    A dendrogram may be reflected without changing what it represents.
    """
    inside = [i for i, x in enumerate(order) if x in group]
    outside = [i for i, x in enumerate(order) if x not in group]
    if inside and outside and np.mean(inside) > np.mean(outside):
        ax.invert_xaxis()


def yticks_by_five(ax, Z):
    top = np.ceil(Z[-1, 2] / 5.0) * 5.0
    ax.set_yticks(np.arange(0, top + 1e-9, 5.0))
