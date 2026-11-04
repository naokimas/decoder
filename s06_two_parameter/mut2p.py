# =====================================================================
# SI Section S6: Two-parameter families of dynamics
#
# This script: the MUT2P model, imported by the sweep drivers
# Feeds      : Figs. S7 and S8, as a module
# Produces   : (module)
# Runtime    : -
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""MUT as a two-parameter family: coupling strength D and immigration B.

The paper sweeps D alone at fixed B = 0.1.  Here B becomes a second axis.

    dx_i/dt = B + x_i (1 - x_i/K)(x_i/C - 1)
              + D * sum_j A_ij * g(x_i, x_j),
    g(a, b) = a b / (Dt + E a + H b).

Everything except B is the paper's MUT verbatim (K = 5, C = 1, Dt = 5,
E = 0.9, H = 0.1), and the integration protocol is the paper's too,
taken from run_all_12_dynamics_v9.py (which produced the published
CSVs, superseding regenerate_mut_gen_wc_v3.py): a FORWARD WARM START
in D, seeded at x = 0.001 at D = 0 only, BDF with jac_sparsity = A + I at
rtol 1e-4 / atol 1e-6, for every network.  The one departure is the
horizon: the paper stops at t_final = 200, whereas equilibrate_settled()
extends each solve until the state has settled.

Positivity.  B > 0 makes f(0) = B > 0, so x = 0 is repelling and no
trajectory started at x > 0 can reach it.  Every B on the grid is
strictly positive, which is what keeps x_i off zero.  B = 0 is excluded
deliberately: there x = 0 is an exact equilibrium, the low-branch
initial condition decays into it for every D (the mutualistic term is
O(x^2) and cannot rescue an O(x) decay), and the whole row would be
identically zero.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent

K_CAP, C_ALLEE = 5.0, 1.0                          # MUT local, unchanged
D_SAT, E_SAT, H_SAT = 5.0, 0.9, 0.1                # MUT kernel, unchanged
X0_LOW = 0.001                                     # paper's low-branch IC
T_FINAL = 200.0                                    # paper's MUT horizon
RTOL, ATOL = 1e-4, 1e-6                            # run_all_12_dynamics_v9
MUT_CONSTANT = 10.258                              # = 2.0 * <k>_dolphin
RES_TOL = 1e-6                    # settle until max_i |dx_i/dt| < RES_TOL
T_CAP = 2.0e5                     # ... but never integrate past this
B_SN = 0.22569                                     # local saddle-node in B


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib")); import paths  # noqa: E402
def load_network(tag):
    """Node order must match the paper's exactly.

    run_all_12_dynamics_v9.py builds A with nodelist=sorted(G.nodes()), so
    column i of every equilibrium CSV is the same node across dynamics.
    convert_node_labels_to_integers would relabel the nodes but leave the
    iteration order unspecified, permuting the state vector.
    """
    G = nx.read_edgelist(paths.network_file(tag), nodetype=int)
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float,
                                 format="csr")
    k = np.asarray(A.sum(axis=1)).ravel()
    return dict(tag=tag, N=A.shape[0], A=A, k=k, kbar=float(k.mean()))


def _f(x, B):
    return B + x * (1.0 - x / K_CAP) * (x / C_ALLEE - 1.0)


def _g(a, b):
    return a * b / (D_SAT + E_SAT * a + H_SAT * b)


def rhs_factory(net, D, B):
    A = net["A"]
    cols = A.indices
    rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
    N = net["N"]

    def rhs(t, x):
        c = np.bincount(rows, weights=_g(x[rows], x[cols]), minlength=N)
        return _f(x, B) + D * c

    return rhs


def jac_sparsity(net):
    """A + I, the per-node pattern run_all_12_dynamics_v9.py uses for MUT."""
    return net["A"] + sp.eye(net["N"])


def equilibrate_settled(net, D, B, x0=None, jac=None, res_tol=RES_TOL,
                        t_cap=T_CAP):
    """Integrate to t=200, then keep going (x10 each time) until settled.

    The paper stops MUT at t=200 unconditionally.  Near the saddle-node
    that is not always long enough: evaluating the rhs at the published
    equilibria gives max_i |dx_i/dt| up to 3.5 on hk and 0.44 on flybi,
    i.e. those points are still in transit rather than at rest.  Here the
    horizon extends until the residual really is small, so every cell of
    the grid is an equilibrium.  Returns (x, ok, t_used, residual).
    """
    if x0 is None:
        x0 = np.full(net["N"], X0_LOW)
    if jac is None:
        jac = jac_sparsity(net)
    rhs = rhs_factory(net, D, B)
    x = x0
    t_used = 0.0
    t = T_FINAL
    ok = True
    while True:
        sol = solve_ivp(rhs, (0.0, t), x, method="BDF", rtol=RTOL,
                        atol=ATOL, jac_sparsity=jac)
        ok = ok and bool(sol.success)
        x = sol.y[:, -1]
        t_used += t
        res = float(np.abs(rhs(0.0, x)).max())
        if res < res_tol or t_used >= t_cap or not sol.success:
            return x, ok, t_used, res
        t *= 10.0
