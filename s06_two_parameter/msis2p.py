# =====================================================================
# SI Section S6: Two-parameter families of dynamics
#
# This script: the MSIS2P model, imported by the sweep drivers
# Feeds      : Figs. S7 and S8, as a module
# Produces   : (module)
# Runtime    : -
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""MSIS as a two-parameter family: infection rate beta and infected-diffusion D_I.

The paper's MSIS is
    d rho_S/dt = -beta rho_S rho_I + mu rho_I + D_S [A (rho_S/k) - rho_S]
    d rho_I/dt = +beta rho_S rho_I - mu rho_I + D_I [A (rho_I/k) - rho_I]
with mu = D_S = 1 and D_I = 1 (MSIS1) or D_I = 0.2 (MSIS2).  Here D_I is a
second control parameter, so MSIS1 and MSIS2 are two columns of one grid.

Recorded quantity is rho_I, as in the paper.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigsh

HERE = Path(__file__).resolve().parent

MU, D_S = 1.0, 1.0
T_FINAL = 100.0                 # the paper's horizon for the metapopulation models
RTOL, ATOL = 1e-7, 1e-9         # tighter than the paper's 1e-4/1e-6
I0 = 0.1                        # the paper's uniform start: rho_S=0.9, rho_I=0.1
THRESHOLD = 0.1                 # the paper's acceptance for beta_max widening


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib")); import paths  # noqa: E402
def load_network(tag):
    G = nx.read_edgelist(paths.network_file(tag), nodetype=int)
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")
    N = A.shape[0]
    k = np.asarray(A.sum(axis=1)).ravel()
    if N < 1500:
        alpha_max = float(np.linalg.eigvalsh(A.toarray()).max())
    else:
        alpha_max = float(eigsh(A, k=1, which="LA",
                                return_eigenvectors=False)[0])
    return dict(tag=tag, N=N, A=A, k=k, alpha_max=alpha_max,
                kbar=float(k.mean()))


def rhs_factory(net, beta, D_I):
    A, k = net["A"], net["k"]

    def rhs(t, y):
        N = y.size // 2
        rS, rI = y[:N], y[N:]
        dS = D_S * (A @ (rS / k) - rS)
        dI = D_I * (A @ (rI / k) - rI)
        inf = beta * rS * rI
        rec = MU * rI
        return np.concatenate([-inf + rec + dS, inf - rec + dI])
    return rhs


def jac_factory(net, beta, D_I):
    """Analytic sparse Jacobian of the 2N-state system."""
    A, k = net["A"], net["k"]
    N = net["N"]
    Ak = A.multiply(1.0 / k[None, :]).tocsr()      # A_ij / k_j
    I_N = sp.identity(N, format="csr")

    def jac(t, y):
        rS, rI = y[:N], y[N:]
        dSS = D_S * (Ak - I_N) - sp.diags(beta * rI)
        dSI = sp.diags(MU - beta * rS)
        dIS = sp.diags(beta * rI)
        dII = D_I * (Ak - I_N) + sp.diags(beta * rS - MU)
        return sp.bmat([[dSS, dSI], [dIS, dII]], format="csr")
    return jac


def equilibrate(net, beta, D_I, y0, t_final=T_FINAL):
    rhs = rhs_factory(net, beta, D_I)
    sol = solve_ivp(rhs, (0.0, t_final), y0, method="BDF",
                    jac=jac_factory(net, beta, D_I), rtol=RTOL, atol=ATOL)
    return sol.y[:, -1], bool(sol.success)


def uniform_ic(N):
    return np.concatenate([np.full(N, 1.0 - I0), np.full(N, I0)])
