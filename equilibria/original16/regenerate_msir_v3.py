"""MSIR cold-start regen for all 12 networks.

beta_max determination: same as MSIS. Start at beta_max = 1/(0.15*alpha_max);
double until a probe at 0.5*beta_max yields mean rho_R >= 0.02. Once
beta_max is determined, do an ascending COLD-start sweep (since MSIR is
transient: no warm-start across sweep points). Each sweep point starts
from rho_S(0)=0.9, rho_I(0)=0.1, rho_R(0)=0, integrates to t_max=100,
records rho_R(t_max).
"""
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import networkx as nx
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
NETWORKS = ["dolphin", "prox", "metabolic", "gkk", "lfr", "er",
            "ba", "hk", "euroroad", "email", "flybi", "reactome"]
NSWEEP = 51
T_FINAL = 100.0
SIS_PERCENTILE = 0.15
THRESHOLD = 0.1     # adaptive beta_max widening acceptance (Methods sec:autotrim)
LARGE_NETS = {"reactome"}


def load_network(tag):
    G = nx.read_edgelist(HERE / f"{tag}_network.edgelist", nodetype=int)
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")
    N = A.shape[0]
    k_vec = np.asarray(A.sum(axis=1)).ravel()
    if N < 200:
        alpha_max = float(np.linalg.eigvalsh(A.toarray()).max())
    else:
        alpha_max = float(eigsh(A, k=1, which="LA",
                                return_eigenvectors=False)[0])
    return N, A, k_vec, alpha_max


def make_msir_rhs(A, k_vec):
    MU, D_S, D_I = 1.0, 1.0, 1.0
    def rhs(t, state, beta):
        N = state.size // 3
        rS = state[:N]; rI = state[N:2*N]
        diff_S = D_S * (A @ (rS / k_vec) - rS)
        diff_I = D_I * (A @ (rI / k_vec) - rI)
        infect = beta * rS * rI
        recov = MU * rI
        return np.concatenate([-infect + diff_S,
                                infect - recov + diff_I,
                                recov])
    return rhs


def msir_jac_pattern(A):
    """3x3 block sparsity, with R block decoupled from S, I except via R
    accumulation:
       J = [[A+I, A+I, 0],
            [A+I, A+I, 0],
            [ 0 ,  I , 0]]"""
    N = A.shape[0]
    A_plus_I = A + sp.eye(N)
    Z = sp.csr_matrix((N, N))
    return sp.bmat([[A_plus_I, A_plus_I, Z],
                    [A_plus_I, A_plus_I, Z],
                    [Z,        sp.eye(N), Z]], format="csr")


def integrate(rhs, beta, state0, jac, large):
    if large:
        sol = solve_ivp(rhs, (0.0, T_FINAL), state0, method="BDF",
                        args=(beta,), rtol=1e-4, atol=1e-6,
                        jac_sparsity=jac)
    else:
        sol = solve_ivp(rhs, (0.0, T_FINAL), state0, method="LSODA",
                        args=(beta,), rtol=1e-7, atol=1e-9)
    if not sol.success:
        raise RuntimeError(f"failed at beta={beta}: {sol.message}")
    return sol.y[:, -1]


def main():
    for tag in NETWORKS:
        print(f"\n========== {tag} ==========")
        N, A, k_vec, alpha_max = load_network(tag)
        large = tag in LARGE_NETS
        jac = msir_jac_pattern(A) if large else None
        rhs = make_msir_rhs(A, k_vec)

        rS_init = np.full(N, 0.999); rI_init = np.full(N, 0.001)
        rR_init = np.zeros(N)
        ic = np.concatenate([rS_init, rI_init, rR_init])

        beta_max = 1.0 / (SIS_PERCENTILE * alpha_max)
        for trial in range(20):
            beta_test = 0.5 * beta_max
            t0 = time.time()
            state_eq = integrate(rhs, beta_test, ic, jac, large)
            rR = state_eq[2*N:]
            mean_rR = float(rR.mean())
            print(f"  trial={trial}  beta_max={beta_max:.4f}  "
                  f"beta_test={beta_test:.4f}  rho_R_mean={mean_rR:.4f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
            if mean_rR >= THRESHOLD:
                break
            beta_max *= 2.0
        else:
            print(f"  WARN: did not reach threshold; "
                  f"using last beta_max={beta_max:.4f}")

        grid = np.linspace(0.0, beta_max, NSWEEP)
        eq = np.empty((NSWEEP, N))
        t0 = time.time()
        for k, beta in enumerate(grid):
            state_eq = integrate(rhs, beta, ic, jac, large)
            eq[k] = state_eq[2*N:]
            if k % 10 == 0 or k == NSWEEP - 1:
                print(f"    MSIR k={k}/{NSWEEP-1} beta={beta:.4f} "
                      f"rho_R_mean={eq[k].mean():.3f} "
                      f"({time.time()-t0:.1f}s)", flush=True)
        print(f"  done in {time.time()-t0:.1f}s, beta_max={beta_max:.4f}, "
              f"rho_R-range [{eq.min():.4f}, {eq.max():.4f}]")

        arr = np.column_stack([grid, eq])
        np.savetxt(HERE / f"{tag}_msir_equilibrium_data.csv", arr,
                   delimiter=",",
                   header="p," + ",".join(f"x_{i+1}" for i in range(N)),
                   comments="", fmt=["%.6f"] + ["%.6f"] * N)

        fn = HERE / f"{tag}_bifurcation_ranges.csv"
        with fn.open() as f:
            header = f.readline().rstrip("\n")
            rows = [line.rstrip("\n").split(",") for line in f if line.strip()]
        by_key = {r[0]: r for r in rows}
        by_key["MSIR"] = ["MSIR", f"{0.0:.6f}", f"{beta_max:.6f}",
                          f"{NSWEEP:d}", f"{eq.min():.6f}",
                          f"{eq.max():.6f}"]
        KEYS = ["SIS", "DW", "MUT", "GEN", "WC", "gLV", "SIR", "NM",
                "EUT", "VEG", "MSIS", "MSIR"]
        out = [header]
        for k in KEYS:
            if k in by_key:
                out.append(",".join(by_key[k]))
        fn.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
