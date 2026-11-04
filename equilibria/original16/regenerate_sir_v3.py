"""SIR cold-start regen for all 12 networks under the SIS-matched beta
range: beta in [0, 1/(0.15*alpha_max)]. SIR is non-equilibrium so we
must cold-start at every sweep point with S(0)=0.99, I(0)=0.01,
integrate to t_final=400, and record R(t_final) = 1 - S - I."""
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
T_FINAL = 400.0
SIS_PERCENTILE = 0.15
LARGE_NETS = {"reactome"}


def load_network(tag):
    G = nx.read_edgelist(HERE / f"{tag}_network.edgelist", nodetype=int)
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")
    N = A.shape[0]
    if N < 200:
        alpha_max = float(np.linalg.eigvalsh(A.toarray()).max())
    else:
        alpha_max = float(eigsh(A, k=1, which="LA",
                                return_eigenvectors=False)[0])
    return N, A, alpha_max


def make_sir_rhs(A):
    GAMMA = 1.0  # mu (recovery rate); unified with SIS's mu in next .tex
    def rhs(t, state, beta):
        N = state.size // 2
        S = state[:N]; I = state[N:]
        AI = A @ I
        dS = -beta * S * AI
        dI = beta * S * AI - GAMMA * I
        return np.concatenate([dS, dI])
    return rhs


def sir_jac_pattern(A):
    """Block sparsity pattern for SIR (S, I) Jacobian:
       J = [[A+I, A+I], [A+I, A+I]] (block 2x2)."""
    N = A.shape[0]
    A_plus_I = A + sp.eye(N)
    return sp.bmat([[A_plus_I, A_plus_I],
                    [A_plus_I, A_plus_I]], format="csr")


def main():
    for tag in NETWORKS:
        print(f"\n========== {tag} ==========")
        N, A, alpha_max = load_network(tag)
        beta_max = 1.0 / (SIS_PERCENTILE * alpha_max)
        large = tag in LARGE_NETS
        jac = sir_jac_pattern(A) if large else None
        rhs = make_sir_rhs(A)
        grid = np.linspace(0.0, beta_max, NSWEEP)
        print(f"  N={N}, alpha_max={alpha_max:.3f}, "
              f"beta_max={beta_max:.4f} (= lambda_max for SIS)")
        eq = np.empty((NSWEEP, N))
        t0 = time.time()
        for k, beta in enumerate(grid):
            S0 = np.full(N, 0.999); I0 = np.full(N, 0.001)
            state0 = np.concatenate([S0, I0])
            if large:
                sol = solve_ivp(rhs, (0.0, T_FINAL), state0, method="BDF",
                                args=(beta,), rtol=1e-4, atol=1e-6,
                                jac_sparsity=jac)
            else:
                sol = solve_ivp(rhs, (0.0, T_FINAL), state0,
                                method="LSODA",
                                args=(beta,), rtol=1e-7, atol=1e-9)
            if not sol.success:
                raise RuntimeError(f"SIR failed at beta={beta}: {sol.message}")
            S_final = sol.y[:N, -1]
            I_final = sol.y[N:, -1]
            eq[k] = 1.0 - S_final - I_final
            if k % 10 == 0 or k == NSWEEP - 1:
                print(f"    SIR k={k}/{NSWEEP-1} beta={beta:.4f} "
                      f"R_mean={eq[k].mean():.3f} ({time.time()-t0:.1f}s)",
                      flush=True)
        print(f"  SIR done in {time.time()-t0:.1f}s, "
              f"R-range [{eq.min():.4f}, {eq.max():.4f}]")

        arr = np.column_stack([grid, eq])
        np.savetxt(HERE / f"{tag}_sir_equilibrium_data.csv", arr,
                   delimiter=",",
                   header="p," + ",".join(f"x_{i+1}" for i in range(N)),
                   comments="", fmt=["%.6f"] + ["%.6f"] * N)

        fn = HERE / f"{tag}_bifurcation_ranges.csv"
        with fn.open() as f:
            header = f.readline().rstrip("\n")
            rows = [line.rstrip("\n").split(",") for line in f if line.strip()]
        by_key = {r[0]: r for r in rows}
        by_key["SIR"] = ["SIR", f"{0.0:.6f}", f"{beta_max:.6f}",
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
