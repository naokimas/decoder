"""MSIS reverse-warm-start regen for all 12 networks. beta range matches
SIS: beta in [0, 1/(0.15*alpha_max)]. At the largest beta we start with
the IC rho_S(0)=0.9, rho_I(0)=0.1 for each node, find the equilibrium,
then warm-start backward through smaller beta. Saved CSVs are in
ascending-beta order."""
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


def make_msis_rhs(A, k_vec):
    MU, D_S, D_I = 1.0, 1.0, 1.0
    def rhs(t, state, beta):
        N = state.size // 2
        rS = state[:N]; rI = state[N:]
        diff_S = D_S * (A @ (rS / k_vec) - rS)
        diff_I = D_I * (A @ (rI / k_vec) - rI)
        infect = beta * rS * rI
        recov = MU * rI
        return np.concatenate([-infect + recov + diff_S,
                                infect - recov + diff_I])
    return rhs


def msis_jac_pattern(A):
    """Sparsity pattern for MSIS Jacobian (state [rho_S, rho_I],
    each block N x N): J = [[A+I, A+I], [A+I, A+I]]."""
    N = A.shape[0]
    A_plus_I = A + sp.eye(N)
    return sp.bmat([[A_plus_I, A_plus_I],
                    [A_plus_I, A_plus_I]], format="csr")


def main():
    for tag in NETWORKS:
        print(f"\n========== {tag} ==========")
        N, A, k_vec, alpha_max = load_network(tag)
        beta_max = 1.0 / (SIS_PERCENTILE * alpha_max)
        large = tag in LARGE_NETS
        jac = msis_jac_pattern(A) if large else None
        rhs = make_msis_rhs(A, k_vec)

        # Adaptive beta_max widening (Methods sec:autotrim): the SIS-matched
        # range can lie below the metapopulation endemic threshold. Probe at
        # beta_max/2 from the reverse-warm IC (rho_S=0.9, rho_I=0.1); if the
        # mean rho_I at t_max is below THRESHOLD, double beta_max and retry.
        probe_ic = np.concatenate([np.full(N, 0.9), np.full(N, 0.1)])
        for _trial in range(20):
            beta_test = 0.5 * beta_max
            if large:
                sp_ = solve_ivp(rhs, (0.0, T_FINAL), probe_ic, method="BDF",
                                args=(beta_test,), rtol=1e-4, atol=1e-6,
                                jac_sparsity=jac)
            else:
                sp_ = solve_ivp(rhs, (0.0, T_FINAL), probe_ic, method="LSODA",
                                args=(beta_test,), rtol=1e-7, atol=1e-9)
            mean_rI = float(sp_.y[N:, -1].mean())
            print(f"  widen trial={_trial} beta_max={beta_max:.4f} "
                  f"beta_test={beta_test:.4f} mean_rho_I={mean_rI:.4f}",
                  flush=True)
            if mean_rI >= THRESHOLD:
                break
            beta_max *= 2.0
        else:
            print(f"  WARN: widening did not reach threshold; "
                  f"using beta_max={beta_max:.4f}")

        grid = np.linspace(0.0, beta_max, NSWEEP)
        print(f"  N={N}, alpha_max={alpha_max:.3f}, "
              f"beta_max={beta_max:.4f}")
        eq = np.empty((NSWEEP, N))

        # IC at largest beta: rho_S=0.9, rho_I=0.1
        rS = np.full(N, 0.9); rI = np.full(N, 0.1)
        state_prev = np.concatenate([rS, rI])

        t0 = time.time()
        for k in range(NSWEEP - 1, -1, -1):
            beta = grid[k]
            if large:
                sol = solve_ivp(rhs, (0.0, T_FINAL), state_prev, method="BDF",
                                args=(beta,), rtol=1e-4, atol=1e-6,
                                jac_sparsity=jac)
            else:
                sol = solve_ivp(rhs, (0.0, T_FINAL), state_prev,
                                method="LSODA",
                                args=(beta,), rtol=1e-7, atol=1e-9)
            if not sol.success:
                raise RuntimeError(f"MSIS failed at beta={beta}: "
                                   f"{sol.message}")
            state_prev = sol.y[:, -1].copy()
            eq[k] = state_prev[N:]  # rho_I
            if k % 10 == 0 or k == NSWEEP - 1:
                print(f"    MSIS k={k}/{NSWEEP-1} beta={beta:.4f} "
                      f"rho_I_mean={eq[k].mean():.3f} "
                      f"({time.time()-t0:.1f}s)", flush=True)
        print(f"  MSIS done in {time.time()-t0:.1f}s, "
              f"rho_I-range [{eq.min():.4f}, {eq.max():.4f}]")

        arr = np.column_stack([grid, eq])
        np.savetxt(HERE / f"{tag}_msis_equilibrium_data.csv", arr,
                   delimiter=",",
                   header="p," + ",".join(f"x_{i+1}" for i in range(N)),
                   comments="", fmt=["%.6f"] + ["%.6f"] * N)

        fn = HERE / f"{tag}_bifurcation_ranges.csv"
        with fn.open() as f:
            header = f.readline().rstrip("\n")
            rows = [line.rstrip("\n").split(",") for line in f if line.strip()]
        by_key = {r[0]: r for r in rows}
        by_key["MSIS"] = ["MSIS", f"{0.0:.6f}", f"{beta_max:.6f}",
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
