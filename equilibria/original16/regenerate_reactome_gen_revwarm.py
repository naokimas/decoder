"""Reactome GEN regen with REVERSE warm-start, on the alpha_max-anchored
D range D in [0, 11.510/alpha_max]. The sweep is computed in descending
order (largest D first, with a high initial condition x_i=10, so the
upper branch is found), and each subsequent (smaller) D is warm-started
from the previous (larger D) equilibrium. After the sweep is done we
reorder the results to ascending D so that the saved CSV has the same
column ordering as the other dynamics. This avoids the warm-start trap
at x=0 that would otherwise propagate from D=0 forward."""
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import networkx as nx
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
NSWEEP = 51
RTOL, ATOL = 1e-4, 1e-6
T_FINAL = 200.0
ALPHA_DOLPHIN = 7.1942
GEN_CONSTANT = 1.6 * ALPHA_DOLPHIN   # 11.510


def main():
    tag = "reactome"
    G = nx.read_edgelist(HERE / f"{tag}_network.edgelist", nodetype=int)
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")
    N = A.shape[0]; M = A.nnz // 2
    kbar = 2.0 * M / N
    alpha_max = float(eigsh(A, k=1, which="LA",
                            return_eigenvectors=False)[0])
    A_plus_I = A + sp.eye(N)
    D_max = GEN_CONSTANT / alpha_max
    print(f"  N={N}, M={M}, <k>={kbar:.3f}, alpha_max={alpha_max:.3f}")
    print(f"  D_max = {GEN_CONSTANT:.3f} / {alpha_max:.3f} = {D_max:.6f}")

    B, F_, H_ = 1.0, 1.0, 2.0
    def rhs(t, x, D):
        x_pos = np.maximum(x, 0.0)
        hill = (x_pos ** H_) / (1.0 + x_pos ** H_)
        return -B * (x_pos ** F_) + D * (A @ hill)

    grid_asc = np.linspace(0.0, D_max, NSWEEP)
    eq = np.empty((NSWEEP, N))

    # Reverse warm-start: descending D, from x_init=10 at the top.
    x_prev = np.full(N, 10.0)
    t0 = time.time()
    for k in range(NSWEEP - 1, -1, -1):
        D = grid_asc[k]
        sol = solve_ivp(rhs, (0.0, T_FINAL), x_prev, method="BDF",
                        args=(D,), rtol=RTOL, atol=ATOL,
                        jac_sparsity=A_plus_I)
        if not sol.success:
            raise RuntimeError(f"failed at D={D}: {sol.message}")
        eq[k] = sol.y[:, -1]
        x_prev = eq[k].copy()
        if k % 5 == 0 or k == NSWEEP - 1:
            print(f"    GEN k={k}/{NSWEEP-1} D={D:.4f} y={eq[k].mean():.3f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    print(f"\nGEN done in {time.time()-t0:.1f}s, "
          f"y-range [{eq.min():.4f}, {eq.max():.4f}]")

    # Save in ascending-D order (eq is already indexed by ascending k = 0..50).
    arr = np.column_stack([grid_asc, eq])
    np.savetxt(HERE / f"{tag}_genereg_equilibrium_data.csv", arr,
               delimiter=",",
               header="p," + ",".join(f"x_{i+1}" for i in range(N)),
               comments="", fmt=["%.6f"] + ["%.6f"] * N)

    fn = HERE / f"{tag}_bifurcation_ranges.csv"
    with fn.open() as f:
        header = f.readline().rstrip("\n")
        rows = [line.rstrip("\n").split(",") for line in f if line.strip()]
    by_key = {r[0]: r for r in rows}
    by_key["GEN"] = ["GEN", "0.000000", f"{D_max:.6f}", f"{NSWEEP:d}",
                     f"{eq.min():.6f}", f"{eq.max():.6f}"]
    KEYS = ["SIS", "DW", "MUT", "GEN", "WC", "gLV", "SIR", "NM",
            "EUT", "VEG", "MSIS", "MSIR"]
    out = [header]
    for k in KEYS:
        if k in by_key:
            out.append(",".join(by_key[k]))
    fn.write_text("\n".join(out) + "\n")
    print(f"  wrote {fn.name}")


if __name__ == "__main__":
    main()
