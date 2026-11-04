"""Build the v23 manuscript assets.

v23 = v22 + (i) Option B (no-bias decoder retrain) is the canonical
cross-transfer; (ii) panels reordered: MinCorr first, QR second, random
third; (iii) Fig.~1 and Fig.~2 redone with NN + Option B + MinCorr and
a shared y-axis across all per-node panels.

Inputs on disk:
  - <tag>_v23_cross_matrix_data_<method>.npz        (NN, Option B)
  - <tag>_v23_linear_cross_matrix_data_<method>.npz (linear, Option B)

Outputs:
  - heatmap_v23_combined.png
  - coverage_v23_C3.png
  - coverage_v23_C2.png
  - covering_v23_chains.tex
  - sis_sis_dolphin_v23.png
  - sis_wc_dolphin_v23.png
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib.colors import ListedColormap

import pipeline_unified as pu
import pipeline_v22 as p22
import test_zero_anchor as tz

HERE = Path(__file__).resolve().parent

NETWORKS = p22.NETWORKS

SYSTEMS_INT = ["SIS", "SIR", "MSIS", "MSIR", "MSIS2", "MSIR2",
               "DW", "MUT", "GEN", "WC", "gLV", "NM",
               "EUT", "VEG", "EUT2", "VEG2"]
SYSTEMS_FIG23 = ["SIS", "SIR", "MSIS", "MSIS2", "MSIR", "MSIR2",
                 "DW", "MUT", "GEN", "WC", "gLV", "NM",
                 "EUT", "EUT2", "VEG", "VEG2"]
SOURCES_INT = SYSTEMS_INT + ["COMBINED"]

DISPLAY = {"MSIS": "MSIS1", "MSIR": "MSIR1", "EUT": "EUT1", "VEG": "VEG1"}
def disp(n):
    return DISPLAY.get(n, n)

SYSTEMS_DISP = [disp(s) for s in SYSTEMS_INT]
SYSTEMS_FIG23_DISP = [disp(s) for s in SYSTEMS_FIG23]
SOURCES_DISP = [disp(s) for s in SOURCES_INT]

CSV_NAMES = {"SIS": "sis", "DW": "doublewell", "MUT": "mutualistic",
             "GEN": "genereg", "WC": "wc", "gLV": "glv",
             "SIR": "sir", "NM": "noymeir",
             "EUT": "eutroph", "VEG": "vegturb",
             "MSIS": "msis", "MSIR": "msir",
             "MSIS2": "msis2", "MSIR2": "msir2",
             "EUT2": "eutroph2", "VEG2": "vegturb2"}

# Reordered: MinCorr first, QR second, random third.
PANELS = [
    ("a", "shallow, MinCorr", "nn",  "minc"),
    ("b", "shallow, QR",      "nn",  "qr"),
    ("c", "shallow, random",  "nn",  "random"),
    ("d", "linear, MinCorr",  "lin", "minc"),
    ("e", "linear, QR",       "lin", "qr"),
    ("f", "linear, random",   "lin", "random"),
]


def load_matrix(tag, decoder, method, target_order):
    suffix = "" if decoder == "nn" else "_linear"
    fn = HERE / f"{tag}_v23{suffix}_cross_matrix_data_{method}.npz"
    if not fn.exists():
        return None
    d = np.load(fn)
    s = list(d["sources"]); t = list(d["targets"]); rel = d["rel"]
    out = np.full((len(SOURCES_INT), len(target_order)), np.nan)
    for i, src in enumerate(SOURCES_INT):
        if src in s:
            si = s.index(src)
            for j, tgt in enumerate(target_order):
                if tgt in t:
                    tj = t.index(tgt)
                    out[i, j] = rel[si, tj]
    return out


def mean_matrix(decoder, method, target_order=None):
    if target_order is None:
        target_order = SYSTEMS_INT
    Ms = []
    for tag in NETWORKS:
        M = load_matrix(tag, decoder, method, target_order)
        if M is None: continue
        Ms.append(M)
    if not Ms:
        return None
    return np.nanmean(np.stack(Ms, axis=0), axis=0)


def plot_heatmap(out_fname):
    TICK_FS, LABEL_FS, TITLE_FS, CELL_FS = 9, 12, 12, 6.5
    VMIN, VMAX = 0.0, 0.5
    fig, axes = plt.subplots(2, 3, figsize=(20.0, 14.5))
    last_im = None
    for k, (lett, title, dec, meth) in enumerate(PANELS):
        r, c = k // 3, k % 3
        ax = axes[r, c]
        M = mean_matrix(dec, meth)
        im = ax.imshow(M * 100.0, vmin=VMIN * 100, vmax=VMAX * 100,
                       cmap="viridis", aspect="auto")
        last_im = im
        ax.set_xticks(range(len(SYSTEMS_DISP)))
        ax.set_xticklabels(SYSTEMS_DISP, rotation=45, fontsize=TICK_FS)
        ax.set_yticks(range(len(SOURCES_DISP)))
        ax.set_yticklabels(SOURCES_DISP, fontsize=TICK_FS)
        ax.set_xlabel("target dynamics", fontsize=LABEL_FS)
        ax.set_ylabel("source decoder", fontsize=LABEL_FS)
        ax.set_title(f"({lett}) {title}", fontsize=TITLE_FS)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j] * 100.0
                if not np.isfinite(v):
                    ax.text(j, i, "--", ha="center", va="center",
                            fontsize=CELL_FS, color="gray")
                    continue
                color = "white" if (v < (VMIN + VMAX) * 50.0) else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=CELL_FS, color=color)
    fig.tight_layout(rect=[0, 0, 0.96, 1.0])
    cbar_ax = fig.add_axes([0.965, 0.10, 0.012, 0.80])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label(r"RMSE/range (\%)", fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    fig.savefig(HERE / out_fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_fname}")


def min_cover_chain(M):
    n_src, n_tgt = M.shape
    finite_vals = M[np.isfinite(M)]
    if not finite_vals.size:
        return []
    taus = sorted(set(finite_vals.tolist()))
    coverable = np.where(np.any(np.isfinite(M), axis=0))[0]
    tgt_full = 0
    for j in coverable:
        tgt_full |= (1 << j)
    last_size = None; chain = []
    for tau in taus:
        adj = (M <= tau) & np.isfinite(M)
        if not adj[:, coverable].any(axis=0).all():
            continue
        sets_per = np.zeros(n_src, dtype=np.int64)
        for s in range(n_src):
            bits = 0
            for t in range(n_tgt):
                if adj[s, t]:
                    bits |= (1 << t)
            sets_per[s] = bits
        min_size, min_combo = None, None
        for k in range(1, n_src + 1):
            for combo in combinations(range(n_src), k):
                covered = 0
                for s in combo:
                    covered |= int(sets_per[s])
                if (covered & tgt_full) == tgt_full:
                    min_size, min_combo = k, combo; break
            if min_size is not None: break
        if last_size is None or min_size < last_size:
            chain.append((tau, min_size, list(min_combo)))
            last_size = min_size
        if min_size == 1: break
    return chain


def get_step(chain, k):
    for tau, kk, si in chain:
        if kk == k:
            return tau, si
    return None


def plot_coverage_figure(target_k, fname):
    TICK_FS, LABEL_FS, TITLE_FS, CELL_FS = 10, 12, 12, 9
    cmap = ListedColormap(["white", "#87CEEB"])
    fig, axes = plt.subplots(2, 3, figsize=(20.0, 7.0 + 0.5 * target_k))
    for k_panel, (lett, title, dec, meth) in enumerate(PANELS):
        r, c = k_panel // 3, k_panel % 3
        ax = axes[r, c]
        M_sec3 = mean_matrix(dec, meth, target_order=SYSTEMS_INT)
        chain = min_cover_chain(M_sec3)
        step = get_step(chain, target_k)
        if step is None:
            ax.set_title(f"({lett}) {title}", fontsize=TITLE_FS)
            ax.axis("off"); continue
        tau, src_idx = step
        M_fig = mean_matrix(dec, meth, target_order=SYSTEMS_FIG23)
        sub_M = M_fig[src_idx, :]
        is_covered = ((sub_M <= tau) & np.isfinite(sub_M)).astype(int)
        ax.imshow(is_covered, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(SYSTEMS_FIG23_DISP)))
        ax.set_xticklabels(SYSTEMS_FIG23_DISP, rotation=45, fontsize=TICK_FS)
        ax.set_yticks(range(len(src_idx)))
        ax.set_yticklabels([SOURCES_DISP[s] for s in src_idx],
                           fontsize=TICK_FS)
        ax.set_xlabel("target dynamics", fontsize=LABEL_FS)
        ax.set_ylabel("source decoder", fontsize=LABEL_FS)
        ax.set_title(f"({lett}) {title}  ($\\tau = {tau*100:.2f}\\%$)",
                     fontsize=TITLE_FS)
        for i in range(sub_M.shape[0]):
            for j in range(sub_M.shape[1]):
                v = sub_M[i, j] * 100.0
                if not np.isfinite(v):
                    ax.text(j, i, "--", ha="center", va="center",
                            fontsize=CELL_FS, color="gray")
                    continue
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=CELL_FS, color="black")
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}")


def latex_master_table():
    parts = []
    parts.append(r"\begin{table}[H]")
    parts.append(r"\centering\small")
    parts.append(r"\caption{Minimum covering-set chain for each of the six "
                 r"(decoder, sentinel-method) panels of "
                 r"Fig.~\ref{fig:heat-combined} under the v23 universe "
                 r"($16$ dynamics, $14$ networks; Option~B no-bias decoder "
                 r"retrain). Each row is a step where $|\mathcal{C}|$ "
                 r"strictly drops as $\tau$ increases. The chain ends at "
                 r"$|\mathcal{C}|=1$.}\label{tab:cover-chain}")
    parts.append(r"\begin{tabular}{l r r r p{0.50\textwidth}}")
    parts.append(r"\toprule")
    parts.append(r"panel & step & $\tau$ (\%) & $|\mathcal{C}|$ & "
                 r"covering set $\mathcal{C}$ \\")
    parts.append(r"\midrule")
    for p_idx, (lett, title, dec, meth) in enumerate(PANELS):
        M = mean_matrix(dec, meth)
        chain = min_cover_chain(M)
        for i, (tau, k, si) in enumerate(chain, start=1):
            srcs = ", ".join(SOURCES_DISP[s] for s in si)
            label = f"({lett}) {title}" if i == 1 else ""
            parts.append(f"{label} & {i} & {tau*100:.2f} & {k} & "
                         + r"\{" + srcs + r"\} \\")
        if p_idx < len(PANELS) - 1:
            parts.append(r"\midrule")
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\end{table}")
    return "\n".join(parts)


# ---------- Fig 1 and Fig 2: NN + Option B + MinCorr, shared y-axis ----------
def build_sis_diagnostic_v23(target_tag, out_png, title_str):
    """Train a no-bias shallow NN on dolphin SIS with MinCorr sentinels,
    deploy on `target_tag` on dolphin via v22 fair-affine, plot per-node
    grid true vs predicted with a SHARED y-axis across all panels
    (global Y_target min/max)."""
    tag = "dolphin"
    G = nx.read_edgelist(HERE / f"{tag}_network.edgelist", nodetype=int)
    N = len(G.nodes()); K = int(np.floor(np.log(N)))
    Y_sis = np.loadtxt(HERE / f"{tag}_sis_equilibrium_data.csv",
                       delimiter=",", skiprows=1)[:, 1:]
    # MinCorr sentinel indices on SIS training split.
    rng = np.random.default_rng(pu.SEED)
    M = Y_sis.shape[0]; idx = np.arange(M); rng.shuffle(idx)
    n_test = int(round(pu.TEST_FRACTION * M))
    Y_train_pick = Y_sis[np.sort(idx[n_test:])]
    sensor_idx = pu.min_corr_sensors(Y_train_pick, K)
    # No-bias NN decoder (Option B) at wd=0.
    ckpt = tz.train_nn_no_bias(Y_sis, sensor_idx, weight_decay=0.0)
    arr_t = np.loadtxt(HERE / f"{tag}_{CSV_NAMES[target_tag]}_equilibrium_data.csv",
                       delimiter=",", skiprows=1)
    ps = arr_t[:, 0]; Y_tgt = arr_t[:, 1:]
    # v22 fair-affine cross-transfer on the no-bias decoder.
    model = tz.ShallowDecoderNoBias(ckpt["k_sentinels"], ckpt["hidden"],
                                    ckpt["n_nodes"])
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    def decode_fn(Xn):
        with torch.no_grad():
            return model(torch.tensor(Xn, dtype=torch.float32)).numpy()
    Y_pred, a, b = p22._fair_v22_core(ckpt, Y_tgt, decode_fn)
    rel = float(np.sqrt(np.mean((Y_pred - Y_tgt) ** 2))
                / (Y_tgt.max() - Y_tgt.min()))
    # Shared y-axis: global Y_tgt min/max.
    y_lim = (float(Y_tgt.min()), float(Y_tgt.max()))
    cols = 9
    rows = int(np.ceil(N / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 1.6 * rows),
                             squeeze=False)
    for n in range(rows * cols):
        r, c = n // cols, n % cols
        ax = axes[r, c]
        if n >= N:
            ax.axis("off"); continue
        ax.plot(ps, Y_tgt[:, n], "-", color="black", lw=1.0, label="true")
        ax.plot(ps, Y_pred[:, n], "--", color="red", lw=1.0, label="predicted")
        ax.set_title(f"node {n+1}", fontsize=7)
        ax.set_ylim(y_lim)
        ax.tick_params(labelsize=5)
        ax.grid(alpha=0.3, linewidth=0.4)
        if n == 0:
            ax.legend(loc="best", fontsize=5)
    fig.suptitle(title_str + f"  RMSE/range = {rel * 100:.2f}\\%",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}  (RMSE/range = {rel*100:.2f}%)")


def report_cover_chains():
    print("\n=== v23 cover chains (informational) ===")
    for lett, title, dec, meth in PANELS:
        M = mean_matrix(dec, meth)
        chain = min_cover_chain(M)
        c3 = get_step(chain, 3); c2 = get_step(chain, 2)
        c3_str = (f"|C|=3 tau={c3[0]*100:.2f}%, "
                  f"{[SOURCES_DISP[s] for s in c3[1]]}") if c3 else "no |C|=3"
        c2_str = (f"|C|=2 tau={c2[0]*100:.2f}%, "
                  f"{[SOURCES_DISP[s] for s in c2[1]]}") if c2 else "no |C|=2"
        print(f"({lett}) {title:25s}  {c3_str}  ;  {c2_str}")


def main():
    plot_heatmap("heatmap_v23_combined.png")
    plot_coverage_figure(3, "coverage_v23_C3.png")
    plot_coverage_figure(2, "coverage_v23_C2.png")
    (HERE / "covering_v23_chains.tex").write_text(latex_master_table())
    print("wrote covering_v23_chains.tex")
    build_sis_diagnostic_v23("SIS", "sis_sis_dolphin_v23.png",
        "Within-dynamics test on dolphin: SIS-source no-bias shallow NN decoder "
        "with MinCorr sentinels, deployed on the SIS target.")
    build_sis_diagnostic_v23("WC", "sis_wc_dolphin_v23.png",
        "Cross-dynamics deployment on dolphin: SIS-source no-bias shallow NN "
        "decoder with MinCorr sentinels, deployed on the WC target via the "
        "fair-affine renormalization.")
    report_cover_chains()


if __name__ == "__main__":
    main()
