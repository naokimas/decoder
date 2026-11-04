# Transferable reconstruction of equilibria in network dynamics from sentinel nodes

Code for the paper

> N. Masuda, B. M. Eldo, and T. Bandara, *Transferable reconstruction of
> equilibria in network dynamics from sentinel nodes*.

The repository contains the code behind every figure, every table and every
number reported in the main text and in the Supplementary Information (SI),
together with the stored numerical results from which all figures and tables
can be redrawn in seconds.

---

## Contents

1. [Quick start](#1-quick-start)
2. [What is and is not in this repository](#2-what-is-and-is-not-in-this-repository)
3. [Input data](#3-input-data)
4. [Folder layout](#4-folder-layout)
5. [Which code produces which figure or table](#5-which-code-produces-which-figure-or-table)
6. [Description of every file](#6-description-of-every-file)
7. [Conventions used throughout](#7-conventions-used-throughout)
8. [Environment](#8-environment)
9. [License and citation](#9-license-and-citation)

---

## 1. Quick start

Install the dependencies (Python 3.9 or later):

```
pip install -r requirements.txt
```

Redraw a figure or rebuild a table from the stored results. No input data is
needed for this. For example:

```
cd fig4_fig5_cross_transfer
python3 plot_fig4_fig5.py            # writes figs/heatmap_rev.png and figs/ratio_rev.png
```

Check a sample of the numbers reported in the SI against the stored results:

```
python3 verify_si.py                 # prints each check; exits with status 1 on any mismatch
```

To recompute the stored results themselves, you need the equilibrium data
and the network edge lists described in [Section 3](#3-input-data).

Every script is run from its own folder (`cd` into the folder, then
`python3 <script>.py`). Every script begins with a banner that states which
figure, table or number it serves, what it writes, and how long it takes.

## 2. What is and is not in this repository

**Included**

* All code used for the results of the paper: the decoders, the
  sentinel-selection methods, the simulation of the equilibria, the analyses
  and the figure and table scripts. Code that was written along the way but
  does not contribute to any reported result is not included.
* The stored numerical results (the `data/` folders and `shared_data/`) that
  the figure and table scripts read, so that every figure and table can be
  redrawn without recomputing anything.
* The LaTeX bodies of the SI tables (the `tables/` folders), as written by
  the table scripts.
* The edge lists of the five model-generated networks used in the paper, and
  of the six Barabási–Albert networks of SI Section S9 (`networks/`).

**Not included, and why**

* **The equilibrium data.** These are about 1 GB of simulation output (one CSV
  per combination of dynamics and network), which is too large for a code
  repository. They are deposited at Zenodo; see the Data availability
  statement of the paper and [Section 3](#3-input-data).
* **The edge lists of the nine empirical networks** (dolphin, proximity,
  metabolic, euroroad, email, flybi, reactome, route_views, spanish). These
  data sets were collected and published by other researchers, and we do not
  redistribute them. They are available from the Netzschleuder repository
  (<https://networks.skewed.de>) or from the original sources cited in SI
  Section S17 (Table S24). [Section 3](#3-input-data) explains how to turn
  them into the edge lists the code expects.
* **The figure files.** They are identical to the figures in the paper, and
  every plotting script regenerates its figure into a `figs/` folder.

## 3. Input data

Only the scripts that recompute results need input data. The plotting and
table scripts, and `verify_si.py`, run on the stored results alone.

All input locations are defined in one module, `lib/paths.py`. By default the
code looks for a folder `input_data/` at the top of the repository:

```
input_data/
  equilibria_submitted/     equilibrium CSVs of the 16 dynamics of the main
                            text produced by equilibria/original16/
  equilibria_revision/      all other equilibrium CSVs: the dynamics LDW,
                            LMUT, ANM, AEUT1, AVEG1, MSIS2P and MUT2P, and
                            the subfolders
      ba_size_sweep/          Barabasi-Albert networks of SI Section S9
      hetero_replicates/      redrawn node parameters of SI Section S7
      doublewell_upper/       double well from the upper state, SI Section S5
  networks_empirical/       edge lists of the nine empirical networks
```

The first two folders are the contents of the Zenodo archive. The folder
names `equilibria_submitted` and `equilibria_revision` record which stage of
the project produced the data; together they are the data of the paper.

Instead of `input_data/`, you can point the code elsewhere with environment
variables:

```
export DECODER_DATA=/path/to/input_data                  # the root folder, or one at a time:
export DECODER_EQ_SUBMITTED=/path/to/equilibria_submitted
export DECODER_EQ_REVISION=/path/to/equilibria_revision
export DECODER_NETWORKS_EMPIRICAL=/path/to/networks_empirical
```

**Equilibrium CSV format.** One file per network and dynamics, named
`<network>_<dynamics>_equilibrium_data.csv`, with a header row
`p,x_1,...,x_N` and one row per control-parameter value (51 rows): the
control-parameter value followed by the equilibrium state of each node.

**Network edge lists.** One file per network, named
`<network>_network.edgelist`, with one undirected edge `i j` per line and
the nodes labeled 0, ..., N-1. To build the edge list of an empirical network
from its original data, do what `equilibria/original16/verify_networks.py`
does:

1. build an undirected graph, removing self-loops and duplicate edges;
2. keep the largest connected component;
3. relabel the nodes 0, ..., N-1 in increasing order of their original node
   IDs.

The resulting numbers of nodes and edges must match SI Table S24. The node
labels must match those used for the equilibrium data. Place the files in
`input_data/networks_empirical/`, which is outside the tracked part of the
repository (see `.gitignore`).

**Do not rerun the equilibrium simulations.** Producing the equilibrium data
took days of CPU time. The generators (`equilibria/` and the `generate_*.py`
and `gen_*.py` scripts in the section folders) are included to document
exactly how the data were made. Use the deposited data instead.

## 4. Folder layout

```
README.md, LICENSE, requirements.txt, .gitignore
verify_si.py                       checks numbers reported in the SI against the stored results

lib/                               modules shared by many scripts
  submitted/                         the decoder pipeline of the first version of this
                                     study, which the analyses reuse unchanged
equilibria/                        simulation of the equilibria of the 17 dynamics of the main text
  original16/                        16 of them
  ldw/                               the Laplacian double well (LDW)
networks/                          edge lists of the model-generated networks
shared_data/                       the 17 x 17 cross-transfer matrices that many scripts read

fig1_concept/                      main-text Fig. 1
fig2_fig3_example_reconstruction/  main-text Figs. 2 and 3
fig4_fig5_cross_transfer/          main-text Figs. 4 and 5
fig6_transfer_network/             main-text Fig. 6
fig7_decoder_rankings/             main-text Fig. 7

s01_nodewise_error/                SI Section S1
s02_effective_rank/                SI Section S2
...                                one folder per SI section
s17_networks/                      SI Section S17
```

There is no folder for SI Section S16, which is an analytical argument
without computation.

Inside a figure or section folder:

```
*.py      the scripts
data/     stored numerical results (.npz, .csv, .json)
tables/   LaTeX table bodies, as included in the SI
figs/     figures, created when a plotting script is run
```

A folder has only the subfolders it needs.

## 5. Which code produces which figure or table

The arrow `→` means "then run". Scripts marked *(input data)* need the input
data of [Section 3](#3-input-data); all other scripts in this table run on
the stored results.

### Main text

The main text has no tables.

| figure | folder | scripts |
|---|---|---|
| Fig. 1 | `fig1_concept` | `concept_fig.py` *(input data)* |
| Figs. 2, 3 | `fig2_fig3_example_reconstruction` | `compute_fig2.py`, `compute_fig3.py` *(input data)* → `plot_fig2_fig3.py` |
| Figs. 4, 5 | `fig4_fig5_cross_transfer` | `compute_matrices.py` *(input data)* → `plot_fig4_fig5.py` |
| Fig. 6 | `fig6_transfer_network` | `plot_wccnet.py` |
| Fig. 7 | `fig7_decoder_rankings` | `plot_rankings.py` (reads `s13_decoder_rankings/data/`) |

### Supplementary Information

| SI section | folder | SI figures | SI tables | scripts |
|---|---|---|---|---|
| S1 Node-wise reconstruction error | `s01_nodewise_error` | S1, S2 | – | `compute_nodewise.py` *(input data)* → `plot_nodewise.py` |
| S2 Effective rank | `s02_effective_rank` | – | S1, S2 | `compute_rank.py` *(input data)* → `make_tables17.py` |
| S3 Gappy POD | `s03_gappy_pod` | S3, S4 | S3, S4 | `compute_gappy.py` *(input data)* → `plot_gappy.py`, `make_tables.py` |
| S4 Interpolation versus extrapolation | `s04_interp_vs_extrap` | – | S5–S8 | the four `compute_*.py` *(input data)* → `make_tables17.py` |
| S5 Double well from the upper state | `s05_doublewell_upper` | S5, S6 | – | `generate_dwup.py` (do not rerun) → `compute_dw45.py` *(input data)* → `plot_dw45.py` |
| S6 Two-parameter families | `s06_two_parameter` | S7, S8 | – | `generate_*2p.py` or `sweep_row*.py` + `merge_rows*.py` (do not rerun) → `compute_2d45.py` *(input data)* → `plot_2d45.py` |
| S7 Variability over networks and random factors | `s07_variability` | S9, S10 | S9, S10 | `gen_hetero.py` (do not rerun) → `compute_repeat.py` *(input data)* → `summarize_repeat.py`, `make_tables.py` |
| S8 Measurement noise | `s08_measurement_noise` | S11, S12 | – | `compute_noise.py` *(input data)* → `plot_noise.py` |
| S9 Number of sentinel nodes | `s09_sentinel_count` | S13–S19 | – | see below |
| S10 Robustness of the cross-transfer network | `s10_transfer_network` | S20, S21 | S11–S13 | `analyze.py`, `plot_net.py`, `cluster.py`, `thresholds.py`, `per_network.py` |
| S11 Four additional dynamics | `s11_four_more_dynamics` | S22 | S14 | `generate_*.py` (do not rerun) → `compute_rand21.py` *(input data)* → `analyze21.py` |
| S12 Why the transferable dynamics transfer | `s12_why_transfer` | S23 | S15 | `compute_corr.py` *(input data)* → `fig_combined.py`, `make_table.py`, `auc.py` |
| S13 Source-decoder rankings | `s13_decoder_rankings` | – | S16–S19 | `compute_rank.py`, `compute_budget.py` *(input data)* → `make_tables.py` |
| S14 NN bias terms and ridge strength | `s14_bias_and_ridge` | S24, S25 | S20, S21 | `s13_decoder_rankings/compute_rank.py` and `compute_budget.py` with the arguments `bias` and `alpha01` *(input data)* → `plot_s14.py`, `make_tables_s14.py` |
| S15 Sentinel sets from network structure | `s15_structural_sentinels` | – | S22, S23 | `compute_sentsel.py`, `compute_struct.py` *(input data)* → `make_tables.py` |
| S16 Positivity and stability of the gLV equilibrium | – | – | – | analytical; no code |
| S17 Networks | `s17_networks` | – | S24 | `make_table.py` *(input data: networks)* |

SI Section S9 combines three experiments:

1. **Effective rank against network size** (Fig. S13): `compute_ranks14.py`
   *(input data)* → `plot_scale.py`.
2. **Sentinel count and size on Barabási–Albert networks** (Figs. S14–S18):
   `gen_ba.py` / `run_par.py` (do not rerun) → `compute_scale.py` and
   `diagnose_why.py` *(input data)* → `plot_scale.py`.
3. **Sentinel count on the 14 networks** (Fig. S19):
   `compute_csweep.py --nets=<list> --out=part_X.npz` *(input data)*, run once
   per group of networks, → `merge_plot.py`, which checks that every cell is
   covered by exactly 14 networks.

## 6. Description of every file

Run times are for one CPU core. "Module" means the file is imported by other
scripts and is not run on its own.

### Top level

| file | description |
|---|---|
| `README.md` | this file |
| `LICENSE` | MIT license |
| `requirements.txt` | Python packages and the versions used |
| `.gitignore` | files that are not tracked: generated figures, input data, working files |
| `verify_si.py` | re-derives a sample of the numbers reported in the SI from the stored results and compares them with the SI. Most of the numbers are printed in the SI text or tables; the Section S8 values and the Section S9 sentinel-count values are read off Figs. S11 and S19. Seconds. |

### `lib/` — shared modules

| file | description |
|---|---|
| `paths.py` | defines where the input data and the network edge lists are found ([Section 3](#3-input-data)); importing it also makes the other modules importable |
| `load_equilibria.py` | loads the equilibrium matrix of any of the 21 dynamics on any of the 14 networks; defines the lists of dynamics and networks and their display names |
| `splits.py` | the split of the 51 control-parameter values into 41 for training and calibration and 10 for evaluation; decoder training on a chosen set of rows; the calibrated prediction for a target dynamics |
| `splits2d.py` | the same for the 8 x 8 parameter grids of the two-parameter dynamics (SI Section S6) |
| `cellcolor.py` | chooses black or white text for each heat-map cell for best contrast |
| `dendro_style.py` | common style of the dendrograms of Figs. S21 and S22 |
| `fig6_style.py` | builds and draws the cross-transfer network of Fig. 6 and Fig. S20 |

### `lib/submitted/` — decoder pipeline of the first version of the study

These modules were written for the first version of this study and are
reused without change, so that the decoders, the sentinel-selection methods
and the random seeds are identical throughout. This repository uses only the
parts listed below, directly or through `lib/splits.py`; the files also
contain functions and command-line drivers of their own that are not used.
Several of the modules import one another, which is why all eight are
needed.

| file | what this repository uses it for |
|---|---|
| `pipeline_unified.py` | the MinCorr and QR sentinel selection; the shallow neural-network decoder with bias terms (SI Section S14); the training settings (hidden units, epochs, learning rate, random seed) |
| `test_zero_anchor.py` | the bias-free shallow neural network, the architecture of the main text's neural-network decoder (`lib/splits.py` trains it on the chosen rows), and the original bias-free training functions, against which `lib/splits.py` checks its own |
| `sweep_v22r_factorial.py` | running a trained decoder on new sentinel readings, and the original deployment of a decoder on a target dynamics, against which `lib/splits.py` checks its own deployment |
| `pipeline_v21.py` | the equilibrium loader for the 16 dynamics produced by `equilibria/original16/` |
| `pipeline_v22.py` | the same loader, as used by the scripts for Figs. 2 and 3 and SI Sections S1 and S8 |
| `pipeline_v22r.py` | decoder training with weight decay; imported by `sweep_v22r_factorial.py` and `test_zero_anchor.py` |
| `pipeline_linear.py` | the ridge-regression solver; imported by `pipeline_v21.py` |
| `build_v23_manuscript.py` | the table of equilibrium file names, used by two scripts of SI Section S9 |

### `equilibria/` — simulation of the equilibria (do not rerun)

| file | description |
|---|---|
| `equilibria/original16/README.txt` | detailed description of the scripts in `original16/`, the order in which they were run, and which script produced which CSV |
| `equilibria/original16/verify_networks.py` | builds the 14 network edge lists from the original network files (largest connected component, nodes relabeled 0, ..., N-1) and checks their sizes |
| `equilibria/original16/generate_data.py` | base generator: 12 dynamics on one network, with automatic trimming of the control-parameter range |
| `equilibria/original16/run_all_12_dynamics_v9.py` | module: right-hand sides, Jacobian patterns and sweep drivers for the 12 dynamics |
| `equilibria/original16/generate_msis2_msir2.py` | MSIS2 and MSIR2 |
| `equilibria/original16/generate_eut2_veg2.py` | EUT2 and VEG2 |
| `equilibria/original16/spanish_param_variants.py` | MSIS2, MSIR2, EUT2 and VEG2 on the spanish network |
| `equilibria/original16/regenerate_sis_dw_v3.py` | SIS and DW with network-rescaled control-parameter ranges |
| `equilibria/original16/regenerate_reactome_v3.py` | the same, for the reactome network only |
| `equilibria/original16/regenerate_sir_v3.py` | SIR |
| `equilibria/original16/regenerate_msis_v3.py` | MSIS (reverse warm start with adaptive widening of the range) |
| `equilibria/original16/regenerate_msir_v3.py` | MSIR (adaptive widening of the range) |
| `equilibria/original16/regenerate_mut_gen_wc_v3.py` | MUT, GEN and WC |
| `equilibria/original16/regenerate_gen_revwarm.py` | GEN by reverse warm start on all networks (used for route_views and spanish) |
| `equilibria/original16/regenerate_reactome_gen_revwarm.py` | GEN by reverse warm start on reactome |
| `equilibria/original16/regenerate_glv_eps.py` | gLV from its analytical coexistence equilibrium |
| `equilibria/original16/regenerate_wc_kbar_v89.py` | WC with the control-parameter range scaled by the mean degree |
| `equilibria/ldw/generate_ldw.py` | LDW, the Laplacian-coupled double well, on the 14 networks. Days. |

### `networks/` — model-generated networks

| file | description |
|---|---|
| `gkk_network.edgelist`, `lfr_network.edgelist`, `er_network.edgelist`, `ba_network.edgelist`, `hk_network.edgelist` | the five model-generated networks among the 14 networks of the paper (Goh–Kahng–Kim, LFR benchmark, Erdős–Rényi, Barabási–Albert, Holme–Kim) |
| `ba_size_sweep/ba1p5_network.edgelist` ... `ba4p0_network.edgelist` | the six Barabási–Albert networks of SI Section S9, with N = 10^1.5, ..., 10^4 nodes, written by `s09_sentinel_count/gen_ba.py --build` |
| `ba_size_sweep/_networks.csv` | size, number of edges, mean and maximum degree, largest eigenvalue and number of sentinels of those six networks |

### `shared_data/`

| file | description |
|---|---|
| `fig45_matrices.npz` | 17 x 17 source-by-target matrices of the n-RMSE and of the mean-prediction baseline, averaged over the 14 networks, for each of the six combinations of decoder and sentinel-selection method; written by `fig4_fig5_cross_transfer/compute_matrices.py` |
| `README.md` | which scripts read `fig45_matrices.npz` and what for |

### Main-text figures

| file | description |
|---|---|
| `fig1_concept/concept_fig.py` | Fig. 1, the overview of the method, drawn from the dolphin network and its SIS and WC equilibria *(input data)*. Seconds. |
| `fig2_fig3_example_reconstruction/compute_fig2.py` | data for Fig. 2: SIS decoder applied to SIS on the dolphin network *(input data)*. Seconds. |
| `fig2_fig3_example_reconstruction/compute_fig3.py` | data for Fig. 3: SIS and C11L1 decoders applied to WC on the dolphin network *(input data)*. Seconds. |
| `fig2_fig3_example_reconstruction/plot_fig2_fig3.py` | Figs. 2 and 3: true and reconstructed equilibria at 10 representative nodes |
| `fig4_fig5_cross_transfer/compute_matrices.py` | the cross-transfer matrices of Figs. 4 and 5, written to `shared_data/` *(input data)*. Hours. |
| `fig4_fig5_cross_transfer/plot_fig4_fig5.py` | Fig. 4 (n-RMSE) and Fig. 5 (ratio to the mean-prediction baseline) |
| `fig6_transfer_network/plot_wccnet.py` | Fig. 6, the cross-transfer network |
| `fig7_decoder_rankings/plot_rankings.py` | Fig. 7, the leading source decoders |

### SI Sections S1–S4

| file | description |
|---|---|
| `s01_nodewise_error/compute_nodewise.py` | node-wise n-RMSE for the four cases of Section S1 *(input data)*. Seconds. |
| `s01_nodewise_error/plot_nodewise.py` | Figs. S1 and S2 |
| `s02_effective_rank/compute_rank.py` | singular values and the four effective ranks of every equilibrium matrix *(input data)*. Seconds. |
| `s02_effective_rank/make_tables17.py` | Tables S1 and S2 |
| `s03_gappy_pod/gappy.py` | module: gappy proper orthogonal decomposition (POD) used in place of a decoder |
| `s03_gappy_pod/compute_gappy.py` | cross-transfer matrices for gappy POD *(input data)*. About half an hour. |
| `s03_gappy_pod/plot_gappy.py` | Figs. S3 and S4 |
| `s03_gappy_pod/make_tables.py` | Tables S3 and S4 |
| `s04_interp_vs_extrap/compute_oos.py` | reconstruction with interspersed and contiguous training sets, for Tables S5 and S6 *(input data)*. About an hour. |
| `s04_interp_vs_extrap/compute_grid9.py` | the nine combinations of source training split and target calibration split, for Table S7 *(input data)*. Hours. |
| `s04_interp_vs_extrap/compute_base_grid.py` | the mean-prediction baselines for the same nine combinations *(input data)*. Under an hour. |
| `s04_interp_vs_extrap/compute_tfrac.py` | dependence on the size of the target calibration set, for Table S8 *(input data)*. Hours. |
| `s04_interp_vs_extrap/make_tables17.py` | Tables S5 to S8, and the numbers quoted in the text of Section S4 |

### SI Sections S5–S8

| file | description |
|---|---|
| `s05_doublewell_upper/generate_dwup.py` | equilibria of the double well started from the upper state (do not rerun). Hours. |
| `s05_doublewell_upper/compute_dw45.py` | cross-transfer matrices with the upper-state double well added *(input data)*. About an hour. |
| `s05_doublewell_upper/plot_dw45.py` | Figs. S5 and S6 |
| `s06_two_parameter/msis2p.py` | module: the MSIS dynamics as a two-parameter family (MSIS2P) |
| `s06_two_parameter/mut2p.py` | module: the mutualistic dynamics as a two-parameter family (MUT2P) |
| `s06_two_parameter/generate_msis2p.py`, `generate_mut2p.py` | equilibria of MSIS2P and MUT2P on an 8 x 8 parameter grid, in one process (do not rerun). Days. |
| `s06_two_parameter/sweep_row.py`, `sweep_row_mut.py` | one row of the same grid, for running the rows in parallel (do not rerun) |
| `s06_two_parameter/merge_rows.py`, `merge_rows_mut.py` | assemble the rows into the equilibrium CSVs |
| `s06_two_parameter/compute_2d45.py` | cross-transfer matrices with MSIS2P and MUT2P added *(input data)*. About an hour. |
| `s06_two_parameter/plot_2d45.py` | Figs. S7 and S8 |
| `s07_variability/gen_hetero.py` | equilibria of five dynamics with redrawn heterogeneous node parameters (do not rerun). Hours. |
| `s07_variability/compute_repeat.py` | repeats the cross-transfer experiment under each source of randomness *(input data)*. Many hours. |
| `s07_variability/summarize_repeat.py` | Figs. S9 and S10 |
| `s07_variability/make_tables.py` | Tables S9 and S10 |
| `s08_measurement_noise/compute_noise.py` | reconstruction from noisy observations, for the four cases of Section S1 *(input data)*. About a minute. |
| `s08_measurement_noise/plot_noise.py` | Figs. S11 and S12 |

### SI Sections S9–S11

| file | description |
|---|---|
| `s09_sentinel_count/compute_ranks14.py` | effective ranks of the equilibrium matrices of the 14 networks *(input data)*. Seconds. |
| `s09_sentinel_count/gen_ba.py` | builds the six Barabási–Albert networks and computes the equilibria of the 17 dynamics on them (do not rerun). Days. |
| `s09_sentinel_count/run_par.py` | runs the steps of `gen_ba.py` in parallel (do not rerun) |
| `s09_sentinel_count/compute_scale.py` | n-RMSE against the number of sentinels and the network size on the Barabási–Albert networks *(input data)*. Minutes. |
| `s09_sentinel_count/diagnose_why.py` | quantities that explain why the n-RMSE falls with the network size *(input data)*. Seconds. |
| `s09_sentinel_count/plot_scale.py` | Figs. S13 to S18 |
| `s09_sentinel_count/compute_csweep.py` | n-RMSE against the number of sentinels on the 14 networks, one group of networks per run *(input data)*. About two hours in all. |
| `s09_sentinel_count/merge_plot.py` | merges the runs of `compute_csweep.py` and draws Fig. S19 |
| `s10_transfer_network/analyze.py` | Table S11: components, edge density and reciprocity of the cross-transfer networks |
| `s10_transfer_network/plot_net.py` | Fig. S20, the cross-transfer networks for all six combinations of decoder and sentinel-selection method |
| `s10_transfer_network/cluster.py` | Fig. S21, hierarchical clustering of the cross-transfer errors |
| `s10_transfer_network/thresholds.py` | Table S12, robustness to the two edge thresholds |
| `s10_transfer_network/per_network.py` | Table S13, the component structure network by network |
| `s11_four_more_dynamics/generate_anm.py`, `generate_aeut.py`, `generate_aveg.py`, `generate_lmut.py` | equilibria of ANM, AEUT1, AVEG1 and LMUT (do not rerun). Days. |
| `s11_four_more_dynamics/compute_rand21.py` | cross-transfer matrices over the 21 dynamics with random sentinels *(input data)*. About an hour. |
| `s11_four_more_dynamics/analyze21.py` | Fig. S22 and Table S14 |

### SI Sections S12–S17

| file | description |
|---|---|
| `s12_why_transfer/compute_corr.py` | correlations between the equilibrium states of different dynamics and with the node degree; prints the block means quoted in Section S12 *(input data)*. Seconds. |
| `s12_why_transfer/fig_combined.py` | Fig. S23 |
| `s12_why_transfer/make_table.py` | Table S15 |
| `s12_why_transfer/auc.py` | the AUC values quoted in Section S12 |
| `s13_decoder_rankings/compute_rank.py` | performance of every source decoder, including the pooled decoders C11, C17, C11L1 and C17L1. With the argument `bias` or `alpha01`, the variants of Section S14 *(input data)*. Hours. |
| `s13_decoder_rankings/compute_budget.py` | the pooled decoders trained on the same amount of data as a single-dynamics decoder (C11B, C11L1B, C17B, C17L1B), also with `bias` or `alpha01` *(input data)*. About an hour. |
| `s13_decoder_rankings/make_tables.py` | Tables S16 to S19, and the values for the budget-matched decoders quoted in Section S13 |
| `s14_bias_and_ridge/plot_s14.py` | Figs. S24 and S25 |
| `s14_bias_and_ridge/make_tables_s14.py` | Tables S20 and S21 |
| `s15_structural_sentinels/compute_sentsel.py` | structural statistics of the MinCorr and QR sentinel sets compared with random sets *(input data)*. About ten minutes. |
| `s15_structural_sentinels/compute_struct.py` | decoders with sentinels chosen by degree or by betweenness *(input data)*. About ten minutes. |
| `s15_structural_sentinels/make_tables.py` | Tables S22 and S23 |
| `s17_networks/make_table.py` | Table S24, the statistics of the 14 networks *(input data: networks)* |

### Stored results (`data/` folders)

Each `data/` folder holds the results that its section's plotting and table
scripts read, written by that section's `compute_*.py` scripts. The file
names are given in the banner of the script that writes them.

## 7. Conventions used throughout

* **Data matrix.** The paper writes the equilibrium data as an N x M matrix
  (row: node; column: control-parameter value). The code stores its
  transpose, an M x N matrix with one row per control-parameter value, as in
  the CSV files. A row index in the code is therefore a column index in the
  paper.
* **Split.** 41 of the 51 control-parameter values, interspersed over the
  sweep, are used to train the decoder and to calibrate it on the target
  dynamics; the error is evaluated on the remaining 10. The same split is
  used on the source and the target side.
* **n-RMSE.** The root-mean-square error over the evaluation values and all
  nodes, divided by the range of the target equilibria over all 51 values and
  all nodes.
* **Ratio r(S→T).** The n-RMSE of the decoder divided by the n-RMSE of the
  mean-prediction baseline, which predicts every node by the mean of the
  sentinel readings at the same control-parameter value.
* **Sentinels.** n = floor(ln N) nodes unless stated otherwise, chosen by
  MinCorr, by QR pivoting, or uniformly at random.
* **Decoders.** A shallow neural network without bias terms, and a linear
  ridge decoder with ridge parameter 0.01, unless stated otherwise
  (SI Section S14).
* **Threads.** Every script limits the numerical libraries to one thread,
  so that several scripts can run at the same time.

## 8. Environment

Python 3.9 with the packages in `requirements.txt` (NumPy, SciPy, NetworkX,
Matplotlib, Pillow and PyTorch). No GPU is used. The results were produced on
macOS; numerical results on other platforms may differ in the last digits.

## 9. License and citation

The code is released under the MIT License (see `LICENSE`). If you use it,
please cite the paper above.
