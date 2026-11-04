# Shared inputs

`fig45_matrices.npz` holds the 17 x 17 source-target matrices of the n-RMSE
and of the mean-prediction baseline, averaged over the 14 networks, for each
of the six combinations of decoder architecture and sentinel-selection
method. It is the backbone of Figs. 4 and 5 of the main text, and nine files
read it:

| reader | what it uses the matrices for |
|---|---|
| `fig4_fig5_cross_transfer/plot_fig4_fig5.py` | Figs. 4 and 5 themselves |
| `s03_gappy_pod/plot_gappy.py` | Fig. S4, the decoder-over-gappy ratio |
| `s03_gappy_pod/make_tables.py` | the decoder rows of Table S3, and Table S4 |
| `s10_transfer_network/analyze.py` | Table S11 |
| `s10_transfer_network/thresholds.py` | Table S12 |
| `s10_transfer_network/cluster.py` | Fig. S21 |
| `lib/fig6_style.py` | Fig. 6 and Fig. S20, via `fig6_transfer_network/plot_wccnet.py` and `s10_transfer_network/plot_net.py` |
| `s12_why_transfer/make_table.py` | Table S15 |
| `s15_structural_sentinels/make_tables.py` | the MinCorr, QR and random rows of Table S23 |

It is produced by `fig4_fig5_cross_transfer/compute_matrices.py`, which
writes it straight into this folder.
