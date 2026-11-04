"""Run only reactome SIS+DW under v3 ranges (after the duplicate-process
issue with regenerate_sis_dw_v3.py). Reactome is the largest network in
the v3 12-network set, N=5973."""
import sys
sys.path.insert(0, ".")
import regenerate_sis_dw_v3 as R

R.NETWORKS = ["reactome"]
R.main()
