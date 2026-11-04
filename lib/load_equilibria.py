"""Equilibrium loader for all 21 dynamics, and the network list.

Every script that reads equilibrium CSVs goes through this module.  It
exposes

  DYNAMICS, NETWORKS   the 21 dynamics and the 14 networks, in display order
  ADJ_21, DIF_21       the adjacency-coupled and Laplacian-coupled dynamics
  DISPLAY              display names (MSIS -> MSIS1, and so on)
  NETDIR               folder of the model-network edge lists
  network_file(tag)    path of the edge list of network `tag`
  load_eq(tag, s)      the M x N equilibrium matrix of dynamics s on network
                       tag, one row per control-parameter value

The 16 dynamics of the first submission are read with the loader of the
submitted code (lib/submitted/pipeline_v21.py); the five dynamics added in
the revision (LDW, LMUT, ANM, AEUT1, AVEG1) are read directly.  The module
computes nothing.
"""
from __future__ import annotations

# Pin the BLAS before numpy/torch import: the decoders are tiny dense
# problems, and multi-threaded BLAS costs more in synchronization than it
# saves.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths                                          # noqa: E402

NETDIR = paths.NET_MODEL
network_file = paths.network_file

import pipeline_v21 as p21                            # noqa: E402

# pipeline_v21.load_eq reads "<its own folder>/<tag>_<stem>_...csv", so its
# base folder is pointed at the submitted equilibria.
p21.HERE = paths.EQ_SUBMITTED

# The 14 networks, in the order of SI Table S24.
NETWORKS_14 = ["dolphin", "prox", "metabolic", "gkk", "lfr", "er",
               "ba", "hk", "euroroad", "email", "flybi", "reactome",
               "route_views", "spanish"]

# Display order: the adjacency-coupled dynamics first and the
# Laplacian-coupled ones last, each new dynamics appended to its class.
ADJ_21 = ["SIS", "SIR", "MSIS", "MSIS2", "MSIR", "MSIR2",
          "DW", "MUT", "GEN", "WC", "gLV", "ANM", "AEUT1", "AVEG1"]
DIF_21 = ["NM", "EUT", "EUT2", "VEG", "VEG2", "LDW", "LMUT"]
DYNAMICS = ADJ_21 + DIF_21
NETWORKS = NETWORKS_14
NEW = {"LDW", "LMUT", "ANM", "AEUT1", "AVEG1"}

DISPLAY = {"MSIS": "MSIS1", "MSIR": "MSIR1", "EUT": "EUT1", "VEG": "VEG1"}

# The five dynamics added in the revision live with the revision's equilibria.
REVDATA = paths.EQ_REVISION
NEW_PATHS = {
    "LDW":   (REVDATA, "ldw"),
    "LMUT":  (REVDATA, "lmut"),
    "ANM":   (REVDATA, "anm"),
    "AEUT1": (REVDATA, "aeut"),
    "AVEG1": (REVDATA, "aveg"),
}

def load_eq(tag, s):
    """Equilibrium matrix (M x N) for dynamics s on network tag."""
    if s in NEW_PATHS:
        folder, stem = NEW_PATHS[s]
        arr = np.loadtxt(folder / f"{tag}_{stem}_equilibrium_data.csv",
                         delimiter=",", skiprows=1)
        return arr[:, 1:]
    return p21.load_eq(tag, s)          # the submitted code's loader


