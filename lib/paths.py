"""Where the code finds its inputs.

Every script that reads equilibrium data or network edge lists takes the
locations from here, so they can be changed in one place.  Importing this
module also puts lib/ and lib/submitted/ on the module search path, which is
what lets the scripts import the shared modules by name.

Inputs that are not part of the repository (see the README, "Input data"):

  DECODER_DATA                  root folder of the input data; default
                                input_data/ inside the repository
    equilibria_submitted/       equilibrium CSVs of the 16 dynamics of the
                                first submission
    equilibria_revision/        equilibrium CSVs computed for the revision,
                                with the subfolders ba_size_sweep/,
                                hetero_replicates/ and doublewell_upper/
    networks_empirical/         edge lists of the 9 empirical networks

Each of the three subfolders can also be set on its own with the environment
variables DECODER_EQ_SUBMITTED, DECODER_EQ_REVISION and
DECODER_NETWORKS_EMPIRICAL, which take precedence over DECODER_DATA.

Inputs that are part of the repository:

  networks/                     edge lists of the 5 model-generated networks
    ba_size_sweep/              edge lists of the 6 Barabasi-Albert networks
                                of SI Section S9
"""
import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent      # the repository root
LIB = CODE / "lib"
SUBMITTED = LIB / "submitted"          # modules of the first submission
ORIGINAL16 = CODE / "equilibria" / "original16"


def _folder(variable, default):
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default


DATA = _folder("DECODER_DATA", CODE / "input_data")
EQ_SUBMITTED = _folder("DECODER_EQ_SUBMITTED", DATA / "equilibria_submitted")
EQ_REVISION = _folder("DECODER_EQ_REVISION", DATA / "equilibria_revision")
NET_EMPIRICAL = _folder("DECODER_NETWORKS_EMPIRICAL",
                        DATA / "networks_empirical")
NET_MODEL = CODE / "networks"
NET_BA_SWEEP = NET_MODEL / "ba_size_sweep"

MODEL_NETWORKS = ("gkk", "lfr", "er", "ba", "hk")
EMPIRICAL_NETWORKS = ("dolphin", "prox", "metabolic", "euroroad", "email",
                      "flybi", "reactome", "route_views", "spanish")


def network_file(tag):
    """Edge list of network `tag`, from the repository or from the input data."""
    name = f"{tag}_network.edgelist"
    for folder in (NET_MODEL, NET_BA_SWEEP, NET_EMPIRICAL):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(
        f"{name} not found in {NET_MODEL}, {NET_BA_SWEEP} or {NET_EMPIRICAL}; "
        "the empirical networks are not distributed with this repository, "
        "see the README, 'Input data'")


for _p in (SUBMITTED, LIB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
