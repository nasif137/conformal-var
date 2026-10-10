"""Global settings shared by every script.

All randomness (the moving-block bootstrap) is seeded from SEED, so every run is
deterministic given the data.
"""
import os
import random

import numpy as np

SEED = 7

DATA_DIR = "data"          # cached copies of the downloaded Ken French files
FIGURE_DIR = "figures"     # PNG figures written by the make_* scripts
RESULTS_DIR = "results"    # text output written by reproduce.py


def set_seeds(seed: int = SEED) -> None:
    """Seed Python and NumPy. Called at the start of every script."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
