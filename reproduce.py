"""Run every experiment and figure script, saving each printout to results/.

    python reproduce.py            # everything
    python reproduce.py run_main   # one script
"""
import contextlib
import importlib
import os
import sys
import time

import config

MODULES = [
    "run_main",          # Tables 1 and 6
    "run_horizon",       # Table 2
    "run_scarce",        # Table 3
    "run_regime",        # Table 4
    "run_panels",        # Table 5
    "run_longonly",      # Table 7
    "make_figures",      # Figures 1, 2, 4
    "make_horizon_fig",  # Figure 3
    "make_stress_fig",   # Figure 5
]


def main(names):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    for name in names:
        t0 = time.time()
        path = os.path.join(config.RESULTS_DIR, f"{name}.txt")
        with open(path, "w") as f, contextlib.redirect_stdout(f):
            importlib.import_module(f"src.{name}").main()
        print(f"{name:<18} {time.time() - t0:6.0f}s  -> {path}")


if __name__ == "__main__":
    main(sys.argv[1:] or MODULES)
