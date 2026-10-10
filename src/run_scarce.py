"""Coverage under scarce calibration (Table 3).

Shrinks the calibration size M and compares conformal, which uses the
finite-sample level ceil((M+1)(1-alpha)), with FHS under the numpy default
("linear") and the conservative ("higher") quantile rules. Every method uses
exactly M past points, and all are scored on the same days.

    python -m src.run_scarce
"""
from __future__ import annotations

import numpy as np

import config
from src.backtest import run_backtest
from src.baselines import band_fhs, ewma_sigma
from src.conformal import conformal_recalibrate
from src.data import load_ff
from src.metrics import common_valid

PANEL, EST, WINDOW = "ff49_industry", "ledoit_wolf", 120
SIZES = (100, 150, 250, 500, 750, 1500)


def main():
    config.set_seeds()
    R = load_ff(PANEL, start="1970-01-01")
    oos = run_backtest(R, estimator=EST, window=WINDOW)["oos"]
    loss = oos["loss"].to_numpy()
    sig = ewma_sigma(loss)

    for alpha in (0.05, 0.01):
        lvl = round((1 - alpha) * 100)
        print(f"\n{lvl}% VaR (target coverage {1 - alpha:.2f}), coverage / mean width")
        print(f"{'M':>6} {'conformal':>20} {'fhs_linear':>20} {'fhs_higher':>20}")
        for M in SIZES:
            bands = [
                conformal_recalibrate(oos, alpha, cal_window=M, min_cal=M)["cal_bound"],
                band_fhs(loss, sig, alpha, window=M, min_cal=M, method="linear"),
                band_fhs(loss, sig, alpha, window=M, min_cal=M, method="higher"),
            ]
            valid = common_valid(loss, bands)
            cells = [f"{np.mean(loss[valid] <= b[valid]):.3f} / {np.mean(b[valid]):.4f}"
                     for b in bands]
            print(f"{M:>6} " + " ".join(f"{c:>20}" for c in cells))
        print(f"days scored at M = {SIZES[-1]}: {int(valid.sum())}")


if __name__ == "__main__":
    main()
