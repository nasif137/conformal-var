"""Coverage by forecast horizon (Table 2).

The plug-in, EWMA and FHS bands are scaled to h days by sqrt(h). The conformal
band is calibrated directly on realized h-day losses (conformal_horizon), so it
assumes no scaling rule. Coverage is measured against the realized h-day loss.

    python -m src.run_horizon
"""
from __future__ import annotations

import numpy as np
from scipy import stats

import config
from src.backtest import run_backtest
from src.baselines import band_fhs, ewma_sigma
from src.conformal import conformal_horizon, horizon_loss
from src.data import load_ff
from src.metrics import common_valid

PANEL, EST, WINDOW = "ff49_industry", "ledoit_wolf", 120
HORIZONS = (1, 5, 10, 20)
CAL_WINDOW, MIN_CAL = 750, 250


def coverages(loss, plug_vol, sig, h, alpha):
    """Coverage of the h-day loss for each method, and the number of days."""
    Lh = horizon_loss(loss, h)
    z = float(stats.norm.ppf(1 - alpha))
    rh = np.sqrt(h)
    bands = {
        "plug": z * rh * plug_vol,
        "ewma": z * rh * sig,
        "fhs": rh * band_fhs(loss, sig, alpha, CAL_WINDOW, MIN_CAL),
        "conformal": conformal_horizon(loss, plug_vol, h, alpha, CAL_WINDOW, MIN_CAL),
    }
    valid = common_valid(Lh, bands.values())
    cov = {k: float(np.mean(Lh[valid] <= b[valid])) for k, b in bands.items()}
    return cov, int(valid.sum())


def load_oos():
    R = load_ff(PANEL, start="1970-01-01")
    oos = run_backtest(R, estimator=EST, window=WINDOW)["oos"]
    loss = oos["loss"].to_numpy()
    return loss, oos["plug_vol"].to_numpy(), ewma_sigma(loss)


def main():
    config.set_seeds()
    loss, plug_vol, sig = load_oos()
    for alpha in (0.05, 0.01):
        lvl = round((1 - alpha) * 100)
        print(f"\n{lvl}% VaR coverage by horizon (target {1 - alpha:.2f})")
        print(f"{'h':>4} {'plug':>8} {'ewma':>8} {'fhs':>8} {'conformal':>10} {'n':>7}")
        for h in HORIZONS:
            cov, nv = coverages(loss, plug_vol, sig, h, alpha)
            print(f"{h:>4} {cov['plug']:>8.3f} {cov['ewma']:>8.3f} {cov['fhs']:>8.3f} "
                  f"{cov['conformal']:>10.3f} {nv:>7}")


if __name__ == "__main__":
    main()
