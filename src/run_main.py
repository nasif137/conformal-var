"""Main comparison (Tables 1 and 6).

FF49 industry portfolios, Ledoit-Wolf covariance, 120-day estimation window.
For the plug-in, historical simulation, EWMA, FHS, conformal and adaptive
conformal bands, at the 95% and 99% levels, prints coverage with a 95%
moving-block bootstrap interval, the Kupiec and Christoffersen p-values and the
mean band width. At 99% it also prints paired coverage differences, and at both
levels the coverage in calm days, the 2008 crisis and the 2020 crisis.

    python -m src.run_main
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.backtest import run_backtest, oos_dates
from src.baselines import band_plugin, band_historical, band_ewma, band_fhs, ewma_sigma
from src.conformal import conformal_recalibrate, aci_recalibrate
from src.data import load_ff
from src.metrics import bootstrap_ci, christoffersen_cc_p, common_valid, kupiec_p

PANEL, EST, WINDOW = "ff49_industry", "ledoit_wolf", 120
CAL_WINDOW, MIN_CAL = 750, 250
BLOCK, B = 50, 5000
CRISES = [("2008", "2008-06-01", "2009-06-30"),
          ("2020", "2020-02-15", "2020-05-31")]
ORDER = ["plugin", "historical", "ewma", "fhs", "conformal", "conformal_aci"]
PAIRS = [("conformal", "ewma"), ("conformal", "historical"),
         ("conformal", "fhs"), ("conformal_aci", "ewma")]


def all_bands(oos, alpha):
    loss = oos["loss"].to_numpy()
    return {
        "plugin": band_plugin(oos, alpha),
        "historical": band_historical(oos, alpha, CAL_WINDOW, MIN_CAL),
        "ewma": band_ewma(oos, alpha),
        "fhs": band_fhs(loss, ewma_sigma(loss), alpha, CAL_WINDOW, MIN_CAL),
        "conformal": conformal_recalibrate(oos, alpha, CAL_WINDOW, MIN_CAL)["cal_bound"],
        "conformal_aci": aci_recalibrate(oos, alpha, CAL_WINDOW, MIN_CAL)["cal_bound"],
    }


def main():
    config.set_seeds()
    rng = np.random.default_rng(config.SEED)
    R = load_ff(PANEL, start="1970-01-01")
    oos = run_backtest(R, estimator=EST, window=WINDOW)["oos"]
    loss = oos["loss"].to_numpy()
    dates = oos_dates(oos)

    for alpha in (0.05, 0.01):
        bands = all_bands(oos, alpha)
        valid = common_valid(loss, bands.values())
        ind = {k: (loss[valid] <= b[valid]).astype(float) for k, b in bands.items()}
        ci, dci = bootstrap_ci(ind, PAIRS, BLOCK, B, rng)

        lvl = round((1 - alpha) * 100)
        print(f"\n{lvl}% VaR, {PANEL}, {EST}, window {WINDOW}")
        print(f"days = {int(valid.sum())}, block = {BLOCK}, B = {B}")
        print(f"{'method':>14} {'cov':>7} {'95% CI':>18} {'kupiec_p':>10} {'cc_p':>9} {'width':>9}")
        for k in ORDER:
            b = bands[k]
            hit = (loss[valid] > b[valid]).astype(int)
            p, lo, hi = ci[k]
            print(f"{k:>14} {p:7.3f} [{lo:6.3f},{hi:6.3f}] "
                  f"{kupiec_p(hit, alpha):10.4g} {christoffersen_cc_p(hit, alpha):9.4g} "
                  f"{float(np.mean(b[valid])):9.5f}")

        if alpha == 0.01:
            print("paired coverage differences:")
            for (a, c), (dd, lo, hi) in dci.items():
                print(f"  {a} - {c}: {dd:+.4f} [{lo:+.4f},{hi:+.4f}]")

        in_crisis = np.zeros(len(loss), dtype=bool)
        masks = {}
        for name, a0, b0 in CRISES:
            m = (dates >= pd.Timestamp(a0)) & (dates <= pd.Timestamp(b0))
            masks[name] = m & valid
            in_crisis |= m
        calm = valid & ~in_crisis
        print(f"coverage by period ({lvl}%):")
        print(f"{'method':>14} {'calm':>7} " + " ".join(f"{n:>7}" for n, _, _ in CRISES))
        for k in ORDER:
            cov = loss <= bands[k]
            print(f"{k:>14} {np.mean(cov[calm]):7.3f} "
                  + " ".join(f"{np.mean(cov[masks[n]]):7.3f}" for n, _, _ in CRISES))
        print("days: calm " + str(int(calm.sum())) + ", "
              + ", ".join(f"{n} {int(masks[n].sum())}" for n, _, _ in CRISES))


if __name__ == "__main__":
    main()
