"""Regime-conditional comparison (Table 4).

Days are assigned to a low, mid or high volatility regime from past data only
(see conformal.regimes). Regime-conditional (Mondrian) conformal and
regime-conditional FHS calibrate within the current regime. Prints overall
coverage, Kupiec and Christoffersen p-values and width, coverage within each
regime, and coverage in calm days and the two crises, at 95% and 99%.

    python -m src.run_regime
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.backtest import run_backtest, oos_dates
from src.baselines import band_plugin, band_ewma, band_fhs, ewma_sigma, regime_fhs
from src.conformal import (conformal_recalibrate, aci_recalibrate, regimes,
                           regime_conformal)
from src.data import load_ff
from src.metrics import christoffersen_cc_p, common_valid, kupiec_p

PANEL, EST, WINDOW = "ff49_industry", "ledoit_wolf", 120
CAL_WINDOW, CAL_REG, MIN_CAL = 750, 2000, 250
CRISES = [("2008", "2008-06-01", "2009-06-30"),
          ("2020", "2020-02-15", "2020-05-31")]
ORDER = ["plugin", "ewma", "fhs", "conformal", "conformal_aci",
         "fhs_regime", "conformal_regime"]


def main():
    config.set_seeds()
    R = load_ff(PANEL, start="1970-01-01")
    oos = run_backtest(R, estimator=EST, window=WINDOW)["oos"]
    loss = oos["loss"].to_numpy()
    plug_vol = oos["plug_vol"].to_numpy()
    dates = oos_dates(oos)
    sig = ewma_sigma(loss)
    g = regimes(loss)
    print(f"regime days low/mid/high = {int((g == 0).sum())}/{int((g == 1).sum())}/"
          f"{int((g == 2).sum())}, unassigned = {int((g < 0).sum())}")

    for alpha in (0.05, 0.01):
        bands = {
            "plugin": band_plugin(oos, alpha),
            "ewma": band_ewma(oos, alpha),
            "fhs": band_fhs(loss, sig, alpha, CAL_WINDOW, MIN_CAL),
            "conformal": conformal_recalibrate(oos, alpha, CAL_WINDOW, MIN_CAL)["cal_bound"],
            "conformal_aci": aci_recalibrate(oos, alpha, CAL_WINDOW, MIN_CAL)["cal_bound"],
            "fhs_regime": regime_fhs(loss, sig, g, alpha, CAL_REG, MIN_CAL),
            "conformal_regime": regime_conformal(loss, plug_vol, g, alpha, CAL_REG, MIN_CAL),
        }
        valid = common_valid(loss, bands.values())
        lvl = round((1 - alpha) * 100)
        print(f"\n{lvl}% VaR, {PANEL}, {EST}, window {WINDOW}; days = {int(valid.sum())}")
        print(f"{'method':>18} {'cov':>7} {'kupiec_p':>10} {'cc_p':>10} {'width':>9}")
        for k in ORDER:
            b = bands[k]
            hit = (loss[valid] > b[valid]).astype(int)
            print(f"{k:>18} {np.mean(loss[valid] <= b[valid]):7.3f} "
                  f"{kupiec_p(hit, alpha):10.4g} {christoffersen_cc_p(hit, alpha):10.4g} "
                  f"{np.mean(b[valid]):9.5f}")

        print(f"coverage by regime (target {1 - alpha:.2f}):")
        print(f"{'method':>18} {'low':>7} {'mid':>7} {'high':>7}")
        for k in ORDER:
            cov = loss <= bands[k]
            cells = []
            for rg in (0, 1, 2):
                m = valid & (g == rg)
                cells.append(f"{np.mean(cov[m]) if m.any() else float('nan'):7.3f}")
            print(f"{k:>18} " + " ".join(cells))

        in_crisis = np.zeros(len(loss), dtype=bool)
        masks = {}
        for name, a0, b0 in CRISES:
            m = (dates >= pd.Timestamp(a0)) & (dates <= pd.Timestamp(b0))
            masks[name] = m & valid
            in_crisis |= m
        calm = valid & ~in_crisis
        print(f"coverage by period ({lvl}%):")
        print(f"{'method':>18} {'calm':>7} " + " ".join(f"{n:>7}" for n, _, _ in CRISES))
        for k in ORDER:
            cov = loss <= bands[k]
            print(f"{k:>18} {np.mean(cov[calm]):7.3f} "
                  + " ".join(f"{np.mean(cov[masks[n]]):7.3f}" for n, _, _ in CRISES))


if __name__ == "__main__":
    main()
