"""Comparison across three Ken French panels (Table 5).

Estimation windows are set from the number of assets N so that N/T takes the
values 0.8, 0.4 and 0.2 in each panel (with T > N, so the sample covariance is
invertible). For each panel, window, covariance estimator and level, the
plug-in, historical, EWMA, conformal and adaptive conformal bands are scored on
the same days: coverage, Kupiec p-value and mean width.

    python -m src.run_panels

The 100-portfolio panel makes this the slowest script (a few minutes).
"""
from __future__ import annotations

import config
from src.backtest import run_backtest
from src.baselines import band_plugin, band_historical, band_ewma
from src.conformal import conformal_recalibrate, aci_recalibrate
from src.data import load_ff
from src.metrics import common_valid, coverage_width, kupiec_p

PANELS = ["ff49_industry", "ff30_industry", "ff100_sizebm"]
NT_TARGETS = (0.8, 0.4, 0.2)
ALPHAS = (0.05, 0.01)
CAL_WINDOW, MIN_CAL = 750, 250


def windows_for(N):
    return sorted({max(int(round(N / t)), N + 5) for t in NT_TARGETS})


def main():
    config.set_seeds()
    hdr = (f"{'panel':>14} {'est':>12} {'win':>4} {'N/T':>5} {'lvl':>5} "
           f"{'method':>14} {'cov':>6} {'Kup p':>7} {'width':>8}")
    for panel in PANELS:
        R = load_ff(panel, start="1970-01-01")
        N = R.shape[1]
        wins = windows_for(N)
        print(f"\n{panel}: {R.shape[0]} days x {N} assets, windows {wins}")
        print(hdr)
        for W in wins:
            for est in ("sample", "ledoit_wolf"):
                oos = run_backtest(R, estimator=est, window=W)["oos"]
                if len(oos) < 1000:
                    continue
                loss = oos["loss"].to_numpy()
                for alpha in ALPHAS:
                    bands = {
                        "plugin": band_plugin(oos, alpha),
                        "historical": band_historical(oos, alpha, CAL_WINDOW, MIN_CAL),
                        "ewma": band_ewma(oos, alpha),
                        "conformal": conformal_recalibrate(
                            oos, alpha, CAL_WINDOW, MIN_CAL)["cal_bound"],
                        "conformal_aci": aci_recalibrate(
                            oos, alpha, CAL_WINDOW, MIN_CAL)["cal_bound"],
                    }
                    valid = common_valid(loss, bands.values())
                    for m, b in bands.items():
                        cov, width, _ = coverage_width(loss, b, valid)
                        hit = (loss[valid] > b[valid]).astype(int)
                        print(f"{panel:>14} {est:>12} {W:>4} {N / W:>5.2f} {1 - alpha:>5.2f} "
                              f"{m:>14} {cov:>6.3f} {kupiec_p(hit, alpha):>7.3g} {width:>8.5f}")


if __name__ == "__main__":
    main()
