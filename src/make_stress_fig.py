"""Figure 5: rolling 250-day coverage of the 99% band, with the 2008 and 2020
crises shaded. Coverage within each crisis is reported by run_main.

Writes figures/fig_stress_99.png.

    python -m src.make_stress_fig
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.backtest import run_backtest, oos_dates
from src.baselines import band_plugin, band_ewma, band_fhs, ewma_sigma
from src.conformal import aci_recalibrate
from src.data import load_ff
from src.metrics import common_valid

PANEL, EST, WINDOW = "ff49_industry", "ledoit_wolf", 120
CAL_WINDOW, MIN_CAL, ROLL, ALPHA = 750, 250, 250, 0.01
CRISES = [("2008 GFC", "2008-06-01", "2009-06-30"),
          ("2020 COVID", "2020-02-15", "2020-05-31")]
STYLE = [
    ("Plug-in",       "#999999", ":"),
    ("EWMA",          "#4C78A8", "--"),
    ("FHS",           "#E45756", "-."),
    ("Conformal+ACI", "#1B1B1B", "-"),
]


def main():
    config.set_seeds()
    R = load_ff(PANEL, start="1970-01-01")
    oos = run_backtest(R, estimator=EST, window=WINDOW)["oos"]
    loss = oos["loss"].to_numpy()
    dates = oos_dates(oos)

    bands = {
        "Plug-in": band_plugin(oos, ALPHA),
        "EWMA": band_ewma(oos, ALPHA),
        "FHS": band_fhs(loss, ewma_sigma(loss), ALPHA, CAL_WINDOW, MIN_CAL),
        "Conformal+ACI": aci_recalibrate(oos, ALPHA, CAL_WINDOW, MIN_CAL)["cal_bound"],
    }
    valid = common_valid(loss, bands.values())

    fig, ax = plt.subplots(figsize=(9, 3.8))
    idx = pd.DatetimeIndex(dates[valid])
    lows = []
    for name, color, ls in STYLE:
        hit = (loss[valid] <= bands[name][valid]).astype(float)
        roll = pd.Series(hit, index=idx).rolling(ROLL).mean()
        ax.plot(roll.index, roll.values, color=color, ls=ls, lw=1.1, label=name)
        lows.append(np.nanmin(roll.values))
    ax.axhline(1 - ALPHA, color="k", lw=0.8, ls="--", alpha=0.6)
    xform = ax.get_xaxis_transform()
    for name, a0, b0 in CRISES:
        a, b = pd.Timestamp(a0), pd.Timestamp(b0)
        ax.axvspan(a, b, color="gray", alpha=0.18)
        ax.text(a + (b - a) / 2, 1.01, name, transform=xform, ha="center",
                va="bottom", fontsize=8, color="#444")
    ax.set_ylim(np.floor((min(lows) - 0.005) * 100) / 100, 1.005)
    ax.set_ylabel(f"Rolling {ROLL}-day coverage")
    ax.set_xlabel("Year")
    ax.grid(True, alpha=0.2, lw=0.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    out = os.path.join(config.FIGURE_DIR, "fig_stress_99.png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
