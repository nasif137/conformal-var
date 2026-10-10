"""Long-only robustness check (Table 7).

Repeats the plug-in versus conformal comparison with the minimum-variance
portfolio constrained to non-negative weights. That problem has no closed form,
so each rebalance solves

    minimize w' Sigma w  subject to  sum(w) = 1, w >= 0

with SLSQP, using the Ledoit-Wolf covariance. As in the main backtest, plug_vol
is computed once per rebalance and held over the 21-day holding period.

    python -m src.run_longonly
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

import config
from src.conformal import conformal_recalibrate
from src.data import load_ff

CAL_WINDOW, MIN_CAL, REBALANCE = 750, 250, 21

# FF49 across three N/T values, and the other two panels near N/T = 0.4
CONFIGS = [
    ("ff49_industry", 60),
    ("ff49_industry", 120),
    ("ff49_industry", 252),
    ("ff30_industry", 75),
    ("ff100_sizebm", 250),
]


def gmv_longonly(cov):
    n = cov.shape[0]
    w0 = np.ones(n) / n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0,
             "jac": lambda w: np.ones_like(w)}]
    res = minimize(lambda w: float(w @ cov @ w), w0, jac=lambda w: 2.0 * cov @ w,
                   bounds=[(0.0, 1.0)] * n, constraints=cons, method="SLSQP",
                   options={"maxiter": 300, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else w0


def backtest_longonly(R, window):
    """Out-of-sample frame [date, plug_vol, loss] for the long-only portfolio."""
    oos_d, oos_pv, oos_loss = [], [], []
    t = window
    while t + REBALANCE <= len(R):
        win = R.iloc[t - window:t]
        hold = R.iloc[t:t + REBALANCE]
        good = win.notna().all() & hold.notna().all()
        assets = good[good].index
        if len(assets) < 5:
            t += REBALANCE
            continue
        cov = LedoitWolf().fit(win[assets].values).covariance_
        w = gmv_longonly(cov)
        plug_vol = float(np.sqrt(max(w @ cov @ w, 1e-18)))
        losses = -(hold[assets].values @ w)
        oos_d.extend(list(hold.index))
        oos_pv.extend([plug_vol] * len(losses))
        oos_loss.extend(losses.tolist())
        t += REBALANCE
    return pd.DataFrame({"date": oos_d, "plug_vol": oos_pv, "loss": oos_loss})


def main():
    config.set_seeds()
    cache = {}
    print(f"{'panel':>14} {'win':>4} {'N/T':>5} {'lvl':>5} {'plug_cov':>9} {'conf_cov':>9}")
    for panel, window in CONFIGS:
        if panel not in cache:
            cache[panel] = load_ff(panel, start="1970-01-01")
        R = cache[panel]
        N = R.shape[1]
        oos = backtest_longonly(R, window)
        loss = oos["loss"].to_numpy()
        for alpha in (0.05, 0.01):
            z = float(stats.norm.ppf(1 - alpha))
            plug = z * oos["plug_vol"].to_numpy()
            c = conformal_recalibrate(oos, alpha, CAL_WINDOW, MIN_CAL)
            valid = c["valid_mask"]
            print(f"{panel:>14} {window:>4} {N / window:>5.2f} {1 - alpha:>5.2f} "
                  f"{np.mean(loss[valid] <= plug[valid]):>9.3f} "
                  f"{np.mean(loss[valid] <= c['cal_bound'][valid]):>9.3f}")


if __name__ == "__main__":
    main()
