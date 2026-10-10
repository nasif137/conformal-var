"""Rolling global minimum-variance (GMV) backtest.

At each rebalance date the covariance is estimated from the trailing window of
`window` days, the GMV weights are formed, and the portfolio is held for the
next `rebalance` days. The plug-in volatility sqrt(w' S w) is computed once per
rebalance and held fixed over the holding period. The function returns a
per-day out-of-sample frame (date, plug_vol, loss) on which every VaR band in
this repository is evaluated, plus summary statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def _sample_cov(window: np.ndarray) -> np.ndarray:
    return np.cov(window, rowvar=False)


def _ledoit_wolf_cov(window: np.ndarray) -> np.ndarray:
    return LedoitWolf().fit(window).covariance_


COV_ESTIMATORS = {
    "sample": _sample_cov,
    "ledoit_wolf": _ledoit_wolf_cov,
}


def gmv_weights(cov: np.ndarray) -> np.ndarray:
    """Unconstrained GMV weights S^-1 1 / (1' S^-1 1).

    Short positions are allowed. The long-only variant is in run_longonly.py.
    """
    n = cov.shape[0]
    ones = np.ones(n)
    inv1 = np.linalg.solve(cov, ones)
    return inv1 / (ones @ inv1)


def oos_dates(oos: pd.DataFrame) -> pd.DatetimeIndex:
    """Dates of the out-of-sample frame as a DatetimeIndex."""
    d = oos["date"]
    if isinstance(d.dtype, pd.PeriodDtype):
        d = d.dt.to_timestamp()
    return pd.to_datetime(np.asarray(pd.to_datetime(d)))


def run_backtest(
    returns: pd.DataFrame,
    estimator: str = "sample",
    window: int = 252,
    rebalance: int = 21,
    z: float = 1.645,
) -> dict:
    """Roll through the sample and record claimed versus realized risk.

    Returns {"summary": dict, "records": DataFrame, "oos": DataFrame}. The "oos"
    frame has one row per out-of-sample day with columns [date, plug_vol, loss],
    in time order. Assets with any missing value in the estimation window or the
    holding period are dropped for that rebalance.
    """
    est = COV_ESTIMATORS[estimator]
    R = returns
    dates = R.index

    records = []
    realized_losses: list = []
    predicted_bounds: list = []
    plug_vols: list = []

    oos_dates_: list = []
    oos_plug_vol: list = []
    oos_loss: list = []

    t = window
    while t + rebalance <= len(R):
        win = R.iloc[t - window:t]
        hold = R.iloc[t:t + rebalance]

        good = win.notna().all() & hold.notna().all()
        assets = good[good].index
        if len(assets) < 5:
            t += rebalance
            continue

        W = win[assets].values
        H = hold[assets].values

        cov = est(W)
        try:
            w = gmv_weights(cov)
        except np.linalg.LinAlgError:
            t += rebalance
            continue

        plug_var = float(w @ cov @ w)
        plug_vol = np.sqrt(max(plug_var, 1e-18))
        var_bound = z * plug_vol

        port_ret = H @ w
        losses = -port_ret

        realized_losses.extend(losses.tolist())
        predicted_bounds.extend([var_bound] * len(losses))
        plug_vols.append(plug_vol)

        oos_dates_.extend(list(hold.index))
        oos_plug_vol.extend([plug_vol] * len(losses))
        oos_loss.extend(losses.tolist())

        records.append({
            "date": dates[t],
            "n_assets": int(len(assets)),
            "plug_vol": plug_vol,
            "realized_vol_hold": float(np.std(port_ret, ddof=1)) if len(port_ret) > 1 else np.nan,
            "var_bound": var_bound,
            "exceed_rate_hold": float(np.mean(losses > var_bound)),
        })
        t += rebalance

    rl = np.array(realized_losses)
    pv = np.array(predicted_bounds)
    records_df = pd.DataFrame.from_records(records)
    oos = pd.DataFrame({"date": oos_dates_, "plug_vol": oos_plug_vol, "loss": oos_loss})

    predicted_vol = float(np.mean(plug_vols)) if plug_vols else np.nan
    realized_vol = float(np.std(rl, ddof=1)) if len(rl) > 1 else np.nan
    vol_ratio = realized_vol / predicted_vol if predicted_vol and predicted_vol > 0 else np.nan
    var_coverage = float(np.mean(rl <= pv)) if len(rl) else np.nan

    summary = {
        "estimator": estimator,
        "window": window,
        "n_holds": len(records),
        "n_obs_oos": int(len(rl)),
        "predicted_vol": predicted_vol,
        "realized_vol": realized_vol,
        "vol_understatement_ratio": vol_ratio,
        "var_nominal_coverage": 0.95,
        "var_empirical_coverage": var_coverage,
        "var_exceedance_rate": (1 - var_coverage) if len(rl) else np.nan,
        "mean_var_bound": float(np.mean(pv)) if len(pv) else np.nan,
    }
    return {"summary": summary, "records": records_df, "oos": oos}
