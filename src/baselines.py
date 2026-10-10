"""Standard VaR baselines on the same out-of-sample loss series.

  - plug-in:    z * plug_vol, the band implied by the estimated covariance
  - historical: empirical quantile of the trailing losses
  - EWMA:       RiskMetrics volatility (lambda = 0.94) times z
  - FHS:        filtered historical simulation, an empirical quantile of
                EWMA-standardized losses rescaled by the current EWMA volatility

Every band for day t uses only data from days before t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def band_plugin(oos: pd.DataFrame, alpha: float = 0.05) -> np.ndarray:
    z = float(stats.norm.ppf(1 - alpha))
    return z * oos["plug_vol"].to_numpy()


def band_historical(oos: pd.DataFrame, alpha: float = 0.05,
                    window: int = 750, min_win: int = 250) -> np.ndarray:
    """Empirical (1-alpha) quantile of the trailing window of losses."""
    loss = oos["loss"].to_numpy()
    n = len(loss)
    b = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - window)
        cal = loss[lo:t]
        if len(cal) < min_win:
            continue
        b[t] = np.quantile(cal, 1 - alpha, method="higher")
    return b


def ewma_sigma(loss: np.ndarray, lam: float = 0.94, warmup: int = 250) -> np.ndarray:
    """RiskMetrics volatility for day t from returns up to day t-1.

    The variance starts at the sample variance of the first `warmup` returns;
    earlier days are NaN.
    """
    r = -np.asarray(loss)
    n = len(r)
    sig = np.full(n, np.nan)
    if n <= warmup:
        return sig
    var = float(np.var(r[:warmup]))
    for t in range(warmup, n):
        var = lam * var + (1 - lam) * r[t - 1] ** 2
        sig[t] = np.sqrt(var)
    return sig


def band_ewma(oos: pd.DataFrame, alpha: float = 0.05,
              lam: float = 0.94, warmup: int = 250) -> np.ndarray:
    z = float(stats.norm.ppf(1 - alpha))
    return z * ewma_sigma(oos["loss"].to_numpy(), lam, warmup)


def standardized(loss: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Losses divided by EWMA volatility; NaN where volatility is unavailable."""
    u = np.full(len(loss), np.nan)
    ok = ~np.isnan(sig) & (sig > 0)
    u[ok] = loss[ok] / sig[ok]
    return u


def band_fhs(loss, sig, alpha, window=750, min_cal=250, method="higher"):
    """Filtered historical simulation.

    The (1-alpha) empirical quantile of the standardized losses from the last
    `window` days, times the current EWMA volatility. `method` is the numpy
    quantile rule; "higher" is the default throughout the paper.
    """
    u = standardized(loss, sig)
    n = len(u)
    b = np.full(n, np.nan)
    for t in range(n):
        if np.isnan(sig[t]):
            continue
        lo = max(0, t - window)
        cal = u[lo:t]
        cal = cal[~np.isnan(cal)]
        if len(cal) < min_cal:
            continue
        b[t] = np.quantile(cal, 1 - alpha, method=method) * sig[t]
    return b


def regime_fhs(loss, sig, g, alpha, cal_window=2000, min_cal=250):
    """Regime-conditional FHS on the same partition as regime_conformal.

    Uses the empirical quantile within the regime, with no finite-sample
    correction. Falls back to all past days in the window when the regime has
    fewer than min_cal of them.
    """
    u = standardized(loss, sig)
    n = len(u)
    b = np.full(n, np.nan)
    for t in range(n):
        if np.isnan(sig[t]):
            continue
        lo = max(0, t - cal_window)
        idx = np.arange(lo, t)
        sel = idx[g[idx] == g[t]] if g[t] >= 0 else idx
        cal = u[sel]
        cal = cal[~np.isnan(cal)]
        if len(cal) < min_cal:
            cal = u[lo:t]
            cal = cal[~np.isnan(cal)]
        if len(cal) < min_cal:
            continue
        b[t] = np.quantile(cal, 1 - alpha, method="higher") * sig[t]
    return b
