"""Conformal recalibration of the plug-in VaR band.

The plug-in band is z * plug_vol. Conformal recalibration replaces z with a
quantile of the standardized scores s = loss / plug_vol observed on earlier days:

  - The calibration set for day t contains only scores from days before t.
  - Each score is standardized by a covariance estimated before its own day,
    so no day informs its own band.
  - The quantile is the k-th smallest score with k = ceil((m+1)(1-alpha)), the
    finite-sample conformal level, rather than a plain empirical percentile.

Variants: adaptive conformal inference (ACI), a multi-day horizon version that
calibrates on realized h-day losses, and a regime-conditional (Mondrian) version
that calibrates within volatility terciles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_recalibrate(
    oos: pd.DataFrame,
    alpha: float = 0.05,
    cal_window: int = 750,
    min_cal: int = 250,
) -> dict:
    """Online split-conformal recalibration of the one-sided VaR band.

    oos        : DataFrame with columns [plug_vol, loss], in time order.
    alpha      : miscoverage target (0.05 for a 95% band, 0.01 for 99%).
    cal_window : number of most recent past scores used for calibration.
    min_cal    : minimum calibration size before a band is issued.

    The band for day t is qhat_t * plug_vol_t, where qhat_t is the finite-sample
    conformal quantile of the scores from the preceding days.
    """
    s = (oos["loss"] / oos["plug_vol"]).to_numpy()
    plug_vol = oos["plug_vol"].to_numpy()
    loss = oos["loss"].to_numpy()
    n = len(s)

    cal_bound = np.full(n, np.nan)
    cal_mult = np.full(n, np.nan)

    for t in range(n):
        lo = max(0, t - cal_window)
        cal = s[lo:t]                       # scores strictly before t
        m = len(cal)
        if m < min_cal:
            continue
        k = int(np.ceil((m + 1) * (1 - alpha)))
        if k > m:
            continue                        # too few scores for this alpha
        qhat = np.sort(cal)[k - 1]          # k-th smallest score
        cal_mult[t] = qhat
        cal_bound[t] = qhat * plug_vol[t]

    valid = ~np.isnan(cal_bound)
    n_valid = int(valid.sum())
    coverage = float(np.mean(loss[valid] <= cal_bound[valid])) if n_valid else np.nan
    mean_bound = float(np.mean(cal_bound[valid])) if n_valid else np.nan
    mean_mult = float(np.mean(cal_mult[valid])) if n_valid else np.nan

    return {
        "coverage": coverage,
        "n_valid": n_valid,
        "mean_bound": mean_bound,           # mean band width
        "mean_multiplier": mean_mult,       # mean qhat, to compare with z
        "cal_bound": cal_bound,
        "cal_mult": cal_mult,
        "valid_mask": valid,
    }


def aci_recalibrate(
    oos: pd.DataFrame,
    alpha: float = 0.05,
    cal_window: int = 750,
    min_cal: int = 250,
    gamma: float = 0.01,
) -> dict:
    """Adaptive conformal inference (Gibbs and Candes, 2021) on the same score.

    The working level is updated online, alpha_{t+1} = alpha_t + gamma * (alpha -
    miss_t), starting at alpha and clamped to [0.001, 0.5]. The band for day t uses
    only scores before t; the miss on day t updates the level for day t+1.
    """
    s = (oos["loss"] / oos["plug_vol"]).to_numpy()
    plug_vol = oos["plug_vol"].to_numpy()
    loss = oos["loss"].to_numpy()
    n = len(s)

    bound = np.full(n, np.nan)
    a_t = alpha
    for t in range(n):
        lo = max(0, t - cal_window)
        cal = s[lo:t]
        m = len(cal)
        if m < min_cal:
            continue
        lvl = min(max(a_t, 0.0), 1.0)
        k = int(np.ceil((m + 1) * (1 - lvl)))
        k = min(max(k, 1), m)               # keep k inside the calibration set
        qhat = np.sort(cal)[k - 1]
        bound[t] = qhat * plug_vol[t]
        miss = 1.0 if loss[t] > bound[t] else 0.0
        a_t = a_t + gamma * (alpha - miss)
        a_t = min(max(a_t, 0.001), 0.5)

    valid = ~np.isnan(bound)
    nv = int(valid.sum())
    coverage = float(np.mean(loss[valid] <= bound[valid])) if nv else np.nan
    return {
        "coverage": coverage,
        "cal_bound": bound,
        "valid_mask": valid,
        "mean_bound": float(np.mean(bound[valid])) if nv else np.nan,
    }


def horizon_loss(loss: np.ndarray, h: int) -> np.ndarray:
    """Realized h-day loss L_t + ... + L_{t+h-1}; NaN where it is not observed."""
    n = len(loss)
    Lh = np.full(n, np.nan)
    for t in range(n - h + 1):
        Lh[t] = np.sum(loss[t:t + h])
    return Lh


def conformal_horizon(loss, plug_vol, h, alpha, cal_window=750, min_cal=250):
    """Conformal band for the h-day loss, calibrated on realized h-day scores.

    The score for day j is L^(h)_j / (sqrt(h) * plug_vol_j). Day t uses only days
    j <= t - h, whose h-day loss is fully realized before t. No square-root-of-time
    scaling of the one-day band is assumed.
    """
    n = len(loss)
    Lh = horizon_loss(loss, h)
    ph = np.sqrt(h) * plug_vol
    sh = Lh / ph
    band = np.full(n, np.nan)
    for t in range(n):
        if np.isnan(ph[t]):
            continue
        hi = t - h
        if hi < 0:
            continue
        lo = max(0, hi - cal_window + 1)
        cal = sh[lo:hi + 1]
        cal = cal[~np.isnan(cal)]
        if len(cal) < min_cal:
            continue
        k = int(np.ceil((len(cal) + 1) * (1 - alpha)))
        if k > len(cal):
            continue
        band[t] = np.sort(cal)[k - 1] * ph[t]
    return band


def regimes(loss, rvwin=21, regwin=1000, min_hist=250):
    """Volatility regime of each day from past data only.

    The regime is the tercile of the trailing rvwin-day realized volatility
    within its own trailing regwin-day history: 0 = low, 1 = mid, 2 = high, and
    -1 when there is too little history.
    """
    r = -loss
    n = len(r)
    rv = np.full(n, np.nan)
    for t in range(rvwin, n):
        rv[t] = np.std(r[t - rvwin:t], ddof=1)
    g = np.full(n, -1, dtype=int)
    for t in range(n):
        if np.isnan(rv[t]):
            continue
        lo = max(0, t - regwin)
        hist = rv[lo:t]
        hist = hist[~np.isnan(hist)]
        if len(hist) < min_hist:
            continue
        q1, q2 = np.percentile(hist, [100 / 3, 200 / 3])
        g[t] = 0 if rv[t] <= q1 else (1 if rv[t] <= q2 else 2)
    return g


def _kth_smallest_band(cal, alpha, scale, min_cal):
    m = len(cal)
    if m < min_cal:
        return np.nan
    k = int(np.ceil((m + 1) * (1 - alpha)))
    if k > m:
        return np.nan
    return float(np.partition(cal, k - 1)[k - 1] * scale)


def regime_conformal(loss, plug_vol, g, alpha, cal_window=2000, min_cal=250):
    """Regime-conditional (Mondrian) conformal band.

    Day t is calibrated on past scores from its own regime within the last
    cal_window days. When that set has fewer than min_cal scores, all past scores
    in the window are used instead.
    """
    s = loss / plug_vol
    n = len(s)
    b = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - cal_window)
        idx = np.arange(lo, t)
        if g[t] >= 0:
            sel = idx[g[idx] == g[t]]
        else:
            sel = idx
        cal = s[sel]
        cal = cal[~np.isnan(cal)]
        if len(cal) < min_cal:
            cal = s[lo:t]
            cal = cal[~np.isnan(cal)]
        b[t] = _kth_smallest_band(cal, alpha, plug_vol[t], min_cal)
    return b
