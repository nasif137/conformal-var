"""Coverage backtests and the moving-block bootstrap.

A hit is a day whose loss exceeds the band. The Kupiec test checks the hit
rate; the Christoffersen test adds a first-order Markov test for clustering.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def coverage_width(loss, band, valid):
    """Coverage, mean band width and number of days on the valid set."""
    l, b = loss[valid], band[valid]
    return float(np.mean(l <= b)), float(np.mean(b)), int(valid.sum())


def common_valid(loss, bands):
    """Days on which the loss and every band are defined."""
    valid = ~np.isnan(loss)
    for b in bands:
        valid &= ~np.isnan(b)
    return valid


def _ll(k, m, p):
    if m == 0:
        return 0.0
    p = min(max(p, 1e-12), 1 - 1e-12)
    return k * np.log(p) + (m - k) * np.log(1 - p)


def kupiec_p(hit, alpha):
    """Kupiec unconditional-coverage likelihood-ratio test, chi-squared(1)."""
    n = len(hit)
    x = int(hit.sum())
    pi = x / n if n else 0.0
    lr = -2 * (_ll(x, n, alpha) - _ll(x, n, pi))
    return float(1 - stats.chi2.cdf(lr, 1))


def christoffersen_cc_p(hit, alpha):
    """Christoffersen conditional-coverage test, chi-squared(2).

    A small p-value flags a wrong hit rate, clustered hits, or both.
    """
    n = len(hit)
    x = int(hit.sum())
    pi = x / n if n else 0.0
    lr_uc = -2 * (_ll(x, n, alpha) - _ll(x, n, pi))
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        a, c = int(hit[i - 1]), int(hit[i])
        if a == 0 and c == 0:
            n00 += 1
        elif a == 0 and c == 1:
            n01 += 1
        elif a == 1 and c == 0:
            n10 += 1
        else:
            n11 += 1
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    den = n00 + n01 + n10 + n11
    pi_ = (n01 + n11) / den if den else 0.0
    ll_ind = _ll(n01, n00 + n01, pi01) + _ll(n11, n10 + n11, pi11)
    lr_ind = -2 * (_ll(n01 + n11, den, pi_) - ll_ind)
    return float(1 - stats.chi2.cdf(lr_uc + lr_ind, 2))


def block_indices(n, block, rng):
    """Indices of one moving-block bootstrap resample of length n."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def bootstrap_ci(indicators, pairs, block, B, rng):
    """95% moving-block bootstrap intervals for coverage and coverage differences.

    indicators : dict name -> 0/1 coverage indicator array, all the same length
    pairs      : list of (a, b); the interval is for coverage(a) - coverage(b)
    Returns ({name: (point, lo, hi)}, {(a, b): (mean, lo, hi)}).
    """
    names = list(indicators)
    n = len(next(iter(indicators.values())))
    point = {k: float(np.mean(v)) for k, v in indicators.items()}
    boot = {k: np.empty(B) for k in names}
    diffs = {p: np.empty(B) for p in pairs}
    for bi in range(B):
        idx = block_indices(n, block, rng)
        means = {k: float(np.mean(indicators[k][idx])) for k in names}
        for k in names:
            boot[k][bi] = means[k]
        for (a, c) in pairs:
            diffs[(a, c)][bi] = means[a] - means[c]
    ci = {k: (point[k], float(np.percentile(boot[k], 2.5)),
              float(np.percentile(boot[k], 97.5))) for k in names}
    dci = {p: (float(np.mean(d)), float(np.percentile(d, 2.5)),
               float(np.percentile(d, 97.5))) for p, d in diffs.items()}
    return ci, dci
