"""Figures 1, 2 and 4, and the values quoted in the text.

FF49 industry portfolios at three estimation windows, sample and Ledoit-Wolf
covariance, 95% level.

  fig1_plugin_vs_realized.png        claimed (plug-in) and realized volatility
  fig3_estimation_error_scaling.png  coverage against N/T
  fig5_band_width_cost.png           conformal band width, sample and Ledoit-Wolf

    python -m src.make_figures
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import config
from src.backtest import run_backtest
from src.conformal import conformal_recalibrate
from src.data import load_ff

WINDOWS = (60, 120, 252)
HEADLINE_WIN = 120
ALPHA = 0.05
CAL_WINDOW, MIN_CAL = 750, 250
Z = float(stats.norm.ppf(1 - ALPHA))


def coverage_panel(R, est, window):
    """Plug-in and conformal coverage, and conformal width, on the valid days."""
    out = run_backtest(R, estimator=est, window=window)
    oos = out["oos"]
    c = conformal_recalibrate(oos, ALPHA, CAL_WINDOW, MIN_CAL)
    v = c["valid_mask"]
    loss = oos["loss"].to_numpy()
    plug = Z * oos["plug_vol"].to_numpy()
    return {
        "plug_cov": float(np.mean(loss[v] <= plug[v])),
        "conf_cov": float(np.mean(loss[v] <= c["cal_bound"][v])),
        "conf_width": float(np.mean(c["cal_bound"][v])),
        "summary": out["summary"],
    }


def fig1_gap(panels, outdir):
    ests = ["sample", "ledoit_wolf"]
    pred = [panels[(HEADLINE_WIN, e)]["summary"]["predicted_vol"] for e in ests]
    real = [panels[(HEADLINE_WIN, e)]["summary"]["realized_vol"] for e in ests]
    print(f"fig1 (window {HEADLINE_WIN}): realized / claimed daily vol")
    for e, p, r in zip(ests, pred, real):
        print(f"  {e:<12} claimed {p:.5f}  realized {r:.5f}  ratio {r / p:.2f}")
    x = np.arange(len(ests))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, pred, w, label="Plug-in (claimed)")
    ax.bar(x + w / 2, real, w, label="Realized (out of sample)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Sample cov", "Ledoit-Wolf"])
    ax.set_ylabel("Daily volatility")
    ax.set_ylim(0, max(real) * 1.25)
    ax.legend(loc="upper center", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_plugin_vs_realized.png"), dpi=150)
    plt.close(fig)


def fig3_scaling(panels, n_assets, outdir):
    n_over_t = [n_assets / w for w in WINDOWS]
    plug_s = [panels[(w, "sample")]["plug_cov"] for w in WINDOWS]
    plug_l = [panels[(w, "ledoit_wolf")]["plug_cov"] for w in WINDOWS]
    conf_l = [panels[(w, "ledoit_wolf")]["conf_cov"] for w in WINDOWS]
    order = np.argsort(n_over_t)
    x = np.array(n_over_t)[order]
    print("fig3: 95% coverage by N/T")
    for i in order:
        print(f"  N/T {n_over_t[i]:.2f}  plug-in sample {plug_s[i]:.3f}  "
              f"plug-in LW {plug_l[i]:.3f}  conformal LW {conf_l[i]:.3f}")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, np.array(plug_s)[order], "o-", label="Plug-in, sample cov")
    ax.plot(x, np.array(plug_l)[order], "s-", label="Plug-in, Ledoit-Wolf")
    ax.plot(x, np.array(conf_l)[order], "^-", label="Conformal (Ledoit-Wolf)")
    ax.axhline(0.95, ls="--", color="k")
    ax.set_xlabel("N / T  (assets over sample length)")
    ax.set_ylabel("Empirical coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_estimation_error_scaling.png"), dpi=150)
    plt.close(fig)


def fig5_band_width(panels, n_assets, outdir):
    s_w = [panels[(w, "sample")]["conf_width"] for w in WINDOWS]
    l_w = [panels[(w, "ledoit_wolf")]["conf_width"] for w in WINDOWS]
    print("fig5: mean conformal 95% band width")
    for w, s, l in zip(WINDOWS, s_w, l_w):
        print(f"  window {w:>3} (N/T {n_assets / w:.2f})  sample {s:.5f}  LW {l:.5f}  "
              f"LW narrower by {100 * (1 - l / s):.1f}%")
    x = np.arange(len(WINDOWS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, s_w, w, label="Sample cov + conformal")
    ax.bar(x + w / 2, l_w, w, label="Ledoit-Wolf + conformal")
    ax.set_xticks(x)
    ax.set_xticklabels([f"win {w_}\nN/T={n_assets / w_:.2f}" for w_ in WINDOWS])
    ax.set_ylabel("Mean certified band width (95%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig5_band_width_cost.png"), dpi=150)
    plt.close(fig)


def main():
    config.set_seeds()
    R = load_ff("ff49_industry", start="1970-01-01")
    n_assets = R.shape[1]
    print(f"panel: {R.shape[0]} days x {n_assets} industries\n")
    panels = {(w, e): coverage_panel(R, e, w)
              for w in WINDOWS for e in ("sample", "ledoit_wolf")}
    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    fig1_gap(panels, config.FIGURE_DIR)
    fig3_scaling(panels, n_assets, config.FIGURE_DIR)
    fig5_band_width(panels, n_assets, config.FIGURE_DIR)


if __name__ == "__main__":
    main()
