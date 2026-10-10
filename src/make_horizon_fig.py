"""Figure 3: VaR coverage by forecast horizon, 95% and 99% panels.

Writes figures/fig_horizon.png.

    python -m src.make_horizon_fig
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src.run_horizon import HORIZONS, coverages, load_oos

STYLE = {
    "plug":      dict(color="#999999", marker="o", ls=":",  label="Plug-in"),
    "ewma":      dict(color="#4C78A8", marker="s", ls="--", label="EWMA"),
    "fhs":       dict(color="#E45756", marker="^", ls="-.", label="FHS"),
    "conformal": dict(color="#1B1B1B", marker="D", ls="-",  label="Conformal"),
}


def main():
    config.set_seeds()
    loss, plug_vol, sig = load_oos()

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharex=True)
    for ax, alpha in zip(axes, (0.05, 0.01)):
        series = {k: [] for k in STYLE}
        for h in HORIZONS:
            cov, _ = coverages(loss, plug_vol, sig, h, alpha)
            for k in STYLE:
                series[k].append(cov[k])
        for k, st in STYLE.items():
            ax.plot(HORIZONS, series[k], color=st["color"], marker=st["marker"],
                    ls=st["ls"], lw=1.4, ms=5, label=st["label"])
        ax.axhline(1 - alpha, color="k", lw=0.8, alpha=0.5)
        ax.set_title(f"{round((1 - alpha) * 100)}% VaR")
        ax.set_xlabel("Horizon $h$ (days)")
        ax.set_xticks(HORIZONS)
        ax.grid(True, alpha=0.25, lw=0.5)
    axes[0].set_ylabel("Out-of-sample coverage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    out = os.path.join(config.FIGURE_DIR, "fig_horizon.png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
