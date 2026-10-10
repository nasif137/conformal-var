# Conformal recalibration of portfolio VaR

Code for the paper's experiments. Every table and figure is produced by one
script in `src/`, using public daily returns from the Ken French Data Library.

## Setup

Python 3.10 or later.

```
pip install -r requirements.txt
```

The first run downloads three panels (FF49 and FF30 industry portfolios, and
the 100 size and book-to-market portfolios) and caches them in `data/`. Later
runs work offline. Results depend on the vintage of the library files; the
paper uses data through July 2026.

## Reproducing the paper

```
python reproduce.py
```

runs every script and writes each printout to `results/<script>.txt` and each
figure to `figures/`. A single script runs with `python -m src.<script>` from
this directory.

| Paper              | Script                  | Output                                  |
|--------------------|-------------------------|-----------------------------------------|
| Table 1, Table 6   | `src/run_main.py`       | printed                                 |
| Table 2            | `src/run_horizon.py`    | printed                                 |
| Table 3            | `src/run_scarce.py`     | printed                                 |
| Table 4            | `src/run_regime.py`     | printed                                 |
| Table 5            | `src/run_panels.py`     | printed                                 |
| Table 7            | `src/run_longonly.py`   | printed                                 |
| Figures 1, 2, 4    | `src/make_figures.py`   | `fig1_plugin_vs_realized.png`, `fig3_estimation_error_scaling.png`, `fig5_band_width_cost.png` |
| Figure 3           | `src/make_horizon_fig.py` | `fig_horizon.png`                     |
| Figure 5           | `src/make_stress_fig.py`  | `fig_stress_99.png`                   |

`run_panels.py` and `run_longonly.py` are the slowest (a few minutes each, mostly
the 100-asset panel).

The bootstrap is seeded from `config.SEED`, so repeated runs on the same data
give identical numbers.

## Layout

```
config.py          seed and output directories
src/data.py        download, parse and cache the Ken French panels
src/backtest.py    rolling global minimum-variance backtest; out-of-sample
                   (date, plug_vol, loss) series
src/conformal.py   split, adaptive, multi-day and regime-conditional conformal bands
src/baselines.py   plug-in, historical simulation, EWMA and FHS bands
src/metrics.py     Kupiec and Christoffersen tests, moving-block bootstrap
src/run_*.py       experiments (tables)
src/make_*.py      figures
```

All bands for day t use only data from days before t.
