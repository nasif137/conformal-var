"""Ken French daily portfolio panels.

Each panel is downloaded from the Ken French Data Library as a zipped CSV and
the first table, the value-weighted daily returns, is parsed. Missing values
(-99.99 and -999) become NaN and percent returns are converted to decimals. The
raw table is cached under config.DATA_DIR, so later calls work offline.

The three panels span cross-sectional sizes N = 30, 49 and 100, so the ratio of
assets to window length N/T can be matched across datasets.
"""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

import config

FF_PANELS = {
    "ff49_industry": "49_Industry_Portfolios_Daily",   # N = 49
    "ff30_industry": "30_Industry_Portfolios_Daily",   # N = 30
    "ff100_sizebm": "100_Portfolios_10x10_Daily",      # N = 100, size x book-to-market
}

_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"


def _parse_ff_csv(text: str) -> pd.DataFrame:
    """Parse the first daily table of a Ken French CSV file."""
    lines = text.splitlines()
    first = None
    for i, ln in enumerate(lines):
        tok = ln.split(",")[0].strip()
        if len(tok) == 8 and tok.isdigit():
            first = i
            break
    if first is None:
        raise ValueError("no daily data rows found in Ken French file")
    h = first - 1
    while h >= 0 and not lines[h].strip():
        h -= 1
    cols = [c.strip() for c in lines[h].split(",")[1:]]
    dates, rows = [], []
    for ln in lines[first:]:
        parts = ln.split(",")
        tok = parts[0].strip()
        if not (len(tok) == 8 and tok.isdigit()):
            break
        vals = parts[1:1 + len(cols)]
        if len(vals) < len(cols):
            vals += ["-99.99"] * (len(cols) - len(vals))
        dates.append(pd.to_datetime(tok, format="%Y%m%d"))
        rows.append([float(x) for x in vals])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=cols)


def _download_panel(panel: str) -> pd.DataFrame:
    url = f"{_BASE}/{FF_PANELS[panel]}_CSV.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        blob = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(blob))
    name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    return _parse_ff_csv(zf.read(name).decode("latin-1"))


def load_ff(panel: str = "ff49_industry", start: str = "1970-01-01", end=None) -> pd.DataFrame:
    """Daily simple returns (decimal) for one panel, between start and end."""
    cache = os.path.join(config.DATA_DIR, f"{panel}_vw_daily.csv")
    raw = None
    if os.path.exists(cache):
        try:
            raw = pd.read_csv(cache, index_col=0, parse_dates=True)
        except Exception:
            raw = None
    if raw is None:
        raw = _download_panel(panel)
        try:
            os.makedirs(config.DATA_DIR, exist_ok=True)
            raw.to_csv(cache)
        except OSError:
            pass  # caching is optional

    df = raw.replace([-99.99, -999.0], np.nan) / 100.0
    df.columns = [str(c).strip() for c in df.columns]
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


if __name__ == "__main__":
    for p in FF_PANELS:
        d = load_ff(p)
        print(f"{p:>16}: {d.shape[0]} days x {d.shape[1]} assets")
