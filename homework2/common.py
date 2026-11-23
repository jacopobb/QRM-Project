"""Homework 2 - shared helpers: data loading (16 stocks + EWP + market index), backtest dates, saving/loading VaR."""
import os, glob, warnings
import numpy as np
import pandas as pd
from settings_hw2 import *


# ------------------------------------------------------------------ data
def load_index():
    """Market index prices -> log returns in % (same unit as the stock returns). None if the file is not there yet."""
    if not os.path.exists(INDEX_FILE):
        warnings.warn(f"{INDEX_FILE} not found: the market index is skipped until it is downloaded.")
        return None
    px = pd.read_csv(INDEX_FILE, index_col=0, parse_dates=True).sort_index().iloc[:, 0].astype(float)
    px = px[~px.index.duplicated()].dropna()
    return (100 * np.log(px).diff()).dropna().rename(INDEX_NAME)


def load_returns(with_index=True):
    """Daily log returns in %: the 16 stocks (company names), the EWP and, if available, the market index.
    EWP = simple average of the 16 log returns, as in HW1 (data_prep.equal_weighted_portfolio).
    The index is aligned to the trading days of the stocks (inner join), so all series share the same dates."""
    r = pd.read_csv(RET_FILE, index_col=0, parse_dates=True).sort_index()
    r = r.rename(columns=RIC_TO_NAME)[list(RIC_TO_NAME.values())]
    r[EWP] = r.mean(axis=1)
    if with_index:
        ix = load_index()
        if ix is not None:
            r = r.join(ix, how="inner")
    return r


def backtest_dates(r):
    """Trading days of the backtesting sample 1/10/2021 - 30/09/2026."""
    return r.loc[BT_START:BT_END].index


# ------------------------------------------------------------------ VaR storage
def save_var(script, series, df):
    """Writes the VaR forecasts of one script for one series in outputs/var/<script>/<series>.csv.
    Columns: <model code>_<alpha> (return quantile, negative number, in %)."""
    d = os.path.join(VAR_DIR, script)
    os.makedirs(d, exist_ok=True)
    df.index.name = "Date"
    df.to_csv(os.path.join(d, f"{series}.csv"))


def load_var(series, r=None):
    """All VaR forecasts available for one series (every script), plus the realised return in column 'ret'."""
    parts = [pd.read_csv(f, index_col=0, parse_dates=True) for f in sorted(glob.glob(os.path.join(VAR_DIR, "*", f"{series}.csv")))]
    if not parts:
        raise FileNotFoundError(f"no VaR forecasts for {series} in {VAR_DIR}")
    out = pd.concat(parts, axis=1)
    out = out[sorted(out.columns)]
    if r is None:
        r = load_returns()
    out.insert(0, "ret", r[series].reindex(out.index))
    return out


def ensure_dirs():
    for d in (VAR_DIR, TABLE_DIR, FIG_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


def run_parallel(func, items, n_jobs=N_JOBS):
    """Runs func(item) for every item, in parallel processes (one series per process). Results keep the input order."""
    if n_jobs <= 1 or len(items) == 1:
        return [func(x) for x in items]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(func, items))
