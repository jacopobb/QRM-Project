"""Homework 2 - shared helpers: data loading (16 stocks + EWP + market index), backtest dates, saving/loading VaR."""
import os, glob, warnings                                    # file handling and warnings
import numpy as np                                           # numerical arrays
import pandas as pd                                          # time series / tables
from settings_hw2 import *                                   # paths, sample dates, names


# ------------------------------------------------------------------ data
def load_index():
    """Market index prices -> log returns in % (same unit as the stock returns). None if the file is not there yet."""
    if not os.path.exists(INDEX_FILE):                       # index not downloaded yet
        warnings.warn(f"{INDEX_FILE} not found: the market index is skipped until it is downloaded.")
        return None                                          # the other series still run
    px = pd.read_csv(INDEX_FILE, index_col=0, parse_dates=True).sort_index().iloc[:, 0].astype(float)  # first column = Close
    px = px[~px.index.duplicated()].dropna()                 # drop duplicated dates and missing prices
    return (100 * np.log(px).diff()).dropna().rename(INDEX_NAME)   # daily log return in %


def load_returns(with_index=True):
    """Daily log returns in %: the 16 stocks (company names), the EWP and, if available, the market index.
    EWP = simple average of the 16 log returns, as in HW1 (data_prep.equal_weighted_portfolio).
    The index is aligned to the trading days of the stocks (inner join), so all series share the same dates."""
    r = pd.read_csv(RET_FILE, index_col=0, parse_dates=True).sort_index()   # HW1 returns, one column per RIC
    r = r.rename(columns=RIC_TO_NAME)[list(RIC_TO_NAME.values())]           # RIC -> company name, fixed column order
    r[EWP] = r.mean(axis=1)                                                  # equally weighted portfolio (weights 1/16)
    if with_index:                                                           # add the market index when requested
        ix = load_index()
        if ix is not None:
            r = r.join(ix, how="inner")                                      # keep only days on which both exist
    return r


def backtest_dates(r):
    """Trading days of the backtesting sample 1/10/2021 - 30/09/2026."""
    return r.loc[BT_START:BT_END].index                                      # inclusive on both ends


# ------------------------------------------------------------------ VaR storage
def save_var(script, series, df):
    """Writes the VaR forecasts of one script for one series in outputs/var/<script>/<series>.csv.
    Columns: <model code>_<alpha> (return quantile, negative number, in %)."""
    d = os.path.join(VAR_DIR, script)                        # one folder per script: no write conflicts between scripts
    os.makedirs(d, exist_ok=True)
    df.index.name = "Date"
    df.to_csv(os.path.join(d, f"{series}.csv"))


def load_var(series, r=None):
    """All VaR forecasts available for one series (every script), plus the realised return in column 'ret'."""
    files = sorted(glob.glob(os.path.join(VAR_DIR, "*", f"{series}.csv")))  # outputs of var_iid, var_evt, var_garch, ...
    parts = [pd.read_csv(f, index_col=0, parse_dates=True) for f in files]
    if not parts:
        raise FileNotFoundError(f"no VaR forecasts for {series} in {VAR_DIR}")
    out = pd.concat(parts, axis=1)                           # side by side, aligned on the date
    out = out[sorted(out.columns)]                           # columns ordered by model code (M01 ... M21)
    if r is None:
        r = load_returns()
    out.insert(0, "ret", r[series].reindex(out.index))       # realised return of each backtest day
    return out


def ensure_dirs():
    for d in (VAR_DIR, TABLE_DIR, FIG_DIR, LOG_DIR):         # create the output folders if missing
        os.makedirs(d, exist_ok=True)


def run_parallel(func, items, n_jobs=N_JOBS):
    """Runs func(item) for every item, in parallel processes (one series per process). Results keep the input order.
    Each series is independent of the others, so the numbers do not depend on the number of processes."""
    if n_jobs <= 1 or len(items) == 1:
        return [func(x) for x in items]                      # sequential run
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(func, items))                     # parallel run, same order as items
