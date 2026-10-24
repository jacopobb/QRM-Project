"""Point 1 (cleaning) and portfolio weights (Point 3).  Run directly to produce the cleaning report and save clean files."""
import numpy as np, pandas as pd
try:
    from settings import *      # renamed config file
except ModuleNotFoundError:
    from config import *

def load_raw():
    px = pd.read_csv(RAW_FILE, index_col=0, parse_dates=True).sort_index()
    return px[TICKERS].loc[START:END]

def clean_prices(px, verbose=True):
    """Remove non-trading days. Returns clean prices and prints what was removed (use it in the report)."""
    n0 = len(px)
    px = px[px.index.dayofweek < 5]                                  # weekends
    px = px.dropna(how="all")                                        # no quote on any stock (holidays)
    flat = (px.diff() == 0) | px.diff().isna()
    holiday = flat.all(axis=1) & px.diff().notna().any(axis=1)       # every price unchanged = market closed
    px = px[~holiday]
    if verbose:
        print(f"Rows: raw {n0} -> after removing weekends/holidays {len(px)}")
        print("Missing values per series:\n", px.isna().sum().to_string())
        print("First valid date per series:\n", px.apply(lambda s: s.first_valid_index().date()).to_string())
    px = px.ffill(limit=3).dropna()                                  # isolated gaps only, then align to common dates
    if verbose: print(f"Final sample: {px.index.min().date()} -> {px.index.max().date()}, {len(px)} days")
    return px

def log_returns(px):
    return (100 * np.log(px).diff()).dropna()                        # log returns in %

def jump_check(r, thr=15):
    """Flag |daily return| > thr %: possible unadjusted splits/data errors (or real events)."""
    big = r[(r.abs() > thr).any(axis=1)]
    return r.where(r.abs() > thr).stack().rename("ret_%").to_frame() if len(big) else pd.DataFrame()

def make_weights(tickers, seed=SEED):
    """Positive random weights (exponential density), normalised to sum to 1. Same seed -> same weights."""
    rng = np.random.default_rng(seed)
    w = rng.exponential(scale=1.0, size=len(tickers))
    return pd.Series(w / w.sum(), index=tickers, name="weight")

def port_returns(rets, w):
    return (rets[w.index] * w).sum(axis=1).rename("PORT")            # daily weighted average of returns

if __name__ == "__main__":
    px = clean_prices(load_raw())
    rets = log_returns(px)
    print("\nPossible jumps (>15%):\n", jump_check(rets))
    w = make_weights(TICKERS); print("\nPORT weights:\n", w.round(4).to_string(), "\nsum =", w.sum())
    px.to_csv(CLEAN_FILE); rets.to_csv(RET_FILE)
    print("\nSaved", CLEAN_FILE, "and", RET_FILE)
