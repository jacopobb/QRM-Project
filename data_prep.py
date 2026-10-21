"""Point 1 - load LSEG prices, remove holidays, compute log returns."""
import numpy as np, pandas as pd
from config import *

def load_clean_prices():
    px = pd.read_csv(RAW_FILE, index_col=0, parse_dates=True).sort_index()[TICKERS]
    px = px.loc[START:END]
    px = px.dropna(how="all")                       # days with no quote at all
    # holiday filter: days where (almost) every price is unchanged = market closed
    unchanged = (px.diff() == 0).mean(axis=1) > 0.9
    px = px[~unchanged]
    return px.ffill().dropna()                      # isolated gaps: forward fill, then drop

def log_returns(px):
    return (100 * np.log(px).diff()).dropna()       # in %, helps the arch optimiser

def make_port(rets, seed=SEED):
    """Point 3 - positive random weights (exponential), normalised to 1, fixed seed."""
    rng = np.random.default_rng(seed)
    w = rng.exponential(size=rets.shape[1]); w /= w.sum()
    w = pd.Series(w, index=rets.columns, name="weight")
    return w, (rets * w).sum(axis=1).rename("PORT")  # buy-and-hold weights, daily weighted return
