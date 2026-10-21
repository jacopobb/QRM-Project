"""Points 3-4 - GARCH estimation, static one-step-ahead forecasts, Gaussian VaR."""
import numpy as np, pandas as pd
from arch import arch_model
from config import *

def spec(r, dist="normal", o=0):
    return arch_model(r, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist=dist)  # o=1 -> GJR

def fit_in_sample(r, dist="normal", o=0):
    res = spec(r.loc[:IN_SAMPLE_END], dist, o).fit(disp="off")
    return res                                    # res.summary() -> parameter significance

def std_resid(res):
    return res.std_resid.dropna()                 # Z, in-sample standardised residuals

def oos_forecasts(r, res, dist="normal", o=0, h=H_REFIT):
    """One-step-ahead mean and sigma for each out-of-sample day.
    h=None: fixed parameters. Otherwise re-estimate every h days on an expanding window."""
    oos = r.loc[IN_SAMPLE_END:].index[1:]
    mu, sig = pd.Series(index=oos, dtype=float), pd.Series(index=oos, dtype=float)
    params = res.params
    for i, t in enumerate(oos):
        if h and i % h == 0:
            params = spec(r.loc[:oos[i - 1] if i else IN_SAMPLE_END], dist, o).fit(disp="off").params
        full = spec(r.loc[:t], dist, o).fix(params)      # parameters fixed, filter on data up to t
        f = full.forecast(horizon=1, start=r.index.get_loc(t) - 1, reindex=False)
        mu[t], sig[t] = f.mean.iloc[-1, 0], np.sqrt(f.variance.iloc[-1, 0])
    return mu, sig
    # speed note: with h=None one fix() on the whole series + conditional_volatility is enough

def var_gaussian(mu, sig, alpha):
    from scipy.stats import norm
    return mu + sig * norm.ppf(alpha)             # VaR Case 1 (return quantile, negative number)
# TODO: Case 2 -> same with t / skew-t quantile (arch.univariate distributions), AIC/BIC choice
