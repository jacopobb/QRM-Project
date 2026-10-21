"""Point 2 - descriptive analysis (full sample)."""
import pandas as pd, numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

def summary_table(r):
    jb = r.apply(lambda s: stats.jarque_bera(s)[1])
    lb_r  = r.apply(lambda s: acorr_ljungbox(s, lags=[10]).iloc[0, 1])
    lb_r2 = r.apply(lambda s: acorr_ljungbox(s**2, lags=[10]).iloc[0, 1])
    arch  = r.apply(lambda s: het_arch(s, nlags=10)[1])
    ext   = (r.sub(r.mean()).abs() > 3 * r.std()).sum()
    return pd.DataFrame({"mean": r.mean(), "std": r.std(), "skew": r.skew(),
                         "ex_kurt": r.kurt(), "JB_p": jb, "LB10_p": lb_r,
                         "LB10_sq_p": lb_r2, "ARCH10_p": arch, "n_|3sigma|": ext})

def rolling_moments(r, w=250):
    """Rolling mean/var/skew/kurt (plot per asset) and average pairwise correlation."""
    out = {"mean": r.rolling(w).mean(), "var": r.rolling(w).var(),
           "skew": r.rolling(w).skew(), "kurt": r.rolling(w).kurt()}
    avg_corr = r.rolling(w).corr().groupby(level=0).apply(
        lambda m: (m.values.sum() - len(m)) / (len(m) ** 2 - len(m)))
    return out, avg_corr
# TODO: ACF plots, QQ-plots, correlation heatmap (seaborn), 3-sigma event dates
