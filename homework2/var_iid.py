"""Homework 2, Point 1 - unconditional VaR on a rolling window of 5 years (W_ROLL days):
    M01 Historical Simulation, M02 Normal, M03 Student-t, M04 Generalized Hyperbolic, M05 Cornish-Fisher,
    M20 EWMA mean and variance with Gaussian quantile.
For every day t of the backtesting sample, the VaR uses only the W_ROLL returns before t (no look-ahead).
VaR is reported as a return quantile (negative number, in %): a violation is ret_t < VaR_t.
Output: outputs/var/iid/<series>.csv (VaR) and outputs/params/iid/<series>.csv (estimated parameters over time).

    python var_iid.py               # all series
    python var_iid.py EWP SMI       # only some series
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats, optimize
from settings_hw2 import *
from common import load_returns, backtest_dates, save_var, ensure_dirs, run_parallel

warnings.filterwarnings("ignore", category=RuntimeWarning)
SCRIPT = "iid"


# ------------------------------------------------------------------ single models
def cornish_fisher_z(a, S, K):
    """Cornish-Fisher expansion of the standard normal quantile z_a with skewness S and EXCESS kurtosis K."""
    z = stats.norm.ppf(a)
    return z + (z**2 - 1) * S / 6 + (z**3 - 3 * z) * K / 24 - (2 * z**3 - 5 * z) * S**2 / 36


def fit_t(x, start=None):
    """Student-t MLE (df, loc, scale). Warm start from the previous day's estimate: same optimum, much faster."""
    if start is None:
        return stats.t.fit(x)
    return stats.t.fit(x, start[0], loc=start[1], scale=start[2])


def _gh_nll(th, x):
    """Negative log-likelihood of the GH (scipy parametrisation p, a, b, loc, scale), on unconstrained parameters:
    a = exp(la) > 0, b = a*tanh(lb) so that |b| < a, scale = exp(ls) > 0."""
    p, la, lb, loc, ls = th
    a = np.exp(la)
    v = stats.genhyperbolic.logpdf(x, p, a, a * np.tanh(lb), loc=loc, scale=np.exp(ls))
    return -v.sum() if np.all(np.isfinite(v)) else 1e10


def fit_gh(x, start=None):
    """Generalized Hyperbolic MLE. First day: scipy's fit. Next days: Nelder-Mead started from the previous estimate
    (checked on our data: same log-likelihood as scipy's fit, about 3x faster). Falls back to scipy if it fails."""
    if start is not None:
        p, a, b, loc, s = start
        th0 = [p, np.log(a), np.arctanh(np.clip(b / a, -0.999, 0.999)), loc, np.log(s)]
        res = optimize.minimize(_gh_nll, th0, args=(x,), method="Nelder-Mead",
                                options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 1000})
        if res.success and res.fun < 1e9:
            p, la, lb, loc, ls = res.x
            a = np.exp(la)
            return p, a, a * np.tanh(lb), loc, np.exp(ls)
    return stats.genhyperbolic.fit(x)


def ewma_mean_var(x, lam=EWMA_LAMBDA, n_init=60):
    """EWMA forecasts for day t+1 made with data up to t:
        m_{t+1} = lam*m_t + (1-lam)*x_t ,   s2_{t+1} = lam*s2_t + (1-lam)*(x_t - m_t)^2.
    Started with the mean/variance of the first n_init days. The weight of the starting value after 5 years is
    lam^1260 ~ 1e-34, so running the recursion on the whole sample is numerically identical to restarting it
    on each 5-year rolling window (and far faster)."""
    m = np.empty(len(x) + 1); s2 = np.empty(len(x) + 1)
    m[0], s2[0] = x[:n_init].mean(), x[:n_init].var(ddof=1)
    for t in range(len(x)):
        s2[t + 1] = lam * s2[t] + (1 - lam) * (x[t] - m[t]) ** 2
        m[t + 1] = lam * m[t] + (1 - lam) * x[t]
    return m[1:], s2[1:]           # element t = forecast for day t+1


# ------------------------------------------------------------------ one series
def run_series(name):
    t0 = time.time()
    r = load_returns()[name].dropna()
    dates = backtest_dates(r.to_frame())
    pos = r.index.get_indexer(dates)                        # position of each backtest day in the full series
    if pos[0] < W_ROLL:
        raise ValueError(f"{name}: not enough history before {BT_START} for a {W_ROLL}-day window")
    x = r.values
    win = sliding_window_view(x, W_ROLL)[pos - W_ROLL]      # row k = the W_ROLL returns before backtest day k

    out = pd.DataFrame(index=dates)
    # M01 Historical Simulation: empirical alpha-quantile of the window
    for a in ALPHAS:
        out[f"M01_HS_{a}"] = np.quantile(win, a, axis=1)
    # M02 Normal and M05 Cornish-Fisher: sample moments of the window
    mu, sd = win.mean(axis=1), win.std(axis=1, ddof=1)
    S, K = stats.skew(win, axis=1, bias=False), stats.kurtosis(win, axis=1, bias=False)   # K = excess kurtosis
    for a in ALPHAS:
        out[f"M02_N_{a}"] = mu + sd * stats.norm.ppf(a)
        out[f"M05_CF_{a}"] = mu + sd * cornish_fisher_z(a, S, K)

    # M03 Student-t and M04 GH: MLE every REFIT days, parameters kept in between
    par_t, par_gh, q_gh = [], [], []
    pt = pg = None
    for k in range(len(dates)):
        if k % REFIT == 0:
            pt = fit_t(win[k], pt)
            pg = fit_gh(win[k], pg)
            qg = stats.genhyperbolic.ppf(ALPHAS, *pg[:3], loc=pg[3], scale=pg[4])   # GH quantile: slow, only on refit days
        par_t.append(pt); par_gh.append(pg); q_gh.append(qg)
    par_t, par_gh, q_gh = np.array(par_t), np.array(par_gh), np.array(q_gh)
    for j, a in enumerate(ALPHAS):
        out[f"M03_T_{a}"] = stats.t.ppf(a, par_t[:, 0], loc=par_t[:, 1], scale=par_t[:, 2])
        out[f"M04_GH_{a}"] = q_gh[:, j]

    # M20 EWMA (mean and variance), Gaussian quantile
    m, s2 = ewma_mean_var(x)
    m_bt, s_bt = m[pos - 1], np.sqrt(s2[pos - 1])           # forecasts made at t-1 for day t
    for a in ALPHAS:
        out[f"M20_EWMA_{a}"] = m_bt + s_bt * stats.norm.ppf(a)

    save_var(SCRIPT, name, out)
    params = pd.DataFrame(np.c_[mu, sd, S, K, par_t, par_gh, m_bt, s_bt], index=dates,
                          columns=["mean", "std", "skew", "exkurt", "t_df", "t_loc", "t_scale",
                                   "gh_p", "gh_a", "gh_b", "gh_loc", "gh_scale", "ewma_mean", "ewma_sd"])
    os.makedirs(os.path.join(OUT_DIR, "params", SCRIPT), exist_ok=True)
    params.to_csv(os.path.join(OUT_DIR, "params", SCRIPT, f"{name}.csv"))
    print(f"  {name:16s} done in {time.time() - t0:6.1f}s", flush=True)
    return name


if __name__ == "__main__":
    ensure_dirs()
    series = sys.argv[1:] or list(load_returns().columns)
    print(f"var_iid: models M01-M05, M20 on {len(series)} series, window {W_ROLL}, refit every {REFIT} day(s), {N_JOBS} processes")
    run_parallel(run_series, series)
