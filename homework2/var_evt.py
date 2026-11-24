"""Homework 2, Point 1 - EVT VaR on the raw returns, i.i.d. assumption, EXPANDING window (all data before day t):
Responsible: member A (reviewer: D).
    M06 GEV on monthly block maxima of the losses, without extremal index (theta = 1)
    M07 GEV with the extremal index theta (blocks estimator)
    M08 GPD on the excesses over a threshold u = GPD_U_Q quantile of the losses.
Losses L = -returns. The loss quantile x_p (p = 1 - alpha) is turned into a return VaR = -x_p (negative number, in %).
Formulas as in HW1 evt.py (course slides): GEV  x_p = G^-1(p^(m*theta)),  G = GEV of the block maxima, m = BLOCK;
GPD  x_p = u + beta/xi * ((alpha / (N_u/T))^(-xi) - 1).
Output: outputs/var/evt/<series>.csv and outputs/params/evt/<series>.csv.

    python var_evt.py               # all series
    python var_evt.py EWP           # only some series
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import genextreme, genpareto                # GEV and GPD (scipy: GEV shape c = -xi)
from settings_hw2 import *
from common import load_returns, backtest_dates, save_var, ensure_dirs, run_parallel

warnings.filterwarnings("ignore", category=RuntimeWarning)
SCRIPT = "evt"


def block_maxima(L, m=BLOCK):
    """Maxima of consecutive blocks of m days; the oldest incomplete block is dropped, so the last block ends at t-1."""
    n = len(L) // m                                          # number of complete blocks
    return L[-n * m:].reshape(n, m).max(axis=1)              # one maximum per block


def extremal_index(L, u_q=THETA_U_Q, k=BLOCK):
    """Blocks estimator: theta = log(1 - G(u)) / (k * log(1 - N(u))), u = u_q quantile of the losses,
    N(u) = share of days above u, G(u) = share of blocks whose maximum is above u. Clipped to (0, 1]."""
    n = len(L) // k
    x = L[-n * k:]                                           # same blocks as the GEV
    u = np.quantile(x, u_q)                                  # threshold
    N_u, G_u = (x > u).mean(), (x.reshape(n, k).max(axis=1) > u).mean()   # exceedance shares: days and blocks
    if G_u >= 1 or N_u <= 0:                                 # estimator not defined
        return np.nan
    return float(min(np.log(1 - G_u) / (k * np.log(1 - N_u)), 1.0))   # theta > 1 means no clustering -> 1


def fit_gev(maxima, start=None):
    """GEV MLE (scipy: shape c = -xi). Warm start from the previous estimate."""
    if start is None:
        return genextreme.fit(maxima)
    return genextreme.fit(maxima, start[0], loc=start[1], scale=start[2])


def fit_gpd(L, u, start=None):
    """GPD MLE on the excesses over u (location fixed at 0). Returns xi, beta."""
    exc = L[L > u] - u                                       # excesses over the threshold
    if start is None:
        xi, _, beta = genpareto.fit(exc, floc=0)
    else:
        xi, _, beta = genpareto.fit(exc, start[0], floc=0, scale=start[1])   # warm start
    return xi, beta


def var_gpd(u, xi, beta, T, N_u, a):
    """Loss quantile at level 1-a from the GPD tail estimator (xi -> 0 limit handled)."""
    if abs(xi) < 1e-8:                                       # exponential tail
        return u - beta * np.log(a / (N_u / T))
    return u + beta / xi * ((a / (N_u / T)) ** (-xi) - 1)    # slide formula


def run_series(name):
    t0 = time.time()
    r = load_returns()[name].dropna()
    dates = backtest_dates(r.to_frame())
    pos = r.index.get_indexer(dates)                         # position of each backtest day
    Lall = -r.values                                         # losses = - returns

    rows, pg, pp = [], None, None                            # pg: GEV parameters, pp: GPD parameters (warm start)
    for k, i in enumerate(pos):
        L = Lall[:i]                                         # expanding window: all losses before day t
        if k % REFIT == 0:                                   # re-estimation day
            M = block_maxima(L)                              # monthly maxima
            pg = fit_gev(M, pg)                              # GEV fit (M06, M07)
            theta = extremal_index(L)                        # extremal index (M07)
            u = np.quantile(L, GPD_U_Q)                      # GPD threshold (M08)
            pp = fit_gpd(L, u, pp)                           # GPD fit
            T, N_u = len(L), int((L > u).sum())              # sample size and number of exceedances
        row = {"n_obs": len(L), "n_blocks": len(L) // BLOCK, "gev_xi": -pg[0], "gev_mu": pg[1], "gev_sigma": pg[2],
               "theta": theta, "gpd_u": u, "gpd_xi": pp[0], "gpd_beta": pp[1], "gpd_Nu": N_u}   # parameters of the day
        for a in ALPHAS:
            p = 1 - a                                        # loss quantile level
            row[f"M06_GEV_{a}"] = -genextreme.ppf(p ** BLOCK, *pg)                    # F(x)=G(x)^(1/m)
            row[f"M07_GEVT_{a}"] = -genextreme.ppf(p ** (BLOCK * theta), *pg) if np.isfinite(theta) else np.nan   # F=G^(1/(m theta))
            row[f"M08_GPD_{a}"] = -var_gpd(u, pp[0], pp[1], T, N_u, a)                # GPD tail quantile
        rows.append(row)

    res = pd.DataFrame(rows, index=dates)
    var_cols = [c for c in res.columns if c.startswith("M0")]   # VaR columns
    save_var(SCRIPT, name, res[var_cols])                    # -> outputs/var/evt/<series>.csv
    os.makedirs(os.path.join(OUT_DIR, "params", SCRIPT), exist_ok=True)
    res.drop(columns=var_cols).to_csv(os.path.join(OUT_DIR, "params", SCRIPT, f"{name}.csv"))   # EVT parameters over time
    print(f"  {name:16s} done in {time.time() - t0:6.1f}s", flush=True)
    return name


if __name__ == "__main__":
    ensure_dirs()
    series = sys.argv[1:] or list(load_returns().columns)
    print(f"var_evt: models M06-M08 on {len(series)} series, expanding window, refit every {REFIT} day(s), {N_JOBS} processes")
    run_parallel(run_series, series)
