"""Homework 2, Point 1 - conditional VaR, models M09-M15: AR(1)-GJR-GARCH(1,1) on a rolling window of W_ROLL days.
The same conditional mean/volatility is combined with different quantiles of the standardised innovations z:
    M09 Normal                       (parameters by Gaussian QML)
    M10 standardised Student-t       (parameters by Student-t MLE, df estimated)
    M11 Historical Simulation of z   (= Filtered Historical Simulation, 1 day)
    M12 Cornish-Fisher on the moments of z
    M13 GEV on the monthly block maxima of the losses -z, M14 the same with the extremal index
    M15 GPD on the excesses of -z over its GPD_U_Q quantile
M11-M15 use the residuals z of the Gaussian QML fit on the window (W_ROLL - 1 values).
VaR_t = mu_t + sigma_t * q_alpha(z), mu_t and sigma_t forecast at t-1 with data of the window only.
Also writes, for Point 4, the Expected Shortfall and the PIT u_t = F_t(r_t) of the GARCH-based models
(outputs/es_inputs/garch/<series>.csv).

    python var_garch.py             # all series
    python var_garch.py EWP SMI
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import genextreme
from settings_hw2 import *
from common import load_returns, backtest_dates, save_var, ensure_dirs, run_parallel
import garch_core as gc                                      # AR(1)-GJR-GARCH estimation and filtering
from var_iid import cornish_fisher_z                         # same Cornish-Fisher formula as M05
from var_evt import block_maxima, extremal_index, fit_gev, fit_gpd, var_gpd   # same EVT tools as M06-M08

warnings.filterwarnings("ignore")
SCRIPT = "garch"


def gpd_es_loss(q, u, xi, beta):
    """ES of the loss beyond its quantile q under the GPD tail (needs xi < 1)."""
    return q / (1 - xi) + (beta - xi * u) / (1 - xi) if xi < 1 else np.nan


def gpd_tail_prob(x, u, xi, beta, n, N_u):
    """P(L > x) for x > u from the GPD tail estimator."""
    return N_u / n * (1 + xi * (x - u) / beta) ** (-1 / xi) if abs(xi) > 1e-8 else N_u / n * np.exp(-(x - u) / beta)


def run_series(name):
    t0 = time.time()
    r = load_returns()[name].dropna()
    dates = backtest_dates(r.to_frame())
    pos = r.index.get_indexer(dates)
    x = r.values

    var_rows, es_rows, par_rows = [], [], []                 # VaR, ES/PIT inputs of Point 4, parameters
    thN = thT = pg = pp = None                               # previous estimates (warm start)
    for k, i in enumerate(pos):
        w = x[i - W_ROLL:i]                                  # rolling window: the W_ROLL returns before day t
        if k % REFIT == 0:                                   # re-estimation day
            thN = gc.fit(w, "normal", thN)                   # Gaussian QML (M09, M11-M15)
            thT = gc.fit(w, "t", thT)                        # Student-t MLE (M10)
        z, mu, sig = gc.forecast(thN, w)                     # Gaussian QML: residuals of the window + forecast for day t
        _, muT, sigT = gc.forecast(thT, w)                   # Student-t model: forecast for day t
        nu = thT[6]                                          # degrees of freedom
        L = -z                                               # losses of the standardised residuals (for EVT)
        if k % REFIT == 0:
            pg = fit_gev(block_maxima(L), pg)                # GEV on monthly maxima of -z (M13, M14)
            theta = extremal_index(L)                        # extremal index of -z (M14)
            u = np.quantile(L, GPD_U_Q)                      # GPD threshold (M15)
            pp = fit_gpd(L, u, pp)                           # GPD on the excesses of -z
            N_u = int((L > u).sum())                         # number of exceedances
        S, K = stats.skew(z, bias=False), stats.kurtosis(z, bias=False)   # moments of z (M12)
        rt = x[i]                                            # realised return of day t (used only for the PIT)
        zr = (rt - mu) / sig                                 # realised standardised residual
        vrow, erow = {}, {}
        for a in ALPHAS:
            p = 1 - a
            qz_hs = np.quantile(z, a)                        # empirical quantile of z
            q_gpd = var_gpd(u, pp[0], pp[1], len(L), N_u, a)    # loss quantile of z
            vrow[f"M09_AG_N_{a}"] = mu + sig * stats.norm.ppf(a)                                   # (9)
            vrow[f"M10_AG_T_{a}"] = muT + sigT * gc.std_t_ppf(a, nu)                               # (10)
            vrow[f"M11_AG_HS_{a}"] = mu + sig * qz_hs                                              # (11) FHS
            vrow[f"M12_AG_CF_{a}"] = mu + sig * cornish_fisher_z(a, S, K)                          # (12)
            vrow[f"M13_AG_GEV_{a}"] = mu - sig * genextreme.ppf(p ** BLOCK, *pg)                   # (13)
            vrow[f"M14_AG_GEVT_{a}"] = mu - sig * genextreme.ppf(p ** (BLOCK * theta), *pg) if np.isfinite(theta) else np.nan   # (14)
            vrow[f"M15_AG_GPD_{a}"] = mu - sig * q_gpd                                             # (15)
            # Expected Shortfall (return units, negative) for Point 4
            erow[f"ES_AG_N_{a}"] = mu - sig * stats.norm.pdf(stats.norm.ppf(a)) / a                # Gaussian ES
            erow[f"ES_AG_T_{a}"] = muT + sigT * gc.std_t_es(a, nu)                                 # Student-t ES
            erow[f"ES_AG_HS_{a}"] = mu + sig * z[z <= qz_hs].mean()                                # FHS ES: mean of z in the tail
            erow[f"ES_AG_GPD_{a}"] = mu - sig * gpd_es_loss(q_gpd, u, pp[0], pp[1])                # GPD ES
        # PIT u_t = F_t(r_t) of each model, and the scale sigma_t (McNeil-Frey test)
        erow["PIT_AG_N"] = stats.norm.cdf(zr)
        erow["PIT_AG_T"] = gc.std_t_cdf((rt - muT) / sigT, nu)
        erow["PIT_AG_HS"] = (z <= zr).mean()                                                       # empirical cdf of z
        erow["PIT_AG_GPD"] = gpd_tail_prob(-zr, u, pp[0], pp[1], len(L), N_u) if -zr > u else (z <= zr).mean()   # GPD in the tail
        erow["SIG_AG_N"] = erow["SIG_AG_HS"] = erow["SIG_AG_GPD"] = sig
        erow["SIG_AG_T"] = sigT
        var_rows.append(vrow); es_rows.append(erow)
        par_rows.append(dict(zip([f"N_{n}" for n in gc.PARAM_NAMES], thN)) | dict(zip([f"T_{n}" for n in gc.PARAM_NAMES + ["nu"]], thT))
                        | {"mu": mu, "sigma": sig, "z_skew": S, "z_exkurt": K, "gev_xi": -pg[0], "theta": theta,
                           "gpd_u": u, "gpd_xi": pp[0], "gpd_beta": pp[1]})   # parameters of the day

    save_var(SCRIPT, name, pd.DataFrame(var_rows, index=dates))   # -> outputs/var/garch/<series>.csv
    for sub, rows in (("es_inputs", es_rows), ("params", par_rows)):   # ES/PIT inputs (Point 4) and parameters
        d = os.path.join(OUT_DIR, sub, SCRIPT); os.makedirs(d, exist_ok=True)
        pd.DataFrame(rows, index=pd.Index(dates, name="Date")).to_csv(os.path.join(d, f"{name}.csv"))
    print(f"  {name:16s} done in {time.time() - t0:6.1f}s", flush=True)
    return name


if __name__ == "__main__":
    ensure_dirs()
    series = sys.argv[1:] or list(load_returns().columns)
    print(f"var_garch: models M09-M15 on {len(series)} series, AR(1)-GJR-GARCH(1,1), window {W_ROLL}, refit every {REFIT} day(s)")
    run_parallel(run_series, series)
