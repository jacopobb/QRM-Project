"""Homework 2, Point 1 - factor-based VaR (stocks and EWP only, NOT the market index):
Responsible: member C (reviewer: B). Bonus models (22+), if any, go in this script.
    mapping   r_i,t = a_i + b_i' f_t + e_i,t   (OLS on the 5-year rolling window, f = Fama-French Europe 5 factors)
    M16 / M17  CCC / DCC(1,1) on the factors, Gaussian:  VaR_i = a_i + b_i' mu_f + z_alpha * sqrt(b_i' Sigma_f b_i + s2_e,i)
    M18 / M19  CCC / DCC + Filtered Historical Simulation: factor innovations devolatilised and decorrelated,
               bootstrapped (jointly with the mapping residuals of the same day) to build N_SIM returns for day t
    M21        X_i,t = b_i' f_t ; AR(1)-GJR-GARCH(1,1) with Hansen skewed-t innovations on X_i, VaR_i = VaR of X_i
Each factor follows an AR(1)-GJR-GARCH(1,1) (Gaussian QML). EWP: weights 1/16 on the stocks, so b_EWP = mean of the
b_i and its residual variance is w' Cov(e) w (full residual covariance of the window).
Factors are in % (simple returns, USD, as published); stock returns are log returns in % (CHF) - see README.

Stage A runs day by day (the factor models are common to all series); Stage B (M21) runs one series per process.
    python var_factors.py
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.signal import lfilter                             # vectorised DCC recursion
from settings_hw2 import *
from common import load_returns, backtest_dates, save_var, ensure_dirs, run_parallel
import garch_core as gc

warnings.filterwarnings("ignore")
SCRIPT = "factors"
N_SIM = 10000                                                # FHS simulations per day
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]             # Fama-French 5 factors (RF not used)
B_FILE = os.path.join(OUT_DIR, "params", SCRIPT, "betas.csv")   # betas of every day, read by Stage B (M21)


# ------------------------------------------------------------------ data
def load_factors(path=FACTORS_FILE):
    """Reads the daily csv of the Kenneth French data library as downloaded (text header, then YYYYMMDD rows)."""
    rows, header = [], None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            cells = [c.strip() for c in line.strip().split(",")]
            if header is None and "Mkt-RF" in cells:         # header line of the daily table
                header = cells[1:]
                continue
            if header is not None:
                if len(cells[0]) == 8 and cells[0].isdigit():    # data line: date YYYYMMDD + values
                    rows.append([cells[0]] + [float(c) for c in cells[1:len(header) + 1]])
                elif rows:
                    break                                    # end of the daily table (copyright line)
    if header is None:
        raise ValueError(f"{path}: header with 'Mkt-RF' not found")
    df = pd.DataFrame(rows, columns=["Date"] + header)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    return df.set_index("Date")[FACTORS]                     # factors in % per day


def joint_data():
    """Stocks (16) and factors on the common trading days. EWP is built from the stocks inside the mapping."""
    r = load_returns(with_index=False)                       # the index is not used in the factor models
    stocks = [c for c in r.columns if c != EWP]
    return r[stocks], load_factors(), r


# ------------------------------------------------------------------ DCC(1,1), second step on the standardised residuals
def dcc_filter(ab, z, Qbar):
    """Q_t = (1-a-b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1}, Q_1 = Qbar. Returns R_1..R_T and R_{T+1}."""
    a, b = ab
    T, n = z.shape
    zz = np.einsum("ti,tj->tij", z, z).reshape(T, n * n)     # outer products z_t z_t', flattened
    x = (1 - a - b) * Qbar.ravel() + a * zz                   # input: Q_{t+1} = x_t + b Q_t
    Q = np.empty((T + 1, n * n)); Q[0] = Qbar.ravel()         # Q_1 = unconditional correlation
    Q[1:] = lfilter([1.0], [1.0, -b], x, axis=0, zi=(b * Qbar.ravel())[None, :])[0]   # whole recursion at once
    Q = Q.reshape(T + 1, n, n)
    d = np.sqrt(np.einsum("tii->ti", Q))                      # square roots of the diagonals
    R = Q / d[:, :, None] / d[:, None, :]                     # R_t = diag(Q)^-1/2 Q diag(Q)^-1/2
    return R[:-1], R[-1]                                      # in-sample correlations, forecast for T+1


def dcc_nll(ab, z, Qbar):
    """Correlation part of the Gaussian log-likelihood (second step of Engle 2002), with minus sign."""
    if ab[0] < 0 or ab[1] < 0 or ab[0] + ab[1] >= 0.999:     # outside the admissible region
        return 1e10
    R, _ = dcc_filter(ab, z, Qbar)
    sign, logdet = np.linalg.slogdet(R)                       # log det R_t for every t
    if np.any(sign <= 0):
        return 1e10
    quad = np.einsum("ti,ti->t", z, np.linalg.solve(R, z[:, :, None])[:, :, 0])   # z_t' R_t^-1 z_t
    return 0.5 * np.mean(logdet + quad)


def fit_dcc(z, start=None):
    """DCC(1,1) parameters (a, b) by MLE, with a + b < 1."""
    Qbar = np.corrcoef(z.T)                                   # correlation targeting
    res = optimize.minimize(dcc_nll, start if start is not None else [0.02, 0.95], args=(z, Qbar), method="SLSQP",
                            bounds=[(0, 0.5), (0, 0.999)], constraints=[{"type": "ineq", "fun": lambda ab: 0.998 - ab[0] - ab[1]}])
    return res.x, Qbar


# ------------------------------------------------------------------ Stage A: M16-M19, day by day
def stage_a(stocks, F, r_all, dates):
    J = stocks.join(F, how="inner")                          # days with both stock returns and factors
    names = list(stocks.columns)
    w = np.full(len(names), 1 / len(names))                  # EWP weights
    th_f = [None] * len(FACTORS); ab = None                  # previous estimates (warm start)
    out, betas, pars = [], [], []
    t0 = time.time()
    for k, t in enumerate(dates):
        win = J.loc[J.index < t].iloc[-W_ROLL:]              # last W_ROLL common days before day t
        Y, Fw = win[names].values, win[FACTORS].values
        # 1) mapping: OLS of every stock on the 5 factors
        X = np.c_[np.ones(len(Fw)), Fw]                      # regressors: constant + factors
        coef = np.linalg.lstsq(X, Y, rcond=None)[0]           # (6, n_stocks)
        A, B = coef[0], coef[1:].T                            # intercepts a_i; betas B: (n_stocks, 5)
        E = Y - X @ coef                                      # mapping residuals e_i,t
        # 2) univariate AR(1)-GJR-GARCH on each factor
        Z, mu_f, sd_f = [], [], []
        for j in range(len(FACTORS)):
            if k % REFIT == 0:
                th_f[j] = gc.fit(Fw[:, j], "normal", th_f[j])   # Gaussian QML on factor j
            z, m, s = gc.forecast(th_f[j], Fw[:, j])          # standardised residuals and forecasts for day t
            Z.append(z); mu_f.append(m); sd_f.append(s)
        Z, mu_f, sd_f = np.array(Z).T, np.array(mu_f), np.array(sd_f)   # Z: (W-1, 5)
        E1 = E[1:]                                              # residuals aligned with Z (AR(1) drops the first day)
        # 3) correlations: CCC (constant) and DCC(1,1)
        R_ccc = np.corrcoef(Z.T)                                # CCC: sample correlation of the standardised residuals
        if k % REFIT == 0:
            ab, Qbar = fit_dcc(Z, ab)                           # DCC parameters
        R_path, R_dcc = dcc_filter(ab, Z, np.corrcoef(Z.T))     # DCC correlations in the window and forecast for day t
        row, s2_e = {}, E.var(axis=0, ddof=1)                   # idiosyncratic variances
        mean_i = A + B @ mu_f                                   # predicted mean of every stock
        rng = np.random.default_rng([SEED, k])                  # same draws whatever the machine
        idx = rng.integers(0, len(Z), N_SIM)                    # bootstrapped days (common to CCC and DCC)
        for lab, R, decor in (("CCC", R_ccc, None), ("DCC", R_dcc, R_path)):
            Sig = np.outer(sd_f, sd_f) * R                      # predicted factor covariance matrix
            v_i = np.einsum("ij,jk,ik->i", B, Sig, B) + s2_e    # stock variance: systematic + idiosyncratic
            bp = w @ B                                          # EWP betas
            v_p = bp @ Sig @ bp + w @ np.cov(E.T) @ w           # EWP variance (full residual covariance)
            # FHS: e_s = L_s^-1 z_s (L_s Cholesky of R_s), f* = mu + sd * (L_{T+1} e_s), r* = a + B f* + resid_s
            Ls = np.linalg.cholesky(R if decor is None else decor[idx])   # CCC: one matrix; DCC: R of each drawn day
            e_s = np.linalg.solve(Ls, Z[idx][:, :, None])[:, :, 0] if decor is not None else np.linalg.solve(Ls, Z[idx].T).T   # decorrelated shocks
            f_sim = mu_f + sd_f * (e_s @ np.linalg.cholesky(R).T)   # simulated factors for day t (correlation of day t)
            r_sim = A + f_sim @ B.T + E1[idx]                   # simulated stock returns (residual of the same drawn day)
            p_sim = r_sim @ w                                   # simulated EWP return
            for a in ALPHAS:
                za = stats.norm.ppf(a)
                g = {"CCC": ("M16_CCC_N", "M18_CCC_FHS"), "DCC": ("M17_DCC_N", "M19_DCC_FHS")}[lab]
                for i, n in enumerate(names):
                    row[(n, f"{g[0]}_{a}")] = mean_i[i] + za * np.sqrt(v_i[i])   # Gaussian VaR (16 / 17)
                    row[(n, f"{g[1]}_{a}")] = np.quantile(r_sim[:, i], a)        # FHS VaR (18 / 19)
                row[(EWP, f"{g[0]}_{a}")] = w @ mean_i + za * np.sqrt(v_p)
                row[(EWP, f"{g[1]}_{a}")] = np.quantile(p_sim, a)
        out.append(row)
        betas.append(np.r_[np.c_[A, B], [[w @ A, *(w @ B)]]].ravel())   # alpha and betas of stocks and EWP
        pars.append(np.r_[ab, [th[5] + th[3] + th[4] / 2 for th in th_f], R_dcc[np.triu_indices(5, 1)]])   # DCC, persistence, correlations
        if k % 100 == 0:
            print(f"  stage A day {k + 1}/{len(dates)} ({t.date()}), {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(out, index=dates)
    res.columns = pd.MultiIndex.from_tuples(res.columns)      # (series, model) columns
    for n in names + [EWP]:
        save_var(SCRIPT, n, res[n].copy())                    # -> outputs/var/factors/<series>.csv
    os.makedirs(os.path.dirname(B_FILE), exist_ok=True)
    cols = pd.MultiIndex.from_product([names + [EWP], ["alpha"] + FACTORS])
    pd.DataFrame(betas, index=pd.Index(dates, name="Date"), columns=cols).to_csv(B_FILE)   # betas over time (also for the report)
    pair = [f"rho_{FACTORS[i]}_{FACTORS[j]}" for i, j in zip(*np.triu_indices(5, 1))]
    pd.DataFrame(pars, index=pd.Index(dates, name="Date"),
                 columns=["dcc_a", "dcc_b"] + [f"persist_{f}" for f in FACTORS] + pair).to_csv(
        os.path.join(OUT_DIR, "params", SCRIPT, "factor_models.csv"))   # factor model parameters over time


# ------------------------------------------------------------------ Stage B: M21, one series per process
def stage_b(name):
    t0 = time.time()
    stocks, F, r_all = joint_data()
    J = stocks.join(F, how="inner")
    dates = backtest_dates(r_all)
    bet = pd.read_csv(B_FILE, header=[0, 1], index_col=0, parse_dates=True)[name]   # betas of this series, every day
    rows, th = [], None
    for k, t in enumerate(dates):
        Fw = J.loc[J.index < t, FACTORS].iloc[-W_ROLL:].values   # factors of the window
        X = Fw @ bet.loc[t, FACTORS].values                     # X = b' f with the betas of the current window
        if k % REFIT == 0:
            th = gc.fit(X, "skewt", th)                         # AR(1)-GJR-GARCH with skewed-t on X
        _, m, s = gc.forecast(th, X)                            # forecast for day t
        rows.append({f"M21_X_SKT_{a}": m + s * gc.skewt_ppf(a, th[6], th[7]) for a in ALPHAS})   # (21)
    d = os.path.join(VAR_DIR, SCRIPT + "_x"); os.makedirs(d, exist_ok=True)
    save_var(SCRIPT + "_x", name, pd.DataFrame(rows, index=dates))   # -> outputs/var/factors_x/<series>.csv
    print(f"  M21 {name:16s} done in {time.time() - t0:6.1f}s", flush=True)
    return name


if __name__ == "__main__":
    ensure_dirs()
    if not os.path.exists(FACTORS_FILE):                      # factors not downloaded yet
        sys.exit(f"ERROR: {FACTORS_FILE} not found. Download 'Europe 5 Factors [Daily]' (CSV) from the Kenneth French "
                 f"data library, unzip it and save the csv with this name.")
    stocks, F, r_all = joint_data()
    dates = backtest_dates(r_all)
    print(f"var_factors: {len(stocks.columns)} stocks + EWP, factors {F.index[0].date()} - {F.index[-1].date()}, "
          f"{len(stocks.join(F, how='inner'))} common days")
    if F.index[-1] < dates[-1] - pd.Timedelta(days=7):        # factors must cover the whole backtest
        print(f"WARNING: the factors end on {F.index[-1].date()}, before the end of the backtest ({dates[-1].date()}).")
    if "--only-m21" not in sys.argv:
        stage_a(stocks, F, r_all, dates)                      # M16-M19
    run_parallel(stage_b, list(stocks.columns) + [EWP])       # M21
