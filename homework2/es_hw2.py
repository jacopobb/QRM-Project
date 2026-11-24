"""Homework 2, Point 4 - Expected Shortfall for the EWP and the market index: five approaches, three backtests,
Responsible: member A (reviewer: D).
a loss function for (VaR, ES) and the Model Confidence Set; full sample and each calendar year.
Approaches (ES and the VaR it is paired with):
  HS      Historical Simulation, rolling 5y: ES = mean of the window returns below the HS VaR          (VaR M01)
  AG_N    AR(1)-GJR-GARCH + Normal:   ES = mu - sigma * phi(z_a)/a                                       (VaR M09)
  AG_T    AR(1)-GJR-GARCH + Student-t: ES of the unit-variance t                                         (VaR M10)
  AG_HS   AR(1)-GJR-GARCH + FHS: ES = mu + sigma * mean of the residuals below their a-quantile           (VaR M11)
  AG_GPD  AR(1)-GJR-GARCH + GPD on the residual losses: ES = mu - sigma*(q/(1-xi) + (beta - xi u)/(1-xi)) (VaR M15)
Tests (all based on the violation days or on the PIT u_t = F_t(r_t) of the model):
  1. McNeil and Frey (2000): on violation days e_t = (r_t - ES_t)/sigma_t has mean 0 if ES is correct.
     One-sided (H1: mean < 0, i.e. losses beyond VaR larger than predicted), bootstrap p-value (B = 10000).
  2. Du and Escanciano (2017) unconditional test U_ES: cumulative violations H_t = (a - u_t)/a * 1{u_t <= a} have mean a/2.
  3. Du and Escanciano (2017) conditional test C_ES(5): Box-Pierce on H_t (no autocorrelation of the cumulative violations).
Loss: FZ0 of Patton, Ziegel and Chen (2019), member of the Fissler-Ziegel (2016) class of losses jointly
strictly consistent for (VaR, ES): see fz0_loss below. Diebold-Mariano of each model vs HS and MCS (90%) on FZ0.
Outputs: tables/es/T4.1_es_tests.csv, T4.2_es_mcs.csv, figures/F4.1_es_<series>_<alpha>.png
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from settings_hw2 import *
from common import load_returns, backtest_dates, load_var, ensure_dirs
import tests_lib as tl

warnings.filterwarnings("ignore")
TAB = os.path.join(TABLE_DIR, "es")
ES_MODELS = {"HS": "M01_HS", "AG_N": "M09_AG_N", "AG_T": "M10_AG_T", "AG_HS": "M11_AG_HS", "AG_GPD": "M15_AG_GPD"}   # ES approach -> paired VaR model
B_MF = 10000                                                 # bootstrap replications of the McNeil-Frey test


def fz0_loss(r, var, es, a):
    """FZ0 loss of Patton, Ziegel and Chen (2019, J. of Econometrics 211), from the Fissler-Ziegel (2016) class:
    L = -1/(a ES) 1{r <= VaR}(VaR - r) + VaR/ES + log(-ES) - 1, with VaR, ES < 0.
    Jointly strictly consistent for (VaR, ES): the only kind of loss that can rank ES forecasts."""
    return -1 / (a * es) * (r <= var) * (var - r) + var / es + np.log(-es) - 1


def es_inputs(name, r):
    """VaR, ES, PIT and scale sigma of every approach, day by day."""
    x = r[name].dropna()
    dates = backtest_dates(x.to_frame())
    pos = x.index.get_indexer(dates)
    win = sliding_window_view(x.values, W_ROLL)[pos - W_ROLL]   # rolling windows (for HS)
    v = load_var(name, r)                                    # VaR forecasts of Point 1
    g = pd.read_csv(os.path.join(OUT_DIR, "es_inputs", "garch", f"{name}.csv"), index_col=0, parse_dates=True)   # ES/PIT of GARCH models
    out = pd.DataFrame({"ret": x.loc[dates].values}, index=dates)
    rt = out.ret.values
    out["PIT_HS"] = (win <= rt[:, None]).mean(axis=1)        # HS PIT: empirical cdf of the window at r_t
    out["SIG_HS"] = win.std(axis=1, ddof=1)                  # HS scale (McNeil-Frey)
    for a in ALPHAS:
        q = np.quantile(win, a, axis=1)                      # HS VaR
        out[f"VaR_HS_{a}"] = v[f"M01_HS_{a}"].values
        out[f"ES_HS_{a}"] = np.nanmean(np.where(win <= q[:, None], win, np.nan), axis=1)   # HS ES: mean of the tail of the window
        for m in ("AG_N", "AG_T", "AG_HS", "AG_GPD"):
            out[f"VaR_{m}_{a}"] = v[f"{ES_MODELS[m]}_{a}"].values
            out[f"ES_{m}_{a}"] = g[f"ES_{m}_{a}"].values
    for m in ("AG_N", "AG_T", "AG_HS", "AG_GPD"):
        out[f"PIT_{m}"] = g[f"PIT_{m}"].values; out[f"SIG_{m}"] = g[f"SIG_{m}"].values
    return out


def mcneil_frey(r, var, es, sig, rng):
    """McNeil-Frey exceedance-residual test: H0 mean(e) = 0 vs H1 mean(e) < 0, bootstrap of the centred residuals."""
    hit = r < var
    e = ((r - es) / sig)[hit]                                # standardised exceedance residuals
    if len(e) < 3:                                           # too few violations
        return len(e), np.nan, np.nan, np.nan
    t = e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))         # observed t-statistic
    ec = e - e.mean()                                        # impose H0
    bs = ec[rng.integers(0, len(ec), (B_MF, len(ec)))]       # bootstrap samples
    tb = bs.mean(axis=1) / (bs.std(axis=1, ddof=1) / np.sqrt(len(ec)))
    return len(e), e.mean(), t, (tb <= t).mean()             # one-sided p-value


def du_escanciano(u, a, m=5):
    """Du-Escanciano (2017) unconditional (U_ES) and conditional (C_ES) backtests on the cumulative violations."""
    H = (a - u) / a * (u <= a)                               # cumulative violation process
    n = len(H)
    U = np.sqrt(n) * (H.mean() - a / 2) / np.sqrt(a * (1 / 3 - a / 4))   # ~ N(0,1) under H0
    Hc = H - a / 2
    g0 = Hc @ Hc / n
    rho = np.array([(Hc[j:] @ Hc[:-j]) / (n - j) / g0 for j in range(1, m + 1)])   # autocorrelations lags 1..m
    C = n * np.sum(rho**2)                                   # Box-Pierce ~ chi2(m) under H0
    return U, 2 * stats.norm.sf(abs(U)), C, stats.chi2.sf(C, m)


def evaluate(name, D, rng):
    """TABLES T4.1 (tests, FZ0, DM) and T4.2 (MCS), full sample and each calendar year."""
    rows, mcs_rows = [], []
    for period, g in [("all", D)] + [(str(y), gg) for y, gg in D.groupby(D.index.year)]:
        r = g.ret.values
        for a in ALPHAS:
            FZ = {}                                          # FZ0 losses of every approach
            for m in ES_MODELS:
                var, es, u, sig = (g[f"VaR_{m}_{a}"].values, g[f"ES_{m}_{a}"].values, g[f"PIT_{m}"].values, g[f"SIG_{m}"].values)
                n_h, e_mean, t_mf, p_mf = mcneil_frey(r, var, es, sig, rng)   # test 1
                U, pU, C, pC = du_escanciano(u, a)                           # tests 2 and 3
                FZ[m] = fz0_loss(r, var, es, a)                           # loss
                hit = r < var
                rows.append({"series": name, "period": period, "alpha": a, "model": m, "n": len(r), "hits": int(hit.sum()),
                             "avg_ES": es.mean(), "avg_loss_beyond_VaR": r[hit].mean() if hit.any() else np.nan,
                             "avg_ES_on_hit_days": es[hit].mean() if hit.any() else np.nan,
                             "MF_mean_resid": e_mean, "MF_t": t_mf, "MF_p": p_mf, "UES": U, "UES_p": pU,
                             "CES5": C, "CES5_p": pC, "FZ0": FZ[m].mean()})
            for m in ES_MODELS:                              # Diebold-Mariano on FZ0 against HS
                if m != "HS":
                    s, p = tl.diebold_mariano(FZ[m], FZ["HS"])
                    rows[-len(ES_MODELS) + list(ES_MODELS).index(m)].update({"DM_FZ0_vs_HS": s, "DM_FZ0_p": p})
            p = tl.mcs(np.column_stack([FZ[m] for m in ES_MODELS]), list(ES_MODELS), 0.10, 1000, 10, SEED)   # 90% MCS on FZ0
            for m in ES_MODELS:
                mcs_rows.append({"series": name, "period": period, "alpha": a, "model": m, "FZ0": FZ[m].mean(),
                                 "MCS_p": p[m], "in_MCS_90": p[m] >= 0.10})
    return rows, mcs_rows


def plot(name, D):
    """FIGURE F4.1: returns, VaR, ES and violations, one panel per approach."""
    for a in ALPHAS:
        fig, axes = plt.subplots(len(ES_MODELS), 1, figsize=(13, 2.4 * len(ES_MODELS)), sharex=True)
        for ax, m in zip(axes, ES_MODELS):
            ax.bar(D.index, D.ret, width=1.5, color="0.7")
            ax.plot(D.index, D[f"VaR_{m}_{a}"], lw=0.8, label="VaR")
            ax.plot(D.index, D[f"ES_{m}_{a}"], lw=0.8, label="ES")
            hit = D.ret < D[f"VaR_{m}_{a}"]
            ax.scatter(D.index[hit], D.ret[hit], s=8, color="red", zorder=3, label="violation")
            ax.set_title(m, fontsize=9); ax.legend(fontsize=7, loc="lower left", ncol=3)
        fig.suptitle(f"Fig F4.1 - {name}: VaR and ES {int(a * 100)}%")
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, f"F4.1_es_{name}_{a}.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    ensure_dirs(); os.makedirs(TAB, exist_ok=True)
    r = load_returns()
    series = sys.argv[1:] or [s for s in (EWP, INDEX_NAME) if s in r.columns]   # EWP and market index only
    rng = np.random.default_rng(SEED)                        # one generator for all McNeil-Frey bootstraps
    rows, mrows = [], []
    for s in series:
        D = es_inputs(s, r)
        D.to_csv(os.path.join(TAB, f"es_forecasts_{s}.csv"))   # ES forecasts day by day
        a1, a2 = evaluate(s, D, rng); rows += a1; mrows += a2
        plot(s, D)                                           # FIGURE F4.1
        print(f"  {s}: done", flush=True)
    t = pd.DataFrame(rows); t.round(5).to_csv(os.path.join(TAB, "T4.1_es_tests.csv"), index=False)   # TABLE T4.1
    m = pd.DataFrame(mrows); m.round(5).to_csv(os.path.join(TAB, "T4.2_es_mcs.csv"), index=False)    # TABLE T4.2
    pd.set_option("display.width", 250)
    print(t[t.period == "all"].round(3).to_string())
    print(m[m.period == "all"].round(3).to_string())
