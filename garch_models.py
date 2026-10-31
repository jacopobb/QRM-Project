"""Points 3-4: GARCH estimation, portfolio, one-step-ahead (static) forecasts, VaR Case 1 and Case 2.

Run:  python garch_models.py
Outputs (all reproducible, SEED fixed in settings.py):
  outputs/tables/G*.csv        summary tables used in the report
  outputs/figures/G*.png       figures used in the report
  outputs/forecasts/<name>.csv one-step-ahead mu, sigma, VaR Case 1 / Case 2 for every series (used by evt.py / backtest.py)
  outputs/Z_insample.csv       in-sample standardised residuals Z of the Gaussian GARCH (input for EVT, points 5-6)
Returns are daily log returns in % (data_prep.log_returns), so VaR is in % too (negative number = loss).
"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
try:
    from settings import *
except ModuleNotFoundError:
    from config import *
import data_prep

warnings.filterwarnings("ignore")                       # arch convergence / scaling notices
TABLE_DIR, FIG_DIR, FC_DIR = "outputs/tables", "outputs/figures", "outputs/forecasts"
for d in (TABLE_DIR, FIG_DIR, FC_DIR): os.makedirs(d, exist_ok=True)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
CANDIDATES = ["t", "skewt", "ged"]                      # alternatives to the Normal (Point 4)

def save_table(df, name, digits=4):
    print(f"\n=== {name} ===\n", df.round(digits).to_string())
    df.round(6).to_csv(f"{TABLE_DIR}/{name}.csv")

# ------------------------------------------------------------------ data
def load_returns():
    """Clean prices -> log returns (%), columns renamed to company names, plus the portfolio PORT."""
    f = CLEAN_FILE if os.path.exists(CLEAN_FILE) else os.path.basename(CLEAN_FILE)
    px = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()[TICKERS]
    r = data_prep.log_returns(px).rename(columns={v[0]: n for n, v in STOCKS.items()})
    # POINT 3: random positive weights (exponential density, SEED) normalised to sum to 1 -> same numbers at every run
    w = data_prep.make_weights(list(r.columns), SEED)
    r["PORT"] = (r[w.index] * w).sum(axis=1)             # daily weighted average of the returns
    return r, w

# ------------------------------------------------------------------ model
def spec(r, dist="normal", o=0):
    """Constant-mean GARCH(1,1). Chosen because it is the standard parsimonious model for daily returns;
    o=1 would give GJR-GARCH (leverage effect)."""
    return arch_model(r, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist=dist)

def fit_in_sample(r, dist="normal", o=0):
    return spec(r.loc[:IN_SAMPLE_END], dist, o).fit(disp="off")      # in-sample = up to IN_SAMPLE_END only

def std_resid(res):
    return res.std_resid.dropna()                                    # Z = (r - mu) / sigma_t

def param_table(results):
    """POINT 3: estimates, (robust) s.e., t-stats, p-values of the Gaussian GARCH for every series."""
    rows = []
    for name, res in results.items():
        for p in res.params.index:
            rows.append({"series": name, "param": p, "coef": res.params[p], "std_err": res.std_err[p],
                         "t": res.tvalues[p], "p_value": res.pvalues[p], "signif_5%": res.pvalues[p] < .05})
    return pd.DataFrame(rows).set_index(["series", "param"])

def persistence_table(results):
    rows = {}
    for name, res in results.items():
        a, b, om = res.params["alpha[1]"], res.params["beta[1]"], res.params["omega"]
        pers = a + b
        rows[name] = {"alpha+beta": pers, "uncond_vol_daily_%": np.sqrt(om / (1 - pers)) if pers < 1 else np.nan,
                      "half_life_days": np.log(.5) / np.log(pers) if pers < 1 else np.inf,
                      "loglik": res.loglikelihood, "AIC": res.aic, "BIC": res.bic}
    return pd.DataFrame(rows).T

# ------------------------------------------------------------------ POINT 4: Z diagnostics and choice of alternative density
def z_diagnostics(Zs, lb=10):
    rows = {}
    for c, z in Zs.items():
        rows[c] = {"mean": z.mean(), "std": z.std(), "skew": stats.skew(z), "ex_kurt": stats.kurtosis(z),
                   "JB_p": stats.jarque_bera(z).pvalue, "KS_norm_p": stats.kstest(z, "norm").pvalue,
                   f"LB{lb}_Z_p": acorr_ljungbox(z, lags=[lb])["lb_pvalue"].iloc[0],
                   f"LB{lb}_Z2_p": acorr_ljungbox(z ** 2, lags=[lb])["lb_pvalue"].iloc[0],
                   f"ARCH_LM{lb}_p": het_arch(z, nlags=lb)[1]}
    return pd.DataFrame(rows).T

def compare_distributions(r):
    """Refit the GARCH(1,1) with Normal / Student-t / skew-t / GED innovations; choose the lowest-BIC density.
    Returns the table of information criteria and the chosen distribution per series."""
    rows, best = [], {}
    for c in r:
        fits = {d: fit_in_sample(r[c], d) for d in ["normal"] + CANDIDATES}
        for d, f in fits.items():
            k = f.model.distribution.num_params
            shape = ", ".join(f"{p}={f.params[p]:.2f}" for p in f.params.index[-k:]) if k else ""
            rows.append({"series": c, "dist": d, "loglik": f.loglikelihood, "AIC": f.aic, "BIC": f.bic, "shape": shape})
        best[c] = min(CANDIDATES, key=lambda d: fits[d].bic)
    tab = pd.DataFrame(rows).set_index(["series", "dist"])
    # LR test Normal vs t (nested, H0: 1/nu = 0 is on the boundary, so p-values are only indicative)
    return tab, pd.Series(best, name="chosen_dist")

# ------------------------------------------------------------------ POINT 4: static one-step-ahead forecasts
def oos_forecasts(r, dist="normal", o=0, h=H_REFIT, alphas=ALPHAS):
    """One-step-ahead mean and sigma for every out-of-sample day (t known only up to t-1) plus the
    standardised quantile q_alpha of the innovation density, re-estimated every h days (expanding window).
    h=None -> parameters fixed at the in-sample estimates."""
    oos = r.index[r.index > IN_SAMPLE_END]
    step = h if h else len(oos)
    out = []
    for s in range(0, len(oos), step):
        block = oos[s:s + step]
        res = spec(r[r.index < block[0]], dist, o).fit(disp="off")             # information up to day before the block
        filt = spec(r.loc[:block[-1]], dist, o).fix(res.params.values)         # filter with fixed parameters on later data
        sig = filt.conditional_volatility.loc[block]                           # sigma_t uses returns up to t-1 only
        k = res.model.distribution.num_params
        if dist == "normal": q = norm.ppf(alphas)
        else: q = res.model.distribution.ppf(np.array(alphas), res.params.iloc[-k:].values)   # unit-variance quantiles
        df = pd.DataFrame({"mu": res.params["mu"], "sigma": sig})
        for a, qa in zip(alphas, q): df[f"q_{a}"] = qa
        out.append(df)
    return pd.concat(out)

def var_from_forecasts(fc, alphas=ALPHAS):
    """VaR = mu + sigma * q_alpha (return quantile, negative number)."""
    return pd.DataFrame({a: fc["mu"] + fc["sigma"] * fc[f"q_{a}"] for a in alphas})

def var_gaussian(mu, sig, alpha):
    return mu + sig * norm.ppf(alpha)                                            # VaR Case 1

# ------------------------------------------------------------------ main
def main():
    r, w = load_returns()
    print(f"Sample {r.index.min().date()} -> {r.index.max().date()} | in-sample until {IN_SAMPLE_END} "
          f"({(r.index <= IN_SAMPLE_END).sum()} days), out-of-sample {(r.index > IN_SAMPLE_END).sum()} days")
    save_table(w.to_frame(), "G0_portfolio_weights", 4); print("sum of weights =", w.sum())

    # ---------- POINT 3: Gaussian GARCH(1,1) on each series and on PORT (in-sample)
    res = {c: fit_in_sample(r[c]) for c in r}
    pt = param_table(res); save_table(pt, "G1_gaussian_garch_parameters")
    share = pt.groupby("param")["signif_5%"].mean().to_frame("share_significant_5%"); save_table(share, "G1b_share_significant")
    save_table(persistence_table(res), "G2_persistence_and_fit")

    # Fig G1: conditional variance of four series incl. PORT, and of all series (grid)
    show = ["Roche", "UBS", "Nestle", "PORT"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), show):
        ax.plot(res[c].conditional_volatility ** 2, lw=.7); ax.set_title(f"{c} - conditional variance (%$^2$)", fontsize=10)
    fig.suptitle("Fig G1 - GARCH(1,1) Gaussian conditional variance, in-sample"); fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/G1_cond_variance.png", dpi=150)
    n = r.shape[1]; nc = 4; nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 3.8, nr * 2.4)); axes = axes.ravel()
    for ax, c in zip(axes, r): ax.plot(res[c].conditional_volatility ** 2, lw=.5); ax.set_title(c, fontsize=9)
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle("Fig G1b - Conditional variance, all series"); fig.tight_layout(); fig.savefig(f"{FIG_DIR}/G1b_cond_variance_all.png", dpi=150)

    # ---------- POINT 4a: in-sample standardised residuals Z and normality check
    Z = pd.DataFrame({c: std_resid(res[c]) for c in r}); Z.to_csv("outputs/Z_insample.csv")   # input for EVT (points 5-6)
    save_table(z_diagnostics(Z), "G3_Z_diagnostics")
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 3.6, nr * 2.8)); axes = axes.ravel()
    for ax, c in zip(axes, Z): stats.probplot(Z[c], dist="norm", plot=ax); ax.set_title(c, fontsize=9); ax.set_xlabel(""); ax.set_ylabel("")
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle("Fig G2 - Normal QQ-plots of in-sample standardised residuals Z"); fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/G2_Z_qqplots.png", dpi=150)

    # ---------- POINT 4b: choice of the alternative density (lowest BIC among t, skew-t, GED)
    ic, best = compare_distributions(r)
    save_table(ic, "G4_distribution_comparison"); save_table(best.to_frame(), "G4b_chosen_distribution")
    print("\nChosen density counts:\n", best.value_counts().to_string())

    # ---------- POINT 4c: one-step-ahead forecasts, VaR Case 1 (Normal) and Case 2 (chosen density)
    summary = []
    for c in r:
        fc1 = oos_forecasts(r[c], "normal"); fc2 = oos_forecasts(r[c], best[c])
        v1, v2 = var_from_forecasts(fc1), var_from_forecasts(fc2)
        f = pd.DataFrame({"ret": r[c].loc[fc1.index], "mu": fc1["mu"], "sigma_normal": fc1["sigma"], "sigma_case2": fc2["sigma"]})
        for a in ALPHAS:
            f[f"VaR_case1_{a}"] = v1[a]; f[f"VaR_case2_{a}"] = v2[a]
        f.to_csv(f"{FC_DIR}/{c}.csv")
        for a in ALPHAS:                                  # quick sanity check; full backtests are Point 8
            summary.append({"series": c, "alpha": a, "n_oos": len(f), "expected_hits": a * len(f),
                            "hits_case1": int((f["ret"] < f[f"VaR_case1_{a}"]).sum()),
                            "hits_case2": int((f["ret"] < f[f"VaR_case2_{a}"]).sum()),
                            "avg_VaR_case1": f[f"VaR_case1_{a}"].mean(), "avg_VaR_case2": f[f"VaR_case2_{a}"].mean(),
                            "case2_dist": best[c]})
    s = pd.DataFrame(summary).set_index(["series", "alpha"]); save_table(s, "G5_var_case1_vs_case2_summary")

    # Fig G3: returns and VaR Case 1 vs Case 2 (5% and 1%) for one asset and PORT
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for ax, c in zip(axes, ["UBS", "PORT"]):
        f = pd.read_csv(f"{FC_DIR}/{c}.csv", index_col=0, parse_dates=True)
        ax.plot(f["ret"], color="grey", lw=.6, label="return")
        ax.plot(f["VaR_case1_0.05"], "b", lw=1, label="VaR 5% Case 1"); ax.plot(f["VaR_case2_0.05"], "b--", lw=1, label=f"VaR 5% Case 2 ({best[c]})")
        ax.plot(f["VaR_case1_0.01"], "r", lw=1, label="VaR 1% Case 1"); ax.plot(f["VaR_case2_0.01"], "r--", lw=1, label=f"VaR 1% Case 2 ({best[c]})")
        ax.set_title(c); ax.legend(fontsize=7, ncol=5)
    fig.suptitle("Fig G3 - Out-of-sample returns and VaR, Case 1 vs Case 2"); fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/G3_var_case1_case2.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
