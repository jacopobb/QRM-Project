"""Points 3-4 - GARCH estimation, static one-step-ahead forecasts, Gaussian VaR."""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2, norm
try:
    from settings import *
except ModuleNotFoundError:
    from config import *
import descriptive
import data_prep   

def spec(r, dist="normal", o=0, p=1, q=1, vol="GARCH"):
    """Builds the model on the data it receives (in-sample or up to day t for forecasts). vol: "GARCH" (GJR if o>0) or "EGARCH"."""
    return arch_model(r, mean="Constant", vol=vol, p=p, o=o, q=q, dist=dist)

def fit_in_sample(r, dist="normal", o=0, p=1, q=1, vol="GARCH"):
    return spec(r.loc[:IN_SAMPLE_END], dist, o, p, q, vol).fit(disp="off")    # in-sample cut here

def find_spec(r, max_p=1, max_q=1, max_o=1, dist="normal"):
    """In-sample grid by BIC. Returns the chosen orders (p, o, q); GJR kept only if gamma is significant."""
    rows = {}
    for p in range(1, max_p + 1):
        for o in range(max_o + 1):
            for q in range(1, max_q + 1):
                res = fit_in_sample(r, dist, o, p, q)
                rows[f"GARCH({p},{o},{q})"] = {"BIC": res.bic, "gamma_p": res.pvalues.get("gamma[1]", np.nan),
                                               "p": p, "o": o, "q": q}
    best = pd.DataFrame.from_dict(rows, orient="index").sort_values("BIC").iloc[0]
    o = int(best["o"]) if best["gamma_p"] < 0.05 else 0                 # NaN < 0.05 is False -> o = 0
    return int(best["p"]), o, int(best["q"])

def persistence(params, vol="GARCH"):
    """GARCH/GJR: sum alpha + sum beta + 0.5 * sum gamma.  EGARCH (log-variance): sum beta."""
    beta = params.filter(regex=r"^beta\[\d+\]$").sum()
    if vol == "EGARCH":
        return beta
    return beta + params.filter(regex=r"^alpha\[\d+\]$").sum() + 0.5 * params.filter(regex=r"^gamma\[\d+\]$").sum()

def param_tables(table, series, res):
    for name in res.params.index:                                    
        table.loc[series, name] = res.params[name]
        table.loc[series, f"{name}_p"] = res.pvalues[name]
    table.loc[series, "persistence"] = persistence(res.params, res.model.volatility.name)
    table.loc[series, "BIC"] = res.bic
    table.loc[series, "AIC"] = res.aic
    table.loc[series, "MLL"] = res.loglikelihood
    return table

def std_resid(res):
    return res.std_resid.dropna() # Z, in-sample standardised residuals

def sign_bias_test(z):
    z = z.dropna()
    z2, z_lag = (z ** 2).iloc[1:], z.shift(1).iloc[1:]
    neg = (z_lag < 0).astype(float)                                  # S- : 1 if yesterday's shock was negative
    X = sm.add_constant(np.column_stack([neg, neg * z_lag, (1 - neg) * z_lag]))
    ols = sm.OLS(z2.values, X).fit()
    return {"sign_bias_p": ols.pvalues[1], "neg_size_bias_p": ols.pvalues[2], "pos_size_bias_p": ols.pvalues[3],
            "joint_p": chi2.sf(len(z2) * ols.rsquared, 3)}

def table_serial_z(Z, lags=10):
    """Tests on the GARCH standardised residuals Z: linear dependence (Z), ARCH effects (Z^2, ARCH-LM)
    and asymmetry left (Engle-Ng sign bias tests)"""
    rows = {}
    for c in Z:
        z = Z[c].dropna()
        band = 1.96 / np.sqrt(len(z))                                # 95% band under iid
        row = {}
        for name, x in [("Z", z), ("Z2", z ** 2)]:
            a = acf(x, nlags=lags, fft=True)[1:]                    
            out = {k + 1: a[k] for k in range(lags) if abs(a[k]) > band}   
            row[f"ACF1_{name}"] = a[0]
            row[f"n_sig_{name}(of {lags})"] = len(out)
            row[f"sig_lags_{name}"] = ", ".join(f"{k}:{v:.3f}" for k, v in out.items()) or "-"   
            row[f"LB{lags}_{name}_p"] = acorr_ljungbox(x, lags=[lags])["lb_pvalue"].iloc[0]
        row[f"ARCH_LM{lags}_p"] = het_arch(z, nlags=lags)[1]
        row.update(sign_bias_test(z))                                # sign_bias_p, neg/pos_size_bias_p, joint_p
        rows[c] = row
    return pd.DataFrame.from_dict(rows, orient="index")

def refine_spec(r, vol, p, o, q, diagn, alpha=0.05):
    bic = lambda m: fit_in_sample(r, o=m[2], p=m[1], q=m[3], vol=m[0]).bic      # m = (vol, p, o, q)
    if diagn["LB10_Z2_p"] < alpha or diagn["ARCH_LM10_p"] < alpha:
        vol, p, o, q = min([(vol, pp, o, qq) for pp in (p, p + 1) for qq in (q, q + 1)], key=bic)
    if diagn["joint_p"] < alpha:
        if o == 0:
           vol, p, o, q = min([("EGARCH",p,1,q),("GARCH",p,0,q)],key=bic)
        elif vol == "GARCH":
            vol, p, o, q = min([("GARCH", p, o, q),("GARCH", p, o + 1, q), ("EGARCH", p, o, q)], key=bic)
    return vol, p, o, q

CANDIDATES = ["t", "skewt", "ged"]                                   # alternatives to the Normal

def z_normality(Z):
    """POINT 4b: is Z Normal? Moments, Jarque-Bera and Kolmogorov-Smirnov (vs N(0,1)) for every series."""
    rows = {}
    for c in Z:
        z = Z[c].dropna()
        rows[c] = {"mean": z.mean(), "std": z.std(), "skew": stats.skew(z), "ex_kurt": stats.kurtosis(z),
                   "JB_p": stats.jarque_bera(z).pvalue, "KS_norm_p": stats.kstest(z, "norm").pvalue}
    return pd.DataFrame(rows).T

def compare_distributions(r, orders):
    """POINT 4c: refit each series' chosen model (vol, p, o, q) with Normal / t / skew-t / GED innovations;
    the alternative density is the one with the lowest BIC among the candidates. Returns the IC table and the choice."""
    rows, best = [], {}
    for c in r:
        vol, p, o, q = orders[c]
        fits_d = {d: fit_in_sample(r[c], d, o, p, q, vol) for d in ["normal"] + CANDIDATES}
        for d, f in fits_d.items():
            k = f.model.distribution.num_params
            shape = ", ".join(f"{n}={f.params[n]:.2f}" for n in f.params.index[-k:]) if k else ""
            rows.append({"series": c, "dist": d, "loglik": f.loglikelihood, "AIC": f.aic, "BIC": f.bic, "shape": shape})
        best[c] = min(CANDIDATES, key=lambda d: fits_d[d].bic)
    return pd.DataFrame(rows).set_index(["series", "dist"]), pd.Series(best, name="chosen_dist")

def oos_forecasts(r, dist="normal", vol="GARCH", p=1, o=0, q=1, h=H_REFIT, alphas=ALPHAS):
    """One-step-ahead mu and sigma for every out-of-sample day (information up to t-1 only) with the series' chosen
    model, plus the unit-variance quantile q_alpha of the innovation density; parameters re-estimated every h days
    on an expanding window (h=None -> fixed at the in-sample estimates)."""
    oos = r.index[r.index > IN_SAMPLE_END]
    step = h if h else len(oos)
    out = []
    for s in range(0, len(oos), step):
        block = oos[s:s + step]
        res = spec(r[r.index < block[0]], dist, o, p, q, vol).fit(disp="off")          # estimated up to the day before the block
        filt = spec(r.loc[:block[-1]], dist, o, p, q, vol).fix(res.params.values)      # filter with fixed parameters
        sig = filt.conditional_volatility.loc[block]                                   # sigma_t uses returns up to t-1 only
        k = res.model.distribution.num_params
        q_a = norm.ppf(alphas) if dist == "normal" else res.model.distribution.ppf(np.array(alphas), res.params.iloc[-k:].values)
        df = pd.DataFrame({"mu": res.params["mu"], "sigma": sig})
        for a, qa in zip(alphas, q_a): df[f"q_{a}"] = qa
        out.append(df)
    return pd.concat(out)

def var_from_forecasts(fc, alphas=ALPHAS):
    """VaR = mu + sigma * q_alpha (return quantile, negative number)."""
    return pd.DataFrame({a: fc["mu"] + fc["sigma"] * fc[f"q_{a}"] for a in alphas})

if __name__ == "__main__":
    #load data
    px, r = descriptive.load_data()
    w = data_prep.make_weights(TICKERS)
    w.index = list(STOCKS.keys())
    print(w.round(4))
    w.to_csv("outputs/tables/G0_portfolio_weights.csv")
    r["PORT"] = data_prep.port_returns(r, w)

    #initialization point 3
    models=pd.DataFrame(index=r.columns, dtype=float)                   
    models.index.name="ticker"
    Z = pd.DataFrame(index=r.loc[:IN_SAMPLE_END].index, columns=r.columns, dtype=float)
    Z.index.name="date"

    #Garch estimation
    orders, fits = {}, {}
    for column in r.columns:
        p, o, q = find_spec(r[column])
        orders[column] = ("GARCH", p, o, q)                          
        res = fit_in_sample(r[column], o=o, p=p, q=q)
        fits[column] = res                                           
        param_tables(models, column, res)
        Z[column] = std_resid(res)
    Z_diagnostic=table_serial_z(Z, lags=10)

    #Save output before refinement 
    models.to_csv("outputs/tables/G1_garch_parameters_pre_refinement.csv")
    Z_diagnostic.assign(model=[fits[c].model.volatility.name for c in Z_diagnostic.index]) \
                .to_csv("outputs/tables/G2_Z_diagnostics_pre_refinement.csv")
    pd.DataFrame(orders, index=["vol", "p", "o", "q"]).T.to_csv("outputs/tables/G0b_chosen_orders_pre_refinement.csv")

    # Refinement
    for column in Z_diagnostic.index:
        new = refine_spec(r[column], *orders[column], Z_diagnostic.loc[column])
        if new != orders[column]:
            print(f"{column}: {orders[column]} -> {new}")
            orders[column] = new
            res = fit_in_sample(r[column], o=new[2], p=new[1], q=new[3], vol=new[0])
            fits[column] = res
            models.loc[column] = np.nan                             
            param_tables(models, column, res)
            Z[column] = std_resid(res)
    Z_diagnostic = table_serial_z(Z, lags=10)

    #Save output after refinement
    models.to_csv("outputs/tables/G1_garch_parameters.csv")# TABLE G1: GARCH parameters and p-values (point 3, significance)
    Z_diagnostic["model"] = [fits[c].model.volatility.name for c in Z_diagnostic.index]   # GARCH / GJR-GARCH / EGARCH
    Z_diagnostic.to_csv("outputs/tables/G2_Z_diagnostics.csv") # TABLE G2: dependence and asymmetry left in Z (justifies the chosen model)
    Z.to_csv("outputs/Z_insample.csv")
    pd.DataFrame(orders, index=["vol", "p", "o", "q"]).T.to_csv("outputs/tables/G0b_chosen_orders.csv")

    # Fig G1: conditional variance of four series incl. PORT, and of all series (grid)
    FIG_DIR = "outputs/figures"
    show = ["Roche", "UBS", "Nestle", "PORT"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), show):
        ax.plot(fits[c].conditional_volatility ** 2, lw=.7); ax.set_title(f"{c} - conditional variance (%$^2$)", fontsize=10)
    fig.suptitle("Fig G1 - GARCH Gaussian conditional variance, in-sample"); fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/G1_cond_variance.png", dpi=150)
    n = r.shape[1]; nc = 4; nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 3.8, nr * 2.4)); axes = axes.ravel()
    for ax, c in zip(axes, r): ax.plot(fits[c].conditional_volatility ** 2, lw=.5); ax.set_title(c, fontsize=9)
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle("Fig G1b - Conditional variance, all series"); fig.tight_layout(); fig.savefig(f"{FIG_DIR}/G1b_cond_variance_all.png", dpi=150)

    # One-step-ahead (static) forecasts of the last year and VaR Case 1 (Normal)
    fc1 = {}
    for c in r:
        vol, p, o, q = orders[c]
        fc1[c] = oos_forecasts(r[c], "normal", vol, p, o, q)        # mu, sigma, Normal quantiles for every OOS day

    # Normality of the in-sample standardised residuals Z (TABLE G3, Fig G2)
    z_normality(Z).to_csv("outputs/tables/G3_Z_normality.csv")
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 3.6, nr * 2.8)); axes = axes.ravel()
    for ax, c in zip(axes, Z): stats.probplot(Z[c].dropna(), dist="norm", plot=ax); ax.set_title(c, fontsize=9); ax.set_xlabel(""); ax.set_ylabel("")
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle("Fig G2 - Normal QQ-plots of in-sample standardised residuals Z"); fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/G2_Z_qqplots.png", dpi=150)

    # ---------- POINT 4c: choice of the alternative density (lowest BIC among t, skew-t, GED) (TABLES G4, G4b)
    ic, best = compare_distributions(r, orders)
    ic.to_csv("outputs/tables/G4_distribution_comparison.csv"); best.to_frame().to_csv("outputs/tables/G4b_chosen_distribution.csv")
    print("\nChosen density counts:\n", best.value_counts().to_string())

    # ---------- POINT 4d: same forecasts with the chosen density in the GARCH -> VaR Case 2; comparison with Case 1 (TABLE G5)
    # one CSV per series in outputs/forecasts: read by evt.py (mu, sigma_normal), backtest.py and es.py (ret, VaR_*)
    FC_DIR = "outputs/forecasts"; os.makedirs(FC_DIR, exist_ok=True)
    summary = []
    for c in r:
        vol, p, o, q = orders[c]
        fc2 = oos_forecasts(r[c], best[c], vol, p, o, q)
        v1, v2 = var_from_forecasts(fc1[c]), var_from_forecasts(fc2)
        f = pd.DataFrame({"ret": r[c].loc[fc1[c].index], "mu": fc1[c]["mu"], "sigma_normal": fc1[c]["sigma"], "sigma_case2": fc2["sigma"]})
        for a in ALPHAS:
            f[f"VaR_case1_{a}"] = v1[a]; f[f"VaR_case2_{a}"] = v2[a]
        f.to_csv(f"{FC_DIR}/{c}.csv")
        for a in ALPHAS:                                             # quick check; full backtests are Point 8
            summary.append({"series": c, "alpha": a, "n_oos": len(f), "expected_hits": a * len(f),
                            "hits_case1": int((f["ret"] < f[f"VaR_case1_{a}"]).sum()),
                            "hits_case2": int((f["ret"] < f[f"VaR_case2_{a}"]).sum()),
                            "avg_VaR_case1": f[f"VaR_case1_{a}"].mean(), "avg_VaR_case2": f[f"VaR_case2_{a}"].mean(),
                            "case2_dist": best[c]})
    pd.DataFrame(summary).set_index(["series", "alpha"]).to_csv("outputs/tables/G5_var_case1_vs_case2_summary.csv")

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
