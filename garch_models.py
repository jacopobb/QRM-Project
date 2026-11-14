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
    """Constant-mean volatility model on the data it receives (GJR = "GARCH" with o>0)."""
    return arch_model(r, mean="Constant", vol=vol, p=p, o=o, q=q, dist=dist)

def fit_in_sample(r, dist="normal", o=0, p=1, q=1, vol="GARCH"):
    """Fits the model on the in-sample period only (up to IN_SAMPLE_END)."""
    return spec(r.loc[:IN_SAMPLE_END], dist, o, p, q, vol).fit(disp="off")    # in-sample cut here

def param_tables(table, series, res):
    """Writes every estimated parameter with its p-value, BIC, AIC and log-likelihood """
    for name in res.params.index:                                    
        table.loc[series, name] = res.params[name]
        table.loc[series, f"{name}_p"] = res.pvalues[name]
    table.loc[series, "BIC"] = res.bic
    table.loc[series, "AIC"] = res.aic
    table.loc[series, "MLL"] = res.loglikelihood
    return table

def std_resid(res):
    """In-sample standardised residuals Z of a fitted model."""
    return res.std_resid.dropna() # Z, in-sample standardised residuals

def sign_bias_test(z):
    """Engle-Ng (1993) sign and size bias tests: z^2_t on S-, S-*z and S+*z at t-1, joint test n*R^2 ~ chi2(3)."""
    z = z.dropna()
    z2, z_lag = (z ** 2).iloc[1:], z.shift(1).iloc[1:]
    neg = (z_lag < 0).astype(float)                                  # S- : 1 if yesterday's shock was negative
    X = sm.add_constant(np.column_stack([neg, neg * z_lag, (1 - neg) * z_lag]))
    ols = sm.OLS(z2.values, X).fit()
    return {"sign_bias_p": ols.pvalues[1], "neg_size_bias_p": ols.pvalues[2], "pos_size_bias_p": ols.pvalues[3],
            "joint_p": chi2.sf(len(z2) * ols.rsquared, 3)}

def table_serial_z(Z, lags=10):
    """Tests on Z: linear dependence (Z), ARCH effects left (Z^2, ARCH-LM) and asymmetry left (sign bias)."""
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

def select_model(r, dist="normal", alpha=0.05):
    """GARCH(1,1) -> GJR if joint rejects -> by BIC among sig. gamma: GJR1/GJR2/EGARCH/APARCH if joint still rejects, EGARCH/APARCH if GJR gamma not sig."""
    rows, fits_m = {}, {}
    def fit(vol, p, o, q):                                                      # fits a candidate and records its row
        res = fit_in_sample(r, dist, o=o, p=p, q=q, vol=vol)
        name = f"{res.model.volatility.name}({p},{o},{q})"                      # e.g. GJR-GARCH(1,1,1), GJR-GARCH(1,2,1)
        fits_m[name] = res
        rows[name] = {"vol": vol, "p": p, "o": o, "q": q, "BIC": res.bic,
                      "gamma": res.params.get(f"gamma[{o}]", np.nan),         # last asymmetric term (gamma[2] if o=2)
                      "gamma_p": res.pvalues.get(f"gamma[{o}]", np.nan),
                      "delta": res.params.get("delta", np.nan), "delta_p": res.pvalues.get("delta", np.nan),
                      **sign_bias_test(std_resid(res))}
        return name
    sig = lambda name: rows[name]["gamma_p"] < alpha                            # asymmetric model admissible only if gamma sig.
    def best_of(names, default):                                                # lowest BIC among those with significant gamma
        adm = [n for n in names if sig(n)]
        return min(adm, key=lambda n: rows[n]["BIC"]) if adm else default
    best = fit("GARCH", 1, 0, 1)
    if rows[best]["joint_p"] < alpha:                                           # asymmetry left in the GARCH(1,1) Z
        gjr = fit("GARCH", 1, 1, 1)                                             # GJR (vol="GARCH" with o=1)
        if not sig(gjr):                                                        # GJR gamma not sig. -> EGARCH vs APARCH
            best = best_of([fit("EGARCH", 1, 1, 1), fit("APARCH", 1, 1, 1)], best)
        elif rows[gjr]["joint_p"] < alpha:                                      # GJR sig. but not enough -> GJR1/GJR2/EGARCH/APARCH
            best = best_of([gjr, fit("GARCH", 1, 2, 1), fit("EGARCH", 1, 1, 1), fit("APARCH", 1, 1, 1)], gjr)
        else:
            best = gjr                                                          # GJR sig. and no asymmetry left
    cand = pd.DataFrame.from_dict(rows, orient="index")
    cand["chosen"] = cand.index == best
    b = rows[best]
    return cand, (b["vol"], b["p"], b["o"], b["q"]), fits_m[best]

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
    """POINT 4c: refits each chosen model with Normal / t / skew-t / GED; alternative density = lowest BIC."""
    rows, best = [], {}
    for c in r:
        vol, p, o, q = orders[c]
        fits_d = {d: fit_in_sample(r[c], d, o, p, q, vol) for d in ["normal"] + CANDIDATES}
        for d, f in fits_d.items():
            k = f.model.distribution.num_params
            shape = ", ".join(f"{n}={f.params[n]:.2f} (p={f.pvalues[n]:.3f})" for n in f.params.index[-k:]) if k else ""
            rows.append({"series": c, "dist": d, "loglik": f.loglikelihood, "AIC": f.aic, "BIC": f.bic, "shape": shape})
        best[c] = min(CANDIDATES, key=lambda d: fits_d[d].bic)
    return pd.DataFrame(rows).set_index(["series", "dist"]), pd.Series(best, name="chosen_dist")

def density_asymmetry_check(r, orders_pre, orders, best):
    """Check: BIC and sign bias of original and refined models, with the Normal and with the chosen density."""
    rows = {}
    for c in r:
        for stage, ords in [("original", orders_pre), ("refined", orders)]:
            if stage == "refined" and orders[c] == orders_pre[c]: continue      # model not refined: same as the original
            vol, p, o, q = ords[c]
            row = {"model": f"{vol}({p},{o},{q})", "dist": best[c]}
            for name, d in [("normal", "normal"), ("chosen", best[c])]:
                f = fit_in_sample(r[c], d, o=o, p=p, q=q, vol=vol)
                row[f"BIC_{name}"] = f.bic
                row.update({f"{k}_{name}": v for k, v in sign_bias_test(std_resid(f)).items()})
            rows[(c, stage)] = row
    return pd.DataFrame(list(rows.values()), index=pd.MultiIndex.from_tuples(rows.keys(), names=["series", "stage"]))

def implausible(params, vol):
    """Economically implausible fit: persistence >= 1, a large negative shock lowering volatility (EGARCH), t/skew-t nu at its bound."""
    a, g, b = (params.filter(regex=rf"^{k}\[\d+\]$").sum() for k in ("alpha", "gamma", "beta"))
    if vol == "EGARCH":
        bad = abs(b) >= 1 or a - g <= 0                      
    elif vol == "APARCH":
        bad = b >= 1 
    else:                                        
        bad = a + b + 0.5 * g >= 1                           
    nu = params.get("nu", params.get("eta", np.inf))
    return bool(bad or nu <= 2.1)                             

def oos_forecasts(r, dist="normal", vol="GARCH", p=1, o=0, q=1, h=H_REFIT, window=W_ROLL, alphas=ALPHAS):
    """One-step-ahead mu, sigma and quantiles q_alpha per OOS day; refit every h days on the last `window` days."""
    oos = r.index[r.index > IN_SAMPLE_END]
    step = h if h else len(oos)
    out, prev = [], None
    for s in range(0, len(oos), step):
        block = oos[s:s + step]
        train = r[r.index < block[0]]                                                  
        if window: train = train.iloc[-window:]                                        
        res = spec(train, dist, o, p, q, vol).fit(disp="off")
        if prev is not None and implausible(res.params, vol):                          
            print(f"  {r.name} {vol} {dist} block {block[0].date()}: implausible parameters, previous parameters kept")
            res = prev
        prev = res
        filt = spec(r.loc[train.index[0]:block[-1]], dist, o, p, q, vol).fix(res.params.values)  
        sig = filt.conditional_volatility.loc[block]                                   
        k = res.model.distribution.num_params
        if dist == "normal": q_a = norm.ppf(alphas)
        else: q_a = res.model.distribution.ppf(np.array(alphas), res.params.iloc[-k:].values)   
        df = pd.DataFrame({"mu": res.params["mu"], "sigma": sig})
        for a, qa in zip(alphas, q_a): df[f"q_{a}"] = qa
        out.append(df)
    return pd.concat(out)

def var_from_forecasts(fc, alphas=ALPHAS):
    """VaR = mu + sigma * q_alpha (return quantile, negative number)."""
    return pd.DataFrame({a: fc["mu"] + fc["sigma"] * fc[f"q_{a}"] for a in alphas})

if __name__ == "__main__":
    # remove the tables written by older versions of this script (no longer produced)
    for f in ["G0b_chosen_orders_pre_refinement", "G0c_asymmetry_candidates", "G1_garch_parameters_pre_refinement",
              "G2_Z_diagnostics_pre_refinement", "T6_port_weights", "T7_garch_params_insample", "T8_Z_insample"]:
        if os.path.exists(f"outputs/tables/{f}.csv"): os.remove(f"outputs/tables/{f}.csv")

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

    # Model selection: GARCH, GJR, EGARCH, APARCH (1,1) for every series, chosen by select_model (gamma significant,
    # then no asymmetry left in the sign bias tests, then lowest BIC)
    orders, fits, candidates = {}, {}, {}
    for column in r.columns:
        candidates[column], orders[column], res = select_model(r[column])
        fits[column] = res
        param_tables(models, column, res)
        Z[column] = std_resid(res)
    candidates = pd.concat(candidates, names=["series", "model"])           # TABLE G0c: every candidate, chosen = True
    print(candidates.round(4).to_string())
    candidates.to_csv("outputs/tables/G0c_model_candidates.csv")
    Z_diagnostic = table_serial_z(Z, lags=10)
    orders_pre = {c: ("GARCH", 1, 0, 1) for c in r}                     # GARCH(1,1) benchmark, for density_asymmetry_check

    #Save output
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

    # Choice of the alternative density (lowest BIC among t, skew-t, GED) (TABLES G4, G4b)
    ic, best = compare_distributions(r, orders)
    ic.to_csv("outputs/tables/G4_distribution_comparison.csv"); best.to_frame().to_csv("outputs/tables/G4b_chosen_distribution.csv")
    print("\nChosen density counts:\n", best.value_counts().to_string())

    # QQ-plots of the standardised residuals of the model with the chosen density 
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 3.6, nr * 2.8)); axes = axes.ravel()
    for ax, c in zip(axes, r):
        vol, p, o, q = orders[c]
        f = fit_in_sample(r[c], best[c], o=o, p=p, q=q, vol=vol)            # model refitted with the chosen density
        z = np.sort(std_resid(f).values)
        k = f.model.distribution.num_params
        pp = (np.arange(1, len(z) + 1) - 0.5) / len(z)                         # plotting positions
        qt = f.model.distribution.ppf(pp, f.params.iloc[-k:].values)            # theoretical quantiles (unit variance)
        ax.scatter(qt, z, s=3); ax.plot(qt, qt, "r", lw=.8)
        ax.set_title(f"{c} ({best[c]})", fontsize=9)
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle("Fig G2b - QQ-plots of standardised residuals vs the chosen density"); fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/G2b_Z_qqplots_chosen_density.png", dpi=150)

    # Personal check, not part of the outputs: uncomment to see if the asymmetry left depends on the Gaussian density
    print(density_asymmetry_check(r, orders_pre, orders, best).round(4).to_string())

    #Same forecasts with the chosen density in the GARCH -> VaR Case 2; comparison with Case 1 (TABLE G5)
    #read by evt.py (mu, sigma_normal), backtest.py and es.py (ret, VaR_*)
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
