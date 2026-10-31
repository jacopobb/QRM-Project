"""Points 5-7: Extreme Value Theory on the in-sample standardised residuals Z (Gaussian GARCH) and
VaR Cases 3, 4, 5 for the out-of-sample returns.

Run AFTER garch_models.py:   python evt.py
Needs : outputs/Z_insample.csv and outputs/forecasts/<series>.csv (made by garch_models.py)
Writes: outputs/tables/E*.csv, outputs/figures/E*.png, and adds VaR_case3/4/5 columns to outputs/forecasts/<series>.csv

Conventions: losses L = -Z (left tail of Z). All EVT quantities are estimated ONCE in-sample (no re-estimation OOS).
VaR of Z is reported as a *return quantile* (negative number) = -(loss quantile).
"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import genextreme, genpareto, gumbel_r, norm
from statsmodels.tools.numdiff import approx_hess
try:
    from settings import *
except ModuleNotFoundError:
    from config import *

warnings.filterwarnings("ignore")
TABLE_DIR, FIG_DIR, FC_DIR = "outputs/tables", "outputs/figures", "outputs/forecasts"
for d in (TABLE_DIR, FIG_DIR, FC_DIR): os.makedirs(d, exist_ok=True)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
THRESH_Q = 0.90                       # GPD threshold = 90th percentile of the losses (see sensitivity table E6)
THETA_Q = 0.95                        # threshold used to estimate the extremal index
RUN_LEN = 5                           # run length of the runs estimator of the extremal index
SENS_Q = [0.85, 0.90, 0.925, 0.95]    # thresholds for the GPD sensitivity analysis

def save_table(df, name, digits=4):
    print(f"\n=== {name} ===\n", df.round(digits).to_string())
    df.round(6).to_csv(f"{TABLE_DIR}/{name}.csv")

# ------------------------------------------------------------------ goodness of fit helpers
def ad_stat(u):
    """Anderson-Darling statistic from probability-integral-transformed data u (no p-value: parameters are estimated)."""
    u = np.clip(np.sort(u), 1e-12, 1 - 1e-12); n = len(u); i = np.arange(1, n + 1)
    return -n - np.sum((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1]))) / n

# ------------------------------------------------------------------ POINT 5: GEV on block maxima, independence
def block_maxima(L, n=BLOCK):
    L = L.iloc[len(L) % n:]                                   # drop the oldest incomplete block -> deterministic
    return L.values.reshape(-1, n).max(axis=1)

def fit_gev(m):
    """MLE of the GEV (xi = shape, xi>0 heavy tail). scipy's c = -xi. Several starting values, keep the best likelihood."""
    g = gumbel_r.fit(m); best = None
    for c0 in (0.1, 0.0, -0.1, -0.2):
        try:
            c, loc, sc = genextreme.fit(m, c0, loc=g[0], scale=g[1])
            ll = genextreme.logpdf(m, c, loc, sc).sum()
            if np.isfinite(ll) and (best is None or ll > best[0]): best = (ll, c, loc, sc)
        except Exception: pass
    ll, c, loc, sc = best
    nll = lambda p: -genextreme.logpdf(m, -p[0], p[1], p[2]).sum()
    try:
        se = np.sqrt(np.diag(np.linalg.inv(approx_hess(np.array([-c, loc, sc]), nll))))   # observed information
    except Exception: se = np.full(3, np.nan)
    return {"xi": -c, "loc": loc, "scale": sc, "se_xi": se[0], "se_loc": se[1], "se_scale": se[2], "ll": ll}

def gev_quality(m, p):
    """Fit quality: KS and Anderson-Darling on PIT values, LR test of Gumbel (xi=0) vs GEV."""
    u = genextreme.cdf(m, -p["xi"], p["loc"], p["scale"])
    ks = stats.kstest(m, genextreme.cdf, args=(-p["xi"], p["loc"], p["scale"]))
    g = gumbel_r.fit(m); ll0 = gumbel_r.logpdf(m, *g).sum(); lr = max(2 * (p["ll"] - ll0), 0)
    return {"KS_stat": ks.statistic, "KS_p(approx)": ks.pvalue, "AD_stat": ad_stat(u),
            "xi_z(H0:xi=0)": p["xi"] / p["se_xi"], "LR_gumbel": lr, "LR_p": stats.chi2.sf(lr, 1)}

def gev_quantile(p, prob):
    return genextreme.ppf(prob, -p["xi"], p["loc"], p["scale"])

# ------------------------------------------------------------------ extremal index
def theta_intervals(L, q=THETA_Q):
    """Ferro-Segers intervals estimator of the extremal index at threshold = q-quantile of the losses."""
    idx = np.flatnonzero(L.values > np.quantile(L.values, q)); T = np.diff(idx)
    if len(T) < 2: return 1.0
    if T.max() <= 2: th = 2 * T.sum() ** 2 / ((len(T)) * (T ** 2).sum())
    else: th = 2 * (T - 1).sum() ** 2 / ((len(T)) * ((T - 1) * (T - 2)).sum())
    return float(min(1.0, th))

def theta_runs(L, q=THETA_Q, r=RUN_LEN):
    """Runs estimator: clusters = exceedances separated by fewer than r non-exceedances; theta = clusters / exceedances."""
    idx = np.flatnonzero(L.values > np.quantile(L.values, q))
    return float((1 + (np.diff(idx) >= r).sum()) / len(idx))

def var_gev_loss(p, alpha, n=BLOCK, theta=1.0):
    """Loss quantile at confidence 1-alpha:  G^{-1}((1-alpha)^(n*theta)).  theta=1 -> independence."""
    return gev_quantile(p, (1 - alpha) ** (n * theta))

# ------------------------------------------------------------------ POINT 6: GPD, peaks over threshold
def fit_gpd(L, q=THRESH_Q):
    u = np.quantile(L.values, q); exc = L.values[L.values > u] - u
    xi, _, beta = genpareto.fit(exc, floc=0)
    return {"u": u, "xi": xi, "beta": beta, "Nu": len(exc), "n": len(L), "se_xi": (1 + xi) / np.sqrt(len(exc)),  # approx. asymptotic s.e.
            "exc": exc}

def gpd_quality(p):
    cdf = genpareto.cdf(p["exc"], p["xi"], 0, p["beta"])
    ks = stats.kstest(p["exc"], genpareto.cdf, args=(p["xi"], 0, p["beta"]))
    return {"Nu": p["Nu"], "KS_stat": ks.statistic, "KS_p(approx)": ks.pvalue, "AD_stat": ad_stat(cdf),
            "xi_z(H0:xi=0)": p["xi"] / p["se_xi"]}

def var_gpd_loss(p, alpha):
    """u + beta/xi * ((n*alpha/Nu)^(-xi) - 1)."""
    k = p["n"] * alpha / p["Nu"]
    return p["u"] + (p["beta"] / p["xi"] * (k ** (-p["xi"]) - 1) if abs(p["xi"]) > 1e-8 else -p["beta"] * np.log(k))

# ------------------------------------------------------------------ main
def main():
    Z = pd.read_csv("outputs/Z_insample.csv", index_col=0, parse_dates=True)
    names = list(Z.columns); n = len(names); nc = 4; nr = int(np.ceil(n / nc))
    G, GQ, TH, GP, GPQ, SENS, QZ = {}, {}, {}, {}, {}, [], []
    qn = {a: norm.ppf(a) for a in ALPHAS}

    for c in names:
        L = -Z[c].dropna()                                     # losses = -Z
        m = block_maxima(L)
        G[c] = fit_gev(m); GQ[c] = gev_quality(m, G[c])
        TH[c] = {"theta_intervals": theta_intervals(L), "theta_runs": theta_runs(L), f"n_blocks(size {BLOCK})": len(m)}
        GP[c] = fit_gpd(L); GPQ[c] = gpd_quality(GP[c])
        th = TH[c]["theta_intervals"]                          # extremal index used in Case 4
        for a in ALPHAS:
            v3 = -var_gev_loss(G[c], a, theta=1.0)             # Z quantile, independence
            v4 = -var_gev_loss(G[c], a, theta=th)              # with extremal index
            v5 = -var_gpd_loss(GP[c], a)
            QZ.append({"series": c, "alpha": a, "Normal": qn[a], "Empirical": np.quantile(Z[c].dropna(), a),
                       "GEV_indep": v3, "GEV_theta": v4, "GPD": v5})
        for q in SENS_Q:                                       # GPD threshold sensitivity
            g = fit_gpd(L, q)
            SENS.append({"series": c, "thr_q": q, "u": g["u"], "Nu": g["Nu"], "xi": g["xi"], "beta": g["beta"],
                         **{f"VaR_Z_{a}": -var_gpd_loss(g, a) for a in ALPHAS}})

    # ---------- POINT 5 tables
    save_table(pd.DataFrame(G).T.drop(columns="ll").assign(loglik=pd.Series({c: G[c]["ll"] for c in G})), "E1_GEV_parameters")
    save_table(pd.DataFrame(GQ).T, "E2_GEV_fit_quality")
    save_table(pd.DataFrame(TH).T, "E3_extremal_index")
    qz = pd.DataFrame(QZ).set_index(["series", "alpha"]); save_table(qz, "E4_VaR_of_Z_quantiles")
    # ---------- POINT 6 tables
    save_table(pd.DataFrame({c: {k: v for k, v in GP[c].items() if k != "exc"} for c in GP}).T, "E5_GPD_parameters")
    save_table(pd.DataFrame(GPQ).T, "E5b_GPD_fit_quality")
    save_table(pd.DataFrame(SENS).set_index(["series", "thr_q"]), "E6_GPD_threshold_sensitivity")

    # ---------- Figures
    # Fig E1: GEV QQ-plots (empirical block maxima vs fitted GEV quantiles)
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 3.4, nr * 3.0)); axes = axes.ravel()
    for ax, c in zip(axes, names):
        m = np.sort(block_maxima(-Z[c].dropna())); pp = (np.arange(1, len(m) + 1) - .5) / len(m)
        th = gev_quantile(G[c], pp); ax.plot(th, m, ".", ms=3); lim = [min(th.min(), m.min()), max(th.max(), m.max())]
        ax.plot(lim, lim, "r", lw=1); ax.set_title(f"{c} (xi={G[c]['xi']:.2f})", fontsize=9)
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle("Fig E1 - GEV QQ-plots of block maxima of losses (x: fitted, y: empirical)"); fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/E1_gev_qq.png", dpi=150)
    # Fig E2: GPD QQ-plots
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 3.4, nr * 3.0)); axes = axes.ravel()
    for ax, c in zip(axes, names):
        e = np.sort(GP[c]["exc"]); pp = (np.arange(1, len(e) + 1) - .5) / len(e)
        th = genpareto.ppf(pp, GP[c]["xi"], 0, GP[c]["beta"]); ax.plot(th, e, ".", ms=3)
        ax.plot([0, max(th.max(), e.max())], [0, max(th.max(), e.max())], "r", lw=1); ax.set_title(f"{c} (xi={GP[c]['xi']:.2f})", fontsize=9)
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle(f"Fig E2 - GPD QQ-plots of exceedances over the {THRESH_Q:.0%} quantile"); fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/E2_gpd_qq.png", dpi=150)
    # Fig E3: threshold choice diagnostics for PORT and UBS: mean residual life + xi stability
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for j, c in enumerate(["UBS", "PORT"]):
        L = (-Z[c].dropna()).values; qs = np.arange(0.80, 0.985, 0.005); us = np.quantile(L, qs)
        axes[0, j].plot(us, [(L[L > u] - u).mean() for u in us]); axes[0, j].set_title(f"{c} - mean residual life"); axes[0, j].set_xlabel("threshold u")
        xs = [fit_gpd(pd.Series(L), q)["xi"] for q in qs]; axes[1, j].plot(qs, xs); axes[1, j].axvline(THRESH_Q, color="r", ls="--")
        axes[1, j].set_title(f"{c} - GPD shape xi vs threshold quantile"); axes[1, j].set_xlabel("quantile of threshold")
    fig.suptitle("Fig E3 - GPD threshold diagnostics"); fig.tight_layout(); fig.savefig(f"{FIG_DIR}/E3_gpd_threshold.png", dpi=150)

    # ---------- POINT 7: VaR Cases 3, 4, 5 for returns = mu_t + sigma_t * q_Z (q_Z: EVT quantile of Z, Gaussian-GARCH sigma_t)
    rows = []
    for c in names:
        f = pd.read_csv(f"{FC_DIR}/{c}.csv", index_col=0, parse_dates=True)
        for a in ALPHAS:
            q = qz.loc[(c, a)]
            f[f"VaR_case3_{a}"] = f["mu"] + f["sigma_normal"] * q["GEV_indep"]
            f[f"VaR_case4_{a}"] = f["mu"] + f["sigma_normal"] * q["GEV_theta"]
            f[f"VaR_case5_{a}"] = f["mu"] + f["sigma_normal"] * q["GPD"]
            rows.append({"series": c, "alpha": a, "expected_hits": a * len(f),
                         **{f"hits_case{k}": int((f["ret"] < f[f"VaR_case{k}_{a}"]).sum()) for k in range(1, 6)}})
        f.to_csv(f"{FC_DIR}/{c}.csv")
    save_table(pd.DataFrame(rows).set_index(["series", "alpha"]), "E7_exceedance_counts_all_cases")

if __name__ == "__main__":
    main()
