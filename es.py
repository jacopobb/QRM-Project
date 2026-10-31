

"""Point 9 (BONUS): Expected Shortfall under VaR Case 5 (Gaussian GARCH + GPD tail of the standardised residuals).

Why Case 5: the GPD models exactly the part of the distribution that ES averages (the exceedances beyond a threshold),
and it gives ES in closed form:  for losses of Z above u,  ES_p = (VaR_p + beta - xi*u) / (1 - xi)   (needs xi < 1).
Return ES_t = mu_t - sigma_t * ES_p(Z) (negative number, like the VaR). The Gaussian ES (Case 1) is added as benchmark:
ES_t = mu_t - sigma_t * phi(z_alpha)/alpha.


##########################################################################################################################
Why Case 5

It models the part ES averages. ES is the average loss beyond the VaR. The GPD is fitted to exactly those exceedances over a threshold, whereas the GEV is fitted to block maxima and does not describe the tail beyond the VaR directly.
It has a closed form. For the loss of Z, ES = (VaR + β − ξu) / (1 − ξ), which needs ξ < 1. Here ξ ranges from 0.04 to 0.31 across the series, so the condition holds. Under the GEV, ES needs numerical integration and is harder to defend.
It behaves well in the backtest. At 1%, Case 5 has 48 violations against 42.5 expected and no rejection by any test. The Gaussian case has 80 violations, and Kupiec rejects it for 3 series.
It is not the best case on every count. Case 4 has slightly fewer violations (46 at 1%, 226 at 5%) and is a bit closer to the target. I still prefer Case 5 for points 1 to 3, and a Case 4 ES would be harder to justify. At 5%, Case 5 has 236 violations against 212.5 expected, which is above the target but not rejected.

##########################################################################################################################
INPUT CONTRACT (so it keeps working if the GARCH/EVT files change)
  - outputs/forecasts/<series>.csv with columns: ret, mu, sigma_normal (names editable below), VaR_case5_<alpha>
  - outputs/tables/E5_GPD_parameters.csv with columns u, xi, beta, Nu, n (one row per series; from evt.py)
Run after garch_models.py and evt.py:   python es.py  [--dir outputs/forecasts] [--params outputs/tables/E5_GPD_parameters.csv]
Outputs: outputs/es/<series>.csv, outputs/tables/S*.csv, outputs/figures/S*.png

##########################################################################################################################
"""
import os, re, glob, argparse, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
MU, SIGMA, RET = "mu", "sigma_normal", "ret"        # column names in the forecast files
VAR5 = "VaR_case5_{a:g}"                            # column of the GPD VaR, used as consistency check
DEFAULT_FOCUS = ["UBS", "PORT"]                     # plotted in detail (+ the series with most violations, added automatically)

# ------------------------------------------------------------------ GPD VaR / ES of the loss of Z
def var_gpd(g, a):
    k = g["n"] * a / g["Nu"]
    return g["u"] + g["beta"] / g["xi"] * (k ** (-g["xi"]) - 1)

def es_gpd(g, a):
    assert g["xi"] < 1, "ES under GPD requires xi < 1"
    return (var_gpd(g, a) + g["beta"] - g["xi"] * g["u"]) / (1 - g["xi"])

def es_gauss(a):
    return norm.pdf(norm.ppf(a)) / a                 # E[-Z | Z < z_alpha] for a standard Normal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/forecasts"); ap.add_argument("--params", default="outputs/tables/E5_GPD_parameters.csv")
    ap.add_argument("--out", default="outputs"); ap.add_argument("--focus", nargs="*", default=DEFAULT_FOCUS)
    ar = ap.parse_args()
    TAB, FIG, ESD = f"{ar.out}/tables", f"{ar.out}/figures", f"{ar.out}/es"
    for d in (TAB, FIG, ESD): os.makedirs(d, exist_ok=True)
    P = pd.read_csv(ar.params, index_col=0)
    files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(f"{ar.dir}/*.csv")}
    alphas = sorted({float(m.group(1)) for f in files.values() for c in pd.read_csv(f, nrows=1).columns
                     if (m := re.match(r"^VaR_case5_(0?\.\d+)$", c))}, reverse=True)
    print("Series:", list(files), "| alphas:", alphas, "| GPD xi range:", P["xi"].min().round(3), P["xi"].max().round(3))

    rows, data, pooled = [], {}, {a: [] for a in alphas}
    for s, f in files.items():
        if s not in P.index: print("  no GPD parameters for", s); continue
        d = pd.read_csv(f, index_col=0, parse_dates=True); g = P.loc[s]; out = d[[RET, MU, SIGMA]].copy()
        for a in alphas:
            vz, ez = var_gpd(g, a), es_gpd(g, a)
            out[f"VaR_case5_{a:g}"] = d[MU] - d[SIGMA] * vz
            out[f"ES_case5_{a:g}"] = d[MU] - d[SIGMA] * ez
            out[f"VaR_case1_{a:g}"] = d[MU] + d[SIGMA] * norm.ppf(a)
            out[f"ES_case1_{a:g}"] = d[MU] - d[SIGMA] * es_gauss(a)
            chk = float(np.abs(out[f"VaR_case5_{a:g}"] - d[VAR5.format(a=a)]).max()) if VAR5.format(a=a) in d else np.nan
            for case in ("case5", "case1"):                    # realised shortfall on violation days vs forecast ES
                v, e = out[f"VaR_{case}_{a:g}"], out[f"ES_{case}_{a:g}"]; h = out[RET] < v
                rows.append({"series": s, "alpha": a, "case": case, "hits": int(h.sum()), "avg_VaR_%": v.mean(), "avg_ES_%": e.mean(),
                             "ES/VaR": e.mean() / v.mean(), "mean_ret_on_hits_%": out[RET][h].mean() if h.any() else np.nan,
                             "mean_ES_on_hits_%": e[h].mean() if h.any() else np.nan, "max_VaR_diff_vs_file": chk if case == "case5" else np.nan})
                if case == "case5": pooled[a] += list(((out[RET] - e)[h] / out[SIGMA][h]))   # standardised ES residuals
        out.to_csv(f"{ESD}/{s}.csv"); data[s] = out
    t = pd.DataFrame(rows).set_index(["series", "alpha", "case"]).sort_index(level=[1, 0], ascending=[False, True])
    print("\n=== S1_ES_summary ===\n", t.round(4).to_string()); t.round(6).to_csv(f"{TAB}/S1_ES_summary.csv")

    # ---------- S2: pooled ES check (McNeil-Frey style): standardised (r - ES)/sigma on violation days should have mean 0
    pr = []
    for a, x in pooled.items():
        x = np.array(x); tt = stats.ttest_1samp(x, 0) if len(x) > 2 else None
        pr.append({"alpha": a, "n_violations_pooled": len(x), "mean_std_residual": x.mean() if len(x) else np.nan,
                   "t_stat": tt.statistic if tt else np.nan, "p_value": tt.pvalue if tt else np.nan})
    pr = pd.DataFrame(pr).set_index("alpha"); print("\n=== S2_ES_pooled_check (Case 5) ===\n", pr.round(4).to_string()); pr.round(6).to_csv(f"{TAB}/S2_ES_pooled_check.csv")

    # ---------- Figures: returns, VaR and ES, one row per focus series, one column per alpha
    small = min(alphas); case5 = t.xs("case5", level="case"); 
    extra = case5.xs(small, level="alpha")["hits"].idxmax()
    focus = list(dict.fromkeys([s for s in ar.focus if s in data] + [extra]))
    fig, axes = plt.subplots(len(focus), len(alphas), figsize=(7 * len(alphas), 4 * len(focus)), squeeze=False)
    for i, s in enumerate(focus):
        d = data[s]
        for j, a in enumerate(alphas):
            ax = axes[i, j]; ax.plot(d[RET], color="grey", lw=.6, label="return")
            ax.plot(d[f"VaR_case5_{a:g}"], "b", lw=1.2, label="VaR (Case 5)"); ax.plot(d[f"ES_case5_{a:g}"], "r", lw=1.2, label="ES (Case 5)")
            ax.plot(d[f"ES_case1_{a:g}"], "g--", lw=.9, label="ES Gaussian (Case 1)")
            h = d[RET] < d[f"VaR_case5_{a:g}"]; ax.plot(d.index[h], d[RET][h], "ko", ms=4, label=f"VaR violations ({int(h.sum())})")
            ax.set_title(f"{s} - alpha = {a:g}"); ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Fig S1 - Returns, VaR and Expected Shortfall (GARCH + GPD, Case 5)"); fig.tight_layout(); fig.savefig(f"{FIG}/S1_ES_VaR_returns.png", dpi=150)
    fig, axes = plt.subplots(1, len(alphas), figsize=(6.5 * len(alphas), 4), squeeze=False)    # S2: ES/VaR ratio, EVT vs Gaussian
    for ax, a in zip(axes.ravel(), alphas):
        r = t.xs(a, level="alpha")["ES/VaR"].unstack("case"); r.plot(kind="bar", ax=ax, width=.8); ax.set_title(f"ES / VaR, alpha = {a:g}")
        ax.axhline(1, color="k", lw=.5); ax.tick_params(axis="x", labelsize=7); ax.set_ylim(.9, None)
    fig.suptitle("Fig S2 - ES/VaR ratio: GPD-based vs Gaussian"); fig.tight_layout(); fig.savefig(f"{FIG}/S2_ES_VaR_ratio.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
