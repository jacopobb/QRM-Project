"""Point 9 (BONUS) - Expected Shortfall (ES) for the out-of-sample year, plotted with returns and VaR.

Cases computed
  Case 5 (main)  : Gaussian GARCH + GPD tail of the standardised residuals (VaR Case 5 of Point 7).
  Case 1 (bench) : Gaussian GARCH, ES_t = mu_t - sigma_t * phi(z_alpha) / alpha.

Why Case 5 for the main ES
  - ES is the average loss beyond the VaR. The GPD is fitted exactly to the exceedances over a threshold, i.e. to the part
    of the distribution that ES averages; the GEV is fitted to block maxima and does not describe the tail beyond the VaR.
  - It has a closed form. For the loss L = -Z above u:   ES_p = (VaR_p + beta - xi*u) / (1 - xi)   (needs xi < 1).
    Under the GEV the ES would need numerical integration.
  - Return ES:  ES_t = mu_t - sigma_t * ES_p(Z)  (negative number, like the VaR), with the same mu_t and sigma_t
    (Gaussian GARCH forecasts) used for VaR Case 5 in evt.py.
  - To justify Case 5 over Case 4 in the report, use the violation counts of backtest.py (tables T8.2*, T8.3);
    do not hard-code numbers here, they change whenever the data or the thresholds change.

INPUT CONTRACT (this script does not import the other modules)
  - <dir>/<series>.csv        (default outputs/forecasts, written by garch_models.py and evt.py)
        columns: ret, mu, sigma_normal (names editable below); VaR_case5_<alpha> is used only as a consistency check.
  - GPD parameters per series, read by default from the tables written by evt.py:
        outputs/tables/E6_var_Z_gpd.csv     (xi, beta per series)
        outputs/tables/E6_exceedances.csv   (threshold u, number of exceedances N(u), sample size T per series)
    A single file with columns u, xi, beta, Nu, n (one row per series) can be passed with --params instead.
  - Alpha levels are detected from the column names VaR_case5_<alpha> (or VaR_case1_<alpha>).

Run after garch_models.py and evt.py:   python es.py [--dir outputs/forecasts] [--tables outputs/tables] [--out outputs]
Outputs (TABLES, in <out>/tables; FIGURES, in <out>/figures; ES series, in <out>/es/<series>.csv)
  S0_ES_Z_gpd_vs_gauss   VaR and ES of the standardised residuals Z: GPD against Gaussian, every series
  S1_ES_summary          average VaR / ES, ES/VaR, violations and realised shortfall on violation days, series x alpha x case
  S2_ES_pooled_check     McNeil-Frey style check of the ES, pooled over the series
  S3_ES_check_by_series  same check, series by series
  Fig S1  returns, VaR and ES (focus series, one column per alpha)
  Fig S2  ES / VaR ratio, GPD against Gaussian, every series
  Fig S3  returns, VaR and ES Case 5, every series (one figure per alpha)
  Fig S4  realised minus forecast shortfall on violation days, every series, Case 5 against Case 1
  Fig S5  ES - VaR spread over time (focus series), GPD against Gaussian
  Fig S6  standardised ES residuals (r - ES)/sigma on violation days, pooled (mean 0 if the ES is right)
  Fig S7  ES of Z, GPD against Gaussian, every series
  Fig S8  return against forecast ES on violation days (focus series): points below the diagonal breach the ES
"""
import os, re, glob, argparse, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

MU, SIGMA, RET = "mu", "sigma_normal", "ret"        # column names in the forecast files
DEFAULT_FOCUS = ["UBS", "PORT"]                     # plotted in detail (+ the series with most violations, added automatically)
TOL_VAR = 1e-3                                      # tolerated gap between rebuilt VaR Case 5 and the one in the forecast file (%)
MIN_HITS = 5                                        # minimum number of violation days to run the ES check


# ------------------------------------------------------------------ GPD VaR / ES of the loss of Z (L = -Z)
def var_gpd(g, a):
    """Slide 72: q_a = u + beta/xi * ((a / (N_u/T))^(-xi) - 1); a is the tail probability (0.05, 0.01)."""
    k = g["n"] * a / g["Nu"]
    if abs(g["xi"]) < 1e-8:                          # xi -> 0: exponential tail
        return g["u"] - g["beta"] * np.log(k)
    return g["u"] + g["beta"] / g["xi"] * (k ** (-g["xi"]) - 1)


def es_gpd(g, a):
    """ES_p = (VaR_p + beta - xi*u) / (1 - xi): finite only if xi < 1 (NaN otherwise)."""
    if g["xi"] >= 1:
        return np.nan
    return (var_gpd(g, a) + g["beta"] - g["xi"] * g["u"]) / (1 - g["xi"])


def es_gauss(a):
    """E[-Z | Z < z_a] for a standard Normal."""
    return norm.pdf(norm.ppf(a)) / a


# ------------------------------------------------------------------ loading
def load_gpd_params(tab_dir, params=None):
    """DataFrame indexed by series with columns u, xi, beta, Nu (exceedances), n (sample size of the losses)."""
    if params:
        P = pd.read_csv(params, index_col=0)
    else:
        f_var, f_exc = f"{tab_dir}/E6_var_Z_gpd.csv", f"{tab_dir}/E6_exceedances.csv"
        for f in (f_var, f_exc):
            if not os.path.exists(f):
                raise FileNotFoundError(f"{f} not found: run evt.py first (or pass --params)")
        v = pd.read_csv(f_var).drop_duplicates("series").set_index("series")[["xi", "beta"]]
        e = pd.read_csv(f_exc, index_col=0).rename(columns={"Threshold_u": "u", "Exceedances_N_u": "Nu", "Total_Obs_T": "n"})
        P = e[["u", "Nu", "n"]].join(v, how="inner")
    missing = {"u", "xi", "beta", "Nu", "n"} - set(P.columns)
    if missing:
        raise KeyError(f"GPD parameter table lacks columns {sorted(missing)}")
    return P[["u", "xi", "beta", "Nu", "n"]]


def find_col(d, prefix, a):
    """Column '<prefix>_<alpha>' whatever the number format of alpha (0.05, .05), or None."""
    for name in (f"{prefix}_{a:g}", f"{prefix}_{a}"):
        if name in d:
            return name
    for c in d:
        m = re.match(rf"^{prefix}_(0?\.\d+)$", c)
        if m and np.isclose(float(m.group(1)), a):
            return c
    return None


def detect_alphas(files):
    al = set()
    for f in files.values():
        for c in pd.read_csv(f, nrows=1).columns:
            m = re.match(r"^VaR_case[15]_(0?\.\d+)$", c)
            if m:
                al.add(float(m.group(1)))
    return sorted(al, reverse=True)


def ttest(x, min_n=MIN_HITS):
    """One-sample t-test of mean 0 (two-sided) and of mean < 0 (ES too small in absolute value)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < min_n:
        return {"n_violations": len(x), "mean_std_residual": x.mean() if len(x) else np.nan, "t_stat": np.nan, "p_two_sided": np.nan, "p_mean_below_0": np.nan}
    t2, t1 = stats.ttest_1samp(x, 0), stats.ttest_1samp(x, 0, alternative="less")
    return {"n_violations": len(x), "mean_std_residual": x.mean(), "t_stat": t2.statistic, "p_two_sided": t2.pvalue, "p_mean_below_0": t1.pvalue}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/forecasts"); ap.add_argument("--tables", default="outputs/tables")
    ap.add_argument("--params", default=None, help="optional single CSV with columns u, xi, beta, Nu, n")
    ap.add_argument("--out", default="outputs"); ap.add_argument("--focus", nargs="*", default=DEFAULT_FOCUS)
    ar = ap.parse_args()
    TAB, FIG, ESD = f"{ar.out}/tables", f"{ar.out}/figures", f"{ar.out}/es"
    for d in (TAB, FIG, ESD):
        os.makedirs(d, exist_ok=True)

    P = load_gpd_params(ar.tables, ar.params)
    files = {os.path.splitext(os.path.basename(f))[0]: f for f in sorted(glob.glob(f"{ar.dir}/*.csv"))}
    alphas = detect_alphas(files)
    if not alphas:
        raise ValueError(f"no VaR_case5_<alpha> / VaR_case1_<alpha> columns in {ar.dir}")
    print("Series with forecasts:", list(files), "| alphas:", alphas,
          "| GPD xi range: %.3f to %.3f" % (P["xi"].min(), P["xi"].max()))

    rows, zrows, data, resid = [], [], {}, {}
    for s, f in files.items():
        if s not in P.index:
            print(f"  {s}: no GPD parameters (was it dropped in evt.py?), skipped")
            continue
        d = pd.read_csv(f, index_col=0, parse_dates=True); g = P.loc[s]; out = d[[RET, MU, SIGMA]].copy()
        for a in alphas:
            vz, ez = var_gpd(g, a), es_gpd(g, a)                                  # loss quantile and ES of Z (positive numbers)
            if vz <= g["u"] or not np.isfinite(ez):
                print(f"  WARNING {s} alpha={a:g}: VaR_Z={vz:.3f} <= u={g['u']:.3f} or xi={g['xi']:.3f} >= 1, GPD ES not valid")
            zrows.append({"series": s, "alpha": a, "u": g["u"], "xi": g["xi"], "beta": g["beta"], "VaR_Z_GPD": vz, "ES_Z_GPD": ez,
                          "ES/VaR_GPD": ez / vz, "VaR_Z_Gauss": -norm.ppf(a), "ES_Z_Gauss": es_gauss(a), "ES/VaR_Gauss": es_gauss(a) / -norm.ppf(a)})
            out[f"VaR_case5_{a:g}"] = d[MU] - d[SIGMA] * vz                       # return VaR/ES: mu_t - sigma_t * (loss quantile of Z)
            out[f"ES_case5_{a:g}"] = d[MU] - d[SIGMA] * ez
            out[f"VaR_case1_{a:g}"] = d[MU] + d[SIGMA] * norm.ppf(a)
            out[f"ES_case1_{a:g}"] = d[MU] - d[SIGMA] * es_gauss(a)
            c5 = find_col(d, "VaR_case5", a)                                      # consistency check with evt.py (Point 7)
            gap = float((out[f"VaR_case5_{a:g}"] - d[c5]).abs().max()) if c5 else np.nan
            if gap > TOL_VAR:
                print(f"  WARNING {s} alpha={a:g}: rebuilt VaR Case 5 differs from the file by up to {gap:.4f}")
            for case in ("case5", "case1"):                                       # realised shortfall on violation days vs forecast ES
                v, e = out[f"VaR_{case}_{a:g}"], out[f"ES_{case}_{a:g}"]; h = out[RET] < v
                rows.append({"series": s, "alpha": a, "case": case, "T": len(out), "expected_hits": a * len(out), "hits": int(h.sum()),
                             "days_below_ES": int((out[RET] < e).sum()), "avg_VaR_%": v.mean(), "avg_ES_%": e.mean(), "ES/VaR": e.mean() / v.mean(),
                             "mean_ret_on_hits_%": out[RET][h].mean() if h.any() else np.nan,
                             "mean_ES_on_hits_%": e[h].mean() if h.any() else np.nan,
                             "max_VaR_diff_vs_file": gap if case == "case5" else np.nan})
                resid[(case, a, s)] = ((out[RET] - e)[h] / out[SIGMA][h]).values   # standardised ES residuals, mean 0 if the ES is right
        out.to_csv(f"{ESD}/{s}.csv"); data[s] = out
    if not data:
        raise RuntimeError("no series with both forecasts and GPD parameters")

    # ---------- TABLE S0: VaR and ES of Z, GPD vs Gaussian
    t0 = pd.DataFrame(zrows).set_index(["series", "alpha"]).sort_index(level=[1, 0], ascending=[False, True])
    print("\n=== S0_ES_Z_gpd_vs_gauss ===\n", t0.round(4).to_string()); t0.round(6).to_csv(f"{TAB}/S0_ES_Z_gpd_vs_gauss.csv")

    # ---------- TABLE S1: ES summary, series x alpha x case
    t = pd.DataFrame(rows).set_index(["series", "alpha", "case"]).sort_index(level=[1, 0], ascending=[False, True])
    t["gap_on_hits_%"] = t["mean_ret_on_hits_%"] - t["mean_ES_on_hits_%"]          # < 0: realised loss worse than the ES
    print("\n=== S1_ES_summary ===\n", t.round(4).to_string()); t.round(6).to_csv(f"{TAB}/S1_ES_summary.csv")

    # ---------- TABLES S2, S3: McNeil-Frey style check, (r - ES)/sigma on violation days should have mean 0
    # S2 pools the violation days of all series: they are cross-correlated (same stress days), so the p-values are optimistic
    pr = [{"case": case, "alpha": a, **ttest(np.concatenate([resid[(case, a, s)] for s in data]), min_n=3)} for case in ("case5", "case1") for a in alphas]
    pr = pd.DataFrame(pr).set_index(["case", "alpha"]); print("\n=== S2_ES_pooled_check ===\n", pr.round(4).to_string()); pr.round(6).to_csv(f"{TAB}/S2_ES_pooled_check.csv")
    bs = pd.DataFrame([{"series": s, "alpha": a, "case": case, **ttest(resid[(case, a, s)])} for s in data for a in alphas for case in ("case5", "case1")])
    bs = bs.set_index(["series", "alpha", "case"]).sort_index(level=[1, 0], ascending=[False, True])
    print(f"\n=== S3_ES_check_by_series (test run only with at least {MIN_HITS} violations) ===\n", bs.round(4).to_string()); bs.round(6).to_csv(f"{TAB}/S3_ES_check_by_series.csv")

    # ---------- Fig S1: returns, VaR and ES, one row per focus series, one column per alpha
    small = min(alphas)
    extra = t.xs("case5", level="case").xs(small, level="alpha")["hits"].idxmax()   # series with most Case 5 violations
    focus = list(dict.fromkeys([s for s in ar.focus if s in data] + [extra]))
    fig, axes = plt.subplots(len(focus), len(alphas), figsize=(7 * len(alphas), 4 * len(focus)), squeeze=False)
    for i, s in enumerate(focus):
        d = data[s]
        for j, a in enumerate(alphas):
            ax = axes[i, j]; ax.plot(d[RET], color="grey", lw=.6, label="return")
            ax.plot(d[f"VaR_case5_{a:g}"], "b", lw=1.2, label="VaR (Case 5)"); ax.plot(d[f"ES_case5_{a:g}"], "r", lw=1.2, label="ES (Case 5)")
            ax.plot(d[f"ES_case1_{a:g}"], "g--", lw=.9, label="ES Gaussian (Case 1)")
            h = d[RET] < d[f"VaR_case5_{a:g}"]; ax.plot(d.index[h], d[RET][h], "ko", ms=4, label=f"VaR violations ({int(h.sum())})")
            ax.set_title(f"{s} - alpha = {a:g}"); ax.set_ylabel("%"); ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Fig S1 - Returns, VaR and Expected Shortfall (GARCH + GPD, Case 5)"); fig.tight_layout(); fig.savefig(f"{FIG}/S1_ES_VaR_returns.png", dpi=150)

    # ---------- Fig S2: ES/VaR ratio, GPD vs Gaussian, every series
    fig, axes = plt.subplots(1, len(alphas), figsize=(6.5 * len(alphas), 4), squeeze=False)
    for ax, a in zip(axes.ravel(), alphas):
        r = t.xs(a, level="alpha")["ES/VaR"].unstack("case"); r.plot(kind="bar", ax=ax, width=.8); ax.set_title(f"ES / VaR, alpha = {a:g}")
        ax.axhline(1, color="k", lw=.5); ax.tick_params(axis="x", labelsize=7); ax.set_ylim(.9, None)
    fig.suptitle("Fig S2 - ES/VaR ratio: GPD-based (case5) vs Gaussian (case1)"); fig.tight_layout(); fig.savefig(f"{FIG}/S2_ES_VaR_ratio.png", dpi=150)

    # ---------- Fig S3: returns, VaR and ES Case 5 for every series, one figure per alpha
    for a in alphas:
        n = len(data); nc = 4; nr = int(np.ceil(n / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(nc * 5, nr * 3.2), sharex=True, squeeze=False); axes = axes.ravel()
        for ax, (s, d) in zip(axes, data.items()):
            ax.plot(d[RET], color="lightgrey", lw=.8, label="return"); ax.plot(d[f"VaR_case5_{a:g}"], "b", lw=1, label="VaR (Case 5)")
            ax.plot(d[f"ES_case5_{a:g}"], "r", lw=1, label="ES (Case 5)"); ax.set_title(s, fontsize=9); ax.tick_params(labelsize=7)
        for ax in axes[n:]: ax.axis("off")
        h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=10)
        fig.suptitle(f"Fig S3 - Returns, VaR and ES (Case 5), alpha = {a:g}, all series", y=.98); fig.tight_layout(rect=(0, .04, 1, .96))
        fig.savefig(f"{FIG}/S3_ES_all_series_alpha_{a:g}.png", dpi=150)

    # ---------- Fig S4: realised minus forecast ES on violation days (< 0: the ES is too small in absolute value)
    fig, axes = plt.subplots(len(alphas), 1, figsize=(12, 3.8 * len(alphas)), squeeze=False)
    for ax, a in zip(axes.ravel(), alphas):
        t.xs(a, level="alpha")["gap_on_hits_%"].unstack("case")[["case5", "case1"]].plot(kind="bar", ax=ax, width=.8)
        ax.axhline(0, color="k", lw=.7); ax.set_ylabel("% points"); ax.set_title(f"alpha = {a:g}"); ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Fig S4 - Mean return minus mean forecast ES on violation days"); fig.tight_layout(); fig.savefig(f"{FIG}/S4_ES_gap_on_hits.png", dpi=150)

    # ---------- Fig S5: ES - VaR spread over time, focus series (how much deeper the tail is beyond the VaR)
    fig, axes = plt.subplots(len(focus), len(alphas), figsize=(7 * len(alphas), 3.5 * len(focus)), squeeze=False)
    for i, s in enumerate(focus):
        d = data[s]
        for j, a in enumerate(alphas):
            ax = axes[i, j]
            ax.plot(d[f"ES_case5_{a:g}"] - d[f"VaR_case5_{a:g}"], "r", lw=1.2, label="ES - VaR, GPD (Case 5)")
            ax.plot(d[f"ES_case1_{a:g}"] - d[f"VaR_case1_{a:g}"], "g--", lw=1, label="ES - VaR, Gaussian (Case 1)")
            ax.axhline(0, color="k", lw=.5); ax.set_title(f"{s} - alpha = {a:g}"); ax.set_ylabel("% points"); ax.legend(fontsize=7)
    fig.suptitle("Fig S5 - Distance between ES and VaR over time"); fig.tight_layout(); fig.savefig(f"{FIG}/S5_ES_VaR_spread.png", dpi=150)

    # ---------- Fig S6: pooled standardised ES residuals on violation days
    fig, axes = plt.subplots(len(alphas), 2, figsize=(11, 3.6 * len(alphas)), squeeze=False)
    for i, a in enumerate(alphas):
        for j, case in enumerate(("case5", "case1")):
            x = np.concatenate([resid[(case, a, s)] for s in data]); ax = axes[i, j]
            ax.hist(x, bins=20, color="steelblue", edgecolor="black", alpha=.7)
            ax.axvline(0, color="k", lw=1); ax.axvline(x.mean() if len(x) else 0, color="r", ls="--", lw=1.3, label=f"mean {x.mean():.2f}" if len(x) else "")
            ax.set_title(f"{case} - alpha = {a:g} (n = {len(x)})"); ax.legend(fontsize=8)
    fig.suptitle("Fig S6 - (return - ES)/sigma on violation days, all series pooled"); fig.tight_layout(); fig.savefig(f"{FIG}/S6_ES_residuals_hist.png", dpi=150)

    # ---------- Fig S7: ES of the standardised residuals, GPD vs Gaussian
    fig, axes = plt.subplots(1, len(alphas), figsize=(6.5 * len(alphas), 4), squeeze=False)
    for ax, a in zip(axes.ravel(), alphas):
        t0.xs(a, level="alpha")[["ES_Z_GPD", "ES_Z_Gauss"]].plot(kind="bar", ax=ax, width=.8)
        ax.set_title(f"ES of Z, alpha = {a:g}"); ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Fig S7 - Expected Shortfall of the standardised residuals: GPD vs Gaussian"); fig.tight_layout(); fig.savefig(f"{FIG}/S7_ES_Z_gpd_vs_gauss.png", dpi=150)

    # ---------- Fig S8: return vs forecast ES (Case 5) on violation days; below the diagonal = ES breached
    fig, axes = plt.subplots(len(focus), len(alphas), figsize=(6 * len(alphas), 4.5 * len(focus)), squeeze=False)
    for i, s in enumerate(focus):
        d = data[s]
        for j, a in enumerate(alphas):
            ax = axes[i, j]; h = d[RET] < d[f"VaR_case5_{a:g}"]; x, y = d[f"ES_case5_{a:g}"][h], d[RET][h]
            ax.scatter(x, y, s=18, color="k")
            lim = [min(x.min(), y.min()) - .2, max(x.max(), y.max()) + .2] if h.any() else [-1, 0]
            ax.plot(lim, lim, "r-", lw=1, label="return = ES"); ax.set_xlabel("forecast ES (%)"); ax.set_ylabel("return (%)")
            ax.set_title(f"{s} - alpha = {a:g}: {int((y < x).sum())} of {int(h.sum())} violations below ES"); ax.legend(fontsize=7)
    fig.suptitle("Fig S8 - Return against forecast ES on VaR violation days (Case 5)"); fig.tight_layout(); fig.savefig(f"{FIG}/S8_ES_scatter_on_hits.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
