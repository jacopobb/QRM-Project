"""Point 2 - descriptive analysis on the FULL sample (log returns, in %).
Run:  python descriptive.py
Prints every table; with SAVE = True it also saves tables in outputs/tables and figures in outputs/figures.

Outputs follow the order of the report (sections 3.1-3.6).
  TABLES                                                 FIGURES
  T3.0   price_performance                               F3.1  prices
  T3.1a  moments_normality                               F3.2  returns
  T3.1b  vol_decomposition                               F3.3  histograms
  T3.1c  lonza_largest_moves                             F3.4  QQ-plots Normal
  T3.2a  extreme_events                                  F3.5  QQ-plots t(3)
  T3.2b  dates_with_most_extremes                        F3.6a ACF returns
  T3.2c  extreme_sigma                                   F3.6b ACF squared returns
  T3.3   serial_dependence_clustering                    F3.7  correlation heatmap
  T3.4a  correlation_matrix                              F3.8  rolling moments
  T3.4b  correlation_summary                             F3.9  rolling correlation
  T3.4c  sector_block_correlation
  T3.4d  avg_corr_by_stock
  T3.5a  annualised_vol_by_year
  T3.5b  levene_test_by_year
  T3.5c  avg_correlation_by_year
  T3.5ac vol_corr_summary_by_year
  T3.5d  rolling_moments_extremes
  T3.5e  rolling_moments_by_year
  T3.5f  rolling_corr_extremes
  T3.5g  rolling_corr_by_year
"""

import os, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf
try:
    from settings import *
except ModuleNotFoundError:
    from config import *
import data_prep

warnings.filterwarnings("ignore", category=FutureWarning)   # statsmodels het_arch notice
SAVE = True       # True: write CSV and PNG files
TABLE_DIR, FIG_DIR = "outputs/tables", "outputs/figures"
os.makedirs(TABLE_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
W = 250            # rolling window (about 1 trading year)
NLAGS = 20         # lags for ACF significance counts
K = 3              # extreme event = |standardised return| > K

def save_table(df, name, digits=4):
    print(f"\n=== {name} ===\n", df.round(digits).to_string())
    if SAVE:
        df.round(6).to_csv(f"{TABLE_DIR}/{name}.csv")


def save_fig(fig, name):
    if SAVE:
        fig.savefig(f"{FIG_DIR}/{name}.png", dpi=150)


def grid(n, ncols=4, h=2.5, w=3.8):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * w, nrows * h))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]: ax.axis("off")
    return fig, axes


# ------------------------------------------------------------------ data
def load_data():
    px = data_prep.clean_prices(data_prep.load_raw(), verbose=False)
    names = {v[0]: name for name, v in STOCKS.items()}               # RIC -> company name
    px = px.rename(columns=names)
    order = sorted(px.columns, key=lambda n: (SECTOR[n], -INDEX_WEIGHT[n]))   # group by sector, then by index weight
    px = px[order]
    return px, data_prep.log_returns(px)                             # log returns in %


# ------------------------------------------------------------------ TABLE 3.1a: moments and normality
def table_moments(r):
    t = pd.DataFrame({
        "mean_daily_%": r.mean(), "ann_mean_%": r.mean() * 252,
        "std_daily_%": r.std(), "ann_vol_%": r.std() * np.sqrt(252),
        "min_%": r.min(), "max_%": r.max(),
        "skew": r.apply(stats.skew), "skew_test_p": r.apply(lambda s: stats.skewtest(s).pvalue),
        "ex_kurt": r.apply(stats.kurtosis), "kurt_test_p": r.apply(lambda s: stats.kurtosistest(s).pvalue),
        "JB_stat": r.apply(lambda s: stats.jarque_bera(s).statistic),
        "JB_p": r.apply(lambda s: stats.jarque_bera(s).pvalue)})
    t.insert(0, "sector", [SECTOR[c] for c in t.index]); t.insert(1, "index_weight_%", [INDEX_WEIGHT[c] for c in t.index])
    return t


# ------------------------------------------------------------------ TABLE 3.1b: volatility decomposition (market vs idiosyncratic)
def table_vol_decomposition(r):
    rows = []
    for c in r.columns:
        mkt = r.drop(columns=c).mean(axis=1)                         # equal-weighted market excluding the stock itself
        rho = r[c].corr(mkt); vol = r[c].std() * np.sqrt(252)
        rows.append({"stock": c, "sector": SECTOR[c], "ann_vol_%": vol,
                     "beta": r[c].cov(mkt) / mkt.var(), "R2_market_%": 100 * rho ** 2,
                     "idio_vol_%": vol * np.sqrt(1 - rho ** 2),
                     "min_%": r[c].min(), "date_min": r[c].idxmin().date()})
    return pd.DataFrame(rows).set_index("stock").sort_values("ann_vol_%")


def table_largest_moves(r, stock="Lonza", peers=("Roche", "Novartis"), n=10):
    """Largest absolute moves of one stock against its peers and the equal-weighted market (without the stock)."""
    peers = list(peers)
    mkt = r.drop(columns=stock).mean(axis=1)
    days = r[stock].abs().sort_values(ascending=False).head(n).index
    ev = pd.DataFrame({f"{stock}_%": r.loc[days, stock], "peers_avg_%": r.loc[days, peers].mean(axis=1),
                       f"mkt_ex_{stock}_%": mkt.loc[days]}).sort_index()
    ev["excess_vs_mkt_%"] = ev[f"{stock}_%"] - ev[f"mkt_ex_{stock}_%"]
    return ev


# ------------------------------------------------------------------ TABLES 3.2a / 3.2b: extreme events
def table_extremes(r):
    z = (r - r.mean()) / r.std()
    expected = len(r) * 2 * stats.norm.sf(K)                         # count expected if returns were Normal
    t = pd.DataFrame({f"n_below_-{K}sd": (z < -K).sum(), f"n_above_+{K}sd": (z > K).sum(),
                      "n_total": (z.abs() > K).sum(), "expected_if_normal": expected})
    t["ratio_to_normal"] = t["n_total"] / expected
    cnt = (z.abs() > K).sum(axis=1)                                  # how many assets are extreme on the same day
    top = pd.DataFrame({"n_assets_extreme": cnt, "avg_return_%": r.mean(axis=1)})
    top = top[top["n_assets_extreme"] > 0].sort_values("n_assets_extreme", ascending=False).head(15)
    return t, top


# ------------------------------------------------------------------ TABLE 3.3: serial dependence / volatility clustering
def table_serial(r, lb=10):
    band = 1.96 / np.sqrt(len(r))                                    # 95% band under iid
    rows = {}
    for c in r:
        a1 = acf(r[c], nlags=NLAGS, fft=True)[1:]
        a2 = acf(r[c] ** 2, nlags=NLAGS, fft=True)[1:]
        rows[c] = {"ACF1_ret": a1[0], f"sig_lags_ret(of {NLAGS})": int((abs(a1) > band).sum()),
                   f"LB{lb}_ret_p": acorr_ljungbox(r[c], lags=[lb])["lb_pvalue"].iloc[0],
                   "ACF1_ret2": a2[0], f"sig_lags_ret2(of {NLAGS})": int((abs(a2) > band).sum()),
                   f"LB{lb}_ret2_p": acorr_ljungbox(r[c] ** 2, lags=[lb])["lb_pvalue"].iloc[0],
                   f"ARCH_LM{lb}_p": het_arch(r[c], nlags=lb)[1]}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------ TABLES 3.4b / 3.4c: correlation summary and sector blocks
def corr_summary(C):
    n = len(C); iu = np.triu_indices(n, 1)
    vals = pd.Series(C.values[iu], index=[f"{C.index[i]} - {C.columns[j]}" for i, j in zip(*iu)])
    return pd.DataFrame({"value": [vals.mean(), vals.min(), vals.max()]},
                        index=["average pairwise corr", f"lowest: {vals.idxmin()}", f"highest: {vals.idxmax()}"]), vals


def sector_corr(C):
    """Average correlation within each sector (diagonal) and between sectors (off-diagonal)."""
    sec = pd.Series({n: SECTOR[n] for n in C.index}); secs = list(dict.fromkeys(sec))
    out = pd.DataFrame(index=secs, columns=secs, dtype=float)
    for a in secs:
        for b in secs:
            ia, ib = sec.index[sec == a], sec.index[sec == b]; sub = C.loc[ia, ib].values
            if a == b:
                out.loc[a, b] = sub[~np.eye(len(ia), dtype=bool)].mean() if len(ia) > 1 else np.nan   # one stock: no pair
            else:
                out.loc[a, b] = sub.mean()
    return out


# ------------------------------------------------------------------ TABLES 3.5a / 3.5b / 3.5c: stability over time (by calendar year)
def table_yearly(r):
    yr = r.index.year
    vol = r.groupby(yr).std() * np.sqrt(252)
    lev = {c: stats.levene(*[g.values for _, g in r[c].groupby(yr)], center="median").pvalue for c in r}
    corr_y = r.groupby(yr).apply(lambda d: d.corr().values[np.triu_indices(d.shape[1], 1)].mean())
    return vol, pd.Series(lev, name="Levene_equal_variance_p"), corr_y


# ------------------------------------------------------------------ FIGURE 3.9, TABLES 3.5f / 3.5g: rolling average pairwise correlation
def rolling_avg_corr(r, w=W):
    n = r.shape[1]
    rc = r.rolling(w).corr().dropna()
    return rc.groupby(level=0).apply(lambda m: (m.values.sum() - n) / (n * n - n))


def main():
    px, r = load_data()
    print(f"Sample: {r.index.min().date()} -> {r.index.max().date()}, {len(r)} daily log returns, {r.shape[1]} assets")

    # ================================================================== 3.1. DATA AND PRICE/RETURN DYNAMICS
    # Fig 3.1: prices rebased to 100 (log scale)
    reb = px / px.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(11, 5)); reb.plot(ax=ax, lw=.9, color=plt.cm.tab20.colors[:16])
    ax.set_title("Fig 3.1 - Prices rebased to 100 (CHF)"); ax.set_yscale("log"); ax.legend(ncol=8, fontsize=7); fig.tight_layout()
    save_fig(fig, "F3.1_prices")

    # Table 3.0: numbers behind Fig 3.1: final rebased value, cumulative and compound annual return (not the log-return mean of Table 3.1a)
    years = (px.index[-1] - px.index[0]).days / 365.25
    perf = pd.DataFrame({"final_rebased": reb.iloc[-1], "cum_return_%": reb.iloc[-1] - 100,
                         "ann_return_%": ((reb.iloc[-1] / 100) ** (1 / years) - 1) * 100,
                         # % price change in the COVID-19 crash and after the SNB decision
                         "chg_19feb_23mar2020_%":
                             (px.loc["2020-03-23"] / px.loc["2020-02-19"] - 1) * 100,
                         "chg_14_16jan2015_%":
                             (px.loc["2015-01-16"] / px.loc["2015-01-14"] - 1) * 100}
                        ).sort_values("final_rebased", ascending=False)
    print(f"\nPeriod: {px.index[0].date()} to {px.index[-1].date()} ({years:.2f} years)")
    save_table(perf, "T3.0_price_performance", 1)
    print("check (rebased prices = exp of summed log returns):", np.allclose(reb.iloc[-1], 100 * np.exp(r.sum() / 100)))
    
    # Averages quoted in Section 3.1 and drawdown of the equally weighted index
    print("Average price change (%): 19 Feb-23 Mar 2020 =",
          round(perf["chg_19feb_23mar2020_%"].mean(), 1),
          "| 14-16 Jan 2015 =", round(perf["chg_14_16jan2015_%"].mean(), 1))
    ew = np.exp(r.mean(axis=1).cumsum() / 100)    # index built from the average log return
    dd = ew / ew.cummax() - 1
    trough = dd.idxmin(); peak = ew[:trough].idxmax()
    print(f"Equally weighted index: max drawdown {100 * dd.min():.1f}% "
          f"from {peak.date()} to {trough.date()}")

    # Fig 3.2: daily log returns (volatility clustering visible)
    fig, axes = grid(r.shape[1])
    for ax, c in zip(axes, r): ax.plot(r.index, r[c], lw=.4); ax.set_title(c, fontsize=9)
    fig.suptitle("Fig 3.2 - Daily log returns (%)"); fig.tight_layout()
    save_fig(fig, "F3.2_returns")

    # ================================================================== 3.2. DISTRIBUTION, ASYMMETRY, NORMALITY
    # Table 3.1a: moments and normality tests
    t1a = table_moments(r)
    save_table(t1a, "T3.1a_moments_normality")

    # Table 3.1b / 3.1c: why Lonza (healthcare) is much more volatile than Roche and Novartis
    save_table(table_vol_decomposition(r), "T3.1b_vol_decomposition", 2)
    save_table(table_largest_moves(r, "Lonza", ("Roche", "Novartis")), "T3.1c_lonza_largest_moves", 2)

    # Fig 3.3: histograms with Normal overlay
    fig, axes = grid(r.shape[1], ncols=4, h=2.4, w=3.4)
    for ax, c in zip(axes, r):
        ax.hist(r[c].dropna(), bins=100, color="steelblue", 
                edgecolor="white", linewidth=0.2, density=True)
        mu, sd = r[c].mean(), r[c].std()
        x = np.linspace(r[c].min(), r[c].max(), 200)
        ax.plot(x, stats.norm.pdf(x, mu, sd), "r-", lw=1.0)
        ax.set_title(c, fontsize=9); ax.tick_params(labelsize=7)
    fig.suptitle("Fig 3.3 - Histograms of daily log returns (%) with Normal overlay", fontsize=11)
    fig.tight_layout(); save_fig(fig, "F3.3_histograms")

    # Fig 3.4: Normal QQ-plots (reference line = least-squares line of scipy probplot)
    fig, axes = grid(r.shape[1], h=2.8, w=3.6)
    for ax, c in zip(axes, r):
        stats.probplot(r[c], dist="norm", plot=ax); ax.set_title(c, fontsize=9); ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("Fig 3.4 - Normal QQ-plots of log returns"); fig.tight_layout()
    save_fig(fig, "F3.4_qqplots")

    # Fig 3.5: Student-t(3) QQ-plots (reference line through the quartiles); bridge to the choice of the density in Section 5
    q1_t, q3_t = stats.t.ppf([.25, .75], df=3)
    fig, axes = grid(r.shape[1], ncols=4, h=2.6, w=3.4)
    for ax, c in zip(axes, r):
        q_emp = np.sort(r[c].dropna().values); n = len(q_emp)
        q_th = stats.t.ppf((np.arange(1, n + 1) - 0.5) / n, df=3)     # standard plotting positions (i - 0.5)/n
        e1, e3 = np.percentile(q_emp, [25, 75])
        slope = (e3 - e1) / (q3_t - q1_t); intercept = e1 - slope * q1_t
        ax.scatter(q_th, q_emp, s=4, color="steelblue")
        xs = np.array([q_th.min(), q_th.max()]); ax.plot(xs, slope * xs + intercept, "r-", lw=1.0)
        ax.set_title(c, fontsize=9); ax.tick_params(labelsize=7)
    fig.suptitle("Fig 3.5 - Student-t (3 df) QQ-plots of daily log returns", fontsize=11)
    fig.tight_layout(); save_fig(fig, "F3.5_qqplots_t3")

    # ================================================================== 3.3. TAILS AND EXTREME EVENTS
    # Table 3.2a / 3.2b: extreme events and dates with most simultaneous extremes
    t2, top = table_extremes(r)
    save_table(t2, "T3.2a_extreme_events")
    save_table(top, "T3.2b_dates_with_most_extremes")

    # Table 3.2c: smallest and largest return of each stock in standard deviations (to read the QQ-plot tails)
    z = (r - r.mean()) / r.std()
    save_table(pd.DataFrame({"min_sigma": z.min(), "date_min": z.idxmin().dt.date,
                             "max_sigma": z.max(), "date_max": z.idxmax().dt.date}
                            ).sort_values("min_sigma"), "T3.2c_extreme_sigma", 1)

    # ================================================================== 3.4. SERIAL DEPENDENCE AND VOLATILITY CLUSTERING
    # Table 3.3: ACF / Ljung-Box / ARCH-LM
    t3 = table_serial(r)
    save_table(t3, "T3.3_serial_dependence_clustering")

    # Fig 3.6a / 3.6b: ACF of returns and of squared returns
    for tag, data, title in [("3.6a", r, "returns"), ("3.6b", r ** 2, "squared returns")]:
        fig, axes = grid(r.shape[1], h=2.4, w=3.8)
        for ax, c in zip(axes, r): plot_acf(data[c], lags=30, ax=ax, title=c, zero=False, auto_ylims=True, bartlett_confint=False)   # same iid band as Table 3.3
        fig.suptitle(f"Fig {tag} - ACF of {title}"); fig.tight_layout()
        save_fig(fig, f"F{tag}_acf_{title.replace(' ', '_')}")

    # ================================================================== 3.5. CORRELATION ACROSS ASSETS
    # Table 3.4a / 3.4b / 3.4c: correlation matrix, correlation summary and sector block correlation
    C = r.corr()
    save_table(C, "T3.4a_correlation_matrix", 2)
    cs, pairs = corr_summary(C)
    save_table(cs, "T3.4b_correlation_summary", 3)
    save_table(sector_corr(C), "T3.4c_sector_block_correlation", 2)

    # Table 3.4d: average correlation of each stock with the other 15 (its mean must give the pairwise mean of Table 3.4b)
    avg_corr = (C.sum() - 1) / (len(C) - 1)
    save_table(avg_corr.sort_values().to_frame("avg_corr_with_others"), "T3.4d_avg_corr_by_stock", 3)
    print("check: mean of stock averages =", round(avg_corr.mean(), 3), "| pairwise mean =", round(pairs.mean(), 3))

    # Fig 3.7: correlation heatmap
    fig, ax = plt.subplots(figsize=(11, 9)); sns.heatmap(C, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, annot_kws={"size": 6})
    cuts = [i for i in range(1, len(C)) if SECTOR[C.index[i]] != SECTOR[C.index[i - 1]]]   # lines between sector blocks
    for k in cuts: ax.axhline(k, color="k", lw=1.2); ax.axvline(k, color="k", lw=1.2)
    ax.set_title("Fig 3.7 - Correlation of daily log returns (full sample), stocks ordered by sector"); fig.tight_layout()
    save_fig(fig, "F3.7_corr_heatmap")

    # ================================================================== 3.6. STABILITY OVER TIME
    # Fig 3.8: rolling mean, variance, skewness, kurtosis (window W; pandas skew/kurt are bias-corrected, scipy ones in Table 3.1a are not)
    roll = {"Rolling mean (% daily)": r.rolling(W).mean(), "Rolling variance (%^2 daily)": r.rolling(W).var(),
            "Rolling skewness": r.rolling(W).skew(), "Rolling excess kurtosis": r.rolling(W).kurt()}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8)); axes = axes.ravel()
    for ax, (name, d) in zip(axes, roll.items()):
        ax.plot(d.index, d.values, lw=.6, alpha=.45); ax.plot(d.index, d.mean(axis=1), "k", lw=2, label="cross-sectional average")
        ax.set_title(name)
        if "mean" in name or "skew" in name: ax.axhline(0, color="grey", lw=.5)
    axes[0].legend(fontsize=8); fig.suptitle(f"Fig 3.8 - Rolling moments ({W}-day window)"); fig.tight_layout()
    save_fig(fig, "F3.8_rolling_moments")

    # Table 3.5d / 3.5e: numbers behind Fig 3.8 (extremes with stock name and date; yearly mean of the average line)
    rows = []
    for name, d in roll.items():
        m = d.mean(axis=1)                                           # black line of the figure
        s = d.unstack().dropna()                                     # index: (stock, date)
        (smax, dmax), (smin, dmin) = s.idxmax(), s.idxmin()
        rows.append({"moment": name,
                     "avg_max": m.max(), "avg_max_date": m.idxmax().date(),
                     "avg_min": m.min(), "avg_min_date": m.idxmin().date(),
                     "stock_max": smax, "stock_max_value": s.max(), "stock_max_date": dmax.date(),
                     "stock_min": smin, "stock_min_value": s.min(), "stock_min_date": dmin.date()})
    save_table(pd.DataFrame(rows).set_index("moment").T, "T3.5d_rolling_moments_extremes", 3)
    by_year = pd.DataFrame({n: d.mean(axis=1).groupby(d.index.year).mean() for n, d in roll.items()})
    by_year.index.name = "year"
    save_table(by_year, "T3.5e_rolling_moments_by_year", 3)

    # Fig 3.9: rolling correlation (average pairwise + most/least correlated pair)
    avg_roll = rolling_avg_corr(r)
    hi, lo = [(p.split(" - ")[0], p.split(" - ")[1]) for p in (pairs.idxmax(), pairs.idxmin())]
    pair_roll = {f"{a}-{b} ({lab} full-sample)": r[a].rolling(W).corr(r[b]) for (a, b), lab in [(hi, "highest"), (lo, "lowest")]}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(avg_roll.index, avg_roll, "k", lw=2, label="average pairwise correlation")
    for lab, rc in pair_roll.items(): ax.plot(rc.index, rc, lw=1, label=lab)
    ax.set_title(f"Fig 3.9 - Rolling correlation ({W}-day window)"); ax.legend(fontsize=8); fig.tight_layout()
    save_fig(fig, "F3.9_rolling_corr")

    # Table 3.5f / 3.5g: numbers behind Fig 3.9
    roll_corr = pd.DataFrame({"average pairwise": avg_roll, **pair_roll}).dropna()
    save_table(pd.DataFrame({"max": roll_corr.max(), "date_max": roll_corr.idxmax().dt.date,
                             "min": roll_corr.min(), "date_min": roll_corr.idxmin().dt.date}),
               "T3.5f_rolling_corr_extremes", 3)
    rcy = roll_corr.groupby(roll_corr.index.year).mean(); rcy.index.name = "year"
    save_table(rcy, "T3.5g_rolling_corr_by_year", 3)

    # Tables 3.5a / 3.5b / 3.5c: volatility, Levene test and average correlation by calendar year
    vol_y, lev, corr_y = table_yearly(r)
    save_table(vol_y, "T3.5a_annualised_vol_by_year", 2)
    save_table(lev.to_frame(), "T3.5b_levene_test_by_year")
    save_table(corr_y.to_frame("avg_pairwise_corr"), "T3.5c_avg_correlation_by_year", 3)
    save_table(pd.DataFrame({"avg_vol_%": vol_y.mean(axis=1), "min_vol_%": vol_y.min(axis=1),
                             "max_vol_%": vol_y.max(axis=1), "avg_pairwise_corr": corr_y}),
               "T3.5ac_vol_corr_summary_by_year", 2)

    plt.show()


if __name__ == "__main__":
    main()
