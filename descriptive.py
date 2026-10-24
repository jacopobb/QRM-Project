"""Point 2 - descriptive analysis on the FULL sample (log returns, in %).
Run:  python descriptive.py   -> prints every table and saves them in outputs/tables, figures in outputs/figures."""
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
TABLE_DIR, FIG_DIR = "outputs/tables", "outputs/figures"
os.makedirs(TABLE_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
W = 250            # rolling window (about 1 trading year)
NLAGS = 20         # lags for ACF significance counts
K = 3              # extreme event = |standardised return| > K

def save_table(df, name, digits=4):
    print(f"\n=== {name} ===\n", df.round(digits).to_string())
    df.round(6).to_csv(f"{TABLE_DIR}/{name}.csv")

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

# ------------------------------------------------------------------ TABLE 1: moments and normality
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

# ------------------------------------------------------------------ TABLE 2: serial dependence / volatility clustering
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

# ------------------------------------------------------------------ TABLE 3: extreme events
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

# ------------------------------------------------------------------ TABLE 4: correlation
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

# ------------------------------------------------------------------ TABLE 5: stability over time (by calendar year)
def table_yearly(r):
    yr = r.index.year
    vol = r.groupby(yr).std() * np.sqrt(252)
    lev = {c: stats.levene(*[g.values for _, g in r[c].groupby(yr)], center="median").pvalue for c in r}
    return vol, pd.Series(lev, name="Levene_equal_variance_p"), r.groupby(yr).apply(lambda d: d.corr().values[np.triu_indices(d.shape[1], 1)].mean())

def rolling_avg_corr(r, w=W):
    n = r.shape[1]
    rc = r.rolling(w).corr().dropna()
    return rc.groupby(level=0).apply(lambda m: (m.values.sum() - n) / (n * n - n))

def main():
    px, r = load_data()
    print(f"Sample: {r.index.min().date()} -> {r.index.max().date()}, {len(r)} daily log returns, {r.shape[1]} assets")

    # ---------- TABLES
    t1 = table_moments(r);                       save_table(t1, "T1_moments_normality")
    t2 = table_serial(r);                        save_table(t2, "T2_serial_dependence_clustering")
    t3, top = table_extremes(r);                 save_table(t3, "T3_extreme_events"); save_table(top, "T3b_dates_with_most_extremes")
    C = r.corr();                                save_table(C, "T4_correlation_matrix", 2)
    cs, pairs = corr_summary(C);                 save_table(cs, "T4b_correlation_summary", 3)
    save_table(sector_corr(C), "T4c_sector_block_correlation", 2)
    vol_y, lev, corr_y = table_yearly(r)
    save_table(vol_y, "T5_annualised_vol_by_year", 2); save_table(lev.to_frame(), "T5b_levene_test_by_year")
    save_table(corr_y.to_frame("avg_pairwise_corr"), "T5c_avg_correlation_by_year", 3)

    # ---------- FIGURES
    # Fig 1: prices rebased to 100 (visual only)
    fig, ax = plt.subplots(figsize=(11, 5)); (px / px.iloc[0] * 100).plot(ax=ax, lw=.9)
    ax.set_title("Fig 1 - Prices rebased to 100 (CHF)"); ax.set_yscale("log"); ax.legend(ncol=8, fontsize=7); fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/F1_prices.png", dpi=150)

    # Fig 2: daily log returns (volatility clustering visible)
    fig, axes = grid(r.shape[1])
    for ax, c in zip(axes, r): ax.plot(r.index, r[c], lw=.4); ax.set_title(c, fontsize=9)
    fig.suptitle("Fig 2 - Daily log returns (%)"); fig.tight_layout(); fig.savefig(f"{FIG_DIR}/F2_returns.png", dpi=150)

    # Fig 3: normal QQ-plots
    fig, axes = grid(r.shape[1], h=2.8, w=3.6)
    for ax, c in zip(axes, r):
        stats.probplot(r[c], dist="norm", plot=ax); ax.set_title(c, fontsize=9); ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("Fig 3 - Normal QQ-plots of log returns"); fig.tight_layout(); fig.savefig(f"{FIG_DIR}/F3_qqplots.png", dpi=150)

    # Fig 4a/4b: ACF of returns and of squared returns
    for tag, data, title in [("4a", r, "returns"), ("4b", r ** 2, "squared returns")]:
        fig, axes = grid(r.shape[1], h=2.4, w=3.8)
        for ax, c in zip(axes, r): plot_acf(data[c], lags=30, ax=ax, title=c, zero=False, auto_ylims=True)
        fig.suptitle(f"Fig {tag} - ACF of {title}"); fig.tight_layout(); fig.savefig(f"{FIG_DIR}/F{tag}_acf_{title.replace(' ', '_')}.png", dpi=150)

    # Fig 5: correlation heatmap
    fig, ax = plt.subplots(figsize=(11, 9)); sns.heatmap(C, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, annot_kws={"size": 6})
    cuts = [i for i in range(1, len(C)) if SECTOR[C.index[i]] != SECTOR[C.index[i - 1]]]   # lines between sector blocks
    for k in cuts: ax.axhline(k, color="k", lw=1.2); ax.axvline(k, color="k", lw=1.2)
    ax.set_title("Fig 5 - Correlation of daily log returns (full sample), stocks ordered by sector"); fig.tight_layout(); fig.savefig(f"{FIG_DIR}/F5_corr_heatmap.png", dpi=150)

    # Fig 6: rolling mean, variance, skewness, kurtosis (window W)
    roll = {"Rolling mean (% daily)": r.rolling(W).mean(), "Rolling variance (%^2 daily)": r.rolling(W).var(),
            "Rolling skewness": r.rolling(W).skew(), "Rolling excess kurtosis": r.rolling(W).kurt()}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8)); axes = axes.ravel()
    for ax, (name, d) in zip(axes, roll.items()):
        ax.plot(d.index, d.values, lw=.6, alpha=.45); ax.plot(d.index, d.mean(axis=1), "k", lw=2, label="cross-sectional average")
        ax.set_title(name); ax.axhline(0 if "mean" in name or "skew" in name else np.nan, color="grey", lw=.5)
    axes[0].legend(fontsize=8); fig.suptitle(f"Fig 6 - Rolling moments ({W}-day window)"); fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/F6_rolling_moments.png", dpi=150)

    # Fig 7: rolling correlation (average pairwise + most/least correlated pair)
    avg = rolling_avg_corr(r)
    hi, lo = [(p.split(" - ")[0], p.split(" - ")[1]) for p in (pairs.idxmax(), pairs.idxmin())]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(avg.index, avg, "k", lw=2, label="average pairwise correlation")
    for (a, b), lab in [(hi, "highest"), (lo, "lowest")]:
        rc = r[a].rolling(W).corr(r[b]); ax.plot(rc.index, rc, lw=1, label=f"{a}-{b} ({lab} full-sample)")
    ax.set_title(f"Fig 7 - Rolling correlation ({W}-day window)"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/F7_rolling_corr.png", dpi=150)

    plt.show()

if __name__ == "__main__":
    main()
