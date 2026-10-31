"""Point 8: backtesting and comparison of ALL the VaR sequences, whatever produced them.

INPUT CONTRACT (the only thing this script relies on - it does not import garch_models.py or evt.py)
  A folder (default outputs/forecasts) with ONE CSV PER SERIES, file name = series name (e.g. UBS.csv, PORT.csv).
  Each CSV: first column = date (index); a column with the realised return (default name "ret", see RET_NAMES);
  and one column per VaR sequence named   VaR_<model>_<alpha>   e.g. VaR_case1_0.05, VaR_case4_0.01, VaR_gpd_0.01.
  VaR is a RETURN quantile (negative number); a "hit" is  return < VaR.
  Models and alpha levels are DETECTED from the column names, so adding/removing a case or a level needs no code change.
  Other columns (mu, sigma_..., ES_...) are ignored here.

Run:   python backtest.py                                   # defaults
       python backtest.py --dir other/folder --focus UBS PORT --out outputs
       python backtest.py --check                           # only show what was detected
Outputs: <out>/tables/B*.csv, <out>/figures/B*.png
"""
import os, re, glob, argparse, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import xlogy, xlog1py
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

# ------------------------------------------------------------------ editable settings
RET_NAMES = ["ret", "return", "returns", "r", "y"]                  # accepted names of the realised-return column
VAR_PATTERN = re.compile(r"^VaR_(.+)_(0?\.\d+)$")                  # VaR_<model>_<alpha>
LABELS = {"case1": "C1 Gaussian GARCH", "case2": "C2 GARCH alt. density", "case3": "C3 GARCH+GEV (indep.)",
          "case4": "C4 GARCH+GEV (ext. index)", "case5": "C5 GARCH+GPD"}   # pretty names; unknown models use their key
SIGNIF = 0.05                                                      # significance level of the tests
DEFAULT_FOCUS = ["UBS", "PORT"]                                    # series plotted in detail (falls back to first/last found)

# ------------------------------------------------------------------ loading and detection
def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def load_series(folder):
    """Return {series: DataFrame(ret, VaR_<model>_<alpha> ...)} and the detected (models, alphas)."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files: raise FileNotFoundError(f"no CSV files in {folder}")
    data, models, alphas = {}, set(), set()
    for f in files:
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        rc = next((c for c in d.columns if c.lower() in RET_NAMES), None)
        vc = {c: VAR_PATTERN.match(c) for c in d.columns}; vc = {c: m for c, m in vc.items() if m}
        if rc is None or not vc: print(f"  skipped {os.path.basename(f)} (no return / VaR columns)"); continue
        d = d[[rc] + list(vc)].rename(columns={rc: "ret"})
        data[os.path.splitext(os.path.basename(f))[0]] = d
        models |= {m.group(1) for m in vc.values()}; alphas |= {float(m.group(2)) for m in vc.values()}
    return data, sorted(models, key=natural_key), sorted(alphas, reverse=True)

def label(m): return LABELS.get(m, m)

# ------------------------------------------------------------------ tests
def kupiec(hits, p):
    """Unconditional coverage (Kupiec POF). H0: hit probability = p."""
    T, x = len(hits), int(hits.sum()); ph = x / T
    ll0 = xlogy(T - x, 1 - p) + xlogy(x, p); ll1 = xlogy(T - x, 1 - ph) + xlogy(x, ph)
    lr = -2 * (ll0 - ll1); return lr, stats.chi2.sf(lr, 1)

def christoffersen(hits):
    """Independence of hits (first-order Markov). H0: P(hit|hit yesterday) = P(hit|no hit yesterday)."""
    h = hits.astype(int).values; a, b = h[:-1], h[1:]
    n00, n01 = ((a == 0) & (b == 0)).sum(), ((a == 0) & (b == 1)).sum()
    n10, n11 = ((a == 1) & (b == 0)).sum(), ((a == 1) & (b == 1)).sum()
    pi01 = n01 / max(n00 + n01, 1); pi11 = n11 / max(n10 + n11, 1); pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll0 = xlog1py(n00 + n10, -pi) + xlogy(n01 + n11, pi)
    ll1 = xlog1py(n00, -pi01) + xlogy(n01, pi01) + xlog1py(n10, -pi11) + xlogy(n11, pi11)
    lr = max(-2 * (ll0 - ll1), 0); return lr, stats.chi2.sf(lr, 1), int(n11)

def pinball(ret, var, p):
    """Quantile (tick) loss of the VaR forecast as the p-quantile of the return; lower = better."""
    u = ret - var; return float(np.mean((p - (u < 0)) * u))

def evaluate(d, m, a):
    col = f"VaR_{m}_{a:g}"
    col = col if col in d else next((c for c in d if (mt := VAR_PATTERN.match(c)) and mt.group(1) == m and np.isclose(float(mt.group(2)), a)), None)
    if col is None: return None
    x = d[["ret", col]].dropna(); r, v = x["ret"], x[col]; hits = r < v; T = len(x)
    lu, pu = kupiec(hits, a); li, pi_, n11 = christoffersen(hits); lcc = lu + li
    return {"T": T, "expected_hits": a * T, "hits": int(hits.sum()), "hit_rate_%": 100 * hits.mean(),
            "Kupiec_p": pu, "Christ_ind_p": pi_, "CC_p": stats.chi2.sf(lcc, 2), "consecutive_hits": n11,
            "avg_VaR_%": v.mean(), "avg_excess_%": (r - v)[hits].mean() if hits.any() else np.nan,
            "worst_excess_%": (r - v)[hits].min() if hits.any() else np.nan, "pinball_loss": pinball(r, v, a)}

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/forecasts"); ap.add_argument("--out", default="outputs")
    ap.add_argument("--focus", nargs="*", default=DEFAULT_FOCUS); ap.add_argument("--check", action="store_true")
    ar = ap.parse_args()
    TAB, FIG = f"{ar.out}/tables", f"{ar.out}/figures"; os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    data, models, alphas = load_series(ar.dir)
    print(f"Series ({len(data)}): {list(data)}\nModels detected: {[label(m) for m in models]}\nAlpha levels: {alphas}")
    d0 = next(iter(data.values())); print(f"OOS sample: {d0.index.min().date()} -> {d0.index.max().date()}, {len(d0)} days")
    if ar.check: return
    def save(df, name, dg=4):
        print(f"\n=== {name} ===\n", df.round(dg).to_string()); df.round(6).to_csv(f"{TAB}/{name}.csv")

    # ---------- B1: full backtest table, every series x alpha x model
    rows = []
    for s, d in data.items():
        for a in alphas:
            for m in models:
                e = evaluate(d, m, a)
                if e: rows.append({"series": s, "alpha": a, "model": label(m), **e})
    full = pd.DataFrame(rows).set_index(["series", "alpha", "model"]); save(full, "B1_backtest_all")

    # ---------- B2: hit counts, wide (series x model), per alpha - the table asked in point 8
    for a in alphas:
        w = full.xs(a, level="alpha")["hits"].unstack("model")[[label(m) for m in models if label(m) in full.index.get_level_values("model")]]
        w.insert(0, "expected", a * full["T"].iloc[0]); save(w, f"B2_hits_alpha_{a:g}", 0)

    # ---------- B3: summary over all series: total hits and number of rejections of each test
    g = full.reset_index()
    summ = g.groupby(["alpha", "model"]).agg(total_hits=("hits", "sum"), expected_total=("expected_hits", "sum"),
            avg_pinball=("pinball_loss", "mean"), avg_VaR=("avg_VaR_%", "mean"),
            rej_Kupiec=("Kupiec_p", lambda x: int((x < SIGNIF).sum())), rej_Indep=("Christ_ind_p", lambda x: int((x < SIGNIF).sum())),
            rej_CC=("CC_p", lambda x: int((x < SIGNIF).sum())), n_series=("hits", "size"))
    summ["hit_rate_%"] = 100 * summ["total_hits"] / (summ["expected_total"] / summ.index.get_level_values("alpha"))
    save(summ.sort_index(ascending=[False, True]), "B3_summary_all_series", 3)

    # ---------- Figures for the focus series
    focus = [s for s in ar.focus if s in data] or [list(data)[0], list(data)[-1]]
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for a in alphas:                                   # B1 figure: returns, VaR lines of every model, hits marked for each model
        fig, axes = plt.subplots(len(focus), 1, figsize=(13, 4.2 * len(focus)), squeeze=False)
        for ax, s in zip(axes.ravel(), focus):
            d = data[s]; ax.plot(d["ret"], color="grey", lw=.6, label="return")
            for i, m in enumerate(models):
                c = f"VaR_{m}_{a:g}"
                if c in d: ax.plot(d[c], lw=1, color=cols[i % 10], label=f"{label(m)} ({int((d['ret'] < d[c]).sum())} hits)")
            ax.set_title(f"{s} - VaR {a:.0%}, expected hits {a * len(d):.1f}"); ax.legend(fontsize=7, ncol=3)
        fig.suptitle(f"Fig B1 - Returns and VaR sequences, alpha = {a:g}"); fig.tight_layout(); fig.savefig(f"{FIG}/B1_var_lines_alpha_{a:g}.png", dpi=150)
    fig, axes = plt.subplots(len(alphas), 1, figsize=(12, 3.8 * len(alphas)), squeeze=False)   # B2: hits per model, focus series
    for ax, a in zip(axes.ravel(), alphas):
        x = np.arange(len(models)); wdt = .8 / len(focus)
        for j, s in enumerate(focus):
            hv = [full.loc[(s, a, label(m)), "hits"] if (s, a, label(m)) in full.index else np.nan for m in models]
            ax.bar(x + j * wdt, hv, wdt, label=s)
        ax.axhline(a * len(d0), color="k", ls="--", label="expected"); ax.set_xticks(x + .4 - wdt / 2); ax.set_xticklabels([label(m) for m in models], fontsize=8)
        ax.set_title(f"alpha = {a:g}"); ax.legend(fontsize=8)
    fig.suptitle("Fig B2 - Number of VaR violations by model"); fig.tight_layout(); fig.savefig(f"{FIG}/B2_hits_bar.png", dpi=150)
    fig, axes = plt.subplots(len(alphas), 1, figsize=(12, 3.8 * len(alphas)), squeeze=False)   # B3: all series, hits vs expected
    for ax, a in zip(axes.ravel(), alphas):
        t = full.xs(a, level="alpha")["hits"].unstack("model"); t.columns = [c for c in t.columns]
        (t / (a * len(d0))).plot(kind="bar", ax=ax, width=.85); ax.axhline(1, color="k", ls="--"); ax.set_ylabel("hits / expected")
        ax.set_title(f"alpha = {a:g}"); ax.legend(fontsize=6, ncol=3); ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Fig B3 - Violations relative to expected, all series (1 = perfect coverage)"); fig.tight_layout(); fig.savefig(f"{FIG}/B3_hits_ratio_all.png", dpi=150)
    fig, axes = plt.subplots(len(alphas), 1, figsize=(12, 3.2 * len(alphas)), squeeze=False)   # B4: timing of hits (clustering?)
    for ax, a in zip(axes.ravel(), alphas):
        for i, m in enumerate(models):
            c = f"VaR_{m}_{a:g}"
            for k, s in enumerate(focus):
                d = data[s]
                if c in d: h = d.index[d["ret"] < d[c]]; ax.plot(h, np.full(len(h), i * len(focus) + k), "|", ms=14, color=cols[i % 10])
        ax.set_yticks([i * len(focus) + k for i in range(len(models)) for k in range(len(focus))])
        ax.set_yticklabels([f"{label(m)[:14]} {s}" for m in models for s in focus], fontsize=6); ax.set_title(f"alpha = {a:g}")
    fig.suptitle("Fig B4 - Dates of VaR violations (clustering = failed independence)"); fig.tight_layout(); fig.savefig(f"{FIG}/B4_hit_timing.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
