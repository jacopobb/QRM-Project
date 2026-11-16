"""Point 8 - comparison of the VaR sequences (Cases 1-5) with graphical tools and descriptive statistics.

No formal backtest (Kupiec, Christoffersen, ...) is performed here: the focus is on the number of days in which the return is below the predicted VaR (a "violation" or "hit").

INPUT CONTRACT (this script does not import garch_models.py or evt.py)
  A folder (default outputs/forecasts) with one CSV per series, file name = series name (e.g. UBS.csv, PORT.csv).
  Each CSV: first column = date; a column with the realised return (name in RET_NAMES, default "ret");
  and one column per VaR sequence named  VaR_<model>_<alpha>  (e.g. VaR_case1_0.05, VaR_case4_0.01).
  VaR is a return quantile (negative number); a hit is  return < VaR.
  Models and alpha levels are detected from the column names; any other column (mu, sigma_..., ES_...) is ignored.

Run:  python backtest.py
      python backtest.py --dir other/folder --out outputs --focus UBS PORT
      python backtest.py --check            # only shows what was detected
Outputs: <out>/tables/T8.*.csv and <out>/figures/F8.*.png

Subsections of Section 8, in the order in which they appear in the report
(when one table or figure is produced for each alpha, the letters a, b, ... follow the alphas from the largest):
  8.1  VaR sequences against the returns      F8.1a, F8.1b   returns and VaR of every model (focus series)
  8.2  Number of violations                   T8.1           descriptive statistics, every series x alpha x model
                                              T8.2a, T8.2b   violations, series x model
                                              T8.3           summary over all series
                                              F8.2           violations by model (focus series)
                                              F8.3           violations relative to the expected number (all series)
  8.3  Timing of the violations               F8.4           dates of the violations (focus series)
                                              F8.5           cumulative violations (focus series)
"""

import os, re, glob, argparse, string, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

# ------------------------------------------------------------------ editable settings
SAVE = True                                                   # True: write CSV and PNG files
RET_NAMES = ["ret", "return", "returns", "r", "y"]            # accepted names of the realised-return column
VAR_PATTERN = re.compile(r"^VaR_(.+)_(0?\.\d+)$")             # VaR_<model>_<alpha>
LABELS = {"case1": "C1 Gaussian GARCH", "case2": "C2 GARCH alt. density", "case3": "C3 GARCH+GEV (indep.)",
          "case4": "C4 GARCH+GEV (ext. index)", "case5": "C5 GARCH+GPD"}   # pretty names; unknown models use their key
DEFAULT_FOCUS = ["UBS", "PORT"]                               # series plotted in detail (falls back to first/last)
TABLE_DIR, FIG_DIR = "outputs/tables", "outputs/figures"      # overridden by --out


def save_table(df, name, digits=4):
    # prints the table rounded for reading, and saves it with 6 decimals so that the report can be checked against it
    print(f"\n=== {name} ===\n", df.round(digits).to_string())
    if SAVE:
        df.round(6).to_csv(f"{TABLE_DIR}/{name}.csv")


def save_fig(fig, name):
    if SAVE:
        fig.savefig(f"{FIG_DIR}/{name}.png", dpi=150)


# ------------------------------------------------------------------ loading and detection
def natural_key(s):
    """Sort key that orders case2 before case10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def label(m):
    return LABELS.get(m, m)


def var_column(d, m, a):
    """Name of the column VaR_<m>_<a> in d, whatever the number format of alpha (0.05, .05), or None."""
    name = f"VaR_{m}_{a:g}"
    if name in d:                            # usual case: the name written by garch_models.py / evt.py
        return name
    for c in d:                              # otherwise compare alpha as a number, not as a string
        mt = VAR_PATTERN.match(c)
        if mt and mt.group(1) == m and np.isclose(float(mt.group(2)), a):
            return c
    return None                              # this model has no VaR at this alpha for this series


def load_series(folder):
    """Return {series: DataFrame(ret, VaR_<model>_<alpha> ...)} and the detected (models, alphas)."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))            # one file per series (16 stocks + PORT)
    if not files:
        raise FileNotFoundError(f"no CSV files in {folder}")
    data, models, alphas = {}, set(), set()
    for f in files:
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        rc = next((c for c in d.columns if c.lower() in RET_NAMES), None)          # realised-return column
        vc = {c: VAR_PATTERN.match(c) for c in d.columns}
        vc = {c: m for c, m in vc.items() if m}                                    # keep only the VaR columns (mu, sigma, ES are dropped)
        if rc is None or not vc:
            print(f"  skipped {os.path.basename(f)} (no return / VaR columns)")
            continue
        data[os.path.splitext(os.path.basename(f))[0]] = d[[rc] + list(vc)].rename(columns={rc: "ret"})
        models |= {m.group(1) for m in vc.values()}
        alphas |= {float(m.group(2)) for m in vc.values()}
    # models in natural order (case1 ... case5), alphas from the largest (5% before 1%, as in the report)
    return data, sorted(models, key=natural_key), sorted(alphas, reverse=True)


# ------------------------------------------------------------------ descriptive statistics of the violations
def hit_stats(d, m, a):
    """Descriptive statistics of the VaR sequence of model m at level a (None if the sequence is missing)."""
    col = var_column(d, m, a)
    if col is None:
        return None
    x = d[["ret", col]].dropna()                                      # only days with both a return and a VaR forecast
    r, v = x["ret"], x[col]
    hits = r < v                                                      # violation: return below the predicted VaR
    T = len(x)                                                        # out-of-sample days
    excess = (r - v)[hits]                                            # by how much the return is below the VaR (< 0)
    # expected number of violations of a correct VaR = alpha * T (12.5 at 5%, 2.5 at 1%);
    # hits/expected = 1 means the hit rate equals alpha, > 1 too many violations, < 1 too few (conservative VaR)   
    return {"T": T, "expected_hits": a * T, "hits": int(hits.sum()), "hit_rate_%": 100 * hits.mean(),
            "hits/expected": hits.sum() / (a * T),
            # a hit on day t that follows a hit on day t-1: descriptive signal of violations clustered in time
            "consecutive_hits": int((hits & hits.shift(1, fill_value=False)).sum()),   # hits that follow a hit
            "avg_VaR_%": v.mean(),
            # size of the violations: average and worst distance of the return below the VaR (NaN if no violation)
            "avg_excess_%": excess.mean() if hits.any() else np.nan,
            "worst_excess_%": excess.min() if hits.any() else np.nan}


# ------------------------------------------------------------------ main
def main():
    global TABLE_DIR, FIG_DIR
    # command-line options: input folder, output folder, focus series and a check-only mode
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/forecasts")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--focus", nargs="*", default=DEFAULT_FOCUS)
    ap.add_argument("--check", action="store_true")
    ar = ap.parse_args()
    TABLE_DIR, FIG_DIR = f"{ar.out}/tables", f"{ar.out}/figures"
    os.makedirs(TABLE_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

    # load every forecast file and report what was found (series, models, alphas, out-of-sample period)
    data, models, alphas = load_series(ar.dir)
    print(f"Series ({len(data)}): {list(data)}\nModels detected: {[label(m) for m in models]}\nAlpha levels: {alphas}")
    d0 = next(iter(data.values()))             # all series share the same out-of-sample dates
    print(f"OOS sample: {d0.index.min().date()} -> {d0.index.max().date()}, {len(d0)} days")
    if ar.check:
        return
    # focus series (UBS and PORT); if none of them exists, the first and the last series are used
    focus = [s for s in ar.focus if s in data] or [list(data)[0], list(data)[-1]]
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]                    # same colour for the same model in every figure
    letter = {a: string.ascii_lowercase[i] for i, a in enumerate(alphas)}       # a, b, ... for each alpha

    # ================================================================== 8.1. VaR SEQUENCES AGAINST THE RETURNS
    # Fig 8.1a, 8.1b: returns and VaR sequences of every model, focus series, one figure per alpha
    for a in alphas:
        fig, axes = plt.subplots(len(focus), 1, figsize=(13, 4.2 * len(focus)), squeeze=False)
        for ax, s in zip(axes.ravel(), focus):
            d = data[s]
            ax.plot(d["ret"], color="grey", lw=.6, label="return")
            for i, m in enumerate(models):
                c = var_column(d, m, a)
                if c:
                    # the number of violations of each model is written in the legend
                    ax.plot(d[c], lw=1, color=cols[i % 10], label=f"{label(m)} ({int((d['ret'] < d[c]).sum())} hits)")
            ax.set_title(f"{s} - VaR {a:.0%}, expected hits {a * len(d):.1f}")
            ax.legend(fontsize=7, ncol=3)
        fig.suptitle(f"Fig 8.1{letter[a]} - Returns and VaR sequences, alpha = {a:g}")
        fig.tight_layout(); save_fig(fig, f"F8.1{letter[a]}_var_lines_alpha_{a:g}")

    # ================================================================== 8.2. NUMBER OF VIOLATIONS
    # Table 8.1: descriptive statistics of the violations, every series x alpha x model
    rows = []
    for s, d in data.items():
        for a in alphas:
            for m in models:
                e = hit_stats(d, m, a)
                if e:
                    rows.append({"series": s, "alpha": a, "model": label(m), **e})
    full = pd.DataFrame(rows).set_index(["series", "alpha", "model"])
    save_table(full, "T8.1_backtest_all")

    # Table 8.2a, 8.2b: number of violations, series x model, one table per alpha
    present = [label(m) for m in models if label(m) in full.index.get_level_values("model")]    # column order C1 ... C5
    for a in alphas:
        fa = full.xs(a, level="alpha")
        w = fa["hits"].unstack("model")[present]                                         # rows = series, columns = models
        w.insert(0, "expected", fa["expected_hits"].groupby("series").first())           # alpha * T, first column for reference
        save_table(w, f"T8.2{letter[a]}_hits_alpha_{a:g}", 1)

    # Table 8.3: summary over all series, per alpha and model
    # totals (hits, expected, T) are sums over the 17 series; avg_VaR and avg_excess are simple averages over series;
    # n_series_above_expected counts the series with more violations than alpha * T
    g = full.reset_index()
    g["above_expected"] = g["hits"] > g["expected_hits"]
    summ = g.groupby(["alpha", "model"]).agg(
        total_hits=("hits", "sum"), expected_total=("expected_hits", "sum"), total_T=("T", "sum"),
        n_series_above_expected=("above_expected", "sum"), n_series=("hits", "size"),
        avg_VaR=("avg_VaR_%", "mean"), avg_excess=("avg_excess_%", "mean"))
    summ["hits/expected"] = summ["total_hits"] / summ["expected_total"]
    summ["hit_rate_%"] = 100 * summ["total_hits"] / summ["total_T"]
    save_table(summ.sort_index(ascending=[False, True]), "T8.3_summary_all_series", 3)      # 5% first, then 1%

    # Fig 8.2: number of violations by model, focus series, with the expected number
    fig, axes = plt.subplots(len(alphas), 1, figsize=(12, 3.8 * len(alphas)), squeeze=False)
    for ax, a in zip(axes.ravel(), alphas):
        x = np.arange(len(models)); wdt = .8 / len(focus)              # one group of bars per model, one bar per focus series
        for j, s in enumerate(focus):
            hv = [full.loc[(s, a, label(m)), "hits"] if (s, a, label(m)) in full.index else np.nan for m in models]
            ax.bar(x + j * wdt, hv, wdt, label=s)
        ax.axhline(a * len(d0), color="k", ls="--", label="expected")
        ax.set_xticks(x + .4 - wdt / 2); ax.set_xticklabels([label(m) for m in models], fontsize=8)      # ticks at the centre of each group
        ax.set_title(f"alpha = {a:g}"); ax.legend(fontsize=8)
    fig.suptitle("Fig 8.2 - Number of VaR violations by model")
    fig.tight_layout(); save_fig(fig, "F8.2_hits_bar")

    # Fig 8.3: violations relative to the expected number, all series (1 = the hit rate equals alpha)
    fig, axes = plt.subplots(len(alphas), 1, figsize=(12, 3.8 * len(alphas)), squeeze=False)
    for ax, a in zip(axes.ravel(), alphas):
        ratio = full.xs(a, level="alpha")["hits/expected"].unstack("model")[present]           # rows = series, columns = models
        ratio.plot(kind="bar", ax=ax, width=.85)
        ax.axhline(1, color="k", ls="--"); ax.set_ylabel("hits / expected")                    # bars above 1: too many violations
        ax.set_title(f"alpha = {a:g}"); ax.legend(fontsize=6, ncol=3); ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Fig 8.3 - Violations relative to the expected number, all series (1 = hit rate equal to alpha)")
    fig.tight_layout(); save_fig(fig, "F8.3_hits_ratio_all")

    # ================================================================== 8.3. TIMING OF THE VIOLATIONS
    # Fig 8.4: dates of the violations, focus series (clusters of dates = violations concentrated in stress periods)
    fig, axes = plt.subplots(len(alphas), 1, figsize=(12, 3.2 * len(alphas)), squeeze=False)
    for ax, a in zip(axes.ravel(), alphas):
        for i, m in enumerate(models):
            for k, s in enumerate(focus):
                d = data[s]; c = var_column(d, m, a)
                if c:
                    h = d.index[d["ret"] < d[c]]                # dates of the violations
                    # one row per (model, series) pair: a vertical tick on each violation date
                    ax.plot(h, np.full(len(h), i * len(focus) + k), "|", ms=14, color=cols[i % 10])
        ax.set_yticks([i * len(focus) + k for i in range(len(models)) for k in range(len(focus))])
        ax.set_yticklabels([f"{label(m)[:14]} {s}" for m in models for s in focus], fontsize=6)
        ax.set_title(f"alpha = {a:g}")
    fig.suptitle("Fig 8.4 - Dates of the VaR violations")
    fig.tight_layout(); save_fig(fig, "F8.4_hit_timing")

    # Fig 8.5: cumulative number of violations against the cumulative expected number, focus series
    # a line below the dashed one = fewer violations than expected so far; steep steps = groups of violations
    fig, axes = plt.subplots(len(alphas), len(focus), figsize=(6.5 * len(focus), 3.8 * len(alphas)), squeeze=False)
    for i, a in enumerate(alphas):
        for j, s in enumerate(focus):
            ax = axes[i, j]; d = data[s]
            ax.plot(d.index, a * np.arange(1, len(d) + 1), "k--", label="expected")       # alpha * t after t days
            for k, m in enumerate(models):
                c = var_column(d, m, a)
                if c:
                    ax.plot(d.index, (d["ret"] < d[c]).cumsum(), color=cols[k % 10], lw=1.2, label=label(m))
            ax.set_title(f"{s} - alpha = {a:g}"); ax.legend(fontsize=6)
    fig.suptitle("Fig 8.5 - Cumulative number of VaR violations")
    fig.tight_layout(); save_fig(fig, "F8.5_cumulative_hits")

    plt.show()


if __name__ == "__main__":
    main()
