# Homework 2 – VaR / ES backtesting (Group 9, Swiss large caps)

Everything for Homework 2 lives in this folder. **No Homework 1 file is modified**: the scripts here only *read*
`../data/returns.csv` (the HW1 log returns, in %) and write their results in `homework2/outputs/`.

Run every script from inside this folder:

```bash
cd homework2
python var_iid.py          # M01-M05, M20  (all series; or e.g. `python var_iid.py EWP SMI`)
python var_evt.py          # M06-M08
```

## Settings
All choices are in `settings_hw2.py` (backtest sample 1/10/2021–30/09/2026, 5-year rolling window = 1260 days,
alphas, GEV block = 21 days, GPD threshold = 90% loss quantile, EWMA lambda = 0.94, refit frequency, seed).
Do not change them between scripts, so that tables and figures stay reproducible.

## Data still to add (in `homework2/data/`)
| File | Content | Source |
|---|---|---|
| `index_prices.csv` | `Date,Close` daily closes of the market index (SMI, RIC `.SSMI`), 26/09/2011–30/09/2026 | LSEG Workspace |
| `Europe_5_Factors_Daily.csv` | Fama-French Europe 5 factors, daily (models 16–21) | Kenneth French data library |

Until `index_prices.csv` exists, the index is skipped with a warning; the other series run normally.

## Series
16 stocks (company names as in HW1) + `EWP` (simple average of the 16 log returns, as in HW1) + `SMI`.

## Output format
`outputs/var/<script>/<series>.csv`: one column per model and level, named `<code>_<alpha>`
(e.g. `M04_GH_0.01`). VaR is a **return quantile** (negative number, %): a violation on day *t* is `ret_t < VaR_t`.
`common.load_var(series)` merges the files of all scripts and adds the realised return `ret`, so the backtesting
code never has to re-estimate the models.
`outputs/params/<script>/<series>.csv`: estimated parameters day by day (useful for the comments on the VaR dynamics).

## Models (codes as in `settings_hw2.MODELS`)
| Code | Model | Window | Script | Status |
|---|---|---|---|---|
| M01 | Historical Simulation | rolling 5y | var_iid.py | done |
| M02–M04 | i.i.d. Normal / Student-t / Generalized Hyperbolic (MLE) | rolling 5y | var_iid.py | done |
| M05 | Cornish-Fisher | rolling 5y | var_iid.py | done |
| M06–M07 | GEV on monthly block maxima, without / with extremal index | expanding | var_evt.py | done |
| M08 | GPD over the 90% loss quantile | expanding | var_evt.py | done |
| M09–M15 | ARMA-GARCH + N / t / HS / CF / GEV / GEV-θ / GPD innovations | rolling 5y | var_garch.py | to do |
| M16–M19 | Factor mapping + CCC / DCC, Gaussian and FHS | rolling 5y | var_factors.py | to do |
| M20 | EWMA mean and variance, Gaussian | rolling 5y | var_iid.py | done |
| M21 | ARMA-GARCH Skew-t on X = Σ β·factors | rolling 5y | var_garch.py | to do |

## Implementation notes (to be reported/justified in the report)
- t and GH are re-estimated every day (`REFIT = 1`); each estimate starts from the previous day's parameters
  (warm start). On our data this gives the same log-likelihood as `scipy.stats.*.fit` and is ~3x faster.
  `REFIT = 5` gives VaR within ~0.1 p.p. and runs 4x faster, useful for quick tests only.
- EWMA: the recursion is run once on the full sample; with λ = 0.94 the weight of the starting value after
  1260 days is ~1e-34, so this equals restarting it on every 5-year window.
- The extremal index uses the blocks estimator with threshold at the 95% loss quantile, clipped to (0, 1].
- Runtime (one core): var_iid ≈ 4.5 min per series, var_evt ≈ 15 s per series. Series run in parallel
  (`N_JOBS` = cores − 1).
