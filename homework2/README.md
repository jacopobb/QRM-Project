# Homework 2 – VaR / ES backtesting (Group 9, Swiss large caps)

Everything for Homework 2 lives in this folder. **No Homework 1 file is modified**: the scripts only *read*
`../data/returns.csv` (HW1 log returns, in %) and write their results in `homework2/outputs/`.
Only numpy, pandas, scipy and matplotlib are needed (the GARCH, DCC and skewed-t are written in `garch_core.py`
and `var_factors.py`, so every formula is visible and no `arch`/`statsmodels` version issue can change the numbers).

```bash
cd homework2
python main_hw2.py            # whole pipeline (about 20-30 min on a laptop); python main_hw2.py --list for the steps
python main_hw2.py --from backtest   # re-run only the evaluation once the VaR forecasts exist
```

## 1. Data to add before running (folder `homework2/data/`)
| File | Content | Source | Who |
|---|---|---|---|
| `index_prices.csv` | two columns `Date,Close`: daily closes of the **SMI** (RIC `.SSMI`), 26/09/2011–30/09/2026 | LSEG Workspace | A |
| `Europe_5_Factors_Daily.csv` | Fama-French **Europe 5 Factors [Daily]**, csv as downloaded (unzipped), not edited | [Kenneth French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) → *Developed Market Factors and Returns* | C |

Without the index file the index is skipped (warning); without the factor file the step `factors` stops with a message.

## 2. Settings (`settings_hw2.py`)
Backtest 1/10/2021–30/09/2026; rolling window 1260 days (5y); alpha 1% and 5%; GEV block 21 days (1 month);
GPD threshold 90% loss quantile; extremal index threshold 95%; EWMA lambda 0.94; parametric models re-estimated every
day (`REFIT = 1`); seed 42 for every simulation/bootstrap. Do not change them between steps.

## 3. Scripts → homework points → owner
| Script | Point | Content | Main outputs | Owner |
|---|---|---|---|---|
| `common.py`, `settings_hw2.py` | – | shared data module: returns, EWP, SMI, backtest window, seed, output format | – | A (settings agreed by all) |
| `var_iid.py` | 1 | M01 HS, M02 Normal, M03 t, M04 GH, M05 Cornish-Fisher (rolling 5y) | `outputs/var/iid/` | A |
| `var_evt.py` | 1 | M06 GEV, M07 GEV + extremal index, M08 GPD (expanding) | `outputs/var/evt/` | A |
| `garch_core.py` | – | AR(1)-GJR-GARCH(1,1) engine (normal, t, skewed-t), FHS simulation | – | B |
| `var_garch.py` | 1 | M09–M15: AR(1)-GJR-GARCH(1,1) + N / t / HS(FHS) / CF / GEV / GEV-θ / GPD quantile of z; M20 EWMA | `outputs/var/garch/`, ES inputs | B |
| `var_plots.py` | 1 (comment) | VaR dynamics for SMI, EWP, UBS; spikes, staleness, lag w.r.t. volatility; local vs global violations | F1.1, V1–V3 | B |
| `var_10d.py` | 3 | 10-day VaR: FHS vs √10 rule, violations, DM with HAC (≥10 lags) and on non-overlapping periods | T3.1, F3.1 | B |
| `var_factors.py` | 1 | M16/M17 CCC/DCC Gaussian, M18/M19 CCC/DCC FHS, M21 X = b'f with skewed-t (stocks + EWP only); bonus 22+ | `outputs/var/factors*/` | C |
| `tests_lib.py` | 2 | VaR tests (UC, IND, CC, Berkowitz, DQ), tick and Lopez losses, Diebold-Mariano, MCS | – | D |
| `backtest_var.py` | 2 | all tests on all models and series, full sample and by year; summaries | T2.1–T2.6 | D |
| `es_hw2.py` | 4 | ES with HS, GARCH-N, GARCH-t, FHS, GARCH-GPD; McNeil-Frey, Du-Escanciano U_ES and C_ES; FZ0 loss, DM, MCS; by year | T4.1, T4.2, F4.1 | A |
| `main_hw2.py`, `report/make_tables.py` | – | runs everything; LaTeX tables and figures | `report/tables`, `report/figures` | D |

**Shared output format (week-1 agreement, already implemented):** every VaR script writes
`outputs/var/<script>/<series>.csv` with dates as rows and one column per model and level, named `<code>_<alpha>`
(e.g. `M04_GH_0.01`). VaR is always a **return quantile** (negative, %): violation on day *t* ⇔ `ret_t < VaR_t`.
`common.load_var(series)` merges all the VaR forecasts of a series with the realised return: the backtesting code
only needs this function, so D can test it with any subset of models (e.g. HS only) before the others are ready.

## 4. Choices to justify in the report (the professor may ask)
- **ARMA-GARCH = AR(1)-GJR-GARCH(1,1)** for every series: one specification for all, so differences between M09–M15
  come only from the innovation quantile. GJR because the asymmetry (gamma) was significant for most series in HW1.
  Gaussian QML parameters for M09, M11–M15; Student-t MLE for M10.
- **Daily re-estimation** with warm start (previous day's estimate as starting value): same optimum, much faster.
- **EVT on GARCH residuals (M13–M15)** uses the W-1 residuals of the window: only ~59 monthly maxima for the GEV.
  The extremal index on residuals is ≈ 1 (GARCH removes the clustering) → M13 ≈ M14: a result to comment.
- **Factors**: Europe 5 factors are in **USD**, our returns in **CHF**. We regress CHF log returns on the USD factors
  (the exchange-rate effect stays in the betas/residuals). Alternative: convert returns to USD. **Ask the professor.**
  RF is not used (stock returns are not in excess of RF, the intercept absorbs it).
- **Mapping VaR includes the idiosyncratic variance** (b'Σb + s²_e); without it single-stock VaR would be far too small.
  M21 (VaR of X = b'f only) deliberately ignores it → many violations on single stocks: comment on it.
- **FHS on factors**: devolatilised and decorrelated factor innovations (Cholesky of R_t) are bootstrapped *together
  with the mapping residuals of the same day*, so the dependence between factors and residuals is kept.
- **Berkowitz test of independence**: implemented as in Berkowitz, Christoffersen and Pelletier (2011), Ljung-Box(5)
  on the demeaned hit sequence. If the course slides define it differently, change `tests_lib.berkowitz_ind`.
- **MCS**: Hansen-Lunde-Nason T_max, 90% confidence, circular block bootstrap (B = 1000, block 10).
- **10-day VaR**: overlapping 10-day returns → DM uses Newey-West with ≥ 10 lags; coverage tests on non-overlapping periods.
- **ES loss**: FZ0 of Patton, Ziegel and Chen (2019), *Journal of Econometrics* 211(2), from the Fissler and Ziegel
  (2016, *Annals of Statistics*) class of losses jointly consistent for (VaR, ES). It is not a loss we designed:
  cite it as taken from the literature.
- The EWMA (M20) is also computed for the index (the assignment requires 15 models for the index, we report 16).

## 5. Division of work (replace A, B, C, D with the names, also in `report/sections/00_roles.tex`)
Point 1 is the bottleneck: it produces the 21 VaR series every other point uses. Each member owns one family of
models; D builds the backtesting code in parallel, on whatever VaR series already exist.
**Responsible** = builds/runs the code of the point, checks its results, writes the section and answers the
professor's questions on it (anyone may be asked about the choices of §4). **Reviewer** = re-reads results and text.

| Member | Points / models | Code to own (must be able to explain it) | Report files | Other tasks | Reviews |
|---|---|---|---|---|---|
| **A** | Data; Point 1 models 1–8; Point 4 (ES) | `common.py`, `var_iid.py`, `var_evt.py`, `es_hw2.py` | `01_data.tex`, `02a_unconditional.tex`, `06_es.tex` | download the SMI (LSEG); owns the shared data module | D (`04_backtesting.tex`) |
| **B** | Point 1 models 9–15, 20; Point 1 comment (VaR evolution); Point 3 | `garch_core.py`, `var_garch.py`, `var_plots.py`, `var_10d.py` | `02b_conditional.tex`, `03_var_dynamics.tex`, `05_var10.tex` | check the event dates of the figures | C (`02c_factor.tex`) |
| **C** | Point 1 models 16–19, 21, bonus 22+ | `var_factors.py` | `02c_factor.tex` | download the Fama-French factors; ask the professor about CHF returns vs USD factors | B (`02b`, `03`, `05`) |
| **D** | Point 2; integration; AI section | `tests_lib.py`, `backtest_var.py`, `main_hw2.py`, `report/make_tables.py` | `04_backtesting.tex`, `02_var_models.tex` (intro), `00_roles.tex`, `07_conclusions.tex`, `08_ai.tex`, `09_appendix.tex`, `main.tex` | check the Berkowitz test against the course slides; final run of the code; commit of `report/tables` and `report/figures`; final PDF | A (`01`, `02a`, `06`) |

Review pairs: **A ↔ D** and **B ↔ C**.

### Report files → who writes them
| File | Section | Responsible | Reviewer |
|---|---|---|---|
| `sections/00_roles.tex` | Contribution of the members | D | all |
| `sections/01_data.tex` | 1. Data and setup | A | D |
| `sections/02_var_models.tex` | 2. VaR models: introduction and table of models | D | all |
| `sections/02a_unconditional.tex` | 2.1 Unconditional models (1–8) | A | D |
| `sections/02b_conditional.tex` | 2.2 Conditional models (9–15, 20) | B | C |
| `sections/02c_factor.tex` | 2.3 Factor models (16–19, 21, bonus) | C | B |
| `sections/03_var_dynamics.tex` | 3. Evolution of the VaR (Point 1, comment) | B | C |
| `sections/04_backtesting.tex` | 4. Backtesting (Point 2) | D | A |
| `sections/05_var10.tex` | 5. 10-day VaR (Point 3) | B | C |
| `sections/06_es.tex` | 6. Expected Shortfall (Point 4) | A | D |
| `sections/07_conclusions.tex` | 7. Conclusions | D (with input from all) | all |
| `sections/08_ai.tex` | 8. Use of AI tools | D (prompts collected by all) | all |
| `sections/09_appendix.tex` | Appendix | D | – |

### Timeline (deadline: 10 January, end of day)
| By | What |
|---|---|
| ~1 December | data module and output format agreed (already implemented: §2, §3); SMI and factors downloaded |
| ~20 December | all VaR series delivered (models 1–21): each owner runs and checks their scripts |
| ~3 January | backtesting (Point 2), 10-day VaR (Point 3) and ES (Point 4) done |
| 3–10 January | writing, cross-review, final check that the code reproduces every number and figure (D) |

### Rules for everyone
- Each member writes their own sections and reviews at least one other (pairs above).
- Each member keeps a log of AI tools, purposes and prompts as they go, and sends it to D for `08_ai.tex`.
- Code must stay commented, marking which table, figure or model estimation each step produces.
- Every result must be commented from both the statistical/econometric and the financial point of view.

## 6. LaTeX report (`report/`)
- `main.tex` joins all the section files with `\input`: nobody merges files by hand, compiling `main.tex` gives the whole PDF.
- Each member edits **only their own section files** (table above); `main.tex`, `00_roles.tex`, `08_ai.tex` are edited by D only.
- `make_tables.py` (last step of `main_hw2.py`) writes `report/tables/` and copies the figures to `report/figures/`:
  never edit them by hand. Until the pipeline has been run, red boxes replace the missing tables and figures.
- Red `\todo{...}` placeholders mark the text still to write: remove them all before submitting.
- Step-by-step guide for VS Code + GitHub (installation, compiling, daily git workflow): `report/README.md`.

## 7. Report checklist (from the assignment)
- Comment every table/figure statistically **and** economically; summaries for the 16 stocks, detail for EWP and SMI.
- Group members and roles for every point. Section on AI use (tools, purpose, prompts, where they were used).
- The code must reproduce every number: run `main_hw2.py` once at the end, from a clean `outputs/`.
