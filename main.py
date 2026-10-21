"""Runs everything in order and saves tables/figures in outputs/.  python main.py"""
import numpy as np, pandas as pd
from config import *
import data_prep, descriptive, garch_models, evt, backtest

np.random.seed(SEED)
px   = data_prep.load_clean_prices()                      # Point 1
rets = data_prep.log_returns(px)                          # Point 2
w, port = data_prep.make_port(rets.loc[:IN_SAMPLE_END])   # Point 3: weights from the loop below
print("Weights:\n", w)
all_r = rets.copy(); all_r["PORT"] = (rets * w).sum(axis=1)
print(descriptive.summary_table(all_r))
for col in all_r:                                         # Points 3-8 per series
    res = garch_models.fit_in_sample(all_r[col])
    Z = garch_models.std_resid(res)
    # ... VaR cases 1-5, backtests, plots, ES: see the modules; TODO per team member
