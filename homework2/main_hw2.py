"""Homework 2 - runs every step in order and stops at the first error. Run from inside homework2/:
Responsible: member D (integration: the code must reproduce every number and figure of the report).

    python main_hw2.py                  # everything
    python main_hw2.py --from backtest  # restart from one step (outputs of the earlier steps must exist)
    python main_hw2.py --only es        # a single step
    python main_hw2.py --list

STEP       POINT  SCRIPT          WHAT IT DOES
  iid      1      var_iid.py      M01-M05 (HS, Normal, t, GH, Cornish-Fisher)                     [A]
  evt      1      var_evt.py      M06-M08 (GEV, GEV with extremal index, GPD; expanding)           [A]
  garch    1      var_garch.py    M09-M15 (AR(1)-GJR-GARCH + 7 innovation quantiles), M20 EWMA     [B]
  factors  1      var_factors.py  M16-M19 (factors + CCC/DCC, Gaussian/FHS), M21 (X skew-t)       [C]
  backtest 2      backtest_var.py UC/IND/CC, BCP, DQ, Diebold-Mariano vs HS, MCS, by year          [D]
  plots    1      var_plots.py    VaR dynamics of index/EWP/UBS, local vs global violations        [B]
  var10    3      var_10d.py      10-day VaR: FHS vs square-root rule, Diebold-Mariano             [B]
  es       4      es_hw2.py       ES with 5 approaches, McNeil-Frey, Du-Escanciano, FZ0 + MCS      [A]
  report   -      report/make_tables.py  LaTeX tables and figures for the report                   [D]
What each script prints is also saved in outputs/logs/<step>.log.
"""
import os, sys, time, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)                                               # every script uses paths relative to homework2/
STEPS = [("iid", "var_iid.py"), ("evt", "var_evt.py"), ("garch", "var_garch.py"), ("factors", "var_factors.py"),
         ("backtest", "backtest_var.py"), ("plots", "var_plots.py"), ("var10", "var_10d.py"), ("es", "es_hw2.py"),
         ("report", os.path.join("report", "make_tables.py"))]   # order matters
LOG_DIR = os.path.join("outputs", "logs")


def run(step, script):
    """Runs one script with the same Python, echoes its output and saves it in outputs/logs/<step>.log."""
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"\n{'=' * 90}\n>>> {step}: {script}\n{'=' * 90}", flush=True)
    t0 = time.time()
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}   # figures saved, no window; output not buffered
    with open(os.path.join(LOG_DIR, f"{step}.log"), "w", encoding="utf-8") as log:
        p = subprocess.Popen([sys.executable, "-W", "ignore", script], env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:                                # echo and log every line
            sys.stdout.write(line); log.write(line)
        code = p.wait()
    if code != 0:                                            # stop at the first failing step
        sys.exit(f"\nERROR: step '{step}' failed (exit code {code}); see {LOG_DIR}/{step}.log. "
                 f"After fixing it: python main_hw2.py --from {step}")
    print(f"<<< {step} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    names = [n for n, _ in STEPS]
    ap.add_argument("--from", dest="start", choices=names)   # first step to run
    ap.add_argument("--only", choices=names)                 # run a single step
    ap.add_argument("--list", action="store_true")           # print the steps and exit
    ar = ap.parse_args()
    if ar.list:
        print(__doc__); sys.exit()
    todo = [s for s in STEPS if s[0] == ar.only] if ar.only else STEPS[names.index(ar.start) if ar.start else 0:]
    t0 = time.time()
    for step, script in todo:
        run(step, script)
    print(f"\nAll steps finished in {(time.time() - t0) / 60:.1f} min. Tables: outputs/tables, figures: outputs/figures")
