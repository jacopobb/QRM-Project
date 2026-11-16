"""Runs the whole project in order and writes tables, figures and forecasts in outputs/.

    python main.py                 # every step (the raw-data step is skipped if data/prices_raw.csv already exists)
    python main.py --from garch    # restart from one step (the outputs of the earlier steps must already exist)
    python main.py --only es       # a single step
    python main.py --list          # the steps and what each one does

Each step is a script that can also be run alone (python <script>.py). This file only calls them in the right order,
from this folder, and stops at the first error so that a failure is never hidden by the later steps.
What each script prints is also saved in outputs/logs/<step>.log (the cleaning report of Point 1 is in outputs/logs/data.log).

BEFORE THE FIRST RUN
  1. Python 3.10+ and:  pip install numpy pandas scipy statsmodels arch matplotlib seaborn openpyxl
  2. The 16 LSEG exports, one Excel file per RIC (e.g. data/raw/NESN.S.xlsx), are in the folder RAW_DIR of settings.py.
     The merged file RAW_FILE (data/prices_raw.csv) is built from them by build_raw_csv.py.
  3. All choices (stocks, dates, alphas, seed, GARCH window) are in settings.py. Random choices (portfolio weights) use SEED:
     same seed -> same weights, tables and figures. Do not change settings between steps.
  4. The GARCH step re-estimates the models every H_REFIT days on a rolling window: it is the slowest one (minutes).

STEP          POINT  SCRIPT            WHAT IT DOES
  raw         1      build_raw_csv.py  merges the Excel exports into data/prices_raw.csv
  data        1      data_prep.py      cleaning report (weekends, holidays, gaps, jumps), clean prices and log returns, weights
  descriptive 2      descriptive.py    descriptive analysis of the returns (tables T*, figures F*)
  garch       3-4    garch_models.py   GARCH on every series and PORT, weights, Z, VaR Cases 1-2 -> outputs/forecasts/<series>.csv
  evt         5-7    evt.py            GEV and GPD on the losses of Z, VaR Cases 3-5 added to outputs/forecasts/<series>.csv
  backtest    8      backtest.py       comparison of the VaR sequences -> outputs/tables/T8.*, outputs/figures/F8.*
  es          9      es.py             Expected Shortfall -> outputs/tables/S*, outputs/figures/S*, outputs/es/<series>.csv
"""
import os, sys, time, argparse, importlib.util, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)                                   # every script uses paths relative to the project folder
sys.path.insert(0, HERE)

STEPS = [("raw", "build_raw_csv.py"), ("data", "data_prep.py"), ("descriptive", "descriptive.py"),
         ("garch", "garch_models.py"), ("evt", "evt.py"), ("backtest", "backtest.py"), ("es", "es.py")]
PACKAGES = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "statsmodels": "statsmodels", "arch": "arch",
            "matplotlib": "matplotlib", "seaborn": "seaborn", "openpyxl": "openpyxl"}   # import name -> pip name
LOG_DIR = os.path.join("outputs", "logs")


def load_settings():
    """The settings file (settings.py, or config.py in older copies) gives the data paths."""
    for name in ("settings", "config"):
        if importlib.util.find_spec(name):
            return __import__(name)
    sys.exit("ERROR: settings.py not found in the project folder.")


def check_environment(names, cfg):
    """Fails early, with a clear message, if a package or an input file is missing."""
    missing = [pip for imp, pip in PACKAGES.items() if importlib.util.find_spec(imp) is None]
    if missing:
        sys.exit(f"ERROR: missing packages: {', '.join(missing)}\n  pip install {' '.join(missing)}")
    for _, script in STEPS:
        if not os.path.exists(script):
            sys.exit(f"ERROR: {script} not found in {HERE}")
    first = names[0]
    if first == "raw" and not os.path.isdir(cfg.RAW_DIR):
        sys.exit(f"ERROR: folder {cfg.RAW_DIR} with the LSEG Excel files not found.")
    if first in ("data", "descriptive", "garch") and not os.path.exists(cfg.RAW_FILE):
        sys.exit(f"ERROR: {cfg.RAW_FILE} not found: run `python main.py --only raw` (needs the Excel files in {cfg.RAW_DIR}).")


def run(step, script):
    """Runs one script with the same Python, echoes its output and saves it in outputs/logs/<step>.log."""
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"\n{'=' * 100}\n>>> {step}: {script}\n{'=' * 100}", flush=True)
    t0 = time.time()
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}   # Agg: figures are saved, no window opens
    with open(os.path.join(LOG_DIR, f"{step}.log"), "w", encoding="utf-8") as log:
        p = subprocess.Popen([sys.executable, script], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line); log.write(line)
        code = p.wait()
    if code != 0:
        sys.exit(f"\nERROR: step '{step}' ({script}) failed with exit code {code}. Later steps were not run; "
                 f"see {os.path.join(LOG_DIR, step + '.log')}. After fixing it: python main.py --from {step}")
    print(f"<<< {step} done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    names = [n for n, _ in STEPS]
    ap.add_argument("--from", dest="start", choices=names, help="first step to run")
    ap.add_argument("--only", choices=names, help="run only this step")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    ar = ap.parse_args()
    if ar.list:
        for n, f in STEPS:
            print(f"{n:12s} {f}")
        return
    todo = [s for s in STEPS if s[0] == ar.only] if ar.only else STEPS[names.index(ar.start) if ar.start else 0:]
    cfg = load_settings()
    explicit = bool(ar.only or ar.start)
    if not explicit and os.path.exists(cfg.RAW_FILE):                  # default run: the merged file is already there
        print(f"{cfg.RAW_FILE} exists: step 'raw' skipped (use --only raw to rebuild it).")
        todo = [s for s in todo if s[0] != "raw"]
    check_environment([s[0] for s in todo], cfg)
    for d in (os.path.dirname(cfg.RAW_FILE) or ".", "outputs/tables", "outputs/figures", "outputs/forecasts"):
        os.makedirs(d, exist_ok=True)
    t0 = time.time()
    for step, script in todo:
        run(step, script)
    print(f"\nAll steps finished in {(time.time() - t0) / 60:.1f} min. Tables: outputs/tables, figures: outputs/figures, "
          f"forecasts: outputs/forecasts, ES: outputs/es, logs: {LOG_DIR}")


if __name__ == "__main__":
    main()
