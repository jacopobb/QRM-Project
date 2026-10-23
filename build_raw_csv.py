"""Point 1 (collection) - merge the 12 LSEG Excel exports into data/prices_raw.csv.
Run once:  python build_raw_csv.py     File names must be the RIC, e.g. data/raw/NESN.S.xlsx"""
import os, pandas as pd
from settings import *

CLOSE_NAMES = ["close", "close price", "price close", "last", "last price", "price", "absolute price", "adj close", "adjusted close"]

def read_one(path):
    """LSEG export layout: row 'Date | <RIC> (TRDPRC_1)', next row 'NaN | Close', then data (newest first)."""
    raw = pd.read_excel(path, header=None)
    hdr = next(i for i in range(min(30, len(raw)))
               if raw.iloc[i].astype(str).str.strip().str.lower().isin(["date", "exchange date"]).any())
    date_col = next(j for j in range(raw.shape[1])
                    if str(raw.iloc[hdr, j]).strip().lower() in ("date", "exchange date"))
    # field-name row: the row under the header (e.g. 'Close'); otherwise the header row itself
    labels = [str(raw.iloc[hdr + 1, j]).strip().lower() for j in range(raw.shape[1])]
    if any(l in CLOSE_NAMES for l in labels):
        price_col, first = next(j for j, l in enumerate(labels) if l in CLOSE_NAMES), hdr + 2
    else:
        names = [str(raw.iloc[hdr, j]).strip().lower() for j in range(raw.shape[1])]
        cand = [j for j, n in enumerate(names) if n in CLOSE_NAMES]
        if not cand:
            raise ValueError(f"No close-price column in {path}. Header: {list(raw.iloc[hdr])}, "
                             f"next row: {list(raw.iloc[hdr + 1])}. Add the right name to CLOSE_NAMES.")
        price_col, first = cand[0], hdr + 1
    data = raw.iloc[first:]
    s = pd.Series(pd.to_numeric(data.iloc[:, price_col], errors="coerce").values,
                  index=pd.to_datetime(data.iloc[:, date_col], errors="coerce"))
    return s[s.index.notna()].sort_index()

if __name__ == "__main__":
    series = {}
    for ric in TICKERS:
        # accept both "NESN.S.xlsx" and "NESN.xlsx" (file named without the ".S" suffix)
        candidates = [os.path.join(RAW_DIR, f"{n}.xlsx") for n in (ric, ric.split(".")[0])]
        path = next(c for c in candidates if os.path.exists(c))
        series[ric] = read_one(path)
        print(f"{ric}: {series[ric].index.min().date()} -> {series[ric].index.max().date()}, {series[ric].notna().sum()} prices")
    pd.DataFrame(series).sort_index().rename_axis("Date").to_csv(RAW_FILE)
    print("saved", RAW_FILE)
