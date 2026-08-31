#!/usr/bin/env python3
"""Turn output/latest_candidates.csv into docs/data.json for the scanner app.

Run by the GitHub Action after screener.py. Kept separate from the screener on purpose:
the screen must never depend on the app, and a failure here must not lose a screen run.
"""
import json, math, os, datetime as dt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

def ndx_members():
    """Nasdaq-100 tickers, fetched independently of the CSV.

    The CSV's own `index` column under-tags membership: dual-listed names (most of the
    NDX) carry only 'SP500'. Known bug, deliberately left in place - work around it here.
    """
    try:
        import screener
        t = screener._constituents("https://en.wikipedia.org/wiki/Nasdaq-100",
                                   ["Ticker", "Symbol"], 50, "NDX")
        s = set(t["ticker"].str.replace(".", "-", regex=False).str.strip())
        if len(s) < 60:
            raise RuntimeError("only %d tickers parsed - refusing a thin list" % len(s))
        return s
    except Exception as e:
        print("  WARNING: Nasdaq-100 fetch failed (%s)" % e)
        print("  Falling back to the static list below (constituents as of 2026-08-31).")
        return set(STATIC_NDX.split(","))

# Fallback constituent list, verified against Wikipedia on 2026-08-31. Index membership
# changes a few names a quarter, so this drifts slowly; the live fetch above is preferred
# and this only catches the runner environments where that parse fails.
STATIC_NDX = ("ADBE,AMD,ABNB,ALNY,GOOGL,GOOG,AMZN,AEP,AMGN,ADI,AAPL,AMAT,APP,ARM,ASML,ADSK,ADP,AXON,BKR,BKNG,AVGO,CDNS,CHTR,CTAS,CSCO,CCEP,CTSH,CMCSA,CEG,CPRT,CSGP,COST,CRWD,CSX,DDOG,DXCM,FANG,DASH,EA,EXC,FAST,FER,FTNT,GEHC,GILD,HON,IDXX,INSM,INTC,INTU,ISRG,KDP,KLAC,KHC,LRCX,LIN,MAR,MRVL,MELI,META,MCHP,MU,MSFT,MSTR,MDLZ,MPWR,MNST,NFLX,NVDA,NXPI,ORLY,ODFL,PCAR,PLTR,PANW,PAYX,PYPL,PDD,PEP,QCOM,REGN,ROP,ROST,SNDK,STX,SHOP,SBUX,SNPS,TMUS,TTWO,TSLA,TXN,TRI,VRSK,VRTX,WMT,WBD,WDC,WDAY,XEL,ZS")

COLS = ['Ticker','name','sector','score','rank','price','ret_1w','ret_1m','ret_3m','ret_6m',
        'rs_3m_chg','pct_from_52w_high','pct_from_52w_low','pct_vs_20dma','pct_vs_50dma',
        'pct_vs_200dma','dist_to_50dma','pct_vs_9ema','pullback_9ema','avg_dollar_vol_m',
        'vol_ratio_20_60','rvol','gap_pct','ndx']

def clean(v):
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 4)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    return v

def main():
    d = pd.read_csv(os.path.join(HERE, "output", "latest_candidates.csv"))
    d = d.rename(columns={d.columns[0]: "Ticker"})
    ndx = ndx_members()
    d["ndx"] = d["Ticker"].isin(ndx) if ndx is not None \
        else d.get("index", pd.Series("", index=d.index)).astype(str).str.contains("NDX")
    d = d[[c for c in COLS if c in d.columns]]
    rows = [{k: clean(v) for k, v in r.items()} for r in d.to_dict("records")]
    payload = {"run": dt.date.today().isoformat(),
               "rows": rows,
               "sectors": sorted(x for x in d["sector"].dropna().unique().tolist() if x)}
    out = os.path.join(HERE, "docs", "data.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, separators=(",", ":"), allow_nan=False)
    print("docs/data.json: %d rows, %d KB, ndx tagged %d" %
          (len(rows), os.path.getsize(out) // 1024, sum(1 for r in rows if r.get("ndx"))))

if __name__ == "__main__":
    raise SystemExit(main())
