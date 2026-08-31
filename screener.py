#!/usr/bin/env python3
"""
screener.py - deterministic momentum / relative-strength screen over S&P 500 + Nasdaq 100.

Design rule (see 00_claude_investing_blueprint.md): this file contains NO judgement and NO
forecasting. It ranks the universe mechanically and writes CSVs. Interpretation happens
afterwards, in Claude, on the shortlist only.

WHY IT RUNS HERE AND NOT IN CLAUDE'S SHELL
------------------------------------------
Both of Claude's shells (the cloud container and the Cowork sandbox VM on this Mac) sit behind
an egress allowlist that blocks Yahoo, SEC, Finnhub, FMP and friends. Your normal Terminal does
not. So this script is run BY YOU, in Terminal, and Claude reads the CSVs it leaves behind -
reading a local file needs no network.

USAGE
-----
    pip3 install yfinance pandas numpy lxml
    python3 screener.py                 # full run, writes CSVs next to this file
    python3 screener.py --self-test     # no network; verifies the maths on synthetic data
    python3 screener.py --universe sp500

OUTPUT (written to ./output/)
    candidates_YYYY-MM-DD.csv   ranked universe, full metrics
    rotation_YYYY-MM-DD.csv     sector / index ETF relative strength
    latest_candidates.csv       copy of the newest run, stable filename for Claude to read
    latest_rotation.csv

SCORED vs REPORTED
------------------
Only the columns in WEIGHTS move the composite. Everything else in the CSV - short-window
returns, pct_from_52w_low, gap_pct, gap_up, rvol - is reported so the interrogation pass can
see it, and is deliberately excluded from the score. Adding a column is cheap; adding a
weight changes what the screen IS, and needs evidence from the dated archives first.
"""

import argparse
import os
import sys
import datetime as dt

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "output")
CACHEDIR = os.path.join(HERE, ".cache")

# Sector and broad-index ETFs for the rotation table.
ROTATION_ETFS = {
    "XLK": "Technology",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "SPY": "S&P 500 (benchmark)",
    "QQQ": "Nasdaq 100 (benchmark)",
    "IWM": "Russell 2000 (small cap)",
    "RSP": "S&P 500 equal weight (breadth check)",
}

# Composite weights. Tune these; they are deliberately exposed rather than buried.
# Every component is converted to a 0-1 percentile rank across the universe first,
# so the weights are comparable.
WEIGHTS = {
    "rs_3m": 0.25,        # medium-term relative strength
    "rs_6m": 0.20,        # longer-term trend
    "rs_1m": 0.10,        # near-term, deliberately underweighted (noisy, mean-reverts)
    "rs_trend": 0.20,     # is the RS rank IMPROVING - improving beats merely high
    "near_high": 0.10,    # proximity to 52w high
    "vol_contraction": 0.10,  # range narrowing into a base
    "volume_surge": 0.05, # recent participation
}

# Overnight-gap threshold for the reported gapper list. 3% is the figure from the
# Humbled Trader prefilter. It is a DISPLAY filter only - gap_pct, gap_up and rvol are
# written to the CSV and printed, and touch no score. See the note in compute_metrics.
GAP_THRESHOLD = 0.03

# "Strong but extended" report. A momentum screen selects for extension almost by
# definition - on the 2026-08-30 run, 31 of the top 40 sat >=10% above their 50-day and the
# MEDIAN was 13.5%. So an absolute threshold flags nearly everything and discriminates
# nothing. The report therefore ranks extension RELATIVE to the shortlist and prints the
# median as the reference line; the threshold below is only used to count how many clear it.
# Display only; touches no score.
EXT_THRESHOLD = 0.10
EXT_SHOW = 12          # how many of the most extended top-ranked names to list


# --------------------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------------------

def _fetch_html(url):
    """Fetch a page with an explicit User-Agent.

    pandas.read_html(url) hands the URL to urllib, which sends a default User-Agent that
    Wikipedia rejects with HTTP 403. Fetching the HTML ourselves with a real UA avoids that.
    """
    import urllib.request
    req = urllib.request.Request(url, headers={
        "User-Agent": "InvestingAssistant/1.0 (personal research script; contact: local user)",
        "Accept": "text/html,application/xhtml+xml",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def pick_constituent_table(tables, ticker_candidates, min_rows):
    """Choose the constituents table by its COLUMNS, not by regex-matching page text.

    read_html(match=...) is brittle - it greps the rendered table text, so a layout change
    or a differently-rendered header silently yields 'No tables found'. Selecting on column
    names plus a row-count floor is far more stable across Wikipedia edits.

    Returns (dataframe, ticker_column_name).
    """
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        for cand in ticker_candidates:
            if cand in cols and len(t) >= min_rows:
                t = t.copy()
                t.columns = cols
                return t, cand
    raise RuntimeError(
        "no table with >=%d rows and one of the columns %s (saw %d tables with columns: %s)"
        % (min_rows, ticker_candidates, len(tables),
           [[str(c) for c in t.columns][:4] for t in tables[:5]]))


def _first_present(cols, candidates, default=None):
    for c in candidates:
        if c in cols:
            return c
    return default


def get_universe(which="both", refresh=False):
    """Index constituents, plus any manual additions in universe_extra.csv."""
    df = _get_universe_base(which, refresh)

    extra_path = os.path.join(HERE, "universe_extra.csv")
    if os.path.exists(extra_path):
        extra = pd.read_csv(extra_path)
        new = extra[~extra["ticker"].isin(set(df["ticker"]))]
        if len(new):
            df = pd.concat([df, new], ignore_index=True)
            print("  + %d names from universe_extra.csv" % len(new))
    return df


def _get_universe_base(which="both", refresh=False):
    """S&P 500 + Nasdaq 100 tickers from Wikipedia, cached to disk."""
    os.makedirs(CACHEDIR, exist_ok=True)
    cache = os.path.join(CACHEDIR, "universe_%s.csv" % which)
    if os.path.exists(cache) and not refresh:
        age_days = (dt.datetime.now().timestamp() - os.path.getmtime(cache)) / 86400
        if age_days < 30:
            df = pd.read_csv(cache)
            print("Universe: %d tickers (cached, %.0f days old)" % (len(df), age_days))
            return df

    try:
        df = _fetch_universe(which)
    except Exception as e:
        fallback = os.path.join(HERE, "universe_fallback.csv")
        if os.path.exists(fallback):
            df = pd.read_csv(fallback)
            if which == "sp500":
                df = df[df["index"].str.contains("SP500")]
            elif which == "ndx":
                df = df[df["index"].str.contains("NDX")]
            print("Universe: %d tickers (Wikipedia failed: %s -- using universe_fallback.csv)"
                  % (len(df), e))
            return df
        msg = str(e)
        hint = ""
        if "html5lib" in msg or "lxml" in msg or "bs4" in msg or "BeautifulSoup" in msg:
            hint = ("\n\nThis is a missing HTML parser, not a network problem. Fix with:\n"
                    "    pip3 install lxml html5lib beautifulsoup4\n"
                    "then re-run. The constituent list is cached for 30 days afterwards, so\n"
                    "this only has to work once a month.")
        raise SystemExit(
            "Could not fetch the index constituents from Wikipedia (%s)\n"
            "and no universe_fallback.csv is present next to this script.%s" % (e, hint))

    df.to_csv(cache, index=False)
    print("Universe: %d tickers (fetched)" % len(df))
    return df


def _constituents(url, ticker_candidates, min_rows, label):
    """Fetch one index's constituent list and normalise it to ticker/name/sector/index."""
    import io
    tables = pd.read_html(io.StringIO(_fetch_html(url)))
    t, tcol = pick_constituent_table(tables, ticker_candidates, min_rows)
    cols = list(t.columns)
    namecol = _first_present(cols, ["Security", "Company", "Company Name", "Name"])
    seccol = _first_present(cols, ["GICS Sector", "ICB Industry", "Sector", "Industry"])
    indcol = _first_present(cols, ["GICS Sub-Industry", "ICB Subsector", "Sub-Industry", "Subsector"])
    return pd.DataFrame({
        "ticker": t[tcol].astype(str),
        "name": t[namecol].astype(str) if namecol else "",
        "sector": t[seccol].astype(str) if seccol else "",
        "industry": t[indcol].astype(str) if indcol else "",
        "index": label,
    })


def _fetch_universe(which):
    frames = []
    if which in ("both", "sp500"):
        frames.append(_constituents(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            ["Symbol", "Ticker"], 400, "SP500"))
    if which in ("both", "ndx"):
        try:
            frames.append(_constituents(
                "https://en.wikipedia.org/wiki/Nasdaq-100",
                ["Ticker", "Symbol"], 50, "NDX"))
        except Exception as e:
            # The Nasdaq-100 adds roughly 30 names the S&P 500 doesn't already cover.
            # Losing it degrades the universe slightly; it must not abort the whole run.
            if which == "ndx":
                raise
            print("  WARNING: Nasdaq-100 list unavailable (%s)" % e)
            print("  Continuing with the S&P 500 only.")

    df = pd.concat(frames, ignore_index=True)
    # Yahoo uses '-' where Wikipedia uses '.' (BRK.B -> BRK-B)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False).str.strip()
    # Collapse duplicates (a name in both indices), remembering both memberships.
    if "industry" not in df.columns:
        df["industry"] = ""
    df = (df.groupby("ticker", as_index=False)
            .agg(name=("name", "first"),
                 sector=("sector", "first"),
                 industry=("industry", "first"),
                 index=("index", lambda s: "+".join(sorted(set(s))))))
    return df


# --------------------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------------------

MIN_SESSIONS = 180


def _download_chunk(chunk, period, threads):
    """One yfinance call. Returns (close, volume, open_), or (None, None, None) on failure.

    Open is pulled only so the overnight gap can be computed. It is optional everywhere
    downstream: if a provider stops returning it, gap_pct goes NaN rather than wrong.
    """
    import yfinance as yf
    d = yf.download(chunk, period=period, interval="1d", auto_adjust=True,
                    progress=False, threads=threads, group_by="column")
    if d is None or len(d) == 0:
        return None, None, None
    if isinstance(d.columns, pd.MultiIndex):
        op = d["Open"] if "Open" in d.columns.get_level_values(0) else None
        return d["Close"], d["Volume"], op
    op = d[["Open"]].rename(columns={"Open": chunk[0]}) if "Open" in d.columns else None
    return (d[["Close"]].rename(columns={"Close": chunk[0]}),
            d[["Volume"]].rename(columns={"Volume": chunk[0]}),
            op)


def fetch_prices(tickers, period="1y", batch=50, threads=4, pause=1.0):
    """Bulk daily OHLCV. Returns (close_df, volume_df, open_df) indexed by date, columns = tickers.

    open_df is None if no chunk returned Open data; every consumer must tolerate that.

    Concurrency is deliberately modest. Large batches with unlimited threads make macOS
    fail with 'getaddrinfo() thread failed to start' and spurious DNS/SSL errors - that is
    local resource exhaustion, not Yahoo rate-limiting, and it silently drops good tickers.
    Anything that still comes back empty gets one serial retry before being given up on.
    """
    import time

    closes, volumes, opens = [], [], []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        print("  downloading %d-%d of %d..." % (i + 1, min(i + batch, len(tickers)), len(tickers)))
        try:
            c, v, o = _download_chunk(chunk, period, threads)
        except Exception as e:
            print("    batch failed (%s) - will retry these individually" % e)
            continue
        if c is not None:
            closes.append(c)
            volumes.append(v)
            if o is not None:
                opens.append(o)
        time.sleep(pause)

    if not closes:
        raise SystemExit("No price data returned at all. Check your internet connection.")

    close = pd.concat(closes, axis=1).sort_index()
    volume = pd.concat(volumes, axis=1).sort_index()

    # Retry anything that came back empty or too thin, serially and slowly.
    got = set(close.columns[close.notna().sum() >= MIN_SESSIONS])
    missing = [t for t in tickers if t not in got]
    if missing:
        print("  retrying %d names serially: %s%s"
              % (len(missing), ", ".join(missing[:12]), " ..." if len(missing) > 12 else ""))
        fixed = []
        for t in missing:
            try:
                c, v, o = _download_chunk([t], period, False)
            except Exception:
                continue
            if c is None or c.notna().sum().sum() < MIN_SESSIONS:
                continue
            c.columns, v.columns = [t], [t]
            close = close.drop(columns=[t], errors="ignore").join(c, how="outer")
            volume = volume.drop(columns=[t], errors="ignore").join(v, how="outer")
            if o is not None:
                o.columns = [t]
                opens.append(o)   # de-duplicated below, keeping this newer copy
            fixed.append(t)
            time.sleep(0.4)
        print("  recovered %d of %d on retry" % (len(fixed), len(missing)))

    keep = close.columns[close.notna().sum() >= MIN_SESSIONS]
    dropped = len(close.columns) - len(keep)
    if dropped:
        lost = sorted(set(close.columns) - set(keep))
        print("  dropped %d tickers with <%d sessions: %s%s"
              % (dropped, MIN_SESSIONS, ", ".join(lost[:15]),
                 " ..." if len(lost) > 15 else ""))
    print("  usable universe: %d names" % len(keep))
    close = close[keep].sort_index()
    volume = volume[keep].sort_index()

    if opens:
        open_ = pd.concat(opens, axis=1)
        # A serially-retried ticker can appear twice; the retry copy is the good one.
        open_ = open_.loc[:, ~open_.columns.duplicated(keep="last")]
        # Force the same shape as close, so a ticker missing an Open is NaN, not misaligned.
        open_ = open_.reindex(index=close.index, columns=close.columns)
    else:
        open_ = None
        print("  WARNING: no Open data returned - gap_pct will be blank this run")
    return close, volume, open_


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def _ret(close, days):
    if len(close) <= days:
        return pd.Series(np.nan, index=close.columns)
    return close.iloc[-1] / close.iloc[-1 - days] - 1.0


def compute_metrics(close, volume, open_=None):
    """All metrics are point-in-time as of the last row. No forward-looking data is used."""
    out = pd.DataFrame(index=close.columns)

    out["price"] = close.iloc[-1]
    # Short windows are REPORTED but not scored. They answer "what moved recently",
    # which is a different question from "what has been strong", and mixing the two
    # into one number would muddle both.
    out["ret_1d"] = _ret(close, 1)
    out["ret_1w"] = _ret(close, 5)
    out["ret_1m"] = _ret(close, 21)
    out["ret_3m"] = _ret(close, 63)
    out["ret_6m"] = _ret(close, 126)

    # Relative strength = cross-sectional percentile of the return (0-1).
    out["rs_1m"] = out["ret_1m"].rank(pct=True)
    out["rs_3m"] = out["ret_3m"].rank(pct=True)
    out["rs_6m"] = out["ret_6m"].rank(pct=True)

    # Is relative strength improving? Compare today's 3m RS rank with the rank as of a month ago.
    if len(close) > 63 + 21:
        past = close.iloc[:-21]
        past_ret_3m = past.iloc[-1] / past.iloc[-1 - 63] - 1.0
        past_rs_3m = past_ret_3m.rank(pct=True)
        out["rs_trend"] = (out["rs_3m"] - past_rs_3m).rank(pct=True)
        out["rs_3m_chg"] = out["rs_3m"] - past_rs_3m
    else:
        out["rs_trend"] = np.nan
        out["rs_3m_chg"] = np.nan

    # Moving averages
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else pd.Series(np.nan, index=close.columns)
    out["pct_vs_50dma"] = out["price"] / sma50 - 1.0
    out["pct_vs_200dma"] = out["price"] / sma200 - 1.0
    out["above_50dma"] = out["pct_vs_50dma"] > 0
    out["above_200dma"] = out["pct_vs_200dma"] > 0

    # How far price must MOVE to reach the 50-day average - negative means it must fall.
    # This is NOT pct_vs_50dma with the sign flipped: +19.2% above the mean is a -16.1%
    # fall back to it, not -19.2%. Reported, not scored. It exists so the size of an
    # ordinary mean-reversion is visible on the CSV instead of being worked out by hand.
    out["dist_to_50dma"] = sma50 / out["price"] - 1.0

    # 52-week high proximity
    hi52 = close.rolling(min(252, len(close))).max().iloc[-1]
    out["pct_from_52w_high"] = out["price"] / hi52 - 1.0
    out["new_52w_high"] = out["pct_from_52w_high"] >= -0.001
    out["near_high"] = out["pct_from_52w_high"].rank(pct=True)

    # Distance from the 52-week LOW. Reported, not scored. A name that is both near its
    # high and 70% off its low is a mature recovery, not a fresh base - a distinction
    # every momentum measure hides, and the one that mattered most in the first
    # interrogation pass. See 00_claude_investing_blueprint.md.
    lo52 = close.rolling(min(252, len(close))).min().iloc[-1]
    out["pct_from_52w_low"] = out["price"] / lo52 - 1.0

    # Volatility contraction: 20d realised vol vs the preceding 60d. Lower ratio = tighter base.
    rets = close.pct_change()
    vol20 = rets.iloc[-20:].std()
    vol60 = rets.iloc[-80:-20].std()
    ratio = vol20 / vol60
    out["vol_ratio_20_60"] = ratio
    out["vol_contraction"] = (-ratio).rank(pct=True)  # tighter ranks higher

    # Volume surge: last 5 sessions vs 50-day average
    v5 = volume.iloc[-5:].mean()
    v50 = volume.iloc[-50:].mean()
    out["volume_x"] = v5 / v50
    out["volume_surge"] = out["volume_x"].rank(pct=True)

    # Overnight gap and single-day relative volume. REPORTED, DELIBERATELY NOT SCORED.
    #
    # Borrowed from the gap-up prefilter idea, with two limits stated rather than hidden:
    #   1. This is the gap of the LAST COMPLETED SESSION, not a live pre-market gap. On a
    #      Sunday run it is Friday's. yfinance daily bars end at the close - there is no
    #      pre-market number in this data, and a column header implying one would be a lie.
    #   2. A gap is an event; this is a trend screen. Weighting it into the composite would
    #      quietly turn a momentum screen into a news screen. If it ever earns a weight it
    #      earns it from the dated scorecard archives, not from a tutorial.
    if open_ is not None and len(close) >= 2:
        out["gap_pct"] = open_.iloc[-1] / close.iloc[-2] - 1.0
    else:
        out["gap_pct"] = np.nan
    out["gap_up"] = out["gap_pct"] >= GAP_THRESHOLD

    # Relative volume: the latest session against the 50 sessions BEFORE it. Excluding the
    # day itself matters - a genuine 5x day included in its own baseline reads as about 4x.
    if len(volume) >= 51:
        out["rvol"] = volume.iloc[-1] / volume.iloc[-51:-1].mean()
    else:
        out["rvol"] = np.nan

    # Composite
    score = pd.Series(0.0, index=out.index)
    total_w = 0.0
    for col, w in WEIGHTS.items():
        c = out[col].fillna(0.5)  # missing component scores neutral rather than disqualifying
        score = score + c * w
        total_w += w
    out["score"] = score / total_w
    out["rank"] = out["score"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("score", ascending=False)


def compute_breadth(m, level="sector", min_names=3):
    """Is a group's move broad, or are two names carrying it?

    A sector ETF's return tells you the group went up. It does NOT tell you whether most
    of its members went up - a cap-weighted index can rise on two giants while the median
    constituent falls. That difference decides whether to buy the sector or hunt inside it.

    `concentration` is mean minus median return. Large and positive means a few big winners
    are dragging the average above the typical stock: a narrow move.
    """
    g = m[m[level].astype(str).str.len() > 0].groupby(level)
    rows = []
    for name, d in g:
        if len(d) < min_names:
            continue
        rows.append({
            level: name,
            "n": len(d),
            "median_1m": d["ret_1m"].median(),
            "median_3m": d["ret_3m"].median(),
            "mean_3m": d["ret_3m"].mean(),
            "concentration": d["ret_3m"].mean() - d["ret_3m"].median(),
            "pct_above_50dma": d["above_50dma"].mean(),
            "pct_above_200dma": d["above_200dma"].mean(),
            "pct_at_52w_high": d["new_52w_high"].mean(),
            "median_score": d["score"].median(),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("median_3m", ascending=False).reset_index(drop=True)


def compute_rotation(close):
    """Relative strength of sector / index ETFs over 1w, 1m, 3m."""
    rows = []
    for t in close.columns:
        rows.append({
            "ticker": t,
            "what": ROTATION_ETFS.get(t, ""),
            "ret_1w": float(_ret(close[[t]], 5).iloc[0]),
            "ret_1m": float(_ret(close[[t]], 21).iloc[0]),
            "ret_3m": float(_ret(close[[t]], 63).iloc[0]),
        })
    df = pd.DataFrame(rows)
    # Excess return vs SPY makes the rotation readable at a glance.
    if "SPY" in df["ticker"].values:
        spy = df[df["ticker"] == "SPY"].iloc[0]
        for h in ("1w", "1m", "3m"):
            df["excess_" + h] = df["ret_" + h] - spy["ret_" + h]
    return df.sort_values("ret_3m", ascending=False)


# --------------------------------------------------------------------------------------
# Self-test (no network)
# --------------------------------------------------------------------------------------

def self_test():
    """Verify the maths on synthetic series with known properties."""
    rng = np.random.default_rng(0)
    n = 60
    idx = pd.bdate_range(end=dt.date.today(), periods=300)
    days = len(idx)
    cols = ["T%02d" % i for i in range(n)]

    # Give each name a deterministic drift; T00 strongest, T59 weakest.
    # Low noise relative to drift, so the ranking is recoverable and the test is meaningful.
    drifts = np.linspace(0.0020, -0.0015, n)
    noise = rng.normal(0, 0.002, size=(days, n))
    logret = drifts + noise
    close = pd.DataFrame(100 * np.exp(np.cumsum(logret, axis=0)), index=idx, columns=cols)
    volume = pd.DataFrame(rng.integers(1e6, 5e6, size=(days, n)), index=idx, columns=cols).astype(float)

    # Opens default to the prior close (zero gap everywhere), then inject a known +5%
    # overnight gap into T00 and a known 4x volume day into T01, so gap_pct and rvol are
    # checked against exact arithmetic rather than "looks about right".
    open_ = close.shift(1).bfill()
    open_.iloc[-1, 0] = close.iloc[-2, 0] * 1.05
    volume.iloc[-1, 1] = volume.iloc[-51:-1, 1].mean() * 4.0

    m = compute_metrics(close, volume, open_)

    checks = []
    checks.append(("all tickers scored", len(m) == n))
    checks.append(("score in [0,1]", bool(m["score"].between(0, 1).all())))
    checks.append(("no NaN scores", bool(m["score"].notna().all())))
    checks.append(("rank 1 is top score", int(m.iloc[0]["rank"]) == 1))
    # Score should recover the underlying drift ordering when noise is small.
    # Spearman correlation is just Pearson correlation of the RANKS, computed here
    # directly rather than via method="spearman" - pandas delegates that to scipy,
    # which this script does not otherwise need and which is absent on a clean runner.
    def spearman(a, b):
        return a.rank().corr(b.rank())

    drift_by_col = pd.Series(drifts, index=cols)
    drift_aligned = drift_by_col.reindex(m.index)
    corr = spearman(m["score"], drift_aligned)
    checks.append(("score recovers drift ordering (spearman %.2f > 0.70)" % corr, corr > 0.70))
    # Momentum components specifically should be near-monotonic in drift.
    corr_rs = spearman(m["rs_3m"], drift_aligned)
    checks.append(("rs_3m recovers drift ordering (spearman %.2f > 0.90)" % corr_rs, corr_rs > 0.90))
    checks.append(("rs_3m is a percentile", bool(m["rs_3m"].between(0, 1).all())))
    checks.append(("52w high proximity <= 0", bool((m["pct_from_52w_high"] <= 1e-9).all())))
    checks.append(("weights sum sane", abs(sum(WEIGHTS.values()) - 1.0) < 1e-9))

    # Rotation maths on a small frame
    rot = compute_rotation(close[cols[:5]])
    checks.append(("rotation returns 1 row per ticker", len(rot) == 5))

    # Short-window and 52w-low columns present and sane
    checks.append(("1d/1w returns computed",
                   m["ret_1d"].notna().all() and m["ret_1w"].notna().all()))
    checks.append(("52w low distance is never negative",
                   bool((m["pct_from_52w_low"] >= -1e-9).all())))

    # Extension: dist_to_50dma must be the exact inverse move, not the negated distance.
    pv, dv = m["pct_vs_50dma"], m["dist_to_50dma"]
    checks.append(("dist_to_50dma is the exact inverse move",
                   float((((1 + pv) * (1 + dv)) - 1).abs().max()) < 1e-12))
    checks.append(("dist_to_50dma is NOT just -pct_vs_50dma where extended",
                   bool(((pv.abs() > 0.02) & ((dv + pv).abs() > 1e-6)).any())))
    checks.append(("sign is opposite to pct_vs_50dma",
                   bool(((pv > 0) == (dv < 0)).all())))
    checks.append(("extension columns never enter the composite",
                   not any(k in WEIGHTS for k in ("pct_vs_50dma", "dist_to_50dma"))))
    checks.append(("EXT_THRESHOLD is a display filter, not a weight",
                   isinstance(EXT_THRESHOLD, float) and "EXT_THRESHOLD" not in str(WEIGHTS)))

    # Gap and relative volume: exact answers, and the failure mode that matters most -
    # a missing Open must produce NaN, never a silently wrong number.
    checks.append(("gap_pct recovers the injected +5% gap",
                   abs(m.loc["T00", "gap_pct"] - 0.05) < 1e-9))
    checks.append(("gap_up flags it at the 3% threshold", bool(m.loc["T00", "gap_up"])))
    checks.append(("no false gap_up on the flat names",
                   int(m["gap_up"].sum()) == 1))
    checks.append(("rvol recovers the injected 4x volume day",
                   abs(m.loc["T01", "rvol"] - 4.0) < 1e-9))
    m_noopen = compute_metrics(close, volume)
    checks.append(("gap_pct is NaN when no Open data is supplied",
                   bool(m_noopen["gap_pct"].isna().all())))
    checks.append(("no gap_up fires without Open data",
                   int(m_noopen["gap_up"].sum()) == 0))
    checks.append(("gap and rvol never enter the composite",
                   not any(k in WEIGHTS for k in ("gap_pct", "gap_up", "rvol"))))
    checks.append(("score is unchanged with and without Open data",
                   float((m["score"] - m_noopen["score"].reindex(m.index)).abs().max()) < 1e-12))

    # Breadth: a group where one name carries the move must show high concentration,
    # and one where every name moves together must show near-zero.
    bm = m.copy()
    bm["sector"] = ["Narrow"] * 30 + ["Broad"] * 30
    bm.loc[bm.index[:30], "ret_3m"] = [3.0] + [0.01] * 29      # one huge winner, rest flat
    bm.loc[bm.index[30:], "ret_3m"] = [0.10] * 30              # all identical
    br = compute_breadth(bm, "sector", min_names=3).set_index("sector")
    checks.append(("breadth flags the narrow group",
                   br.loc["Narrow", "concentration"] > 0.05))
    checks.append(("breadth shows ~zero concentration when uniform",
                   abs(br.loc["Broad", "concentration"]) < 1e-9))
    checks.append(("breadth skips groups below min_names",
                   len(compute_breadth(bm.head(2), "sector", min_names=3)) == 0))

    # Constituent-table picker: must skip decoy tables and select on columns, not page text.
    decoy = pd.DataFrame({"Ticker": ["AAA", "BBB"], "Company": ["a", "b"]})          # too few rows
    noise = pd.DataFrame({"Date": range(80), "Note": ["x"] * 80})                    # wrong columns
    real = pd.DataFrame({"Ticker": ["T%03d" % i for i in range(100)],
                         "Company": ["C%03d" % i for i in range(100)],
                         "ICB Industry": ["Tech"] * 100})
    picked, tcol = pick_constituent_table([decoy, noise, real], ["Ticker", "Symbol"], 50)
    checks.append(("picker skips short and irrelevant tables",
                   len(picked) == 100 and tcol == "Ticker"))
    try:
        pick_constituent_table([decoy, noise], ["Ticker", "Symbol"], 50)
        checks.append(("picker raises when nothing qualifies", False))
    except RuntimeError:
        checks.append(("picker raises when nothing qualifies", True))
    checks.append(("sector column falls back to ICB Industry",
                   _first_present(list(real.columns),
                                  ["GICS Sector", "ICB Industry"]) == "ICB Industry"))

    ok = True
    for name, passed in checks:
        print(("  PASS  " if passed else "  FAIL  ") + name)
        ok = ok and passed
    print("\nSELF-TEST %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=["both", "sp500", "ndx"], default="both")
    ap.add_argument("--top", type=int, default=40, help="rows to print to the terminal")
    ap.add_argument("--refresh-universe", action="store_true")
    ap.add_argument("--self-test", action="store_true", help="run offline maths checks and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    os.makedirs(OUTDIR, exist_ok=True)
    today = dt.date.today().isoformat()

    uni = get_universe(args.universe, refresh=args.refresh_universe)
    tickers = sorted(uni["ticker"].unique().tolist())

    print("Fetching prices...")
    close, volume, open_ = fetch_prices(tickers)
    print("Computing metrics on %d names, %d sessions..." % (close.shape[1], close.shape[0]))
    m = compute_metrics(close, volume, open_)

    join_cols = [c for c in ("name", "sector", "industry", "index") if c in uni.columns]
    m = m.join(uni.set_index("ticker")[join_cols])
    cols = ["rank", "score", "name", "sector", "industry", "index", "price",
            "ret_1d", "ret_1w", "ret_1m", "ret_3m", "ret_6m",
            "rs_1m", "rs_3m", "rs_6m", "rs_3m_chg",
            "pct_from_52w_high", "pct_from_52w_low", "new_52w_high",
            "pct_vs_50dma", "pct_vs_200dma", "dist_to_50dma",
            "above_50dma", "above_200dma", "vol_ratio_20_60", "volume_x",
            "gap_pct", "gap_up", "rvol"]
    m = m[[c for c in cols if c in m.columns]]

    print("Fetching rotation ETFs...")
    etf_close, _, _ = fetch_prices(list(ROTATION_ETFS.keys()))
    rot = compute_rotation(etf_close)

    sector_breadth = compute_breadth(m, "sector", min_names=3)
    industry_breadth = (compute_breadth(m, "industry", min_names=4)
                        if "industry" in m.columns else pd.DataFrame())

    cand_path = os.path.join(OUTDIR, "candidates_%s.csv" % today)
    rot_path = os.path.join(OUTDIR, "rotation_%s.csv" % today)
    m.to_csv(cand_path)
    rot.to_csv(rot_path, index=False)
    m.to_csv(os.path.join(OUTDIR, "latest_candidates.csv"))
    rot.to_csv(os.path.join(OUTDIR, "latest_rotation.csv"), index=False)
    for df, stem in ((sector_breadth, "sector_breadth"), (industry_breadth, "industry_breadth")):
        if len(df):
            df.to_csv(os.path.join(OUTDIR, "%s_%s.csv" % (stem, today)), index=False)
            df.to_csv(os.path.join(OUTDIR, "latest_%s.csv" % stem), index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== TOP %d BY COMPOSITE ===" % args.top)
    show = ["rank", "score", "name", "sector", "ret_3m", "rs_3m", "rs_3m_chg",
            "pct_from_52w_high", "pct_vs_50dma", "vol_ratio_20_60", "volume_x"]
    print(m.head(args.top)[[c for c in show if c in m.columns]].round(3).to_string())

    print("\n=== BIGGEST MOVES YESTERDAY (top 15 by 1-day return) ===")
    mv = ["name", "sector", "ret_1d", "ret_1w", "volume_x", "rank"]
    print(m.sort_values("ret_1d", ascending=False).head(15)[
        [c for c in mv if c in m.columns]].round(3).to_string())

    if "pct_vs_50dma" in m.columns:
        top = m.head(args.top)
        med = float(top["pct_vs_50dma"].median())
        over = int((top["pct_vs_50dma"] >= EXT_THRESHOLD).sum())
        ext = top.sort_values("pct_vs_50dma", ascending=False).head(EXT_SHOW)
        print("\n=== MOST EXTENDED OF THE TOP %d BY COMPOSITE ===" % args.top)
        print("  The composite scores STRENGTH. It says nothing about how far price has run")
        print("  from its own mean, and a momentum screen selects for that by construction:")
        print("  median extension in this shortlist is %+.1f%%, and %d of %d are >=%.0f%% above."
              % (med * 100, over, len(top), EXT_THRESHOLD * 100))
        print("  So judge these against the median, not against zero. dist_to_50dma is the")
        print("  move needed to get back to the average - which would break no trend at all.")
        print("  A good trend and a good entry are different questions; this column is the second.")
        ev = ["name", "sector", "rank", "score", "pct_vs_50dma", "dist_to_50dma",
              "pct_vs_200dma", "pct_from_52w_high", "vol_ratio_20_60"]
        print(ext[[c for c in ev if c in ext.columns]].round(3).to_string())

    if "gap_pct" in m.columns and m["gap_pct"].notna().any():
        print("\n=== GAPPED UP >=%.0f%% ON THE LAST COMPLETED SESSION (sorted by rvol) ==="
              % (GAP_THRESHOLD * 100))
        print("  That session's OPEN vs the prior CLOSE. Not a live pre-market gap -")
        print("  on a weekend run this is Friday's gap and may already be closed.")
        gappers = m[m["gap_up"].fillna(False)].sort_values("rvol", ascending=False)
        if len(gappers):
            gv = ["name", "sector", "gap_pct", "rvol", "ret_1d", "pct_from_52w_high", "rank"]
            print(gappers.head(15)[[c for c in gv if c in gappers.columns]].round(3).to_string())
        else:
            print("  none")

    print("\n=== SECTOR / INDEX ROTATION (ETFs, 3m sorted) ===")
    print(rot.round(4).to_string(index=False))

    if len(sector_breadth):
        print("\n=== SECTOR BREADTH (the MEDIAN stock, not the cap-weighted ETF) ===")
        print("  concentration = mean minus median 3m return; large positive = a few names carrying it")
        print(sector_breadth.round(3).to_string(index=False))

    if len(industry_breadth):
        print("\n=== STRONGEST INDUSTRIES (>=4 names, by median 3m) ===")
        print(industry_breadth.head(12).round(3).to_string(index=False))
        print("\n=== WEAKEST INDUSTRIES ===")
        print(industry_breadth.tail(8).round(3).to_string(index=False))

    print("\nWritten:\n  %s\n  %s" % (cand_path, rot_path))
    print("\nThese are CANDIDATES, not decisions. Most breakouts fail. Next step is to hand")
    print("output/latest_candidates.csv to Claude for the interrogation pass - catalyst,")
    print("bear case, invalidation level - on a handful of names, not the whole list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
