# CONTEXT — read this first

This repo is the deterministic half of a personal, AI-assisted stock research system for
**US swing trading (days to ~1 year), executed manually through Interactive Brokers**.

It exists so that a Claude session with **no access to the owner's machine** — a session started
on mobile or web, which runs entirely in the cloud — can pick up the rules in one fetch instead of
rediscovering them badly. Read this before doing any analysis for this project.

Nothing here is advice, and nothing here is a position. This file documents *method*, not holdings.

---

## 1. The rule that governs everything

**Deterministic Python surfaces candidates. Claude interrogates the shortlist. The human decides.**

- Never rank names by "attractiveness" — that is prediction wearing a different hat.
- Never answer "what looks promising?" from model knowledge. Answer it from `output/`.
- The screener has no opinion and must not be given one.

## 2. Hard constraints — not negotiable

- **No investment advice. No buy or sell recommendations.** Assessment only.
- **No auto-execution, ever**, even where a broker connector is available.
- **No handling of credentials, API keys or personal access tokens.**
- Every claim traceable to a filing, a named source, or a tool call. Unsourced claims are
  **dropped, not softened**.
- Every analysis ends with **what could not be established**. That section is the point, not a
  disclaimer — it is what makes the rest trustworthy.

## 3. What is in this repo

| Path | What it is |
|---|---|
| `screener.py` | The whole screen. Self-contained, no judgement, writes CSVs. |
| `output/latest_candidates.csv` | Newest full ranked universe (~513 names). |
| `output/latest_rotation.csv` | Sector / index ETF relative strength. |
| `output/latest_sector_breadth.csv` | Median-stock breadth per sector. |
| `output/candidates_YYYY-MM-DD.csv` | **Dated archives — the forward-test record.** |
| `.github/workflows/screener.yml` | Runs the screen on GitHub Actions, weekdays. |

**Read the CSVs with an unauthenticated fetch of the `raw.githubusercontent.com` URL.** That works
from any cloud session with no local machine. `raw` caches aggressively for recently committed
files — add a cache-busting query parameter and a `Cache-Control: no-cache` header when freshness
matters. The GitHub API is often unavailable to these sessions; the raw host generally is not.

`screener.py --self-test` runs the maths offline against synthetic data with known answers.
**Run it after any change.** It is the only guard against a silently wrong column.

## 4. How to read the screen — the part that is easy to get wrong

The composite is **~85% momentum** (relative strength over 1/3/6 months, whether that strength is
improving, and proximity to the 52-week high), 10% volatility contraction, 5% volume. It is
effectively a **single-factor momentum screen**. It contains no value, quality, growth, size or
estimate-revision factor, and it **has never read a financial statement**.

**SCORED vs REPORTED is the governing distinction.** Only the columns in `WEIGHTS` move the
composite. These are written to the CSV and score *nothing*:

| Column | What it answers |
|---|---|
| `ret_1d`, `ret_1w` | what moved recently — a different question from what is strong |
| `pct_from_52w_low` | is this a fresh base or a mature recovery? Momentum hides this |
| `pct_vs_50dma` | how extended is it |
| `dist_to_50dma` | the move needed to get *back* to the average — **not** `pct_vs_50dma` negated |
| `gap_pct`, `gap_up` | last completed session's open vs prior close |
| `rvol` | that session's volume vs the 50 sessions *before* it |

Adding a column is cheap. **Adding a weight changes what the screen is** and requires evidence from
the dated archives first.

Two traps this design encodes, both learned the hard way:

- **A high rank is not a good entry.** A momentum screen selects for extension by construction —
  on one recent run, 31 of the top 40 sat ≥10% above their 50-day average, median +13.5%. So judge
  extension *against that median*, not against zero, and read `dist_to_50dma` before calling
  anything a setup. A stock can have an excellent trend and a poor price on the same day.
- **`gap_pct` is not a pre-market gap.** Daily bars end at the close. On a Monday run it is
  Friday's gap and may already have closed. Never imply otherwise.

Known weakness of the whole approach: momentum fails hardest at market bottoms. See
[Daniel & Moskowitz](https://www.nber.org/papers/w20439) — 14 of the 15 worst momentum months came
when the trailing two-year market return was negative, all in months the market rose.

## 5. Data sources — each has exactly one job

- **SEC EDGAR — all fundamentals.** Free, unlimited, authoritative. Ticker→CIK from
  `sec.gov/files/company_tickers.json`, **zero-padded to 10 digits**; figures from
  `data.sec.gov/api/xbrl/companyconcept/CIK##########/us-gaap/<Tag>.json`.
  - **Trap:** `filings.recent` is **parallel arrays**, not a list of objects. Pair `form[]`,
    `filingDate[]` and `reportDate[]` **by index** or dates attach to the wrong filings. A summarised
    read of that endpoint once produced a 10-K "filed" eight days before its own period ended.
    The per-fact `filed` field is self-consistent — prefer it.
  - **Trap:** tag coverage is uneven and can be years stale. **Check the date on every value.**
  - **Results are announced by 8-K before the 10-Q appears.** A quarter can be reported and public
    while its 10-Q is not yet filed — in which case say the figure is derived or second-hand.
- **A broker connector — all price and portfolio data.** Bars, snapshots with 13/26/52-week
  extremes, option chains, alerts. **No fundamentals at all.** Snapshot calls are one per contract,
  so any "scan the universe" idea belongs in `screener.py`, never in a loop of tool calls.
- **A market-data API — consensus estimates.** EPS and revenue consensus by quarter and fiscal year,
  analyst counts, revision counts, price targets, rating buckets.
  - **Trap: mixed vintages inside one response.** A single payload has returned a *stale* market cap
    and price-to-sales alongside a *current* P/E. **Recompute every ratio from the broker's closing
    price.** Taking one such feed at face value would have overstated a valuation by a third.
  - **Trap:** two endpoints can give different EPS estimates for the same quarter (likely GAAP vs
    non-GAAP). Always say which endpoint a number came from; never mix them in one comparison.
  - **Free tiers are rationed.** Spend the daily allowance on single names the human asks about —
    never on discovery, never looping over a candidate list. A source that stops answering silently
    is worse than not having one.

**Check the actual tool list; do not reason from what a vendor is famous for.** That mistake has
been made three times on this project and been wrong three times.

## 6. Evidence constraints

- **Lookahead bias:** an LLM "forecasting" a pre-cutoff date is reciting, not forecasting
  (~32% inflated, and the effect vanishes after the cutoff). **Never backtest a judgement-based
  signal on pre-cutoff history.** The dated `candidates_*.csv` archives are the only honest forward
  test this system has.
- The Kim/Muhn/Nikolaev paper (arXiv 2407.17866, "GPT-4 beats analysts") is **withdrawn**. Do not
  cite it.
- **Frictions are real:** ~10bp a day turns a 113%/yr paper return into 65%. Cap discovery at
  roughly 5–8 candidates a week; churn is the enemy.

## 7. What a cloud-only session can and cannot do

**Can:** read everything in this repo; pull filings from EDGAR; use broker and market-data
connectors; run a full single-name analysis end to end.

**Cannot:** reach the owner's machine — no local runtime folder, no working notes, no watchlist, no
thesis files, and **no project memory**. Deliver results into the conversation and say plainly that
they were not filed anywhere durable.

If an analysis matters, it should be redone or saved from a desktop-started session, where the
written record lives.
