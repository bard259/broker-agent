import os
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import yfinance as yf
import praw
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from psaw import PushshiftAPI


# Try Reddit & VADER sentiment
_HAS_PRAW = False
_HAS_VADER = False
try:
    import praw
    _HAS_PRAW = True
except:
    pass

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    _vader = SentimentIntensityAnalyzer()
    _HAS_VADER = True
except:
    _vader = None


def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    ups = delta.clip(lower=0)
    downs = -delta.clip(upper=0)
    roll_up = ups.ewm(alpha=1/period, adjust=False).mean()
    roll_down = downs.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _fetch_prices(ticker: str, lookback_days: int = 90) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days + 5)
    df = yf.download(ticker, start=start.date(), end=end.date(), interval="1d", progress=False, auto_adjust=False, group_by='column')
    if df.empty:
        raise RuntimeError(f"No price data for {ticker}")
    # df = df.rename(columns=str.lower)
    # return df[["open", "high", "low", "close", "volume"]].dropna()
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def _init_reddit():
    if not _HAS_PRAW:
        return None
    needed = ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT")
    if not all(k in os.environ and os.environ[k] for k in needed):
        return None
    try:
        reddit = praw.Reddit(
            client_id=os.environ["REDDIT_CLIENT_ID"],
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=os.environ["REDDIT_USER_AGENT"],
        )
        reddit.read_only = True
        return reddit
    except:
        return None


def _fetch_reddit_texts(ticker: str, hours: int = 36, max_posts: int = 100) -> list:
    reddit = _init_reddit()
    if reddit is None:
        return []
    subs = ["stocks", "investing", "wallstreetbets", "StockMarket"]
    query = f'"{ticker}" OR "${ticker}"'
    cutoff = time.time() - hours * 3600

    texts = []
    for s in subs:
        try:
            for p in reddit.subreddit(s).search(query=query, sort="new", limit=max_posts // len(subs)):
                if getattr(p, "created_utc", 0) >= cutoff:
                    txt = (getattr(p, "title", "") or "") + " " + (getattr(p, "selftext", "") or "")
                    if txt.strip():
                        texts.append(txt.strip())
        except:
            continue
    return texts


def _score_sentiment(texts: list) -> dict:
    if not _HAS_VADER or not texts:
        return {"mean": 0.0, "count": 0, "pos_share": 0.0}
    scores, pos = [], 0
    for t in texts:
        sc = _vader.polarity_scores(t)
        scores.append(sc["compound"])
        if sc["compound"] >= 0.05:
            pos += 1
    return {
        "mean": float(np.mean(scores)) if scores else 0.0,
        "count": len(scores),
        "pos_share": pos / len(scores) if scores else 0.0,
    }


def _build_features(ticker: str) -> dict:
    prices = _fetch_prices(ticker)
    close = prices["Close"]

    r1 = float((close.iloc[-1] / close.iloc[-2] - 1.0)) if len(close) >= 2 else 0.0 # 1-day return
    r5 = float((close.iloc[-1] / close.iloc[-6] - 1.0)) if len(close) >= 6 else 0.0 #5-day return
    rsi14 = float(_rsi(close, period=14)) #14-day Relative Strength Index

    texts = _fetch_reddit_texts(ticker, hours=36, max_posts=100)
    senti = _score_sentiment(texts)

    return {
        "r1": float(r1),
        "r5": float(r5),
        "rsi14": float(rsi14),
        "senti_mean": float(senti.get("mean", 0.0)), #Average sentiment score across posts mentioning the ticker (range: −1 = very negative, +1 = very positive)
        "senti_count": int(senti.get("count", 0)), #Number of posts analyzed in the sample
        "senti_pos_share": float(senti.get("pos_share", 0.0)), #Share of posts with positive sentiment (compound ≥ 0.05)
    }

def _decide_row_from_feats(row: pd.Series) -> str:
    # Defensive casts to ensure pure Python/numpy scalars
    rsi   = float(row["rsi14"])
    sm    = float(row["senti_mean"])
    spos  = float(row["senti_pos_share"])
    r5    = float(row["r5"])
    scnt  = int(row["senti_count"])

    overbought = rsi >= 70.0
    oversold   = rsi <= 30.0

    BUY_SM_THR   =  0.10
    SHORT_SM_THR = -0.10
    # small momentum guards
    MOM_THR_UP = -0.02
    MOM_THR_DN =  0.02

    # BUY: positive sentiment, not overbought, and benign to positive recent trend
    if (sm >= BUY_SM_THR) and (not overbought) and ((r5 > MOM_THR_UP) or (spos >= 0.60)):
        return "buy"

    # SHORT: negative sentiment, not oversold, and weak recent trend
    if (sm <= SHORT_SM_THR) and (not oversold) and ((r5 < -MOM_THR_DN) or (spos <= 0.35)):
        return "short"

    # Fallback: no sentiment data → simple technicals
    if scnt == 0:
        if (r5 > 0.02) and (40.0 < rsi < 68.0):
            return "buy"
        if (r5 < -0.02) and (32.0 < rsi < 60.0):
            return "short"

    return "hold"

def consult(ticker: str, explain: bool = False):
    """
    consult('NVDA') -> 'buy' | 'hold' | 'short'
    consult('NVDA', explain=True) -> (decision, explanation_dict)
    """
    feats = _build_features(ticker.upper())
    decision, rationale = _decide_with_explain(feats)
    if explain:
        return decision, {"features": feats, "rationale": rationale}
    else:
        return decision


def _decide_with_explain(feats: dict):
    # Cast to scalars
    rsi   = float(feats["rsi14"])
    sm    = float(feats["senti_mean"])
    spos  = float(feats["senti_pos_share"])
    r5    = float(feats["r5"])
    scnt  = int(feats["senti_count"])

    overbought = rsi >= 70.0
    oversold   = rsi <= 30.0

    rationale = []

    # BUY
    if (sm >= 0.10) and (not overbought) and ((r5 > -0.02) or (spos >= 0.60)):
        rationale.append(f"Positive sentiment (mean={sm:.2f}, pos_share={spos:.2f}), RSI={rsi:.1f} not overbought, r5={r5:.2%}")
        return "buy", rationale

    # SHORT
    if (sm <= -0.10) and (not oversold) and ((r5 < -0.02) or (spos <= 0.35)):
        rationale.append(f"Negative sentiment (mean={sm:.2f}, pos_share={spos:.2f}), RSI={rsi:.1f} not oversold, r5={r5:.2%}")
        return "short", rationale

    # Fallback
    if scnt == 0:
        if (r5 > 0.02) and (40.0 < rsi < 68.0):
            rationale.append(f"No sentiment, technicals uptrend: r5={r5:.2%}, RSI={rsi:.1f}")
            return "buy", rationale
        if (r5 < -0.02) and (32.0 < rsi < 60.0):
            rationale.append(f"No sentiment, technicals downtrend: r5={r5:.2%}, RSI={rsi:.1f}")
            return "short", rationale

    rationale.append("Conditions mixed or weak → default HOLD")
    return "hold", rationale

# === PSAW functions ===

def _utc_ts(dt_str):
    # 'YYYY-MM-DD' -> unix epoch
    return int(pd.Timestamp(dt_str, tz='UTC').timestamp())

def _text_ok(s):
    return isinstance(s, str) and len(s.strip()) > 0

    def _score_texts_to_row(texts, day):
        s = _score_sentiment(texts)  # uses your VADER helper
        return dict(date=pd.to_datetime(day), senti_mean=s['mean'], senti_pos_share=s['pos_share'], senti_count=s['count'])

    # We iterate day by day to avoid massive result sets and to keep memory low
    day_index = pd.date_range(start=start_date, end=end_date, tz='UTC', freq='D')
    for d0 in day_index:
        d1 = d0 + pd.Timedelta(days=1)
        after  = int(d0.timestamp())
        before = int(d1.timestamp())

        collected = []
        for kind in kinds:
            for sub in subreddits:
                # PSAW query (submissions or comments)
                if kind == 'submission':
                    gen = api.search_submissions(
                        after=after, before=before,
                        q=query_extra or None,
                        subreddit=sub, filter=['title','selftext'], limit=None
                    )
                    for s in gen:
                        title = getattr(s, 'title', '') or ''
                        body  = getattr(s, 'selftext', '') or ''
                        txt = (title + ' ' + body).strip()
                        if _text_ok(txt) and pat.search(txt):
                            collected.append(txt)
                            if max_per_day and len(collected) >= max_per_day:
                                break
                    # small backoff
                    time.sleep(rate_sleep)
                else:  # comments
                    gen = api.search_comments(
                        after=after, before=before,
                        q=query_extra or None,
                        subreddit=sub, filter=['body'], limit=None
                    )
                    for c in gen:
                        body = getattr(c, 'body', '') or ''
                        if _text_ok(body) and pat.search(body):
                            collected.append(body)
                            if max_per_day and len(collected) >= max_per_day:
                                break
                    time.sleep(rate_sleep)

        if collected:
            rows.append(_score_texts_to_row(collected, d0.date()))
        else:
            # record an explicit zero-info day (optional but helpful)
            rows.append(dict(date=pd.to_datetime(d0.date()), senti_mean=0.0, senti_pos_share=0.0, senti_count=0))

    df = pd.DataFrame(rows).set_index('date').sort_index()
    # Drop timezone (simulate() expects naive dates)
    df.index = df.index.tz_localize(None)
    return df
import math
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
from .consultant import _decide_row_from_feats, _score_sentiment # Import necessary functions from consultant.py


# =========================
# Helpers to normalize yfinance frames
# =========================


_FIELD_NAMES = {"Open","High","Low","Close","Adj Close","Volume"}

def _normalize_yf(px: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """
    Normalize yfinance DataFrame to:
      - Single-level DatetimeIndex
      - Columns: Open, High, Low, Close, Volume (Close falls back to Adj Close if needed)
    Handles:
      * Single-level columns already fine
      * MultiIndex with (ticker, field)
      * MultiIndex with (field, ticker)
    """
    df = px.copy()

    # 1) Ensure DatetimeIndex (take first level if MultiIndex index)
    if isinstance(df.index, pd.MultiIndex):
        df.index = pd.to_datetime(df.index.get_level_values(0))
    else:
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # 2) Flatten/standardize columns
    if isinstance(df.columns, pd.MultiIndex):
        # find which level holds field names
        lvl_with_fields = None
        for lvl in range(df.columns.nlevels):
            vals = set(map(str, df.columns.get_level_values(lvl)))
            if _FIELD_NAMES & vals:
                lvl_with_fields = lvl
                break

        if lvl_with_fields is not None:
            # Build a new frame by selecting each field across that level
            parts = {}
            for f in ["Open","High","Low","Close","Adj Close","Volume"]:
                if f in set(map(str, df.columns.get_level_values(lvl_with_fields))):
                    try:
                        sel = df.xs(f, axis=1, level=lvl_with_fields)
                        # If xs returns a Series for some reason, make it a 1-col DF
                        if isinstance(sel, pd.Series):
                            sel = sel.to_frame(f)
                        # If there are multiple tickers, pick the first column
                        if isinstance(sel.columns, pd.MultiIndex):
                            # flatten again if weird
                            sel.columns = [c[-1] for c in sel.columns]
                        if sel.shape[1] > 1:
                            # prefer the requested ticker if present
                            if ticker and ticker in sel.columns:
                                parts[f] = sel[ticker]
                            else:
                                parts[f] = sel.iloc[:, 0]
                        else:
                            parts[f] = sel.iloc[:, 0]
                    except Exception:
                        pass
            df = pd.DataFrame(parts, index=df.index)
        else:
            # No level clearly contains field names -> flatten by last level
            df.columns = [str(c[-1]) for c in df.columns]

    # 3) Standardize column names (case-insensitive)
    lower_map = {c: str(c).strip().lower().replace(" ", "") for c in df.columns}
    df = df.rename(columns=lower_map)

    # 4) Create canonical frame
    out = pd.DataFrame(index=df.index)
    def pick(*names):
        for n in names:
            if n in df.columns:
                return df[n]
        return pd.Series(np.nan, index=df.index)

    out["Open"]   = pick("open")
    out["High"]   = pick("high")
    out["Low"]    = pick("low")
    # prefer close; fall back to adjclose
    close_series  = pick("close")
    if close_series.isna().all():
        close_series = pick("adjclose","adjustedclose","adjustclose")
    out["Close"]  = close_series
    out["Volume"] = pick("volume")

    # 5) Validate & drop rows with any missing
    missing_all = [c for c in ["Open","High","Low","Close","Volume"] if out[c].isna().all()]
    if missing_all:
        raise RuntimeError(
            f"Missing required price columns after normalization: {missing_all}. "
            f"Raw columns: {list(px.columns)}"
        )

    out = out.dropna(how="any")
    return out

# =========================
# Feature engineering
# =========================

def _compute_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    ups = delta.clip(lower=0)
    downs = -delta.clip(upper=0)
    roll_up = ups.ewm(alpha=1/period, adjust=False).mean()
    roll_down = downs.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _build_features_historical(px: pd.DataFrame) -> pd.DataFrame:
    """Build features as-of t-1 close, to decide at day t open."""
    close = px["Close"]
    feats = pd.DataFrame(index=px.index)
    feats["r1"] = (close / close.shift(1) - 1).shift(1)
    feats["r5"] = (close / close.shift(5) - 1).shift(1)
    feats["rsi14"] = _compute_rsi_series(close, 14).shift(1)
    return feats

# =========================
# Core backtest (single ticker)
# =========================

def simulate(
    ticker: str,
    start: str = "2019-01-01",
    end: str | None = None,
    fraction_per_trade: float = 0.10,
    cost_bps: float = 5.0,
    initial_cash: float = 100_000.0,
    allow_short: bool = True,
    senti_daily: pd.DataFrame | None = None,
):
    """
    Backtest one ticker with your decision rules.
    - Decide at day t open using features through t-1 close
    - Trade to a target exposure of ±fraction_per_trade * equity at the OPEN
    - Mark P&L at the CLOSE
    - cost_bps applied per side on traded notional
    - senti_daily (optional): index=Date, columns ['senti_mean','senti_pos_share','senti_count'].
      It is shifted by 1 day internally to avoid look-ahead.

    Returns: (summary: dict, ledger: pd.DataFrame)
    """
    end = end or datetime.now().strftime("%Y-%m-%d")
    # Explicitly set auto_adjust=False and group_by='column' to avoid surprises
    px = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=False, group_by='column')
    if px.empty:
        raise RuntimeError(f"No price data for {ticker} in range {start}..{end}")
    px = _normalize_yf(px, ticker)

    feats = _build_features_historical(px)

    if senti_daily is not None:
        sd = senti_daily.copy()
        # normalize index
        if isinstance(sd.index, pd.MultiIndex):
            sd.index = pd.to_datetime(sd.index.get_level_values(0))
        else:
            sd.index = pd.to_datetime(sd.index)
        sd = sd.sort_index()
        # Ensure expected columns
        for c in ["senti_mean","senti_pos_share","senti_count"]:
            if c not in sd.columns:
                sd[c] = np.nan
        # Only data up to t-1 can be used at t
        feats = feats.join(sd[["senti_mean","senti_pos_share","senti_count"]].shift(1), how="left")

    df = px.join(feats, how="left")

    cash = initial_cash
    shares = 0.0
    prev_equity = initial_cash
    rows = []

    for dt, row in df.iterrows():
        open_p  = float(row["Open"])
        close_p = float(row["Close"])

        # Decide at today's OPEN
        action = _decide_row_from_feats(row)

        # Compute target shares
        equity_mark = cash + shares * close_p
        target_shares = shares
        if action == "buy":
            target_val = fraction_per_trade * equity_mark
            target_shares = math.floor(target_val / open_p)
        elif action == "short" and allow_short:
            target_val = -fraction_per_trade * equity_mark
            target_shares = -math.floor(abs(target_val) / open_p)

        # Execute once at OPEN
        delta = target_shares - shares
        if delta != 0:
            fees = abs(delta) * open_p * (cost_bps / 10_000.0)
            cash -= delta * open_p
            cash -= fees
            shares = target_shares

        # Mark at CLOSE
        equity = cash + shares * close_p
        daily_ret = (equity / prev_equity - 1.0) if prev_equity > 0 else 0.0
        rows.append({
            "date": dt, "action": action, "open": open_p, "close": close_p,
            "shares": shares, "cash": cash, "equity": equity, "ret": daily_ret
        })
        prev_equity = equity

    ledger = pd.DataFrame(rows).set_index("date")
    daily = ledger["ret"].fillna(0.0)
    ann = 252.0
    total_ret = ledger["equity"].iloc[-1] / initial_cash - 1.0
    cagr = (1 + total_ret) ** (ann / max(1, len(daily))) - 1.0
    sharpe = (daily.mean() / (daily.std() + 1e-12)) * np.sqrt(ann) if daily.std() > 0 else np.nan

    # Max drawdown
    peaks = ledger["equity"].cummax()
    dd = ledger["equity"] / peaks - 1.0
    mdd = dd.min()
    mdd_end = dd.idxmin()
    mdd_start = ledger["equity"].loc[:mdd_end].idxmax()

    summary = {
        "ticker": ticker,
        "final_equity": float(ledger["equity"].iloc[-1]),
        "total_return": float(total_ret),
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "MaxDrawdown": float(mdd),
        "MDD_start": mdd_start,
        "MDD_end": mdd_end,
        "trading_days": int(len(ledger)),
        "fraction_per_trade": fraction_per_trade,
        "cost_bps": cost_bps,
    }
    return summary, ledger

# Convenience runner

def simulate_and_show(ticker="NVDA", start="2019-01-01"):
    s, ledg = simulate(ticker, start=start)
    print("Summary:", {k: (round(v,4) if isinstance(v, float) else v) for k,v in s.items()})
    display(ledg.tail(10))
    return s, ledg

def run_demo(ticker="NVDA", start="2019-01-01"):
    """
    Runs a demo of the stock consultant and backtesting for a given ticker.
    """
    print(f"--- Running Demo for {ticker.upper()} ---")

    # Consult
    decision, explanation = consult(ticker, explain=True)
    print("\nConsult Decision:", decision)
    print("Consult Features:", explanation["features"])
    print("Consult Rationale:", explanation["rationale"])

    # Simulate
    print(f"\n--- Running Simulation for {ticker.upper()} from {start} ---")
    simulate_and_show(ticker, start=start)

    print(f"\n--- Demo for {ticker.upper()} Complete ---")
