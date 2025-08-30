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

def backfill_senti_psaw(
    ticker: str,
    start_date: str,          # '2019-01-01'
    end_date: str,            # '2021-12-31'
    subreddits=('stocks','investing','wallstreetbets','StockMarket'),
    kinds=('submission',),    # ('submission','comment') to include comments
    query_extra=None,         # e.g. '"NVDA" OR "$NVDA"'
    rate_sleep=1.0,           # seconds between pages
    max_per_day=None          # optional cap per day to control runtime
) -> pd.DataFrame:
    """
    Returns a daily sentiment DataFrame indexed by date with columns:
      ['senti_mean','senti_pos_share','senti_count']
    Uses your existing VADER via _score_sentiment().
    """
    api = PushshiftAPI()
    cashtag = f"${ticker.upper()}"
    bare    = ticker.upper()
    # Build a regex we apply locally to text to reduce false positives
    pat = re.compile(rf'(^|\W)({re.escape(bare)}|{re.escape(cashtag)})($|\W)', re.IGNORECASE)

    start_ts = _utc_ts(start_date)
    end_ts   = _utc_ts(end_date)

    rows = []

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
