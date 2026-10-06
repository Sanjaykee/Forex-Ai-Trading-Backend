import json
import urllib.request
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

PAIRS = {
    "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"), "USDJPY": ("USD", "JPY"),
    "USDCHF": ("USD", "CHF"), "AUDUSD": ("AUD", "USD"), "NZDUSD": ("NZD", "USD"),
    "USDCAD": ("USD", "CAD"), "EURJPY": ("EUR", "JPY"), "GBPJPY": ("GBP", "JPY"),
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def _fetch_yahoo_chart(symbol: str, interval: str = "1m", range_str: str = "1d") -> dict:
    """Fetch raw JSON chart from Yahoo Finance API."""
    ticker = f"{symbol}=X"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&range={range_str}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=6) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_yahoo_live_price(symbol: str) -> dict:
    """Fetch real-time Forex tick price from Yahoo Finance."""
    try:
        data = _fetch_yahoo_chart(symbol, interval="1m", range_str="1d")
        result = data["chart"]["result"][0]
        meta = result.get("meta", {})
        price = meta.get("regularMarketPrice")

        if not price or price <= 0:
            quotes = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])
            valid_quotes = [q for q in quotes if q is not None]
            if valid_quotes:
                price = valid_quotes[-1]

        if not price or price <= 0:
            return {}

        pip_spread = 0.00010 if "JPY" not in symbol else 0.010
        bid = round(float(price), 5 if "JPY" not in symbol else 3)
        ask = round(bid + pip_spread, 5 if "JPY" not in symbol else 3)
        return {"bid": bid, "ask": ask, "spread": 1.0}
    except Exception as e:
        logger.debug(f"[YahooFetcher] Failed to get live price for {symbol}: {e}")
        return {}


def _get_yahoo_candles(symbol: str, timeframe: str, count: int = 200) -> pd.DataFrame:
    """Fetch genuine historical candles from Yahoo Finance."""
    try:
        if timeframe in ("M1", "M3", "M5"):
            interval, range_str = "5m", "2d"
        elif timeframe == "M15":
            interval, range_str = "15m", "5d"
        elif timeframe == "H1":
            interval, range_str = "1h", "1mo"
        elif timeframe == "H4":
            interval, range_str = "1h", "3mo"
        else:
            interval, range_str = "1d", "1y"

        data = _fetch_yahoo_chart(symbol, interval=interval, range_str=range_str)
        result = data["chart"]["result"][0]
        timestamps = result.get("timestamp", [])
        quote = result.get("indicators", {}).get("quote", [{}])[0]

        opens = quote.get("open", [])
        highs = quote.get("high", [])
        lows = quote.get("low", [])
        closes = quote.get("close", [])
        volumes = quote.get("volume", [])

        rows = []
        for ts, o, h, l, c, v in zip(timestamps, opens, highs, lows, closes, volumes):
            if None not in (o, h, l, c):
                rows.append({
                    "time": datetime.utcfromtimestamp(ts),
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v) if v is not None else 1000.0
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if timeframe == "H4":
            # Resample 1h candles into 4h candles
            df = df.set_index("time").resample("4h").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna().reset_index()

        return df.tail(count).reset_index(drop=True)
    except Exception as e:
        logger.debug(f"[YahooFetcher] Failed to get candles for {symbol} ({timeframe}): {e}")
        return pd.DataFrame()


# ==============================================================
# FALLBACK: Frankfurter API (Daily European Central Bank Rates)
# ==============================================================

def _get_rate_frankfurter(base: str, quote: str) -> float:
    try:
        r = requests.get(f"https://api.frankfurter.app/latest?from={base}&to={quote}", timeout=5)
        data = r.json()
        return float(data["rates"][quote])
    except Exception:
        return 0.0


def _get_historical_frankfurter(base: str, quote: str, days: int = 60) -> pd.DataFrame:
    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=days)
        r = requests.get(
            f"https://api.frankfurter.app/{start}..{end}?from={base}&to={quote}",
            timeout=10
        )
        data = r.json()
        if "rates" not in data:
            return pd.DataFrame()
        rows = []
        for date_str, rates in sorted(data["rates"].items()):
            price = rates.get(quote)
            if price:
                rows.append({"time": pd.Timestamp(date_str), "close": float(price)})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df["close"] * 1.0005
        df["low"] = df["close"] * 0.9995
        df["volume"] = 1000.0
        return df[["time", "open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


# ==============================================================
# PUBLIC API FUNCTIONS
# ==============================================================

def get_live_price(symbol: str) -> dict:
    """Returns real-time tick price from Yahoo Finance, falling back to Frankfurter."""
    # 1. Primary: Yahoo Finance Real-Time API
    y_price = _get_yahoo_live_price(symbol)
    if y_price:
        return y_price

    # 2. Fallback: Frankfurter API
    pair = PAIRS.get(symbol)
    if not pair:
        return {}
    base, quote = pair
    price = _get_rate_frankfurter(base, quote)
    if price == 0.0:
        return {}
    offset = 0.00010 if "JPY" not in symbol else 0.010
    return {"bid": round(price, 5), "ask": round(price + offset, 5), "spread": 1.0}


def get_candles(symbol: str, timeframe: str, count: int = 200) -> pd.DataFrame:
    """Returns historical candles from Yahoo Finance, falling back to Frankfurter."""
    # 1. Primary: Genuine candles from Yahoo Finance
    df = _get_yahoo_candles(symbol, timeframe, count)
    if not df.empty and len(df) >= 10:
        return df

    # 2. Fallback: Frankfurter
    pair = PAIRS.get(symbol)
    if not pair:
        return pd.DataFrame()
    base, quote = pair
    df = _get_historical_frankfurter(base, quote, days=365)
    return df.tail(count).reset_index(drop=True) if not df.empty else pd.DataFrame()


def get_symbol_info(symbol: str) -> dict:
    point = 0.0001 if "JPY" not in symbol else 0.01
    return {
        "point": point,
        "digits": 5 if "JPY" not in symbol else 3,
        "trade_contract_size": 100000,
        "volume_min": 0.01,
        "volume_step": 0.01
    }


def get_all_pairs() -> list:
    return list(PAIRS.keys())
