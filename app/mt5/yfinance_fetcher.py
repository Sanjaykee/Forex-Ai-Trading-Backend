import pandas as pd
import requests
from datetime import datetime, timedelta

PAIRS = {
    "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"), "USDJPY": ("USD", "JPY"),
    "USDCHF": ("USD", "CHF"), "AUDUSD": ("AUD", "USD"), "NZDUSD": ("NZD", "USD"),
    "USDCAD": ("USD", "CAD"), "EURJPY": ("EUR", "JPY"), "GBPJPY": ("GBP", "JPY"),
}

def _get_rate(base: str, quote: str) -> float:
    try:
        r = requests.get(f"https://api.frankfurter.app/latest?from={base}&to={quote}", timeout=5)
        data = r.json()
        return float(data["rates"][quote])
    except Exception:
        return 0.0

def _get_historical(base: str, quote: str, days: int = 60) -> pd.DataFrame:
    try:
        end   = datetime.utcnow().date()
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
        # Synthesize OHLV from close (daily data only)
        df["open"]   = df["close"].shift(1).fillna(df["close"])
        df["high"]   = df["close"] * 1.0005
        df["low"]    = df["close"] * 0.9995
        df["volume"] = 1000
        return df[["time", "open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()

def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df.set_index("time").resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna().reset_index()

def get_candles(symbol: str, timeframe: str, count: int = 200) -> pd.DataFrame:
    pair = PAIRS.get(symbol)
    if not pair:
        return pd.DataFrame()
    base, quote = pair
    # Frankfurter only has daily data — resample to simulate timeframes
    df = _get_historical(base, quote, days=365)
    if df.empty:
        return pd.DataFrame()
    if timeframe == "H4":
        df = _resample(df, "4D")   # 4-day bars simulate H4 structure
    elif timeframe == "H1":
        df = _resample(df, "1D")   # daily bars simulate H1 structure
    elif timeframe == "M15":
        df = _resample(df, "1D")   # daily bars simulate M15 structure
    return df.tail(count).reset_index(drop=True)

def get_live_price(symbol: str) -> dict:
    pair = PAIRS.get(symbol)
    if not pair:
        return {}
    base, quote = pair
    price = _get_rate(base, quote)
    if price == 0.0:
        return {}
    offset = 0.00010 if "JPY" not in symbol else 0.010
    return {"bid": round(price, 5), "ask": round(price + offset, 5), "spread": 1.0}

def get_symbol_info(symbol: str) -> dict:
    point = 0.0001 if "JPY" not in symbol else 0.01
    return {"point": point, "digits": 5, "trade_contract_size": 100000,
            "volume_min": 0.01, "volume_step": 0.01}

def get_all_pairs() -> list:
    return list(PAIRS.keys())
