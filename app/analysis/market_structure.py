import pandas as pd

def detect_structure(df: pd.DataFrame, lookback: int = 50) -> str:
    if len(df) < lookback:
        return "RANGING"

    highs = df["high"].values[-lookback:]
    lows  = df["low"].values[-lookback:]
    mid   = lookback // 2

    # Compare recent half vs older half for proper HH/HL/LL/LH
    recent_high = max(highs[mid:])
    recent_low  = min(lows[mid:])
    prior_high  = max(highs[:mid])
    prior_low   = min(lows[:mid])

    hh = recent_high > prior_high
    hl = recent_low  > prior_low
    ll = recent_low  < prior_low
    lh = recent_high < prior_high

    if hh and hl:
        return "BULLISH"
    if ll and lh:
        return "BEARISH"
    return "RANGING"

def detect_trend(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "RANGING"
    last = df.iloc[-1]
    if "ema20" not in df.columns or "ema50" not in df.columns:
        return "RANGING"
    if last["ema20"] > last["ema50"] and last["close"] > last["ema50"]:
        return "BULLISH"
    if last["ema20"] < last["ema50"] and last["close"] < last["ema50"]:
        return "BEARISH"
    return "RANGING"
