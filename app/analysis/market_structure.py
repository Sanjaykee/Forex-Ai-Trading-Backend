import pandas as pd

def detect_structure(df: pd.DataFrame) -> str:
    if len(df) < 10:
        return "RANGING"
    highs = df["high"].values[-10:]
    lows  = df["low"].values[-10:]

    hh = highs[-1] > highs[-5]
    hl = lows[-1]  > lows[-5]
    ll = lows[-1]  < lows[-5]
    lh = highs[-1] < highs[-5]

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
