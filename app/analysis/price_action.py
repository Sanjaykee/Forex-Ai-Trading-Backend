import pandas as pd

def detect_price_action(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        return {"pattern": None, "direction": None}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    body      = abs(last["close"] - last["open"])
    upper_wick= last["high"] - max(last["close"], last["open"])
    lower_wick= min(last["close"], last["open"]) - last["low"]
    candle_range = last["high"] - last["low"]

    if candle_range == 0:
        return {"pattern": None, "direction": None}

    # Bullish engulfing
    if (prev["close"] < prev["open"] and
        last["close"] > last["open"] and
        last["close"] > prev["open"] and
        last["open"]  < prev["close"]):
        return {"pattern": "ENGULFING", "direction": "BULLISH"}

    # Bearish engulfing
    if (prev["close"] > prev["open"] and
        last["close"] < last["open"] and
        last["close"] < prev["open"] and
        last["open"]  > prev["close"]):
        return {"pattern": "ENGULFING", "direction": "BEARISH"}

    # Bullish rejection (hammer)
    if lower_wick > body * 2 and upper_wick < body * 0.5:
        return {"pattern": "REJECTION", "direction": "BULLISH"}

    # Bearish rejection (shooting star)
    if upper_wick > body * 2 and lower_wick < body * 0.5:
        return {"pattern": "REJECTION", "direction": "BEARISH"}

    # Strong momentum candle
    if body > candle_range * 0.7:
        direction = "BULLISH" if last["close"] > last["open"] else "BEARISH"
        return {"pattern": "MOMENTUM", "direction": direction}

    return {"pattern": None, "direction": None}
