import pandas as pd

def detect_liquidity_sweep(df: pd.DataFrame) -> dict:
    if len(df) < 5:
        return {"detected": False, "direction": None}

    recent = df.iloc[-5:]
    prev_high = recent["high"].iloc[:-1].max()
    prev_low  = recent["low"].iloc[:-1].min()
    last      = df.iloc[-1]

    # Bullish sweep: price swept below previous low then closed back above
    if last["low"] < prev_low and last["close"] > prev_low:
        return {"detected": True, "direction": "BULLISH", "level": round(prev_low, 5)}

    # Bearish sweep: price swept above previous high then closed back below
    if last["high"] > prev_high and last["close"] < prev_high:
        return {"detected": True, "direction": "BEARISH", "level": round(prev_high, 5)}

    return {"detected": False, "direction": None}


def detect_inducement_sweep(df: pd.DataFrame, ob_high: float, ob_low: float, direction: str) -> dict:
    """
    Detects an inducement sweep: a fake move away from the OB zone that sweeps
    a nearby liquidity level, then price reverses back toward the OB for the real entry.

    Pattern (BUY example):
      - OB is below current price (bullish OB)
      - Price makes a lower-low sweep (inducement) beyond a recent swing low
      - Then closes back above that swing low (rejection wick)
      - Price is now heading back into the OB zone = high-conviction entry

    Returns: {"detected": bool, "level": float, "direction": str}
    """
    if len(df) < 8:
        return {"detected": False, "direction": None, "level": None}

    # Look at the last 10 candles for the inducement pattern
    window = df.iloc[-10:]
    current_price = float(df.iloc[-1]["close"])

    if direction == "BUY":
        # Find a swing low in the middle of the window (the inducement target)
        swing_lows = []
        for j in range(1, len(window) - 1):
            l = float(window.iloc[j]["low"])
            if l < float(window.iloc[j - 1]["low"]) and l < float(window.iloc[j + 1]["low"]):
                swing_lows.append((j, l))
        if not swing_lows:
            return {"detected": False, "direction": None, "level": None}

        # Most recent swing low
        idx, swing_low = swing_lows[-1]
        last = window.iloc[-1]

        # Inducement: wick swept below swing_low but closed back above it
        # AND current price is moving back toward or inside the OB zone
        swept = float(last["low"]) < swing_low and float(last["close"]) > swing_low
        returning_to_ob = current_price <= ob_high * 1.002  # within 0.2% of OB top
        if swept and returning_to_ob:
            return {"detected": True, "direction": "BULLISH", "level": round(swing_low, 5)}

    elif direction == "SELL":
        # Find a swing high in the middle of the window (the inducement target)
        swing_highs = []
        for j in range(1, len(window) - 1):
            h = float(window.iloc[j]["high"])
            if h > float(window.iloc[j - 1]["high"]) and h > float(window.iloc[j + 1]["high"]):
                swing_highs.append((j, h))
        if not swing_highs:
            return {"detected": False, "direction": None, "level": None}

        idx, swing_high = swing_highs[-1]
        last = window.iloc[-1]

        swept = float(last["high"]) > swing_high and float(last["close"]) < swing_high
        returning_to_ob = current_price >= ob_low * 0.998
        if swept and returning_to_ob:
            return {"detected": True, "direction": "BEARISH", "level": round(swing_high, 5)}

    return {"detected": False, "direction": None, "level": None}
