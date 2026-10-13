import pandas as pd

def _find_swing_high(h, lookback: int) -> float:
    """Highest high in the lookback window excluding the last candle."""
    return max(h[-lookback:-1])

def _find_swing_low(l, lookback: int) -> float:
    """Lowest low in the lookback window excluding the last candle."""
    return min(l[-lookback:-1])

def detect_bos_choch(df: pd.DataFrame, lookback: int = 20) -> dict:
    if len(df) < lookback + 1:
        return {"bos": False, "choch": False, "direction": None}

    c = df["close"].values
    h = df["high"].values
    l = df["low"].values

    swing_high = _find_swing_high(h, lookback)
    swing_low  = _find_swing_low(l, lookback)

    # Require close to break pivot by at least 1 pip (not just touch)
    pip = 0.01 if (swing_high > 10) else 0.0001  # crude Gold vs Forex detect
    bos_bull = c[-1] > swing_high + pip
    bos_bear = c[-1] < swing_low  - pip

    # CHoCH: prior structure (mid-window) was opposite direction
    mid = lookback // 2
    prev_trend_bull = c[-(mid + 1)] > c[-lookback]
    prev_trend_bear = c[-(mid + 1)] < c[-lookback]

    choch_bull = bos_bull and prev_trend_bear
    choch_bear = bos_bear and prev_trend_bull

    if choch_bull:
        return {"bos": True, "choch": True,  "direction": "BULLISH",
                "swing_high": round(swing_high, 5), "swing_low": round(swing_low, 5)}
    if choch_bear:
        return {"bos": True, "choch": True,  "direction": "BEARISH",
                "swing_high": round(swing_high, 5), "swing_low": round(swing_low, 5)}
    if bos_bull:
        return {"bos": True, "choch": False, "direction": "BULLISH",
                "swing_high": round(swing_high, 5), "swing_low": round(swing_low, 5)}
    if bos_bear:
        return {"bos": True, "choch": False, "direction": "BEARISH",
                "swing_high": round(swing_high, 5), "swing_low": round(swing_low, 5)}

    return {"bos": False, "choch": False, "direction": None}
