import pandas as pd

def detect_bos_choch(df: pd.DataFrame) -> dict:
    if len(df) < 6:
        return {"bos": False, "choch": False, "direction": None}

    c = df["close"].values
    h = df["high"].values
    l = df["low"].values

    # Bullish BOS: close breaks above recent swing high
    swing_high = max(h[-6:-1])
    swing_low  = min(l[-6:-1])

    bos_bull   = c[-1] > swing_high
    bos_bear   = c[-1] < swing_low

    # CHoCH: previous structure was opposite, now breaking
    prev_trend_bull = c[-4] > c[-6]
    prev_trend_bear = c[-4] < c[-6]

    choch_bull = bos_bull and prev_trend_bear
    choch_bear = bos_bear and prev_trend_bull

    if choch_bull:
        return {"bos": True, "choch": True, "direction": "BULLISH"}
    if choch_bear:
        return {"bos": True, "choch": True, "direction": "BEARISH"}
    if bos_bull:
        return {"bos": True, "choch": False, "direction": "BULLISH"}
    if bos_bear:
        return {"bos": True, "choch": False, "direction": "BEARISH"}

    return {"bos": False, "choch": False, "direction": None}
