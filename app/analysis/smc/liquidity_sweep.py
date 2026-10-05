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
