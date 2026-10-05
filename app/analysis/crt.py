import pandas as pd

def detect_crt(df: pd.DataFrame) -> dict:
    """
    Candle Range Theory (CRT):
    - Asian candle forms a range (high/low)
    - London/NY sweeps one side (manipulation)
    - Price reverses and targets the other side (distribution)
    """
    if len(df) < 10:
        return {"detected": False, "direction": None, "swept_level": None, "target": None}

    # Use last 10 candles — look for a consolidation range then sweep
    recent = df.tail(10).reset_index(drop=True)

    # Find the range candle (smallest body = consolidation)
    bodies = (recent["close"] - recent["open"]).abs()
    range_idx = bodies.iloc[:5].idxmin()  # range candle in first 5
    range_candle = recent.iloc[range_idx]

    range_high = float(range_candle["high"])
    range_low  = float(range_candle["low"])
    range_size = range_high - range_low

    if range_size == 0:
        return {"detected": False, "direction": None, "swept_level": None, "target": None}

    # Check candles after range for sweep
    after = recent.iloc[range_idx + 1:]

    for _, candle in after.iterrows():
        # Bullish CRT: swept below range low then closed back above
        if float(candle["low"]) < range_low and float(candle["close"]) > range_low:
            return {
                "detected":     True,
                "direction":    "BULLISH",
                "swept_level":  round(range_low, 5),
                "target":       round(range_high, 5),
                "pattern":      "CRT Bullish — Low swept, targeting range high",
            }
        # Bearish CRT: swept above range high then closed back below
        if float(candle["high"]) > range_high and float(candle["close"]) < range_high:
            return {
                "detected":     True,
                "direction":    "BEARISH",
                "swept_level":  round(range_high, 5),
                "target":       round(range_low, 5),
                "pattern":      "CRT Bearish — High swept, targeting range low",
            }

    return {"detected": False, "direction": None, "swept_level": None, "target": None}
