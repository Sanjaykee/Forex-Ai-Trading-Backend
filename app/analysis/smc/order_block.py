import pandas as pd

def detect_order_block(df: pd.DataFrame) -> dict:
    if len(df) < 5:
        return {"detected": False, "direction": None, "zone_high": None, "zone_low": None}

    for i in range(len(df) - 4, len(df) - 1):
        candle = df.iloc[i]
        next_c = df.iloc[i + 1]

        body = abs(candle["close"] - candle["open"])
        next_body = abs(next_c["close"] - next_c["open"])

        # Bullish OB: bearish candle followed by strong bullish move
        if candle["close"] < candle["open"] and next_c["close"] > next_c["open"] and next_body > body * 1.5:
            return {
                "detected":  True,
                "direction": "BULLISH",
                "zone_high": round(candle["high"], 5),
                "zone_low":  round(candle["low"], 5),
            }

        # Bearish OB: bullish candle followed by strong bearish move
        if candle["close"] > candle["open"] and next_c["close"] < next_c["open"] and next_body > body * 1.5:
            return {
                "detected":  True,
                "direction": "BEARISH",
                "zone_high": round(candle["high"], 5),
                "zone_low":  round(candle["low"], 5),
            }

    return {"detected": False, "direction": None, "zone_high": None, "zone_low": None}
