import pandas as pd

def detect_order_block(df: pd.DataFrame, lookback: int = 20) -> dict:
    if len(df) < lookback + 1:
        return {"detected": False, "direction": None, "zone_high": None, "zone_low": None}

    current_price = float(df["close"].iloc[-1])
    start = max(0, len(df) - lookback - 1)

    # Scan from oldest to newest so we prefer the most recent valid OB
    best = None
    for i in range(start, len(df) - 1):
        candle = df.iloc[i]
        next_c = df.iloc[i + 1]

        body      = abs(float(candle["close"]) - float(candle["open"]))
        next_body = abs(float(next_c["close"]) - float(next_c["open"]))
        if body == 0 or next_body < body * 1.5:
            continue

        zh = round(float(candle["high"]), 5)
        zl = round(float(candle["low"]),  5)

        # Bullish OB: last bearish candle before strong bullish impulse
        if float(candle["close"]) < float(candle["open"]) and float(next_c["close"]) > float(next_c["open"]):
            # Mitigation check: current price must be inside or touching the OB zone
            if zl <= current_price <= zh:
                best = {"detected": True, "direction": "BULLISH", "zone_high": zh, "zone_low": zl}

        # Bearish OB: last bullish candle before strong bearish impulse
        elif float(candle["close"]) > float(candle["open"]) and float(next_c["close"]) < float(next_c["open"]):
            if zl <= current_price <= zh:
                best = {"detected": True, "direction": "BEARISH", "zone_high": zh, "zone_low": zl}

    if best:
        return best

    return {"detected": False, "direction": None, "zone_high": None, "zone_low": None}
