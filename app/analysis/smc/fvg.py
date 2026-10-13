import pandas as pd

def detect_fvg(df: pd.DataFrame, lookback: int = 15) -> dict:
    if len(df) < 3:
        return {"detected": False, "direction": None}

    current_price = float(df["close"].iloc[-1])
    start = max(0, len(df) - lookback - 2)

    # Scan oldest→newest, keep most recent unmitigated FVG
    best = None
    for i in range(start, len(df) - 2):
        c1 = df.iloc[i]
        c3 = df.iloc[i + 2]

        gap_high_bull = round(float(c3["low"]),  5)
        gap_low_bull  = round(float(c1["high"]), 5)

        gap_high_bear = round(float(c1["low"]),  5)
        gap_low_bear  = round(float(c3["high"]), 5)

        # Bullish FVG: c3 low > c1 high — gap above
        if float(c3["low"]) > float(c1["high"]):
            # Unmitigated: current price is inside the gap (pulling back into it)
            if gap_low_bull <= current_price <= gap_high_bull:
                best = {"detected": True, "direction": "BULLISH",
                        "gap_high": gap_high_bull, "gap_low": gap_low_bull}

        # Bearish FVG: c3 high < c1 low — gap below
        elif float(c3["high"]) < float(c1["low"]):
            if gap_low_bear <= current_price <= gap_high_bear:
                best = {"detected": True, "direction": "BEARISH",
                        "gap_high": gap_high_bear, "gap_low": gap_low_bear}

    if best:
        return best

    return {"detected": False, "direction": None}
