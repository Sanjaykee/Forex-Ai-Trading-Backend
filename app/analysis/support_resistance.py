import pandas as pd
import numpy as np

def detect_sr_zones(df: pd.DataFrame) -> dict:
    if len(df) < 20:
        return {"support": None, "resistance": None, "at_support": False, "at_resistance": False}

    highs = df["high"].values[-50:]
    lows  = df["low"].values[-50:]
    close = df["close"].values[-1]
    atr   = df["atr"].values[-1] if "atr" in df.columns else 0.001

    resistance = float(np.percentile(highs, 90))
    support    = float(np.percentile(lows, 10))

    at_resistance = abs(close - resistance) < atr * 0.5
    at_support    = abs(close - support)    < atr * 0.5

    return {
        "support":        round(support, 5),
        "resistance":     round(resistance, 5),
        "at_support":     at_support,
        "at_resistance":  at_resistance,
    }
