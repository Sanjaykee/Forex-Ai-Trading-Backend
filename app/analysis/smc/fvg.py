import pandas as pd

def detect_fvg(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        return {"detected": False, "direction": None}

    for i in range(len(df) - 3, len(df) - 2):
        c1 = df.iloc[i]
        c3 = df.iloc[i + 2]

        if float(c3["low"]) > float(c1["high"]):
            return {"detected": True, "direction": "BULLISH",
                    "gap_high": round(float(c3["low"]), 5),
                    "gap_low":  round(float(c1["high"]), 5)}

        if float(c3["high"]) < float(c1["low"]):
            return {"detected": True, "direction": "BEARISH",
                    "gap_high": round(float(c1["low"]), 5),
                    "gap_low":  round(float(c3["high"]), 5)}

    return {"detected": False, "direction": None}
