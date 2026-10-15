import pandas as pd


def detect_fvg(df: pd.DataFrame) -> dict:
    return {"detected": False, "direction": None, "gap_high": None, "gap_low": None}
