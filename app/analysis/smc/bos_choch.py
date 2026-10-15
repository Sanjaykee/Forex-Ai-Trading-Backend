import pandas as pd


def detect_bos_choch(df: pd.DataFrame, lookback: int = 20) -> dict:
    return {"bos": False, "choch": False, "direction": None}
