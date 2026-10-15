import pandas as pd


def detect_order_block(df: pd.DataFrame, lookback: int = 30) -> dict:
    return {"detected": False, "direction": None, "zone_high": None, "zone_low": None}
