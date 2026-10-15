import pandas as pd


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> dict:
    return {"detected": False, "direction": None, "level": None}


def detect_inducement_sweep(df: pd.DataFrame, ob_high: float, ob_low: float,
                             direction: str) -> dict:
    return {"detected": False, "direction": None, "level": None}
