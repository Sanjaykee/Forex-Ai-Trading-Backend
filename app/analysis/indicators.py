import pandas as pd
import pandas_ta as ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"]  = ta.ema(df["close"], length=20)
    df["ema50"]  = ta.ema(df["close"], length=min(50, len(df) - 1))
    df["rsi"]    = ta.rsi(df["close"], length=14)
    df["atr"]    = ta.atr(df["high"], df["low"], df["close"], length=14)
    adx          = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx"]    = adx["ADX_14"] if adx is not None else 0
    macd         = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"]        = macd["MACD_12_26_9"] if macd is not None else 0
    df["macd_signal"] = macd["MACDs_12_26_9"] if macd is not None else 0
    return df.dropna(subset=["ema20", "atr", "rsi"])
