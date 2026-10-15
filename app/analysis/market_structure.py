import pandas as pd
import numpy as np


def detect_structure(df: pd.DataFrame) -> str:
    """
    Detect market bias: BULLISH, BEARISH, or RANGING.
    Uses EMA20 vs EMA50 + last swing sequence (HH/HL or LL/LH).
    Returns RANGING if signals conflict.
    """
    if df is None or len(df) < 30:
        return "RANGING"

    closes = df["close"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)

    # EMA bias
    def ema(arr, p):
        k = 2 / (p + 1)
        e = arr[0]
        for v in arr[1:]:
            e = v * k + e * (1 - k)
        return e

    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    ema_bias = "BULLISH" if ema20 > ema50 else "BEARISH"

    # Swing structure bias
    swing_bias = "RANGING"
    sh_prices, sl_prices = [], []
    n = 3
    for i in range(n, len(highs) - n):
        if highs[i] == max(highs[i - n: i + n + 1]):
            sh_prices.append(highs[i])
        if lows[i] == min(lows[i - n: i + n + 1]):
            sl_prices.append(lows[i])

    if len(sh_prices) >= 2 and len(sl_prices) >= 2:
        hh = sh_prices[-1] > sh_prices[-2]
        hl = sl_prices[-1] > sl_prices[-2]
        ll = sl_prices[-1] < sl_prices[-2]
        lh = sh_prices[-1] < sh_prices[-2]
        if hh and hl:
            swing_bias = "BULLISH"
        elif ll and lh:
            swing_bias = "BEARISH"

    # Both agree → strong bias, one agrees → use it, conflict → RANGING
    if ema_bias == swing_bias:
        return ema_bias
    if swing_bias != "RANGING":
        return swing_bias
    return ema_bias


def detect_trend(df: pd.DataFrame) -> str:
    """Simple EMA20 vs EMA50 trend direction."""
    if df is None or len(df) < 25:
        return "RANGING"
    closes = df["close"].values.astype(float)

    def ema(arr, p):
        k = 2 / (p + 1)
        e = arr[0]
        for v in arr[1:]:
            e = v * k + e * (1 - k)
        return e

    return "BULLISH" if ema(closes, 20) > ema(closes, 50) else "BEARISH"
