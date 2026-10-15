import pandas as pd
import numpy as np


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 30) -> dict:
    """
    Liquidity sweep: price wicks beyond a significant swing high/low
    then CLOSES back on the opposite side within the same candle.

    Bullish sweep : wick below swing low, close back above → stop hunt of longs
                    → smart money bought → expect move UP
    Bearish sweep : wick above swing high, close back below → stop hunt of shorts
                    → smart money sold → expect move DOWN

    Checks last 3 candles for the sweep pattern (not just the last one).
    """
    if len(df) < 10:
        return {"detected": False, "direction": None, "level": None, "candles_ago": None}

    use   = df.tail(max(lookback, 15)).reset_index(drop=True)
    n     = len(use)
    ref   = use.iloc[:-3]   # reference window for swing levels

    if len(ref) < 5:
        return {"detected": False, "direction": None, "level": None, "candles_ago": None}

    swing_high = float(ref["high"].max())
    swing_low  = float(ref["low"].min())

    # Check last 3 candles for sweep pattern
    for ago in range(1, 4):
        idx = n - ago
        if idx < 0:
            continue
        c = use.iloc[idx]
        ch = float(c["high"])
        cl = float(c["low"])
        cc = float(c["close"])

        if cl < swing_low and cc > swing_low:
            return {"detected": True, "direction": "BULLISH",
                    "level": round(swing_low, 5), "candles_ago": ago}

        if ch > swing_high and cc < swing_high:
            return {"detected": True, "direction": "BEARISH",
                    "level": round(swing_high, 5), "candles_ago": ago}

    return {"detected": False, "direction": None, "level": None, "candles_ago": None}


def detect_inducement(df: pd.DataFrame, direction: str) -> dict:
    """
    Inducement: a minor swing level gets swept (fake move) BEFORE price
    reaches the Order Block — trapping retail traders the wrong way.

    Pattern (BUY direction):
      Within last 20 candles, find a minor swing low that got swept
      (wick below it, close back above) — this is the inducement trap.

    Pattern (SELL direction): mirror.
    """
    if len(df) < 12:
        return {"detected": False, "level": None}

    use = df.tail(20).reset_index(drop=True)
    n   = len(use)

    if direction == "BULLISH":
        # Find minor swing lows (local low with 1 neighbour each side)
        for j in range(1, n - 3):
            sl = float(use.iloc[j]["low"])
            is_swing = (sl < float(use.iloc[j - 1]["low"]) and
                        sl < float(use.iloc[j + 1]["low"]))
            if not is_swing:
                continue
            # Check if any later candle swept below and closed back above
            for k in range(j + 1, n):
                ck = use.iloc[k]
                if float(ck["low"]) < sl and float(ck["close"]) > sl:
                    return {"detected": True, "level": round(sl, 5)}

    elif direction == "BEARISH":
        for j in range(1, n - 3):
            sh = float(use.iloc[j]["high"])
            is_swing = (sh > float(use.iloc[j - 1]["high"]) and
                        sh > float(use.iloc[j + 1]["high"]))
            if not is_swing:
                continue
            for k in range(j + 1, n):
                ck = use.iloc[k]
                if float(ck["high"]) > sh and float(ck["close"]) < sh:
                    return {"detected": True, "level": round(sh, 5)}

    return {"detected": False, "level": None}
