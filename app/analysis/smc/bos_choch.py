import pandas as pd
import numpy as np


def _find_swings(highs: np.ndarray, lows: np.ndarray, n: int = 3):
    """Return list of (index, price) for swing highs and swing lows."""
    sh, sl = [], []
    for i in range(n, len(highs) - n):
        if highs[i] == max(highs[i - n: i + n + 1]):
            sh.append((i, highs[i]))
        if lows[i] == min(lows[i - n: i + n + 1]):
            sl.append((i, lows[i]))
    return sh, sl


def detect_bos_choch(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Detect BOS and CHoCH from real swing structure.

    BOS   : close breaks last swing high (BULLISH) or last swing low (BEARISH)
            in the SAME direction as the prior trend (HH/HL or LL/LH).

    CHoCH : close breaks last swing high/low but AGAINST the prior trend
            — signals a reversal. Higher confidence than BOS.

    Returns:
        detected  : bool
        type      : "CHoCH" | "BOS" | None
        direction : "BULLISH" | "BEARISH" | None
        broke_level : float — the swing level that was broken
        swing_high  : float — last swing high
        swing_low   : float — last swing low
    """
    use = df.tail(max(lookback, 20)).reset_index(drop=True)
    if len(use) < 15:
        return {"detected": False, "type": None, "direction": None}

    highs = use["high"].values.astype(float)
    lows  = use["low"].values.astype(float)
    close = float(use["close"].iloc[-1])

    sh_list, sl_list = _find_swings(highs, lows, n=2)

    if len(sh_list) < 2 or len(sl_list) < 2:
        return {"detected": False, "type": None, "direction": None}

    # Last two swing highs and lows
    last_sh,  last_sh_price  = sh_list[-1]
    prev_sh,  prev_sh_price  = sh_list[-2]
    last_sl,  last_sl_price  = sl_list[-1]
    prev_sl,  prev_sl_price  = sl_list[-2]

    # Prior trend from swing sequence
    prior_bullish = (last_sh_price > prev_sh_price) and (last_sl_price > prev_sl_price)
    prior_bearish = (last_sh_price < prev_sh_price) and (last_sl_price < prev_sl_price)

    pip = 0.01 if last_sh_price > 20 else 0.0001

    broke_high = close > last_sh_price + pip
    broke_low  = close < last_sl_price - pip

    if broke_high:
        sig_type = "CHoCH" if prior_bearish else "BOS"
        return {
            "detected": True, "type": sig_type, "direction": "BULLISH",
            "broke_level": round(last_sh_price, 5),
            "swing_high":  round(last_sh_price, 5),
            "swing_low":   round(last_sl_price, 5),
        }

    if broke_low:
        sig_type = "CHoCH" if prior_bullish else "BOS"
        return {
            "detected": True, "type": sig_type, "direction": "BEARISH",
            "broke_level": round(last_sl_price, 5),
            "swing_high":  round(last_sh_price, 5),
            "swing_low":   round(last_sl_price, 5),
        }

    return {"detected": False, "type": None, "direction": None}
