import pandas as pd
import numpy as np


def detect_order_block(df: pd.DataFrame, direction: str, lookback: int = 40) -> dict:
    """
    Find the most recent unmitigated Order Block in the given direction.

    Bullish OB : Last bearish candle before a bullish impulse (BOS/CHoCH move up).
                 Unmitigated = no close below OB low after formation.
                 Price must be AT or NEAR the OB zone (within 1.5x ATR).

    Bearish OB : Last bullish candle before a bearish impulse.
                 Unmitigated = no close above OB high after formation.
                 Price must be AT or NEAR the OB zone.

    Returns:
        detected   : bool
        zone_high  : float
        zone_low   : float
        midpoint   : float
        touching   : bool — price is currently inside or touching the zone
    """
    use = df.tail(max(lookback, 15)).reset_index(drop=True)
    if len(use) < 8:
        return {"detected": False, "zone_high": None, "zone_low": None}

    opens  = use["open"].values.astype(float)
    closes = use["close"].values.astype(float)
    highs  = use["high"].values.astype(float)
    lows   = use["low"].values.astype(float)

    current = closes[-1]
    atr = float(np.mean(highs[-14:] - lows[-14:])) or 0.0005

    best = None

    for i in range(len(use) - 3):
        # Need a strong impulse candle after OB candle
        impulse_body = abs(closes[i + 1] - opens[i + 1])
        ob_body      = abs(closes[i] - opens[i])
        if ob_body == 0 or impulse_body < ob_body * 1.5:
            continue

        zh = round(highs[i], 5)
        zl = round(lows[i],  5)

        if direction == "BULLISH":
            # OB candle must be bearish
            if closes[i] >= opens[i]:
                continue
            # Impulse candle must be bullish
            if closes[i + 1] <= opens[i + 1]:
                continue
            # Unmitigated: no close below OB low after formation
            if any(closes[j] < zl for j in range(i + 2, len(use))):
                continue
            # Price must be near or inside OB (approaching from above)
            near = current <= zh + atr * 1.5 and current >= zl - atr * 1.5
            if not near:
                continue
            touching = zl <= current <= zh
            best = {
                "detected": True, "zone_high": zh, "zone_low": zl,
                "midpoint": round((zh + zl) / 2, 5), "touching": touching,
                "ob_index": i,
            }

        elif direction == "BEARISH":
            # OB candle must be bullish
            if closes[i] <= opens[i]:
                continue
            # Impulse candle must be bearish
            if closes[i + 1] >= opens[i + 1]:
                continue
            # Unmitigated: no close above OB high after formation
            if any(closes[j] > zh for j in range(i + 2, len(use))):
                continue
            near = current >= zl - atr * 1.5 and current <= zh + atr * 1.5
            if not near:
                continue
            touching = zl <= current <= zh
            best = {
                "detected": True, "zone_high": zh, "zone_low": zl,
                "midpoint": round((zh + zl) / 2, 5), "touching": touching,
                "ob_index": i,
            }

    return best or {"detected": False, "zone_high": None, "zone_low": None}


def detect_accumulation_ob(df: pd.DataFrame, direction: str, lookback: int = 60) -> dict:
    """
    Pattern C: Detect accumulation range then BOS breakout.
    Accumulation = at least 8 candles with ATR < 0.5x average ATR (tight range).
    OB = last candle of the accumulation range before the breakout candle.
    """
    use = df.tail(max(lookback, 20)).reset_index(drop=True)
    if len(use) < 20:
        return {"detected": False, "zone_high": None, "zone_low": None}

    highs  = use["high"].values.astype(float)
    lows   = use["low"].values.astype(float)
    closes = use["close"].values.astype(float)

    avg_atr = float(np.mean(highs - lows))
    current = closes[-1]

    # Scan for tight consolidation window (min 8 candles)
    for start in range(len(use) - 12):
        window = 8
        seg_h = highs[start: start + window]
        seg_l = lows[start:  start + window]
        seg_atr = float(np.mean(seg_h - seg_l))

        if seg_atr > avg_atr * 0.55:
            continue  # not tight enough

        # Breakout candle right after consolidation
        bo_idx = start + window
        if bo_idx >= len(use) - 1:
            continue

        bo_body = abs(closes[bo_idx] - use["open"].values[bo_idx])
        if bo_body < avg_atr * 0.8:
            continue  # weak breakout

        # OB = last candle of consolidation
        ob_idx = bo_idx - 1
        zh = round(highs[ob_idx], 5)
        zl = round(lows[ob_idx],  5)

        if direction == "BULLISH" and closes[bo_idx] > closes[bo_idx - 1]:
            # Unmitigated
            if any(closes[j] < zl for j in range(bo_idx + 1, len(use))):
                continue
            near = current <= zh + avg_atr * 2 and current >= zl - avg_atr
            if near:
                return {
                    "detected": True, "zone_high": zh, "zone_low": zl,
                    "midpoint": round((zh + zl) / 2, 5), "touching": zl <= current <= zh,
                }

        elif direction == "BEARISH" and closes[bo_idx] < closes[bo_idx - 1]:
            if any(closes[j] > zh for j in range(bo_idx + 1, len(use))):
                continue
            near = current >= zl - avg_atr * 2 and current <= zh + avg_atr
            if near:
                return {
                    "detected": True, "zone_high": zh, "zone_low": zl,
                    "midpoint": round((zh + zl) / 2, 5), "touching": zl <= current <= zh,
                }

    return {"detected": False, "zone_high": None, "zone_low": None}
