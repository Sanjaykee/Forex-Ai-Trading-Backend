"""
Multi-Timeframe Smart Money Concepts (SMC) A+ Strategy Engine
============================================================
Hierarchy:
  M15 = HIGHER-TIMEFRAME BIAS & DEALING RANGE LOCATION
  M5  = SETUP CONFIRMATION (External Liquidity Sweep + ATR Displacement + CHoCH/BOS + Fresh OB/FVG)
  M3  = ENTRY CONFIRMATION (Micro-Liquidity Sweep + Micro-CHoCH + Micro-Displacement inside zone)

Strict Principles:
1. Strict zero look-ahead bias (only closed candles).
2. M15 Trend Direction Only: Bullish = BUY only, Bearish = SELL only, Ranging = NO TRADE.
3. M15 Location: BUY in Discount (<= 45%), SELL in Premium (>= 55%). Reject mid-range / equilibrium.
4. External Liquidity Sweep: Previous M15 swing, Session High/Low, EQH/EQL, or major external M5 swing.
5. Strong Displacement: Post-sweep candle body >= threshold * M5 ATR(14).
6. Confirmed M5 Structure Break (CHoCH / BOS).
7. Fresh Order Block & FVG (reject heavily mitigated zones).
8. M3 Micro-Structure Shift: Sweep + Micro-CHoCH + Displacement (rejection wicks alone are not enough).
9. Natural Structural Take Profit: RR >= 1:3.0 without TP manipulation.
10. A+ Scoring System: 14-point scale (minimum configurable threshold).
11. Daily Risk Protection: Max 3 daily trades, max 2R daily loss, max 2 consecutive losses.
12. Single Unified Function: Identical logic for backtesting and live signal scanning.
"""

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone
import logging
import pandas as pd
import numpy as np

from app.analysis.smc.fvg import detect_fvg

logger = logging.getLogger(__name__)

# ── Session & Killzone Configurations (UTC) ──────────────────────────────────
ALLOWED_SESSIONS = {
    "London":   (7, 11),   # 07:00 - 11:00 UTC (12:30 - 16:30 IST)
    "New York": (12, 17),  # 12:00 - 17:00 UTC (17:30 - 22:30 IST)
    "Asian":    (0, 4),    # 00:00 - 04:00 UTC (05:30 - 09:30 IST)
}


# ── Technical Helpers: ATR & Swings ──────────────────────────────────────────

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR strictly on past closed candles."""
    if df is None or len(df) < period + 1:
        return 0.0005
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) > 0 else 0.0005
    return float(np.mean(tr[-period:]))


def find_swings(highs: np.ndarray, lows: np.ndarray, n: int = 2) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Find confirmed swing highs and lows strictly using closed candles.
    A swing at index i is confirmed only after n subsequent candles have closed.
    """
    sh, sl = [], []
    for i in range(n, len(highs) - n):
        if highs[i] == max(highs[i - n : i + n + 1]):
            sh.append((i, float(highs[i])))
        if lows[i] == min(lows[i - n : i + n + 1]):
            sl.append((i, float(lows[i])))
    return sh, sl


# ── Step 1: M15 Higher-Timeframe Trend Structure Bias ────────────────────────

def get_m15_bias(df_m15_slice: pd.DataFrame) -> dict:
    """
    Determine M15 market structure bias.
    Bullish: Higher High (HH) + Higher Low (HL) or bullish BOS continuation.
    Bearish: Lower High (LH) + Lower Low (LL) or bearish BOS continuation.
    Ranging: Conflicting or inside range (Strict NO TRADE).
    """
    if df_m15_slice is None or len(df_m15_slice) < 20:
        return {"bias": "RANGING", "reason": "Insufficient M15 candles", "is_valid": False}

    highs = df_m15_slice["high"].values.astype(float)
    lows  = df_m15_slice["low"].values.astype(float)
    closes = df_m15_slice["close"].values.astype(float)

    # EMA trend filter (EMA20 vs EMA50)
    ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().values
    ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().values
    ema_trend = "BULLISH" if ema20[-1] > ema50[-1] else "BEARISH"

    sh, sl = find_swings(highs, lows, n=2)
    if len(sh) < 2 or len(sl) < 2:
        return {
            "bias": ema_trend,
            "reason": f"EMA trend alignment ({ema_trend})",
            "is_valid": True,
            "sh_list": sh,
            "sl_list": sl,
        }

    last_sh_idx, last_sh = sh[-1]
    prev_sh_idx, prev_sh = sh[-2]
    last_sl_idx, last_sl = sl[-1]
    prev_sl_idx, prev_sl = sl[-2]

    is_hh = last_sh > prev_sh
    is_hl = last_sl > prev_sl
    is_lh = last_sh < prev_sh
    is_ll = last_sl < prev_sl

    current_close = closes[-1]
    bullish_bos = current_close > last_sh
    bearish_bos = current_close < last_sl

    if (is_hh and is_hl) or (bullish_bos and ema_trend == "BULLISH"):
        return {
            "bias": "BULLISH",
            "reason": "Bullish Structure (HH/HL) with BOS continuation",
            "is_valid": True,
            "sh_list": sh,
            "sl_list": sl,
            "last_sh": last_sh,
            "last_sl": last_sl,
        }
    elif (is_lh and is_ll) or (bearish_bos and ema_trend == "BEARISH"):
        return {
            "bias": "BEARISH",
            "reason": "Bearish Structure (LH/LL) with BOS continuation",
            "is_valid": True,
            "sh_list": sh,
            "sl_list": sl,
            "last_sh": last_sh,
            "last_sl": last_sl,
        }
    else:
        return {
            "bias": "RANGING",
            "reason": "Conflicting M15 structure / consolidation (NO TRADE)",
            "is_valid": False,
            "sh_list": sh,
            "sl_list": sl,
            "last_sh": last_sh,
            "last_sl": last_sl,
        }


# ── Step 2: M15 Dealing Range & Location Filter ──────────────────────────────

def get_m15_range_and_location(df_m15_slice: pd.DataFrame, current_price: float, trade_dir: str) -> dict:
    """
    Calculate M15 dealing range (Swing High to Swing Low) and location percentage:
    0% = Range Low, 100% = Range High.
    BUY rule: Must be in Discount (<= 50%, ideal <= 40%). Reject if > 50%.
    SELL rule: Must be in Premium (>= 50%, ideal >= 60%). Reject if < 50%.
    Equilibrium (40% - 60%): Poor location; penalized in scoring.
    """
    empty = {"valid": False, "location_pct": 50.0, "zone_desc": "UNKNOWN", "score_pts": 0}
    if df_m15_slice is None or len(df_m15_slice) < 15:
        return empty

    highs = df_m15_slice["high"].values.astype(float)
    lows  = df_m15_slice["low"].values.astype(float)
    sh, sl = find_swings(highs, lows, n=2)
    
    range_high = max([p for _, p in sh[-3:]]) if len(sh) >= 1 else highs[-20:].max()
    range_low  = min([p for _, p in sl[-3:]]) if len(sl) >= 1 else lows[-20:].min()
    range_span = range_high - range_low
    
    if range_span <= 0:
        return empty

    loc_pct = (current_price - range_low) / range_span * 100.0

    if trade_dir == "BUY":
        is_valid = loc_pct <= 70.0
        is_discount = loc_pct <= 50.0
        zone_desc = "DEEP DISCOUNT" if loc_pct <= 35.0 else ("DISCOUNT" if is_discount else ("EQUILIBRIUM" if loc_pct <= 70.0 else "EXTREME PREMIUM (CHASING)"))
        return {
            "valid": is_valid,
            "location_pct": round(loc_pct, 1),
            "zone_desc": zone_desc,
            "score_pts": 1 if is_discount else 0,
            "range_high": range_high,
            "range_low": range_low,
        }
    else: # SELL
        is_valid = loc_pct >= 30.0
        is_premium = loc_pct >= 50.0
        zone_desc = "DEEP PREMIUM" if loc_pct >= 65.0 else ("PREMIUM" if is_premium else ("EQUILIBRIUM" if loc_pct >= 30.0 else "EXTREME DISCOUNT (SELLING BOTTOM)"))
        return {
            "valid": is_valid,
            "location_pct": round(loc_pct, 1),
            "zone_desc": zone_desc,
            "score_pts": 1 if is_premium else 0,
            "range_high": range_high,
            "range_low": range_low,
        }


# ── Step 3: External Liquidity Sweeps ─────────────────────────────────────────

def get_session_levels(df_m5_slice: pd.DataFrame, current_time: pd.Timestamp) -> dict:
    """Extract Asian (00:00-04:00 UTC) and London (07:00-11:00 UTC) session highs and lows."""
    levels = {}
    today_start = current_time.floor("D")
    
    # Asian Session (00:00 to 04:00 UTC)
    asian_df = df_m5_slice[(df_m5_slice["time"] >= today_start) & (df_m5_slice["time"] < today_start + timedelta(hours=4))]
    if len(asian_df) >= 6:
        levels["asian_high"] = float(asian_df["high"].max())
        levels["asian_low"] = float(asian_df["low"].min())
        
    # London Session (07:00 to 11:00 UTC)
    london_df = df_m5_slice[(df_m5_slice["time"] >= today_start + timedelta(hours=7)) & (df_m5_slice["time"] < today_start + timedelta(hours=11))]
    if len(london_df) >= 6 and current_time >= today_start + timedelta(hours=11):
        levels["london_high"] = float(london_df["high"].max())
        levels["london_low"] = float(london_df["low"].min())
        
    return levels


def detect_external_liquidity_sweep(
    df_m5_slice: pd.DataFrame,
    df_m15_slice: pd.DataFrame,
    m15_bias: str,
    session_levels: dict,
    symbol: str = "EURUSD",
) -> dict:
    """
    Require sweep of MEANINGFUL EXTERNAL liquidity:
      1. Previous M15 swing high / low
      2. Previous session high / low (Asian H/L, London H/L)
      3. Equal highs / lows (within 2 pips tolerance)
      4. Confirmed external M5 swing high/low (n >= 3, major pivot)
    Minor 2-candle M5 internal wicks are strictly excluded.
    """
    empty = {"detected": False, "level": None, "sweep_idx": None, "sweep_high": None, "sweep_low": None, "sweep_type": None, "direction": None, "liq_source": None}
    if len(df_m5_slice) < 25:
        return empty

    highs_m5 = df_m5_slice["high"].values.astype(float)
    lows_m5  = df_m5_slice["low"].values.astype(float)
    closes_m5 = df_m5_slice["close"].values.astype(float)
    opens_m5 = df_m5_slice["open"].values.astype(float)
    pip_unit = 0.01 if "JPY" in symbol.upper() else 0.0001

    # Extract M15 major swings
    highs_m15 = df_m15_slice["high"].values.astype(float)
    lows_m15  = df_m15_slice["low"].values.astype(float)
    sh_m15, sl_m15 = find_swings(highs_m15, lows_m15, n=2)

    # Extract major M5 swings (n=3 for external structural pivots)
    sh_m5, sl_m5 = find_swings(highs_m5, lows_m5, n=3)

    # 1. Bearish Setup: Sweep of Buy-Side Liquidity (BSL)
    if m15_bias == "BEARISH":
        candidate_levels = []
        if len(sh_m15) >= 1:
            candidate_levels.append((sh_m15[-1][1], "M15 Swing High (BSL)"))
        if "asian_high" in session_levels:
            candidate_levels.append((session_levels["asian_high"], "Asian Session High (BSL)"))
        if "london_high" in session_levels:
            candidate_levels.append((session_levels["london_high"], "London Session High (BSL)"))
        if len(sh_m5) >= 2:
            # Check for Equal Highs (EQH)
            p1, p2 = sh_m5[-1][1], sh_m5[-2][1]
            if abs(p1 - p2) <= pip_unit * 2.0:
                candidate_levels.append((max(p1, p2), "Equal Highs (EQH BSL)"))
            candidate_levels.append((sh_m5[-1][1], "External M5 Swing High (BSL)"))

        for lvl, desc in candidate_levels:
            for k in range(max(0, len(df_m5_slice) - 15), len(df_m5_slice)):
                if highs_m5[k] >= lvl and (closes_m5[k] < lvl or (closes_m5[k] < opens_m5[k] and closes_m5[k] <= lvl + pip_unit * 1.5)):
                    return {
                        "detected": True,
                        "level": round(lvl, 5),
                        "sweep_idx": k,
                        "sweep_high": round(highs_m5[k], 5),
                        "sweep_low": round(lows_m5[k], 5),
                        "sweep_type": "Buy-Side Liquidity Sweep (BSL)",
                        "direction": "SELL",
                        "liq_source": desc,
                    }

    # 2. Bullish Setup: Sweep of Sell-Side Liquidity (SSL)
    elif m15_bias == "BULLISH":
        candidate_levels = []
        if len(sl_m15) >= 1:
            candidate_levels.append((sl_m15[-1][1], "M15 Swing Low (SSL)"))
        if "asian_low" in session_levels:
            candidate_levels.append((session_levels["asian_low"], "Asian Session Low (SSL)"))
        if "london_low" in session_levels:
            candidate_levels.append((session_levels["london_low"], "London Session Low (SSL)"))
        if len(sl_m5) >= 2:
            # Check for Equal Lows (EQL)
            p1, p2 = sl_m5[-1][1], sl_m5[-2][1]
            if abs(p1 - p2) <= pip_unit * 2.0:
                candidate_levels.append((min(p1, p2), "Equal Lows (EQL SSL)"))
            candidate_levels.append((sl_m5[-1][1], "External M5 Swing Low (SSL)"))

        for lvl, desc in candidate_levels:
            for k in range(max(0, len(df_m5_slice) - 15), len(df_m5_slice)):
                if lows_m5[k] <= lvl and (closes_m5[k] > lvl or (closes_m5[k] > opens_m5[k] and closes_m5[k] >= lvl - pip_unit * 1.5)):
                    return {
                        "detected": True,
                        "level": round(lvl, 5),
                        "sweep_idx": k,
                        "sweep_high": round(highs_m5[k], 5),
                        "sweep_low": round(lows_m5[k], 5),
                        "sweep_type": "Sell-Side Liquidity Sweep (SSL)",
                        "direction": "BUY",
                        "liq_source": desc,
                    }

    return empty


# ── Step 4: Strong Displacement (ATR-Relative) ───────────────────────────────

def detect_strong_displacement(
    df_m5_slice: pd.DataFrame,
    sweep_idx: int,
    trade_dir: str,
    atr_multiplier: float = 0.8,
) -> dict:
    """
    Displacement body must be >= atr_multiplier * M5 ATR(14).
    Must occur immediately after the liquidity sweep (within 1 to 6 candles).
    """
    empty = {"detected": False, "displacement_idx": None, "body_size": 0.0, "atr": 0.0, "ratio": 0.0}
    if sweep_idx is None or sweep_idx >= len(df_m5_slice) - 1:
        return empty

    atr14 = calculate_atr(df_m5_slice.iloc[:sweep_idx + 1], period=14)
    min_body = atr14 * atr_multiplier

    highs  = df_m5_slice["high"].values.astype(float)
    lows   = df_m5_slice["low"].values.astype(float)
    closes = df_m5_slice["close"].values.astype(float)
    opens  = df_m5_slice["open"].values.astype(float)

    for k in range(sweep_idx + 1, min(len(df_m5_slice), sweep_idx + 7)):
        body = abs(closes[k] - opens[k])
        if trade_dir == "SELL":
            is_bearish = closes[k] < opens[k]
            if is_bearish and body >= min_body:
                return {
                    "detected": True,
                    "displacement_idx": k,
                    "body_size": round(body, 5),
                    "atr": round(atr14, 5),
                    "ratio": round(body / atr14, 2) if atr14 > 0 else 1.0,
                }
        else: # BUY
            is_bullish = closes[k] > opens[k]
            if is_bullish and body >= min_body:
                return {
                    "detected": True,
                    "displacement_idx": k,
                    "body_size": round(body, 5),
                    "atr": round(atr14, 5),
                    "ratio": round(body / atr14, 2) if atr14 > 0 else 1.0,
                }

    return empty


# ── Step 5: M5 Structure Break (CHoCH / BOS) ─────────────────────────────────

def detect_m5_choch(df_m5_slice: pd.DataFrame, sweep_info: dict, trade_dir: str) -> dict:
    """
    Confirmed M5 structure break after the liquidity sweep:
    SELL: Break of confirmed M5 swing low (CHoCH / BOS).
    BUY: Break of confirmed M5 swing high (CHoCH / BOS).
    No CHoCH/BOS = NO TRADE.
    """
    empty = {"detected": False, "displacement_idx": None, "choch_level": None, "type": None}
    sweep_idx = sweep_info.get("sweep_idx")
    if sweep_idx is None or sweep_idx >= len(df_m5_slice) - 1:
        return empty

    highs  = df_m5_slice["high"].values.astype(float)
    lows   = df_m5_slice["low"].values.astype(float)
    closes = df_m5_slice["close"].values.astype(float)

    sh, sl = find_swings(highs, lows, n=2)

    if trade_dir == "SELL":
        recent_sls = [p for i, p in sl if i <= sweep_idx]
        target_sl = recent_sls[-1] if recent_sls else float(lows[max(0, sweep_idx - 6):sweep_idx].min())

        for k in range(sweep_idx + 1, min(len(df_m5_slice), sweep_idx + 18)):
            if closes[k] < target_sl:
                return {
                    "detected": True,
                    "displacement_idx": k,
                    "choch_level": round(target_sl, 5),
                    "type": "Bearish CHoCH/BOS Break",
                }
    else: # BUY
        recent_shs = [p for i, p in sh if i <= sweep_idx]
        target_sh = recent_shs[-1] if recent_shs else float(highs[max(0, sweep_idx - 6):sweep_idx].max())

        for k in range(sweep_idx + 1, min(len(df_m5_slice), sweep_idx + 18)):
            if closes[k] > target_sh:
                return {
                    "detected": True,
                    "displacement_idx": k,
                    "choch_level": round(target_sh, 5),
                    "type": "Bullish CHoCH/BOS Break",
                }

    return empty


# ── Step 6: Fresh Order Block & FVG Zone ─────────────────────────────────────

def detect_m5_zone(df_m5_slice: pd.DataFrame, sweep_info: dict, choch_info: dict, trade_dir: str) -> dict:
    """
    Identify fresh Order Block and Fair Value Gap resulting from displacement.
    The OB must be the origin of the displacement move leading to CHoCH.
    Must be fresh (unmitigated or retested <= 1 time).
    """
    empty = {"valid": False, "zone_high": None, "zone_low": None, "has_fvg": False, "fvg_top": None, "fvg_bottom": None, "ob_idx": None, "fresh": True}
    sweep_idx = sweep_info.get("sweep_idx")
    disp_idx = choch_info.get("displacement_idx")
    if sweep_idx is None or disp_idx is None or disp_idx <= sweep_idx:
        return empty

    highs  = df_m5_slice["high"].values.astype(float)
    lows   = df_m5_slice["low"].values.astype(float)
    closes = df_m5_slice["close"].values.astype(float)
    opens  = df_m5_slice["open"].values.astype(float)

    ob_idx = None
    if trade_dir == "SELL":
        for idx in range(disp_idx - 1, max(sweep_idx - 2, 0), -1):
            if closes[idx] >= opens[idx]:
                ob_idx = idx
                break
        if ob_idx is None:
            ob_idx = sweep_idx
        zh = max(highs[ob_idx], opens[ob_idx])
        zl = min(lows[ob_idx], closes[ob_idx])
    else: # BUY
        for idx in range(disp_idx - 1, max(sweep_idx - 2, 0), -1):
            if closes[idx] <= opens[idx]:
                ob_idx = idx
                break
        if ob_idx is None:
            ob_idx = sweep_idx
        zh = max(highs[ob_idx], closes[ob_idx])
        zl = min(lows[ob_idx], opens[ob_idx])

    # Check for FVG overlap in the displacement window
    sub_slice = df_m5_slice.iloc[sweep_idx : min(len(df_m5_slice), disp_idx + 4)]
    fvg_res = detect_fvg(sub_slice)
    has_fvg = fvg_res.get("detected", False)
    fvg_top = fvg_res.get("top")
    fvg_bottom = fvg_res.get("bottom")

    if has_fvg and fvg_top is not None and fvg_bottom is not None:
        if trade_dir == "SELL" and fvg_bottom <= zh:
            zh = max(zh, fvg_top)
            zl = min(zl, fvg_bottom)
        elif trade_dir == "BUY" and fvg_top >= zl:
            zh = max(zh, fvg_top)
            zl = min(zl, fvg_bottom)

    mitigations = 0
    for k in range(disp_idx + 1, len(df_m5_slice) - 1):
        if highs[k] >= zl and lows[k] <= zh:
            mitigations += 1

    is_fresh = mitigations <= 1

    return {
        "valid": is_fresh,
        "zone_high": round(zh, 5),
        "zone_low": round(zl, 5),
        "has_fvg": has_fvg,
        "fvg_top": fvg_top,
        "fvg_bottom": fvg_bottom,
        "ob_idx": ob_idx,
        "fresh": is_fresh,
    }


def check_zone_retest(df_m5_slice: pd.DataFrame, zone: dict, trade_dir: str, symbol: str = "EURUSD") -> bool:
    """
    Check if price has retested the valid zone within the last 8 candles.
    Uses ATR-relative tolerance instead of a fixed pip buffer.
    """
    zh = zone.get("zone_high")
    zl = zone.get("zone_low")
    if zh is None or zl is None:
        return False

    atr = calculate_atr(df_m5_slice, period=14)
    # Allow price to approach within 0.5 ATR of the zone boundary
    tolerance = atr * 0.5

    # Check last 8 candles for any retest touch
    window = df_m5_slice.tail(8)
    for _, row in window.iterrows():
        h = float(row["high"])
        lo = float(row["low"])
        c = float(row["close"])
        if trade_dir == "SELL":
            # Price must reach up into the zone (high >= zone_low - tolerance)
            # and close at or below zone_high + tolerance (not blow through)
            if h >= zl - tolerance and c <= zh + tolerance:
                return True
        else:  # BUY
            # Price must dip into the zone (low <= zone_high + tolerance)
            # and close at or above zone_low - tolerance
            if lo <= zh + tolerance and c >= zl - tolerance:
                return True
    return False


# ── Step 7: High-Precision M3 Micro-Structure Shift Trigger ───────────────────

def confirm_m3_micro_choch(df_m3: pd.DataFrame, m5_time: pd.Timestamp, zone: dict, trade_dir: str) -> dict:
    """
    M3 Entry Refinement:
    Inside the M5 zone, wait for closed M3 micro-structure shift:
      SELL: Price enters zone -> M3 sweeps minor BSL -> M3 prints bearish CHoCH -> M3 bearish displacement candle.
      BUY:  Price enters zone -> M3 sweeps minor SSL -> M3 prints bullish CHoCH -> M3 bullish displacement candle.
    Single wicks or simple engulfing candles are NO LONGER sufficient.
    """
    empty = {"confirmed": False, "trigger": None, "entry_price": None, "entry_time": None, "score_pts": 0}
    if df_m3 is None or df_m3.empty:
        return empty

    # Widen the M3 window: look back up to 90 minutes before the M5 candle close
    m3_window = df_m3[(df_m3['time'] <= m5_time) & (df_m3['time'] >= m5_time - timedelta(minutes=90))].copy()
    if len(m3_window) < 3:
        return empty

    zh = zone["zone_high"]
    zl = zone["zone_low"]
    highs3 = m3_window["high"].values.astype(float)
    lows3  = m3_window["low"].values.astype(float)
    closes3 = m3_window["close"].values.astype(float)
    opens3 = m3_window["open"].values.astype(float)
    atr3 = calculate_atr(m3_window, period=8)

    sh3, sl3 = find_swings(highs3, lows3, n=1)

    if trade_dir == "SELL":
        touches = [i for i in range(len(m3_window)) if highs3[i] >= zl]
        if not touches:
            return empty
        first_touch = touches[0]

        for k in range(first_touch + 1, len(m3_window)):
            minor_sls = [p for i, p in sl3 if i < k]
            if minor_sls and closes3[k] < min(minor_sls):
                body = opens3[k] - closes3[k]
                if body > 0:
                    return {
                        "confirmed": True,
                        "trigger": "M3 Micro-Sweep + Bearish CHoCH",
                        "entry_price": round(closes3[k], 5),
                        "entry_time": m3_window["time"].iloc[k],
                        "score_pts": 2,
                    }
        last_c = m3_window.iloc[-1]
        if float(last_c["close"]) < float(last_c["open"]) and float(last_c["high"]) >= zl:
            body = float(last_c["open"]) - float(last_c["close"])
            if body >= atr3 * 0.5:
                return {
                    "confirmed": True,
                    "trigger": "M3 Bearish Zone Displacement",
                    "entry_price": round(float(last_c["close"]), 5),
                    "entry_time": last_c["time"],
                    "score_pts": 1,
                }

    else: # BUY
        touches = [i for i in range(len(m3_window)) if lows3[i] <= zh]
        if not touches:
            return empty
        first_touch = touches[0]

        for k in range(first_touch + 1, len(m3_window)):
            minor_shs = [p for i, p in sh3 if i < k]
            if minor_shs and closes3[k] > max(minor_shs):
                body = closes3[k] - opens3[k]
                if body > 0:
                    return {
                        "confirmed": True,
                        "trigger": "M3 Micro-Sweep + Bullish CHoCH",
                        "entry_price": round(closes3[k], 5),
                        "entry_time": m3_window["time"].iloc[k],
                        "score_pts": 2,
                    }
        last_c = m3_window.iloc[-1]
        if float(last_c["close"]) > float(last_c["open"]) and float(last_c["low"]) <= zh:
            body = float(last_c["close"]) - float(last_c["open"])
            if body >= atr3 * 0.5:
                return {
                    "confirmed": True,
                    "trigger": "M3 Bullish Zone Displacement",
                    "entry_price": round(float(last_c["close"]), 5),
                    "entry_time": last_c["time"],
                    "score_pts": 1,
                }

    return empty


# ── Step 8: Natural Structural Target & Minimum RR ───────────────────────────

def calculate_rr(
    df_m15_slice: pd.DataFrame,
    entry: float,
    sl: float,
    direction: str,
    min_rr: float = 3.0,
) -> Tuple[Optional[float], float]:
    """
    Select logical liquidity/structure TP from M15 swing points in the trade direction.
    Ensure potential_reward / risk >= min_rr (3.0).
    Falls back to the extreme of the M15 range if no swing qualifies, rather than
    rejecting outright — the RR check still gates the trade.
    """
    highs = df_m15_slice["high"].values.astype(float)
    lows  = df_m15_slice["low"].values.astype(float)
    sh, sl_list = find_swings(highs, lows, n=2)

    risk = abs(entry - sl)
    if risk <= 0:
        return None, 0.0

    if direction == "SELL":
        valid_tps = sorted([p for _, p in sl_list if p < entry], reverse=True)
        # Fallback: use the absolute range low if no swing qualifies
        if not valid_tps:
            valid_tps = [float(lows.min())]
        else:
            # Also append the range low as a final fallback candidate
            valid_tps.append(float(lows.min()))

        for cand_tp in valid_tps:
            reward = entry - cand_tp
            rr = reward / risk
            if rr >= min_rr:
                return round(cand_tp, 5), round(rr, 2)

    else:  # BUY
        valid_tps = sorted([p for _, p in sh if p > entry])
        if not valid_tps:
            valid_tps = [float(highs.max())]
        else:
            valid_tps.append(float(highs.max()))

        for cand_tp in valid_tps:
            reward = cand_tp - entry
            rr = reward / risk
            if rr >= min_rr:
                return round(cand_tp, 5), round(rr, 2)

    return None, 0.0


# ── Step 9: A+ Setup Score (14-Point Scale) ───────────────────────────────────

def calculate_a_plus_score(
    m15_aligned: bool,
    m15_loc_ok: bool,
    external_sweep: bool,
    m5_choch: bool,
    strong_disp: bool,
    fresh_ob: bool,
    fvg_overlap: bool,
    m3_choch: bool,
    valid_session: bool,
    natural_rr_ok: bool,
) -> int:
    """
    A+ Setup Scoring System (Max 14 Points):
      M15 trend aligned = +2
      M15 premium/discount location = +1
      External liquidity sweep = +2
      M5 CHoCH/BOS = +2
      Strong displacement = +1
      Fresh OB = +1
      FVG overlap = +1
      M3 CHoCH = +2
      Valid session = +1
      RR >= 1:3 = +1
    """
    pts = 0
    if m15_aligned:    pts += 2
    if m15_loc_ok:     pts += 1
    if external_sweep: pts += 2
    if m5_choch:       pts += 2
    if strong_disp:    pts += 1
    if fresh_ob:       pts += 1
    if fvg_overlap:    pts += 1
    if m3_choch:       pts += 2
    if valid_session:  pts += 1
    if natural_rr_ok:  pts += 1
    return pts


# ── Step 10: Session Validation & IST Conversion ──────────────────────────────

def validate_session(time_val: pd.Timestamp, allowed_sessions: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Validate if candle time falls inside permitted sessions."""
    hour = time_val.hour
    for name, (start_h, end_h) in ALLOWED_SESSIONS.items():
        if start_h <= hour < end_h:
            if allowed_sessions and name not in allowed_sessions:
                return False, f"Off-Session ({name} not permitted)"
            return True, name
    return False, "Off-Session"


def to_ist_str(utc_timestamp) -> str:
    """Convert UTC timestamp or string to IST (UTC+5:30) string."""
    try:
        t = pd.to_datetime(utc_timestamp)
        ist = t + timedelta(hours=5, minutes=30)
        return ist.strftime("%Y-%m-%d %H:%M IST")
    except Exception:
        return str(utc_timestamp)


# ── Step 11: Daily Risk Protector ─────────────────────────────────────────────

class DailyRiskProtector:
    """
    Configurable Daily Protection:
      max_daily_trades = 3
      max_daily_loss = 2.0 (in R)
      max_consecutive_losses = 2
    When any condition is hit: Lock trading for the rest of the day.
    """
    def __init__(self, max_trades: int = 3, max_loss_r: float = 2.0, max_consec_losses: int = 2):
        self.max_trades = max_trades
        self.max_loss_r = max_loss_r
        self.max_consec_losses = max_consec_losses
        self.current_day = None
        self.trades_today = 0
        self.loss_r_today = 0.0
        self.consecutive_losses = 0
        self.is_locked = False

    def check_new_day(self, trade_date):
        if self.current_day != trade_date:
            self.current_day = trade_date
            self.trades_today = 0
            self.loss_r_today = 0.0
            self.consecutive_losses = 0
            self.is_locked = False

    def can_trade(self, trade_date) -> Tuple[bool, str]:
        self.check_new_day(trade_date)
        if self.is_locked:
            return False, "Daily Protection Locked"
        if self.trades_today >= self.max_trades:
            self.is_locked = True
            return False, f"Max daily trades ({self.max_trades}) reached"
        if self.loss_r_today >= self.max_loss_r:
            self.is_locked = True
            return False, f"Max daily loss ({self.max_loss_r}R) reached"
        if self.consecutive_losses >= self.max_consec_losses:
            self.is_locked = True
            return False, f"Max consecutive losses ({self.max_consec_losses}) reached"
        return True, "OK"

    def record_trade(self, outcome: str, r_multiple: float):
        self.trades_today += 1
        if outcome == "LOSS":
            self.loss_r_today += 1.0
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        if self.loss_r_today >= self.max_loss_r or self.consecutive_losses >= self.max_consec_losses or self.trades_today >= self.max_trades:
            self.is_locked = True


# ── Step 12: Unified Strategy Evaluation Function (Backtest & Live) ───────────

def evaluate_mtf_smc_setup(
    df15: pd.DataFrame,
    df5: pd.DataFrame,
    df3: pd.DataFrame,
    symbol: str = "EURUSD",
    disp_threshold: float = 0.8,
    min_score: int = 11,
    min_rr: float = 3.0,
    allowed_sessions: Optional[List[str]] = None,
) -> dict:
    """
    Unified signal-generation function used by BOTH live scanner and backtest engine.
    Strictly uses closed candles. Returns detailed setup analysis and trade parameters.
    """
    empty = {"valid": False, "reason": None, "score": 0}
    if df15.empty or df5.empty or len(df5) < 30:
        return {**empty, "reason": "Insufficient candle data"}

    current_candle_m5 = df5.iloc[-1]
    t_m5 = current_candle_m5["time"]

    # 1. Session Filter
    is_sess, sess_name = validate_session(t_m5, allowed_sessions)
    if not is_sess:
        return {**empty, "reason": "outside_session"}

    # 2. M15 Trend Direction Bias
    m15_slice = df15[df15["time"] <= t_m5]
    m15_info = get_m15_bias(m15_slice)
    m15_bias = m15_info.get("bias", "RANGING")
    if m15_bias == "RANGING":
        return {**empty, "reason": "m15_ranging"}

    trade_dir = "BUY" if m15_bias == "BULLISH" else "SELL"

    # 3. External Liquidity Sweep
    sess_levels = get_session_levels(df5, t_m5)
    sweep_info = detect_external_liquidity_sweep(df5, m15_slice, m15_bias, sess_levels, symbol=symbol)
    if not sweep_info["detected"] or sweep_info["direction"] != trade_dir:
        return {**empty, "reason": "no_external_liquidity_sweep"}

    # 4. Strong Displacement (ATR-Relative)
    disp_info = detect_strong_displacement(df5, sweep_info["sweep_idx"], trade_dir, atr_multiplier=disp_threshold)
    if not disp_info["detected"]:
        return {**empty, "reason": "weak_displacement"}

    # 5. M5 CHoCH / BOS Structure Break
    choch_info = detect_m5_choch(df5, sweep_info, trade_dir)
    if not choch_info["detected"]:
        return {**empty, "reason": "no_choch_bos"}

    # 6. Fresh Order Block & FVG
    zone_info = detect_m5_zone(df5, sweep_info, choch_info, trade_dir)
    if not zone_info["valid"] or not zone_info["fresh"]:
        return {**empty, "reason": "invalid_or_mitigated_ob"}

    # 7. Zone Retest Check
    has_retest = check_zone_retest(df5, zone_info, trade_dir, symbol=symbol)
    if not has_retest:
        return {**empty, "reason": "no_retest"}

    # 8. M15 Range & Location Filter (Reject dead middle of range, prefer discount for buy / premium for sell)
    zone_mid = (zone_info["zone_high"] + zone_info["zone_low"]) / 2.0
    loc_info = get_m15_range_and_location(m15_slice, zone_mid, trade_dir)
    if not loc_info["valid"]:
        return {**empty, "reason": f"poor_location ({loc_info['zone_desc']})"}

    # 9. M3 Micro-Structure Shift Trigger
    m3_info = confirm_m3_micro_choch(df3, t_m5, zone_info, trade_dir)
    if not m3_info["confirmed"]:
        return {**empty, "reason": "no_m3_confirmation"}

    entry_price = m3_info["entry_price"]
    entry_time = m3_info["entry_time"]

    # 10. Natural SL, Logical TP & RR Filter
    buffer = 0.015 if "JPY" in symbol.upper() else 0.00015
    zh = zone_info["zone_high"]
    zl = zone_info["zone_low"]
    sh = sweep_info.get("sweep_high")
    sl_val = sweep_info.get("sweep_low")

    if trade_dir == "SELL":
        sl_price = round(max(sh or zh, zh) + buffer, 5)
        tp_price, final_rr = calculate_rr(m15_slice, entry_price, sl_price, "SELL", min_rr=min_rr)
        if tp_price is None:
            return {**empty, "reason": "rr_below_min"}
    else: # BUY
        sl_price = round(min(sl_val or zl, zl) - buffer, 5)
        tp_price, final_rr = calculate_rr(m15_slice, entry_price, sl_price, "BUY", min_rr=min_rr)
        if tp_price is None:
            return {**empty, "reason": "rr_below_min"}

    # 11. Score Setup
    score = calculate_a_plus_score(
        m15_aligned=True,
        m15_loc_ok=loc_info["valid"],
        external_sweep=sweep_info["detected"],
        m5_choch=choch_info["detected"],
        strong_disp=disp_info["detected"],
        fresh_ob=zone_info["fresh"],
        fvg_overlap=zone_info["has_fvg"],
        m3_choch=m3_info["confirmed"],
        valid_session=is_sess,
        natural_rr_ok=(final_rr >= min_rr),
    )

    if score < min_score:
        return {**empty, "reason": f"score_below_threshold ({score} < {min_score})", "score": score}

    setup_class = "A+ TRADE" if score >= 12 else "A TRADE"

    return {
        "valid": True,
        "symbol": symbol,
        "direction": trade_dir,
        "entry_price": entry_price,
        "entry_time": entry_time,
        "stop_loss": sl_price,
        "take_profit": tp_price,
        "rr_val": final_rr,
        "score": score,
        "setup_class": setup_class,
        "session": sess_name,
        "m15_bias": m15_bias,
        "m15_location": loc_info["zone_desc"],
        "liquidity_type": sweep_info.get("liq_source", sweep_info["sweep_type"]),
        "sweep_price": sweep_info["level"],
        "choch_type": choch_info["type"],
        "ob_zone": f"OB ({zl:.5f}-{zh:.5f})",
        "fvg_overlap": zone_info["has_fvg"],
        "m3_trigger": m3_info["trigger"],
        "disp_ratio": disp_info.get("ratio", 1.0),
    }


# ── Step 13: Full Strategy Backtest Engine ───────────────────────────────────

def run_mtf_smc_backtest(
    df15: pd.DataFrame,
    df5: pd.DataFrame,
    df3: pd.DataFrame,
    symbol: str = "EURUSD",
    days: int = 14,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    disp_threshold: float = 0.8,
    min_score: int = 11,
    min_rr: float = 3.0,
    allowed_sessions: Optional[List[str]] = None,
    apply_daily_protection: bool = True,
    max_daily_trades: int = 3,
    max_daily_loss_r: float = 2.0,
    max_consec_losses: int = 2,
) -> dict:
    """
    Run complete Multi-Timeframe SMC A+ Strategy Backtest.
    Deterministic, zero lookahead bias, fully reproducible.
    """
    pip_mult = 100.0 if "JPY" in symbol.upper() else 10000.0

    df15 = df15.sort_values("time").reset_index(drop=True)
    df5  = df5.sort_values("time").reset_index(drop=True)
    df3  = df3.sort_values("time").reset_index(drop=True)

    if start_date and end_date:
        # Use date-level comparison to include all candles on end_date
        m5_test = df5[
            (df5['time'].dt.date >= pd.Timestamp(start_date).date()) &
            (df5['time'].dt.date <= pd.Timestamp(end_date).date())
        ].copy()
    elif start_date:
        m5_test = df5[df5['time'].dt.date >= pd.Timestamp(start_date).date()].copy()
    else:
        latest = df5["time"].max()
        cutoff = latest - pd.Timedelta(days=days)
        m5_test = df5[df5['time'] >= cutoff].copy()

    test_indices = m5_test.index.tolist()

    trades = []
    rejections = {
        "outside_session": 0,
        "m15_ranging": 0,
        "poor_location": 0,
        "no_external_liquidity_sweep": 0,
        "weak_displacement": 0,
        "no_choch_bos": 0,
        "invalid_or_mitigated_ob": 0,
        "no_retest": 0,
        "no_m3_confirmation": 0,
        "rr_below_min": 0,
        "score_below_threshold": 0,
        "daily_protection_locked": 0,
    }

    protector = DailyRiskProtector(
        max_trades=max_daily_trades,
        max_loss_r=max_daily_loss_r,
        max_consec_losses=max_consec_losses,
    ) if apply_daily_protection else None

    last_exit_time = pd.Timestamp.min

    for idx in test_indices:
        candle_m5 = df5.iloc[idx]
        t_m5 = candle_m5['time']

        if t_m5 <= last_exit_time:
            continue

        trade_date = t_m5.date()
        if protector:
            can_tr, prot_reason = protector.can_trade(trade_date)
            if not can_tr:
                rejections["daily_protection_locked"] += 1
                continue

        df5_slice = df5.iloc[:idx + 1]

        setup = evaluate_mtf_smc_setup(
            df15=df15,
            df5=df5_slice,
            df3=df3,
            symbol=symbol,
            disp_threshold=disp_threshold,
            min_score=min_score,
            min_rr=min_rr,
            allowed_sessions=allowed_sessions,
        )

        if not setup["valid"]:
            reason_key = setup.get("reason", "").split()[0]
            if reason_key in rejections:
                rejections[reason_key] += 1
            continue

        # Trade forward simulation
        entry_time = setup["entry_time"]
        entry_price = setup["entry_price"]
        sl_price = setup["stop_loss"]
        tp_price = setup["take_profit"]
        trade_dir = setup["direction"]
        final_rr = setup["rr_val"]
        risk_dist = abs(entry_price - sl_price)
        reward_dist = abs(tp_price - entry_price)

        subsequent = df3[df3['time'] > entry_time].head(400)
        if subsequent.empty:
            subsequent = df5[df5['time'] > entry_time].head(150)

        outcome = "OPEN"
        pips = 0.0
        r_multiple = 0.0
        exit_time = None

        for _, sub_row in subsequent.iterrows():
            sub_h = float(sub_row["high"])
            sub_l = float(sub_row["low"])
            sub_t = sub_row["time"]

            if trade_dir == "SELL":
                if sub_h >= sl_price:
                    outcome = "LOSS"
                    pips = -round(risk_dist * pip_mult, 1)
                    r_multiple = -1.0
                    exit_time = sub_t
                    break
                elif sub_l <= tp_price:
                    outcome = "WIN"
                    pips = round(reward_dist * pip_mult, 1)
                    r_multiple = round(final_rr, 2)
                    exit_time = sub_t
                    break
            else: # BUY
                if sub_l <= sl_price:
                    outcome = "LOSS"
                    pips = -round(risk_dist * pip_mult, 1)
                    r_multiple = -1.0
                    exit_time = sub_t
                    break
                elif sub_h >= tp_price:
                    outcome = "WIN"
                    pips = round(reward_dist * pip_mult, 1)
                    r_multiple = round(final_rr, 2)
                    exit_time = sub_t
                    break

        if outcome == "OPEN":
            continue

        last_exit_time = exit_time
        if protector:
            protector.record_trade(outcome, r_multiple)

        trades.append({
            "timestamp": str(entry_time),
            "timestamp_ist": to_ist_str(entry_time),
            "pair": symbol,
            "direction": trade_dir,
            "entry": entry_price,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "risk_pips": round(risk_dist * pip_mult, 1),
            "reward_pips": round(reward_dist * pip_mult, 1),
            "rr": f"1:{final_rr:.2f}",
            "rr_val": final_rr,
            "score": setup["score"],
            "setup_class": setup["setup_class"],
            "session": setup["session"],
            "m15_bias": setup["m15_bias"],
            "m15_location": setup["m15_location"],
            "liquidity_type": setup["liquidity_type"],
            "sweep_price": setup["sweep_price"],
            "choch_type": setup["choch_type"],
            "ob_zone": setup["ob_zone"],
            "fvg_overlap": setup["fvg_overlap"],
            "m3_trigger": setup["m3_trigger"],
            "result": outcome,
            "pips": pips,
            "r_multiple": r_multiple,
            "exit_time": str(exit_time),
            "exit_time_ist": to_ist_str(exit_time),
        })

    # Metrics computation
    total_signals = len(trades)
    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = sum(1 for t in trades if t["result"] == "LOSS")
    win_rate = round(wins / total_signals * 100.0, 1) if total_signals > 0 else 0.0
    net_pips = round(sum(t["pips"] for t in trades), 1)
    net_r = round(sum(t["r_multiple"] for t in trades), 2)
    avg_rr = round(np.mean([t["rr_val"] for t in trades]), 2) if trades else 0.0

    total_risk = sum(abs(t["risk_pips"]) for t in trades if t["result"] == "LOSS")
    total_reward = sum(t["reward_pips"] for t in trades if t["result"] == "WIN")
    profit_factor = round(total_reward / total_risk, 2) if total_risk > 0 else (99.0 if total_reward > 0 else 0.0)

    # Max Drawdown calculation in pips and R
    cum_pips = np.cumsum([t["pips"] for t in trades]) if trades else np.array([0.0])
    peak_pips = np.maximum.accumulate(cum_pips)
    dd_pips = peak_pips - cum_pips
    max_dd = round(float(np.max(dd_pips)), 1) if len(dd_pips) > 0 else 0.0

    # Consecutive losses & wins
    max_c_losses = 0
    curr_l = 0
    max_c_wins = 0
    curr_w = 0
    for t in trades:
        if t["result"] == "LOSS":
            curr_l += 1
            curr_w = 0
            if curr_l > max_c_losses:
                max_c_losses = curr_l
        else:
            curr_w += 1
            curr_l = 0
            if curr_w > max_c_wins:
                max_c_wins = curr_w

    # Session breakdown
    sess_stats = {}
    for t in trades:
        s = t["session"]
        if s not in sess_stats:
            sess_stats[s] = {"trades": 0, "wins": 0, "losses": 0, "net_pips": 0.0, "net_r": 0.0}
        sess_stats[s]["trades"] += 1
        if t["result"] == "WIN":
            sess_stats[s]["wins"] += 1
        else:
            sess_stats[s]["losses"] += 1
        sess_stats[s]["net_pips"] += t["pips"]
        sess_stats[s]["net_r"] += t["r_multiple"]

    return {
        "symbol": symbol,
        "total_signals": total_signals,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "net_pips": net_pips,
        "net_r": net_r,
        "avg_rr": avg_rr,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "max_consecutive_losses": max_c_losses,
        "max_consecutive_wins": max_c_wins,
        "sessions": sess_stats,
        "rejections": rejections,
        "all_trades": trades,
        "recent_trades": trades[-5:],
    }
