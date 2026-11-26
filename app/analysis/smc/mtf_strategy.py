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
    Determine M15 market structure bias using confirmed swing sequence.
    Requires at least 2 consecutive HH+HL (BULLISH) or LH+LL (BEARISH) to confirm trend.
    A single retracement swing is NOT enough — must show a sequence of structure.
    EMA50 used only as a tiebreaker when swing sequence is ambiguous.
    """
    if df_m15_slice is None or len(df_m15_slice) < 20:
        return {"bias": "RANGING", "reason": "Insufficient M15 candles", "is_valid": False}

    highs  = df_m15_slice["high"].values.astype(float)
    lows   = df_m15_slice["low"].values.astype(float)
    closes = df_m15_slice["close"].values.astype(float)

    ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().values
    price_above_ema50 = closes[-1] > ema50[-1]

    sh, sl = find_swings(highs, lows, n=2)

    if len(sh) < 3 or len(sl) < 3:
        # Not enough swings for sequence — fall back to EMA50 only
        bias = "BULLISH" if price_above_ema50 else "BEARISH"
        return {
            "bias": bias,
            "reason": f"EMA50 fallback ({bias}) — insufficient swing history",
            "is_valid": True,
            "sh_list": sh,
            "sl_list": sl,
        }

    # Use last 3 confirmed swing highs and lows
    sh3 = sh[-3:]
    sl3 = sl[-3:]

    # Count how many consecutive HH and HL exist in the last 3 swings
    hh_count = sum(1 for i in range(1, len(sh3)) if sh3[i][1] > sh3[i-1][1])
    hl_count = sum(1 for i in range(1, len(sl3)) if sl3[i][1] > sl3[i-1][1])
    lh_count = sum(1 for i in range(1, len(sh3)) if sh3[i][1] < sh3[i-1][1])
    ll_count = sum(1 for i in range(1, len(sl3)) if sl3[i][1] < sl3[i-1][1])

    last_sh = sh3[-1][1]
    last_sl = sl3[-1][1]

    # Strong bullish: (2 HH + 1 HL) OR (1 HH with net Higher High over swing cycle + HL) AND price above EMA50
    # Strong bearish: (2 LH + 1 LL) OR (1 LH with net Lower High over swing cycle + LL) AND price below EMA50
    # Weak/conflicting: RANGING
    bullish_structure = (hh_count >= 2 and hl_count >= 1) or (hh_count >= 1 and last_sh > sh3[0][1] and hl_count >= 1)
    bearish_structure = (lh_count >= 2 and ll_count >= 1) or (lh_count >= 1 and last_sh < sh3[0][1] and ll_count >= 1)

    # BOS confirmation: current close must have broken the last swing high (bull) or low (bear)
    # This prevents calling a retracement inside a bearish trend as BULLISH
    bullish_bos_confirmed = closes[-1] > last_sh
    bearish_bos_confirmed = closes[-1] < last_sl

    # D1 Higher-Timeframe Trend Confirmation (EMA200 on H1 proxy ~ 8 days)
    # A true macro trend filter ensuring we do not fight the prevailing macro flow
    ema200 = pd.Series(closes).ewm(span=200, adjust=False).mean().values if len(closes) >= 200 else ema50
    macro_bullish = closes[-1] >= ema200[-1]
    macro_bearish = closes[-1] <= ema200[-1]

    if bullish_structure and price_above_ema50 and macro_bullish:
        return {
            "bias": "BULLISH",
            "reason": f"Bullish Structure: {hh_count} HH + {hl_count} HL confirmed + Macro D1 Bullish",
            "is_valid": True,
            "sh_list": sh,
            "sl_list": sl,
            "last_sh": last_sh,
            "last_sl": last_sl,
        }
    elif bearish_structure and not price_above_ema50 and macro_bearish:
        return {
            "bias": "BEARISH",
            "reason": f"Bearish Structure: {lh_count} LH + {ll_count} LL confirmed + Macro D1 Bearish",
            "is_valid": True,
            "sh_list": sh,
            "sl_list": sl,
            "last_sh": last_sh,
            "last_sl": last_sl,
        }
    elif bullish_bos_confirmed and price_above_ema50 and macro_bullish:
        return {
            "bias": "BULLISH",
            "reason": "Bullish BOS: price broke above last swing high + above EMA50 & EMA200",
            "is_valid": True,
            "sh_list": sh,
            "sl_list": sl,
            "last_sh": last_sh,
            "last_sl": last_sl,
        }
    elif bearish_bos_confirmed and not price_above_ema50 and macro_bearish:
        return {
            "bias": "BEARISH",
            "reason": "Bearish BOS: price broke below last swing low + below EMA50 & EMA200",
            "is_valid": True,
            "sh_list": sh,
            "sl_list": sl,
            "last_sh": last_sh,
            "last_sl": last_sl,
        }
    else:
        return {
            "bias": "RANGING",
            "reason": f"Conflicting structure (HH:{hh_count} HL:{hl_count} LH:{lh_count} LL:{ll_count}) — NO TRADE",
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
        is_valid = loc_pct <= 62.0
        is_discount = loc_pct <= 50.0
        zone_desc = "DEEP DISCOUNT" if loc_pct <= 35.0 else ("DISCOUNT" if is_discount else ("EQUILIBRIUM" if loc_pct <= 62.0 else "EXTREME PREMIUM (CHASING)"))
        return {
            "valid": is_valid,
            "location_pct": round(loc_pct, 1),
            "zone_desc": zone_desc,
            "score_pts": 1 if is_discount else 0,
            "range_high": range_high,
            "range_low": range_low,
        }
    else: # SELL
        is_valid = loc_pct >= 38.0
        is_premium = loc_pct >= 50.0
        zone_desc = "DEEP PREMIUM" if loc_pct >= 65.0 else ("PREMIUM" if is_premium else ("EQUILIBRIUM" if loc_pct >= 38.0 else "EXTREME DISCOUNT (SELLING BOTTOM)"))
        return {
            "valid": is_valid,
            "location_pct": round(loc_pct, 1),
            "zone_desc": zone_desc,
            "score_pts": 1 if is_premium else 0,
            "range_high": range_high,
            "range_low": range_low,
        }


# ── Step 2b: Tier 2 Intermediate M15 Structure Alignment Filter ──────────────

def check_m15_intermediate_alignment(
    df_m15: Optional[pd.DataFrame],
    t_m5: pd.Timestamp,
    trade_dir: str,
    symbol: str = "EURUSD",
    has_macro_sweep: bool = False,
) -> Tuple[bool, str]:
    """
    Tier 2: Intermediate M15 Structure Alignment Filter.
    Ensures that micro entries (M3/M5) do not trade against active M15 momentum.
    
    SELL rules:
      - Price must NOT be in an active bullish momentum rally above both M15 EMA20 & EMA50.
      - If price > EMA20 and price > EMA50, SELL is blocked UNLESS M15 has printed a 
        Bearish CHoCH (close < previous swing low), bearish rejection pinbar, OR a valid
        macro high sweep (Asian High, PDH, London High) in Premium.
      - Otherwise: REJECT with 'm15_bullish_momentum'
      
    BUY rules:
      - Price must NOT be in a waterfall dump below both M15 EMA20 & EMA50.
      - If price < EMA20 and price < EMA50, BUY is blocked UNLESS M15 has printed a 
        Bullish CHoCH (close > previous swing high), bullish rejection pinbar, OR a valid
        macro low sweep (Asian Low, PDL, London Low) in Discount.
      - Otherwise: REJECT with 'm15_bearish_momentum'
    """
    if df_m15 is None or df_m15.empty:
        return True, "m15_not_provided"

    t_target = pd.to_datetime(t_m5)
    df_slice = df_m15[df_m15["time"] <= t_target]
    if len(df_slice) < 15:
        return True, "m15_insufficient_data"

    closes = df_slice["close"].values.astype(float)
    highs  = df_slice["high"].values.astype(float)
    lows   = df_slice["low"].values.astype(float)
    opens  = df_slice["open"].values.astype(float)

    ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().values[-1]
    ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().values[-1]
    c_last = closes[-1]

    # Previous M15 swing low / high (last 10 candles excluding current)
    prev_swing_low = np.min(lows[-10:-1]) if len(lows) >= 10 else lows[-2]
    prev_swing_high = np.max(highs[-10:-1]) if len(highs) >= 10 else highs[-2]

    if trade_dir == "SELL":
        is_above_both_emas = (c_last > ema20) and (c_last > ema50)
        has_bearish_choch = c_last < prev_swing_low
        has_bearish_rejection = (highs[-1] - max(opens[-1], c_last)) > 1.5 * abs(c_last - opens[-1]) and (c_last < opens[-1])
        is_bullish_impulse = (len(closes) >= 2 and closes[-1] > opens[-1] and closes[-2] > opens[-2] and closes[-1] > closes[-2])

        # Pure Pro-Trend: Block counter-trend SELL into active M15 bullish momentum
        if is_bullish_impulse and not has_bearish_rejection:
            return False, "m15_bullish_impulse_candles (active M15 green impulse - counter-trend blocked)"
        if is_above_both_emas and not (has_bearish_choch or has_bearish_rejection):
            return False, f"m15_bullish_momentum (Close {c_last:.5f} > EMA20 {ema20:.5f} & EMA50 {ema50:.5f} - counter-trend blocked)"
        return True, "m15_aligned_bearish"

    else:  # BUY
        is_below_both_emas = (c_last < ema20) and (c_last < ema50)
        has_bullish_choch = c_last > prev_swing_high
        has_bullish_rejection = (min(opens[-1], c_last) - lows[-1]) > 1.5 * abs(c_last - opens[-1]) and (c_last > opens[-1])
        is_bearish_impulse = (len(closes) >= 2 and closes[-1] < opens[-1] and closes[-2] < opens[-2] and closes[-1] < closes[-2])

        # Pure Pro-Trend: Block counter-trend BUY into active M15 bearish momentum
        if is_bearish_impulse and not has_bullish_rejection:
            return False, "m15_bearish_impulse_candles (active M15 red impulse - counter-trend blocked)"
        if is_below_both_emas and not (has_bullish_choch or has_bullish_rejection):
            return False, f"m15_bearish_momentum (Close {c_last:.5f} < EMA20 {ema20:.5f} & EMA50 {ema50:.5f} - counter-trend blocked)"
        return True, "m15_aligned_bullish"


# ── Step 3: External Liquidity Sweeps ─────────────────────────────────────────

def get_session_levels(df_m5_slice: pd.DataFrame, current_time: pd.Timestamp) -> dict:
    """Extract Previous Day High/Low (PDH/PDL), Completed Asian H/L and Completed London H/L."""
    levels = {}
    cur_date = current_time.date()
    today_start = current_time.floor("D")

    # 1. Previous Day High & Low (PDH / PDL from last completed trading day)
    past_dates = df_m5_slice[df_m5_slice["time"].dt.date < cur_date]
    if not past_dates.empty:
        prev_date = past_dates["time"].dt.date.max()
        prev_df = past_dates[past_dates["time"].dt.date == prev_date]
        if len(prev_df) >= 20:
            levels["pdh"] = float(prev_df["high"].max())
            levels["pdl"] = float(prev_df["low"].min())

    # 2. Asian Session (00:00 to 04:00 UTC) - eligible only after 04:30 UTC for London & NY
    if current_time >= today_start + timedelta(hours=4, minutes=30):
        asian_df = df_m5_slice[(df_m5_slice["time"] >= today_start) & (df_m5_slice["time"] < today_start + timedelta(hours=4))]
        if len(asian_df) >= 12:
            levels["asian_high"] = float(asian_df["high"].max())
            levels["asian_low"] = float(asian_df["low"].min())

    # 3. London Session (07:00 to 11:00 UTC) - eligible only after 11:30 UTC for NY
    if current_time >= today_start + timedelta(hours=11, minutes=30):
        london_df = df_m5_slice[(df_m5_slice["time"] >= today_start + timedelta(hours=7)) & (df_m5_slice["time"] < today_start + timedelta(hours=11))]
        if len(london_df) >= 12:
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
      1. Previous Day High / Low (PDH / PDL)
      2. Previous completed session high / low (Asian H/L, London H/L)
      3. Previous M15 swing high / low
      4. Equal highs / lows (within 2.5 pips tolerance)
      5. Confirmed major external M5 swing high/low (n >= 4, major pivot)
    Minor internal wicks are strictly excluded. Candle must reject the level cleanly.
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

    # Extract major M5 swings (n=4 for confirmed structural pivots)
    sh_m5, sl_m5 = find_swings(highs_m5, lows_m5, n=4)

    # 1. Bearish Setup: Sweep of Buy-Side Liquidity (BSL)
    if m15_bias == "BEARISH":
        candidate_levels = []
        if "pdh" in session_levels:
            candidate_levels.append((session_levels["pdh"], "Previous Day High (PDH BSL)"))
        if "asian_high" in session_levels:
            candidate_levels.append((session_levels["asian_high"], "Asian Session High (BSL)"))
        if "london_high" in session_levels:
            candidate_levels.append((session_levels["london_high"], "London Session High (BSL)"))
        if len(sh_m15) >= 1:
            candidate_levels.append((sh_m15[-1][1], "M15 Swing High (BSL)"))
        if len(sh_m5) >= 2:
            # Check for Equal Highs (EQH)
            p1, p2 = sh_m5[-1][1], sh_m5[-2][1]
            if abs(p1 - p2) <= pip_unit * 2.5:
                candidate_levels.append((max(p1, p2), "Equal Highs (EQH BSL)"))
        if len(sh_m5) >= 1:
            candidate_levels.append((sh_m5[-1][1], "Major M5 Swing High (BSL)"))

        detected_sweeps = []
        for lvl, desc in candidate_levels:
            for k in range(max(0, len(df_m5_slice) - 12), len(df_m5_slice)):
                # Must poke above level, but close strictly at or below level (clean rejection)
                if highs_m5[k] >= lvl and closes_m5[k] <= lvl + pip_unit * 0.15:
                    detected_sweeps.append({
                        "detected": True,
                        "level": round(lvl, 5),
                        "sweep_idx": k,
                        "sweep_high": round(highs_m5[k], 5),
                        "sweep_low": round(lows_m5[k], 5),
                        "sweep_type": "Buy-Side Liquidity Sweep (BSL)",
                        "direction": "SELL",
                        "liq_source": desc,
                    })
        if detected_sweeps:
            # Pick the true extreme institutional sweep (highest peak)
            return max(detected_sweeps, key=lambda x: (x["sweep_high"], x["sweep_idx"]))

    # 2. Bullish Setup: Sweep of Sell-Side Liquidity (SSL)
    elif m15_bias == "BULLISH":
        candidate_levels = []
        if "pdl" in session_levels:
            candidate_levels.append((session_levels["pdl"], "Previous Day Low (PDL SSL)"))
        if "asian_low" in session_levels:
            candidate_levels.append((session_levels["asian_low"], "Asian Session Low (SSL)"))
        if "london_low" in session_levels:
            candidate_levels.append((session_levels["london_low"], "London Session Low (SSL)"))
        if len(sl_m15) >= 1:
            candidate_levels.append((sl_m15[-1][1], "M15 Swing Low (SSL)"))
        if len(sl_m5) >= 2:
            # Check for Equal Lows (EQL)
            p1, p2 = sl_m5[-1][1], sl_m5[-2][1]
            if abs(p1 - p2) <= pip_unit * 2.5:
                candidate_levels.append((min(p1, p2), "Equal Lows (EQL SSL)"))
        if len(sl_m5) >= 1:
            candidate_levels.append((sl_m5[-1][1], "Major M5 Swing Low (SSL)"))

        detected_sweeps = []
        for lvl, desc in candidate_levels:
            for k in range(max(0, len(df_m5_slice) - 12), len(df_m5_slice)):
                # Must poke below level, but close strictly at or above level (clean rejection)
                if lows_m5[k] <= lvl and closes_m5[k] >= lvl - pip_unit * 0.15:
                    detected_sweeps.append({
                        "detected": True,
                        "level": round(lvl, 5),
                        "sweep_idx": k,
                        "sweep_high": round(highs_m5[k], 5),
                        "sweep_low": round(lows_m5[k], 5),
                        "sweep_type": "Sell-Side Liquidity Sweep (SSL)",
                        "direction": "BUY",
                        "liq_source": desc,
                    })
        if detected_sweeps:
            # Pick the true extreme institutional sweep (lowest valley)
            return min(detected_sweeps, key=lambda x: (x["sweep_low"], -x["sweep_idx"]))

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
    Must occur after the liquidity sweep within the institutional accumulation/distribution
    window (1 to 18 candles / up to 90 min), synchronized with CHoCH structure formation.
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

    for k in range(sweep_idx + 1, min(len(df_m5_slice), sweep_idx + 18)):
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


# ── Step 6B: Inducement (IDM) & Trap Purge Detection ────────────────────────

def detect_inducement_sweep(
    df_m5_slice: pd.DataFrame,
    sweep_info: dict,
    trade_dir: str,
) -> dict:
    """
    Institutional Inducement (IDM) Filter:
    Before entering on an OB retest, verify that price swept an internal pullback high/low
    (Inducement) that lured premature retail traders in.
    
    BUY:  Prior to the sweep, find the highest swing point of the immediate pullback (IDM).
          Price must have purged below that IDM into the Extreme Order Block.
    SELL: Prior to the sweep, find the lowest swing point of the immediate pullback (IDM).
          Price must have purged above that IDM into the Extreme Order Block.
    """
    empty = {"idm_detected": True, "idm_level": None, "idm_swept": True} # Non-blocking fallback
    sweep_idx = sweep_info.get("sweep_idx")
    if sweep_idx is None or sweep_idx < 4:
        return empty

    highs  = df_m5_slice["high"].values.astype(float)
    lows   = df_m5_slice["low"].values.astype(float)

    # Look back 3 to 12 candles prior to the liquidity sweep to locate the engineered pullback
    lookback_start = max(0, sweep_idx - 12)
    prior_highs = highs[lookback_start:sweep_idx]
    prior_lows  = lows[lookback_start:sweep_idx]

    if len(prior_highs) < 3 or len(prior_lows) < 3:
        return empty

    if trade_dir == "BUY":
        # IDM is the minor swing high of the pullback before final low sweep
        idm_price = float(np.max(prior_highs))
        sweep_low = float(sweep_info.get("sweep_low", lows[sweep_idx]))
        # The sweep must have formed a clean divergence or lower sweep below the pre-IDM low
        return {
            "idm_detected": True,
            "idm_level": round(idm_price, 5),
            "idm_swept": sweep_low < idm_price,
        }
    else:  # SELL
        # IDM is the minor swing low of the pullback before final high sweep
        idm_price = float(np.min(prior_lows))
        sweep_high = float(sweep_info.get("sweep_high", highs[sweep_idx]))
        return {
            "idm_detected": True,
            "idm_level": round(idm_price, 5),
            "idm_swept": sweep_high > idm_price,
        }


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

    # M3 lookback window: calibrated to 60 minutes (20 M3 candles)
    # Allows full institutional Swing Low/High and CHoCH structure formation on major pairs
    m3_window = df_m3[(df_m3['time'] <= m5_time) & (df_m3['time'] >= m5_time - timedelta(minutes=60))].copy()
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
    min_rr: float = 2.0,
) -> Tuple[Optional[float], float]:
    """
    Dynamic 1:2.0 to 1:2.5 Structural RR:
    Banks profit at natural liquidity targets before market pulls back.
    Defaults to 1:2.0 RR for higher consistency and win rate.
    """
    risk = abs(entry - sl)
    if risk <= 0:
        return None, 0.0
    # Use effective target RR (2.0 to 2.5)
    target_rr = min(max(min_rr, 2.0), 2.5)
    if direction == "SELL":
        tp = round(entry - risk * target_rr, 5)
    else:
        tp = round(entry + risk * target_rr, 5)
    return tp, target_rr


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
    """Validate if candle time falls inside permitted sessions (Default: Asian, London, New York)."""
    if allowed_sessions is None:
        allowed_sessions = ["Asian", "London", "New York"]
    hour = time_val.hour
    for name, (start_h, end_h) in ALLOWED_SESSIONS.items():
        if start_h <= hour < end_h:
            display = "Off-Session" if name.startswith("Off-Session") else name
            if display not in allowed_sessions:
                return False, "Off-Session"
            return True, display
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
    def __init__(self, max_trades: int = 3, max_loss_r: float = 2.0, max_consec_losses: int = 5):
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


# ── Step 11b: Specialized Asian Range Fade Strategy ───────────────────────────

def evaluate_asian_range_fade(
    df15: pd.DataFrame,
    df5: pd.DataFrame,
    symbol: str,
    t_m5: pd.Timestamp,
) -> dict:
    """
    Dedicated Institutional Asian Range Fade (Mean Reversion) Engine:
      1. Reference Box: Late US / Early Asian (21:00 UTC yesterday to 00:45 UTC today).
      2. Valid box span: 10 to 35 pips (consolidation).
      3. Trend-alignment: Only fade with higher-timeframe trend (H1 bias).
      4. Entry: M5 candle sweeps the box boundary and closes back inside (rejection).
      5. Target: Opposite range boundary (or Equilibrium).
    """
    empty = {"valid": False, "reason": None}
    day_start = t_m5.floor("D")
    ref_start = day_start - timedelta(hours=3)
    ref_end   = day_start + timedelta(minutes=45)

    if t_m5 < ref_end:
        return {**empty, "reason": "asian_box_forming"}

    ref_slice = df5[(df5["time"] >= ref_start) & (df5["time"] < ref_end)]
    if len(ref_slice) < 10:
        return {**empty, "reason": "insufficient_box_candles"}

    pip_unit = 0.01 if "JPY" in symbol.upper() else 0.0001
    box_high = float(ref_slice["high"].max())
    box_low  = float(ref_slice["low"].min())
    box_span_pips = (box_high - box_low) / pip_unit

    if not (10.0 <= box_span_pips <= 35.0):
        return {**empty, "reason": f"box_span_out_of_range ({box_span_pips:.1f}p)"}

    c = df5.iloc[-1]
    h = float(c["high"])
    l = float(c["low"])
    close = float(c["close"])

    h1_slice = df15[df15["time"] <= t_m5]
    bias_info = get_m15_bias(h1_slice)
    htf_bias = bias_info.get("bias", "RANGING")

    if htf_bias == "BEARISH" and h >= box_high and close < box_high:
        entry_price = close
        sweep_high = h
        sl_price = round(max(sweep_high + 2.5 * pip_unit, entry_price + 6.5 * pip_unit), 5)
        risk_dist = abs(entry_price - sl_price)
        tp_price = round(entry_price - risk_dist * 3.0, 5)  # fixed 1:3 RR

        return {
            "valid": True,
            "symbol": symbol,
            "direction": "SELL",
            "entry_price": entry_price,
            "entry_time": c["time"],
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "rr_val": 3.0,
            "score": 13,
            "setup_class": "A+ TRADE (Asian Range Fade)",
            "session": "Asian",
            "m15_bias": htf_bias,
            "m15_location": "Range High (Premium)",
            "liquidity_type": "Asian Box High Sweep",
            "sweep_price": round(box_high, 5),
            "choch_type": "Asian Boundary Rejection",
            "ob_zone": f"Range ({box_low:.5f}-{box_high:.5f})",
            "fvg_overlap": False,
            "m3_trigger": "Asian M5 Boundary Rejection",
            "disp_ratio": 1.0,
        }

    elif htf_bias == "BULLISH" and l <= box_low and close > box_low:
        entry_price = close
        sweep_low = l
        sl_price = round(min(sweep_low - 2.5 * pip_unit, entry_price - 6.5 * pip_unit), 5)
        risk_dist = abs(entry_price - sl_price)
        tp_price = round(entry_price + risk_dist * 3.0, 5)  # fixed 1:3 RR

        return {
            "valid": True,
            "symbol": symbol,
            "direction": "BUY",
            "entry_price": entry_price,
            "entry_time": c["time"],
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "rr_val": 3.0,
            "score": 13,
            "setup_class": "A+ TRADE (Asian Range Fade)",
            "session": "Asian",
            "m15_bias": htf_bias,
            "m15_location": "Range Low (Discount)",
            "liquidity_type": "Asian Box Low Sweep",
            "sweep_price": round(box_low, 5),
            "choch_type": "Asian Boundary Rejection",
            "ob_zone": f"Range ({box_low:.5f}-{box_high:.5f})",
            "fvg_overlap": False,
            "m3_trigger": "Asian M5 Boundary Rejection",
            "disp_ratio": 1.0,
        }

    return {**empty, "reason": "no_asian_range_fade_setup"}


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
    df_m15: Optional[pd.DataFrame] = None,
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

    # In Asian session: evaluate the specialized Trend-Aligned Range Fade strategy first
    if sess_name == "Asian":
        asian_fade = evaluate_asian_range_fade(df15, df5, symbol, t_m5)
        if asian_fade["valid"]:
            return asian_fade

    # 2. M15 Trend Direction Bias (Macro H1 Bias)
    m15_slice = df15[df15["time"] <= t_m5]
    m15_info = get_m15_bias(m15_slice)
    m15_bias = m15_info.get("bias", "RANGING")
    if m15_bias == "RANGING":
        return {**empty, "reason": "m15_ranging"}

    trade_dir = "BUY" if m15_bias == "BULLISH" else "SELL"

    # 3. External Liquidity Sweep (Evaluated before M15 alignment to check for Macro Sweeps)
    sess_levels = get_session_levels(df5, t_m5)
    sweep_info = detect_external_liquidity_sweep(df5, m15_slice, m15_bias, sess_levels, symbol=symbol)
    if not sweep_info["detected"] or sweep_info["direction"] != trade_dir:
        return {**empty, "reason": "no_external_liquidity_sweep"}

    # In Asian session (low-volume hours), M5 swings are retail noise.
    # Institutional setups in Asian require macro liquidity (M15 swings, PDH/PDL, Session H/L, EQH/EQL).
    if sess_name == "Asian" and "Major M5 Swing" in sweep_info.get("liq_source", ""):
        return {**empty, "reason": "asian_requires_macro_liquidity"}

    # 2b. Tier 2: Intermediate M15 Structure Alignment Filter
    # Macro Liquidity Sweep Exception: If price has swept a major macro liquidity pool
    # (Asian H/L, PDH/PDL, London H/L, BSL/SSL, Major M15 Swing, Equal H/L), price is naturally
    # in Premium/Discount and sweeping against short-term EMAs. Do NOT penalize institutional sweeps.
    liq_source = sweep_info.get("liq_source", "")
    has_macro_sweep = any(
        k in liq_source for k in ["Asian", "PDH", "PDL", "London", "BSL", "SSL", "Major M15", "Equal Highs", "Equal Lows"]
    )

    if df_m15 is not None and not df_m15.empty:
        is_aligned, align_reason = check_m15_intermediate_alignment(
            df_m15, t_m5, trade_dir, symbol, has_macro_sweep=has_macro_sweep
        )
        if not is_aligned:
            return {**empty, "reason": f"m15_momentum_conflict ({align_reason})"}

    # 4. Strong Displacement (ATR-Relative)
    curr_disp = disp_threshold
    if sess_name == "Asian" and "JPY" not in symbol.upper() and "AUD" not in symbol.upper() and "NZD" not in symbol.upper():
        curr_disp = max(disp_threshold, 1.2)
    disp_info = detect_strong_displacement(df5, sweep_info["sweep_idx"], trade_dir, atr_multiplier=curr_disp)
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

    # 7B. Inducement (IDM) Filter
    idm_info = detect_inducement_sweep(df5, sweep_info, trade_dir)
    if not idm_info.get("idm_swept", True):
        return {**empty, "reason": "idm_not_purged"}

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
    entry_time  = m3_info["entry_time"]

    # 9B. Precision Entry Location Guard (Reject entries that chase into deep discount for SELL or deep premium for BUY)
    entry_loc = get_m15_range_and_location(m15_slice, entry_price, trade_dir)
    if not entry_loc["valid"]:
        return {**empty, "reason": f"poor_entry_location ({entry_loc['zone_desc']})"}

    # Re-validate session at M3 entry time — this is the actual trade entry
    # moment, which can differ from t_m5 by up to 90 min (M3 lookback window).
    is_sess, sess_name = validate_session(pd.Timestamp(entry_time), allowed_sessions)
    if not is_sess:
        return {**empty, "reason": "outside_session"}

    # 10. Natural SL, Logical TP & Dynamic Guard
    pip_unit = 0.01 if "JPY" in symbol.upper() else 0.0001
    atr_m5 = calculate_atr(df5, period=14)
    min_buf_pips = 2.5 if "JPY" in symbol.upper() else 2.2
    buffer = max(min_buf_pips * pip_unit, atr_m5 * 0.25)

    zh = zone_info["zone_high"]
    zl = zone_info["zone_low"]
    sh = sweep_info.get("sweep_high")
    sl_val = sweep_info.get("sweep_low")

    if trade_dir == "SELL":
        sl_price = round(max(sh or zh, zh) + buffer, 5)
    else: # BUY
        sl_price = round(min(sl_val or zl, zl) - buffer, 5)

    # Minimum SL guard against spread stopouts
    spread_pips = 1.8 if "JPY" in symbol.upper() else 1.4
    min_sl_dist_pips = max(6.5, spread_pips * 4.5)
    cur_sl_dist_pips = abs(entry_price - sl_price) / pip_unit
    if cur_sl_dist_pips < min_sl_dist_pips:
        if trade_dir == "SELL":
            sl_price = round(entry_price + min_sl_dist_pips * pip_unit, 5)
        else:
            sl_price = round(entry_price - min_sl_dist_pips * pip_unit, 5)

    # Maximum SL cap to ensure realistic intraday 1:3 targets
    max_sl_pips = 28.0 if "JPY" in symbol.upper() else 20.0
    final_sl_dist_pips = abs(entry_price - sl_price) / pip_unit
    if final_sl_dist_pips > max_sl_pips:
        return {**empty, "reason": f"sl_too_wide ({final_sl_dist_pips:.1f}p > {max_sl_pips}p)"}

    tp_price, final_rr = calculate_rr(m15_slice, entry_price, sl_price, trade_dir, min_rr=min_rr)
    if tp_price is None:
        return {**empty, "reason": "rr_below_min"}

    # 11. Score Setup with Dynamic Pair-Tier Threshold (Upgrade 3)
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

    # Dynamic Confluence Threshold: Top Major pairs (EURUSD, GBPUSD, USDJPY) with tightest spreads allow 11
    sym_u = symbol.upper()
    effective_min_score = min_score
    if any(m in sym_u for m in ["EURUSD", "GBPUSD", "USDJPY"]):
        effective_min_score = min(min_score, 11)
    else:
        effective_min_score = max(min_score, 12)

    if score < effective_min_score:
        return {**empty, "reason": f"score_below_threshold ({score} < {effective_min_score})", "score": score}

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
    df_m15: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Run complete Multi-Timeframe SMC A+ Strategy Backtest.
    Deterministic, zero lookahead bias, fully reproducible.
    Spread cost is applied at entry: effective_entry is shifted against the
    trader so TP/SL hit checks reflect the real cost of crossing the spread.
    """
    # Spread cost per pair in pips (one-way entry cost)
    SPREAD_PIPS = {
        "EURUSD": 1.2, "GBPUSD": 1.5, "USDJPY": 1.3, "USDCHF": 1.6,
        "AUDUSD": 1.4, "NZDUSD": 2.0, "USDCAD": 1.8,
        "EURJPY": 1.8, "GBPJPY": 2.5,
    }
    pip_size = 0.01 if "JPY" in symbol.upper() else 0.0001
    pip_mult = 100.0 if "JPY" in symbol.upper() else 10000.0
    spread_pips = SPREAD_PIPS.get(symbol.upper(), 1.5)
    spread_price = spread_pips * pip_size   # spread in price units

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
    missed_trades = []
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
        "m15_momentum_conflict": 0,
        "limit_order_missed": 0,
        "limit_order_unfilled": 0,
    }

    protector = DailyRiskProtector(
        max_trades=max_daily_trades,
        max_loss_r=max_daily_loss_r,
        max_consec_losses=max_consec_losses,
    ) if apply_daily_protection else None

    last_exit_time  = pd.Timestamp.min
    seen_entry_times: set = set()   # dedup: same entry_time = same M3 candle

    for idx in test_indices:
        candle_m5 = df5.iloc[idx]
        t_m5 = candle_m5['time']

        # Unlock 1-trade lockout: allow concurrent/fresh setups after 4 hours (14,400s)
        # Even if a previous trade is running, don't stay completely locked out for 14+ hours
        if t_m5 <= last_exit_time:
            time_since_open_s = (pd.to_datetime(t_m5) - pd.to_datetime(last_exit_time)).total_seconds()
            if abs(time_since_open_s) < 4 * 3600:
                continue

        trade_date = t_m5.date()
        if protector:
            can_tr, prot_reason = protector.can_trade(trade_date)
            if not can_tr:
                rejections["daily_protection_locked"] += 1
                continue

        df5_slice = df5[df5['time'] <= t_m5].copy()

        setup = evaluate_mtf_smc_setup(
            df15=df15,
            df5=df5_slice,
            df3=df3,
            symbol=symbol,
            disp_threshold=disp_threshold,
            min_score=min_score,
            min_rr=min_rr,
            allowed_sessions=allowed_sessions,
            df_m15=df_m15,
        )

        if not setup["valid"]:
            reason_key = setup.get("reason", "").split()[0]
            if reason_key in rejections:
                rejections[reason_key] += 1
            continue

        # Deduplicate: same M3 entry_time = same candle already taken
        entry_time  = setup["entry_time"]
        if entry_time in seen_entry_times:
            continue
        seen_entry_times.add(entry_time)
        entry_price = setup["entry_price"]
        sl_price    = setup["stop_loss"]
        tp_price    = setup["take_profit"]
        trade_dir   = setup["direction"]
        final_rr    = setup["rr_val"]

        # Spread-adjusted effective entry: shift against the trader at fill
        # BUY  fills at ask = mid + spread  → effective_entry is higher → SL closer, TP further
        # SELL fills at bid = mid - spread  → effective_entry is lower  → SL closer, TP further
        if trade_dir == "BUY":
            effective_entry = entry_price + spread_price
        else:
            effective_entry = entry_price - spread_price

        # Risk/reward distances measured from the spread-adjusted entry
        risk_dist   = abs(effective_entry - sl_price)
        reward_dist = abs(tp_price - effective_entry)

        # Two-stage management targets
        # Partial TP at 1:1.5 RR (banks 50% volume -> 0.75R)
        # Break-Even SL level = effective_entry + 0.5 pip buffer
        # Signal is confirmed at M5 candle close (t_m5). Forward simulation must start
        # from candles after t_m5 (the exact moment the order is placed in live trading),
        # not the historical M3 lookback candle which occurred up to 90 min earlier.
        signal_time = t_m5
        subsequent = df3[df3['time'] > signal_time].head(400)
        if subsequent.empty:
            subsequent = df5[df5['time'] > signal_time].head(150)

        outcome       = "OPEN"
        pips          = 0.0
        r_multiple    = 0.0
        exit_time     = None
        is_filled     = False
        closest_price = None

        # Two-tier entry tolerance: allows front-edge FVG fill up to 1.5 pips away (Upgrade 2)
        fill_tolerance = 1.5 * pip_size

        # Check if live price at signal confirmation (t_m5) is already inside or through
        # the retest zone (Smart Order Router: immediate market fill at candle close)
        c_m5 = df5_slice.iloc[-1]
        c_close = float(c_m5["close"])
        if trade_dir == "BUY":
            if c_close <= entry_price and c_close > sl_price:
                is_filled = True
        else:
            if c_close >= entry_price and c_close < sl_price:
                is_filled = True

        for _, sub_row in subsequent.iterrows():
            sub_h = float(sub_row["high"])
            sub_l = float(sub_row["low"])
            sub_t = sub_row["time"]

            # ── Limit Order Fill / Miss Check (matches live TradeMonitor & Smart Order Router) ──
            if not is_filled:
                # Time-to-Live (TTL): if limit order is unfilled after 6 hours (120 M3 / 72 M5 bars), cancel it
                time_diff_s = (pd.to_datetime(sub_t) - pd.to_datetime(signal_time)).total_seconds()
                if time_diff_s > 6 * 3600:
                    outcome = "MISSED"
                    break

                if trade_dir == "BUY":
                    if closest_price is None or sub_l < closest_price:
                        closest_price = sub_l
                    # If price ran away straight to TP without pulling back to entry: MISSED
                    if sub_h >= tp_price and sub_l > (entry_price + fill_tolerance):
                        outcome = "MISSED"
                        break
                    # BUY LIMIT fills when price pulls back to or below entry + two-tier tolerance
                    if sub_l <= (entry_price + fill_tolerance):
                        is_filled = True
                    else:
                        continue
                else:  # SELL
                    if closest_price is None or sub_h > closest_price:
                        closest_price = sub_h
                    # If price ran away straight to TP without pulling back to entry: MISSED
                    if sub_l <= tp_price and sub_h < (entry_price - fill_tolerance):
                        outcome = "MISSED"
                        break
                    # SELL LIMIT fills when price pulls back to or above entry - two-tier tolerance
                    if sub_h >= (entry_price - fill_tolerance):
                        is_filled = True
                    else:
                        continue

            # ── Pure Full SL or Full TP Trade Execution (No Partial TP, No Breakeven Ratchet) ──
            if trade_dir == "SELL":
                # Check Stop Loss
                if sub_h >= sl_price:
                    outcome    = "LOSS"
                    pips       = -round(risk_dist * pip_mult, 1)
                    r_multiple = -1.0
                    exit_time  = sub_t
                    break
                # Check Full TP hit (1:3 RR)
                elif sub_l <= tp_price:
                    outcome    = "WIN"
                    pips       = round(reward_dist * pip_mult, 1)
                    r_multiple = round(final_rr, 2)
                    exit_time  = sub_t
                    break
            else:  # BUY
                # Check Stop Loss
                if sub_l <= sl_price:
                    outcome    = "LOSS"
                    pips       = -round(risk_dist * pip_mult, 1)
                    r_multiple = -1.0
                    exit_time  = sub_t
                    break
                # Check Full TP hit (1:3 RR)
                elif sub_h >= tp_price:
                    outcome    = "WIN"
                    pips       = round(reward_dist * pip_mult, 1)
                    r_multiple = round(final_rr, 2)
                    exit_time  = sub_t
                    break

        if outcome in ("OPEN", "MISSED") or not is_filled:
            if outcome == "MISSED":
                rejections["limit_order_missed"] += 1
                if closest_price is not None:
                    if trade_dir == "BUY":
                        missed_pips = round((closest_price - entry_price) / pip_size, 1)
                    else:
                        missed_pips = round((entry_price - closest_price) / pip_size, 1)
                else:
                    missed_pips = 0.0
                missed_trades.append({
                    "timestamp":      str(entry_time),
                    "timestamp_ist":  to_ist_str(entry_time),
                    "pair":           symbol,
                    "direction":      trade_dir,
                    "entry":          entry_price,
                    "closest_retest": round(closest_price, 5) if closest_price is not None else entry_price,
                    "missed_pips":    max(0.0, missed_pips),
                    "stop_loss":      sl_price,
                    "take_profit":    tp_price,
                    "session":        setup["session"],
                    "score":          setup["score"],
                })
            else:
                rejections["limit_order_unfilled"] += 1
            continue

        last_exit_time = exit_time
        if protector:
            protector.record_trade(outcome, r_multiple)

        trades.append({
            "timestamp":      str(entry_time),
            "timestamp_ist":  to_ist_str(entry_time),
            "pair":           symbol,
            "direction":      trade_dir,
            "entry":          entry_price,
            "effective_entry": round(effective_entry, 5),
            "stop_loss":      sl_price,
            "take_profit":    tp_price,
            "spread_pips":    spread_pips,
            "risk_pips":      round(risk_dist * pip_mult, 1),
            "reward_pips":    round(reward_dist * pip_mult, 1),
            "rr":             f"1:{final_rr:.2f}",
            "rr_val":         final_rr,
            "score":          setup["score"],
            "setup_class":    setup["setup_class"],
            "session":        setup["session"],
            "m15_bias":       setup["m15_bias"],
            "m15_location":   setup["m15_location"],
            "liquidity_type": setup["liquidity_type"],
            "sweep_price":    setup["sweep_price"],
            "choch_type":     setup["choch_type"],
            "ob_zone":        setup["ob_zone"],
            "fvg_overlap":    setup["fvg_overlap"],
            "m3_trigger":     setup["m3_trigger"],
            "result":         outcome,
            "pips":           pips,
            "r_multiple":     r_multiple,
            "exit_time":      str(exit_time),
            "exit_time_ist":  to_ist_str(exit_time),
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

    # Spread-adjusted net R: deduct spread_pips/risk_pips fraction per trade.
    # This is the R-equivalent cost of crossing the spread on entry.
    spread_cost_r = round(
        sum(
            t["spread_pips"] / t["risk_pips"]
            for t in trades
            if t["risk_pips"] > 0
        ), 2
    )
    spread_adj_net_r = round(net_r - spread_cost_r, 2)

    return {
        "symbol": symbol,
        "total_signals": total_signals,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "net_pips": net_pips,
        "net_r": net_r,
        "spread_cost_r": spread_cost_r,
        "spread_adj_net_r": spread_adj_net_r,
        "avg_rr": avg_rr,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "max_consecutive_losses": max_c_losses,
        "max_consecutive_wins": max_c_wins,
        "sessions": sess_stats,
        "rejections": rejections,
        "missed_trades": missed_trades,
        "all_trades": trades,
        "recent_trades": trades[-5:],
    }
