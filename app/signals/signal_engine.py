"""
Signal Engine — 6-Step SMC Entry Confirmation Chain

Step 1  D1 bias          — top-of-chain direction filter
Step 2  H4 structure     — macro bias must match D1
Step 3  H1 trend         — intermediate trend must not oppose H4
Step 4  M15 sweep + BOS  — liquidity sweep then structure break (CHoCH preferred)
Step 5  M15 OB/FVG zone  — price must be at an institutional entry zone
Step 6  M3 trigger       — bullish/bearish engulfing OR M3 CHoCH inside the zone

Every signal includes a `training_label` dict so outputs can be used
directly as AI training data without hindsight labeling.
"""
import logging
from app.analysis.indicators          import add_indicators
from app.analysis.market_structure    import detect_structure, detect_trend
from app.analysis.smc.liquidity_sweep import detect_liquidity_sweep, detect_inducement_sweep
from app.analysis.smc.bos_choch      import detect_bos_choch
from app.analysis.smc.order_block    import detect_order_block
from app.analysis.smc.fvg            import detect_fvg
from app.analysis.support_resistance import detect_sr_zones
from app.analysis.price_action       import detect_price_action
from app.filters.session_filter      import get_active_session, is_tradeable_session
from app.filters.spread_filter       import is_spread_acceptable
from app.filters.news_filter         import is_news_safe
from app.risk.position_sizer         import calculate_sl_tp, calculate_lot_size
from app.analysis.crt                import detect_crt
from app.filters.killzone_filter     import get_active_killzone
from app.mt5.source_router           import get_fetcher
from app.ml.predictor                import predict_win_probability

logger = logging.getLogger(__name__)

MIN_WIN_PROB    = 65   # minimum score to emit a signal
MIN_SL_PIPS    = 5    # skip if SL < 5 pips (spread eats too much R)


# ── Step 6 helpers — M3 confirmation ─────────────────────────────────────────

def _is_engulfing(df, direction: str) -> bool:
    """
    Bullish engulfing (BUY):
      - current candle is bullish (close > open)
      - current body > previous body
      - current close > previous open  (engulfs the previous candle)
    Bearish engulfing (SELL): mirror rules.
    Requires at least 2 candles.
    """
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    prev_body = abs(float(prev["close"]) - float(prev["open"]))
    curr_body = abs(float(curr["close"]) - float(curr["open"]))
    if direction == "BUY":
        return (
            float(curr["close"]) > float(curr["open"])   # bullish candle
            and curr_body > prev_body                     # larger body
            and float(curr["close"]) > float(prev["open"])  # engulfs prev
        )
    # SELL
    return (
        float(curr["close"]) < float(curr["open"])       # bearish candle
        and curr_body > prev_body
        and float(curr["close"]) < float(prev["open"])   # engulfs prev
    )


def _m3_choch(df, direction: str) -> bool:
    """
    M3 CHoCH:
      BUY  — current close breaks above the last swing lower-high in the last 10 candles.
      SELL — current close breaks below the last swing lower-low.
    A swing high = candle whose high > both neighbours.
    """
    if len(df) < 5:
        return False
    curr_close = float(df.iloc[-1]["close"])
    window = df.iloc[-10:-1]  # last 10 candles excluding current

    if direction == "BUY":
        # find the most recent swing high (lower-high) in the window
        for j in range(len(window) - 2, 0, -1):
            h = float(window.iloc[j]["high"])
            if h > float(window.iloc[j - 1]["high"]) and h > float(window.iloc[j + 1]["high"]):
                return curr_close > h   # close breaks above it = CHoCH up
    else:
        # find the most recent swing low (higher-low) in the window
        for j in range(len(window) - 2, 0, -1):
            l = float(window.iloc[j]["low"])
            if l < float(window.iloc[j - 1]["low"]) and l < float(window.iloc[j + 1]["low"]):
                return curr_close < l   # close breaks below it = CHoCH down
    return False


def _detect_m3_confirmation(df_m3, direction: str, ob_high: float, ob_low: float) -> dict:
    """
    Step 6: M3 confirmation inside the M15 OB zone.
    Price must be inside the zone, then either:
      - Engulfing candle in trade direction, OR
      - M3 CHoCH in trade direction
    Returns {"confirmed": bool, "trigger": str}
    """
    if df_m3 is None or len(df_m3) < 5:
        return {"confirmed": False, "trigger": None}

    current_price = float(df_m3.iloc[-1]["close"])
    in_zone = ob_low <= current_price <= ob_high

    if not in_zone:
        return {"confirmed": False, "trigger": None}

    if _is_engulfing(df_m3, direction):
        return {"confirmed": True, "trigger": "M3 Engulfing"}

    if _m3_choch(df_m3, direction):
        return {"confirmed": True, "trigger": "M3 CHoCH"}

    return {"confirmed": False, "trigger": None}


# ── Confirmation score ────────────────────────────────────────────────────────

CONFIRMATION_WEIGHTS = {
    "Liquidity Sweep":  10,
    "BOS":              15,
    "CHoCH":            20,
    "Order Block":      15,
    "Fair Value Gap":   10,
    "Inducement Sweep": 20,
    "Support Zone":      5,
    "Resistance Zone":   5,
    "ICT Killzone":     15,
    "M3 Engulfing":     20,
    "M3 CHoCH":         25,
    "CRT":              10,
    "Price Action":      5,
}


def _score_confirmations(confirmations: list, kz: dict) -> int:
    score = 20  # base score
    for c in confirmations:
        for key, weight in CONFIRMATION_WEIGHTS.items():
            if key.lower() in c.lower():
                score += weight
                break
    if kz.get("high_probability"):
        score += 5
    return min(score, 99)


# ── Pip / SL helpers ──────────────────────────────────────────────────────────

def _pip_config(symbol: str) -> tuple:
    """Return (pip_size, min_sl_pips, max_sl_pips)."""
    s = symbol.upper()
    if any(x in s for x in ("XAU", "GOLD")):   return 0.01,  150, 400
    if any(x in s for x in ("XAG", "SILVER")): return 0.001, 100, 300
    if any(x in s for x in ("OIL", "WTI")):    return 0.01,  100, 300
    if any(x in s for x in ("BTC", "ETH")):    return 1.0,   200, 800
    if "JPY" in s:                              return 0.01,    3,  15
    return 0.0001, 3, 15


# ── Main signal function ──────────────────────────────────────────────────────

def analyze_pair(symbol: str, risk_usd: float = 1.0, rr_ratio: float = 2.0, data_source: str = "auto") -> dict:
    """
    Run the full 6-step confirmation chain for a symbol.
    Returns a signal dict with `confirmations`, `confirmation_steps`,
    `win_probability`, and `training_label` for AI training.
    """
    get_candles, get_live_price, get_symbol_info, _ = get_fetcher(data_source)

    base = {
        "symbol":    symbol,
        "direction": "NO TRADE",
        "reason":    [],
        "session":   get_active_session(),
    }

    # ── Pre-flight filters ────────────────────────────────────────────────────
    if not is_tradeable_session():
        base["reason"].append("Asian session — no trade")
        return base

    kz = get_active_killzone()
    if not kz["active"]:
        base["reason"].append("Not in London Open or NY Open killzone")
        return base

    price_info = get_live_price(symbol)
    if not price_info:
        base["reason"].append("Cannot get live price")
        return base

    spread = price_info.get("spread", 99)
    if not is_spread_acceptable(spread):
        base["reason"].append(f"Spread too high: {spread} pips")
        return base

    if not is_news_safe():
        base["reason"].append("High-impact news within 30 min")
        return base

    # ── Load candles ──────────────────────────────────────────────────────────
    df_d1  = get_candles(symbol, "D1",  200)
    df_h4  = get_candles(symbol, "H4",  200)
    df_h1  = get_candles(symbol, "H1",  200)
    df_m15 = get_candles(symbol, "M15", 200)
    df_m3  = get_candles(symbol, "M3",  100)

    if df_h4.empty or df_h1.empty or df_m15.empty:
        base["reason"].append("No candle data available")
        return base

    df_h4  = add_indicators(df_h4)
    df_h1  = add_indicators(df_h1)
    df_m15 = add_indicators(df_m15)
    if not df_d1.empty:
        df_d1 = add_indicators(df_d1)

    confirmation_steps = {}   # tracks each step result for training label

    # ── Step 1: D1 bias ───────────────────────────────────────────────────────
    d1_bias = detect_structure(df_d1) if not df_d1.empty else "ANY"
    confirmation_steps["step1_d1_bias"] = d1_bias
    if d1_bias == "RANGING":
        base["reason"].append("Step 1 FAIL: D1 ranging — no institutional bias")
        return base

    # ── Step 2: H4 structure must match D1 ───────────────────────────────────
    h4_structure = detect_structure(df_h4)
    confirmation_steps["step2_h4_structure"] = h4_structure
    if h4_structure == "RANGING":
        base["reason"].append("Step 2 FAIL: H4 ranging")
        return base
    if d1_bias != "ANY" and h4_structure != d1_bias:
        base["reason"].append(f"Step 2 FAIL: H4 {h4_structure} conflicts D1 {d1_bias}")
        return base

    # ── Step 3: H1 trend must not oppose H4 ──────────────────────────────────
    h1_trend = detect_trend(df_h1)
    confirmation_steps["step3_h1_trend"] = h1_trend
    if h1_trend not in (h4_structure, "RANGING", "ANY"):
        base["reason"].append(f"Step 3 FAIL: H1 {h1_trend} opposes H4 {h4_structure}")
        return base

    direction = "BUY" if h4_structure == "BULLISH" else "SELL"

    # ── Step 4: M15 liquidity sweep + structure break ─────────────────────────
    sweep = detect_liquidity_sweep(df_m15)
    bos   = detect_bos_choch(df_m15)

    sweep_ok = sweep["detected"] and sweep["direction"] == h4_structure
    bos_ok   = (bos["choch"] or bos["bos"]) and bos["direction"] == h4_structure

    confirmation_steps["step4_sweep"]    = sweep_ok
    confirmation_steps["step4_bos_choch"] = "CHoCH" if bos["choch"] else ("BOS" if bos["bos"] else "None")

    if not sweep_ok:
        base["reason"].append("Step 4 FAIL: No M15 liquidity sweep in trade direction")
        return base
    if not bos_ok:
        base["reason"].append("Step 4 FAIL: No M15 BOS/CHoCH after sweep")
        return base

    # ── Step 5: M15 OB or FVG zone ───────────────────────────────────────────
    ob  = detect_order_block(df_m15)
    fvg = detect_fvg(df_m15)
    sr  = detect_sr_zones(df_m15)

    ob_ok  = ob["detected"]  and ob["direction"]  == h4_structure
    fvg_ok = fvg["detected"] and fvg["direction"] == h4_structure

    confirmation_steps["step5_ob"]  = ob_ok
    confirmation_steps["step5_fvg"] = fvg_ok

    if not ob_ok and not fvg_ok:
        base["reason"].append("Step 5 FAIL: No OB or FVG zone in trade direction")
        return base

    # Determine zone boundaries for Step 6
    if ob_ok:
        zone_high = float(ob["zone_high"])
        zone_low  = float(ob["zone_low"])
        zone_type = "Order Block"
    else:
        zone_high = float(fvg["gap_high"])
        zone_low  = float(fvg["gap_low"])
        zone_type = "Fair Value Gap"

    # ── Step 5b: Inducement sweep (optional — high-conviction bonus) ──────────
    # Price makes a fake move sweeping a nearby swing level then returns to OB.
    # Adds +20 to win probability and "Inducement Sweep" to confirmations.
    inducement = detect_inducement_sweep(df_m15, zone_high, zone_low, direction)
    confirmation_steps["step5b_inducement"] = inducement["detected"]

    # ── Step 6: M3 confirmation inside the zone ───────────────────────────────
    m3_result = _detect_m3_confirmation(df_m3, direction, zone_high, zone_low)
    confirmation_steps["step6_m3_trigger"] = m3_result["trigger"] or "None"

    if not m3_result["confirmed"]:
        base["reason"].append(f"Step 6 FAIL: No M3 engulfing or CHoCH inside {zone_type} zone")
        return base

    # ── Build confirmations list ──────────────────────────────────────────────
    confirmations = ["Liquidity Sweep"]
    confirmations.append("CHoCH" if bos["choch"] else "BOS")
    confirmations.append(zone_type)
    confirmations.append(m3_result["trigger"])
    if inducement["detected"]:
        confirmations.append("Inducement Sweep")
    if kz["active"]:
        confirmations.append(f"ICT Killzone: {kz['name']}")

    # Optional extras
    pa  = detect_price_action(df_m15)
    crt = detect_crt(df_m15)
    if pa["direction"] == h4_structure:
        confirmations.append(f"Price Action: {pa['pattern']}")
    if crt["detected"] and crt["direction"] == ("BULLISH" if direction == "BUY" else "BEARISH"):
        confirmations.append(f"CRT: {crt['pattern']}")
    if direction == "BUY"  and sr["at_support"]:    confirmations.append("Support Zone")
    if direction == "SELL" and sr["at_resistance"]: confirmations.append("Resistance Zone")

    # ── Win probability ───────────────────────────────────────────────────────
    ml_prob  = predict_win_probability(df_h4, df_h1, df_m15, confirmations, direction)
    win_prob = ml_prob if ml_prob is not None else _score_confirmations(confirmations, kz)

    if win_prob < MIN_WIN_PROB:
        base["reason"].append(f"Win probability {win_prob}% below {MIN_WIN_PROB}% threshold")
        return base

    # ── Entry, SL, TP ─────────────────────────────────────────────────────────
    pip_size, min_sl_pips, max_sl_pips = _pip_config(symbol)
    buffer       = 1.0 * pip_size
    market_price = round(float(price_info["ask"] if direction == "BUY" else price_info["bid"]), 5)
    atr          = float(df_m15.iloc[-1]["atr"])

    # Entry at 50% of zone (OB midpoint or FVG midpoint)
    midpoint   = round((zone_high + zone_low) / 2.0, 5)
    entry      = midpoint if abs(market_price - midpoint) <= atr * 1.5 else market_price
    entry_type = f"limit_{zone_type.lower().replace(' ', '_')}_50"

    # SL beyond the sweep low/high (Step 4 sweep level) + buffer
    sweep_level = float(sweep.get("level", zone_low if direction == "BUY" else zone_high))
    if direction == "BUY":
        sl      = round(sweep_level - buffer, 5)
        sl_dist = abs(entry - sl)
    else:
        sl      = round(sweep_level + buffer, 5)
        sl_dist = abs(sl - entry)

    # Clamp SL to instrument range
    sl_dist = min(max_sl_pips * pip_size, max(min_sl_pips * pip_size, sl_dist))

    # Skip if SL too tight (spread eats too much R)
    sl_pips = round(sl_dist / pip_size, 1)
    if sl_pips < MIN_SL_PIPS:
        base["reason"].append(f"SL {sl_pips} pips too tight — spread would eat >15% of R")
        return base

    if direction == "BUY":
        sl = round(entry - sl_dist, 5)
        tp = round(entry + sl_dist * rr_ratio, 5)
        m1_tp = round(entry + sl_dist * rr_ratio, 5)
        m1_sl = round(entry + sl_dist * (rr_ratio - 0.5), 5)
        m2_tp = round(entry + sl_dist * (rr_ratio + 1.0), 5)
        m2_sl = round(entry + sl_dist * (rr_ratio + 0.5), 5)
        m3_tp = round(entry + sl_dist * (rr_ratio + 2.0), 5)
    else:
        sl = round(entry + sl_dist, 5)
        tp = round(entry - sl_dist * rr_ratio, 5)
        m1_tp = round(entry - sl_dist * rr_ratio, 5)
        m1_sl = round(entry - sl_dist * (rr_ratio - 0.5), 5)
        m2_tp = round(entry - sl_dist * (rr_ratio + 1.0), 5)
        m2_sl = round(entry - sl_dist * (rr_ratio + 0.5), 5)
        m3_tp = round(entry - sl_dist * (rr_ratio + 2.0), 5)

    tp_pips  = round(sl_dist * rr_ratio / pip_size, 1)
    symbol_info = get_symbol_info(symbol)
    lot         = calculate_lot_size(risk_usd, entry, sl, symbol_info, symbol=symbol)

    # Entry validity check
    current_price  = round(float(price_info["bid"]), 5)
    distance_pips  = round(abs(current_price - entry) / pip_size, 1)
    max_stale_pips = 50 if min_sl_pips >= 100 else 10
    if entry_type.startswith("limit") and distance_pips > max_stale_pips:
        entry_valid  = False
        entry_status = f"Stale — price moved {distance_pips} pips from entry zone. Wait for next setup."
    else:
        entry_valid  = True
        entry_status = f"Valid — enter at {entry} ({zone_type} 50% level)"

    # ── Training label ────────────────────────────────────────────────────────
    # outcome is filled in later by the backtest/trade monitor after SL or TP is hit.
    # All input features are captured here at signal time (no hindsight).
    training_label = {
        "symbol":          symbol,
        "direction":       direction,
        "d1_bias":         d1_bias,
        "h4_structure":    h4_structure,
        "h1_trend":        h1_trend,
        "sweep_detected":  sweep_ok,
        "bos_type":        confirmation_steps["step4_bos_choch"],
        "zone_type":       zone_type,
        "inducement":      inducement["detected"],
        "m3_trigger":      m3_result["trigger"],
        "kz_active":       kz["active"],
        "kz_name":         kz.get("name"),
        "kz_high_prob":    kz.get("high_probability", False),
        "sl_pips":         sl_pips,
        "rr_ratio":        rr_ratio,
        "win_probability": win_prob,
        "spread_pips":     spread,
        "num_confirmations": len(confirmations),
        "outcome":         None,   # filled by trade monitor: "WIN" / "LOSS"
        "outcome_pips":    None,   # filled by trade monitor
    }

    return {
        "symbol":          symbol,
        "direction":       direction,
        "strategy":        "SMC 6-Step Confirmation (M3 Trigger)",
        "current_price":   current_price,
        "entry_price":     entry,
        "entry_type":      entry_type,
        "entry_valid":     entry_valid,
        "entry_status":    entry_status,
        "stop_loss":       sl,
        "take_profit":     tp,
        "sl_pips":         sl_pips,
        "tp_pips":         tp_pips,
        "rr_ratio":        f"1:{rr_ratio}",
        "lot_size":        lot,
        "risk_usd":        risk_usd,
        "d1_bias":         d1_bias,
        "h4_trend":        h4_structure,
        "h1_trend":        h1_trend,
        "session":         get_active_session(),
        "killzone":        kz["name"],
        "spread":          spread,
        "zone_type":       zone_type,
        "zone_high":       zone_high,
        "zone_low":        zone_low,
        "confirmations":   confirmations,
        "confirmation_steps": confirmation_steps,
        "win_probability": win_prob,
        "ml_powered":      ml_prob is not None,
        "dynamic_ladder": {
            "milestone_1": {
                "target_rr":        f"1:{rr_ratio}",
                "target_price":     m1_tp,
                "action":           "Close 50% position",
                "ratchet_sl_price": m1_sl,
                "ratchet_sl_rr":    f"+{round(rr_ratio - 0.5, 1)} RR",
            },
            "milestone_2": {
                "target_rr":        f"1:{round(rr_ratio + 1.0, 1)}",
                "target_price":     m2_tp,
                "action":           "Close 25% position",
                "ratchet_sl_price": m2_sl,
                "ratchet_sl_rr":    f"+{round(rr_ratio + 0.5, 1)} RR",
            },
            "milestone_3": {
                "target_rr":    f"1:{round(rr_ratio + 2.0, 1)}",
                "target_price": m3_tp,
                "action":       "Close remaining 25% runner",
            },
        },
        "training_label":  training_label,
        "reason":          confirmations,
    }
