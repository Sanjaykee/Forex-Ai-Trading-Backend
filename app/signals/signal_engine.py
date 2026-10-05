from app.analysis.indicators        import add_indicators
from app.analysis.market_structure  import detect_structure, detect_trend
from app.analysis.smc.liquidity_sweep import detect_liquidity_sweep
from app.analysis.smc.bos_choch     import detect_bos_choch
from app.analysis.smc.order_block   import detect_order_block
from app.analysis.smc.fvg           import detect_fvg
from app.analysis.support_resistance import detect_sr_zones
from app.analysis.price_action      import detect_price_action
from app.filters.session_filter     import get_active_session, is_tradeable_session
from app.filters.spread_filter      import is_spread_acceptable
from app.filters.news_filter        import is_news_safe
from app.risk.position_sizer        import calculate_sl_tp, calculate_lot_size
from app.analysis.crt               import detect_crt
from app.filters.killzone_filter    import get_active_killzone
from app.mt5.source_router          import get_fetcher
from app.ml.predictor               import predict_win_probability

CONFIRMATION_WEIGHTS = {
    "Liquidity sweep":   10,
    "BOS confirmed":     15,
    "CHoCH confirmed":   15,
    "Order Block":       15,
    "Fair Value Gap":    10,
    "At Support zone":    5,
    "At Resistance zone": 5,
    "ICT Killzone":      15,
    "CRT":               10,
    "Price Action":       5,
}

def calculate_win_probability(confirmations: list, h4: str, h1: str, kz: dict) -> int:
    score = 20
    for c in confirmations:
        for key, weight in CONFIRMATION_WEIGHTS.items():
            if key in c:
                score += weight
                break
    if kz.get("high_probability"):
        score += 5
    return min(score, 99)


def analyze_pair(symbol: str, risk_usd: float = 1.0, rr_ratio: float = 2.0, data_source: str = "yfinance") -> dict:
    get_candles, get_live_price, get_symbol_info, _ = get_fetcher(data_source)
    base = {
        "symbol":    symbol,
        "direction": "NO TRADE",
        "reason":    [],
        "session":   get_active_session(),
    }

    if not is_tradeable_session():
        base["reason"].append("Asian session — no trade")
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

    df_h4  = get_candles(symbol, "H4", 200)
    df_h1  = get_candles(symbol, "H1", 200)
    df_m15 = get_candles(symbol, "M15", 200)

    if df_h4.empty or df_h1.empty or df_m15.empty:
        base["reason"].append("No candle data")
        return base

    df_h4  = add_indicators(df_h4)
    df_h1  = add_indicators(df_h1)
    df_m15 = add_indicators(df_m15)

    h4_structure = detect_structure(df_h4)
    if h4_structure == "RANGING":
        base["reason"].append("H4 ranging — no trade")
        return base

    h1_trend = detect_trend(df_h1)
    if h1_trend == "RANGING":
        base["reason"].append("H1 ranging — no trade")
        return base

    if h4_structure != h1_trend:
        base["reason"].append(f"H4 {h4_structure} vs H1 {h1_trend} — conflict")
        return base

    direction = "BUY" if h4_structure == "BULLISH" else "SELL"

    sweep = detect_liquidity_sweep(df_m15)
    bos   = detect_bos_choch(df_m15)
    ob    = detect_order_block(df_m15)
    fvg   = detect_fvg(df_m15)
    sr    = detect_sr_zones(df_m15)
    pa    = detect_price_action(df_m15)
    crt   = detect_crt(df_m15)
    kz    = get_active_killzone()

    confirmations = []
    if sweep["detected"] and sweep["direction"] == h4_structure:
        confirmations.append("Liquidity sweep")
    if bos["bos"] and bos["direction"] == h4_structure:
        confirmations.append("BOS confirmed")
    if bos["choch"] and bos["direction"] == h4_structure:
        confirmations.append("CHoCH confirmed")
    if ob["detected"] and ob["direction"] == h4_structure:
        confirmations.append("Order Block")
    if fvg["detected"] and fvg["direction"] == h4_structure:
        confirmations.append("Fair Value Gap")
    if direction == "BUY" and sr["at_support"]:
        confirmations.append("At Support zone")
    if direction == "SELL" and sr["at_resistance"]:
        confirmations.append("At Resistance zone")
    if pa["direction"] == h4_structure:
        confirmations.append(f"Price Action: {pa['pattern']}")
    if crt["detected"] and crt["direction"] == ("BULLISH" if direction == "BUY" else "BEARISH"):
        confirmations.append(f"CRT: {crt['pattern']}")
    if kz["active"]:
        confirmations.append(f"ICT Killzone: {kz['name']}")

    if len(confirmations) < 3:
        base["reason"].append(f"Only {len(confirmations)}/3 confirmations — no trade")
        return base

    # --- Trade levels ---
    last         = df_m15.iloc[-1]
    market_price = round(float(price_info["ask"] if direction == "BUY" else price_info["bid"]), 5)
    atr          = float(last["atr"])

    # Precision Sniper Entry at 50% Equilibrium & Wick Stop Loss
    pip_size = 0.01 if "JPY" in symbol else 0.0001
    buffer   = 1.0 * pip_size

    entry      = market_price
    entry_type = "market"

    if ob["detected"] and ob["direction"] == h4_structure:
        zh = float(ob["zone_high"]) if ob.get("zone_high") is not None else market_price
        zl = float(ob["zone_low"]) if ob.get("zone_low") is not None else market_price
        midpoint = round((zh + zl) / 2.0, 5)
        if abs(market_price - midpoint) <= atr * 1.5:
            entry      = midpoint
            entry_type = "limit_ob_50"

        # Sniper Stop Loss at Order Block Wick
        if direction == "BUY":
            sl = zl - buffer
            sl_dist = entry - sl
        else:
            sl = zh + buffer
            sl_dist = sl - entry
    elif fvg["detected"] and fvg["direction"] == h4_structure:
        gh = float(fvg.get("gap_high", market_price))
        gl = float(fvg.get("gap_low", market_price))
        midpoint = round((gh + gl) / 2.0, 5)
        if abs(market_price - midpoint) <= atr * 1.5:
            entry      = midpoint
            entry_type = "limit_fvg_50"

        if direction == "BUY":
            sl = gl - buffer
            sl_dist = entry - sl
        else:
            sl = gh + buffer
            sl_dist = sl - entry
    else:
        sl_dist = max(atr * 1.2, 0.0004 if "JPY" not in symbol else 0.04)
        sl = entry - sl_dist if direction == "BUY" else entry + sl_dist

    # Clamp SL between 3.5 pips and 12.0 pips for institutional sniper protection
    min_sl  = 3.5 * pip_size
    max_sl  = 12.0 * pip_size
    sl_dist = min(max_sl, max(min_sl, abs(sl_dist)))

    if direction == "BUY":
        sl  = round(entry - sl_dist, 5)
        tp  = round(entry + (sl_dist * rr_ratio), 5)
        # Dynamic Ladder Milestones with 0.5 RR stepped trailing offset
        m1_rr = rr_ratio
        m2_rr = rr_ratio + 1.0
        m3_rr = rr_ratio + 2.0
        m1_sl_rr = round(m1_rr - 0.5, 1)
        m2_sl_rr = round(m2_rr - 0.5, 1)

        m1_tp = round(entry + (sl_dist * m1_rr), 5)
        m1_sl = round(entry + (sl_dist * m1_sl_rr), 5)
        m2_tp = round(entry + (sl_dist * m2_rr), 5)
        m2_sl = round(entry + (sl_dist * m2_sl_rr), 5)
        m3_tp = round(entry + (sl_dist * m3_rr), 5)
    else:
        sl  = round(entry + sl_dist, 5)
        tp  = round(entry - (sl_dist * rr_ratio), 5)
        m1_rr = rr_ratio
        m2_rr = rr_ratio + 1.0
        m3_rr = rr_ratio + 2.0
        m1_sl_rr = round(m1_rr - 0.5, 1)
        m2_sl_rr = round(m2_rr - 0.5, 1)

        m1_tp = round(entry - (sl_dist * m1_rr), 5)
        m1_sl = round(entry - (sl_dist * m1_sl_rr), 5)
        m2_tp = round(entry - (sl_dist * m2_rr), 5)
        m2_sl = round(entry - (sl_dist * m2_sl_rr), 5)
        m3_tp = round(entry - (sl_dist * m3_rr), 5)

    sl_pips  = round(sl_dist / pip_size, 1)
    tp_pips  = round((sl_dist * rr_ratio) / pip_size, 1)
    tp1_pips = round((sl_dist * m1_rr) / pip_size, 1)

    symbol_info = get_symbol_info(symbol)
    lot         = calculate_lot_size(risk_usd, entry, sl, symbol_info, symbol=symbol)
    ml_prob     = predict_win_probability(df_h4, df_h1, df_m15, confirmations, direction)
    win_prob    = ml_prob if ml_prob is not None else calculate_win_probability(confirmations, h4_structure, h1_trend, kz)

    current_price = round(float(price_info["bid"]), 5)

    # --- Entry validity check ---
    price_distance = abs(current_price - entry)

    if entry_type == "market":
        entry_valid  = True
        entry_status = "Enter now at market price"
    elif direction == "BUY" and current_price < entry:
        entry_valid  = True
        entry_status = f"Wait — price needs to pull back to {entry}"
    elif direction == "SELL" and current_price > entry:
        entry_valid  = True
        entry_status = f"Wait — price needs to pull back to {entry}"
    elif price_distance <= atr:
        entry_valid  = True
        entry_status = f"Valid — price within {round(price_distance / pip_size, 1)} pips of entry"
    else:
        entry_valid  = False
        entry_status = f"Stale — price moved {round(price_distance / pip_size, 1)} pips from entry. Wait for NY Open."

    return {
        "symbol":        symbol,
        "direction":     direction,
        "strategy":      "SMC Precision Sniper (Dynamic Ladder)",
        "current_price": current_price,
        "entry_price":   entry,
        "entry_type":    entry_type,
        "entry_valid":   entry_valid,
        "entry_status":  entry_status,
        "stop_loss":     sl,
        "take_profit":   m1_tp,
        "tp1":           m1_tp,
        "tp1_action":    f"Milestone 1 (1:{m1_rr} RR): Close 50% & Ratchet SL to +{m1_sl_rr} RR ({m1_sl})",
        "sl_pips":       sl_pips,
        "tp_pips":       tp_pips,
        "tp1_pips":      tp1_pips,
        "rr_ratio":      f"1:{rr_ratio}",
        "lot_size":      lot,
        "risk_usd":      risk_usd,
        "h4_trend":      h4_structure,
        "h1_trend":      h1_trend,
        "session":       get_active_session(),
        "spread":        spread,
        "confirmations": confirmations,
        "reason":        confirmations,
        "win_probability": win_prob,
        "ml_powered":    ml_prob is not None,
        "dynamic_ladder": {
            "milestone_1": {
                "target_rr": f"1:{m1_rr}",
                "target_price": m1_tp,
                "action": "Close 50% position",
                "ratchet_sl_price": m1_sl,
                "ratchet_sl_rr": f"+{m1_sl_rr} RR",
            },
            "milestone_2": {
                "target_rr": f"1:{m2_rr}",
                "target_price": m2_tp,
                "action": "Close 25% position",
                "ratchet_sl_price": m2_sl,
                "ratchet_sl_rr": f"+{m2_sl_rr} RR",
            },
            "milestone_3": {
                "target_rr": f"1:{m3_rr}",
                "target_price": m3_tp,
                "action": "Close remaining 25% runner (or trail with 0.5 RR offset)",
            },
        },
    }
