"""
Backtest Engine — Simulates SMC strategy execution candle-by-candle on intraday & swing timeframes.
Computes Win/Loss ratio, Pips gained, Session analytics, and Multi-Timeframe Institutional Win Probability.
"""
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from app.analysis.indicators import add_indicators
from app.analysis.market_structure import detect_structure, detect_trend
from app.analysis.smc.liquidity_sweep import detect_liquidity_sweep
from app.analysis.smc.bos_choch import detect_bos_choch
from app.analysis.smc.order_block import detect_order_block
from app.analysis.smc.fvg import detect_fvg
from app.analysis.support_resistance import detect_sr_zones
from app.analysis.price_action import detect_price_action

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"

# Universal Multi-Timeframe Triad Hierarchy Matrix
TIMEFRAME_HIERARCHY = {
    # ── Scalping Tier ──
    "M1":  {"bias": "H1", "trend": "M15", "style": "Micro Scalp"},
    "M3":  {"bias": "H1", "trend": "M15", "style": "Fast Scalp"},
    "M5":  {"bias": "H1", "trend": "M15", "style": "Intraday Scalp"},

    # ── Day Trading Tier ──
    "M15": {"bias": "H4", "trend": "H1",  "style": "Intraday Trend"},
    "M30": {"bias": "H4", "trend": "H1",  "style": "Day Trading"},

    # ── Swing & Position Tier ──
    "H1":  {"bias": "D1", "trend": "H4",  "style": "Short-term Swing"},
    "H4":  {"bias": "D1", "trend": "H4",  "style": "Swing Trading"},
    "D1":  {"bias": "D1", "trend": "D1",  "style": "Position Trading"},
}

CONFIRMATION_WEIGHTS = {
    "Liquidity Sweep":   10,
    "Liquidity sweep":   10,
    "BOS":               15,
    "BOS confirmed":     15,
    "CHoCH":             15,
    "CHoCH confirmed":   15,
    "Order Block":       15,
    "Fair Value Gap":    10,
    "Support Zone":       5,
    "At Support zone":    5,
    "Resistance Zone":    5,
    "At Resistance zone": 5,
    "ICT Killzone":      15,
    "CRT":               10,
    "Price Action":       5,
}

# ICT Killzones in UTC
KILLZONES = {
    "London Open":    (7,  10),  # 07:00 - 10:00 UTC
    "New York Open":  (12, 15),  # 12:00 - 15:00 UTC
    "London Close":   (15, 17),  # 15:00 - 17:00 UTC
    "Asian Range":    (0,  4),   # 00:00 - 04:00 UTC
}
HIGH_PROBABILITY_KZ = ["London Open", "New York Open"]


def _get_killzone_for_hour(hour: int) -> dict:
    for name, (start, end) in KILLZONES.items():
        if start <= hour < end:
            return {
                "active": True,
                "name": name,
                "high_probability": name in HIGH_PROBABILITY_KZ,
            }
    return {"active": False, "name": None, "high_probability": False}


def calculate_win_probability(confirmations: list, kz: dict) -> int:
    score = 20
    for c in confirmations:
        for key, weight in CONFIRMATION_WEIGHTS.items():
            if key.lower() in c.lower():
                score += weight
                break
    if kz.get("high_probability"):
        score += 5
    return min(score, 99)


def _load_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load candles from local sample_data or MT5 fallback."""
    csv_path = SAMPLE_DIR / f"{symbol}_{timeframe}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df["time"] = pd.to_datetime(df["time"])
            return df.sort_values("time").reset_index(drop=True)
        except Exception:
            pass

    try:
        from app.mt5.data_fetcher import get_candles
        return get_candles(symbol, timeframe, 10000)
    except Exception:
        return pd.DataFrame()


def _get_session(hour: int) -> str:
    """Identify trading session by UTC hour."""
    if 7 <= hour <= 11:
        return "London"
    elif 12 <= hour <= 16:
        return "New York"
    elif 17 <= hour <= 21:
        return "NY Close"
    else:
        return "Asian"


def run_backtest(
    symbol: str = "EURUSD",
    timeframe: str = "M3",
    days: int = 7,
    session_filter: Optional[str] = None,
    rr_ratio: float = 2.0,
    min_win_prob: int = 60,
    ladder_mode: bool = True,
) -> dict:
    """
    Run backtest on specified symbol and timeframe for the last `days` days.
    Matches the Scanner API entry setup, timeframe hierarchy, and win probability filtering.
    """
    df = _load_candles(symbol, timeframe)
    if df.empty or len(df) < 50:
        return {"error": f"Insufficient historical candle data for {symbol} on {timeframe}."}

    # Filter for the last N days
    latest_time = df["time"].max()
    cutoff_time = latest_time - pd.Timedelta(days=days)
    df_window = df[df["time"] >= cutoff_time].copy().reset_index(drop=True)

    if len(df_window) < 30:
        df_window = df.tail(1500).copy().reset_index(drop=True)

    df_window = add_indicators(df_window)

    # Resolve Multi-Timeframe Triad Hierarchy
    tf_cfg = TIMEFRAME_HIERARCHY.get(timeframe.upper(), {"bias": "H4", "trend": "H1", "style": "Intraday"})
    bias_tf = tf_cfg["bias"]
    trend_tf = tf_cfg["trend"]

    df_bias = _load_candles(symbol, bias_tf) if timeframe != bias_tf else pd.DataFrame()
    df_trend = _load_candles(symbol, trend_tf) if timeframe != trend_tf else pd.DataFrame()
    if not df_trend.empty:
        df_trend = add_indicators(df_trend)

    pip_mult = 100.0 if "JPY" in symbol else 10000.0
    trades = []
    confs_win_counter = {}

    step = 2 if timeframe in ("M1", "M3") else 1
    total_len = len(df_window)
    last_exit_idx = -1

    for i in range(25, total_len - 30, step):
        # 1. Cooldown & Overlap Prevention
        if i <= last_exit_idx:
            continue

        current_slice = df_window.iloc[:i + 1]
        candle = current_slice.iloc[-1]
        c_time = candle["time"]
        session = _get_session(c_time.hour)

        # 2. Session Filter
        if session_filter:
            if session_filter.upper() not in session.upper():
                continue
        else:
            if session == "Asian" and timeframe in ("M1", "M3", "M5"):
                continue

        # 3. Dual Higher-Timeframe Alignment Gate (Macro Bias == Intermediate Trend)
        macro_bias = "ANY"
        if not df_bias.empty:
            b_slice = df_bias[df_bias["time"] <= c_time].tail(25)
            if len(b_slice) >= 10:
                macro_bias = detect_structure(b_slice)

        inter_trend = "ANY"
        if not df_trend.empty:
            t_slice = df_trend[df_trend["time"] <= c_time].tail(25)
            if len(t_slice) >= 10:
                inter_trend = detect_trend(t_slice)

        # Skip if macro structure is ranging or if there is trend conflict
        if macro_bias in ("RANGING", "ANY") or inter_trend in ("RANGING", "ANY"):
            continue
        if macro_bias != inter_trend:
            continue

        direction = "BUY" if macro_bias == "BULLISH" else "SELL"

        # 4. Lower-Timeframe SMC Detectors
        bos = detect_bos_choch(current_slice)
        ob = detect_order_block(current_slice)
        fvg = detect_fvg(current_slice)
        sweep = detect_liquidity_sweep(current_slice)
        sr = detect_sr_zones(current_slice)
        kz = _get_killzone_for_hour(c_time.hour)

        confs = []
        if bos["bos"] and bos["direction"] == macro_bias:
            confs.append("BOS")
        if bos["choch"] and bos["direction"] == macro_bias:
            confs.append("CHoCH")
        if ob["detected"] and ob["direction"] == macro_bias:
            confs.append("Order Block")
        if fvg["detected"] and fvg["direction"] == macro_bias:
            confs.append("Fair Value Gap")
        if sweep["detected"] and sweep["direction"] == macro_bias:
            confs.append("Liquidity Sweep")
        if direction == "BUY" and sr["at_support"]:
            confs.append("Support Zone")
        if direction == "SELL" and sr["at_resistance"]:
            confs.append("Resistance Zone")
        if kz["active"]:
            confs.append(f"ICT Killzone: {kz['name']}")

        if len(confs) < 2:
            continue

        # 5. Strict Win Probability Gatekeeper (>= min_win_prob, default 60%)
        win_prob = calculate_win_probability(confs, kz)
        if win_prob < min_win_prob:
            continue

        # 6. Precision Sniper Entry at 50% Equilibrium & Wick Stop Loss
        pip_unit = 0.01 if "JPY" in symbol else 0.0001
        buffer = 1.0 * pip_unit
        market_price = float(candle["close"])
        atr = float(candle["atr"]) if "atr" in candle and not np.isnan(candle["atr"]) else (0.0010 if "JPY" not in symbol else 0.10)
        entry_price = market_price
        entry_type = "market"

        if ob["detected"] and ob["direction"] == macro_bias:
            zh = float(ob["zone_high"]) if ob.get("zone_high") is not None else market_price
            zl = float(ob["zone_low"]) if ob.get("zone_low") is not None else market_price
            midpoint = round((zh + zl) / 2.0, 5)
            if abs(market_price - midpoint) <= atr * 1.5:
                entry_price = midpoint
                entry_type = "limit_ob_50"

            # Sniper Stop Loss at Order Block Wick
            if direction == "BUY":
                sl = zl - buffer
                sl_dist = entry_price - sl
            else:
                sl = zh + buffer
                sl_dist = sl - entry_price
        elif fvg["detected"] and fvg["direction"] == macro_bias:
            gh = float(fvg.get("gap_high", market_price))
            gl = float(fvg.get("gap_low", market_price))
            midpoint = round((gh + gl) / 2.0, 5)
            if abs(market_price - midpoint) <= atr * 1.5:
                entry_price = midpoint
                entry_type = "limit_fvg_50"

            if direction == "BUY":
                sl = gl - buffer
                sl_dist = entry_price - sl
            else:
                sl = gh + buffer
                sl_dist = sl - entry_price
        else:
            sl_dist = max(atr * 1.2, 0.0004 if "JPY" not in symbol else 0.04)
            sl = entry_price - sl_dist if direction == "BUY" else entry_price + sl_dist

        # Institutional sanity check: SL between 3.5 pips and 12 pips
        min_sl = 3.5 * pip_unit
        max_sl = 12.0 * pip_unit
        sl_dist = min(max_sl, max(min_sl, abs(sl_dist)))

        if direction == "BUY":
            sl = entry_price - sl_dist
            tp = entry_price + (sl_dist * rr_ratio)
        else:
            sl = entry_price + sl_dist
            tp = entry_price - (sl_dist * rr_ratio)

        # 7. Forward Simulation: Dynamic Profit Laddering & Stepped Offset Trailing
        # - Full 100% volume runs to primary target (rr_ratio, e.g. 1:5 RR) without early TP1 scale-out.
        # - Milestone 1 (rr_ratio): Close 50% volume, ratchet SL to (rr_ratio - 0.5) RR in locked profit, target moves to +1.0 RR.
        # - Milestone 2 (rr_ratio + 1.0): Close 25% volume, ratchet SL to (rr_ratio + 0.5) RR, target moves to +2.0 RR.
        # - Milestone 3 (rr_ratio + 2.0): Close remaining 25% runner (or trail with 0.5 offset).
        # - If market pulls back into ratcheted SL at any time, banked profit is preserved.
        future = df_window.iloc[i + 1: i + 41]
        outcome = "OPEN"
        pips = 0.0
        exit_idx = i + len(future)
        active_sl = sl
        ladder_level = 0
        remaining_vol = 1.0
        realized_r = 0.0
        current_target_r = rr_ratio

        if direction == "BUY":
            active_tp = entry_price + (sl_dist * current_target_r)
        else:
            active_tp = entry_price - (sl_dist * current_target_r)

        for f_step, (_, f_row) in enumerate(future.iterrows()):
            f_high = float(f_row["high"])
            f_low = float(f_row["low"])

            if direction == "BUY":
                # Check Stop Loss first
                if f_low <= active_sl:
                    if ladder_level == 0:
                        outcome = "LOSS"
                        pips = -round(sl_dist * pip_mult, 1)
                    else:
                        outcome = "WIN"
                        trailed_r = (active_sl - entry_price) / sl_dist
                        realized_r += remaining_vol * trailed_r
                        pips = round(sl_dist * realized_r * pip_mult, 1)
                    exit_idx = i + 1 + f_step
                    break

                # Dynamic Milestone Progression
                while f_high >= active_tp and ladder_level < 3:
                    if ladder_level == 0:
                        # Milestone 1 Hit (Primary target reached)
                        ladder_level = 1
                        close_pct = 0.50
                        realized_r += close_pct * current_target_r
                        remaining_vol -= close_pct
                        # Ratchet SL dynamically to current_target_r - 0.5 RR
                        ratchet_r = current_target_r - 0.5
                        active_sl = entry_price + (sl_dist * ratchet_r)
                        current_target_r += 1.0
                        active_tp = entry_price + (sl_dist * current_target_r)
                    elif ladder_level == 1:
                        # Milestone 2 Hit
                        ladder_level = 2
                        close_pct = 0.25
                        realized_r += close_pct * current_target_r
                        remaining_vol -= close_pct
                        ratchet_r = current_target_r - 0.5
                        active_sl = entry_price + (sl_dist * ratchet_r)
                        current_target_r += 1.0
                        active_tp = entry_price + (sl_dist * current_target_r)
                    elif ladder_level == 2:
                        # Milestone 3 Hit
                        ladder_level = 3
                        realized_r += remaining_vol * current_target_r
                        remaining_vol = 0.0
                        outcome = "WIN"
                        pips = round(sl_dist * realized_r * pip_mult, 1)
                        exit_idx = i + 1 + f_step
                        break

                if ladder_level >= 3:
                    break

            else:  # direction == "SELL"
                # Check Stop Loss first
                if f_high >= active_sl:
                    if ladder_level == 0:
                        outcome = "LOSS"
                        pips = -round(sl_dist * pip_mult, 1)
                    else:
                        outcome = "WIN"
                        trailed_r = (entry_price - active_sl) / sl_dist
                        realized_r += remaining_vol * trailed_r
                        pips = round(sl_dist * realized_r * pip_mult, 1)
                    exit_idx = i + 1 + f_step
                    break

                # Dynamic Milestone Progression
                while f_low <= active_tp and ladder_level < 3:
                    if ladder_level == 0:
                        ladder_level = 1
                        close_pct = 0.50
                        realized_r += close_pct * current_target_r
                        remaining_vol -= close_pct
                        ratchet_r = current_target_r - 0.5
                        active_sl = entry_price - (sl_dist * ratchet_r)
                        current_target_r += 1.0
                        active_tp = entry_price - (sl_dist * current_target_r)
                    elif ladder_level == 1:
                        ladder_level = 2
                        close_pct = 0.25
                        realized_r += close_pct * current_target_r
                        remaining_vol -= close_pct
                        ratchet_r = current_target_r - 0.5
                        active_sl = entry_price - (sl_dist * ratchet_r)
                        current_target_r += 1.0
                        active_tp = entry_price - (sl_dist * current_target_r)
                    elif ladder_level == 2:
                        ladder_level = 3
                        realized_r += remaining_vol * current_target_r
                        remaining_vol = 0.0
                        outcome = "WIN"
                        pips = round(sl_dist * realized_r * pip_mult, 1)
                        exit_idx = i + 1 + f_step
                        break

                if ladder_level >= 3:
                    break

        if outcome == "OPEN" and ladder_level > 0:
            outcome = "WIN"
            last_close = float(future.iloc[-1]["close"])
            if direction == "BUY":
                floating_r = max(0.0, (last_close - entry_price) / sl_dist)
            else:
                floating_r = max(0.0, (entry_price - last_close) / sl_dist)
            realized_r += remaining_vol * floating_r
            pips = round(sl_dist * realized_r * pip_mult, 1)

        if outcome in ("WIN", "LOSS"):
            last_exit_idx = exit_idx + (3 if timeframe in ("M1", "M3") else 2)
            trades.append({
                "time": str(c_time),
                "direction": direction,
                "entry": entry_price,
                "entry_type": entry_type,
                "sl": sl,
                "tp": tp,
                "ladder_milestone": f"M{ladder_level}" if ladder_level > 0 else "None",
                "outcome": outcome,
                "pips": pips,
                "session": session,
                "confirmations": confs,
                "win_probability": win_prob,
            })
            if outcome == "WIN":
                for c in confs:
                    confs_win_counter[c] = confs_win_counter.get(c, 0) + 1

    total_signals = len(trades)
    if total_signals == 0:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "days": days,
            "total_signals": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_pips": 0.0,
            "min_win_prob_filter": min_win_prob,
            "timeframe_hierarchy": {
                "macro_bias_tf": bias_tf,
                "intermediate_trend_tf": trend_tf,
                "style": tf_cfg.get("style", "Intraday"),
            },
            "message": f"No setup met the strict >={min_win_prob}% win probability and multi-timeframe alignment threshold.",
        }

    wins = [t for t in trades if t["outcome"] == "WIN"]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = round((win_count / total_signals) * 100, 1)
    net_pips = round(sum(t["pips"] for t in trades), 1)
    avg_win_prob = round(sum(t["win_probability"] for t in trades) / total_signals, 1)

    # Session stats
    sessions = {}
    for s_name in ["London", "New York", "Asian", "NY Close"]:
        s_trades = [t for t in trades if t["session"] == s_name]
        if s_trades:
            s_wins = len([t for t in s_trades if t["outcome"] == "WIN"])
            s_rate = round(s_wins / len(s_trades) * 100, 1)
            sessions[s_name] = {"total": len(s_trades), "wins": s_wins, "win_rate": s_rate}

    # Best & worst trade
    sorted_trades = sorted(trades, key=lambda x: x["pips"], reverse=True)
    best_trade = sorted_trades[0] if sorted_trades else None
    worst_trade = sorted_trades[-1] if sorted_trades else None

    # Top confirmation
    top_confs = sorted(confs_win_counter.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "days": days,
        "start_date": str(df_window["time"].iloc[0].date()),
        "end_date": str(df_window["time"].iloc[-1].date()),
        "total_signals": total_signals,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": win_rate,
        "net_pips": net_pips,
        "min_win_prob_filter": min_win_prob,
        "avg_win_probability": avg_win_prob,
        "timeframe_hierarchy": {
            "macro_bias_tf": bias_tf,
            "intermediate_trend_tf": trend_tf,
            "style": tf_cfg.get("style", "Intraday"),
        },
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "sessions": sessions,
        "top_confirmations": top_confs,
        "recent_trades": trades[-5:],
    }
