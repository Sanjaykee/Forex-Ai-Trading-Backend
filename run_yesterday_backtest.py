"""
Variant C Backtest — Live Data (dynamic source)
Full live-scenario display: SL pips, risk $, target $, lot size per trade.
Usage: python run_yesterday_backtest.py [START_DATE] [END_DATE]
       Dates in YYYY-MM-DD format. Defaults: today
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import date, datetime, timezone
import pandas as pd
from app.analysis.smc.mtf_strategy import run_mtf_smc_backtest
from app.risk.position_sizer import (
    calculate_lot_size, actual_risk_usd, _get_instrument_params
)
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
         "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"]

TODAY      = str(date.today())
START_DATE = sys.argv[1] if len(sys.argv) > 1 else TODAY
# If only one date arg given, END_DATE = same day (single-day backtest)
END_DATE   = sys.argv[2] if len(sys.argv) > 2 else START_DATE
MIN_SCORE  = 11
MIN_RR     = 3.0
RISK_USD   = 1.0
SYM_INFO   = {"volume_min": 0.01, "volume_step": 0.01, "volume_max": 100.0}

# ── Default risk limits (used when no user settings found in DB) ──────────────
# These match the Settings page defaults shown in the UI.
DEFAULT_MAX_TRADES_DAY  = 10
DEFAULT_MAX_LOSSES_DAY  = 5
DEFAULT_MAX_CONSEC_LOSS = 5   # not configurable via UI


def _load_user_risk_settings(target_email: str = "sanjaykee79@gmail.com") -> tuple:
    """
    Read max_trades_day, max_losses_day, max_risk_usd, rr_ratio, and data_source from the specified user's DB settings.
    Defaults to checking sanjaykee79@gmail.com.
    Falls back to DEFAULT_* values if DB is unavailable or settings not saved.
    Returns (max_trades_day, max_losses_day, max_consec_loss, risk_usd, rr_ratio, user_email, data_source).
    """
    try:
        from app.db.database import SessionLocal
        from app.db.models import User, Settings
        from app.db import crud
        db = SessionLocal()

        # 1. First look up directly by user email
        user_obj = db.query(User).filter(User.email.ilike(target_email)).first()
        
        # 2. Fallback to user who has configured settings
        if not user_obj:
            setting_row = db.query(Settings.user_id).filter(Settings.key == "max_trades_day").order_by(Settings.id.desc()).first()
            if setting_row:
                user_obj = db.query(User).filter(User.id == setting_row[0]).first()
            else:
                user_obj = db.query(User).order_by(User.id.desc()).first()

        user_email = user_obj.email if user_obj else "default"
        if user_obj:
            target_uid = user_obj.id
            raw_trades = crud.get_setting(db, target_uid, "max_trades_day")
            raw_losses = crud.get_setting(db, target_uid, "max_losses_day")
            raw_risk   = crud.get_setting(db, target_uid, "max_risk_usd")
            raw_rr     = crud.get_setting(db, target_uid, "rr_ratio")
            raw_ds     = crud.get_setting(db, target_uid, "data_source")

            max_trades = int(raw_trades) if raw_trades else DEFAULT_MAX_TRADES_DAY
            max_losses = int(raw_losses) if raw_losses else DEFAULT_MAX_LOSSES_DAY
            risk_usd   = float(raw_risk) if raw_risk else 1.0
            rr_ratio   = float(raw_rr) if raw_rr else 3.0
            data_source = raw_ds.strip().lower() if raw_ds else "auto"
        else:
            max_trades, max_losses = DEFAULT_MAX_TRADES_DAY, DEFAULT_MAX_LOSSES_DAY
            risk_usd, rr_ratio, data_source = 1.0, 3.0, "auto"

        db.close()
        return max_trades, max_losses, DEFAULT_MAX_CONSEC_LOSS, risk_usd, rr_ratio, user_email, data_source
    except Exception as e:
        return DEFAULT_MAX_TRADES_DAY, DEFAULT_MAX_LOSSES_DAY, DEFAULT_MAX_CONSEC_LOSS, 1.0, 3.0, f"error: {e}", "auto"


from app.mt5.source_router import get_candles_range_fetcher, _mt5_available
from app.signals.signal_logger import load_signals
from datetime import timedelta

try:
    import MetaTrader5 as mt5
    mt5.initialize()
except Exception:
    pass

# Context lookback needed before start_date so the strategy has enough history
_CONTEXT = {"H1": timedelta(days=120), "M15": timedelta(days=60), "M5": timedelta(days=30), "M3": timedelta(days=20)}

_range_fetcher = get_candles_range_fetcher("auto")
_candle_cache: dict = {}


def load_candles(symbol: str, tf: str) -> pd.DataFrame:
    """Always fetch by date range: (START_DATE - context) → (END_DATE + 1 day)."""
    key = f"{symbol}_{tf}"
    if key in _candle_cache:
        return _candle_cache[key]
    try:
        from_dt = datetime.strptime(START_DATE, "%Y-%m-%d") - _CONTEXT.get(tf, timedelta(days=7))
        to_dt   = datetime.strptime(END_DATE,   "%Y-%m-%d") + timedelta(days=1)
        df = _range_fetcher(symbol, tf, from_dt, to_dt)
        if df.empty:
            _candle_cache[key] = pd.DataFrame()
            return pd.DataFrame()
        if df["time"].dt.tz is not None:
            df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
        df = df.sort_values("time").reset_index(drop=True)
        df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    except Exception as e:
        print(f"  [WARN] Data fetch failed for {symbol} {tf}: {e}")
        df = pd.DataFrame()
    _candle_cache[key] = df
    return df


def trade_financials(t):
    """
    Compute lot, actual risk $, target reward $, and actual P&L $
    for a single trade dict from the backtest engine.
    """
    symbol    = t["pair"]
    entry     = float(t["entry"])
    sl        = float(t["stop_loss"])
    tp        = float(t["take_profit"])
    direction = t["direction"]

    pip_size, pip_val = _get_instrument_params(symbol, entry)

    sl_dist   = abs(entry - sl)
    tp_dist   = abs(tp - entry)
    sl_pips   = round(sl_dist / pip_size, 1)
    tp_pips   = round(tp_dist / pip_size, 1)

    if t.get("lot") and float(t["lot"]) > 0:
        lot = float(t["lot"])
    else:
        lot = calculate_lot_size(RISK_USD, entry, sl, SYM_INFO, symbol)

    risk_amt  = actual_risk_usd(lot, entry, sl, symbol)
    reward_amt = round(lot * (tp_pips * pip_val), 2)

    if t["result"] == "WIN":
        pnl = reward_amt
    else:
        pnl = -risk_amt

    return {
        "lot":        lot,
        "sl_pips":    sl_pips,
        "tp_pips":    tp_pips,
        "risk_amt":   risk_amt,
        "reward_amt": reward_amt,
        "pnl":        pnl,
    }


def print_trade(idx, t, fin):
    result  = t["result"]
    tag     = "[WIN] " if result == "WIN" else "[LOSS]"
    pnl_str = "+$%.2f" % fin["pnl"] if fin["pnl"] >= 0 else "-$%.2f" % abs(fin["pnl"])

    print("")
    print("  Trade #%d  %s  %s %s" % (idx, tag, t["pair"], t["direction"]))
    print("  %-18s %s" % ("Entry Price:",    t["entry"]))
    print("  %-18s %s  (%.1f pips)" % ("Stop Loss:",  t["stop_loss"],  fin["sl_pips"]))
    print("  %-18s %s  (%.1f pips)" % ("Take Profit:", t["take_profit"], fin["tp_pips"]))
    print("  %-18s %.2f lots" % ("Lot Size:",   fin["lot"]))
    print("  %-18s $%.4f  (if SL hit)" % ("Risk Amount:",   fin["risk_amt"]))
    print("  %-18s $%.4f  (if TP hit)" % ("Target Reward:", fin["reward_amt"]))
    print("  %-18s %s  (RR %s | Score %d/14)" % ("Outcome:", pnl_str, t["rr"], t["score"]))
    print("  %-18s %s" % ("Session:",    t["session"]))
    print("  %-18s %s  ->  %s" % ("Time (IST):", t["timestamp_ist"], t["exit_time_ist"]))


def run():
    global RISK_USD, MIN_RR, _range_fetcher, _CONTEXT
    all_trades   = []
    all_missed   = []
    pair_results = []
    trade_idx    = 0

    using_log    = False

    max_trades_day, max_losses_day, max_consec_loss, db_risk_usd, db_rr, user_email, user_ds = _load_user_risk_settings()
    RISK_USD = db_risk_usd
    MIN_RR   = db_rr

    # Dynamically bind range fetcher and context for active data source
    global _range_fetcher, _candle_cache
    _range_fetcher = get_candles_range_fetcher(user_ds)
    _candle_cache.clear()

    effective_ds = user_ds
    if effective_ds == "auto":
        if _mt5_available():
            effective_ds = "mt5"
        else:
            effective_ds = "ctrader"

    if effective_ds == "ctrader":
        source_label = "cTrader Cloud (WebSocket JSON)"
    elif effective_ds == "metaapi":
        source_label = "MetaApi Cloud (REST)"
    elif effective_ds == "mt5":
        source_label = "Local MT5 Terminal"
    else:
        source_label = f"Auto ({effective_ds})"

    print("=" * 65)
    print("  VARIANT C BACKTEST  %s to %s" % (START_DATE, END_DATE))
    print("  Strategy : MTF SMC A+  |  RR 1:%.1f  |  Score >= %d/14" % (MIN_RR, MIN_SCORE))
    print("  Source   : %s  |  Risk/Trade: $%.2f  |  Lot step: 0.01" % (source_label, RISK_USD))
    print("  User Profile : %s (Settings Loaded from Database)" % user_email)
    if using_log:
        print("  Signal log found — using exact live entry prices for %s" % START_DATE)
    print("=" * 65)
    print("  Risk Limits : Max %d trades/day | Max %dR loss/day | Max %d consec losses" % (
        max_trades_day, max_losses_day, max_consec_loss))
    print("=" * 65)

    for symbol in PAIRS:
        print("  Fetching %s..." % symbol, end=" ", flush=True)
        df15   = load_candles(symbol, "H1")
        df_m15 = load_candles(symbol, "M15")
        df5    = load_candles(symbol, "M5")
        df3    = load_candles(symbol, "M3")
        print("H1=%d M15=%d M5=%d M3=%d" % (len(df15), len(df_m15), len(df5), len(df3)), flush=True)

        if df15.empty or df5.empty or df3.empty:
            print("\n[%s] SKIPPED - no data from active source" % symbol)
            continue

        # ── If signal log exists for this date, replay from logged prices ──
        if using_log and symbol in signal_log:
            log_records = signal_log[symbol]
            trades = []
            seen_log = set()   # dedup existing log records by (entry, sl, tp, direction)
            for rec in log_records:
                entry     = float(rec["entry"])
                sl_price  = float(rec["sl"])
                tp_price  = float(rec["tp"])
                direction = rec["direction"]

                sig_key = (entry, sl_price, tp_price, direction)
                if sig_key in seen_log:
                    continue
                seen_log.add(sig_key)

                time_ist  = rec.get("time_ist", "?")
                session   = rec.get("session", "?")
                score     = rec.get("score", 0)
                pip_size  = 0.01 if "JPY" in symbol else 0.0001
                pip_mult  = 100.0 if "JPY" in symbol else 10000.0

                # Forward simulate from M3 candles after signal time
                # Parse signal time to find candles after it
                try:
                    sig_date = pd.Timestamp(START_DATE)
                    sig_time_str = time_ist.replace(" IST", "")
                    sig_ist = pd.Timestamp("%s %s" % (START_DATE, sig_time_str))
                    sig_utc = sig_ist - pd.Timedelta(hours=5, minutes=30)
                except Exception:
                    sig_utc = pd.Timestamp(START_DATE)

                # Re-validate session from actual UTC signal time
                # (stored session label may be wrong if M3 entry was in different session)
                from app.analysis.smc.mtf_strategy import validate_session as _vs
                _sess_valid, session = _vs(sig_utc)

                subsequent = df3[df3["time"] > sig_utc].head(400)
                if subsequent.empty:
                    subsequent = df5[df5["time"] > sig_utc].head(150)

                outcome   = "OPEN"
                exit_time = None
                is_filled = False
                for _, row in subsequent.iterrows():
                    h, l, t = float(row["high"]), float(row["low"]), row["time"]

                    # ── Limit Order Fill / Miss Check (matches live TradeMonitor) ──
                    if not is_filled:
                        if direction == "BUY":
                            if h >= tp_price and l > entry:
                                outcome = "MISSED"
                                break
                            if l <= entry:
                                is_filled = True
                            else:
                                continue
                        else:  # SELL
                            if l <= tp_price and h < entry:
                                outcome = "MISSED"
                                break
                            if h >= entry:
                                is_filled = True
                            else:
                                continue

                    if direction == "SELL":
                        if h >= sl_price:  outcome = "LOSS"; exit_time = t; break
                        if l <= tp_price:  outcome = "WIN";  exit_time = t; break
                    else:
                        if l <= sl_price:  outcome = "LOSS"; exit_time = t; break
                        if h >= tp_price:  outcome = "WIN";  exit_time = t; break

                if outcome in ("OPEN", "MISSED") or not is_filled:
                    continue

                exit_ist = (exit_time + pd.Timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M IST") if exit_time is not None else "?"

                trades.append({
                    "timestamp":     str(sig_utc),
                    "timestamp_ist": "%s %s" % (START_DATE, time_ist),
                    "pair":          symbol,
                    "direction":     direction,
                    "entry":         entry,
                    "stop_loss":     sl_price,
                    "take_profit":   tp_price,
                    "lot":           float(rec.get("lot", 0.01)),
                    "rr":            "1:3.00",
                    "rr_val":        3.0,
                    "score":         score,
                    "session":       session,   # re-validated from sig_utc
                    "result":        outcome,
                    "exit_time_ist": exit_ist,
                    "risk_pips":     round(abs(entry - sl_price) / pip_size, 1),
                    "spread_pips":   1.5,
                })

            total = len(trades)
            wins   = sum(1 for t in trades if t["result"] == "WIN")
            losses = sum(1 for t in trades if t["result"] == "LOSS")
            wr     = round(wins / total * 100, 1) if total else 0.0
            spread_adj_net_r = 0.0

        else:
            # No signal log — use MT5 backtest engine
            result = run_mtf_smc_backtest(
                df15=df15, df5=df5, df3=df3,
                symbol=symbol,
                start_date=START_DATE,
                end_date=END_DATE,
                min_score=MIN_SCORE,
                min_rr=MIN_RR,
                allowed_sessions=["Asian", "London", "New York"],
                apply_daily_protection=True,
                max_daily_trades=max_trades_day,
                max_daily_loss_r=float(max_losses_day),
                max_consec_losses=max_consec_loss,
                df_m15=df_m15,
            )
            trades = result["all_trades"]
            total  = result["total_signals"]
            wins   = result["wins"]
            losses = result["losses"]
            wr     = result["win_rate"]
            spread_adj_net_r = result.get("spread_adj_net_r", 0.0)
            missed = result.get("rejections", {}).get("limit_order_missed", 0)
            pair_missed_list = result.get("missed_trades", [])
            all_missed.extend(pair_missed_list)

        # compute financials per trade
        pair_pnl = 0.0
        pair_risk = 0.0
        pair_reward = 0.0
        fin_list = []
        for t in trades:
            fin = trade_financials(t)
            if fin["lot"] <= 0:          # skip zero-lot trades
                continue
            fin_list.append(fin)
            pair_pnl    += fin["pnl"]
            pair_risk   += fin["risk_amt"]
            pair_reward += fin["reward_amt"]

        pair_results.append({
            "symbol": symbol, "signals": total,
            "wins": wins, "losses": losses,
            "win_rate": wr, "net_pnl": round(pair_pnl, 2),
            "spread_adj_net_r": spread_adj_net_r if not using_log else 0.0,
            "missed": missed if not using_log else 0,
        })
        all_trades.extend(zip(trades, fin_list))

        if total == 0:
            print("\n[%s]  No signals this period." % symbol)
            continue

        print("\n" + "-" * 65)
        print("  [%s]  Signals=%d  W=%d  L=%d  WR=%.1f%%  Net P&L=%s$%.2f" % (
            symbol, total, wins, losses, wr,
            "+" if pair_pnl >= 0 else "-", abs(pair_pnl)
        ))
        print("-" * 65)

        for t, fin in zip(trades, fin_list):
            trade_idx += 1
            print_trade(trade_idx, t, fin)

    # ── Overall Summary ───────────────────────────────────────────────────────
    total_signals = len(all_trades)
    total_wins    = sum(1 for t, _ in all_trades if t["result"] == "WIN")
    total_losses  = sum(1 for t, _ in all_trades if t["result"] == "LOSS")
    win_rate      = round(total_wins / total_signals * 100, 1) if total_signals else 0.0
    total_pnl     = round(sum(fin["pnl"]        for _, fin in all_trades), 2)
    total_risked  = round(sum(fin["risk_amt"]    for _, fin in all_trades), 2)
    total_target  = round(sum(fin["reward_amt"]  for t, fin in all_trades if t["result"] == "WIN"), 2)
    total_lost    = round(sum(fin["risk_amt"]     for t, fin in all_trades if t["result"] == "LOSS"), 2)
    spread_adj_net_r_total = round(sum(p.get("spread_adj_net_r", 0.0) for p in pair_results), 2)
    # Convert spread-adj net R to USD at $1 risk/trade
    spread_adj_pnl = round(spread_adj_net_r_total * RISK_USD, 2)

    print("\n" + "=" * 65)
    print("  FINAL SUMMARY  %s to %s" % (START_DATE, END_DATE))
    print("=" * 65)
    print("  Total Trades    : %d" % total_signals)
    print("  Wins            : %d" % total_wins)
    print("  Losses          : %d" % total_losses)
    print("  Win Rate        : %.1f%%" % win_rate)
    total_missed  = sum(p.get("missed", 0) for p in pair_results)
    if total_missed > 0:
        print("  Missed Orders   : %d  (limit orders cancelled — reached TP before entry fill)" % total_missed)
    print("  Total Risked    : $%.2f  (across all trades)" % total_risked)
    print("  Total Won       : +$%.2f  (from %d winning trades)" % (total_target, total_wins))
    print("  Total Lost      : -$%.2f  (from %d losing trades)"  % (total_lost,   total_losses))
    print("  Net P&L         : %s$%.2f" % ("+" if total_pnl >= 0 else "-", abs(total_pnl)))
    print("  Spread-adjusted : %s$%.2f  (after entry spread cost)" % ("+" if spread_adj_pnl >= 0 else "-", abs(spread_adj_pnl)))
    print("=" * 65)

    active = [p for p in pair_results if p["signals"] > 0 or p.get("missed", 0) > 0]
    if active:
        print("\n  Per-Pair Summary:")
        print("  %-10s %7s %4s %4s %6s %7s %10s" % (
            "Pair", "Signals", "W", "L", "WR%", "Missed", "Net P&L"))
        print("  " + "-" * 58)
        for p in active:
            sign = "+" if p["net_pnl"] >= 0 else ""
            print("  %-10s %7d %4d %4d %5.1f%% %7d  %s$%.2f" % (
                p["symbol"], p["signals"], p["wins"], p["losses"],
                p["win_rate"], p.get("missed", 0), sign, p["net_pnl"]
            ))

    # ── Per-Session Summary ───────────────────────────────────────────────────
    sess_map: dict = {}
    for t, fin in all_trades:
        s = t.get("session", "Unknown")
        if s not in sess_map:
            sess_map[s] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        sess_map[s]["trades"] += 1
        if t["result"] == "WIN":
            sess_map[s]["wins"] += 1
        else:
            sess_map[s]["losses"] += 1
        sess_map[s]["pnl"] += fin["pnl"]

    if sess_map:
        print("\n  Per-Session Summary:")
        print("  %-14s %7s %4s %4s %6s %10s" % (
            "Session", "Trades", "W", "L", "WR%", "Net P&L"))
        print("  " + "-" * 54)
        for sess in ["Asian", "London", "New York", "Off-Session"]:
            if sess not in sess_map:
                continue
            d = sess_map[sess]
            wr = round(d["wins"] / d["trades"] * 100, 1) if d["trades"] else 0.0
            sign = "+" if d["pnl"] >= 0 else ""
            print("  %-14s %7d %4d %4d %5.1f%%  %s$%.2f" % (
                sess, d["trades"], d["wins"], d["losses"], wr, sign, abs(d["pnl"])
            ))

    # ── Missed Orders Detailed Breakdown ──────────────────────────────────────
    if all_missed:
        print("\n" + "=" * 65)
        print("  MISSED ORDERS DETAILED BREAKDOWN (%d Total Missed)" % len(all_missed))
        print("  (Price retested close to entry, then reached 1:3 TP without fill)")
        print("=" * 65)

        # 1. Distance Distribution
        le_1 = sum(1 for m in all_missed if m["missed_pips"] <= 1.0)
        le_2 = sum(1 for m in all_missed if 1.0 < m["missed_pips"] <= 2.0)
        le_3 = sum(1 for m in all_missed if 2.0 < m["missed_pips"] <= 3.0)
        le_5 = sum(1 for m in all_missed if 3.0 < m["missed_pips"] <= 5.0)
        gt_5 = sum(1 for m in all_missed if m["missed_pips"] > 5.0)

        print("\n  Missed Distance Distribution:")
        print("    Within <= 1.0 pip  : %2d setups  (Nearly filled! Retest was <= 1 pip away)" % le_1)
        print("    Within 1.1 - 2.0 p : %2d setups" % le_2)
        print("    Within 2.1 - 3.0 p : %2d setups" % le_3)
        print("    Within 3.1 - 5.0 p : %2d setups" % le_5)
        print("    Farther > 5.0 p    : %2d setups" % gt_5)

        # 2. Individual Missed Setups List
        print("\n  List of All Missed Setups:")
        print("  %-3s %-7s %-4s %-16s %-9s %-12s %-10s %-9s" % (
            "#", "Pair", "Dir", "Time (IST)", "Limit", "Retested To", "Missed By", "TP Target"
        ))
        print("  " + "-" * 78)
        for idx, m in enumerate(all_missed, 1):
            print("  %-3d %-7s %-4s %-16s %-9s %-12s %-10s %-9s" % (
                idx, m["pair"], m["direction"], m["timestamp_ist"],
                m["entry"], m["closest_retest"],
                "%.1f pips" % m["missed_pips"], m["take_profit"]
            ))


if __name__ == "__main__":
    run()
