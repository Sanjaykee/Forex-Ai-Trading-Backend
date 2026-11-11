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

from app.mt5.source_router import get_candles_fetcher
from app.signals.signal_logger import load_signals

CANDLE_COUNT = {"H1": 1500, "M5": 4500, "M3": 7500}
_fetcher     = get_candles_fetcher()
_candle_cache: dict = {}

def load_candles(symbol: str, tf: str) -> pd.DataFrame:
    """Fetch candles from MT5 — same source as the live scanner."""
    key = f"{symbol}_{tf}"
    if key in _candle_cache:
        return _candle_cache[key]
    try:
        df = _fetcher(symbol, tf, count=CANDLE_COUNT.get(tf, 500))
        if df.empty:
            _candle_cache[key] = pd.DataFrame()
            return pd.DataFrame()
        if df["time"].dt.tz is not None:
            df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
        df = df.sort_values("time").reset_index(drop=True)
        needed = ["time", "open", "high", "low", "close", "volume"]
        df = df[needed].copy()
    except Exception as e:
        print(f"  [WARN] MT5 fetch failed for {symbol} {tf}: {e}")
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

    lot       = calculate_lot_size(RISK_USD, entry, sl, SYM_INFO, symbol)
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
    all_trades   = []
    pair_results = []
    trade_idx    = 0

    # Load signal log for the date range (use START_DATE for single-day)
    signal_log = load_signals(START_DATE)
    using_log  = bool(signal_log)
    source_label = "signal_log" if using_log else "mt5"

    print("=" * 65)
    print("  VARIANT C BACKTEST  %s to %s" % (START_DATE, END_DATE))
    print("  Strategy : MTF SMC A+  |  RR 1:%.1f  |  Score >= %d/14" % (MIN_RR, MIN_SCORE))
    print("  Source   : %s  |  Risk/Trade: $%.2f  |  Lot step: 0.01" % (source_label, RISK_USD))
    if using_log:
        print("  Signal log found — using exact live entry prices for %s" % START_DATE)
    print("=" * 65)

    for symbol in PAIRS:
        print("  Fetching %s..." % symbol, end=" ", flush=True)
        df15 = load_candles(symbol, "H1")
        df5  = load_candles(symbol, "M5")
        df3  = load_candles(symbol, "M3")
        print("H1=%d M5=%d M3=%d" % (len(df15), len(df5), len(df3)))

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
                # (stored session label may be wrong if M3 entry was Off-Session)
                from app.analysis.smc.mtf_strategy import validate_session as _vs
                sess_valid, session = _vs(sig_utc)
                if not sess_valid:
                    continue

                subsequent = df3[df3["time"] > sig_utc].head(400)
                if subsequent.empty:
                    subsequent = df5[df5["time"] > sig_utc].head(150)

                outcome   = "OPEN"
                exit_time = None
                for _, row in subsequent.iterrows():
                    h, l, t = float(row["high"]), float(row["low"]), row["time"]
                    if direction == "SELL":
                        if h >= sl_price:  outcome = "LOSS"; exit_time = t; break
                        if l <= tp_price:  outcome = "WIN";  exit_time = t; break
                    else:
                        if l <= sl_price:  outcome = "LOSS"; exit_time = t; break
                        if h >= tp_price:  outcome = "WIN";  exit_time = t; break

                if outcome == "OPEN":
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
                apply_daily_protection=False,
            )
            trades = result["all_trades"]
            total  = result["total_signals"]
            wins   = result["wins"]
            losses = result["losses"]
            wr     = result["win_rate"]
            spread_adj_net_r = result.get("spread_adj_net_r", 0.0)

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
    print("  Total Risked    : $%.2f  (across all trades)" % total_risked)
    print("  Total Won       : +$%.2f  (from %d winning trades)" % (total_target, total_wins))
    print("  Total Lost      : -$%.2f  (from %d losing trades)"  % (total_lost,   total_losses))
    print("  Net P&L         : %s$%.2f" % ("+" if total_pnl >= 0 else "-", abs(total_pnl)))
    print("  Spread-adjusted : %s$%.2f  (after entry spread cost)" % ("+" if spread_adj_pnl >= 0 else "-", abs(spread_adj_pnl)))
    print("=" * 65)

    active = [p for p in pair_results if p["signals"] > 0]
    if active:
        print("\n  Per-Pair Summary:")
        print("  %-10s %7s %4s %4s %6s %10s" % (
            "Pair", "Signals", "W", "L", "WR%", "Net P&L"))
        print("  " + "-" * 50)
        for p in active:
            sign = "+" if p["net_pnl"] >= 0 else ""
            print("  %-10s %7d %4d %4d %5.1f%%  %s$%.2f" % (
                p["symbol"], p["signals"], p["wins"], p["losses"],
                p["win_rate"], sign, p["net_pnl"]
            ))


if __name__ == "__main__":
    run()
