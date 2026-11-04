"""
Variant C Backtest - 2026-11-02 to 2026-11-03
Full live-scenario display: SL pips, risk $, target $, lot size per trade.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from app.analysis.smc.mtf_strategy import run_mtf_smc_backtest
from app.risk.position_sizer import (
    calculate_lot_size, actual_risk_usd, _get_instrument_params
)

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
         "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"]

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data", "historical_backup")
START_DATE = "2026-11-02"
END_DATE   = "2026-11-03"
MIN_SCORE  = 11
MIN_RR     = 3.0
RISK_USD   = 1.0          # $1 risk per trade
SYM_INFO   = {"volume_min": 0.01, "volume_step": 0.01, "volume_max": 100.0}


def load_csv(symbol, tf):
    path = os.path.join(DATA_DIR, f"{symbol}_{tf}_2026-09-01_2026-11-04.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["time"])
    return df.sort_values("time").reset_index(drop=True)


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

    print("=" * 65)
    print("  VARIANT C BACKTEST  %s to %s" % (START_DATE, END_DATE))
    print("  Strategy : MTF SMC A+  |  RR 1:%.1f  |  Score >= %d/14" % (MIN_RR, MIN_SCORE))
    print("  Risk/Trade: $%.2f  |  Lot step: 0.01" % RISK_USD)
    print("=" * 65)

    for symbol in PAIRS:
        df15 = load_csv(symbol, "H1")
        df5  = load_csv(symbol, "M5")
        df3  = load_csv(symbol, "M3")

        if df15.empty or df5.empty or df3.empty:
            print("\n[%s] SKIPPED - missing data" % symbol)
            continue

        result = run_mtf_smc_backtest(
            df15=df15, df5=df5, df3=df3,
            symbol=symbol,
            start_date=START_DATE,
            end_date=END_DATE,
            min_score=MIN_SCORE,
            min_rr=MIN_RR,
            apply_daily_protection=True,
            max_daily_trades=3,
            max_daily_loss_r=2.0,
            max_consec_losses=2,
        )

        trades = result["all_trades"]
        total  = result["total_signals"]
        wins   = result["wins"]
        losses = result["losses"]
        wr     = result["win_rate"]

        # compute financials per trade
        pair_pnl = 0.0
        pair_risk = 0.0
        pair_reward = 0.0
        fin_list = []
        for t in trades:
            fin = trade_financials(t)
            fin_list.append(fin)
            pair_pnl    += fin["pnl"]
            pair_risk   += fin["risk_amt"]
            pair_reward += fin["reward_amt"]

        pair_results.append({
            "symbol": symbol, "signals": total,
            "wins": wins, "losses": losses,
            "win_rate": wr, "net_pnl": round(pair_pnl, 2),
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
