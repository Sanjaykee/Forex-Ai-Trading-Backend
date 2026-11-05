"""
Today's Backtest — fetches live candles from MetaApi and runs the SMC engine.
Usage: python run_today_backtest.py [SYMBOL] [TF] [RR]
       Defaults: all pairs, M15, 1:3
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from datetime import date, timedelta
import pandas as pd
from app.backtest.engine import run_backtest
from app.risk.position_sizer import (
    calculate_lot_size, actual_risk_usd, _get_instrument_params
)

def to_ist(utc_str: str) -> str:
    try:
        t = pd.to_datetime(utc_str)
        return (t + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M IST")
    except Exception:
        return utc_str

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
         "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"]

# Spread cost per pair in pips (one-way entry cost) — same as Strategy 2
SPREAD_PIPS = {
    "EURUSD": 1.2, "GBPUSD": 1.5, "USDJPY": 1.3, "USDCHF": 1.6,
    "AUDUSD": 1.4, "NZDUSD": 2.0, "USDCAD": 1.8,
    "EURJPY": 1.8, "GBPJPY": 2.5,
}

TF      = sys.argv[2].upper() if len(sys.argv) > 2 else "M15"
RR      = float(sys.argv[3])  if len(sys.argv) > 3 else 3.0
DAYS    = 1
TODAY   = str(date.today())
RISK_USD = 1.0
SYM_INFO = {"volume_min": 0.01, "volume_step": 0.01, "volume_max": 100.0}

pairs_to_run = [sys.argv[1].upper()] if len(sys.argv) > 1 else PAIRS


def trade_financials(symbol, entry, sl, tp, outcome):
    pip_size, pip_val = _get_instrument_params(symbol, entry)
    sl_pips  = round(abs(entry - sl) / pip_size, 1)
    tp_pips  = round(abs(tp - entry) / pip_size, 1)
    lot      = calculate_lot_size(RISK_USD, entry, sl, SYM_INFO, symbol)
    risk_amt = actual_risk_usd(lot, entry, sl, symbol)
    reward_amt = round(lot * tp_pips * pip_val, 2)
    pnl      = reward_amt if outcome == "WIN" else -risk_amt

    spread_p = SPREAD_PIPS.get(symbol.upper(), 1.5)
    spread_cost = round(lot * (spread_p / (100 if "JPY" in symbol else 10000)) * (
        pip_val / pip_size * pip_size), 4)
    # simpler: spread cost in $ = spread_pips * pip_val * lot
    spread_cost = round(spread_p * pip_val * lot, 4)
    spread_adj_pnl = round(pnl - spread_cost, 4)

    return {
        "lot": lot, "sl_pips": sl_pips, "tp_pips": tp_pips,
        "risk_amt": risk_amt, "reward_amt": reward_amt,
        "pnl": pnl, "spread_pips": spread_p, "spread_cost": spread_cost,
        "spread_adj_pnl": spread_adj_pnl,
    }


print("=" * 65)
print(f"  TODAY'S BACKTEST  {TODAY}")
print(f"  Timeframe: {TF}  |  RR 1:{RR:.1f}  |  Source: MetaApi")
print(f"  Risk/Trade: ${RISK_USD:.2f}  |  Lot step: 0.01")
print("=" * 65)

all_trades   = []
pair_results = []
trade_idx    = 0

for symbol in pairs_to_run:
    r = run_backtest(symbol=symbol, timeframe=TF, days=DAYS, rr_ratio=RR)

    if "error" in r:
        print(f"\n[{symbol}]  ERROR: {r['error']}")
        continue

    trades = r.get("all_trades", [])
    total  = r["total_signals"]
    wins   = r["wins"]
    losses = r["losses"]
    wr     = r["win_rate"]

    if total == 0:
        print(f"\n[{symbol}]  {r.get('message', 'No signals this period.')}")
        continue

    # compute financials
    fin_list  = []
    pair_pnl  = 0.0
    pair_sadj = 0.0
    for t in trades:
        fin = trade_financials(symbol, float(t["entry"]), float(t["sl"]),
                               float(t["tp"]), t["outcome"])
        fin_list.append(fin)
        pair_pnl  += fin["pnl"]
        pair_sadj += fin["spread_adj_pnl"]

    pair_results.append({
        "symbol": symbol, "total": total, "wins": wins, "losses": losses,
        "win_rate": wr, "net_pnl": round(pair_pnl, 2),
        "spread_adj": round(pair_sadj, 2),
    })
    all_trades.extend(zip(trades, fin_list))

    sign = "+" if pair_pnl >= 0 else "-"
    print(f"\n{'-'*65}")
    print(f"  [{symbol}]  Signals={total}  W={wins}  L={losses}  "
          f"WR={wr:.1f}%  Net P&L={sign}${abs(pair_pnl):.2f}")
    print(f"{'-'*65}")

    for t, fin in zip(trades, fin_list):
        trade_idx += 1
        tag = "[WIN] " if t["outcome"] == "WIN" else "[LOSS]"
        pnl_str = ("+$%.2f" % fin["pnl"]) if fin["pnl"] >= 0 else ("-$%.2f" % abs(fin["pnl"]))
        print(f"\n  Trade #{trade_idx}  {tag}  {symbol}  {t['direction']}")
        print(f"  {'Pattern:':<18} {t['pattern']}  (score {t['score']}  {t['confidence']})")
        print(f"  {'Entry Price:':<18} {t['entry']}")
        print(f"  {'Stop Loss:':<18} {t['sl']}  ({fin['sl_pips']:.1f} pips)")
        print(f"  {'Take Profit:':<18} {t['tp']}  ({fin['tp_pips']:.1f} pips)")
        print(f"  {'Lot Size:':<18} {fin['lot']:.2f} lots")
        print(f"  {'Risk Amount:':<18} ${fin['risk_amt']:.4f}  (if SL hit)")
        print(f"  {'Target Reward:':<18} ${fin['reward_amt']:.4f}  (if TP hit)")
        print(f"  {'Outcome:':<18} {pnl_str}  (RR 1:{RR:.2f} | Score {t['score']})")
        print(f"  {'Session:':<18} {t['session']}  KZ={t['kz']}")
        print(f"  {'HTF Bias:':<18} {t['htf_bias']}")
        print(f"  {'Time (IST):':<18} {to_ist(t['time'])}  ->  {to_ist(t['exit_time'])}")

# ── Summary ───────────────────────────────────────────────────────────────────
total_t   = len(all_trades)
total_w   = sum(1 for t, _ in all_trades if t["outcome"] == "WIN")
total_l   = sum(1 for t, _ in all_trades if t["outcome"] == "LOSS")
wr_all    = round(total_w / total_t * 100, 1) if total_t else 0.0
total_won = round(sum(fin["reward_amt"] for t, fin in all_trades if t["outcome"] == "WIN"), 2)
total_lost= round(sum(fin["risk_amt"]   for t, fin in all_trades if t["outcome"] == "LOSS"), 2)
net_pnl   = round(sum(fin["pnl"]        for _, fin in all_trades), 2)
spread_adj= round(sum(fin["spread_adj_pnl"] for _, fin in all_trades), 2)
total_risked = round(sum(fin["risk_amt"] for _, fin in all_trades), 2)

print(f"\n{'='*65}")
print(f"  FINAL SUMMARY  {TODAY}  ({TF}  RR 1:{RR:.1f})")
print(f"{'='*65}")
print(f"  Total Trades    : {total_t}")
print(f"  Wins            : {total_w}")
print(f"  Losses          : {total_l}")
print(f"  Win Rate        : {wr_all:.1f}%")
print(f"  Total Risked    : ${total_risked:.2f}  (across all trades)")
print(f"  Total Won       : +${total_won:.2f}  (from {total_w} winning trades)")
print(f"  Total Lost      : -${total_lost:.2f}  (from {total_l} losing trades)")
print(f"  Net P&L         : {'+'if net_pnl>=0 else ''}${net_pnl:.2f}")
print(f"  Spread-adjusted : {'+'if spread_adj>=0 else ''}${spread_adj:.2f}  (after entry spread cost)")
print(f"{'='*65}")

active = [p for p in pair_results if p["total"] > 0]
if active:
    print(f"\n  Per-Pair Summary:")
    print(f"  {'Pair':<10} {'Signals':>7} {'W':>4} {'L':>4} {'WR%':>6} {'Net P&L':>10}")
    print("  " + "-" * 50)
    for p in active:
        sign = "+" if p["net_pnl"] >= 0 else ""
        print(f"  {p['symbol']:<10} {p['total']:>7} {p['wins']:>4} {p['losses']:>4} "
              f"{p['win_rate']:>5.1f}%  {sign}${p['net_pnl']:.2f}")
