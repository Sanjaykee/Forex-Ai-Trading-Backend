import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from app.risk.position_sizer import calculate_lot_size, _get_instrument_params

# MT5 minimum stop distance in pips per pair (standard broker rules)
MT5_MIN_SL_PIPS = {
    "EURUSD": 3,  "GBPUSD": 3,  "AUDUSD": 3,  "NZDUSD": 3,
    "USDCHF": 3,  "USDCAD": 3,  "USDJPY": 3,
    "EURJPY": 3,  "GBPJPY": 3,
}

# Actual trades from the 2-day backtest
trades = [
    ("EURUSD", "BUY",  1.11861, 1.11765, 1.12159),
    ("EURUSD", "BUY",  1.11941, 1.11835, 1.12261),
    ("EURUSD", "BUY",  1.11806, 1.11748, 1.11981),
    ("GBPUSD", "BUY",  1.34029, 1.33936, 1.34334),
    ("GBPUSD", "BUY",  1.34156, 1.34083, 1.34594),
    ("GBPUSD", "BUY",  1.34127, 1.34031, 1.34478),
    ("GBPUSD", "BUY",  1.34127, 1.34044, 1.34478),
    ("USDJPY", "BUY",  157.862, 157.77,  158.153),
    ("USDCHF", "SELL", 0.82482, 0.82549, 0.82280),
    ("USDCHF", "SELL", 0.82283, 0.82354, 0.82061),
    ("USDCHF", "SELL", 0.82272, 0.82320, 0.82061),
    ("USDCHF", "SELL", 0.81951, 0.82015, 0.81758),
    ("AUDUSD", "SELL", 0.70264, 0.70297, 0.70150),
    ("NZDUSD", "BUY",  0.56236, 0.56180, 0.56405),
    ("NZDUSD", "BUY",  0.56227, 0.56145, 0.56510),
    ("USDCAD", "BUY",  1.39623, 1.39582, 1.39760),
    ("EURJPY", "BUY",  176.005, 175.814, 176.580),
    ("EURJPY", "SELL", 176.692, 176.728, 176.537),
    ("EURJPY", "SELL", 176.590, 176.713, 176.209),
    ("EURJPY", "BUY",  176.689, 176.565, 177.078),
    ("GBPJPY", "BUY",  211.152, 210.956, 211.811),
    ("GBPJPY", "SELL", 211.636, 211.732, 211.281),
]

sym_info = {"volume_min": 0.01, "volume_step": 0.01}
RISK_USD = 1.0

print("=" * 80)
print("  RISK AUDIT — risk_usd=$%.2f  target range $0.75-$1.50" % RISK_USD)
print("=" * 80)
print("%-10s %-5s %8s %6s %12s %10s %10s" % (
    "Pair", "Dir", "SL_pips", "Lot", "ActualRisk$", "MT5_MinSL", "SL_OK"))
print("-" * 80)

issues = []
for symbol, direction, entry, sl, tp in trades:
    pip_size, pip_val = _get_instrument_params(symbol, entry)
    sl_dist  = abs(entry - sl)
    sl_pips  = sl_dist / pip_size
    sl_dollar_per_lot = sl_pips * pip_val
    lot      = calculate_lot_size(RISK_USD, entry, sl, sym_info, symbol)
    actual_risk = lot * sl_dollar_per_lot

    min_sl   = MT5_MIN_SL_PIPS.get(symbol, 3)
    sl_ok    = "OK" if sl_pips >= min_sl else "FAIL-TOO_TIGHT"
    risk_ok  = "OK" if 0.75 <= actual_risk <= 1.50 else "FAIL-OUT_RANGE"

    flag = ""
    if sl_ok != "OK":
        flag += " [SL_TIGHT]"
        issues.append((symbol, direction, "SL too tight: %.1f pips < %d pip min" % (sl_pips, min_sl)))
    if risk_ok != "OK":
        flag += " [RISK_WRONG]"
        issues.append((symbol, direction, "Actual risk $%.4f outside $0.75-$1.50" % actual_risk))

    print("%-10s %-5s %8.1f %6.2f %12.4f %10s %10s%s" % (
        symbol, direction, sl_pips, lot, actual_risk, str(min_sl)+" pips", sl_ok, flag))

print("=" * 80)
print("\n  ISSUES FOUND: %d" % len(issues))
if issues:
    for sym, d, msg in issues:
        print("  [%s %s] %s" % (sym, d, msg))
else:
    print("  None — all trades within acceptable range.")

# Show rounding loss analysis
print("\n" + "=" * 80)
print("  LOT ROUNDING IMPACT (volume_step=0.01)")
print("=" * 80)
print("%-10s %-5s %10s %10s %10s %10s" % (
    "Pair", "Dir", "RawLot", "RoundedLot", "IdealRisk$", "ActualRisk$"))
print("-" * 80)
seen = set()
for symbol, direction, entry, sl, tp in trades:
    key = (symbol, direction, round(entry,5), round(sl,5))
    if key in seen:
        continue
    seen.add(key)
    pip_size, pip_val = _get_instrument_params(symbol, entry)
    sl_dist  = abs(entry - sl)
    sl_pips  = sl_dist / pip_size
    sl_dollar_per_lot = sl_pips * pip_val
    if sl_dollar_per_lot <= 0:
        continue
    raw_lot  = RISK_USD / sl_dollar_per_lot
    step     = 0.01
    rounded  = max(0.01, round(round(raw_lot / step) * step, 2))
    ideal    = raw_lot * sl_dollar_per_lot
    actual   = rounded * sl_dollar_per_lot
    diff_pct = abs(actual - ideal) / ideal * 100 if ideal > 0 else 0
    flag     = "  <-- HIGH ROUNDING ERROR" if diff_pct > 20 else ""
    print("%-10s %-5s %10.4f %10.2f %10.4f %10.4f%s" % (
        symbol, direction, raw_lot, rounded, ideal, actual, flag))
