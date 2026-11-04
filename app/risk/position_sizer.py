import math

# ── MT5 minimum stop distance in pips per instrument ─────────────────────────
MT5_MIN_SL_PIPS = {
    "EURUSD": 3,  "GBPUSD": 3,  "AUDUSD": 3,  "NZDUSD": 3,
    "USDCHF": 3,  "USDCAD": 3,  "USDJPY": 3,
    "EURJPY": 5,  "GBPJPY": 5,  "AUDJPY": 5,  "CADJPY": 5,
    "XAUUSD": 50, "XAGUSD": 20,
}
MT5_MIN_SL_PIPS_DEFAULT = 3


def _get_instrument_params(symbol: str, entry: float) -> tuple:
    """
    Return (pip_size, pip_val_usd_per_standard_lot).

    pip_val_usd = USD value of 1 pip on 1 standard lot (100,000 units).

    Correct formulas:
      xxxUSD  (EURUSD, GBPUSD, AUDUSD, NZDUSD) : pip_val = 10.0  (fixed, quote=USD)
      USDxxx  (USDCHF, USDCAD)                 : pip_val = 10.0 / entry
      JPY cross (USDJPY, EURJPY, GBPJPY)        : pip_val = (0.01 * 100000) / entry
      Gold XAUUSD                               : pip_val = 1.0
    """
    s = symbol.upper()

    if "XAU" in s or "GOLD" in s:
        return 0.01, 1.0
    if "XAG" in s or "SILVER" in s:
        return 0.001, 5.0
    if "OIL" in s or "WTI" in s or "BRENT" in s:
        return 0.01, 10.0
    if "BTC" in s:
        return 1.0, 1.0

    if "JPY" in s:
        pip_size = 0.01
        pip_val  = (pip_size * 100000) / entry if entry > 0 else 10.0
        return pip_size, pip_val

    pip_size = 0.0001

    if s.endswith("USD"):
        # Quote = USD, pip value is always $10/lot regardless of price
        return pip_size, 10.0

    if s.startswith("USD") and entry > 0:
        # Base = USD, quote is foreign currency — convert to USD
        # e.g. USDCHF at 0.82: $10 / 0.82 = $12.20/lot
        # e.g. USDCAD at 1.39: $10 / 1.39 = $7.19/lot
        return pip_size, 10.0 / entry

    return pip_size, 10.0


def get_min_sl_pips(symbol: str) -> int:
    """Return MT5 broker minimum SL distance in pips for the symbol."""
    return MT5_MIN_SL_PIPS.get(symbol.upper(), MT5_MIN_SL_PIPS_DEFAULT)


def enforce_min_sl(symbol: str, direction: str, entry: float, sl: float) -> float:
    """
    Push SL out to MT5 minimum distance if it is too tight.
    Returns the adjusted SL price. Does NOT move SL silently in live trading —
    caller should log a warning when this triggers.
    """
    pip_size = _get_instrument_params(symbol, entry)[0]
    min_dist = get_min_sl_pips(symbol) * pip_size
    if abs(entry - sl) < min_dist:
        sl = round(entry - min_dist if direction == "BUY" else entry + min_dist, 5)
    return sl


def calculate_lot_size(
    risk_usd: float,
    entry: float,
    stop_loss: float,
    symbol_info: dict,
    symbol: str = "",
) -> float:
    """
    Calculate lot size so that hitting SL costs approximately risk_usd.

    Strategy:
    - Compute exact (raw) lot from risk formula.
    - Round to nearest volume_step (not floor) — this keeps actual risk
      within ±1 step of target, which for 0.01 step is ±$0.10-$0.15 max.
    - Clamp to [volume_min, volume_max].
    - If raw_lot < volume_min (SL too wide for $1 risk), return volume_min
      and accept the under-risk rather than over-risking.

    Actual risk range with $1 target and 0.01 step:
      Best case: raw_lot rounds up  -> actual risk up to ~$1.50 (acceptable)
      Worst case: raw_lot rounds down -> actual risk down to ~$0.75 (acceptable)
      The $0.75-$1.50 band is inherent to 0.01 lot granularity at small risk.
      To tighten this band, broker must support 0.001 micro-lots.
    """
    volume_min  = float(symbol_info.get("volume_min",  0.01))
    volume_step = float(symbol_info.get("volume_step", 0.01))
    volume_max  = float(symbol_info.get("volume_max",  100.0))

    sl_distance = abs(entry - stop_loss)
    if sl_distance == 0:
        return volume_min

    pip_size, pip_val_usd = _get_instrument_params(symbol, entry)

    # Enforce MT5 minimum SL distance before sizing
    min_dist = get_min_sl_pips(symbol) * pip_size
    if sl_distance < min_dist:
        sl_distance = min_dist

    sl_pips           = sl_distance / pip_size
    sl_dollar_per_lot = sl_pips * pip_val_usd

    if sl_dollar_per_lot <= 0:
        return volume_min

    raw_lot = risk_usd / sl_dollar_per_lot

    # Round to nearest step (not floor) — keeps risk closest to target
    steps = round(raw_lot / volume_step)
    lot   = round(steps * volume_step, 8)

    # Clamp to broker limits
    lot = max(volume_min, min(volume_max, lot))

    return round(lot, 2)


def actual_risk_usd(lot: float, entry: float, stop_loss: float, symbol: str) -> float:
    """Return the actual USD risk for a given lot size and SL distance."""
    pip_size, pip_val_usd = _get_instrument_params(symbol, entry)
    sl_pips = abs(entry - stop_loss) / pip_size
    return round(lot * sl_pips * pip_val_usd, 4)


def calculate_sl_tp(
    direction: str,
    entry: float,
    atr: float,
    rr_ratio: float = 2.0,
    atr_multiplier: float = 1.5,
    symbol: str = "",
) -> dict:
    sl_distance = atr * atr_multiplier
    tp_distance = sl_distance * rr_ratio

    # Enforce MT5 minimum SL
    pip_size, _ = _get_instrument_params(symbol, entry)
    min_dist    = get_min_sl_pips(symbol) * pip_size
    sl_distance = max(sl_distance, min_dist)
    tp_distance = sl_distance * rr_ratio

    if direction == "BUY":
        sl = round(entry - sl_distance, 5)
        tp = round(entry + tp_distance, 5)
    else:
        sl = round(entry + sl_distance, 5)
        tp = round(entry - tp_distance, 5)

    sl_pips = round(sl_distance / pip_size, 1)
    tp_pips = round(tp_distance / pip_size, 1)

    return {"sl": sl, "tp": tp, "sl_pips": sl_pips, "tp_pips": tp_pips}
