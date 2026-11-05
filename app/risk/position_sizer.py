import math
import os

# ── Per-trade risk amount ─────────────────────────────────────────────────────
# Override via environment: RISK_PER_TRADE_USD=1.00
# Default is $0.10 (scaling phase). auto_scanner.py enforces the scaling cap.
_RISK_PER_TRADE_USD_DEFAULT = 0.10

# ── Over-risk tolerance ───────────────────────────────────────────────────────
# When the requested risk cannot be expressed with the broker's minimum lot, the
# sizer must choose between over-risking and refusing the trade. Silently
# over-risking is what made a $0.10 target execute at ~$1.00 of real risk (10x).
# Default: refuse when the minimum achievable risk exceeds the target by more
# than this multiple. Set ALLOW_MIN_LOT_OVER_RISK=true to trade anyway.
MAX_RISK_MULTIPLE       = float(os.environ.get("MAX_RISK_MULTIPLE", "1.5"))
ALLOW_MIN_LOT_OVER_RISK = os.environ.get("ALLOW_MIN_LOT_OVER_RISK", "false").strip().lower() == "true"

# ── USDJPY rate for valuing JPY-cross pip risk ────────────────────────────────
# Pip value on a JPY pair is 1000 JPY per lot. Converting that to USD needs the
# USDJPY rate. For USDJPY itself the pair's own price IS that rate (exact). For
# EURJPY / GBPJPY the cross price is NOT a valid substitute — the previous code
# used it anyway, understating GBPJPY risk by ~20%. Used only as a fallback when
# the caller does not supply a live rate.
USDJPY_RATE_ESTIMATE    = float(os.environ.get("USDJPY_RATE_ESTIMATE", "150.0"))


def get_risk_per_trade() -> float:
    """Return RISK_PER_TRADE_USD env var as float, defaulting to 0.10."""
    try:
        return float(os.environ.get("RISK_PER_TRADE_USD", str(_RISK_PER_TRADE_USD_DEFAULT)))
    except (TypeError, ValueError):
        return _RISK_PER_TRADE_USD_DEFAULT


# ── MT5 minimum stop distance in pips per instrument ─────────────────────────
MT5_MIN_SL_PIPS = {
    "EURUSD": 3,  "GBPUSD": 3,  "AUDUSD": 3,  "NZDUSD": 3,
    "USDCHF": 3,  "USDCAD": 3,  "USDJPY": 3,
    "EURJPY": 5,  "GBPJPY": 5,  "AUDJPY": 5,  "CADJPY": 5,
    "XAUUSD": 50, "XAGUSD": 20,
}
MT5_MIN_SL_PIPS_DEFAULT = 3


def _get_instrument_params(symbol: str, entry: float, usd_jpy: float | None = None) -> tuple:
    """
    Return (pip_size, pip_val_usd_per_standard_lot).

    pip_val_usd = USD value of 1 pip on 1 standard lot (100,000 units).

    Correct formulas:
      xxxUSD  (EURUSD, GBPUSD, AUDUSD, NZDUSD) : pip_val = 10.0  (fixed, quote=USD)
      USDxxx  (USDCHF, USDCAD)                 : pip_val = 10.0 / entry
      JPY pair                                 : pip_val = 1000 JPY / USDJPY rate
                                                 - USDJPY: rate == entry (exact)
                                                 - crosses: rate must be supplied
                                                   via `usd_jpy`, else fallback
      Gold XAUUSD                               : pip_val = 1.0

    `usd_jpy` lets the caller pass a live USDJPY rate so JPY-cross risk is
    valued correctly instead of being derived from the cross's own price.
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
        # 1 pip on 100,000 units of a JPY pair = 1000 JPY, valued in USD.
        if s.startswith("USD"):
            rate = entry if entry > 0 else USDJPY_RATE_ESTIMATE
        elif usd_jpy and usd_jpy > 0:
            rate = usd_jpy
        else:
            rate = USDJPY_RATE_ESTIMATE
        pip_val = (1000.0 / rate) if rate > 0 else 10.0
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
    usd_jpy: float | None = None,
) -> float:
    """
    Calculate lot size so that hitting SL costs approximately risk_usd.

    Returns 0.0 when the requested risk cannot be honoured — meaning "skip this
    trade". Callers MUST treat 0.0 as a refusal, not as a size.

    Strategy:
    - Compute exact (raw) lot from risk formula.
    - Round to nearest volume_step.
    - Clamp to [volume_min, volume_max].
    - If the clamped lot risks more than `risk_usd * MAX_RISK_MULTIPLE`, the
      broker cannot express this risk and the trade is refused (0.0) unless
      ALLOW_MIN_LOT_OVER_RISK=true.

    Why the refusal exists:
    The previous implementation clamped raw_lot up to volume_min unconditionally.
    With a $0.10 target, a 10-pip EURUSD stop and a 0.01 minimum lot, raw_lot is
    0.001 -> rounds to 0 -> clamps to 0.01 -> real risk $1.00, i.e. 10x the
    target. That silently broke the scaling phase and made the 3% daily / 6%
    weekly circuit breakers trip ~10x sooner than the per-trade arithmetic
    predicted.
    """
    volume_min  = float(symbol_info.get("volume_min",  0.01))
    volume_step = float(symbol_info.get("volume_step", 0.01))
    volume_max  = float(symbol_info.get("volume_max",  100.0))

    sl_distance = abs(entry - stop_loss)
    if sl_distance == 0:
        return 0.0

    pip_size, pip_val_usd = _get_instrument_params(symbol, entry, usd_jpy)

    # Enforce MT5 minimum SL distance before sizing
    min_dist = get_min_sl_pips(symbol) * pip_size
    if sl_distance < min_dist:
        sl_distance = min_dist

    sl_pips           = sl_distance / pip_size
    sl_dollar_per_lot = sl_pips * pip_val_usd

    if sl_dollar_per_lot <= 0:
        return 0.0

    raw_lot = risk_usd / sl_dollar_per_lot

    # Round to nearest step (not floor) — keeps risk closest to target
    steps = round(raw_lot / volume_step)
    lot   = round(steps * volume_step, 8)

    # Clamp to broker limits
    lot = max(volume_min, min(volume_max, lot))

    # Refuse if the smallest tradeable size still exceeds the risk tolerance
    actual_risk = lot * sl_dollar_per_lot
    if actual_risk > risk_usd * MAX_RISK_MULTIPLE and not ALLOW_MIN_LOT_OVER_RISK:
        return 0.0

    return round(lot, 8)


def actual_risk_usd(lot: float, entry: float, stop_loss: float, symbol: str,
                    usd_jpy: float | None = None) -> float:
    """Return the actual USD risk for a given lot size and SL distance."""
    pip_size, pip_val_usd = _get_instrument_params(symbol, entry, usd_jpy)
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
