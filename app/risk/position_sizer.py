def _get_instrument_params(symbol: str, entry: float) -> tuple:
    """Return (pip_size, pip_val_usd_per_lot) for any instrument."""
    s = symbol.upper()
    if "XAU" in s or "GOLD" in s:
        # Gold: contract = 100 oz, pip = $0.01, pip value = 100 * 0.01 = $1 per lot
        return 0.01, 1.0
    if "XAG" in s or "SILVER" in s:
        # Silver: contract = 5000 oz, pip = $0.001
        return 0.001, 5.0
    if "OIL" in s or "WTI" in s or "BRENT" in s:
        # Oil: contract = 1000 barrels, pip = $0.01
        return 0.01, 10.0
    if "BTC" in s:
        # Bitcoin: 1 pip = $1, contract = 1 BTC
        return 1.0, 1.0
    if "JPY" in s:
        pip_size = 0.01
        pip_val = (pip_size * 100000) / entry if entry > 0 else 10.0
        return pip_size, pip_val
    if s.endswith("USD"):
        return 0.0001, 10.0   # e.g. EURUSD, GBPUSD — $10 per pip per standard lot
    if s.startswith("USD") and entry > 0:
        return 0.0001, (0.0001 * 100000) / entry
    return 0.0001, 10.0


def calculate_lot_size(
    risk_usd: float,
    entry: float,
    stop_loss: float,
    symbol_info: dict,
    symbol: str = "",
) -> float:
    volume_min  = symbol_info.get("volume_min",  0.01)
    volume_step = symbol_info.get("volume_step", 0.01)

    sl_distance = abs(entry - stop_loss)
    if sl_distance == 0:
        return volume_min

    pip_size, pip_val_usd = _get_instrument_params(symbol, entry)
    sl_pips = sl_distance / pip_size
    sl_dollar_per_lot = sl_pips * pip_val_usd

    if sl_dollar_per_lot <= 0:
        return volume_min

    raw_lot = risk_usd / sl_dollar_per_lot
    lot = max(volume_min, round(round(raw_lot / volume_step) * volume_step, 2))
    return lot


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

    if direction == "BUY":
        sl = round(entry - sl_distance, 5)
        tp = round(entry + tp_distance, 5)
    else:
        sl = round(entry + sl_distance, 5)
        tp = round(entry - tp_distance, 5)

    pip_size, _ = _get_instrument_params(symbol, entry)
    sl_pips = round(sl_distance / pip_size, 1)
    tp_pips = round(tp_distance / pip_size, 1)

    return {"sl": sl, "tp": tp, "sl_pips": sl_pips, "tp_pips": tp_pips}
