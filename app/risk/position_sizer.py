def calculate_lot_size(
    risk_usd: float,
    entry: float,
    stop_loss: float,
    symbol_info: dict,
    symbol: str = "",
) -> float:
    contract_size = symbol_info.get("trade_contract_size", 100000)
    volume_min    = symbol_info.get("volume_min", 0.01)
    volume_step   = symbol_info.get("volume_step", 0.01)

    sl_distance   = abs(entry - stop_loss)
    if sl_distance == 0:
        return volume_min

    pip_size = 0.01 if "JPY" in symbol or entry > 50 else 0.0001
    sl_pips  = sl_distance / pip_size

    # Calculate pip value in USD per 1.0 standard lot
    if symbol.endswith("USD"):
        pip_val_usd = pip_size * contract_size
    elif "JPY" in symbol and entry > 0:
        pip_val_usd = (pip_size * contract_size) / entry
    elif symbol.startswith("USD") and entry > 0:
        pip_val_usd = (pip_size * contract_size) / entry
    else:
        pip_val_usd = 10.0

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

    pip_size = 0.01 if "JPY" in symbol else 0.0001
    sl_pips  = round(sl_distance / pip_size, 1)
    tp_pips  = round(tp_distance / pip_size, 1)

    return {"sl": sl, "tp": tp, "sl_pips": sl_pips, "tp_pips": tp_pips}
