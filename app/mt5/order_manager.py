import MetaTrader5 as mt5

def place_order(symbol: str, direction: str, lot: float, entry: float, sl: float, tp: float, comment: str = "ForexAI") -> dict:
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         order_type,
        "price":        entry,
        "sl":           sl,
        "tp":           tp,
        "deviation":    10,
        "magic":        20240101,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"success": False, "error": result.comment}
    return {"success": True, "ticket": result.order}

def close_order(ticket: int, symbol: str, lot: float, direction: str) -> dict:
    close_type = mt5.ORDER_TYPE_SELL if direction == "BUY" else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)
    price = tick.bid if direction == "BUY" else tick.ask
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         close_type,
        "position":     ticket,
        "price":        price,
        "deviation":    10,
        "magic":        20240101,
        "comment":      "ForexAI Close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"success": False, "error": result.comment}
    return {"success": True, "ticket": result.order}
