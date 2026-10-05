try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    mt5 = None
    HAS_MT5 = False

def place_order(symbol: str, direction: str, lot: float, entry: float, sl: float, tp: float, comment: str = "ForexAI") -> dict:
    # 1. MetaApi Cloud execution (Priority for 24/7 cloud deployments like Render)
    from app.mt5.metaapi_client import metaapi_client
    if metaapi_client.is_configured:
        return metaapi_client.place_order(symbol=symbol, direction=direction, lot=lot, sl=sl, tp=tp, comment=comment)

    # 2. Local Windows MT5 Terminal execution
    if HAS_MT5 and mt5 is not None:
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
        try:
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return {"success": True, "ticket": result.order}
            return {"success": False, "error": getattr(result, "comment", "MT5 order rejected")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    return {
        "success": False,
        "error": "No execution bridge available. Configure MetaApi in Settings or run MT5 locally on Windows."
    }

def close_order(ticket: int, symbol: str, lot: float, direction: str) -> dict:
    from app.mt5.metaapi_client import metaapi_client
    if metaapi_client.is_configured:
        return metaapi_client.close_order(str(ticket))

    if HAS_MT5 and mt5 is not None:
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
        try:
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return {"success": True, "ticket": result.order}
            return {"success": False, "error": getattr(result, "comment", "Close failed")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    return {"success": False, "error": "No terminal available"}
