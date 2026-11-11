try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    mt5 = None
    HAS_MT5 = False


def _place_order_mt5(symbol: str, direction: str, lot: float, sl: float, tp: float,
                     comment: str, entry: float = None) -> dict:
    """Place a limit order at exact signal entry on the local MT5 terminal.

    Always uses LIMIT at signal entry price to match backtest retest fill behavior.
    If entry is None: falls back to MARKET at current tick (manual/emergency use only).
    """
    if not HAS_MT5 or mt5 is None:
        return {"success": False, "error": "MetaTrader5 package not installed"}
    if not mt5.initialize():
        return {"success": False, "error": "MT5 terminal not running or not logged in"}

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"success": False, "error": f"No tick for {symbol} — add it to Market Watch"}

    digits = 3 if "JPY" in symbol.upper() else 5

    if entry:
        price      = round(float(entry), digits)
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction.upper() == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT
        action     = mt5.TRADE_ACTION_PENDING
        filling    = mt5.ORDER_FILLING_RETURN
        otype_str  = "ORDER_TYPE_BUY_LIMIT" if direction.upper() == "BUY" else "ORDER_TYPE_SELL_LIMIT"
    else:
        price      = tick.ask if direction.upper() == "BUY" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if direction.upper() == "BUY" else mt5.ORDER_TYPE_SELL
        action     = mt5.TRADE_ACTION_DEAL
        filling    = mt5.ORDER_FILLING_IOC
        otype_str  = "ORDER_TYPE_MARKET"

    request = {
        "action":       action,
        "symbol":       symbol,
        "volume":       float(lot),
        "type":         order_type,
        "price":        price,
        "sl":           float(sl) if sl else None,
        "tp":           float(tp) if tp else None,
        "deviation":    20,
        "magic":        20240101,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

    try:
        result = mt5.order_send(request)
    except Exception as e:
        return {"success": False, "error": f"MT5 order_send exception: {e}"}

    if result is None:
        return {"success": False, "error": f"MT5 order_send returned None: {mt5.last_error()}"}

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        return {
            "success":     True,
            "ticket":      str(result.order),
            "order_type":  otype_str,
            "fill_price":  getattr(result, "price", price),
            "actual_lots": float(lot),
        }
    return {"success": False, "error": f"MT5 rejected ({result.retcode}): {result.comment}"}


def _close_order_mt5(ticket, symbol: str, lot: float, direction: str) -> dict:
    """Close a position on the local MT5 terminal."""
    if not HAS_MT5 or mt5 is None:
        return {"success": False, "error": "MetaTrader5 package not installed"}
    if not mt5.initialize():
        return {"success": False, "error": "MT5 terminal not running"}

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"success": False, "error": f"No tick for {symbol}"}

    close_type = mt5.ORDER_TYPE_SELL if direction.upper() == "BUY" else mt5.ORDER_TYPE_BUY
    price = tick.bid if direction.upper() == "BUY" else tick.ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       float(lot),
        "type":         close_type,
        "position":     int(ticket),
        "price":        price,
        "deviation":    20,
        "magic":        20240101,
        "comment":      "ForexAI Close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    try:
        result = mt5.order_send(request)
    except Exception as e:
        return {"success": False, "error": str(e)}

    if result is None:
        return {"success": False, "error": f"MT5 close returned None: {mt5.last_error()}"}

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        return {"success": True, "ticket": str(result.order)}
    return {"success": False, "error": f"MT5 close rejected ({result.retcode}): {result.comment}"}


def place_order(symbol: str, direction: str, lot: float, entry: float, sl: float, tp: float,
                comment: str = "ForexAI", metaapi_token: str = None, metaapi_account_id: str = None,
                client_order_id: str = None) -> dict:
    """Place an order through the ACTIVE data source (dynamic).

    Routes by the saved `data_source` preference:
      mt5 -> local MT5 terminal, ctrader -> cTrader, metaapi -> MetaApi cloud.
    """
    from app.mt5.source_router import _resolve_user_data_source, _build_ctrader_client, _mt5_available
    ds = _resolve_user_data_source("auto")

    # 1. cTrader Open API
    if ds == "ctrader":
        client = _build_ctrader_client()
        if client is not None:
            res = client.place_order(symbol=symbol, direction=direction, volume=lot, sl=sl, tp=tp, comment=comment)
            if res.get("success"):
                return res
            return {"success": False, "error": res.get("error", "cTrader order failed")}

    # 2. Local MT5 terminal
    if ds in ("mt5", "auto") and _mt5_available():
        return _place_order_mt5(symbol, direction, lot, sl, tp, comment, entry=entry)

    # 3. MetaApi cloud
    if ds in ("metaapi", "auto"):
        from app.mt5.metaapi_client import MetaApiClient
        client = MetaApiClient(token=metaapi_token, account_id=metaapi_account_id)
        if client.is_configured:
            return client.place_order(symbol=symbol, direction=direction, lot=lot, entry=entry,
                                      sl=sl, tp=tp, comment=comment, client_order_id=client_order_id)

    return {"success": False, "error": "No execution bridge available. Select a data source in Settings."}


def cancel_pending_order(ticket: str, symbol: str = None, direction: str = None) -> dict:
    """Cancel a pending LIMIT order through the active data source."""
    from app.mt5.source_router import _resolve_user_data_source, _mt5_available
    ds = _resolve_user_data_source("auto")

    if ds in ("mt5", "auto") and _mt5_available():
        if not HAS_MT5 or mt5 is None:
            return {"success": False, "error": "MT5 not available"}
        if not mt5.initialize():
            return {"success": False, "error": "MT5 terminal not running"}
        request = {
            "action":  mt5.TRADE_ACTION_REMOVE,
            "order":   int(ticket),
        }
        try:
            result = mt5.order_send(request)
        except Exception as e:
            return {"success": False, "error": str(e)}
        if result is None:
            return {"success": False, "error": f"MT5 cancel returned None: {mt5.last_error()}"}
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {"success": True, "ticket": ticket}
        return {"success": False, "error": f"MT5 cancel rejected ({result.retcode}): {result.comment}"}

    if ds in ("metaapi", "auto"):
        from app.mt5.metaapi_client import metaapi_client
        if metaapi_client.is_configured:
            return metaapi_client.cancel_order(str(ticket))

    return {"success": False, "error": "No terminal available"}


def close_order(ticket, symbol: str, lot: float, direction: str) -> dict:
    """Close a position through the ACTIVE data source (dynamic)."""
    from app.mt5.source_router import _resolve_user_data_source, _build_ctrader_client, _mt5_available
    ds = _resolve_user_data_source("auto")

    # cTrader Open API
    if ds == "ctrader":
        client = _build_ctrader_client()
        if client is not None:
            return client.close_order(str(ticket), volume=lot)

    # Local MT5 terminal
    if ds in ("mt5", "auto") and _mt5_available():
        return _close_order_mt5(ticket, symbol, lot, direction)

    # MetaApi cloud
    if ds in ("metaapi", "auto"):
        from app.mt5.metaapi_client import metaapi_client
        if metaapi_client.is_configured:
            return metaapi_client.close_order(str(ticket))

    return {"success": False, "error": "No terminal available"}
