def get_fetcher(data_source: str = "auto"):
    # 1. Check MetaApi Cloud first (if configured in cloud/Render)
    from app.mt5.metaapi_client import metaapi_client
    if (data_source in ("auto", "metaapi", "mt5")) and metaapi_client.is_configured:
        from app.mt5.yfinance_fetcher import get_symbol_info, get_all_pairs
        return metaapi_client.get_candles, metaapi_client.get_live_price, get_symbol_info, get_all_pairs

    # 2. Check local Windows MT5 terminal
    if data_source in ("auto", "mt5"):
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                from app.mt5.data_fetcher import get_candles, get_live_price, get_symbol_info, get_all_pairs
                return get_candles, get_live_price, get_symbol_info, get_all_pairs
        except Exception:
            pass

    # 3. Fallback to Yahoo Finance
    from app.mt5.yfinance_fetcher import get_candles, get_live_price, get_symbol_info, get_all_pairs
    return get_candles, get_live_price, get_symbol_info, get_all_pairs
