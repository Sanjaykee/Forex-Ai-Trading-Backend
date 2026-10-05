def get_fetcher(data_source: str = "yfinance"):
    if data_source == "mt5":
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                from app.mt5.data_fetcher import get_candles, get_live_price, get_symbol_info, get_all_pairs
                return get_candles, get_live_price, get_symbol_info, get_all_pairs
        except Exception:
            pass
    from app.mt5.yfinance_fetcher import get_candles, get_live_price, get_symbol_info, get_all_pairs
    return get_candles, get_live_price, get_symbol_info, get_all_pairs
