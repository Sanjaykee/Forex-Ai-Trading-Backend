"""
Data-source router.

Resolves the per-user `data_source` preference (saved from the Settings page)
and returns matching candle / live-price / symbol-info fetchers.

Resolved priority (NO Yahoo fallback):
  'mt5'      -> local MetaTrader 5 terminal (Windows, free)
  'ctrader'  -> cTrader Open API
  'metaapi'  -> MetaApi cloud
  'auto'     -> local MT5 -> MetaApi cloud

If the resolved source is unavailable, no-data fetchers are returned so the
app reports "no data" instead of silently substituting a different feed.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def _resolve_user_data_source(explicit: str = "auto") -> str:
    """Return the effective data source ('mt5'|'metaapi'|'ctrader'|'auto').

    Reads the per-user `data_source` setting from the DB when `explicit` is
    'auto'. This lets a user who chose 'Local PC (MT5)' in Settings stop the
    scanner/backtest from using MetaApi cloud.
    """
    if explicit and str(explicit).strip().lower() not in ("", "auto"):
        return str(explicit).strip().lower()
    try:
        from app.db.database import SessionLocal
        from app.db.models import User
        from app.db import crud
        db = SessionLocal()
        try:
            for u in db.query(User).order_by(User.id.desc()).all():
                v = crud.get_setting(db, u.id, "data_source")
                if v:
                    return v.strip().lower()
        finally:
            db.close()
    except Exception:
        pass
    return "auto"


def _mt5_available() -> bool:
    """Return True if a local MT5 terminal is reachable."""
    try:
        import MetaTrader5 as mt5
        return bool(mt5.initialize())
    except Exception:
        return False


_cached_ctrader_client = None

def _build_ctrader_client():
    """Return a configured CTraderClient singleton, or None."""
    global _cached_ctrader_client
    try:
        from app.mt5.ctrader_client import CTraderClient
        from app.config import settings as _cfg
        from app.db.database import SessionLocal
        from app.db.models import User

        if not (_cfg.CTRADER_CLIENT_ID and _cfg.CTRADER_CLIENT_SECRET):
            return None

        u = None
        db = SessionLocal()
        try:
            u = db.query(User).filter(
                User.ctrader_account_id.isnot(None),
                User.ctrader_account_id != "",
            ).order_by(User.id.desc()).first()
        finally:
            db.close()
        if not u or not u.ctrader_access_token:
            return None

        # Reuse existing client if tokens/account match
        if _cached_ctrader_client and _cached_ctrader_client.is_configured:
            if (_cached_ctrader_client.account_id == str(u.ctrader_account_id).strip() and
                _cached_ctrader_client._token == str(u.ctrader_access_token).strip()):
                return _cached_ctrader_client

        _cached_ctrader_client = CTraderClient(
            client_id=_cfg.CTRADER_CLIENT_ID,
            client_secret=_cfg.CTRADER_CLIENT_SECRET,
            account_id=u.ctrader_account_id,
            server=u.ctrader_server or "demo.ctrader.com",
            access_token=u.ctrader_access_token,
            refresh_token=u.ctrader_refresh_token,
        )
        return _cached_ctrader_client if _cached_ctrader_client.is_configured else None
    except Exception:
        return None


# ── No-data fetchers (used instead of any external fallback) ────────────────

def _no_data_candles(symbol: str, timeframe: str, count: int = 500):
    return pd.DataFrame()


def _no_data_candles_range(symbol: str, timeframe: str, from_date, to_date):
    return pd.DataFrame()


def _no_data_live_price(symbol: str) -> dict:
    return {}


def _no_data_symbol_info(symbol: str) -> dict:
    return {}


PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
         "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"]


def _all_pairs() -> list:
    return list(PAIRS)


# ── Resolved fetchers ────────────────────────────────────────────────────────

def get_candles_fetcher(data_source: str = "auto"):
    """Return a callable `get_candles(symbol, timeframe, count) -> DataFrame`
    for the resolved data source. Returns a no-data fetcher when unavailable."""
    ds = _resolve_user_data_source(data_source)

    if ds == "mt5":
        if _mt5_available():
            from app.mt5.data_fetcher import get_candles as _candles
            return _candles
        logger.warning("[SourceRouter] data_source='mt5' but MT5 terminal is not available — no data")
        return _no_data_candles

    if ds == "ctrader":
        client = _build_ctrader_client()
        if client is not None:
            return client.get_candles
        logger.warning("[SourceRouter] data_source='ctrader' but cTrader is not connected — no data")
        return _no_data_candles

    if ds == "metaapi":
        from app.mt5.metaapi_client import metaapi_client
        if metaapi_client.is_configured:
            return metaapi_client.get_candles
        logger.warning("[SourceRouter] data_source='metaapi' but MetaApi is not configured — no data")
        return _no_data_candles

    if ds == "auto":
        if _mt5_available():
            from app.mt5.data_fetcher import get_candles as _candles
            return _candles
        from app.mt5.metaapi_client import metaapi_client
        if metaapi_client.is_configured:
            return metaapi_client.get_candles
        logger.warning("[SourceRouter] no data source available — no data")
        return _no_data_candles

    logger.warning("[SourceRouter] unknown data_source=%r — no data", ds)
    return _no_data_candles


def get_candles_range_fetcher(data_source: str = "auto"):
    """Return a callable `get_candles_range(symbol, timeframe, from_date, to_date) -> DataFrame`
    for date-range fetching (no count limit). Falls back to no-data when unavailable."""
    ds = _resolve_user_data_source(data_source)

    if ds in ("mt5", "auto") and _mt5_available():
        from app.mt5.data_fetcher import get_candles_range as _cr
        return _cr

    # For historical backtests: if local MT5 is running on PC, use it regardless of cloud execution setting
    if _mt5_available():
        from app.mt5.data_fetcher import get_candles_range as _cr
        return _cr

    logger.warning("[SourceRouter] get_candles_range_fetcher: only MT5 supports range fetch — no data")
    return _no_data_candles_range


def get_fetcher(data_source: str = "auto"):
    """Return (get_candles, get_live_price, get_symbol_info, get_all_pairs)
    for the resolved data source. Kept for existing callers."""
    ds = _resolve_user_data_source(data_source)
    candles = get_candles_fetcher(ds)

    # Local MT5 terminal
    if ds in ("mt5", "auto") and _mt5_available():
        from app.mt5.data_fetcher import (
            get_live_price as _lp,
            get_symbol_info as _si,
            get_all_pairs as _ap,
        )
        return candles, _lp, _si, _ap

    # cTrader Open API
    if ds == "ctrader":
        client = _build_ctrader_client()
        if client is not None:
            return candles, client.get_live_price, _no_data_symbol_info, _all_pairs

    # MetaApi cloud
    if ds in ("metaapi", "auto"):
        from app.mt5.metaapi_client import metaapi_client
        if metaapi_client.is_configured:
            return candles, metaapi_client.get_live_price, _no_data_symbol_info, _all_pairs

    # No available source
    return candles, _no_data_live_price, _no_data_symbol_info, _all_pairs
