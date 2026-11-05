import requests
from datetime import datetime, timedelta, timezone

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
HIGH_IMPACT_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]
BUFFER_BEFORE_MINUTES = 30   # block 30 min before news
BUFFER_AFTER_MINUTES  = 15   # block 15 min after news (spread stays wide)

# Symbol -> currencies affected
SYMBOL_CURRENCIES = {
    "EURUSD": ["EUR", "USD"], "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"], "USDCHF": ["USD", "CHF"],
    "AUDUSD": ["AUD", "USD"], "NZDUSD": ["NZD", "USD"],
    "USDCAD": ["USD", "CAD"], "EURJPY": ["EUR", "JPY"],
    "GBPJPY": ["GBP", "JPY"],
}

_cache = {"events": [], "fetched_at": None}
CACHE_TTL_MINUTES = 15   # Issue 8: refresh every 15 min (was 30)


def _fetch_and_store() -> None:
    """Fetch news from calendar and update cache. Called by background thread."""
    now = datetime.now(timezone.utc)
    try:
        res = requests.get(CALENDAR_URL, timeout=5)
        events = res.json()
        _cache["events"] = [
            e for e in events
            if e.get("impact") == "High"
            and e.get("currency") in HIGH_IMPACT_CURRENCIES
        ]
        _cache["fetched_at"] = now
    except Exception:
        pass  # keep stale cache on failure


def _start_background_refresh() -> None:
    """Issue 8: Background thread refreshes news cache every 15 min on a fixed schedule."""
    import threading

    def _loop():
        while True:
            import time
            time.sleep(CACHE_TTL_MINUTES * 60)
            _fetch_and_store()

    t = threading.Thread(target=_loop, daemon=True, name="news-cache-refresh")
    t.start()


# Kick off background refresh on module import
_fetch_and_store()          # initial load
_start_background_refresh() # schedule periodic refresh


def get_news_events() -> list:
    now = datetime.now(timezone.utc)
    # Force refresh if cache is stale (background thread may not have run yet)
    if _cache["fetched_at"] is None or \
            (now - _cache["fetched_at"]).total_seconds() > CACHE_TTL_MINUTES * 60:
        _fetch_and_store()
    return _cache["events"]


def is_news_safe(symbol: str = "") -> bool:
    """
    Return False if a high-impact news event is within the buffer window
    for any currency in the given symbol.
    If symbol is empty, checks all high-impact currencies (legacy behaviour).
    """
    now = datetime.now(timezone.utc)
    affected = SYMBOL_CURRENCIES.get(symbol.upper(), []) if symbol else HIGH_IMPACT_CURRENCIES

    for event in get_news_events():
        currency = event.get("currency", "")
        if affected and currency not in affected:
            continue
        try:
            event_time = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
            diff_minutes = (event_time - now).total_seconds() / 60
            # block BUFFER_BEFORE_MINUTES before and BUFFER_AFTER_MINUTES after
            if -BUFFER_AFTER_MINUTES <= diff_minutes <= BUFFER_BEFORE_MINUTES:
                return False
        except Exception:
            continue
    return True


def get_upcoming_news(symbol: str = "", lookahead_minutes: int = 60) -> list:
    """
    Return list of upcoming high-impact events within lookahead_minutes
    for the given symbol's currencies. Used for Telegram alerts.
    """
    now = datetime.now(timezone.utc)
    affected = SYMBOL_CURRENCIES.get(symbol.upper(), HIGH_IMPACT_CURRENCIES)
    upcoming = []
    for event in get_news_events():
        if event.get("currency") not in affected:
            continue
        try:
            event_time = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
            diff_minutes = (event_time - now).total_seconds() / 60
            if 0 <= diff_minutes <= lookahead_minutes:
                upcoming.append({
                    "currency": event.get("currency"),
                    "title":    event.get("title", ""),
                    "minutes":  round(diff_minutes),
                    "time_utc": event_time.strftime("%H:%M UTC"),
                })
        except Exception:
            continue
    return upcoming
