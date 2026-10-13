import requests
from datetime import datetime, timedelta, timezone

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
HIGH_IMPACT_CURRENCIES = ["USD", "EUR", "GBP", "JPY"]
BUFFER_MINUTES = 30

# Cache: avoid hitting ForexFactory on every pair scan
_cache = {"events": [], "fetched_at": None}
CACHE_TTL_MINUTES = 30


def get_news_events() -> list:
    now = datetime.now(timezone.utc)
    if _cache["fetched_at"] and (now - _cache["fetched_at"]).total_seconds() < CACHE_TTL_MINUTES * 60:
        return _cache["events"]
    try:
        res = requests.get(CALENDAR_URL, timeout=5)
        _cache["events"] = res.json()
        _cache["fetched_at"] = now
        return _cache["events"]
    except Exception:
        return _cache["events"]  # return stale cache on failure rather than empty


def is_news_safe() -> bool:
    now = datetime.now(timezone.utc)
    for event in get_news_events():
        if event.get("impact") != "High":
            continue
        if event.get("currency") not in HIGH_IMPACT_CURRENCIES:
            continue
        try:
            event_time = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
            diff = abs((event_time - now).total_seconds() / 60)
            if diff <= BUFFER_MINUTES:
                return False
        except Exception:
            continue
    return True
