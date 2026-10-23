from datetime import datetime, timezone

KILLZONES = {
    "Asian Open":    (0,  5),   # 00:00 - 05:00 UTC  (05:30 - 10:30 IST)
    "London Open":   (7, 11),   # 07:00 - 11:00 UTC  (12:30 - 16:30 IST)
    "New York Open": (12, 17),  # 12:00 - 17:00 UTC  (17:30 - 22:30 IST)
}

def get_active_killzone() -> dict:
    hour = datetime.now(timezone.utc).hour
    for name, (start, end) in KILLZONES.items():
        if start <= hour < end:
            return {"active": True, "name": name, "high_probability": True}
    return {"active": False, "name": None, "high_probability": False}

def is_killzone_active() -> bool:
    return get_active_killzone()["active"]
