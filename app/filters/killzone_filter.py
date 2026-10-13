from datetime import datetime, timezone

# Only high-probability ICT Killzones — London Open & New York Open
KILLZONES = {
    "London Open":   (7,  9),   # 07:00 - 09:00 UTC  (12:30 - 14:30 IST)
    "New York Open": (12, 14),  # 12:00 - 14:00 UTC  (17:30 - 19:30 IST)
}

def get_active_killzone() -> dict:
    hour = datetime.now(timezone.utc).hour
    for name, (start, end) in KILLZONES.items():
        if start <= hour < end:
            return {"active": True, "name": name, "high_probability": True}
    return {"active": False, "name": None, "high_probability": False}

def is_killzone_active() -> bool:
    return get_active_killzone()["active"]
