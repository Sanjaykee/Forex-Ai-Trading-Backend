from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db import crud
from app.api.dependencies import get_current_user
from app.api.schemas.signal_schema import SettingsUpdate, MT5Setup

trades_router      = APIRouter(prefix="/trades",      tags=["trades"])
performance_router = APIRouter(prefix="/performance", tags=["performance"])
settings_router    = APIRouter(prefix="/settings",    tags=["settings"])
pairs_router       = APIRouter(prefix="/pairs",       tags=["pairs"])

@trades_router.get("/")
def get_trades(db: Session = Depends(get_db), user=Depends(get_current_user)):
    return crud.get_trades_by_user(db, user.id)

@performance_router.get("/")
def get_performance(db: Session = Depends(get_db), user=Depends(get_current_user)):
    return crud.get_performance_by_user(db, user.id)

@settings_router.get("/")
def get_settings(db: Session = Depends(get_db), user=Depends(get_current_user)):
    keys = ["max_risk_usd", "max_trades_day", "max_losses_day", "rr_ratio",
            "london_session", "newyork_session", "telegram_alerts", "data_source"]
    res = {k: crud.get_setting(db, user.id, k) for k in keys}
    res["mt5_login"] = user.mt5_login or ""
    res["mt5_server"] = user.mt5_server or ""
    res["telegram_chat_id"] = user.telegram_chat_id or ""
    res["has_mt5_password"] = bool(user.mt5_password)
    try:
        from app.chat.gemini_service import get_gemini_api_key
        res["has_gemini_key"] = bool(get_gemini_api_key())
    except Exception:
        res["has_gemini_key"] = False
    if not res.get("data_source"):
        res["data_source"] = "mt5" if user.mt5_login else "yfinance"
    return res

@settings_router.put("/")
def update_settings(data: SettingsUpdate, db: Session = Depends(get_db), user=Depends(get_current_user)):
    from app.db.models import User
    for key, value in data.model_dump(exclude_none=True).items():
        if key == "telegram_chat_id":
            u = db.query(User).filter(User.id == user.id).first()
            if u:
                u.telegram_chat_id = str(value).strip()
                db.commit()
        else:
            crud.upsert_setting(db, user.id, key, str(value))
    return {"message": "Settings updated"}

@settings_router.post("/telegram/test")
def test_telegram(data: dict = None, db: Session = Depends(get_db), user=Depends(get_current_user)):
    from app.notifications.telegram_bot import test_telegram_bot
    chat_id = (data.get("telegram_chat_id") if data else None) or user.telegram_chat_id
    return test_telegram_bot(chat_id)

@settings_router.post("/mt5")
def setup_mt5(data: MT5Setup, db: Session = Depends(get_db), user=Depends(get_current_user)):
    password_to_use = data.mt5_password if (data.mt5_password and data.mt5_password.strip()) else user.mt5_password
    crud.update_mt5_credentials(db, user.id, data.mt5_login, password_to_use, data.mt5_server)
    if data.telegram_chat_id is not None:
        from app.db.models import User
        u = db.query(User).filter(User.id == user.id).first()
        if u:
            u.telegram_chat_id = str(data.telegram_chat_id).strip()
            db.commit()
    
    crud.upsert_setting(db, user.id, "data_source", "mt5")

    ok = False
    try:
        from app.mt5.connection import connect
        ok = connect(login=int(data.mt5_login), password=password_to_use, server=data.mt5_server.strip())
    except Exception:
        ok = False

    if not ok:
        import platform
        if platform.system() != "Windows":
            return {
                "message": "MT5 credentials saved to cloud database. Note: Desktop MT5 terminal runs locally on Windows (cloud server automatically uses Yahoo Finance market data).",
                "connected": False
            }
        return {"message": "Credentials saved, but MT5 terminal failed to authorize. Check login & password.", "connected": False}

    return {"message": "MT5 connected successfully", "connected": True}

@settings_router.post("/metaapi")
def setup_metaapi(data: dict, db: Session = Depends(get_db), user=Depends(get_current_user)):
    from app.db.models import User
    from app.mt5.metaapi_client import MetaApiClient

    token = data.get("metaapi_token", "").strip()
    account_id = data.get("metaapi_account_id", "").strip()
    region = data.get("metaapi_region", "new-york").strip() or "new-york"

    # Save per-user credentials directly on the User row
    u = db.query(User).filter(User.id == user.id).first()
    if u:
        if token:
            u.metaapi_token = token
        if account_id:
            u.metaapi_account_id = account_id
        db.commit()
    crud.upsert_setting(db, user.id, "data_source", "metaapi")

    client = MetaApiClient(token=token or (u.metaapi_token if u else ""), account_id=account_id or (u.metaapi_account_id if u else ""), region=region)
    info = client.get_account_information()
    if info.get("connected"):
        return {
            "success": True,
            "connected": True,
            "message": f"Connected to MT5 ({info.get('server')})! Balance: ${info.get('balance')} {info.get('currency')}",
            "account": info
        }
    return {
        "success": False,
        "connected": False,
        "message": "MetaApi credentials saved, but could not authorize. Check Token and Account ID in app.metaapi.cloud."
    }

@settings_router.get("/metaapi/status")
def get_metaapi_status(db: Session = Depends(get_db), user=Depends(get_current_user)):
    from app.db.models import User
    from app.mt5.metaapi_client import MetaApiClient

    u = db.query(User).filter(User.id == user.id).first()
    token = u.metaapi_token if u else None
    account_id = u.metaapi_account_id if u else None
    region = crud.get_setting(db, user.id, "metaapi_region") or "new-york"

    if not token or not account_id:
        return {"configured": False, "connected": False}

    client = MetaApiClient(token=token, account_id=account_id, region=region)
    info = client.get_account_information()
    return {
        "configured": True,
        "connected": bool(info.get("connected")),
        "account": info
    }

@pairs_router.get("/")
def get_pairs():
    try:
        from app.mt5.source_router import get_fetcher
        _, _, _, get_all_pairs = get_fetcher()
        return get_all_pairs()
    except Exception:
        return [
            {"symbol": "EURUSD", "digits": 5, "spread": 1.2, "active": True},
            {"symbol": "GBPUSD", "digits": 5, "spread": 1.5, "active": True},
            {"symbol": "USDJPY", "digits": 3, "spread": 1.4, "active": True},
            {"symbol": "AUDUSD", "digits": 5, "spread": 1.3, "active": True},
            {"symbol": "USDCAD", "digits": 5, "spread": 1.6, "active": True},
            {"symbol": "USDCHF", "digits": 5, "spread": 1.5, "active": True},
            {"symbol": "NZDUSD", "digits": 5, "spread": 1.8, "active": True},
            {"symbol": "EURJPY", "digits": 3, "spread": 1.7, "active": True},
            {"symbol": "GBPJPY", "digits": 3, "spread": 2.1, "active": True},
        ]
