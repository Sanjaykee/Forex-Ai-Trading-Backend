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
    res["has_metaapi"] = bool(user.metaapi_token and user.metaapi_account_id)
    res["metaapi_account_id"] = user.metaapi_account_id or ""
    res["ctrader_account_id"] = user.ctrader_account_id or ""
    res["ctrader_server"]     = user.ctrader_server     or "demo.ctrader.com"
    res["ctrader_connected"]  = bool(user.ctrader_access_token)
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
    password_to_use = data.mt5_password if (data.mt5_password and data.mt5_password.strip()) else None
    if not password_to_use and user.mt5_password:
        from app.db.crud import get_mt5_password
        password_to_use = get_mt5_password(db.query(User).filter(User.id == user.id).first())
    crud.update_mt5_credentials(db, user.id, data.mt5_login, password_to_use or "", data.mt5_server)
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
        from app.db.crud import get_mt5_password
        u_fresh = db.query(User).filter(User.id == user.id).first()
        ok = connect(login=int(data.mt5_login), password=get_mt5_password(u_fresh), server=data.mt5_server.strip())
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

    u = db.query(User).filter(User.id == user.id).first()
    if u:
        if token:
            u.metaapi_token = token
        if account_id:
            u.metaapi_account_id = account_id
        db.commit()
    crud.upsert_setting(db, user.id, "data_source", "metaapi")
    crud.upsert_setting(db, user.id, "metaapi_region", region)

    # Try the selected region first, then auto-detect
    saved_token = token or (u.metaapi_token if u else "")
    saved_account = account_id or (u.metaapi_account_id if u else "")
    info = {}
    regions_to_try = [r for r in [region, "backup-new-york", "new-york", "london"] if r]
    # Deduplicate while preserving order
    regions_to_try = list(dict.fromkeys(regions_to_try))
    for try_region in regions_to_try:
        client = MetaApiClient(token=saved_token, account_id=saved_account, region=try_region)
        info = client.get_account_information()
        if info.get("connected"):
            crud.upsert_setting(db, user.id, "metaapi_region", try_region)
            break
    if info.get("connected"):
        crud.upsert_setting(db, user.id, "metaapi_connected", "true")
        return {
            "success": True,
            "connected": True,
            "message": f"Connected to MT5 ({info.get('server')})! Balance: ${info.get('balance')} {info.get('currency')}",
            "account": info
        }
    crud.upsert_setting(db, user.id, "metaapi_connected", "false")
    return {
        "success": False,
        "connected": False,
        "message": info.get("error") or "MetaApi credentials saved, but could not authorize. Check Token and Account ID in app.metaapi.cloud."
    }

@settings_router.post("/ctrader")
def setup_ctrader(data: dict, db: Session = Depends(get_db), user=Depends(get_current_user)):
    from app.db.models import User
    u = db.query(User).filter(User.id == user.id).first()
    if not u:
        return {"success": False, "message": "User not found"}
    if data.get("ctrader_account_id"): u.ctrader_account_id = data["ctrader_account_id"].strip()
    if data.get("ctrader_server"):     u.ctrader_server     = data["ctrader_server"].strip()
    db.commit()
    crud.upsert_setting(db, user.id, "data_source", "ctrader")
    return {
        "success": True,
        "message": "✅ Account saved! Now click 'Connect cTrader Account' to authorize via browser."
    }

@settings_router.get("/ctrader/auth")
def ctrader_auth(db: Session = Depends(get_db), user=Depends(get_current_user)):
    from app.db.models import User
    from app.config import settings
    u = db.query(User).filter(User.id == user.id).first()
    client_id = settings.CTRADER_CLIENT_ID
    if not client_id:
        return {"error": "CTRADER_CLIENT_ID not set in server .env"}
    if not u or not u.ctrader_account_id:
        return {"error": "Save your Account ID first"}
    is_prod      = "localhost" not in settings.FRONTEND_URL
    base         = "https://forex-ai-trading-backend.onrender.com" if is_prod else "http://localhost:8000"
    redirect_uri = f"{base}/settings/ctrader/callback"
    auth_url = (
        f"https://openapi.ctrader.com/oauth/v2/auth"
        f"?client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&response_type=code"
        f"&scope=trading"
        f"&state={user.id}"
    )
    return {"auth_url": auth_url, "redirect_uri": redirect_uri}


@settings_router.get("/ctrader/callback")
def ctrader_callback(code: str = None, state: str = None, error: str = None,
                     db: Session = Depends(get_db)):
    """Receive OAuth2 code from cTrader and exchange for token."""
    from app.db.models import User
    from app.config import settings
    from fastapi.responses import HTMLResponse
    import requests as req

    if error:
        return HTMLResponse(f"<h2>cTrader connection failed: {error}</h2>")
    if not code or not state:
        return HTMLResponse("<h2>Missing code or state</h2>")

    user_id = int(state)
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        return HTMLResponse("<h2>User not found</h2>")

    is_prod     = "localhost" not in settings.FRONTEND_URL
    base        = "https://forex-ai-trading-backend.onrender.com" if is_prod else "http://localhost:8000"
    redirect_uri = f"{base}/settings/ctrader/callback"

    try:
        from app.config import settings as app_settings
        r = req.post("https://openapi.ctrader.com/oauth/v2/token", data={
            "grant_type":    "authorization_code",
            "code":          code,
            "redirect_uri":  redirect_uri,
            "client_id":     app_settings.CTRADER_CLIENT_ID,
            "client_secret": app_settings.CTRADER_CLIENT_SECRET,
        }, timeout=15, verify=False)
        data = r.json()
        if "access_token" not in data:
            return HTMLResponse(f"<h2>Token exchange failed: {data}</h2>")
        u.ctrader_access_token  = data["access_token"]
        u.ctrader_refresh_token = data.get("refresh_token", "")
        db.commit()
        frontend = settings.FRONTEND_URL or "http://localhost:5173"
        return HTMLResponse(
            f"""<html><body>
            <h2>✅ cTrader Connected Successfully!</h2>
            <p>Account {u.ctrader_account_id} is now linked.</p>
            <script>setTimeout(()=>window.location='{frontend}/settings',2000)</script>
            <p>Redirecting back to your app...</p>
            </body></html>"""
        )
    except Exception as e:
        return HTMLResponse(f"<h2>Error: {e}</h2>")


@settings_router.get("/metaapi/status")
def get_metaapi_status(db: Session = Depends(get_db), user=Depends(get_current_user)):
    from app.db.models import User
    u = db.query(User).filter(User.id == user.id).first()
    if not u or not u.metaapi_token or not u.metaapi_account_id:
        return {"configured": False, "connected": False}
    return {
        "configured": True,
        "connected": crud.get_setting(db, user.id, "metaapi_connected") == "true",
        "account_id": u.metaapi_account_id,
        "region": crud.get_setting(db, user.id, "metaapi_region") or "new-york"
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
