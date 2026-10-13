from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from app.db.database import get_db
from app.db import crud
from app.signals.scanner import scan_all_pairs
from app.notifications.telegram_bot import send_signal_alert
from app.api.dependencies import get_current_user
from app.ml.predictor import reload_model

router = APIRouter(prefix="/signals", tags=["signals"])

@router.get("/scan")
def scan(db: Session = Depends(get_db), user=Depends(get_current_user)):
    try:
        rr          = float(crud.get_setting(db, user.id, "rr_ratio") or 2.0)
        risk        = float(crud.get_setting(db, user.id, "max_risk_usd") or 1.0)
        data_source = crud.get_setting(db, user.id, "data_source") or "yfinance"
        max_trades  = int(crud.get_setting(db, user.id, "max_trades_day") or 999)
        max_losses  = int(crud.get_setting(db, user.id, "max_losses_day") or 999)

        # Enforce daily trade & loss limits
        from datetime import date
        from app.db.models import Trade
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        todays_trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.open_time >= today_start
        ).all()
        todays_losses = [t for t in todays_trades if t.result == "LOSS"]
        if len(todays_trades) >= max_trades:
            return [{"symbol": "ALL", "direction": "NO TRADE",
                     "reason": [f"Daily trade limit reached ({max_trades} trades)"]}]
        if len(todays_losses) >= max_losses:
            return [{"symbol": "ALL", "direction": "NO TRADE",
                     "reason": [f"Daily loss limit reached ({max_losses} losses) — trading stopped for today"]}]

        # Auto-connect MT5 using saved credentials if source is mt5
        if data_source == "mt5" and all([user.mt5_login, user.mt5_password, user.mt5_server]):
            try:
                from app.mt5.connection import connect
                from app.db.crud import get_mt5_password
                connect(login=int(user.mt5_login), password=get_mt5_password(user), server=user.mt5_server)
            except Exception:
                pass

        results = scan_all_pairs(risk_usd=risk, rr_ratio=rr, data_source=data_source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

    # Deduplicate: skip saving if this pair already has a pending/approved signal
    from app.db.models import Signal as SignalModel
    pending_symbols = {
        s.symbol for s in db.query(SignalModel).filter(
            SignalModel.user_id == user.id,
            SignalModel.status.in_(["pending", "approved"])
        ).all()
    }

    for r in results:
        if r.get("direction") in ("BUY", "SELL"):
            if r["symbol"] in pending_symbols:
                r["reason"] = ["Signal already active for this pair — skipped duplicate"]
                continue
            try:
                signal_data = {
                    "user_id":          user.id,
                    "symbol":           r["symbol"],
                    "direction":        r["direction"],
                    "strategy":         r.get("strategy"),
                    "entry_price":      r.get("entry_price"),
                    "stop_loss":        r.get("stop_loss"),
                    "take_profit":      r.get("take_profit"),
                    "sl_pips":          r.get("sl_pips"),
                    "tp_pips":          r.get("tp_pips"),
                    "rr_ratio":         r.get("rr_ratio"),
                    "lot_size":         r.get("lot_size"),
                    "risk_usd":         r.get("risk_usd", 1.0),
                    "h4_trend":         r.get("h4_trend"),
                    "h1_trend":         r.get("h1_trend"),
                    "session":          r.get("session"),
                    "spread_at_signal": r.get("spread"),
                    "status":           "pending",
                }
                saved_signal = crud.create_signal(db, signal_data)
                if user.telegram_chat_id:
                    r["id"] = saved_signal.id
                    send_signal_alert(user.telegram_chat_id, r)
            except Exception:
                pass

    return results

@router.post("/retrain")
def retrain_model(db: Session = Depends(get_db), user=Depends(get_current_user)):
    """Trigger ML retraining using user's saved MT5 credentials."""
    try:
        import threading
        from app.ml.trainer import train

        login    = user.mt5_login
        password = user.mt5_password
        server   = user.mt5_server
        source   = "mt5" if all([login, password, server]) else "frankfurter"

        def _run():
            if source == "mt5":
                try:
                    from app.mt5.connection import connect
                    from app.db.crud import get_mt5_password
                    connect(login=int(login), password=get_mt5_password(user), server=server)
                except Exception:
                    pass
            train(source=source)
            reload_model()

        threading.Thread(target=_run, daemon=True).start()
        return {"status": f"Retraining started using {source}. Check backend logs."}
    except Exception as e:
        raise HTTPException(500, f"Retrain failed: {str(e)}")

@router.get("/history")
def history(db: Session = Depends(get_db), user=Depends(get_current_user)):
    return crud.get_signals_by_user(db, user.id)

@router.post("/{signal_id}/approve")
def approve(signal_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    from app.signals.trade_monitor import activate_trade_from_signal
    res = activate_trade_from_signal(db, user.id, signal_id)
    if not res.get("success"):
        raise HTTPException(400, res.get("message", "Could not activate trade"))
    signal = crud.update_signal_status(db, signal_id, "approved")
    return {"status": "approved", "trade_id": res.get("trade_id"), "signal": signal}

@router.post("/{signal_id}/reject")
def reject(signal_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    signal = crud.update_signal_status(db, signal_id, "rejected")
    return signal
