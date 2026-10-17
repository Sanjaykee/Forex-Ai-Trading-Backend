from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from app.db.database import get_db
from app.db import crud
from app.signals.scanner import scan_all_pairs, scan_pair
from app.api.dependencies import get_current_user

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("/scan")
def scan(db: Session = Depends(get_db), user=Depends(get_current_user)):
    """
    Scan all 9 pairs on M15 + M3 for live SMC signals.
    Returns signals sorted by confidence score descending.
    """
    try:
        results = scan_all_pairs(timeframes=("M15", "M3"), min_confidence="LOW")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

    # Save new signals to DB and send Telegram alerts
    from app.db.models import Signal as SignalModel
    pending_symbols = {
        s.symbol for s in db.query(SignalModel).filter(
            SignalModel.user_id == user.id,
            SignalModel.status.in_(["pending", "approved"])
        ).all()
    }

    for r in results:
        key = f"{r['symbol']}_{r['timeframe']}"
        if key in pending_symbols:
            continue
        try:
            signal_data = {
                "user_id":          user.id,
                "symbol":           r["symbol"],
                "direction":        r["direction"],
                "strategy":         r.get("pattern_name"),
                "entry_price":      r.get("entry"),
                "stop_loss":        r.get("sl"),
                "take_profit":      r.get("tp_1_3"),   # default 1:3
                "sl_pips":          r.get("sl_pips"),
                "tp_pips":          None,
                "rr_ratio":         "1:3",
                "lot_size":         None,
                "risk_usd":         1.0,
                "h4_trend":         r.get("htf_bias"),
                "h1_trend":         None,
                "session":          r.get("session"),
                "spread_at_signal": None,
                "status":           "pending",
            }
            saved = crud.create_signal(db, signal_data)
            # Telegram alert
            if getattr(user, "telegram_chat_id", None):
                try:
                    from app.notifications.telegram_bot import send_signal_alert
                    r["id"] = saved.id
                    send_signal_alert(user.telegram_chat_id, r)
                except Exception:
                    pass
        except Exception:
            pass

    return results


@router.get("/scan/{symbol}")
def scan_single(symbol: str, tf: str = "M15",
                db: Session = Depends(get_db),
                user=Depends(get_current_user)):
    """Scan a single pair on a specific timeframe."""
    try:
        result = scan_pair(symbol.upper(), tf.upper())
        if result is None:
            return {"symbol": symbol, "timeframe": tf,
                    "direction": "NO TRADE",
                    "reason": "No pattern detected or outside killzone"}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
def history(db: Session = Depends(get_db), user=Depends(get_current_user)):
    return crud.get_signals_by_user(db, user.id)


@router.post("/{signal_id}/approve")
def approve(signal_id: int, db: Session = Depends(get_db),
            user=Depends(get_current_user)):
    from app.signals.trade_monitor import activate_trade_from_signal
    res = activate_trade_from_signal(db, user.id, signal_id)
    if not res.get("success"):
        raise HTTPException(400, res.get("message", "Could not activate trade"))
    signal = crud.update_signal_status(db, signal_id, "approved")
    return {"status": "approved", "trade_id": res.get("trade_id"), "signal": signal}


@router.post("/{signal_id}/reject")
def reject(signal_id: int, db: Session = Depends(get_db),
           user=Depends(get_current_user)):
    signal = crud.update_signal_status(db, signal_id, "rejected")
    return signal
