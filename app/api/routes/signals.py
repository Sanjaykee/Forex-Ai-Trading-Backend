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
    Scan all 9 pairs on M15 + M5 for live SMC signals.
    Returns signals sorted by confidence score descending.
    """
    try:
        results = scan_all_pairs(timeframes=("M15", "M5"), min_confidence="LOW")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

    # Load user's actual risk settings
    risk_usd = float(crud.get_setting(db, user.id, "max_risk_usd") or 1.0)
    rr_ratio = float(crud.get_setting(db, user.id, "rr_ratio") or 3.0)
    rr_key   = f"tp_1_{int(rr_ratio)}" if int(rr_ratio) in (2, 3, 5) else "tp_1_3"

    from app.risk.position_sizer import calculate_lot_size
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
            entry = float(r.get("entry_price") or r.get("entry") or 0)
            sl    = float(r.get("stop_loss")   or r.get("sl")    or 0)
            tp    = float(r.get(rr_key) or r.get("tp_1_3") or 0)
            lot   = calculate_lot_size(
                risk_usd=risk_usd,
                entry=entry,
                stop_loss=sl,
                symbol_info={"volume_min": 0.01, "volume_step": 0.01},
                symbol=r["symbol"]
            ) if entry and sl else 0.01

            signal_data = {
                "user_id":          user.id,
                "symbol":           r["symbol"],
                "direction":        r["direction"],
                "strategy":         r.get("pattern_name"),
                "entry_price":      entry,
                "stop_loss":        sl,
                "take_profit":      tp,
                "sl_pips":          r.get("sl_pips"),
                "tp_pips":          None,
                "rr_ratio":         f"1:{int(rr_ratio)}",
                "lot_size":         lot,
                "risk_usd":         risk_usd,
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
                    alert_payload = dict(r)
                    alert_payload["id"]       = saved.id
                    alert_payload["risk_usd"] = risk_usd
                    alert_payload["rr_ratio"] = f"1:{int(rr_ratio)}"
                    alert_payload["lot_size"] = lot
                    alert_payload["take_profit"] = tp
                    send_signal_alert(user.telegram_chat_id, alert_payload)
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


@router.get("/backtest-quick")
def backtest_quick(days: int = 15, tf: str = "M3", db: Session = Depends(get_db), user=Depends(get_current_user)):
    """
    Run backtest across all 9 pairs. Default: 15 days on M15.
    Params: days (default 15), tf (default M15, options: M5/M15/both)
    """
    from app.backtest.engine import run_backtest

    PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
             "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"]

    timeframes = ("M15", "M5") if tf.upper() == "BOTH" else (tf.upper(),)

    all_trades = []
    summary = {"total": 0, "wins": 0, "losses": 0, "pairs_scanned": 0, "errors": []}

    for symbol in PAIRS:
        for timeframe in timeframes:
            try:
                result = run_backtest(symbol=symbol, timeframe=timeframe, days=days, rr_ratio=3.0, min_win_prob=50)
                if result.get("total_signals", 0) > 0:
                    for t in result.get("all_trades", []):
                        t["symbol"] = symbol
                        t["timeframe"] = timeframe
                        all_trades.append(t)
                    summary["wins"]   += result.get("wins", 0)
                    summary["losses"] += result.get("losses", 0)
                summary["pairs_scanned"] += 1
            except Exception as e:
                summary["errors"].append(f"{symbol} {timeframe}: {str(e)}")

    summary["total"] = summary["wins"] + summary["losses"]
    summary["win_rate"] = round(summary["wins"] / summary["total"] * 100, 1) if summary["total"] > 0 else 0.0

    all_trades.sort(key=lambda x: x.get("score", 0), reverse=True)

    return {
        "days": days,
        "timeframe": tf.upper(),
        "summary": summary,
        "trades": all_trades,
    }


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
