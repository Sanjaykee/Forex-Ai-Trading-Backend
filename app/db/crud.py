from sqlalchemy.orm import Session
from app.db import models
import bcrypt
from passlib.context import CryptContext
from datetime import datetime

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    try:
        pwd_bytes = password[:72].encode("utf-8")
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")
    except Exception:
        return pwd_context.hash(password[:72])

# --- Users ---
def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_user_by_id(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def create_user(db: Session, email: str, password: str, full_name: str):
    user = models.User(
        email=email,
        password_hash=hash_password(password),
        full_name=full_name
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    sub = models.Subscription(user_id=user.id, plan="free", status="active")
    db.add(sub)
    db.commit()
    return user

def verify_password(plain: str, hashed: str) -> bool:
    try:
        plain_bytes = plain[:72].encode("utf-8")
        hashed_bytes = hashed.encode("utf-8")
        return bcrypt.checkpw(plain_bytes, hashed_bytes)
    except Exception:
        try:
            return pwd_context.verify(plain[:72], hashed)
        except Exception:
            return False

def update_mt5_credentials(db: Session, user_id: int, login: str, password: str, server: str):
    user = get_user_by_id(db, user_id)
    user.mt5_login = login
    user.mt5_password = password
    user.mt5_server = server
    db.commit()
    return user

# --- Signals ---
def create_signal(db: Session, signal_data: dict):
    signal = models.Signal(**signal_data)
    db.add(signal)
    db.commit()
    db.refresh(signal)
    return signal

def get_signals_by_user(db: Session, user_id: int, limit: int = 20):
    return db.query(models.Signal).filter(
        models.Signal.user_id == user_id
    ).order_by(models.Signal.created_at.desc()).limit(limit).all()

def update_signal_status(db: Session, signal_id: int, status: str):
    signal = db.query(models.Signal).filter(models.Signal.id == signal_id).first()
    signal.status = status
    db.commit()
    return signal

# --- Trades ---
def create_trade(db: Session, trade_data: dict):
    trade = models.Trade(**trade_data)
    db.add(trade)
    db.commit()
    db.refresh(trade)
    return trade

def get_trades_by_user(db: Session, user_id: int, limit: int = 50):
    return db.query(models.Trade).filter(
        models.Trade.user_id == user_id
    ).order_by(models.Trade.open_time.desc()).limit(limit).all()

def close_trade(db: Session, trade_id: int, close_price: float, profit_usd: float, result: str, actual_rr: float):
    trade = db.query(models.Trade).filter(models.Trade.id == trade_id).first()
    trade.close_price = close_price
    trade.close_time = datetime.utcnow()
    trade.profit_usd = profit_usd
    trade.result = result
    trade.actual_rr = actual_rr
    db.commit()
    return trade

# --- Performance ---
def get_performance_by_user(db: Session, user_id: int):
    trades = db.query(models.Trade).filter(models.Trade.user_id == user_id).all()
    if not trades:
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0, "net_pnl_usd": 0}
    wins   = [t for t in trades if t.result == "WIN"]
    losses = [t for t in trades if t.result == "LOSS"]
    net    = sum(t.profit_usd for t in trades if t.profit_usd)
    return {
        "total_trades":     len(trades),
        "wins":             len(wins),
        "losses":           len(losses),
        "win_rate":         round(len(wins) / len(trades) * 100, 1),
        "net_pnl_usd":      round(net, 2),
        "total_profit_usd": round(sum(t.profit_usd for t in wins if t.profit_usd), 2),
        "total_loss_usd":   round(sum(t.profit_usd for t in losses if t.profit_usd), 2),
    }

# --- Settings ---
def get_setting(db: Session, user_id: int, key: str):
    s = db.query(models.Settings).filter(
        models.Settings.user_id == user_id,
        models.Settings.key == key
    ).first()
    return s.value if s else None

def upsert_setting(db: Session, user_id: int, key: str, value: str):
    s = db.query(models.Settings).filter(
        models.Settings.user_id == user_id,
        models.Settings.key == key
    ).first()
    if s:
        s.value = value
    else:
        s = models.Settings(user_id=user_id, key=key, value=value)
        db.add(s)
    db.commit()
    return s
