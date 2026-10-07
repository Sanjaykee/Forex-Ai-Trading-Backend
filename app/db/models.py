from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base

class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    email           = Column(String, unique=True, index=True, nullable=False)
    password_hash   = Column(String, nullable=False)
    full_name       = Column(String)
    telegram_chat_id= Column(String)
    mt5_login           = Column(String)
    mt5_password        = Column(String)
    mt5_server          = Column(String)
    metaapi_token       = Column(String)
    metaapi_account_id  = Column(String)
    is_active           = Column(Boolean, default=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    subscriptions   = relationship("Subscription", back_populates="user")
    signals         = relationship("Signal", back_populates="user")
    trades          = relationship("Trade", back_populates="user")

class Subscription(Base):
    __tablename__ = "subscriptions"
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    plan        = Column(String, default="free")
    status      = Column(String, default="active")
    start_date  = Column(DateTime(timezone=True), server_default=func.now())
    end_date    = Column(DateTime(timezone=True))
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    user        = relationship("User", back_populates="subscriptions")
    payments    = relationship("Payment", back_populates="subscription")

class Payment(Base):
    __tablename__ = "payments"
    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id"))
    amount_usd      = Column(Float, nullable=False)
    payment_method  = Column(String)
    transaction_id  = Column(String)
    status          = Column(String, default="pending")
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    subscription    = relationship("Subscription", back_populates="payments")

class Pair(Base):
    __tablename__ = "pairs"
    id          = Column(Integer, primary_key=True, index=True)
    symbol      = Column(String, unique=True, nullable=False)
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())

class Candle(Base):
    __tablename__ = "candles"
    id          = Column(Integer, primary_key=True, index=True)
    symbol      = Column(String, nullable=False, index=True)
    timeframe   = Column(String, nullable=False)
    open        = Column(Float)
    high        = Column(Float)
    low         = Column(Float)
    close       = Column(Float)
    volume      = Column(Float)
    time        = Column(DateTime(timezone=True), index=True)

class Signal(Base):
    __tablename__ = "signals"
    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    symbol          = Column(String, nullable=False)
    direction       = Column(String, nullable=False)
    strategy        = Column(String)
    entry_price     = Column(Float)
    stop_loss       = Column(Float)
    take_profit     = Column(Float)
    sl_pips         = Column(Float)
    tp_pips         = Column(Float)
    rr_ratio        = Column(String)
    lot_size        = Column(Float)
    risk_usd        = Column(Float, default=1.0)
    ml_probability  = Column(Float)
    h4_trend        = Column(String)
    h1_trend        = Column(String)
    session         = Column(String)
    spread_at_signal= Column(Float)
    status          = Column(String, default="pending")
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    user            = relationship("User", back_populates="signals")
    trade           = relationship("Trade", back_populates="signal", uselist=False)

class Trade(Base):
    __tablename__ = "trades"
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    signal_id   = Column(Integer, ForeignKey("signals.id"))
    symbol      = Column(String, nullable=False)
    direction   = Column(String)
    entry_price = Column(Float)
    stop_loss   = Column(Float)
    take_profit = Column(Float)
    lot_size    = Column(Float)
    risk_usd    = Column(Float, default=1.0)
    mt5_ticket  = Column(String)
    open_time   = Column(DateTime(timezone=True))
    close_time  = Column(DateTime(timezone=True))
    close_price = Column(Float)
    profit_usd  = Column(Float)
    result      = Column(String)
    actual_rr   = Column(Float)
    strategy    = Column(String)
    user        = relationship("User", back_populates="trades")
    signal      = relationship("Signal", back_populates="trade")

class Performance(Base):
    __tablename__ = "performance"
    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    date            = Column(DateTime(timezone=True), server_default=func.now())
    total_trades    = Column(Integer, default=0)
    wins            = Column(Integer, default=0)
    losses          = Column(Integer, default=0)
    win_rate        = Column(Float, default=0.0)
    total_profit_usd= Column(Float, default=0.0)
    total_loss_usd  = Column(Float, default=0.0)
    net_pnl_usd     = Column(Float, default=0.0)
    profit_factor   = Column(Float, default=0.0)
    max_drawdown_usd= Column(Float, default=0.0)
    best_rr_achieved= Column(Float, default=0.0)

class Settings(Base):
    __tablename__ = "settings"
    id      = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key     = Column(String, nullable=False)
    value   = Column(String)
