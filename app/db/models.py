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
    ctrader_client_id     = Column(String)
    ctrader_client_secret  = Column(String)
    ctrader_account_id     = Column(String)
    ctrader_server         = Column(String, default="demo.ctrader.com")
    ctrader_access_token   = Column(String)
    ctrader_refresh_token  = Column(String)
    is_active              = Column(Boolean, default=True)
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
    entry_price     = Column(Float)   # M3 trigger candle close — signal reference price
    stop_loss       = Column(Float)
    take_profit     = Column(Float)
    sl_pips         = Column(Float)
    tp_pips         = Column(Float)
    rr_ratio        = Column(String)
    lot_size        = Column(Float)
    risk_usd        = Column(Float, default=1.0)
    ml_probability  = Column(Float)   # normalized score (raw/14*100) for Variant C signals
    raw_score       = Column(Integer) # raw 0-14 score from calculate_a_plus_score()
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
    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False)
    signal_id        = Column(Integer, ForeignKey("signals.id"))
    symbol           = Column(String, nullable=False)
    direction        = Column(String)
    entry_price      = Column(Float)   # M3 signal reference price (strategy reference)
    actual_fill_price= Column(Float)   # actual broker execution price (nullable until confirmed)
    stop_loss        = Column(Float)
    take_profit      = Column(Float)
    lot_size         = Column(Float)
    risk_usd         = Column(Float, default=1.0)
    mt5_ticket       = Column(String)
    open_time        = Column(DateTime(timezone=True))
    close_time       = Column(DateTime(timezone=True))
    close_price      = Column(Float)
    profit_usd       = Column(Float)
    result           = Column(String)
    actual_rr        = Column(Float)
    strategy         = Column(String)
    user             = relationship("User", back_populates="trades")
    signal           = relationship("Signal", back_populates="trade")

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


# ── Dry-Run Forward-Test Tables ───────────────────────────────────────────────
# These tables are STRICTLY for paper-trade simulation.
# No broker execution function is ever called from code that writes here.
# Kept separate from Signal/Trade so a schema migration cannot accidentally
# route dry-run rows into live execution paths.

class DryRunSignal(Base):
    """
    One row per signal candidate observed by the dry-run scanner.
    Includes both accepted signals and rejected ones (with rejection_reason).
    """
    __tablename__ = "dry_run_signals"

    id                  = Column(Integer, primary_key=True, index=True)
    # Identity
    signal_id           = Column(String, nullable=False, unique=True, index=True)  # UUID
    strategy            = Column(String, nullable=False)   # "A_Classic" | "C_NoM3Sweep"
    pair                = Column(String, nullable=False, index=True)
    direction           = Column(String, nullable=False)   # "BUY" | "SELL"
    # Timing
    signal_ts           = Column(DateTime(timezone=True), nullable=False, index=True)  # UTC candle close
    candle_closed       = Column(Boolean, default=True)    # always True — we only act on closed candles
    # Prices at signal time
    entry_ref           = Column(Float)    # OB midpoint / M3 close
    stop_loss           = Column(Float)
    take_profit         = Column(Float)
    risk_pips           = Column(Float)
    planned_rr          = Column(Float)
    # Market context
    session             = Column(String)
    m15_bias            = Column(String)
    m5_confirmation     = Column(String)   # CHoCH/BOS type
    m3_trigger          = Column(String)   # M3 confirmation label
    ob_zone             = Column(String)   # "OB (low-high)" string
    fvg_overlap         = Column(Boolean, default=False)
    score               = Column(Integer, default=0)
    # Spread / cost info at signal time
    spread_pips_observed = Column(Float)   # actual bid-ask spread if available
    spread_pips_modelled = Column(Float)   # cost model estimate
    cost_source         = Column(String, default="modelled")  # "observed" | "modelled"
    # Status
    status              = Column(String, default="pending")  # pending|entered|missed|invalidated|rejected
    rejection_reason    = Column(String)   # populated when status="rejected"
    # Timestamps
    created_at          = Column(DateTime(timezone=True), server_default=func.now())

    paper_trade         = relationship("DryRunTrade", back_populates="signal", uselist=False)


class DryRunTrade(Base):
    """
    One row per paper-trade entry derived from a DryRunSignal.
    Outcome is determined from subsequent market candles only — never from
    candles that existed before the signal was recorded.
    """
    __tablename__ = "dry_run_trades"

    id                  = Column(Integer, primary_key=True, index=True)
    signal_id           = Column(String, ForeignKey("dry_run_signals.signal_id"),
                                 nullable=False, unique=True, index=True)
    strategy            = Column(String, nullable=False)
    pair                = Column(String, nullable=False, index=True)
    direction           = Column(String, nullable=False)
    # Entry
    entry_price         = Column(Float)    # actual fill price (bid/ask adjusted)
    entry_ts            = Column(DateTime(timezone=True), index=True)
    entry_candle_ts     = Column(DateTime(timezone=True))  # candle that triggered fill
    entry_type          = Column(String, default="limit")  # "limit" | "market"
    # Levels
    stop_loss           = Column(Float)
    take_profit         = Column(Float)
    risk_pips           = Column(Float)
    planned_rr          = Column(Float)
    # Exit
    exit_price          = Column(Float)
    exit_ts             = Column(DateTime(timezone=True))
    exit_candle_ts      = Column(DateTime(timezone=True))
    exit_reason         = Column(String)   # "TP"|"SL"|"ambiguous_SL_conservative"|"expired"
    result              = Column(String)   # "WIN"|"LOSS"|"OPEN"|"MISSED"|"EXPIRED"
    # R metrics
    r_gross             = Column(Float)    # realized R before costs
    r_net               = Column(Float)    # realized R after modelled costs
    cost_r              = Column(Float)    # r_gross - r_net
    cost_pips           = Column(Float)    # total modelled cost in pips
    # Spread info
    spread_pips_at_entry = Column(Float)   # observed spread at entry candle if available
    spread_source       = Column(String, default="modelled")  # "observed" | "modelled"
    # Flags
    ambiguous_candle    = Column(Boolean, default=False)  # True when SL+TP both in same candle
    missed_entry        = Column(Boolean, default=False)  # True when price never pulled back
    data_gap            = Column(Boolean, default=False)  # True when candle data was missing
    # Timestamps
    created_at          = Column(DateTime(timezone=True), server_default=func.now())
    updated_at          = Column(DateTime(timezone=True), onupdate=func.now())

    signal              = relationship("DryRunSignal", back_populates="paper_trade")
