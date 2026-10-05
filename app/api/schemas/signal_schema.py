from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class UserRegister(BaseModel):
    email:     str
    password:  str
    full_name: str

class UserLogin(BaseModel):
    email:    str
    password: str

class MT5Setup(BaseModel):
    mt5_login:    str
    mt5_password: str
    mt5_server:   str
    telegram_chat_id: Optional[str] = None

class SignalResponse(BaseModel):
    id:            Optional[int]
    symbol:        str
    direction:     str
    strategy:      Optional[str]
    entry_price:   Optional[float]
    stop_loss:     Optional[float]
    take_profit:   Optional[float]
    sl_pips:       Optional[float]
    tp_pips:       Optional[float]
    rr_ratio:      Optional[str]
    lot_size:      Optional[float]
    risk_usd:      Optional[float]
    ml_probability:Optional[float]
    h4_trend:      Optional[str]
    h1_trend:      Optional[str]
    session:       Optional[str]
    status:        Optional[str]
    confirmations: Optional[List[str]]
    created_at:    Optional[datetime]

    class Config:
        from_attributes = True

class TradeResponse(BaseModel):
    id:          int
    symbol:      str
    direction:   str
    entry_price: Optional[float]
    stop_loss:   Optional[float]
    take_profit: Optional[float]
    lot_size:    Optional[float]
    risk_usd:    Optional[float]
    open_time:   Optional[datetime]
    close_time:  Optional[datetime]
    close_price: Optional[float]
    profit_usd:  Optional[float]
    result:      Optional[str]
    actual_rr:   Optional[float]
    strategy:    Optional[str]

    class Config:
        from_attributes = True

class PerformanceResponse(BaseModel):
    total_trades:     int
    wins:             int
    losses:           int
    win_rate:         float
    net_pnl_usd:      float
    total_profit_usd: float
    total_loss_usd:   float

class SettingsUpdate(BaseModel):
    max_risk_usd:      Optional[float] = 1.0
    max_trades_day:    Optional[int]   = 2
    max_losses_day:    Optional[int]   = 2
    rr_ratio:          Optional[float] = 2.0
    london_session:    Optional[bool]  = True
    newyork_session:   Optional[bool]  = True
    telegram_alerts:   Optional[bool]  = True
    data_source:       Optional[str]   = "yfinance"
    gemini_api_key:    Optional[str]   = None
    telegram_chat_id:  Optional[str]   = None
    pairs:             Optional[List[str]] = None
