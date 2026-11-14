"""
Phase 5.9.3 — Replacement 3: C1 (IntegrityError) + C2 (Live-Price) Verification
==================================================================================
All broker calls and DB interactions are mocked.
No real orders are placed. No real DB records are modified.

C1 tests (concurrent duplicate protection):
  C1-01  IntegrityError during PENDING_APPROVAL commit -> rollback called
  C1-02  IntegrityError -> existing trade re-fetched, returned as success
  C1-03  IntegrityError -> place_order NOT called
  C1-04  IntegrityError -> no existing trade found after rollback -> safe failure
  C1-05  Normal successful commit -> place_order called exactly once
  C1-06  Concurrent duplicate: second request cannot submit a broker order
  C1-07  IntegrityError -> signal status restored to original (not left approved)

C2 tests (live-price risk validation):
  C2-01  BUY uses Ask as executable entry
  C2-02  SELL uses Bid as executable entry
  C2-03  Empty live-price response -> rejected
  C2-04  Missing ask -> rejected
  C2-05  Missing bid -> rejected
  C2-06  Zero price (ask or bid <= 0) -> rejected
  C2-07  Negative price -> rejected
  C2-08  NaN price -> rejected
  C2-09  Infinite price -> rejected
  C2-10  Non-numeric price -> rejected
  C2-11  Live price crosses structural SL (BUY: ask <= sl) -> rejected
  C2-12  Live price crosses structural SL (SELL: bid >= sl) -> rejected
  C2-13  Live-price movement makes stored lot unsafe (risk > $1.00) -> rejected
  C2-13b Live-price movement makes stored lot inconsistent (lot mismatch) -> rejected
  C2-14  Actual risk exactly $1.00 -> accepted
  C2-15  Actual risk above $1.00 (tiny excess) -> rejected
  C2-16  Original structural entry, SL, TP, lot unchanged after rejection
  C2-17  Original structural entry, SL, TP, lot unchanged after success; Trade row entry is structural
  C2-18  No lot increase or SL adjustment to force trades

Run:
    python backend/tests/phase593_verify.py
"""

import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch, call
from sqlalchemy.exc import IntegrityError as SAIntegrityError

from app.signals.trade_monitor import activate_trade_from_signal, get_live_price

PASS = []
FAIL = []

def chk(name: str, cond: bool, detail: str = ""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


# ── Shared broker specs (valid GBPUSD) ────────────────────────────────────────
_SPECS = {
    "volumeMin": 0.01, "volumeMax": 500.0, "volumeStep": 0.01,
    "volume_min": 0.01, "volume_max": 500.0, "volume_step": 0.01,
    "min_stop_distance": 0.0001, "point": 0.00001, "digits": 5,
}

# Standard live prices for GBPUSD:
# Pip size: 0.0001, pip value per lot: $10.00.
# Lot 0.01 * 10 pips * $10/pip = $1.0000.
# BUY: executable entry = Ask. With SL=1.27410, Ask=1.27510 gives sl_dist=0.00100 (10 pips) -> risk = $1.00.
# SELL: executable entry = Bid. With SL=1.27600, Bid=1.27500 gives sl_dist=0.00100 (10 pips) -> risk = $1.00.
_LIVE_BUY_EXACT = {"ask": 1.27510, "bid": 1.27508}
_LIVE_SELL_EXACT = {"ask": 1.27502, "bid": 1.27500}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _signal(
    id=1, user_id=10, symbol="GBPUSD", direction="BUY",
    entry_price=1.27500, stop_loss=1.27410, take_profit=1.28000,
    lot_size=0.01, risk_usd=1.0, rr_ratio="1:3", status="pending",
    strategy="smc_mtf",
):
    s = MagicMock()
    s.id          = id
    s.user_id     = user_id
    s.symbol      = symbol
    s.direction   = direction
    s.entry_price = entry_price
    s.stop_loss   = stop_loss
    s.take_profit = take_profit
    s.lot_size    = lot_size
    s.risk_usd    = risk_usd
    s.rr_ratio    = rr_ratio
    s.status      = status
    s.strategy    = strategy
    return s


def _user(id=10, token="tok", account_id="acct123", telegram_chat_id=None):
    u = MagicMock()
    u.id                 = id
    u.metaapi_token      = token
    u.metaapi_account_id = account_id
    u.telegram_chat_id   = telegram_chat_id
    return u


def _make_db(signal=None, existing_trade=None, user=None):
    db = MagicMock()

    def _query_side_effect(model):
        from app.db import models as m
        q = MagicMock()
        if model is m.Signal:
            q.filter.return_value.first.return_value = signal
        elif model is m.Trade:
            inner = MagicMock()
            inner.first.return_value = existing_trade
            inner.with_for_update.return_value = inner
            q.filter.return_value = inner
        elif model is m.User:
            q.filter.return_value.first.return_value = user
        else:
            q.filter.return_value.first.return_value = None
        return q

    db.query.side_effect = _query_side_effect
    db.add      = MagicMock()
    db.commit   = MagicMock()
    db.refresh  = MagicMock()
    db.rollback = MagicMock()
    return db


def _make_integrity_error():
    orig = MagicMock()
    orig.args = ("duplicate key value violates unique constraint",)
    return SAIntegrityError("INSERT", {}, orig)


# ─────────────────────────────────────────────────────────────────────────────
# C1 tests (concurrent duplicate protection)
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== C1-01: IntegrityError during commit -> rollback called ===")
sig1 = _signal(lot_size=0.01, risk_usd=1.0, status="pending",
               entry_price=1.27500, stop_loss=1.27410)
db1 = _make_db(signal=sig1, existing_trade=None, user=_user())
db1.commit.side_effect = _make_integrity_error()
existing_after1 = MagicMock(); existing_after1.id = 88

def _query_after_rollback1(model):
    from app.db import models as m
    q = MagicMock()
    if model is m.Signal:
        q.filter.return_value.first.return_value = sig1
    elif model is m.Trade:
        inner = MagicMock()
        inner.first.side_effect = [None, existing_after1]
        inner.with_for_update.return_value = inner
        q.filter.return_value = inner
    elif model is m.User:
        q.filter.return_value.first.return_value = _user()
    return q

db1.query.side_effect = _query_after_rollback1

with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.get_live_price", return_value=_LIVE_BUY_EXACT), \
     patch("app.mt5.order_manager.place_order") as mock_place1:
    res1 = activate_trade_from_signal(db1, user_id=10, signal_id=1)

chk("C1-01 rollback called", db1.rollback.called)
chk("C1-01 place_order not called", not mock_place1.called)


print("\n=== C1-02: IntegrityError -> existing trade re-fetched, returned ===")
sig2 = _signal(lot_size=0.01, risk_usd=1.0, status="pending",
               entry_price=1.27500, stop_loss=1.27410)
existing_after2 = MagicMock(); existing_after2.id = 99
db2 = _make_db(signal=sig2, existing_trade=None, user=_user())
db2.rollback = MagicMock()

call_counts2 = {"n": 0}
def _q2(model):
    from app.db import models as m
    q = MagicMock()
    if model is m.Signal:
        q.filter.return_value.first.return_value = sig2
    elif model is m.Trade:
        inner = MagicMock()
        call_counts2["n"] += 1
        inner.first.return_value = None if call_counts2["n"] == 1 else existing_after2
        inner.with_for_update.return_value = inner
        q.filter.return_value = inner
    elif model is m.User:
        q.filter.return_value.first.return_value = _user()
    return q

db2.query.side_effect = _q2
db2.commit.side_effect = _make_integrity_error()

with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.get_live_price", return_value=_LIVE_BUY_EXACT), \
     patch("app.mt5.order_manager.place_order") as mock_place2:
    res2 = activate_trade_from_signal(db2, user_id=10, signal_id=1)

chk("C1-02 success=True (idempotent)", res2["success"], str(res2))
chk("C1-02 trade_id is existing trade", res2.get("trade_id") == 99,
    f"got {res2.get('trade_id')}")
chk("C1-02 place_order not called", not mock_place2.called)


print("\n=== C1-03: IntegrityError -> place_order NOT called ===")
sig3 = _signal(lot_size=0.01, risk_usd=1.0, status="pending",
               entry_price=1.27500, stop_loss=1.27410)
db3 = _make_db(signal=sig3, existing_trade=None, user=_user())
db3.rollback = MagicMock()
existing3 = MagicMock(); existing3.id = 77

cnt3 = {"n": 0}
def _q3(model):
    from app.db import models as m
    q = MagicMock()
    if model is m.Signal:
        q.filter.return_value.first.return_value = sig3
    elif model is m.Trade:
        inner = MagicMock()
        cnt3["n"] += 1
        inner.first.return_value = None if cnt3["n"] == 1 else existing3
        inner.with_for_update.return_value = inner
        q.filter.return_value = inner
    elif model is m.User:
        q.filter.return_value.first.return_value = _user()
    return q

db3.query.side_effect = _q3
db3.commit.side_effect = _make_integrity_error()

with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.get_live_price", return_value=_LIVE_BUY_EXACT), \
     patch("app.mt5.order_manager.place_order") as mock_place3:
    activate_trade_from_signal(db3, user_id=10, signal_id=1)

chk("C1-03 place_order never called on IntegrityError", not mock_place3.called,
    f"called {mock_place3.call_count} times")


print("\n=== C1-04: IntegrityError -> no existing trade found -> safe failure ===")
sig4 = _signal(lot_size=0.01, risk_usd=1.0, status="pending",
               entry_price=1.27500, stop_loss=1.27410)
db4 = _make_db(signal=sig4, existing_trade=None, user=_user())
db4.rollback = MagicMock()
db4.commit.side_effect = _make_integrity_error()

with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.get_live_price", return_value=_LIVE_BUY_EXACT), \
     patch("app.mt5.order_manager.place_order") as mock_place4:
    res4 = activate_trade_from_signal(db4, user_id=10, signal_id=1)

chk("C1-04 success=False", not res4["success"], str(res4))
chk("C1-04 rollback called", db4.rollback.called)
chk("C1-04 place_order not called", not mock_place4.called)
chk("C1-04 message mentions constraint or duplicate",
    any(w in res4.get("message", "").lower()
        for w in ("constraint", "duplicate", "manual")),
    res4.get("message"))


print("\n=== C1-05: Normal successful commit -> place_order called exactly once ===")
sig5 = _signal(lot_size=0.01, risk_usd=1.0, status="pending",
               entry_price=1.27500, stop_loss=1.27410)
db5 = _make_db(signal=sig5, existing_trade=None, user=_user())

trade_obj5 = MagicMock()
trade_obj5.id = 55
trade_obj5.result = "PENDING_APPROVAL"
trade_obj5.actual_fill_price = None
trade_obj5.mt5_ticket = None
trade_obj5.symbol = "GBPUSD"
trade_obj5.direction = "BUY"
trade_obj5.broker_position_id = None

with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.models.Trade", return_value=trade_obj5), \
     patch("app.signals.trade_monitor.get_live_price", return_value=_LIVE_BUY_EXACT), \
     patch("app.mt5.order_manager.place_order") as mock_place5:
    mock_place5.return_value = {
        "success": True, "ticket": "MT5-555",
        "order_type": "ORDER_TYPE_BUY", "data": {"openPrice": 1.27510},
    }
    res5 = activate_trade_from_signal(db5, user_id=10, signal_id=1)

chk("C1-05 success=True", res5["success"], str(res5))
chk("C1-05 place_order called exactly once", mock_place5.call_count == 1,
    f"called {mock_place5.call_count} times")
chk("C1-05 rollback NOT called", not db5.rollback.called)


print("\n=== C1-06: Concurrent duplicate: second request cannot submit broker order ===")
place_call_count = {"n": 0}

def _counting_place(**kwargs):
    place_call_count["n"] += 1
    return {"success": True, "ticket": "MT5-R1",
            "order_type": "ORDER_TYPE_BUY", "data": {"openPrice": 1.27510}}

sig_a = _signal(lot_size=0.01, risk_usd=1.0, status="pending",
                entry_price=1.27500, stop_loss=1.27410)
sig_b = _signal(lot_size=0.01, risk_usd=1.0, status="pending",
                entry_price=1.27500, stop_loss=1.27410)

trade_a = MagicMock(); trade_a.id = 10; trade_a.result = "PENDING_APPROVAL"
trade_a.actual_fill_price = None; trade_a.mt5_ticket = None
trade_a.symbol = "GBPUSD"; trade_a.direction = "BUY"; trade_a.broker_position_id = None

db_a = _make_db(signal=sig_a, existing_trade=None, user=_user())
db_b = _make_db(signal=sig_b, existing_trade=None, user=_user())
db_b.rollback = MagicMock()
db_b.commit.side_effect = _make_integrity_error()
existing_b = MagicMock(); existing_b.id = 10

cnt_b = {"n": 0}
def _q_b(model):
    from app.db import models as m
    q = MagicMock()
    if model is m.Signal:
        q.filter.return_value.first.return_value = sig_b
    elif model is m.Trade:
        inner = MagicMock()
        cnt_b["n"] += 1
        inner.first.return_value = None if cnt_b["n"] == 1 else existing_b
        inner.with_for_update.return_value = inner
        q.filter.return_value = inner
    elif model is m.User:
        q.filter.return_value.first.return_value = _user()
    return q
db_b.query.side_effect = _q_b

with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.models.Trade", return_value=trade_a), \
     patch("app.signals.trade_monitor.get_live_price", return_value=_LIVE_BUY_EXACT), \
     patch("app.mt5.order_manager.place_order", side_effect=_counting_place):
    res_a = activate_trade_from_signal(db_a, user_id=10, signal_id=1)
    res_b = activate_trade_from_signal(db_b, user_id=10, signal_id=1)

chk("C1-06 first request succeeds", res_a["success"], str(res_a))
chk("C1-06 second request returns success (idempotent)", res_b["success"], str(res_b))
chk("C1-06 place_order called exactly once total", place_call_count["n"] == 1,
    f"called {place_call_count['n']} times")


print("\n=== C1-07: IntegrityError -> signal status restored to original ===")
sig7 = _signal(lot_size=0.01, risk_usd=1.0, status="pending",
               entry_price=1.27500, stop_loss=1.27410)
db7 = _make_db(signal=sig7, existing_trade=None, user=_user())
db7.rollback = MagicMock()
db7.commit.side_effect = _make_integrity_error()

with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.get_live_price", return_value=_LIVE_BUY_EXACT), \
     patch("app.mt5.order_manager.place_order"):
    res7 = activate_trade_from_signal(db7, user_id=10, signal_id=1)

chk("C1-07 signal.status remains 'pending' on rollback", sig7.status == "pending",
    f"got status {sig7.status}")


# ─────────────────────────────────────────────────────────────────────────────
# C2 tests (live-price risk validation)
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== C2-01: BUY uses Ask as executable entry ===")
sig_buy = _signal(direction="BUY", entry_price=1.27500, stop_loss=1.27410, lot_size=0.01)
db_buy = _make_db(signal=sig_buy, existing_trade=None, user=_user())

with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.get_live_price", return_value={"ask": 1.27510, "bid": 1.27490}), \
     patch("app.signals.trade_monitor.validate_trade_risk") as mock_vr_buy, \
     patch("app.mt5.order_manager.place_order") as mock_place_buy:
    mock_vr_res = MagicMock()
    mock_vr_res.is_valid = True
    mock_vr_res.lot_size = 0.01
    mock_vr_res.actual_risk_usd = 1.00
    mock_vr_res.pip_size = 0.0001
    mock_vr_res.pip_value_usd_per_lot = 10.0
    mock_vr_buy.return_value = mock_vr_res
    mock_place_buy.return_value = {
        "success": True, "ticket": "MT5-B1", "order_type": "ORDER_TYPE_BUY",
        "data": {"openPrice": 1.27510}
    }

    res_buy = activate_trade_from_signal(db_buy, user_id=10, signal_id=1)
    chk("C2-01 validate_trade_risk called with entry=live_ask (1.27510)",
        mock_vr_buy.call_args[1].get("entry") == 1.27510,
        f"entry was {mock_vr_buy.call_args[1].get('entry')}")
    chk("C2-01 validate_trade_risk NOT called with live_bid (1.27490)",
        mock_vr_buy.call_args[1].get("entry") != 1.27490)
    chk("C2-01 validate_trade_risk NOT called with structural entry (1.27500)",
        mock_vr_buy.call_args[1].get("entry") != 1.27500)


print("\n=== C2-02: SELL uses Bid as executable entry ===")
sig_sell = _signal(direction="SELL", entry_price=1.27480, stop_loss=1.27600, lot_size=0.01)
db_sell = _make_db(signal=sig_sell, existing_trade=None, user=_user())

with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.get_live_price", return_value={"ask": 1.27520, "bid": 1.27500}), \
     patch("app.signals.trade_monitor.validate_trade_risk") as mock_vr_sell, \
     patch("app.mt5.order_manager.place_order") as mock_place_sell:
    mock_vr_res = MagicMock()
    mock_vr_res.is_valid = True
    mock_vr_res.lot_size = 0.01
    mock_vr_res.actual_risk_usd = 1.00
    mock_vr_res.pip_size = 0.0001
    mock_vr_res.pip_value_usd_per_lot = 10.0
    mock_vr_sell.return_value = mock_vr_res
    mock_place_sell.return_value = {
        "success": True, "ticket": "MT5-S1", "order_type": "ORDER_TYPE_SELL",
        "data": {"openPrice": 1.27500}
    }

    res_sell = activate_trade_from_signal(db_sell, user_id=10, signal_id=1)
    chk("C2-02 validate_trade_risk called with entry=live_bid (1.27500)",
        mock_vr_sell.call_args[1].get("entry") == 1.27500,
        f"entry was {mock_vr_sell.call_args[1].get('entry')}")
    chk("C2-02 validate_trade_risk NOT called with live_ask (1.27520)",
        mock_vr_sell.call_args[1].get("entry") != 1.27520)


print("\n=== C2-03: Empty live-price response -> rejected ===")
sig_c2 = _signal(direction="BUY", entry_price=1.27500, stop_loss=1.27410, lot_size=0.01)
db_c2 = _make_db(signal=sig_c2, existing_trade=None, user=_user())

for empty_val in ({}, None):
    with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
         patch("app.signals.trade_monitor.get_live_price", return_value=empty_val), \
         patch("app.mt5.order_manager.place_order") as mock_place:
        r = activate_trade_from_signal(db_c2, user_id=10, signal_id=1)
        chk(f"C2-03 empty price ({empty_val!r}) rejected", not r["success"] and not mock_place.called)


print("\n=== C2-04 & C2-05: Missing ask or bid -> rejected ===")
with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.get_live_price", return_value={"bid": 1.27500}), \
     patch("app.mt5.order_manager.place_order") as mock_place:
    r_no_ask = activate_trade_from_signal(db_c2, user_id=10, signal_id=1)
    chk("C2-04 missing ask rejected", not r_no_ask["success"] and not mock_place.called, r_no_ask.get("message"))

with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.get_live_price", return_value={"ask": 1.27510}), \
     patch("app.mt5.order_manager.place_order") as mock_place:
    r_no_bid = activate_trade_from_signal(db_c2, user_id=10, signal_id=1)
    chk("C2-05 missing bid rejected", not r_no_bid["success"] and not mock_place.called, r_no_bid.get("message"))


print("\n=== C2-06 through C2-10: Zero, negative, NaN, infinite, non-numeric prices -> rejected ===")
invalid_price_cases = [
    ("zero ask", {"ask": 0.0, "bid": 1.27500}),
    ("zero bid", {"ask": 1.27510, "bid": 0.0}),
    ("negative ask", {"ask": -1.27510, "bid": 1.27500}),
    ("negative bid", {"ask": 1.27510, "bid": -1.27500}),
    ("NaN ask", {"ask": float("nan"), "bid": 1.27500}),
    ("NaN bid", {"ask": 1.27510, "bid": float("nan")}),
    ("inf ask", {"ask": float("inf"), "bid": 1.27500}),
    ("inf bid", {"ask": 1.27510, "bid": float("inf")}),
    ("non-numeric ask", {"ask": "abc", "bid": 1.27500}),
    ("non-numeric bid", {"ask": 1.27510, "bid": "xyz"}),
]

for label, p_dict in invalid_price_cases:
    with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
         patch("app.signals.trade_monitor.get_live_price", return_value=p_dict), \
         patch("app.mt5.order_manager.place_order") as mock_place:
        r_inv = activate_trade_from_signal(db_c2, user_id=10, signal_id=1)
        chk(f"C2 invalid price ({label}) rejected", not r_inv["success"] and not mock_place.called, r_inv.get("message"))


print("\n=== C2-11: Live price crosses structural SL (BUY: ask <= sl) -> rejected ===")
# BUY SL is 1.27410. Live ask reaches or drops below 1.27410
with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.get_live_price", return_value={"ask": 1.27410, "bid": 1.27408}), \
     patch("app.mt5.order_manager.place_order") as mock_place:
    r_sl_buy = activate_trade_from_signal(db_c2, user_id=10, signal_id=1)
    chk("C2-11 BUY ask <= SL rejected", not r_sl_buy["success"] and not mock_place.called)
    chk("C2-11 BUY message mentions crossed SL", "crossed" in r_sl_buy.get("message", "").lower(), r_sl_buy.get("message"))


print("\n=== C2-12: Live price crosses structural SL (SELL: bid >= sl) -> rejected ===")
# SELL SL is 1.27600. Live bid reaches or rises above 1.27600
with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.get_live_price", return_value={"ask": 1.27602, "bid": 1.27600}), \
     patch("app.mt5.order_manager.place_order") as mock_place:
    r_sl_sell = activate_trade_from_signal(db_sell, user_id=10, signal_id=1)
    chk("C2-12 SELL bid >= SL rejected", not r_sl_sell["success"] and not mock_place.called)
    chk("C2-12 SELL message mentions crossed SL", "crossed" in r_sl_sell.get("message", "").lower(), r_sl_sell.get("message"))


print("\n=== C2-13: Live-price movement makes stored lot unsafe (risk > $1.00) -> rejected ===")
# Signal created at lot=0.01 with SL=1.27410.
# Live ask moved further up to 1.27560 (SL dist = 15 pips).
# At 15 pips, 0.01 lot would risk $1.50 > $1.00.
# validate_trade_risk returns safe lot = 0.00 (min lot exceeds risk budget).
with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.get_live_price", return_value={"ask": 1.27560, "bid": 1.27558}), \
     patch("app.mt5.order_manager.place_order") as mock_place:
    r_unsafe_lot = activate_trade_from_signal(db_c2, user_id=10, signal_id=1)
    chk("C2-13 stored lot unsafe rejected", not r_unsafe_lot["success"] and not mock_place.called)
    chk("C2-13 message explains risk/lot mismatch",
        any(w in r_unsafe_lot.get("message", "").lower() for w in ("exceed", "match", "unsafe")),
        r_unsafe_lot.get("message"))


print("\n=== C2-13b: Live-price movement makes stored lot inconsistent -> rejected ===")
# Signal created at lot=0.01 with SL=1.27410.
# Live ask moved closer to SL to 1.27460 (SL dist = 5 pips).
# At 5 pips, safe lot for $1.00 risk is 0.02.
# Stored lot 0.01 != safe lot 0.02 -> rejected to flag inconsistency.
with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.get_live_price", return_value={"ask": 1.27460, "bid": 1.27458}), \
     patch("app.mt5.order_manager.place_order") as mock_place:
    r_inconsistent = activate_trade_from_signal(db_c2, user_id=10, signal_id=1)
    chk("C2-13b inconsistent lot rejected", not r_inconsistent["success"] and not mock_place.called)
    chk("C2-13b message explains unsafe or inconsistent",
        "inconsistent" in r_inconsistent.get("message", "").lower() or "not match" in r_inconsistent.get("message", "").lower(),
        r_inconsistent.get("message"))


print("\n=== C2-14: Actual risk exactly $1.00 -> accepted ===")
# BUY: Ask=1.27510, SL=1.27410 -> sl_dist=0.00100 (10 pips).
# Lot=0.01 -> risk = 0.01 * 10 * 10 = $1.0000.
trade_exact = MagicMock(); trade_exact.id = 14; trade_exact.result = "PENDING_APPROVAL"
trade_exact.actual_fill_price = None; trade_exact.mt5_ticket = None
trade_exact.symbol = "GBPUSD"; trade_exact.direction = "BUY"; trade_exact.broker_position_id = None

db_exact = _make_db(signal=sig_c2, existing_trade=None, user=_user())

with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.models.Trade", return_value=trade_exact), \
     patch("app.signals.trade_monitor.get_live_price", return_value=_LIVE_BUY_EXACT), \
     patch("app.mt5.order_manager.place_order") as mock_place_exact:
    mock_place_exact.return_value = {
        "success": True, "ticket": "MT5-14", "order_type": "ORDER_TYPE_BUY",
        "data": {"openPrice": 1.27510}
    }
    r_exact = activate_trade_from_signal(db_exact, user_id=10, signal_id=1)
    chk("C2-14 exact $1.00 risk accepted", r_exact["success"], str(r_exact))
    chk("C2-14 broker order placed once", mock_place_exact.call_count == 1)


print("\n=== C2-15: Actual risk above $1.00 (tiny excess) -> rejected ===")
# BUY: Ask=1.27511, SL=1.27410 -> sl_dist=0.00101 (10.1 pips).
# Lot=0.01 -> risk = 0.01 * 10.1 * 10 = $1.0100 > $1.00.
db_excess = _make_db(signal=sig_c2, existing_trade=None, user=_user())
sig15 = _signal(direction="BUY", entry_price=1.27500, stop_loss=1.27410, lot_size=0.01, status="pending")
db_excess = _make_db(signal=sig15, existing_trade=None, user=_user())

with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.get_live_price", return_value={"ask": 1.27511, "bid": 1.27509}), \
     patch("app.mt5.order_manager.place_order") as mock_place_excess:
    r_excess = activate_trade_from_signal(db_excess, user_id=10, signal_id=1)
    chk("C2-15 tiny excess over $1.00 rejected", not r_excess["success"], str(r_excess))
    chk("C2-15 no broker order placed", not mock_place_excess.called)
    chk("C2-15 message mentions cap or lot mismatch",
        any(w in r_excess.get("message", "").lower() for w in ("exceed", "match", "cap")),
        r_excess.get("message"))


print("\n=== C2-16: Original structural entry, SL, TP, lot unchanged after rejection ===")
sig16 = _signal(entry_price=1.27500, stop_loss=1.27410, take_profit=1.28000, lot_size=0.01)
db16 = _make_db(signal=sig16, existing_trade=None, user=_user())

with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.get_live_price", return_value={"ask": 1.27560, "bid": 1.27558}), \
     patch("app.mt5.order_manager.place_order"):
    activate_trade_from_signal(db16, user_id=10, signal_id=1)

chk("C2-16 entry_price unchanged after rejection", sig16.entry_price == 1.27500)
chk("C2-16 stop_loss unchanged after rejection", sig16.stop_loss == 1.27410)
chk("C2-16 take_profit unchanged after rejection", sig16.take_profit == 1.28000)
chk("C2-16 lot_size unchanged after rejection", sig16.lot_size == 0.01)


print("\n=== C2-17: Original structural entry, SL, TP, lot unchanged after success; Trade row entry is structural ===")
sig17 = _signal(entry_price=1.27500, stop_loss=1.27410, take_profit=1.28000, lot_size=0.01)
trade_obj17 = MagicMock()
trade_obj17.id = 17
trade_obj17.result = "PENDING_APPROVAL"
trade_obj17.actual_fill_price = None
trade_obj17.mt5_ticket = None
trade_obj17.symbol = "GBPUSD"
trade_obj17.direction = "BUY"
trade_obj17.broker_position_id = None
db17 = _make_db(signal=sig17, existing_trade=None, user=_user())

captured_trade_kwargs = {}
def _mock_trade_constructor(**kwargs):
    captured_trade_kwargs.update(kwargs)
    for k, v in kwargs.items():
        setattr(trade_obj17, k, v)
    return trade_obj17

with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.models.Trade", side_effect=_mock_trade_constructor), \
     patch("app.signals.trade_monitor.get_live_price", return_value={"ask": 1.27510, "bid": 1.27508}), \
     patch("app.mt5.order_manager.place_order") as mock_place17:
    mock_place17.return_value = {
        "success": True, "ticket": "MT5-17", "order_type": "ORDER_TYPE_BUY",
        "data": {"openPrice": 1.27512}
    }
    r17 = activate_trade_from_signal(db17, user_id=10, signal_id=1)

chk("C2-17 signal.entry_price unchanged after success", sig17.entry_price == 1.27500)
chk("C2-17 signal.stop_loss unchanged after success", sig17.stop_loss == 1.27410)
chk("C2-17 signal.take_profit unchanged after success", sig17.take_profit == 1.28000)
chk("C2-17 signal.lot_size unchanged after success", sig17.lot_size == 0.01)
chk("C2-17 Trade.entry_price is structural signal entry (1.27500)",
    captured_trade_kwargs.get("entry_price") == 1.27500,
    f"got {captured_trade_kwargs.get('entry_price')}")
chk("C2-17 Trade.actual_fill_price captures broker execution (1.27512)",
    trade_obj17.actual_fill_price == 1.27512,
    f"got {trade_obj17.actual_fill_price}")


print("\n=== C2-18: No lot increase or SL adjustment ===")
# Ensure lot is never increased to compensate or SL adjusted to make trade valid
sig18 = _signal(entry_price=1.27500, stop_loss=1.27410, lot_size=0.01)
db18 = _make_db(signal=sig18, existing_trade=None, user=_user())

with patch("app.mt5.metaapi_client.MetaApiClient.get_symbol_specs", return_value=_SPECS), \
     patch("app.signals.trade_monitor.get_live_price", return_value={"ask": 1.27530, "bid": 1.27528}), \
     patch("app.mt5.order_manager.place_order") as mock_place18:
    r18 = activate_trade_from_signal(db18, user_id=10, signal_id=1)
    chk("C2-18 trade rejected without adjusting parameters", not r18["success"])
    chk("C2-18 signal.lot_size never increased", sig18.lot_size == 0.01)
    chk("C2-18 signal.stop_loss never moved", sig18.stop_loss == 1.27410)
    chk("C2-18 no broker order placed", not mock_place18.called)


# ─────────────────────────────────────────────────────────────────────────────
# Final results
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  RESULTS: {len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print(f"    - {f}")
print("=" * 60)
sys.exit(0 if not FAIL else 1)
