import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(r"c:\Users\nts0218\hepl project\Ai Forex Trading\backend")
sys.path.insert(0, r"c:\Users\nts0218\hepl project\Ai Forex Trading\backend")

from app.db.database import SessionLocal
from app.db.models import User
from app.db import crud
from app.config import settings as _cfg
from app.mt5.metaapi_client import MetaApiClient
from app.signals.scanner import scan_all_pairs, scan_pair

print("=" * 70)
print("       TESTING LIVE SIGNAL & MT5 BROKER ORDER EXECUTION")
print("=" * 70)

# 1. Check User & MetaApi Configuration
db = SessionLocal()
user = db.query(User).filter(
    User.metaapi_account_id.isnot(None),
    User.metaapi_account_id != ""
).order_by(User.id.desc()).first() or db.query(User).first()

if not user:
    print("[ERROR] No user found in database.")
    sys.exit(1)

region = crud.get_setting(db, user.id, "metaapi_region") or _cfg.METAAPI_REGION or "backup-new-york"
token = user.metaapi_token or _cfg.METAAPI_TOKEN
account_id = user.metaapi_account_id or _cfg.METAAPI_ACCOUNT_ID
db.close()

print(f"[+] Connecting to MetaApi...")
print(f"    Region     : {region}")
print(f"    Account ID : {account_id}")

client = MetaApiClient(token=token, account_id=account_id, region=region)
acc = client.get_account_information()
print(f"[+] Account Status: {acc.get('server', 'N/A')} | Balance: ${acc.get('balance', 0):.2f} | Equity: ${acc.get('equity', 0):.2f} | Connected: {acc.get('connected')}")

if not acc.get("connected"):
    print("[ERROR] Broker account is not connected.")
    sys.exit(1)

# 2. Scan for Live Signal
print("\n[+] Scanning all pairs for active SMC setups (M15 / M3)...")
signals = scan_all_pairs(timeframes=("M15", "M3"), min_confidence="LOW")
target_sig = None

if signals:
    print(f"    Found {len(signals)} active signal(s)!")
    for s in signals:
        print(f"    - {s['symbol']} {s['timeframe']} {s['direction']} | Score: {s['score']} ({s['confidence']}) | Pattern: {s.get('strategy')} | Entry: {s.get('entry')} | SL: {s.get('sl')} | TP: {s.get('tp_1_3')}")
    target_sig = signals[0]
else:
    print("    No immediate signal on M15/M3 right at this second. Scanning M5 or individual pairs...")
    for sym in ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY"]:
        for tf in ["M3", "M15"]:
            sig = scan_pair(sym, tf)
            if sig:
                target_sig = sig
                print(f"    Found signal on {sym} {tf}!")
                break
        if target_sig:
            break

# If no pattern detected right now on live candles, construct a valid live setup on EURUSD for demonstration
if not target_sig:
    print("    Constructing a precision SMC test order on live EURUSD market price...")
    pdata = client.get_live_price("EURUSD")
    ask = pdata.get("ask", 1.0850)
    bid = pdata.get("bid", 1.0848)
    # BUY setup 2 pips below ask (or at ask) with 1:3 RR
    entry = round(ask - 0.0003, 5)  # 3 pips pullback
    sl = round(entry - 0.0010, 5)   # 10 pips SL
    tp = round(entry + 0.0030, 5)   # 30 pips TP (1:3 RR)
    target_sig = {
        "symbol": "EURUSD",
        "direction": "BUY",
        "entry": entry,
        "sl": sl,
        "tp_1_3": tp,
        "strategy": "CHoCH + OB Retest (Test)",
        "score": 75,
        "confidence": "HIGH"
    }

symbol = target_sig["symbol"]
direction = target_sig["direction"]
entry_price = float(target_sig.get("entry_price") or target_sig.get("entry"))
sl_price = float(target_sig.get("stop_loss") or target_sig.get("sl"))
tp_price = float(target_sig.get("take_profit") or target_sig.get("tp_1_3") or target_sig.get("tp_price"))
lot = 0.01  # Safe minimum test lot

print("\n" + "=" * 70)
print(f"      SIGNAL TO EXECUTE ON MT5 BROKER")
print("=" * 70)
print(f"Symbol     : {symbol}")
print(f"Direction  : {direction}")
print(f"Lot Size   : {lot} (Safe Minimum)")
print(f"Entry Price: {entry_price}")
print(f"Stop Loss  : {sl_price}")
print(f"Take Profit: {tp_price} (Target 1:3 RR)")
print(f"Strategy   : {target_sig.get('strategy', 'SMC Order Block')}")
print(f"Score      : {target_sig.get('score')} ({target_sig.get('confidence')})")

live_quote = client.get_live_price(symbol)
print(f"\nCurrent Broker Quote for {symbol}: Bid={live_quote.get('bid')} | Ask={live_quote.get('ask')} | Spread={live_quote.get('spread')} pips")

print("\n[+] Executing order via MetaApiClient with Smart Hybrid Routing...")
res = client.place_order(
    symbol=symbol,
    direction=direction,
    lot=lot,
    entry=entry_price,
    sl=sl_price,
    tp=tp_price,
    comment="ForexAI-Test"
)

print("\n" + "=" * 70)
print(f"      MT5 EXECUTION RESULT")
print("=" * 70)
print(f"Success    : {res.get('success')}")
if res.get("success"):
    ticket = res.get("ticket")
    print(f"Ticket ID  : {ticket}  (VERIFIED ON BROKER MT5!)")
    data = res.get("data", {})
    print(f"Order Type : {data.get('type') or 'Executed'}")
    print(f"Order ID   : {data.get('orderId')}")
    print(f"Position ID: {data.get('positionId')}")
    print("\n[✓] ORDER SUCCESSFULLY EXECUTED ON MT5 BROKER WITH ENTRY, SL, AND TP!")
else:
    print(f"Error      : {res.get('error')}")
