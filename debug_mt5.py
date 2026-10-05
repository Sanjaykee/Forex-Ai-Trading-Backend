import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import MetaTrader5 as mt5
from app.db.database import SessionLocal
from app.db.models import User

db = SessionLocal()
user = db.query(User).filter(User.mt5_login != None).first()
db.close()

print(f"Login in DB : {user.mt5_login}")
print(f"Server in DB: {user.mt5_server}")
print(f"Password    : {user.mt5_password}")
print()

# Step 1: try plain initialize first
print("Step 1: plain mt5.initialize()...")
result = mt5.initialize()
print(f"  Result: {result}")
print(f"  Error : {mt5.last_error()}")
if result:
    info = mt5.terminal_info()
    acc  = mt5.account_info()
    print(f"  Terminal path: {info.path if info else 'N/A'}")
    print(f"  Account login: {acc.login if acc else 'N/A'}")
    print(f"  Account server: {acc.server if acc else 'N/A'}")
mt5.shutdown()
print()

# Step 2: try with explicit path
MT5_PATH = "C:\\Program Files\\MetaTrader 5 IC Markets Global\\terminal64.exe"
print(f"Step 2: mt5.initialize(path={MT5_PATH})...")
result = mt5.initialize(path=MT5_PATH)
print(f"  Result: {result}")
print(f"  Error : {mt5.last_error()}")
if result:
    info = mt5.terminal_info()
    acc  = mt5.account_info()
    print(f"  Terminal path: {info.path if info else 'N/A'}")
    print(f"  Account login: {acc.login if acc else 'N/A'}")
    print(f"  Account server: {acc.server if acc else 'N/A'}")
mt5.shutdown()
print()

# Step 3: try login after initialize
print("Step 3: initialize then login...")
result = mt5.initialize(path=MT5_PATH)
print(f"  Initialize: {result} | {mt5.last_error()}")
if result:
    login_result = mt5.login(
        login=int(user.mt5_login),
        password=user.mt5_password,
        server=user.mt5_server
    )
    print(f"  Login result: {login_result}")
    print(f"  Error: {mt5.last_error()}")
    acc = mt5.account_info()
    print(f"  Account: {acc.login if acc else 'None'} @ {acc.server if acc else 'None'}")
mt5.shutdown()
