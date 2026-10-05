import MetaTrader5 as mt5
import os

print("MT5 version:", mt5.__version__)
print("MT5 package location:", mt5.__file__)
print()

# Try to find running MT5 terminal
import subprocess
result = subprocess.run(['tasklist'], capture_output=True, text=True)
mt5_processes = [line for line in result.stdout.split('\n') if 'terminal' in line.lower() or 'metatrader' in line.lower()]
print("Running MT5 processes:")
for p in mt5_processes:
    print(" ", p)
print()

# Try initialize with timeout
print("Trying mt5.initialize()...")
r = mt5.initialize()
print("Result:", r)
print("Error:", mt5.last_error())
mt5.shutdown()
