import requests

urls = [
    ("Yahoo Finance", "https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X"),
    ("Frankfurter",   "https://api.frankfurter.app/latest"),
    ("ExchangeRate",  "https://open.er-api.com/v6/latest/USD"),
    ("Fixer",         "https://data.fixer.io/api/latest"),
    ("Google",        "https://www.google.com"),
]

for name, url in urls:
    try:
        r = requests.get(url, timeout=5)
        print(f"{name}: OK ({r.status_code})")
    except Exception as e:
        print(f"{name}: FAILED - {str(e)[:80]}")
