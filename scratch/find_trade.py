import sqlite3
import glob

dbs = glob.glob('*.db') + glob.glob('*/*.db') + glob.glob('*/*/*.db')
print('Found dbs:', dbs)
for db in dbs:
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cur.fetchall()
        print(db, 'tables:', tables)
        for t in tables:
            if 'trade' in t[0].lower():
                cur.execute(f"SELECT * FROM {t[0]} WHERE symbol LIKE '%EUR%' ORDER BY id DESC LIMIT 5;")
                print(f'Trades in {db}:', cur.fetchall())
    except Exception as e:
        print(db, e)
