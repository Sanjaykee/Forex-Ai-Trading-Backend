"""Add cTrader columns to users table."""
from app.db.database import engine
from sqlalchemy import text

cols = [
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS ctrader_client_id VARCHAR",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS ctrader_client_secret VARCHAR",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS ctrader_account_id VARCHAR",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS ctrader_server VARCHAR DEFAULT 'demo.ctrader.com'",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS ctrader_access_token VARCHAR",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS ctrader_refresh_token VARCHAR",
]

with engine.connect() as conn:
    for sql in cols:
        conn.execute(text(sql))
    conn.commit()
    print("cTrader columns added successfully.")
