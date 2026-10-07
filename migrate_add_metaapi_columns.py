"""
One-time migration: adds metaapi_token and metaapi_account_id columns to the users table.
Run once: python migrate_add_metaapi_columns.py
Safe to run multiple times — skips if columns already exist.
"""
from app.db.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    for col in ("metaapi_token", "metaapi_account_id"):
        try:
            conn.execute(text(f"ALTER TABLE users ADD COLUMN {col} VARCHAR"))
            conn.commit()
            print(f"Added column: {col}")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print(f"Column already exists (skipped): {col}")
            else:
                raise

print("Migration complete.")
