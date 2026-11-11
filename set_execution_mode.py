"""
One-time script: sets execution_mode=auto for all users in the DB.
Run once from the backend/ directory:
    python set_execution_mode.py
"""
from app.db.database import SessionLocal
from app.db import crud, models

db = SessionLocal()
try:
    users = db.query(models.User).all()
    if not users:
        print("No users found in DB.")
    for user in users:
        crud.upsert_setting(db, user.id, "execution_mode", "auto")
        current_mode = crud.get_setting(db, user.id, "execution_mode")
        print(f"User id={user.id} email={user.email} execution_mode={current_mode}")
    print("Done.")
finally:
    db.close()
