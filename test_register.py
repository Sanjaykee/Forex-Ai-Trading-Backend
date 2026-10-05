import sys
sys.path.insert(0, '.')

try:
    from passlib.context import CryptContext
    ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
    print("passlib OK:", ctx.hash("test"))
except Exception as e:
    print("passlib ERROR:", e)

try:
    from app.db.database import engine
    from app.db.models import Base
    Base.metadata.create_all(bind=engine)
    print("DB OK")
except Exception as e:
    print("DB ERROR:", e)

try:
    from app.db.database import SessionLocal
    from app.db import crud
    db = SessionLocal()
    user = crud.create_user(db, "debugtest@test.com", "Test@1234", "Debug User")
    print("Register OK, user id:", user.id)
    db.close()
except Exception as e:
    print("Register ERROR:", e)
