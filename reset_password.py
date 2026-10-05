import sys
sys.path.insert(0, '.')
from app.db.database import SessionLocal
from app.db import models
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
db = SessionLocal()
user = db.query(models.User).filter(models.User.email == "sanjaykee79@gmail.com").first()
if user:
    user.password_hash = pwd_context.hash("Dhachu@2007"[:72])
    db.commit()
    print("Password reset OK")
else:
    print("User not found")
db.close()
