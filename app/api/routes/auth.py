from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db import crud
from app.api.schemas.signal_schema import UserRegister, UserLogin
from app.config import settings
from jose import jwt
from datetime import datetime, timedelta

from sqlalchemy.exc import IntegrityError
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

def create_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": str(user_id), "exp": expire}, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

@router.post("/register")
def register(data: UserRegister, db: Session = Depends(get_db)):
    if crud.get_user_by_email(db, data.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    try:
        user = crud.create_user(db, data.email, data.password, data.full_name)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already registered")
    except Exception as e:
        db.rollback()
        logger.error(f"User registration error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Registration error: {str(e)}")
    token = create_token(user.id)
    return {"access_token": token, "token_type": "bearer", "user_id": user.id}

@router.post("/login")
def login(data: UserLogin, db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, data.email)
    if not user or not crud.verify_password(data.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    token = create_token(user.id)
    return {"access_token": token, "token_type": "bearer", "user_id": user.id}
