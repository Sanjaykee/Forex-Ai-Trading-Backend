from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.database import Base, engine
from app.api.routes.auth        import router as auth_router
from app.api.routes.signals     import router as signals_router
from app.api.routes.other_routes import (
    trades_router, performance_router,
    settings_router, pairs_router
)
from app.api.routes.chat import router as chat_router
from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create tables — wrapped so server starts even if DB is briefly unavailable
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.warning(f"Database connection issue on startup: {e}")

app = FastAPI(title="ForexAI API", version="1.0.0", debug=True)

# CORS: Support localhost, custom FRONTEND_URL, and all Vercel domains
cors_origins = [settings.FRONTEND_URL, "http://localhost:5173", "http://localhost:3000"]
if settings.FRONTEND_URL not in cors_origins:
    cors_origins.append(settings.FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(signals_router)
app.include_router(trades_router)
app.include_router(performance_router)
app.include_router(settings_router)
app.include_router(pairs_router)
app.include_router(chat_router)

@app.get("/")
def root():
    return {"status": "ForexAI API running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}
