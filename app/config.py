import os
from pathlib import Path
from pydantic_settings import BaseSettings

_backend_dir = Path(__file__).resolve().parent.parent

# Read ACTIVE_PROFILE: system env first (Render sets this), then .env file
_profile = os.getenv("ACTIVE_PROFILE") or os.getenv("APP_ENV")
if not _profile:
    _master_env = _backend_dir / ".env"
    if _master_env.exists():
        for line in _master_env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ACTIVE_PROFILE="):
                _profile = line.strip().split("=", 1)[1].strip()
                break

_profile = (_profile or "local").lower()
is_prod  = _profile in ("prod", "production")

_target_file = _backend_dir / (".env.prod" if is_prod else ".env.local")

class Settings(BaseSettings):
    ACTIVE_PROFILE: str = "prod" if is_prod else "local"
    DATABASE_URL: str = "postgresql+psycopg2://postgres:root@localhost:5432/forexai"
    SECRET_KEY: str = "forexai_super_secret_key_2024"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080
    FRONTEND_URL: str = "http://localhost:5173"
    TELEGRAM_BOT_TOKEN: str = ""
    GEMINI_API_KEY: str = ""
    CTRADER_CLIENT_ID: str = ""
    CTRADER_CLIENT_SECRET: str = ""
    METAAPI_TOKEN: str = ""
    METAAPI_ACCOUNT_ID: str = ""
    METAAPI_REGION: str = "backup-new-york"

    @property
    def is_production(self) -> bool:
        return self.ACTIVE_PROFILE in ("prod", "production")

    class Config:
        env_file = str(_target_file)
        extra = "ignore"

settings = Settings()
print(f"[ForexAI] Profile: [{settings.ACTIVE_PROFILE.upper()}] -> {_target_file.name}")
