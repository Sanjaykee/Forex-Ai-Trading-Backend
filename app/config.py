import os
from pathlib import Path
from pydantic_settings import BaseSettings

_backend_dir = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------------------
# Active Profile Selector (Exact same concept as Spring Boot application.yml)
# Priority:
#   1. System Environment Variable: ACTIVE_PROFILE or APP_ENV (Cloud Hosts e.g. Render)
#   2. Master .env file line: ACTIVE_PROFILE=local or ACTIVE_PROFILE=prod
# ------------------------------------------------------------------------------
_profile = os.getenv("ACTIVE_PROFILE") or os.getenv("APP_ENV")

if not _profile:
    _master_env = _backend_dir / ".env"
    if _master_env.exists():
        try:
            with open(_master_env, "r", encoding="utf-8") as f:
                for line in f:
                    clean_line = line.strip()
                    if clean_line.startswith("ACTIVE_PROFILE="):
                        _profile = clean_line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        except Exception:
            pass

_profile = (_profile or "local").lower()
is_prod = _profile in ("prod", "production")
active_profile = "prod" if is_prod else "local"

# Select target profile file (1-to-1 match: 'local' -> .env.local, 'prod' -> .env.prod)
_target_file = _backend_dir / (".env.prod" if is_prod else ".env.local")
if not _target_file.exists():
    _target_file = _backend_dir / (".env.production" if is_prod else ".env.development")
if not _target_file.exists():
    _target_file = _backend_dir / ".env"

class Settings(BaseSettings):
    ACTIVE_PROFILE: str = active_profile
    APP_ENV: str = "production" if is_prod else "development"
    DATABASE_URL: str = "postgresql+psycopg2://postgres:root@localhost:5432/forexai"
    SECRET_KEY: str = "forexai_super_secret_key_2024_do_not_share"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080
    TELEGRAM_BOT_TOKEN: str = "8885177423:AAFhpQFKhqCvc50OhtwvYywlu0vLs8BVcmk"
    FRONTEND_URL: str = "http://localhost:5173"
    GEMINI_API_KEY: str = ""
    STRIPE_SECRET_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    METAAPI_TOKEN: str = ""
    METAAPI_ACCOUNT_ID: str = ""
    METAAPI_REGION: str = "new-york"

    @property
    def is_production(self) -> bool:
        return self.ACTIVE_PROFILE in ("prod", "production")

    class Config:
        env_file = str(_target_file)
        extra = "ignore"

settings = Settings()

# Startup Log
print(f"[ForexAI] Active Profile: [{settings.ACTIVE_PROFILE.upper()}] -> Loaded file: {_target_file.name}")
