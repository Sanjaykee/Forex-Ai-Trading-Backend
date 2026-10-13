"""
Symmetric encryption for sensitive fields (MT5 password) stored in DB.
Uses Fernet (AES-128-CBC + HMAC-SHA256) from the cryptography package.
Key is derived from SECRET_KEY in config — no separate key management needed.
"""
import base64
import hashlib
from cryptography.fernet import Fernet


def _get_fernet() -> Fernet:
    from app.config import settings
    # Derive a 32-byte key from SECRET_KEY using SHA-256
    raw = hashlib.sha256(settings.SECRET_KEY.encode()).digest()
    key = base64.urlsafe_b64encode(raw)
    return Fernet(key)


def encrypt_password(plain: str) -> str:
    """Encrypt a plain-text password. Returns base64 ciphertext string."""
    if not plain:
        return ""
    try:
        return _get_fernet().encrypt(plain.encode()).decode()
    except Exception:
        return plain  # fallback: store as-is if crypto fails


def decrypt_password(cipher: str) -> str:
    """Decrypt a previously encrypted password. Returns plain text."""
    if not cipher:
        return ""
    try:
        return _get_fernet().decrypt(cipher.encode()).decode()
    except Exception:
        return cipher  # fallback: return as-is (handles legacy plain-text passwords)
