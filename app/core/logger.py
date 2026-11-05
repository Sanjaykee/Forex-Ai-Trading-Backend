"""
app.core.logger
Thin wrapper so any module can do:
    from app.core.logger import get_logger
    logger = get_logger(__name__)
Uses the standard Python logging hierarchy — no extra dependencies.
"""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Configures a StreamHandler on first call."""
    logger = logging.getLogger(name)
    if not logging.root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)
    return logger
