"""
app.core.logger
Thin wrapper so any module can do:
    from app.core.logger import get_logger
    logger = get_logger(__name__)
Logs to both console and a daily rotating file under logs/YYYY-MM-DD.log
"""
import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
LOGS_DIR.mkdir(exist_ok=True)

FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Configures handlers on first call."""
    logger = logging.getLogger(name)
    if not logging.root.handlers:
        formatter = logging.Formatter(FORMAT, datefmt=DATE_FMT)

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)

        # Daily rotating file: logs/YYYY-MM-DD.log, keeps 30 days
        log_file = LOGS_DIR / "app.log"
        file_handler = TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=30, encoding="utf-8"
        )
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setFormatter(formatter)

        logging.root.addHandler(console)
        logging.root.addHandler(file_handler)
        logging.root.setLevel(logging.INFO)
    return logger
