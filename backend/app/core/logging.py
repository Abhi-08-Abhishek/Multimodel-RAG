import logging
import sys
from typing import Optional


def setup_logging(level: Optional[str] = "INFO") -> logging.Logger:
    """
    Configure a clean, humanized console logging formatter for the backend.
    """
    log_format = "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger("novacore_rag")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


logger = setup_logging()
