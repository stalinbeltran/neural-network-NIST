"""Logging simple para las corridas. TODO: afinar formato/handlers."""
from __future__ import annotations

import logging


def get_logger(name: str = "nnist") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger
