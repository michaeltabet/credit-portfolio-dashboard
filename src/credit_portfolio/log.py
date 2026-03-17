"""Structured logging factory for the credit portfolio system."""

import logging

from credit_portfolio.config import load_config


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger, reading level/format from config.yaml."""
    cfg = load_config()
    log_cfg = cfg.get("logging", {})
    level = log_cfg.get("level", "INFO")
    fmt = log_cfg.get("format", "%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger
