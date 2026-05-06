"""Centralized logging configuration for NARE.

Logs are written to `.nare_memory/nare.log` inside the working directory.
Nothing is printed to stdout/stderr so the CLI stays clean.
When NARE_LOG_LEVEL=DEBUG or --debug is passed, a StreamHandler is added
automatically by `app.py` via `logging.basicConfig`.
"""

import logging
import os
import sys
import threading
from typing import Optional


# Thread-safe lock for handler configuration
_handler_lock = threading.Lock()
_configured_loggers = set()
_log_file_handler: Optional[logging.FileHandler] = None


def _get_file_handler() -> logging.FileHandler:
    """Lazily create a single shared file handler."""
    global _log_file_handler
    if _log_file_handler is None:
        log_dir = os.path.join(os.getcwd(), ".nare_memory")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "nare.log")
        _log_file_handler = logging.FileHandler(log_path, encoding="utf-8")
        _log_file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
    return _log_file_handler


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name (defaults to caller's module)
    
    Returns:
        Configured logger with file-only output (no console spam).
    """
    logger = logging.getLogger(name or __name__)
    
    # Thread-safe configuration check
    with _handler_lock:
        logger_id = id(logger)
        if logger_id not in _configured_loggers:
            # Clear any existing handlers to prevent duplicates
            logger.handlers.clear()
            
            # File handler only — keeps the terminal clean
            logger.addHandler(_get_file_handler())
            
            # Set level from environment or default to INFO
            log_level = os.getenv('NARE_LOG_LEVEL', 'INFO').upper()
            logger.setLevel(getattr(logging, log_level, logging.INFO))
            
            # Prevent propagation to root logger
            logger.propagate = False
            
            # Mark as configured
            _configured_loggers.add(logger_id)
    
    return logger


def cleanup_logger(name: Optional[str] = None) -> None:
    """Remove handlers from a logger to prevent memory leaks.
    
    Args:
        name: Logger name to cleanup
    """
    logger = logging.getLogger(name or __name__)
    with _handler_lock:
        logger_id = id(logger)
        if logger_id in _configured_loggers:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            _configured_loggers.discard(logger_id)
