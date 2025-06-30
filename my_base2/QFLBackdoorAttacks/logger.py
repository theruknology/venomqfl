"""
logger.py: Logging utility for QFLBackdoorAttacks
Sets up both console and file logging, controlled by config verbosity.
"""
import logging
import os
from datetime import datetime

LOG_FORMAT = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'

# Map config verbosity to logging levels
def get_log_level(verbosity):
    if verbosity >= 2:
        return logging.DEBUG
    elif verbosity == 1:
        return logging.INFO
    else:
        return logging.WARNING

def setup_logger(log_dir=None, verbosity=1, log_name='qfl'):
    logger = logging.getLogger(log_name)
    logger.setLevel(get_log_level(verbosity))
    logger.handlers = []  # Remove old handlers

    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (if log_dir provided)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'{log_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
