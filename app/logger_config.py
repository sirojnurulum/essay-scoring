"""
Centralized logging configuration for the Essay Scoring application.

This module provides a function to set up a standardized logger that can be
used across different parts of the application, such as the training script
and the FastAPI server. It configures logging to output to both the console
and a rotating file.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
import os


def setup_logger():
    """
    Configures and returns a logger for the application.

    The logger is configured to:
    - Log messages of level INFO and above.
    - Output logs to the console (stdout).
    - Output logs to a rotating file `logs/app.log`, with a max size of 5MB
      and keeping one backup file.
    - Use a standardized format for log messages.

    Returns:
        logging.Logger: A configured logger instance.
    """
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    logger = logging.getLogger('essay_scorer')
    logger.setLevel(logging.INFO)

    # Add console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    if not logger.handlers:
        logger.addHandler(stream_handler)

        # Add rotating file handler
        file_handler = RotatingFileHandler(os.path.join(log_dir, 'app.log'), maxBytes=5*1024*1024, backupCount=1)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    return logger