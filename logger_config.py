"""
Logging configuration - writes to both file and console
"""

import logging
import sys
from datetime import datetime


def setup_logger(log_file: str = None):
    """Setup logger to write to file and console"""
    
    # Default log file name with timestamp
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/run_{timestamp}.log"
    
    # Create logs directory if needed
    import os
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger("wipro_agent")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler (writes to file)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Console handler (shows on screen)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')  # Simpler for console
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Global logger instance
logger = setup_logger()