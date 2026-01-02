"""
Logger utility functions for ScrollWeaver.
"""

import os
import logging
import datetime
from .file_utils import get_root_dir, create_dir


def get_logger(experiment_name: str):
    """Get logger for experiment."""
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(get_root_dir(), f"log/{experiment_name}")
    create_dir(log_dir)
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{current_time}.log"),
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # Avoid logging duplication
    logger.propagate = False

    return logger

