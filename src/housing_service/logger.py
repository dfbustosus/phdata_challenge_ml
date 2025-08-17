"""
Colored logging system following best practices.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any


# ANSI color codes for terminal output
class LogColors:
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels"""

    LEVEL_COLORS = {
        logging.DEBUG: LogColors.BLUE,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.BOLD + LogColors.RED,
    }

    def format(self, record):
        # Save the original levelname
        levelname = record.levelname
        # Add color to the levelname
        if record.levelno in self.LEVEL_COLORS:
            color = self.LEVEL_COLORS[record.levelno]
            record.levelname = f"{color}{levelname}{LogColors.RESET}"

        # Format the message using the parent formatter
        return super().format(record)


def setup_logger(
    name: str = __name__, level: int | None = None, log_to_file: bool = False
) -> logging.Logger:
    """Set up and configure a logger with colored output for the application.

    Args:
        name (str): The logger's name, typically the module's __name__.
        level (int, optional): The logging level (default is from LOG_LEVEL env var or INFO).
        log_to_file (bool): Whether to also log to a file.

    Returns:
        logging.Logger: Configured logger instance with colored output.
    """
    # Get the level if it's not provided
    if level is None:
        # Convert string level from environment to int if present
        level_str = os.environ.get("LOG_LEVEL", "INFO")
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = level_map.get(level_str, logging.INFO)

    # Set up logging with specified level
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create a colored formatter
    formatter = ColoredFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add formatter to the handler
    console_handler.setFormatter(formatter)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.hasHandlers():
        logger.addHandler(console_handler)

    # Add file handler if requested
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Ensure propagation is set correctly to avoid duplicate logs
    logger.propagate = False

    return logger
