import logging
import sys
from logging.handlers import RotatingFileHandler


def setup_custom_logger(name):
    # 1. Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture everything from DEBUG up

    # 2. Define the log format (Time - Name - Level - Message)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 3. Create a Console Handler (prints to terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Only show INFO and above in console
    console_handler.setFormatter(formatter)

    # 4. Create a File Handler (saves to a file with rotation so it doesn't grow forever)
    file_handler = RotatingFileHandler(
        "app.log", maxBytes=1024 * 1024 * 5, backupCount=3  # 5MB per file, keeps 3 backups
    )
    file_handler.setLevel(logging.DEBUG)  # Save everything (including DEBUG) to the file
    file_handler.setFormatter(formatter)

    # 5. Add handlers to the logger
    # Prevent duplicate handlers if function is called multiple times
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


"""
# --- How to use it in your code ---
log = setup_custom_logger("MyApp")

log.debug("This is a debug message (only goes to the file).")
log.info("This is an info message (goes to console AND file).")
log.warning("Uh oh, something might be wrong.")
log.error("An error occurred!")
log.critical("The application is crashing!")
"""
