import logging
import sys
import numpy as np
import pandas as pd
from logging.handlers import RotatingFileHandler
from .config import TIME_PENALTY_THRESHOLD


def got_end_penalty(driver_abb: str, race_laps: pd.DataFrame, race_results: pd.DataFrame):
    """
    Checks if the driver got a penalty at the end of the race
    """
    driver_laps = race_laps[race_laps["Driver"] == driver_abb]
    total_laps = race_laps["LapNumber"].max()
    ending_position = driver_laps.loc[driver_laps["LapNumber"] == total_laps, "Position"]
    if ending_position.empty:
        return False  # driver did not finish the race
    official_position = race_results.loc[race_results["Abbreviation"] == driver_abb, "Position"].iloc[0]
    if np.isnan(official_position) or np.isnan(ending_position.iloc[0]):
        return False  # defensive

    # Considering only big results changes to be excluded
    fixed_ending_position = ending_position.iloc[0] + TIME_PENALTY_THRESHOLD
    return official_position > fixed_ending_position


def get_driver_fastest_quali_time(quali_results_df: pd.DataFrame, driver_id: str):
    q1 = quali_results_df.loc[quali_results_df["driver_id"] == driver_id, "Q1"].dt.total_seconds()
    q2 = quali_results_df.loc[quali_results_df["driver_id"] == driver_id, "Q2"].dt.total_seconds()
    q3 = quali_results_df.loc[quali_results_df["driver_id"] == driver_id, "Q3"].dt.total_seconds()

    times = [q1, q2, q3]
    return np.nanmin(times)


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
