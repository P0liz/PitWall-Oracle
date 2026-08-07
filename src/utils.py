import logging
import sys
import numpy as np
import pandas as pd
from logging.handlers import RotatingFileHandler
from .config import TIME_PENALTY_THRESHOLD, SESSION_MAPPING

# Status che indicano una vettura regolarmente classificata al termine della
# gara. Jolpica/FastF1 normalizza normalmente i distacchi in giri come "Lapped".
CLASSIFIED_FINISH_STATUSES = frozenset({"Finished", "Lapped"})

# Eventi amministrativi successivi alla gara: non rappresentano un ritiro
# avvenuto durante la simulazione e non devono alimentare il rischio DNF.
POST_RACE_EXCLUSION_STATUSES = frozenset({"Disqualified", "Excluded"})


def is_race_dnf(status: str | None) -> bool:
    """
    Classifica un DNF all-cause usando esclusivamente lo stato finale.

    Qualsiasi stato valorizzato che non rappresenti un arrivo classificato o
    un'esclusione amministrativa viene trattato come DNF. In questo modo sono
    inclusi sia incidenti/DNS sia motivazioni tecniche specifiche (per esempio
    "Brakes"), senza dipendere da una whitelist incompleta.
    """
    if status is None or pd.isna(status):
        return False

    normalized_status = str(status).strip()
    if not normalized_status:
        return False

    is_dnf = normalized_status not in CLASSIFIED_FINISH_STATUSES | POST_RACE_EXCLUSION_STATUSES
    return is_dnf


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
    # remove NaN values to avoid nanmin warning
    times = np.asarray(times, dtype=float)
    return np.nan if times.size == 0 or np.isnan(times).all() else np.nanmin(times)


def get_session_mapping(year: int, is_conventional: bool, race_type: str, data: str):
    if is_conventional:
        year = 0
    return SESSION_MAPPING.get((year, is_conventional, race_type, data), None)


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
