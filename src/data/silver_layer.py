import math
import os
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR
from .bronze_layer import BronzeLayer
from .history_builder import build_driver_id, map_team_id

TIME_THRESHOLD = 1.04


class SilverLayer:
    data_dir = Path(DATA_DIR) / "silver"

    def __init__(self):
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.bronze = BronzeLayer()

    def get_clean_laps(self, year: int, race_number: int, session: int, force: bool):
        """
        Returns a cleaned version of the laps data for a given year, race number and session.

        The cleaning process involves removing laps that are deleted, inaccurate,
        or have lap times that deviate significantly from the driver's previous lap time.

        Required columns in the raw data: 'LapTime', 'Deleted', 'IsAccurate', 'Driver' (Abbreviation).
        """
        assert (session >= 1) & (session <= 5), "Session number not valid: 1 <= session <= 5"

        filename = f"{year}_{race_number}_{session}_clean_laps.parquet"
        if filename in os.listdir(self.data_dir) and not force:
            # Load from file
            df = pd.read_parquet(self.data_dir / filename)
        else:
            # Load raw data
            df = self.bronze.get_raw_laps(year, race_number, session)
            if df["LapTime"].empty:
                raise ValueError("LapTime column is empty")

            time_reference = 0
            driver_ref = ""
            to_drop = []
            # print("Tot: ", df.shape[0])
            for i in range(0, df.shape[0]):
                # Default data cleaning
                if pd.isna(df.iloc[i]["LapTime"]):
                    to_drop.append(i)
                    continue
                if df.iloc[i]["Deleted"] == True:
                    to_drop.append(i)
                    continue
                if df.iloc[i]["IsAccurate"] == False:
                    to_drop.append(i)
                    continue
                # Get reference for new driver
                if df.iloc[i]["Driver"] != driver_ref:
                    time_reference = df.iloc[i]["LapTime"]
                    driver_ref = df.iloc[i]["Driver"]
                    continue

                # Keep only laps that are within 4% of the driver's last lap time
                # Discard any Nan laptimes
                if df.iloc[i]["LapTime"] > time_reference * TIME_THRESHOLD:
                    # Check if the times switch from quali to race sim
                    if i < df.shape[0] - 3:
                        new_ref_first = df.iloc[i]["LapTime"]
                        new_ref_second = df.iloc[i + 1]["LapTime"]
                        new_ref_third = df.iloc[i + 2]["LapTime"]
                        # If the next 3 laps are all within 4% of each other, we assume they are valid
                        if (
                            new_ref_second < new_ref_first * TIME_THRESHOLD
                            and new_ref_third < new_ref_second * TIME_THRESHOLD
                        ):
                            time_reference = new_ref_first
                            continue
                    to_drop.append(i)
                else:
                    time_reference = df.iloc[i]["LapTime"]

            df = df.drop(to_drop).reset_index(drop=True)
            df["LapTime"] = df["LapTime"].dt.total_seconds()
            df.to_parquet(self.data_dir / filename)
        return df

    def get_clean_results(self, year: int, race_number: int, session: int, force: bool):
        """
        Returns a cleaned version of the results data for a given year, race number and session.

        Required columns in the raw data: 'Position', 'Abbreviation', 'Points', 'Status', 'Laps'.
        """
        assert (session >= 1) & (session <= 5), "Session number not valid: 1 <= session <= 5"

        filename = f"{year}_{race_number}_{session}_clean_results.parquet"
        if filename in os.listdir(self.data_dir) and not force:
            # Load from file
            df = pd.read_parquet(self.data_dir / filename)
        else:
            # Load raw data
            df = self.bronze.get_raw_results(year, race_number, session)
            if df["Position"].empty:
                # TODO: possibly handle if too much data is corrupted
                raise ValueError("Position column is empty")

            if df["Abbreviation"].empty:
                # TODO: possibly handle if too much data is corrupted
                raise ValueError("Abbreviation column is empty")

            # TODO: handle if necessary
            if session == 5:  # race
                if df["Points"].empty or df["Laps"].empty or df["Status"].empty:
                    raise ValueError("Points, Laps or Status column is empty")
                # Solve issues with columns Points, Laps, Status
                # TODO: map position to points to write Points column
                # TODO: find a way to know how many laps a driver has completed
                # That info could also help write the Status column

            # Custom correction cause data sucks
            if (df.loc[df["Abbreviation"] == "ANT", "FirstName"] == "Andrea Kimi").any():
                df.loc[df["Abbreviation"] == "ANT", "FirstName"] = "Kimi"

            # Add driver_id and team_id columns
            df["driver_id"] = df.apply(
                lambda r: build_driver_id(r["Abbreviation"], r["FirstName"], r["LastName"]), axis=1
            )
            df["team_id"] = df["TeamName"].map(map_team_id)
            df.to_parquet(self.data_dir / filename)
        return df

    def get_event_metadata(self, year: int, race_number: int):
        df = self.bronze.get_event_metadata(year, race_number)
        if df.empty:
            raise ValueError("Event metadata is empty")
        return df

    def get_clean_weather(
        self,
        year: int,
        race_number: int,
        session: int,
        latitude: float,
        longitude: float,
        race_datetime_utc: pd.Timestamp,
        future: bool = False,
        force: bool = False,
    ):
        assert (session >= 1) & (session <= 5), "Session number not valid: 1 <= session <= 5"

        # Read from parquet
        filename = f"{year}_{race_number}_{session}_clean_weather.parquet"
        if filename in os.listdir(self.data_dir) and not force:
            return pd.read_parquet(self.data_dir / filename)

        # Get raw weather data and compute probability
        raw_df = self.bronze.get_raw_weather(year, race_number, session, latitude, longitude, race_datetime_utc, future)
        hourly_df = raw_df.dropna(subset=["value"])

        if hourly_df.empty:
            rain_probability = 0.0
        else:
            nearest_idx = (hourly_df["time"] - race_datetime_utc).dt.total_seconds().abs().argmin()
            mm = hourly_df["value"].iloc[nearest_idx]
            if future:
                rain_probability = mm / 100.0
            else:
                rain_probability = 1 / (1 + math.exp(-2 * (mm - 1)))  # 0mm->0.12, 1mm->0.5, 4mm->0.98

        df = pd.DataFrame(
            {"year": [year], "race_number": [race_number], "session": [session], "rain_probability": [rain_probability]}
        )
        df.to_parquet(self.data_dir / filename, index=False)
        return df
