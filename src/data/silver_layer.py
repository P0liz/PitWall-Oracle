import math
import os
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR
from .bronze_layer import BronzeLayer
from .history_builder import build_driver_id, map_team_id
from src.utils import setup_custom_logger

TIME_THRESHOLD = 1.04


class SilverLayer:

    def __init__(self):
        self.data_dir = Path(DATA_DIR) / "silver"
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.bronze = BronzeLayer()
        self.log = setup_custom_logger("DataLoader")

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
                self.log.error("LapTime column is empty")
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
                    # Discard stupidly long times
                    if df.iloc[i]["LapTime"] > time_reference * 1.2:
                        to_drop.append(i)
                        continue
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

    def get_untouched_laps(self, year: int, race_number: int, session: int, force: bool):
        return self.bronze.get_raw_laps(year, race_number, session)

    def get_clean_results(self, year: int, race_number: int, session: int, force: bool):
        """
        Returns a cleaned version of the results data for a given year, race number and session.

        Required columns in the raw data: 'Position', 'Abbreviation', 'Status', 'Laps'.
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
                self.log.error("Position column is empty")
                raise ValueError("Position column is empty")

            if df["Abbreviation"].empty:
                # TODO: possibly handle if too much data is corrupted
                self.log.error("Abbreviation column is empty")
                raise ValueError("Abbreviation column is empty")

            # TODO: handle if necessary
            # corrections for 2023 or before not supported (hard to distinguish sprint weekend from normal)
            if (session == 5 or session == 3) and year >= 2024:  # race or sprint race
                if df["Laps"].empty or df["Status"].empty:
                    self.log.error("Laps or Status column is empty")
                    raise ValueError("Laps or Status column is empty")

                # Solve issues with NaN values in Laps column
                if df["Laps"].isnull().any().any():
                    self.log.warning(f"NaN values found in Laps column, correcting - {year}_{race_number}_{session}")
                    max_laps = df["Laps"].max()
                    null_idx = df[df["Laps"].isnull()].index
                    for idx in null_idx:
                        position = df.at[idx, "Position"]
                        status = df.at[idx, "Status"]
                        if pd.isna(position):
                            df.at[idx, "Laps"] = 0
                        elif pd.isna(status):
                            df.at[idx, "Laps"] = 0
                        elif status == "Finished":
                            df.at[idx, "Laps"] = max_laps
                        elif status == "Lapped":
                            df.at[idx, "Laps"] = max_laps - 1
                        elif status == "Withdrew":
                            df.at[idx, "Laps"] = 0
                        else:
                            df.at[idx, "Laps"] = max_laps // 2

            # Custom correction cause data sucks
            if (df.loc[df["Abbreviation"] == "ANT", "FirstName"] == "Andrea Kimi").any():
                self.log.warning(f"Found Andrea Kimi in ANT row, changing to Kimi - {year}_{race_number}_{session}")
                df.loc[df["Abbreviation"] == "ANT", "FirstName"] = "Kimi"

            # Add driver_id and team_id columns
            df["driver_id"] = df.apply(
                lambda r: build_driver_id(r["Abbreviation"], r["FirstName"], r["LastName"]), axis=1
            )
            df["team_id"] = df["TeamName"].map(map_team_id)
            df.to_parquet(self.data_dir / filename)
        return df

    def get_clean_pit_stops(
        self, year: int, race_number: int, race_results: pd.DataFrame | None = None, force: bool = False
    ) -> pd.DataFrame:
        """Parse Jolpica pit stops and attach the project's driver/team identifiers."""
        filename = f"{year}_{race_number}_clean_pit_stops.parquet"
        path = self.data_dir / filename
        if path.exists() and not force:
            cached = pd.read_parquet(path)
            if {"driver_id", "team_id"}.issubset(cached.columns):
                return cached

        # Load pit stops data
        raw_df = self.bronze.get_raw_pit_stops(year, race_number)
        columns = ["driver_id", "team_id", "lap", "stop", "time", "duration_seconds"]
        if raw_df.empty:
            df = pd.DataFrame(columns=columns)
            df.to_parquet(path, index=False)
            self.log.warning(f"Empty pit stop dataFrame saved to {path}")
            return df
        if race_results.empty:
            self.log.error("Race results is empty")
            raise ValueError("Race results is empty")

        required_columns = {"driverId", "lap", "stop", "duration"}
        missing_columns = required_columns.difference(raw_df.columns)
        if missing_columns:
            self.log.error(f"Colonne pit stop mancanti: {sorted(missing_columns)}")
            raise ValueError(f"Colonne pit stop mancanti: {sorted(missing_columns)}")

        # Data cleaning
        df = raw_df.rename(columns={"driverId": "DriverId"}).copy()
        df["lap"] = pd.to_numeric(df["lap"], errors="coerce")
        df["stop"] = pd.to_numeric(df["stop"], errors="coerce")
        df["duration_seconds"] = pd.to_timedelta(df["duration"], errors="coerce").dt.total_seconds()
        # Join with race results to assign driver_id and team_id
        identities = race_results.loc[:, ["DriverId", "driver_id", "team_id"]].drop_duplicates("DriverId")
        df = df.merge(identities, on="DriverId", how="inner", validate="many_to_one")
        df = (
            df.dropna(subset=["driver_id", "team_id", "lap", "stop", "duration_seconds"])
            .loc[lambda frame: (frame["lap"] > 0) & (frame["stop"] > 0) & (frame["duration_seconds"] > 0)]
            .drop_duplicates(subset=["driver_id", "stop"])
            .sort_values(["driver_id", "stop"])
        )
        df["lap"] = df["lap"].astype(int)
        df["stop"] = df["stop"].astype(int)
        df = df.reindex(columns=columns).reset_index(drop=True)
        df.to_parquet(path, index=False)
        return df

    def get_clean_event_metadata(self, year: int, race_number: int, force: bool):
        filename = f"{year}_{race_number}_clean_event.parquet"
        if filename in os.listdir(self.data_dir) and not force:
            # Load from file
            df = pd.read_parquet(self.data_dir / filename)
        else:
            df = self.bronze.get_event_metadata(year, race_number)
            if df.empty:
                self.log.error("Event metadata is empty")
                raise ValueError("Event metadata is empty")
            # Custom correction cause data sucks
            if (df["Location"] == "Miami Gardens").any():
                self.log.warning(f"Found Miami Gardens in Location, changing to Miami - {year}_{race_number}")
                df.loc[df["Location"] == "Miami Gardens", "Location"] = "Miami"
            if (df["Location"] == "Monte Carlo").any():
                self.log.warning(f"Found Monte Carlo in Location, changing to Monaco - {year}_{race_number}")
                df.loc[df["Location"] == "Monte Carlo", "Location"] = "Monaco"
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
