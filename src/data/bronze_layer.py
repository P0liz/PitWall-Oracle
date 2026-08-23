import fastf1
import pandas as pd
import os
from pathlib import Path
import requests
from src.config import ARCHIVE_URL, DATA_DIR, FORECAST_URL
from src.utils import setup_custom_logger

import time
from fastf1.exceptions import DataNotLoadedError

log = setup_custom_logger("DataLoader")


def _load_data_with_retry(
    session, required_attr: str, max_attempts: int = 3, base_delay_seconds: float = 60.0, **load_kwargs
):
    """
    Carica una sessione FastF1 con retry: FastF1 non solleva eccezioni sui
    singoli fetch falliti (solo warning), quindi il controllo di successo
    reale avviene leggendo `required_attr` dopo il load, non catturando
    un'eccezione su session.load() stesso.
    """
    last_error = None
    fastf1.set_log_level("DEBUG")
    for attempt in range(1, max_attempts + 1):
        session.load(**load_kwargs)
        try:
            data = getattr(session, required_attr)
            if data is not None and len(data) > 0:
                return session
        except DataNotLoadedError as error:
            last_error = error

        if attempt < max_attempts:
            delay = base_delay_seconds * attempt  # backoff lineare/esponenziale a scelta
            log.warning(
                f"Tentativo {attempt}/{max_attempts} fallito nel caricare "
                f"'{required_attr}', ritento tra {delay:.0f}s"
            )
            time.sleep(delay)

    raise RuntimeError(f"Impossibile caricare '{required_attr}' dopo {max_attempts} tentativi") from last_error


class BronzeLayer:
    data_dir = Path(DATA_DIR) / "bronze"

    def __init__(self):
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def get_raw_laps(self, year: int, race_number: int, session: int):
        filename = f"{year}_{race_number}_{session}_raw_laps.parquet"
        if filename in os.listdir(self.data_dir):
            # Load from file
            df = pd.read_parquet(self.data_dir / filename)
        else:
            # Load from API
            data = fastf1.get_session(year, race_number, session)
            _load_data_with_retry(data, required_attr="laps")
            laps = data.laps
            df = pd.DataFrame(data=laps).reset_index(drop=True)
            df.to_parquet(self.data_dir / filename)
        return df

    def get_raw_results(self, year: int, race_number: int, session: int):
        filename = f"{year}_{race_number}_{session}_raw_results.parquet"
        if filename in os.listdir(self.data_dir):
            # Load from file
            df = pd.read_parquet(self.data_dir / filename)
        else:
            # Load from API
            data = fastf1.get_session(year, race_number, session)
            _load_data_with_retry(data, required_attr="results")
            results = data.results
            df = pd.DataFrame(data=results).reset_index(drop=True)
            df.to_parquet(self.data_dir / filename)
        return df

    def get_raw_pit_stops(self, year: int, race_number: int):
        """
        Return the main-race pit stops published by the Jolpica Ergast-compatible API.

        Jolpica's ``duration`` is the total pit-lane traversal time, not only
        the stationary wheel-change time. The raw value is retained here and
        converted to seconds in Silver.
        """
        filename = f"{year}_{race_number}_raw_pit_stops.parquet"
        path = self.data_dir / filename
        if path.exists():
            return pd.read_parquet(path)

        response = fastf1.ergast.Ergast(result_type="pandas", auto_cast=True, limit=100).get_pit_stops(
            season=year, round=race_number
        )
        columns = ["driverId", "lap", "stop", "time", "duration"]
        frames = [pd.DataFrame(frame) for frame in response.content if not frame.empty]
        df = pd.concat(frames, ignore_index=True).reindex(columns=columns) if frames else pd.DataFrame(columns=columns)
        df.to_parquet(path, index=False)
        return df

    def get_event_metadata(self, year: int, race_number: int):
        filename = f"{year}_{race_number}_event.parquet"
        if filename in os.listdir(self.data_dir):
            # Load from file
            df = pd.read_parquet(self.data_dir / filename)
        else:
            # Load from API
            data = fastf1.get_session(year, race_number, 1)
            _load_data_with_retry(data, required_attr="event")
            df = pd.DataFrame([data.event]).reset_index(drop=True)
            df.to_parquet(self.data_dir / filename)
        return df

    # TODO: could use for historical track weather, but does not do predictions
    def get_raw_weather_old(self, year: int, race_number: int, session: int):
        filename = f"{year}_{race_number}_{session}_raw_weather_fastf1.parquet"
        if filename in os.listdir(self.data_dir):
            # Load from file
            df = pd.read_parquet(self.data_dir / filename)
        else:
            # Load from API
            data = fastf1.get_session(year, race_number, session)
            data.load()
            event = data.weather_data
            df = pd.DataFrame(data=event).reset_index(drop=True)
            df.to_parquet(self.data_dir / filename)
        return df

    def get_raw_weather(
        self,
        year: int,
        race_number: int,
        session: int,
        latitude: float,
        longitude: float,
        race_datetime_utc: pd.Timestamp,
        future: bool = False,
    ):
        """
        future=False -> Archive API, precipitation in mm (historical training set).
        future=True  -> Forecast API, precipitation_probability in % (future races).
        Saves raw hourly data as parquet, same pattern as FastF1 raw data.
        """
        filename = f"{year}_{race_number}_{session}_raw_weather.parquet"
        if filename in os.listdir(self.data_dir) and not future:
            return pd.read_parquet(self.data_dir / filename)

        race_date = race_datetime_utc.date().isoformat()
        url, field = (FORECAST_URL, "precipitation_probability") if future else (ARCHIVE_URL, "precipitation")

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": race_date,
            "end_date": race_date,
            "hourly": field,
            "timezone": "UTC",
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()

        df = pd.DataFrame(
            {
                "time": pd.to_datetime(payload["hourly"]["time"]),
                "value": payload["hourly"][field],
                "field": field,
                "future": future,
            }
        )
        df.to_parquet(self.data_dir / filename, index=False)
        return df
