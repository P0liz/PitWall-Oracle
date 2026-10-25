import fastf1
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
import requests
from src.config import ARCHIVE_URL, DATA_DIR, FORECAST_URL, OPENF1_BASE_URL
from src.utils import setup_custom_logger
from fastf1.exceptions import DataNotLoadedError

log = setup_custom_logger("DataLoader")


def _load_data_with_retry(
    session, required_attr: str, max_attempts: int = 1, base_delay_seconds: float = 0.0, **load_kwargs
):
    """
    Carica una sessione FastF1 con singolo tentativo immediato senza attese.
    Mantenuta per retrocompatibilità.
    """
    session.load(**load_kwargs)
    try:
        data = getattr(session, required_attr)
        if data is not None and len(data) > 0:
            return session
    except DataNotLoadedError as error:
        raise RuntimeError(f"Impossibile caricare '{required_attr}' da FastF1") from error
    raise RuntimeError(f"Impossibile caricare '{required_attr}' da FastF1")


class BronzeLayer:
    data_dir = Path(DATA_DIR) / "bronze"

    def __init__(self):
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self._openf1_sessions_cache = {}

    def _resolve_openf1_session_key(self, year: int, race_number: int, session: int) -> int | None:
        """Risolve il session_key di OpenF1 associato all'evento/sessione richiesto."""
        if year not in self._openf1_sessions_cache:
            url = f"{OPENF1_BASE_URL}/sessions?year={year}"
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                self._openf1_sessions_cache[year] = resp.json()
            except Exception as e:
                log.warning(f"Errore nel recuperare le sessioni OpenF1 per l'anno {year}: {e}")
                return None

        openf1_sessions = self._openf1_sessions_cache.get(year, [])
        if not openf1_sessions:
            return None

        f1_session = fastf1.get_session(year, race_number, session)
        target_name = str(f1_session.name or f1_session.event.get(f"Session{session}", "")).strip().lower()
        target_date_raw = f1_session.event.get(f"Session{session}DateUtc", getattr(f1_session, "date", None))
        target_date = pd.to_datetime(target_date_raw, utc=True) if pd.notna(target_date_raw) else None

        def _normalize_name(name: str) -> str:
            return name.lower().replace("shootout", "qualifying").strip()

        norm_target_name = _normalize_name(target_name)

        candidates = []
        for s in openf1_sessions:
            s_name = _normalize_name(str(s.get("session_name", "")))
            if s_name == norm_target_name or (norm_target_name in s_name):
                s_date = pd.to_datetime(s.get("date_start"), format="ISO8601", utc=True)
                diff = abs((s_date - target_date).total_seconds()) if target_date is not None else 0.0
                candidates.append((diff, s["session_key"]))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            if target_date is None or candidates[0][0] <= 36 * 3600:
                return candidates[0][1]

        loc = str(f1_session.event.get("Location", "")).strip().lower()
        loc_candidates = []
        for s in openf1_sessions:
            s_loc = str(s.get("location", "")).strip().lower()
            s_short = str(s.get("circuit_short_name", "")).strip().lower()
            if loc and (loc in s_loc or loc in s_short or s_loc in loc):
                s_date = pd.to_datetime(s.get("date_start"), format="ISO8601", utc=True)
                diff = abs((s_date - target_date).total_seconds()) if target_date is not None else 0.0
                loc_candidates.append((diff, s["session_key"]))

        if loc_candidates:
            loc_candidates.sort(key=lambda x: x[0])
            if target_date is None or loc_candidates[0][0] <= 12 * 3600:
                return loc_candidates[0][1]

        return None

    def _get_openf1_raw_laps(self, year: int, race_number: int, session: int) -> pd.DataFrame:
        """Costruisce un DataFrame compatibile con FastF1 laps interrogando OpenF1."""
        session_key = self._resolve_openf1_session_key(year, race_number, session)
        if session_key is None:
            log.warning(f"Nessuna session_key OpenF1 trovata per {year}_{race_number}_{session}")
            return pd.DataFrame()

        try:
            laps_resp = requests.get(f"{OPENF1_BASE_URL}/laps?session_key={session_key}", timeout=15)
            laps_resp.raise_for_status()
            laps_data = laps_resp.json()
        except Exception as e:
            log.warning(f"Errore caricamento laps da OpenF1 ({session_key}): {e}")
            return pd.DataFrame()

        if not laps_data or not isinstance(laps_data, list):
            return pd.DataFrame()

        laps_df = pd.DataFrame(laps_data)
        if laps_df.empty or "lap_number" not in laps_df.columns:
            return pd.DataFrame()

        drv_map = {}
        team_map = {}
        try:
            drv_resp = requests.get(f"{OPENF1_BASE_URL}/drivers?session_key={session_key}", timeout=10)
            if drv_resp.status_code == 200:
                for d in drv_resp.json():
                    num = int(d.get("driver_number", -1))
                    drv_map[num] = str(d.get("name_acronym", ""))
                    team_map[num] = str(d.get("team_name", ""))
        except Exception as e:
            log.warning(f"Errore caricamento drivers da OpenF1 ({session_key}): {e}")

        stints_by_driver = {}
        try:
            stints_resp = requests.get(f"{OPENF1_BASE_URL}/stints?session_key={session_key}", timeout=10)
            if stints_resp.status_code == 200:
                for s in stints_resp.json():
                    num = int(s.get("driver_number", -1))
                    stints_by_driver.setdefault(num, []).append(s)
        except Exception as e:
            log.warning(f"Errore caricamento stints da OpenF1 ({session_key}): {e}")

        compounds, stint_numbers, tyre_lifes = [], [], []
        for _, lap in laps_df.iterrows():
            d_num = int(lap.get("driver_number", -1))
            lap_num = lap.get("lap_number", 0)
            matched = False
            for s in stints_by_driver.get(d_num, []):
                l_start = s.get("lap_start")
                l_end = s.get("lap_end")
                if l_start is not None and l_end is not None and l_start <= lap_num <= l_end:
                    compounds.append(str(s.get("compound", "UNKNOWN")).upper())
                    stint_numbers.append(float(s.get("stint_number", 1)))
                    start_age = s.get("tyre_age_at_start", 0) or 0
                    tyre_lifes.append(float(start_age + (lap_num - l_start) + 1))
                    matched = True
                    break
            if not matched:
                compounds.append("UNKNOWN")
                stint_numbers.append(np.nan)
                tyre_lifes.append(np.nan)

        deleted_set = set()
        try:
            rc_resp = requests.get(f"{OPENF1_BASE_URL}/race_control?session_key={session_key}", timeout=10)
            if rc_resp.status_code == 200:
                del_pattern = re.compile(r"CAR\s+(\d+).*?(?:LAP|L)\s*(\d+)", re.IGNORECASE)
                for msg_obj in rc_resp.json():
                    msg = str(msg_obj.get("message", ""))
                    if "DELETED" in msg:
                        m = del_pattern.search(msg)
                        if m:
                            deleted_set.add((int(m.group(1)), int(m.group(2))))
        except Exception as e:
            log.warning(f"Errore caricamento race_control da OpenF1 ({session_key}): {e}")

        deleted_laps = []
        for _, lap in laps_df.iterrows():
            d_num = int(lap.get("driver_number", -1))
            l_num = int(lap.get("lap_number", 0))
            deleted_laps.append((d_num, l_num) in deleted_set)

        positions = []
        try:
            pos_resp = requests.get(f"{OPENF1_BASE_URL}/position?session_key={session_key}", timeout=10)
            if pos_resp.status_code == 200 and pos_resp.json():
                pos_df = pd.DataFrame(pos_resp.json())
                pos_df["date"] = pd.to_datetime(pos_df["date"], format="ISO8601", utc=True)
                laps_df["_dt_start"] = pd.to_datetime(laps_df["date_start"], format="ISO8601", utc=True)
                laps_df["_dt_end"] = laps_df["_dt_start"] + pd.to_timedelta(laps_df["lap_duration"].fillna(0), unit="s")
                pos_by_driver = {int(k): v.sort_values("date") for k, v in pos_df.groupby("driver_number")}

                for _, lap in laps_df.iterrows():
                    d_num = int(lap.get("driver_number", -1))
                    d_pos = pos_by_driver.get(d_num)
                    if d_pos is None or d_pos.empty:
                        positions.append(np.nan)
                        continue
                    lap_end = lap["_dt_end"]
                    prior = d_pos.loc[d_pos["date"] <= lap_end, "position"]
                    positions.append(float(prior.iloc[-1]) if not prior.empty else float(d_pos["position"].iloc[0]))
            else:
                positions = [np.nan] * len(laps_df)
        except Exception as e:
            log.warning(f"Errore caricamento position da OpenF1 ({session_key}): {e}")
            positions = [np.nan] * len(laps_df)

        is_out_lap = laps_df.get("is_pit_out_lap", pd.Series([False] * len(laps_df))).fillna(False).astype(bool)
        is_accurate = laps_df["lap_duration"].notna() & (~is_out_lap)

        raw_df = pd.DataFrame({
            "Time": pd.to_datetime(laps_df["date_start"], format="ISO8601", utc=True),
            "Driver": laps_df["driver_number"].astype(int).map(drv_map).fillna(""),
            "DriverNumber": laps_df["driver_number"].astype(str),
            "LapTime": pd.to_timedelta(laps_df["lap_duration"], unit="s"),
            "LapNumber": laps_df["lap_number"].astype(float),
            "Stint": stint_numbers,
            "PitOutTime": [pd.Timedelta(seconds=0) if x else pd.NaT for x in is_out_lap],
            "PitInTime": pd.NaT,
            "Sector1Time": pd.to_timedelta(laps_df.get("duration_sector_1"), unit="s"),
            "Sector2Time": pd.to_timedelta(laps_df.get("duration_sector_2"), unit="s"),
            "Sector3Time": pd.to_timedelta(laps_df.get("duration_sector_3"), unit="s"),
            "SpeedI1": laps_df.get("i1_speed"),
            "SpeedI2": laps_df.get("i2_speed"),
            "SpeedST": laps_df.get("st_speed"),
            "IsPersonalBest": False,
            "Compound": compounds,
            "TyreLife": tyre_lifes,
            "FreshTyre": True,
            "Team": laps_df["driver_number"].astype(int).map(team_map).fillna(""),
            "Position": positions,
            "Deleted": deleted_laps,
            "IsAccurate": is_accurate,
        })

        raw_df = raw_df.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
        return raw_df

    def _get_openf1_raw_results(self, year: int, race_number: int, session: int) -> pd.DataFrame:
        """Costruisce un DataFrame compatibile con FastF1 results interrogando OpenF1."""
        session_key = self._resolve_openf1_session_key(year, race_number, session)
        if session_key is None:
            log.warning(f"Nessuna session_key OpenF1 trovata per {year}_{race_number}_{session}")
            return pd.DataFrame()

        try:
            res_resp = requests.get(f"{OPENF1_BASE_URL}/session_result?session_key={session_key}", timeout=15)
            res_resp.raise_for_status()
            res_data = res_resp.json()
        except Exception as e:
            log.warning(f"Errore caricamento session_result da OpenF1 ({session_key}): {e}")
            return pd.DataFrame()

        if not res_data or not isinstance(res_data, list):
            return pd.DataFrame()

        res_df = pd.DataFrame(res_data)

        drivers = {}
        try:
            drv_resp = requests.get(f"{OPENF1_BASE_URL}/drivers?session_key={session_key}", timeout=10)
            if drv_resp.status_code == 200:
                for d in drv_resp.json():
                    drivers[int(d.get("driver_number", -1))] = d
        except Exception as e:
            log.warning(f"Errore caricamento drivers da OpenF1 ({session_key}): {e}")

        grid_map = {}
        try:
            meeting_key = res_df.iloc[0].get("meeting_key")
            if meeting_key:
                grid_resp = requests.get(f"{OPENF1_BASE_URL}/starting_grid?meeting_key={meeting_key}", timeout=10)
                if grid_resp.status_code == 200 and isinstance(grid_resp.json(), list):
                    for g in grid_resp.json():
                        grid_map[int(g.get("driver_number", -1))] = float(g.get("position", np.nan))
        except Exception as e:
            log.warning(f"Errore caricamento starting_grid da OpenF1: {e}")

        rows = []
        is_quali = session in (2, 4)

        for _, r in res_df.iterrows():
            d_num = int(r.get("driver_number", -1))
            drv = drivers.get(d_num, {})
            pos = float(r.get("position", np.nan)) if pd.notna(r.get("position")) else np.nan
            dns = bool(r.get("dns", False))
            dnf = bool(r.get("dnf", False))
            dsq = bool(r.get("dsq", False))
            gap = str(r.get("gap_to_leader", ""))

            if dns:
                status = "Did not start"
            elif dsq:
                status = "Disqualified"
            elif dnf:
                status = "Retired"
            elif "LAP" in gap.upper():
                status = "Lapped"
            elif not np.isnan(pos):
                status = "Finished"
            else:
                status = "Retired"

            dur = r.get("duration")
            dur_list = dur if isinstance(dur, list) else ([dur] if pd.notna(dur) else [])
            q1 = pd.to_timedelta(dur_list[0], unit="s") if len(dur_list) > 0 and pd.notna(dur_list[0]) else pd.NaT
            q2 = pd.to_timedelta(dur_list[1], unit="s") if len(dur_list) > 1 and pd.notna(dur_list[1]) else pd.NaT
            q3 = pd.to_timedelta(dur_list[2], unit="s") if len(dur_list) > 2 and pd.notna(dur_list[2]) else pd.NaT

            time_val = pd.NaT
            if not is_quali and pd.notna(r.get("duration")) and isinstance(r.get("duration"), (int, float)):
                time_val = pd.to_timedelta(r["duration"], unit="s")
            elif is_quali:
                q_valid = [q for q in (q3, q2, q1) if pd.notna(q)]
                time_val = q_valid[0] if q_valid else pd.NaT

            grid_pos = grid_map.get(d_num, pos if is_quali else np.nan)

            first_name = drv.get("first_name", "")
            last_name = drv.get("last_name", "")
            abb = drv.get("name_acronym", "")
            full_name = drv.get("full_name", f"{first_name} {last_name}".strip())
            team_name = drv.get("team_name", "")
            driver_id_slug = f"{first_name}_{last_name}".strip().lower().replace(" ", "_")

            rows.append({
                "DriverNumber": str(d_num),
                "Abbreviation": abb,
                "DriverId": driver_id_slug,
                "TeamName": team_name,
                "FirstName": first_name,
                "LastName": last_name,
                "FullName": full_name,
                "Position": pos,
                "ClassifiedPosition": str(int(pos)) if not np.isnan(pos) else "R",
                "GridPosition": grid_pos,
                "Q1": q1,
                "Q2": q2,
                "Q3": q3,
                "Time": time_val,
                "Status": status,
                "Points": float(r.get("points", 0.0) or 0.0),
                "Laps": float(r.get("number_of_laps", 0) or 0),
            })

        df = pd.DataFrame(rows)
        return df.sort_values("Position", na_position="last").reset_index(drop=True)

    def get_raw_laps(self, year: int, race_number: int, session: int):
        filename = f"{year}_{race_number}_{session}_raw_laps.parquet"
        if filename in os.listdir(self.data_dir):
            return pd.read_parquet(self.data_dir / filename)

        df = None
        try:
            data = fastf1.get_session(year, race_number, session)
            data.load(telemetry=False, weather=False, messages=False)
            laps = getattr(data, "laps", None)
            if laps is not None and len(laps) > 0:
                df = pd.DataFrame(data=laps).reset_index(drop=True)
        except Exception as error:
            log.warning(f"FastF1 load non riuscito per raw_laps ({year}_{race_number}_{session}): {error}")

        if df is None or df.empty:
            log.info(f"Fallback su OpenF1 per raw_laps ({year}_{race_number}_{session})...")
            df = self._get_openf1_raw_laps(year, race_number, session)

        if df is None or df.empty:
            raise RuntimeError(f"Impossibile caricare 'laps' per {year}_{race_number}_{session} sia da FastF1 sia da OpenF1")

        df.to_parquet(self.data_dir / filename)
        return df

    def get_raw_results(self, year: int, race_number: int, session: int):
        filename = f"{year}_{race_number}_{session}_raw_results.parquet"
        if filename in os.listdir(self.data_dir):
            return pd.read_parquet(self.data_dir / filename)

        df = None
        try:
            data = fastf1.get_session(year, race_number, session)
            data.load(laps=False, telemetry=False, weather=False, messages=False)
            results = getattr(data, "results", None)
            if results is not None and len(results) > 0:
                df = pd.DataFrame(data=results).reset_index(drop=True)
        except Exception as error:
            log.warning(f"FastF1 load non riuscito per raw_results ({year}_{race_number}_{session}): {error}")

        if df is None or df.empty:
            log.info(f"Fallback su OpenF1 per raw_results ({year}_{race_number}_{session})...")
            df = self._get_openf1_raw_results(year, race_number, session)

        if df is None or df.empty:
            raise RuntimeError(f"Impossibile caricare 'results' per {year}_{race_number}_{session} sia da FastF1 sia da OpenF1")

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
            return pd.read_parquet(self.data_dir / filename)

        data = fastf1.get_session(year, race_number, 1)
        event_obj = getattr(data, "event", None)
        if event_obj is None or event_obj.empty:
            try:
                data.load(laps=False, telemetry=False, weather=False, messages=False)
                event_obj = data.event
            except Exception as e:
                log.warning(f"FastF1 load per event metadata ({year}_{race_number}): {e}")

        if event_obj is None or event_obj.empty:
            raise RuntimeError(f"Impossibile caricare 'event' per {year}_{race_number}")

        df = pd.DataFrame([event_obj]).reset_index(drop=True)
        df.to_parquet(self.data_dir / filename)
        return df

    # TODO: could use for historical track weather, but does not do predictions
    def get_raw_weather_old(self, year: int, race_number: int, session: int):
        filename = f"{year}_{race_number}_{session}_raw_weather_fastf1.parquet"
        if filename in os.listdir(self.data_dir):
            return pd.read_parquet(self.data_dir / filename)

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
