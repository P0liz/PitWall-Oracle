import pandas as pd
import numpy as np
from pathlib import Path
from src.config import DATA_DIR, TEAM_ID_MAPPING
from src.utils import (
    POST_RACE_EXCLUSION_STATUSES,
    get_driver_fastest_quali_time,
    got_end_penalty,
    is_race_dnf,
    setup_custom_logger,
)

log = setup_custom_logger("DataLoader")


class HistoryBuilder:
    def __init__(self, silver, feature_engineer):
        self.history_path = Path(DATA_DIR) / "gold" / "driver_team_history.parquet"
        self.silver = silver
        self.feature_engineer = feature_engineer
        self.log = log

    def get_history_up_to(self, race_date):
        if self.history_path.exists():
            history_df = pd.read_parquet(self.history_path)
            return history_df[history_df["race_date"] < race_date]  # always <
        return pd.DataFrame()

    def get_history(self):
        try:
            return pd.read_parquet(self.history_path)
        except:
            self.log.warning("History parquet vuoto")
            return None

    # TODO: adesso che per necessità passo year, race_number, session ad update_history
    # avrebbe senso che questa chiamasse da sola il silver per i dati che necessita?
    def update_history(
        self,
        year: int,
        race_number: str,
        session: int,
        quali_results: pd.DataFrame,
        race_results: pd.DataFrame,
        race_laps: pd.DataFrame,
        circuit_location: str,
        race_date,
        force: bool = False,
    ):
        if not force and race_date in self.get_history()["race_date"].values:
            self.log.info(f"History già presente per la gara {race_number}, skippo")
            return

        # Upfront validation checks
        if race_results.empty or quali_results.empty:
            self.log.error(f"Dati mancanti per race {race_number}")
            raise ValueError(f"Dati mancanti per race {race_number}")

        required_cols = ["Abbreviation"]
        if not all(col in race_results.columns for col in required_cols):
            self.log.error(f"Colonne {required_cols} mancanti in race_results")
            raise ValueError(f"Colonne {required_cols} mancanti in race_results")
        if race_results[required_cols].isnull().any().any():
            self.log.error(f"Valori nulli nelle colonne {required_cols}")
            raise ValueError(f"Valori nulli nelle colonne {required_cols}")

        # Load pit stops data only for main race sessions
        pit_stops = (
            self.silver.get_clean_pit_stops(year, race_number, race_results=race_results, force=force)
            if session == 5
            else pd.DataFrame()
        )

        new_history_rows = self.build_history_rows(
            quali_results=quali_results,
            race_results=race_results,
            race_laps=race_laps,
            pit_stops=pit_stops,
            race_number=race_number,
            session=session,
            race_date=race_date,
            year=year,
            circuit_location=circuit_location,
        )

        if self.history_path.exists():
            history_df = pd.read_parquet(self.history_path)
            # get df without the race rows, if it exists
            history_df = history_df.loc[history_df["race_date"] != race_date]
            updated_history = pd.concat([history_df, new_history_rows], ignore_index=True)
        else:
            updated_history = new_history_rows

        updated_history.to_parquet(self.history_path, index=False)

    def build_history_rows(
        self,
        quali_results: pd.DataFrame,
        race_results: pd.DataFrame,
        race_laps: pd.DataFrame,
        pit_stops: pd.DataFrame | None,
        race_number: str,
        session: int,
        race_date: pd.Timestamp,
        year: int,
        circuit_location: str,
    ):
        rows = []
        team_pit_metrics = compute_team_pit_metrics(pit_stops)
        for _, race_row in race_results.iterrows():
            driver_id = build_driver_id(race_row["Abbreviation"], race_row["FirstName"], race_row["LastName"])
            team_id = map_team_id(race_row["TeamName"])
            quali_row = quali_results.loc[quali_results["driver_id"] == driver_id]
            quali_position = (
                quali_row["Position"].iloc[0]
                if not quali_row.empty and pd.notna(quali_row["Position"].iloc[0])
                else (float(race_row["GridPosition"]) if pd.notna(race_row["GridPosition"]) else np.nan)
            )
            grid_position = (
                float(race_row["GridPosition"])
                if pd.notna(race_row["GridPosition"])
                else (float(quali_row["Position"].iloc[0]) if not quali_row.empty else np.nan)
            )
            is_podium = race_row["Position"] in (1, 2, 3) if pd.notna(race_row["Position"]) else False

            status = race_row["Status"]
            position = race_row["Position"]
            if (
                is_race_dnf(status)
                or pd.isna(position)
                or got_end_penalty(race_row["Abbreviation"], race_laps, race_results)
                or status in POST_RACE_EXCLUSION_STATUSES
            ):
                position = np.nan  # esclusa, non "ultimo posto"

            # super_time = float(race_laps.loc[race_laps["Driver"] == race_row["Abbreviation"], "LapTime"].dt.total_seconds().min())
            quali_time = get_driver_fastest_quali_time(quali_results, driver_id)

            race_pace = (
                race_laps.loc[race_laps["Driver"] == race_row["Abbreviation"], "LapTime"].dt.total_seconds().mean()
                if not race_laps.empty
                else np.nan
            )

            lap_1_position = race_laps.loc[
                (race_laps["Driver"] == race_row["Abbreviation"]) & (race_laps["LapNumber"] == 1), "Position"
            ]

            rain_probability = self.feature_engineer.get_rain_probability(
                year, race_number, session, circuit_location, race_date, force=True
            )

            # WATCH OUT: when chainging data in the history, delete the parquet
            rows.append(
                {
                    "race_date": race_date,  # race identifier
                    "race_number": race_number,  # race data
                    "session": session,  # race data
                    "year": year,  # race data
                    "driver_id": driver_id,  # driver identifier
                    "team_id": team_id,  # team identifier
                    "circuit_id": circuit_location,  # circuit identifier
                    "quali_position": quali_position,
                    "grid_position": grid_position,
                    "lap_1_position": lap_1_position.iloc[0] if not lap_1_position.empty else np.nan,
                    "race_position": position,
                    "points_scored": race_row["Points"],
                    "laps_completed": race_row["Laps"],
                    "quali_time": quali_time,
                    "race_pace": race_pace,
                    "pit_execution_zscore": team_pit_metrics.get(team_id, {}).get("pit_execution_zscore", np.nan),
                    "first_stop_lap_zscore": team_pit_metrics.get(team_id, {}).get("first_stop_lap_zscore", np.nan),
                    "status_raw": status,
                    "is_podium": is_podium,
                    "rain_probability": rain_probability,
                }
            )

        return pd.DataFrame(rows)


# Helpers
def compute_team_pit_metrics(pit_stops: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    """
    Compute within-race team pit metrics.

    ``pit_execution_zscore`` is based on each team's median total pit-lane
    duration; lower values mean faster execution.
    ``first_stop_lap_zscore`` compares the team's earliest first stop with
    the field median; lower values mean an earlier, more aggressive stop.
    """
    if pit_stops is None or pit_stops.empty:
        return {}

    required_pit_columns = {"driver_id", "team_id", "lap", "stop", "duration_seconds"}
    stops = pit_stops.dropna(subset=required_pit_columns)
    if stops.empty:
        return {}

    team_duration = stops.groupby("team_id")["duration_seconds"].median()
    duration_std = team_duration.std(ddof=0)
    if duration_std > 0:
        execution_zscore = (team_duration - team_duration.mean()) / duration_std
    else:
        execution_zscore = pd.Series(np.nan, index=team_duration.index, dtype=float)

    first_driver_stops = stops.sort_values(["driver_id", "stop", "lap"]).drop_duplicates("driver_id", keep="first")
    team_first_stop = first_driver_stops.groupby("team_id")["lap"].min()
    first_stop_std = team_first_stop.std(ddof=0)
    if first_stop_std > 0:
        first_stop_zscore = (team_first_stop - team_first_stop.median()) / first_stop_std
    else:
        first_stop_zscore = pd.Series(np.nan, index=team_first_stop.index, dtype=float)

    team_ids = execution_zscore.index.union(first_stop_zscore.index)
    return {
        team_id: {
            "pit_execution_zscore": float(execution_zscore.get(team_id, np.nan)),
            "first_stop_lap_zscore": float(first_stop_zscore.get(team_id, np.nan)),
        }
        for team_id in team_ids
    }


def build_driver_id(abbreviation: str, first_name: str, last_name: str) -> str:
    if not abbreviation or not first_name or not last_name:
        log.warning(f"Abbreviation, FirstName o LastName mancanti -" f"possibile driver_id incompleto, da sistemare")
    raw = f"{abbreviation}_{first_name}_{last_name}"
    return raw.strip().lower().replace(" ", "_")


def map_team_id(team_name: str) -> str:
    if team_name not in TEAM_ID_MAPPING:
        log.warning(
            f"TeamName '{team_name}' non presente in TEAM_ID_MAPPING -",
            f"fallback al nome grezzo, aggiungilo manualmente al dizionario",
        )
        return team_name  # fallback visibile, non silenzioso
    return TEAM_ID_MAPPING[team_name]
