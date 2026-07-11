import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from src.config import DATA_DIR, TEAM_ID_MAPPING


class HistoryBuilder:
    def __init__(self, silver):
        self.history_path = Path(DATA_DIR) / "gold" / "driver_team_history.parquet"
        self.silver = silver

    def get_history_up_to(self, race_date):
        if self.history_path.exists():
            history_df = pd.read_parquet(self.history_path)
            return history_df[history_df["race_date"] < race_date]  # always <
        return pd.DataFrame()

    def update_history(self, year: int, race_number: str, session: int, circuit_location: str, race_date, force: bool):
        quali_results = self.silver.get_clean_results(year, race_number, session - 1, force)
        race_results = self.silver.get_clean_results(year, race_number, session, force)

        # Upfront validation checks
        if race_results.empty or quali_results.empty:
            raise ValueError(f"Dati mancanti per race {race_number}")

        required_cols = ["Abbreviation", "Position", "Points", "Status", "Laps"]
        if not all(col in race_results.columns for col in required_cols):
            raise ValueError(f"Colonne mancanti in race_results")
        if race_results[required_cols].isnull().any().any():
            raise ValueError("Valori nulli nelle colonne richieste")

        new_history_rows = build_history_rows(
            quali_results=quali_results,
            race_results=race_results,
            race_number=race_number,
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


# Helpers
def build_history_rows(
    quali_results: pd.DataFrame,
    race_results: pd.DataFrame,
    race_number: str,
    race_date: pd.Timestamp,
    year: int,
    circuit_location: str,
):
    rows = []
    for _, race_row in race_results.iterrows():
        quali_row = quali_results.loc[quali_results["Abbreviation"] == race_row["Abbreviation"]]
        quali_position = (
            quali_row["Position"].iloc[0]
            if pd.notna(quali_row["Position"].iloc[0])
            else race_row.get("GridPosition", np.nan)
        )
        is_podium = race_row["Position"] in (1, 2, 3) if pd.notna(race_row["Position"]) else False

        rows.append(
            {
                "race_date": race_date,  # race identifier
                "race_number": race_number,  # race data
                "year": year,  # race data
                "driver_id": build_driver_id(
                    race_row["Abbreviation"], race_row["FirstName"], race_row["LastName"]
                ),  # driver identifier
                "team_id": map_team_id(race_row["TeamName"]),  # team identifier
                "circuit_id": circuit_location,  # circuit identifier
                "quali_position": quali_position,
                "grid_position": race_row["GridPosition"],
                "race_position": race_row["Position"],
                "points_scored": race_row["Points"],
                "laps_completed": race_row["Laps"],
                "status_raw": race_row["Status"],
                "is_podium": is_podium,
            }
        )

    return pd.DataFrame(rows)


def build_driver_id(abbreviation: str, first_name: str, last_name: str) -> str:
    if not abbreviation or not first_name or not last_name:
        warnings.warn("Abbreviation, FirstName o LastName mancanti, possibile driver_id incompleto")
    raw = f"{abbreviation}_{first_name}_{last_name}"
    return raw.strip().lower().replace(" ", "_")


def map_team_id(team_name: str) -> str:
    if team_name not in TEAM_ID_MAPPING:
        warnings.warn(
            f"TeamName '{team_name}' non presente in TEAM_ID_MAPPING — "
            f"fallback al nome grezzo, aggiungilo manualmente al dizionario"
        )
        return team_name  # fallback visibile, non silenzioso
    return TEAM_ID_MAPPING[team_name]
