import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from src.config import DATA_DIR, TEAM_ID_MAPPING
from src.utils import setup_custom_logger, got_end_penalty

log = setup_custom_logger("DataLoader")


class HistoryBuilder:
    def __init__(self, silver):
        self.history_path = Path(DATA_DIR) / "gold" / "driver_team_history.parquet"
        self.silver = silver
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

    def update_history(
        self,
        year: int,
        race_number: str,
        quali_results: pd.DataFrame,
        race_results: pd.DataFrame,
        race_laps: pd.DataFrame,
        circuit_location: str,
        race_date,
    ):
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

        new_history_rows = self.build_history_rows(
            quali_results=quali_results,
            race_results=race_results,
            race_laps=race_laps,
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

    def _is_unclassified_dnf(self, status: str) -> bool:
        """True if the driver did not finish and should be excluded from the target."""
        # Not including Disqualified because usually it happens after the race
        if status is None:
            return False  # fallback > no signal
        if (
            status == "Retired"
            or status == "Accident"
            or status == "Withdrew"
            or status == "Did not start"
            or status == "Collision damage"
        ):
            return True
        else:
            return False

    def build_history_rows(
        self,
        quali_results: pd.DataFrame,
        race_results: pd.DataFrame,
        race_laps: pd.DataFrame,
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
                self._is_unclassified_dnf(status)
                or pd.isna(position)
                or got_end_penalty(quali_results["Abbreviation"].iloc[0], race_laps, race_results)
            ):
                position = np.nan  # esclusa, non "ultimo posto"

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
                    "grid_position": grid_position,
                    "race_position": position,
                    "points_scored": race_row["Points"],
                    "laps_completed": race_row["Laps"],
                    "status_raw": status,
                    "is_podium": is_podium,
                }
            )

        return pd.DataFrame(rows)


# Helpers
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
