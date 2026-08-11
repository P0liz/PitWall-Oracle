import os
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import *
from src.utils import (
    setup_custom_logger,
    got_end_penalty,
    is_race_dnf,
    get_session_mapping,
    POST_RACE_EXCLUSION_STATUSES,
)
from .silver_layer import SilverLayer
from .history_builder import HistoryBuilder
from .feature_engineer import FeatureEngineering


class GoldLayer:

    def __init__(self):
        # Path where all the parquet files with features are saved
        self.data_dir = Path(DATA_DIR) / "gold"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.log = setup_custom_logger("DataLoader")
        self.silver = SilverLayer()
        self.feature_engineer = FeatureEngineering(self.silver)
        self.history_builder = HistoryBuilder(self.silver, self.feature_engineer)

    # Get features to parquet: one parquet for each weekend, with one row for each driver
    # The rows are the groups used by XGBRanker to calculate the ranking
    def build_features(self, year: int, race_number: int, force: bool = False):
        """
        Main function to call for the data_loader
        """
        assert year >= 2023, "Year not supported: must be >= 2023"
        assert 1 <= race_number <= 24, f"Race number {race_number} does not exist: max 24 races"
        event = self.silver.get_clean_event_metadata(year, race_number, force)
        results = []
        self.log.info(f"Building features for {year} Grand Prix #{race_number}...")

        if event["EventFormat"].iloc[0] != "conventional":  # Sprint race
            sprint_race_session = get_session_mapping(year, False, "sr", "race")
            results.append(self.get_features(event, year, race_number, session=sprint_race_session, force=force))
        results.append(self.get_features(event, year, race_number, session=5, force=force))

        return results

    # Does not use race_results, only for predictions
    def build_prediction_features(self, year: int, race_number: int, session: int, force: bool = False):
        assert year >= 2024, "Predictions only on 2024+ seasons"
        assert 1 <= race_number <= 24, f"Race number {race_number} does not exist: max 24 races"
        assert session in [3, 5], "Predictions only on race sessions"
        event = self.silver.get_clean_event_metadata(year, race_number, force)

        results = self.get_features(event, year, race_number, session, force, prediction_mode=True)
        results.to_parquet(self.data_dir / f"latest_race_pred.parquet", index=False)
        return results

    def get_features(
        self,
        event: pd.DataFrame,
        year: int,
        race_number: int,
        session: int,
        force: bool = False,
        prediction_mode: bool = False,
    ):
        filename = f"{year}_{race_number}_{session}_features.parquet"
        if filename in os.listdir(self.data_dir) and not force:
            # Load from file
            gold_df = pd.read_parquet(self.data_dir / filename)
            required_metadata = {"dnf_target", "raw_driver_id", "raw_team_id", "session_type"}
            missing_metadata = required_metadata - set(gold_df.columns)
            if missing_metadata:
                self.log.info(f"Rigenerazione di {filename}: colonne mancanti {sorted(missing_metadata)}.")
                gold_df = self.get_gp_features(event, year, race_number, session, force, prediction_mode)
                gold_df.to_parquet(self.data_dir / filename, index=False)
        else:
            # Compute from features and save
            gold_df = self.get_gp_features(event, year, race_number, session, force, prediction_mode)
            gold_df.to_parquet(self.data_dir / filename, index=False)
        return gold_df

    # Just calculate features and return a dataframe
    def get_gp_features(
        self, event: pd.DataFrame, year: int, race_number: int, session: int, force: bool, prediction_mode: bool
    ):
        # Load raw data
        silver = self.silver
        gold_df = pd.DataFrame()

        is_conventional = event["EventFormat"].iloc[0] == "conventional"
        race_type = "sr" if session <= 4 else "gp"
        practice_laps_session = get_session_mapping(year, is_conventional, race_type, "practice_laps")
        practice_laps_df = silver.get_clean_laps(year, race_number, practice_laps_session, force)
        practice_results_session = get_session_mapping(year, is_conventional, race_type, "practice_results")
        practice_results_laps_df = silver.get_clean_laps(year, race_number, practice_results_session, force)
        quali_session = get_session_mapping(year, is_conventional, race_type, "quali")
        quali_results_df = silver.get_clean_results(year, race_number, quali_session, force)
        race_session = get_session_mapping(year, is_conventional, race_type, "race")
        race_date = event[f"Session{race_session}Date"].iloc[0]
        race_results_df = silver.get_clean_results(year, race_number, race_session, force)
        raw_race_laps_df = (
            silver.get_untouched_laps(year, race_number, race_session, force) if not prediction_mode else pd.DataFrame()
        )
        clean_race_laps_df = (
            silver.get_clean_laps(year, race_number, race_session, force) if not prediction_mode else pd.DataFrame()
        )

        # Normalize to tz-naive UTC scalar
        if hasattr(race_date, "tzinfo") and race_date.tzinfo is not None:
            race_date = race_date.tz_convert("UTC").tz_localize(None)

        circuit_location = event["Location"].iloc[0]
        if prediction_mode:
            driver_ids = quali_results_df["driver_id"].unique()
        else:
            driver_ids = np.intersect1d(quali_results_df["driver_id"].unique(), race_results_df["driver_id"].unique())

        # leggi la history SOLO fino alla gara precedente (mai quella corrente)
        history_before = self.history_builder.get_history_up_to(race_date)

        # Update file containig the state with the new data
        if not prediction_mode:
            self.history_builder.update_history(
                year,
                race_number,
                session,
                quali_results_df,
                race_results_df,
                raw_race_laps_df,
                clean_race_laps_df,
                circuit_location,
                race_date,
                force=force,
            )

        # Get longest stint for each driver in practice laps
        race_sim_best_stint_df = self.find_race_sim_stint(practice_laps_df, quali_results_df["Abbreviation"].unique())
        # race_sim_best_stint_df.to_parquet(f"{year}_{race_number}_{session}_stints.parquet", index=False)

        # Get practice rankings
        practice_rankings = self.compute_practice_rankings(
            practice_results_laps_df, quali_results_df["Abbreviation"].unique()
        )
        # Get weather data
        rain_probability = self.feature_engineer.get_rain_probability(
            year, race_number, session, circuit_location, race_date, force
        )

        # Get circuit data
        overtaking_difficulty = self.feature_engineer.compute_overtaking_difficulty(history_before, circuit_location)

        gold_rows = []
        for driver_id in driver_ids:
            team_id = quali_results_df.loc[quali_results_df["driver_id"] == driver_id, "team_id"].iloc[0]
            regs_era = self.feature_engineer.get_regs_era(year)
            abbreviation = quali_results_df.loc[quali_results_df["driver_id"] == driver_id, "Abbreviation"].iloc[0]
            team = quali_results_df.loc[quali_results_df["driver_id"] == driver_id, "TeamName"].iloc[0]

            quali_position = quali_results_df.loc[quali_results_df["driver_id"] == driver_id, "Position"].iloc[0]
            grid_position = (
                self.feature_engineer.get_grid_position(race_results_df, driver_id)
                if not prediction_mode
                else CUSTOM_GRID[driver_id]
            )

            row = {
                # Weekend's identifiers
                "race_date": race_date,
                "race_number": race_number,
                "year": year,
                "session_type": "race" if session == 5 else "sprint",
                "raw_driver_id": driver_id,
                "raw_team_id": team_id,
                "driver_id": driver_id,
                "team_id": team_id,
                "circuit_id": circuit_location,
                # Categoria A: solo dati del weekend corrente, nessuna history necessaria
                "degradation_rate": self.feature_engineer.compute_degradation_rate(
                    race_sim_best_stint_df, abbreviation
                ),
                "teammate_delta_deg": np.nan,
                "mean_race_pace": self.feature_engineer.compute_race_pace(race_sim_best_stint_df, abbreviation),
                "teammate_delta_pace": np.nan,
                "teammate_recent_race_h2h": np.nan,
                "late_stint_dropoff": self.feature_engineer.compute_late_stint_dropoff(
                    race_sim_best_stint_df, abbreviation
                ),
                "team_race_pace": self.feature_engineer.compute_team_race_pace(race_sim_best_stint_df, team),
                "quali_pace": self.feature_engineer.compute_quali_pace(quali_results_df, driver_id),
                "teammate_delta_quali": np.nan,
                "practice_position": self.feature_engineer.get_practice_position(practice_rankings, abbreviation),
                "grid_position": grid_position,
                "teammate_delta_grid_position": np.nan,
                "teammate_recent_quali_h2h": np.nan,
                # Categoria B: causali, derivate da fatti grezzi nella history (safe, non target-derived)
                "team_dnf_rate": self.feature_engineer.compute_team_dnf_rate(history_before, year, team_id),
                "driver_dnf_rate": self.feature_engineer.compute_driver_dnf_rate(history_before, year, driver_id),
                "car_age_proxy": self.feature_engineer.compute_car_age_proxy(
                    history_before, driver_id, year, quali_position, grid_position
                ),
                "driver_track_affinity": self.feature_engineer.compute_driver_track_affinity(
                    history_before, driver_id, circuit_location
                ),
                "team_track_affinity": self.feature_engineer.compute_team_track_affinity(
                    history_before, team_id, circuit_location
                ),
                "driver_current_form": self.feature_engineer.compute_driver_current_form(history_before, driver_id),
                "team_current_form": self.feature_engineer.compute_team_current_form(history_before, team_id),
                "quali_current_form": self.feature_engineer.compute_quali_current_form(history_before, driver_id),
                "teammate_delta_race_form": np.nan,
                "driver_consistency": self.feature_engineer.compute_driver_consistency(history_before, driver_id),
                "teammate_delta_consistency": np.nan,
                "avg_positions_gained": self.feature_engineer.compute_avg_positions_gained(history_before, driver_id),
                "teammate_delta_pos_gained": np.nan,
                "lap1_avg_pos_gained": self.feature_engineer.compute_lap1_avg_positions_gained(
                    history_before, driver_id
                ),
                "teammate_delta_lap1_pos_gained": np.nan,
                "driver_recent_race_pace": self.feature_engineer.compute_recent_race_pace(history_before, driver_id),
                "teammate_delta_recent_pace": np.nan,
                "overtaking_difficulty": overtaking_difficulty,
                "team_development": self.feature_engineer.compute_team_development_trend(history_before, team_id),
                "team_pit_execution_index": self.feature_engineer.compute_team_pit_execution_index(
                    history_before, team_id
                ),
                "team_strategy_aggressiveness_score": (
                    self.feature_engineer.compute_team_strategy_aggressiveness_score(history_before, team_id)
                ),
                "wet_affinity": self.feature_engineer.compute_wet_affinity(history_before, driver_id),
                "teammate_delta_wet_affinity": np.nan,
                # Target DNF all-cause
                "dnf_target": np.nan,
                # Categoria C: esterna
                "forecast_rain_probability": rain_probability,
                "is_street_circuit": IS_STREET_CIRCUIT[circuit_location],
                "regulation_era": regs_era,
            }

            if not prediction_mode:
                result = race_results_df.loc[race_results_df["driver_id"] == driver_id, ["Position", "Status"]]

                if len(result) != 1:
                    raise ValueError(f"Risultato assente o duplicato: {driver_id}")

                position = result["Position"].iat[0]
                status = result["Status"].iat[0]

                normalized_status = str(status).strip() if pd.notna(status) and str(status).strip() else None

                is_dnf = is_race_dnf(normalized_status)
                row["dnf_target"] = np.nan if normalized_status is None else int(is_dnf)

                # Target
                if (
                    is_dnf
                    or pd.isna(position)
                    or got_end_penalty(abbreviation, raw_race_laps_df, race_results_df)
                    or normalized_status in POST_RACE_EXCLUSION_STATUSES
                ):
                    row["target"] = np.nan  # esclusa dal training
                else:
                    n_drivers = race_results_df["driver_id"].nunique()
                    row["target"] = n_drivers - position + 1

            gold_rows.append(row)

        gold_df = pd.DataFrame(gold_rows)
        self._compute_teammate_features(gold_df, history_before, quali_results_df)

        return gold_df

    def _compute_teammate_features(
        self, gold_df: pd.DataFrame, history_before: pd.DataFrame, quali_results_df: pd.DataFrame
    ):
        """Compute teammate-based features using the current gold_df."""
        # Group by team_id
        for team_id in gold_df["team_id"].unique():
            team_drivers = gold_df[gold_df["team_id"] == team_id]
            if len(team_drivers) != 2:
                self.log.warning(
                    f"Team {team_id} does not have exactly 2 drivers, skipping teammate features computation"
                )
                continue
            driver_a = team_drivers.iloc[0]["driver_id"]  # First driver
            driver_b = team_drivers.iloc[1]["driver_id"]  # Second driver

            self.feature_engineer.quali_results = quali_results_df

            # Compute teammate deltas
            self.feature_engineer.compute_teammate_deltas(gold_df, history_before, driver_a, driver_b)

    def find_race_sim_stint(self, practice_laps: pd.DataFrame, drivers: list) -> pd.DataFrame:
        """Find longest stint for each driver, in the given laps (should be laps from only on session)"""
        stints = pd.DataFrame()
        for driver in drivers:
            driver_laps = practice_laps[practice_laps["Driver"] == driver]

            longest_stint = pd.DataFrame()
            current_stint = pd.DataFrame()

            for i in range(len(driver_laps)):
                if current_stint.empty:
                    current_stint = driver_laps.iloc[[i]]
                else:
                    last_row = current_stint.iloc[-1]  # Get the last row of the current stint
                    current_row = driver_laps.iloc[i]

                    # If the compound is the same and the lap number is consecutive, add to current stint
                    if (
                        current_row["Compound"] == last_row["Compound"]
                        and current_row["LapNumber"] == last_row["LapNumber"] + 1
                    ):
                        current_stint = pd.concat([current_stint, current_row.to_frame().T])
                    # If the compound changes or the lap number is not consecutive
                    else:
                        # Check if the current stint is the longest
                        if len(current_stint) > len(longest_stint):
                            longest_stint = current_stint
                        # Prepare for the next stint
                        current_stint = driver_laps.iloc[[i]]

                # Check at the end of the loop
                if len(current_stint) > len(longest_stint):
                    longest_stint = current_stint

            stints = pd.concat([stints, longest_stint], ignore_index=True)
        return stints

    def compute_practice_rankings(self, practice_laps: pd.DataFrame, drivers: list) -> pd.DataFrame:
        """Compute practice rankings based on fastest lap times for each driver."""
        rankings = []
        for driver in drivers:
            driver_laps = practice_laps[practice_laps["Driver"] == driver]
            if not driver_laps.empty:
                fastest_lap = driver_laps.loc[driver_laps["LapTime"].idxmin()]
                rankings.append({"Driver": driver, "FastestLapTime": fastest_lap["LapTime"]})
        return pd.DataFrame(rankings).sort_values("FastestLapTime").reset_index(drop=True)
