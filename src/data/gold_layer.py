import os
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import *
from src.utils import setup_custom_logger, got_end_penalty
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
        assert year >= 2022, "Year not supported: must be >= 2022"
        assert (race_number <= 24) & (race_number >= 1), "Race number {id} does not exist: max 24 races"
        event = self.silver.get_clean_event_metadata(year, race_number, force)
        results = []
        self.log.info(f"Building features for {year} Grand Prix #{race_number}...")

        if event["EventFormat"].iloc[0] != "conventional":  # Sprint race
            sprint_race_session = 3 if year >= 2024 else 4
            results.append(self.get_features(event, year, race_number, session=sprint_race_session, force=force))
        results.append(self.get_features(event, year, race_number, session=5, force=force))

        return results

    # Does not use race_results, only for predictions
    def build_prediction_features(self, year: int, race_number: int, session: int, force: bool = False):
        """
        driver_list: DataFrame con colonne [driver_id, team_id, Abbreviation, GridPosition]
        """
        assert year >= 2022
        assert 1 <= race_number <= 24
        assert (session == 3) or (session == 5), "Predictions only on race sessions"
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
            if "technical_dnf_target" not in gold_df.columns:
                self.log.info(f"Rigenerazione di {filename}: target DNF assente nel parquet Gold.")
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

        if event["EventFormat"].iloc[0] != "conventional" and session == 3 and year >= 2024:  # Sprint race 2024+
            practice_laps_df = [silver.get_clean_laps(year, race_number, 1, force)]  # FP1
            race_date = event["Session3Date"].iloc[0]
            quali_results_df = silver.get_clean_results(year, race_number, 2, force)
            race_results_df = (
                silver.get_clean_results(year, race_number, 3, force) if not prediction_mode else pd.DataFrame()
            )
            race_laps_df = (
                silver.get_untouched_laps(year, race_number, 3, force) if not prediction_mode else pd.DataFrame()
            )
        elif event["EventFormat"].iloc[0] != "conventional" and session == 4 and year == 2023:  # Sprint race 2023
            practice_laps_df = [silver.get_clean_laps(year, race_number, 1, force)]  # FP1
            race_date = event["Session4Date"].iloc[0]
            quali_results_df = silver.get_clean_results(year, race_number, 3, force)
            race_results_df = (
                silver.get_clean_results(year, race_number, 4, force) if not prediction_mode else pd.DataFrame()
            )
            race_laps_df = (
                silver.get_untouched_laps(year, race_number, 4, force) if not prediction_mode else pd.DataFrame()
            )
        elif event["EventFormat"].iloc[0] != "conventional" and session == 5:
            if year == 2023:  # main race 2023
                practice_laps_df = [silver.get_clean_laps(year, race_number, 4, force)]  # Sprint for data
                quali_results_df = silver.get_clean_results(year, race_number, 2, force)
            elif year >= 2024:  # main race 2024+
                practice_laps_df = [silver.get_clean_laps(year, race_number, 3, force)]  # Sprint for data
                quali_results_df = silver.get_clean_results(year, race_number, 4, force)
            else:
                self.log.error("Years before 2022 are not correctly implemented")
                raise ValueError("Years before 2022 are not correctly implemented")
            race_date = event["Session5Date"].iloc[0]
            race_results_df = (
                silver.get_clean_results(year, race_number, 5, force) if not prediction_mode else pd.DataFrame()
            )
            race_laps_df = (
                silver.get_untouched_laps(year, race_number, 5, force) if not prediction_mode else pd.DataFrame()
            )
        else:  # normal weekend
            fp1 = silver.get_clean_laps(year, race_number, 1, force)
            fp2 = silver.get_clean_laps(year, race_number, 2, force)
            fp3 = silver.get_clean_laps(year, race_number, 3, force)
            practice_laps_df = [fp1, fp2, fp3]  # All FP sessions
            race_date = event["Session5Date"].iloc[0]
            quali_results_df = silver.get_clean_results(year, race_number, session - 1, force)
            race_results_df = (
                silver.get_clean_results(year, race_number, session, force) if not prediction_mode else pd.DataFrame()
            )
            race_laps_df = (
                silver.get_untouched_laps(year, race_number, session, force) if not prediction_mode else pd.DataFrame()
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
                year, race_number, session, quali_results_df, race_results_df, race_laps_df, circuit_location, race_date
            )

        # Get longest stint for each driver in practice laps
        race_sim_best_stint_df = self.find_race_sim_stint(practice_laps_df, quali_results_df["Abbreviation"].unique())
        # race_sim_best_stint_df.to_parquet(f"{year}_{race_number}_{session}_stints.parquet", index=False)

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
                "driver_id": driver_id,
                "team_id": team_id,
                "circuit_id": circuit_location,
                # Categoria A: solo dati del weekend corrente, nessuna history necessaria
                "degradation_rate": self.feature_engineer.compute_degradation_rate(
                    race_sim_best_stint_df, abbreviation
                ),
                "teammate_delta_deg": np.nan,  # computed at the end
                "simulated_race_pace": self.feature_engineer.compute_race_pace(race_sim_best_stint_df, abbreviation),
                "teammate_delta_pace": np.nan,  # computed at the end
                "quali_pace": self.feature_engineer.compute_quali_pace(quali_results_df, driver_id),
                "grid_position": grid_position,
                "teammate_delta_quali": np.nan,  # computed at the end
                # Categoria B: causali, derivate da fatti grezzi nella history (safe, non target-derived)
                "rolling_tech_dnf_rate": self.feature_engineer.compute_rolling_dnf_rate(history_before, team_id),
                "car_age_proxy": self.feature_engineer.compute_car_age_proxy(
                    history_before, driver_id, year, quali_position, grid_position
                ),
                "track_affinity_score": self.feature_engineer.compute_track_affinity(
                    history_before, driver_id, circuit_location
                ),
                "race_current_form": self.feature_engineer.compute_race_current_form(history_before, driver_id),
                "teammate_delta_race_form": np.nan,
                # "quali_current_form": self.compute_quali_current_form(history_before, driver_id),
                "team_current_form": self.feature_engineer.compute_team_current_form(history_before, team_id),
                "driver_consistency": self.feature_engineer.compute_driver_consistency(history_before, driver_id),
                "avg_positions_gained": self.feature_engineer.compute_avg_positions_gained(history_before, driver_id),
                "teammate_delta_pos_gained": np.nan,
                "lap1_avg_pos_gained": self.feature_engineer.compute_lap1_avg_positions_gained(
                    history_before, driver_id
                ),
                "teammate_delta_lap1_pos_gained": np.nan,
                "overtaking_difficulty": overtaking_difficulty,
                "team_development": self.feature_engineer.compute_team_development_trend(history_before, team_id, year),
                "wet_affinity": self.feature_engineer.compute_wet_affinity(history_before, driver_id),
                "teammate_delta_wet_affinity": np.nan,
                # Target del modello DNF. In prediction mode resta ignoto.
                "technical_dnf_target": np.nan,
                # Categoria C: esterna
                "forecast_rain_probability": rain_probability,
                "is_street_circuit": IS_STREET_CIRCUIT[circuit_location],
                "regulation_era": regs_era,
            }

            if not prediction_mode:
                position = race_results_df.loc[race_results_df["driver_id"] == driver_id, "Position"].iloc[0]
                status = race_results_df.loc[race_results_df["driver_id"] == driver_id, "Status"].iloc[0]
                is_dnf = self.feature_engineer.is_unclassified_dnf(status)
                row["technical_dnf_target"] = int(is_dnf)  # True = 1 e False = 0

                # Target
                if is_dnf or pd.isna(position) or got_end_penalty(abbreviation, race_laps_df, race_results_df):
                    row["target"] = np.nan  # esclusa dal training
                else:
                    n_drivers = race_results_df["driver_id"].nunique()
                    row["target"] = n_drivers - position + 1

            gold_rows.append(row)

        gold_df = pd.DataFrame(gold_rows)
        self._compute_teammate_features(gold_df, quali_results_df)

        return gold_df

    def _compute_teammate_features(self, gold_df: pd.DataFrame, quali_results_df: pd.DataFrame):
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
            self.feature_engineer.compute_teammate_deltas(gold_df, driver_a, driver_b)

    def find_race_sim_stint(
        self, practice_sessions: list, drivers: list, min_valid_laps: int = MIN_VALID_LAPS
    ) -> pd.DataFrame:
        """Find longest stint for each driver, giving preference to FP3, than FP2 and last FP1"""
        stints = pd.DataFrame()
        for driver in drivers:
            fp_stints = {}
            for idx, fp in enumerate(practice_sessions):
                driver_laps = fp[fp["Driver"] == driver]

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
                fp_stints[idx + 1] = longest_stint

            # Choosing preferred session (highest index = latest FP)
            best_stint = pd.DataFrame()
            for i in range(len(practice_sessions), 0, -1):
                if not fp_stints[i].empty and len(fp_stints[i]) >= min_valid_laps:
                    best_stint = fp_stints[i]
                    break
            stints = pd.concat([stints, best_stint], ignore_index=True)
        return stints
