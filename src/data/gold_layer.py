import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR, CIRCUIT_COORDS
from src.utils import setup_custom_logger
from .silver_layer import SilverLayer
from .history_builder import HistoryBuilder
from sklearn.linear_model import HuberRegressor

MIN_VALID_LAPS = 5
ROLLING_DNF_WINDOW = 10
CURRENT_FORM_RACES = 5
PENALTY_THRESHOLD = 3

"""
Possibili features da inserire in futuro (in caso manchino dati)
1. Average degradation rate, basato sulla history delle gare passate
2. Average positions gained, sempre calcolato sui risultati delle gare passate,
    normalizzata sulla base del margine di guadagno: posizioni_guadagnate / (N_piloti - grid_position)
3. Historical weather: andando a prendere negli ultimi, bho 10 anni, tramite i dati di fast f1, se la gara è stata bagnata; 
    costruire una history a parte che comprenda tutti i circuiti facendo gare_bagnate/tot_gare
"""
# TODO: aggiungere o modificare features che lavorino sulla storia molto recente (ultme 3 gare)
# bisogna fare in modo che il modello impari ad adattarsi in fretta ai cambiamenti
# current quali form?
# quali pace (best laptime)
# per entrambe le current form si può avere più colonne: 3,5,10 last races
# current form però del team
# cambiare decay_rate dei pesi? (più aggressivo?)
# un pò di dati sul circuito:
#  - is street circuit? (mapping a mano ma si fa)
#  - overtaking difficulty (non so come implementarlo)
#  - anche altre: elevation height o safety car rate, ma possono creare problemi troppe features sul circuito?
# alla fine sono uguali per tutti i piloti quindi bho

# TODO: verificare attendibilità dell'ottimizzazione


class GoldLayer:

    def __init__(self):
        # Path where all the parquet files with features are saved
        self.data_dir = Path(DATA_DIR) / "gold"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.log = setup_custom_logger("DataLoader")
        self.silver = SilverLayer()
        self.history_builder = HistoryBuilder(self.silver)

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

    def get_features(self, event: pd.DataFrame, year: int, race_number: int, session: int, force: bool = False):
        filename = f"{year}_{race_number}_{session}_features.parquet"
        if filename in os.listdir(self.data_dir) and not force:
            # Load from file
            gold_df = pd.read_parquet(self.data_dir / filename)
        else:
            # Compute from features and save
            gold_df = self.get_gp_features(event, year, race_number, session, force)
            gold_df.to_parquet(self.data_dir / filename, index=False)
        return gold_df

    # Just calculate features and return a dataframe
    def get_gp_features(self, event: pd.DataFrame, year: int, race_number: int, session: int, force: bool):
        # Load raw data
        silver = self.silver
        gold_df = pd.DataFrame()

        if event["EventFormat"].iloc[0] != "conventional" and session == 3:  # Sprint race 2024+
            practice_laps_df = silver.get_clean_laps(year, race_number, 1, force)  # FP1
            race_date = event["Session3Date"].iloc[0]
            quali_results_df = silver.get_clean_results(year, race_number, 2, force)
            race_results_df = silver.get_clean_results(year, race_number, 3, force)
        elif event["EventFormat"].iloc[0] != "conventional" and session == 4:  # Sprint race 2023 or before
            practice_laps_df = silver.get_clean_laps(year, race_number, 1, force)  # FP1
            race_date = event["Session4Date"].iloc[0]
            quali_results_df = silver.get_clean_results(year, race_number, 3, force)
            race_results_df = silver.get_clean_results(year, race_number, 4, force)
        elif event["EventFormat"].iloc[0] != "conventional" and session == 5:
            if year <= 2023:  # main race 2023 or before
                practice_laps_df = silver.get_clean_laps(year, race_number, 4, force)  # Sprint for data
                quali_results_df = silver.get_clean_results(year, race_number, 2, force)
            else:  # main race 2024+
                practice_laps_df = silver.get_clean_laps(year, race_number, 3, force)  # Sprint for data
                quali_results_df = silver.get_clean_results(year, race_number, 4, force)
            race_date = event["Session5Date"].iloc[0]
            race_results_df = silver.get_clean_results(year, race_number, 5, force)
        else:  # normal weekend
            fp1 = silver.get_clean_laps(year, race_number, 1, force)
            fp2 = silver.get_clean_laps(year, race_number, 2, force)
            fp3 = silver.get_clean_laps(year, race_number, 3, force)
            practice_laps_df = pd.concat([fp1, fp2, fp3], ignore_index=True)  # All FP sessions
            race_date = event["Session5Date"].iloc[0]
            quali_results_df = silver.get_clean_results(year, race_number, session - 1, force)
            race_results_df = silver.get_clean_results(year, race_number, session, force)

        # Normalize to tz-naive UTC scalar
        if hasattr(race_date, "tzinfo") and race_date.tzinfo is not None:
            race_date = race_date.tz_convert("UTC").tz_localize(None)

        circuit_location = event["Location"].iloc[0]
        driver_ids = race_results_df["driver_id"].unique()

        # leggi la history SOLO fino alla gara precedente (mai quella corrente)
        history_before = self.history_builder.get_history_up_to(race_date)

        # Update file containig the state with the new data
        self.history_builder.update_history(
            year, race_number, quali_results_df, race_results_df, circuit_location, race_date
        )

        # Get longest stint for each driver in practice laps
        race_sim_best_stint_df = self.find_race_sim_stint(practice_laps_df, race_results_df["Abbreviation"].unique())
        # race_sim_best_stint_df.to_parquet(f"{year}_{race_number}_{session}_stints.parquet", index=False)

        # Get weather data
        longitude, latitude = CIRCUIT_COORDS[circuit_location]
        future = race_date > pd.Timestamp.now(tz="UTC").tz_localize(None)
        weather_df = silver.get_clean_weather(year, race_number, session, latitude, longitude, race_date, future, force)

        gold_rows = []
        for driver_id in driver_ids:
            team_id = race_results_df.loc[race_results_df["driver_id"] == driver_id, "team_id"].iloc[0]
            regs_era = self._get_regs_era(year)
            abbreviation = race_results_df.loc[race_results_df["driver_id"] == driver_id, "Abbreviation"].iloc[0]

            quali_position = quali_results_df.loc[quali_results_df["driver_id"] == driver_id, "Position"].iloc[0]
            grid_position = self._get_grid_position(race_results_df, driver_id)

            row = {
                # Weekend's identifiers
                "race_date": race_date,
                "race_number": race_number,
                "year": year,
                "driver_id": driver_id,
                "team_id": team_id,
                "circuit_id": circuit_location,
                # Categoria A: solo dati del weekend corrente, nessuna history necessaria
                "degradation_rate": self._compute_degradation_rate(race_sim_best_stint_df, abbreviation),
                "simulated_race_pace": self._compute_race_pace_delta(race_sim_best_stint_df, abbreviation),
                "pace_imputed": False,  # updated in _compute_teammate_features
                "teammate_delta_pace": np.nan,  # computed at the end
                "grid_position": grid_position,
                "teammate_delta_q": np.nan,  # computed at the end
                # Categoria B: causali, derivate da fatti grezzi nella history (safe, non target-derived)
                "rolling_tech_dnf_rate": self._compute_rolling_dnf_rate(history_before, team_id),
                "car_age_proxy": self._compute_car_age_proxy(
                    history_before, driver_id, year, quali_position, grid_position
                ),
                "track_affinity_score": self._compute_track_affinity(history_before, driver_id, circuit_location),
                "current_form": self._compute_current_form(history_before, driver_id),
                # Categoria C: esterna
                "forecast_rain_probability": self._get_rain_probability(weather_df),
                "regulation_era": regs_era,
            }

            position = race_results_df.loc[race_results_df["driver_id"] == driver_id, "Position"].iloc[0]
            status = race_results_df.loc[race_results_df["driver_id"] == driver_id, "Status"].iloc[0]

            # Target
            if self._is_unclassified_dnf(status) or pd.isna(position):
                row["target"] = np.nan  # esclusa dal training, non "ultimo posto"
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

            # Compute teammate deltas
            driver_a_race_pace = gold_df.loc[gold_df["driver_id"] == driver_a, "simulated_race_pace"].iloc[0]
            driver_b_race_pace = gold_df.loc[gold_df["driver_id"] == driver_b, "simulated_race_pace"].iloc[0]
            self._compute_teammate_delta_pace(gold_df, driver_a, driver_b, driver_a_race_pace, driver_b_race_pace)

            driver_a_quali_time = self._get_driver_fastest_quali_time(quali_results_df, driver_a)
            driver_b_quali_time = self._get_driver_fastest_quali_time(quali_results_df, driver_b)
            self._compute_teammate_delta_q(gold_df, driver_a, driver_b, driver_a_quali_time, driver_b_quali_time)

    def find_race_sim_stint(self, practice_laps_df: pd.DataFrame, drivers: list) -> pd.DataFrame:
        """
        Iterare sui laps e salvare solo lo stint più lungo, sulle stesse gomme
        Practice_laps_df è un concat di tutti i lap delle sessioni di practice,
        a cui non viene assegnata una precedenza perchè il migliore è a prescindere il più lungo
        """
        stints = pd.DataFrame()
        for driver in drivers:
            driver_laps = practice_laps_df[practice_laps_df["Driver"] == driver]

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

    def _compute_degradation_rate(
        self, long_run_laps: pd.DataFrame, abbreviation: str, min_valid_laps: int = MIN_VALID_LAPS
    ) -> float:
        """
        Da capire come funziona l'Huber Regressor
        """
        if long_run_laps.empty:
            return np.nan
        long_run_laps = long_run_laps[long_run_laps["Driver"] == abbreviation]
        if len(long_run_laps) < min_valid_laps:
            return np.nan

        long_run_laps = long_run_laps.sort_values("LapNumber")
        X = (long_run_laps["LapNumber"] - long_run_laps["LapNumber"].min()).values.reshape(-1, 1)
        y = long_run_laps["LapTime"].values

        # Robust Regression con HuberRegressor
        # Configura l'epsilon (es. 1.35 è lo standard industry per l'efficienza al 95% su dati normali)
        model = HuberRegressor(epsilon=1.35, max_iter=1000)
        model.fit(X, y)

        # La pendenza rappresenta quanti secondi si perdono (o guadagnano) a ogni giro
        degradation_rate = model.coef_[0]

        return float(degradation_rate)

    def _compute_race_pace_delta(
        self, long_run_laps: pd.DataFrame, abbreviation: str, min_valid_laps: int = MIN_VALID_LAPS
    ) -> float:
        """
        Il delta rappresenta il gap (in secondi) dal
        Mediano dei tempi del long run, a cui sottraggo il tempo del leader (giro più veloce tra i long run di tutti)
        Facendo ciò vado a rendere il campo confrontabile tra diverse gare, piste e sessioni con diverse condizioni climatiche
        """
        if long_run_laps.empty:
            return np.nan
        driver_laps = long_run_laps[long_run_laps["Driver"] == abbreviation]
        if len(driver_laps) < min_valid_laps:
            return np.nan

        leader_time = long_run_laps["LapTime"].min()
        deltas = driver_laps["LapTime"] - leader_time
        return float(deltas.median())

    def _compute_teammate_delta_pace(
        self, gold_df: pd.DataFrame, driver_a: str, driver_b: str, driver_a_race_pace: float, driver_b_race_pace: float
    ):
        a_nan = np.isnan(driver_a_race_pace)
        b_nan = np.isnan(driver_b_race_pace)
        # if both drivers are missing the race_pace
        if a_nan and b_nan:
            gold_df.loc[gold_df["driver_id"] == driver_a, "teammate_delta_pace"] = np.nan
            gold_df.loc[gold_df["driver_id"] == driver_b, "teammate_delta_pace"] = np.nan
        # if one of the drivers is missing the race_pace
        # asymmetry for teammate delta: the one missing gets nan, the other gets 0
        elif a_nan:
            gold_df.loc[gold_df["driver_id"] == driver_a, "simulated_race_pace"] = driver_b_race_pace
            gold_df.loc[gold_df["driver_id"] == driver_a, "pace_imputed"] = True
            gold_df.loc[gold_df["driver_id"] == driver_a, "teammate_delta_pace"] = np.nan
            gold_df.loc[gold_df["driver_id"] == driver_b, "teammate_delta_pace"] = 0.0
        elif b_nan:
            gold_df.loc[gold_df["driver_id"] == driver_b, "simulated_race_pace"] = driver_a_race_pace
            gold_df.loc[gold_df["driver_id"] == driver_b, "pace_imputed"] = True
            gold_df.loc[gold_df["driver_id"] == driver_a, "teammate_delta_pace"] = 0.0
            gold_df.loc[gold_df["driver_id"] == driver_b, "teammate_delta_pace"] = np.nan
            print("Missing race pace for driver_b, copied from driver_a")
        # if both drivers have a race pace, compute the delta
        else:
            delta = driver_a_race_pace - driver_b_race_pace
            gold_df.loc[gold_df["driver_id"] == driver_a, "teammate_delta_pace"] = delta
            gold_df.loc[gold_df["driver_id"] == driver_b, "teammate_delta_pace"] = -delta

    def _get_grid_position(self, race_results_df: pd.DataFrame, driver_id: str) -> float:
        grid_position = race_results_df.loc[race_results_df["driver_id"] == driver_id, "GridPosition"].iloc[0]
        if grid_position <= 0 or np.isnan(grid_position):
            self.log.warning(f"Driver {driver_id} has grid position {grid_position}")
            # starting from pitlane probably
            grid_position = float(race_results_df["GridPosition"].max() + 2) if not np.isnan(grid_position) else np.nan

        return grid_position

    def _compute_teammate_delta_q(
        self,
        gold_df: pd.DataFrame,
        driver_a: str,
        driver_b: str,
        driver_a_quali_time: float,
        driver_b_quali_time: float,
    ):
        a_nan = np.isnan(driver_a_quali_time)
        b_nan = np.isnan(driver_b_quali_time)
        # if both drivers are missing the quali time
        if a_nan and b_nan:
            gold_df.loc[gold_df["driver_id"] == driver_a, "teammate_delta_q"] = np.nan
            gold_df.loc[gold_df["driver_id"] == driver_b, "teammate_delta_q"] = np.nan
        # if one of the drivers is missing the quali time
        elif a_nan:
            gold_df.loc[gold_df["driver_id"] == driver_a, "teammate_delta_q"] = np.nan
            gold_df.loc[gold_df["driver_id"] == driver_b, "teammate_delta_q"] = 0.0
        elif b_nan:
            gold_df.loc[gold_df["driver_id"] == driver_a, "teammate_delta_q"] = 0.0
            gold_df.loc[gold_df["driver_id"] == driver_b, "teammate_delta_q"] = np.nan
            print("Missing quali time for driver_b, copied from driver_a")
        # if both drivers have a quali time, compute the delta
        else:
            delta = driver_a_quali_time - driver_b_quali_time
            gold_df.loc[gold_df["driver_id"] == driver_a, "teammate_delta_q"] = delta
            gold_df.loc[gold_df["driver_id"] == driver_b, "teammate_delta_q"] = -delta

    def _get_driver_fastest_quali_time(self, quali_results_df: pd.DataFrame, driver_id: str):
        q1 = quali_results_df.loc[quali_results_df["driver_id"] == driver_id, "Q1"].dt.total_seconds()
        q2 = quali_results_df.loc[quali_results_df["driver_id"] == driver_id, "Q2"].dt.total_seconds()
        q3 = quali_results_df.loc[quali_results_df["driver_id"] == driver_id, "Q3"].dt.total_seconds()

        times = [q1, q2, q3]
        if all(np.isnan(t).all() for t in times):
            self.log.warning(f"Driver {driver_id} has no quali times in Q1/Q2/Q3")
        return np.nanmax(times)

    def _compute_rolling_dnf_rate(
        self, history_before: pd.DataFrame, team_id: str, window: int = ROLLING_DNF_WINDOW
    ) -> float:
        team_history = history_before.loc[history_before["team_id"] == team_id].tail(window * 2).copy()
        if team_history.empty:
            return np.nan

        is_dns = team_history["status_raw"] == "Did not start"
        is_retired = team_history["status_raw"] == "Retired"
        is_first_lap_crash = is_retired & (team_history["laps_completed"] <= 1)
        is_dnf = is_dns | (is_retired & ~is_first_lap_crash)

        return float(is_dnf.mean())

    def _compute_car_age_proxy(
        self,
        history_before: pd.DataFrame,
        driver_id: str,
        year: int,
        current_quali_position: float,
        current_grid_position: float,
    ) -> float:
        """
        Cumulative laps completed by the driver since the last detected
        component-reset event (or since the start of the season if none
        detected) — proxy for Car's mileage exposure.

        Causal safety: both grid_position and quali_position for a given
        race are known before that race starts (the official starting grid,
        including any steward-applied penalties, is published ahead of
        lights-out), so evaluating a reset for the CURRENT race is not
        leakage — it uses only information available pre-race.
        """
        season_races = history_before[
            (history_before["driver_id"] == driver_id) & (history_before["year"] == year)
        ].sort_values("race_number")

        cumulative_laps = 0
        for _, race in season_races.iterrows():
            if _is_reset_event(race["grid_position"], race["quali_position"]):
                cumulative_laps = 0
            cumulative_laps += race["laps_completed"]

        # Reset also applies to the race being predicted right now,
        # if its own grid shows a drop relative to quali (known pre-race)
        # TODO: not sure it is right, but consider adding a decrement factor instead of hard reset to 0
        if _is_reset_event(current_grid_position, current_quali_position):
            cumulative_laps = 0

        return float(cumulative_laps)

    def _compute_track_affinity(self, history_before: pd.DataFrame, driver_id: str, circuit_id: str) -> float:
        """Driver's average position at this circuit in past seasons."""
        driver_history = history_before[history_before["driver_id"] == driver_id]
        if driver_history.empty:
            return np.nan  # rookie o nessuno storico ancora disponibile

        career_avg = driver_history["race_position"].mean()
        circuit_avg = driver_history.loc[driver_history["circuit_id"] == circuit_id, "race_position"].mean()

        if np.isnan(career_avg) or np.isnan(circuit_avg):
            return np.nan

        return float(circuit_avg / career_avg)

    def _compute_current_form(self, history_before: pd.DataFrame, driver_id: str):
        driver_history = history_before[history_before["driver_id"] == driver_id]
        if driver_history.empty:
            return np.nan
        recent_form = driver_history.sort_values("race_date")["race_position"].tail(CURRENT_FORM_RACES).mean()
        return float(recent_form)

    def _get_rain_probability(self, weather_df: pd.DataFrame) -> float:
        """Rain probability for race day, pre-fetched and stored at Bronze time."""
        rain_probability = weather_df["rain_probability"].iloc[0]
        return float(rain_probability)

    def _is_unclassified_dnf(self, status: str) -> bool:
        """True if the driver did not finish and should be excluded from the target."""
        # Not including Disqualified because usually it happens after the race
        if status is None:
            return False  # fallback > no signal
        if status == "Retired" or status == "Accident" or status == "Withdrew" or status == "Did not start":
            return True
        else:
            return False

    def _get_regs_era(self, year: int):
        """Return the regulation era based on the year."""
        if year >= 2026:
            return 2026
        elif year >= 2022:
            return 2022
        else:
            return np.nan  # Should not happen due to assertion in build_features


@staticmethod
def _is_reset_event(grid_position: float, quali_position: float, penalty_threshold: int = PENALTY_THRESHOLD) -> bool:
    """
    Flags a race as a likely PU/component-change event based on the gap
    between the driver's official starting grid and their classified
    qualifying position.

    Coarse by design: cannot distinguish a PU penalty from a gearbox,
    unsafe-release, or driving-standards penalty. Accepted as a proxy
    given that PU/gearbox element penalties are the dominant cause of
    grid drops of this magnitude, consistent with the same "accept a
    coarse heuristic over a bespoke classifier" approach used for
    Rolling Tech DNF Rate.
    """
    if pd.isna(grid_position):
        return False  # e.g. DNS/withdrawn before the race — defensive, no signal

    if grid_position == 0:
        # Pit lane start (parc fermé infringement). Magnitude unknown,
        # but treated conservatively as a reset trigger.
        return True

    if grid_position == -1:
        # Known FastF1 data-quality bug for 2026 sessions (GH issue #871):
        # GridPosition returns -1.0 instead of the real value. Treated as
        # missing, not as a signal, to avoid false resets contaminating
        # the 2026 regulation_era data.
        return False

    if pd.isna(quali_position):
        return False  # can't compute a delta — defensive fallback

    delta = grid_position - quali_position
    return delta > penalty_threshold
