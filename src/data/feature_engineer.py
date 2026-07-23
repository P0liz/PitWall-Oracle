import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from src.config import *
from src.utils import get_driver_fastest_quali_time, setup_custom_logger

"""
Possibili features da inserire in futuro (in caso manchino dati)
 - Dati sul circuito: elevation height o safety car rate,
 - Average degradation rate, basato sulla history delle gare passate
 - Historical weather: andando a prendere negli ultimi, bho 10 anni, tramite i dati di fast f1, se la gara è stata bagnata; 
    costruire una history a parte che comprenda tutti i circuiti facendo gare_bagnate/tot_gare
"""

# TODO: alcune idee per feature da inserire per il dnf_model
# Inserite tutte nei parquet gold, ma poi quelle che non centrano con il ranking le escluderei
# e terrei solo per per trainare il dnf regressor
# Magari fare anche ricerca online per capire se esistono già modelli simili
""" 
DNF tecnico
Feature promettenti e disponibili prima della gara:
DNF tecnici del team nelle ultime 5/10 gare;
tasso DNF con decadimento temporale;
numero di gare consecutive senza guasti;
età/chilometraggio di motore, cambio, turbo e componenti ibridi;
sostituzioni di componenti nel weekend;
penalità in griglia dovute a sostituzioni;
problemi tecnici osservati nelle prove libere;
guasti recenti della vettura del compagno;
temperatura prevista;
altitudine;
lunghezza della gara;
stress del circuito su motore, cambio e freni;
nuova regulation era o aggiornamento importante della vettura.

DNF per incidente
incidenti del pilota nelle ultime 5/10 gare;
incident rate al primo giro;
posizioni perse/guadagnate mediamente al primo giro;
esperienza del pilota;
probabilità di pioggia;
circuito cittadino;
difficoltà di sorpasso;
posizione in griglia;
densità della zona di partenza;
storico di incidenti e Safety Car del circuito;
affinità del pilota sul bagnato.
"""

log = setup_custom_logger("DataLoader")


class FeatureEngineering:
    def __init__(self, silver):
        self.log = log
        self.silver = silver
        self.quali_results = None

    def compute_degradation_rate(
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

    def compute_race_pace(
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

    def compute_quali_pace(self, quali_results_df: pd.DataFrame, driver_id: str):
        """
        Delta qualifiche: quanto è veloce il pilota rispetto al miglior tempo di qualifica
        in quella sessione (non è un gap dal leader, ma dal miglior tempo della sessione)
        """
        driver_time = get_driver_fastest_quali_time(quali_results_df, driver_id)
        session_best_time = quali_results_df[["Q1", "Q2", "Q3"]].min().min().total_seconds()
        return float(driver_time - session_best_time)

    def compute_teammate_deltas(self, gold_df: pd.DataFrame, driver_a: str, driver_b: str):
        """Compute the difference between the two teammates for each feature in the list."""
        teammate_features = {
            "degradation_rate": "deg",
            "simulated_race_pace": "pace",
            "quali_pace": "quali",
            "race_current_form": "race_form",
            "avg_positions_gained": "pos_gained",
            "lap1_avg_pos_gained": "lap1_pos_gained",
            "wet_affinity": "wet_affinity",
        }
        for feature in teammate_features.keys():
            teammate_f = teammate_features[feature]
            if feature == "quali_pace":
                driver_a_feature = get_driver_fastest_quali_time(self.quali_results, driver_a)
                driver_b_feature = get_driver_fastest_quali_time(self.quali_results, driver_b)
            else:
                driver_a_feature = gold_df.loc[gold_df["driver_id"] == driver_a, f"{feature}"]
                driver_b_feature = gold_df.loc[gold_df["driver_id"] == driver_b, f"{feature}"]

                if not driver_a_feature.empty:
                    driver_a_feature = driver_a_feature.iloc[0]
                else:
                    driver_a_feature = np.nan
                if not driver_b_feature.empty:
                    driver_b_feature = driver_b_feature.iloc[0]
                else:
                    driver_b_feature = np.nan
            a_nan = np.isnan(driver_a_feature)
            b_nan = np.isnan(driver_b_feature)
            if a_nan and b_nan:
                gold_df.loc[gold_df["driver_id"] == driver_a, f"teammate_delta_{teammate_f}"] = np.nan
                gold_df.loc[gold_df["driver_id"] == driver_b, f"teammate_delta_{teammate_f}"] = np.nan
            # no feature_imputed column since it proved to be mostly useless
            elif a_nan:
                # if feature != "quali_pace":
                gold_df.loc[gold_df["driver_id"] == driver_a, f"{feature}"] = driver_b_feature
                gold_df.loc[gold_df["driver_id"] == driver_a, f"teammate_delta_{teammate_f}"] = np.nan
                gold_df.loc[gold_df["driver_id"] == driver_b, f"teammate_delta_{teammate_f}"] = 0.0
            elif b_nan:
                # if feature != "quali_pace":
                gold_df.loc[gold_df["driver_id"] == driver_b, f"{feature}"] = driver_a_feature
                gold_df.loc[gold_df["driver_id"] == driver_a, f"teammate_delta_{teammate_f}"] = 0.0
                gold_df.loc[gold_df["driver_id"] == driver_b, f"teammate_delta_{teammate_f}"] = np.nan
            else:
                delta = driver_a_feature - driver_b_feature
                gold_df.loc[gold_df["driver_id"] == driver_a, f"teammate_delta_{teammate_f}"] = delta
                gold_df.loc[gold_df["driver_id"] == driver_b, f"teammate_delta_{teammate_f}"] = -delta

    def get_grid_position(self, race_results_df: pd.DataFrame, driver_id: str) -> float:
        grid_position = race_results_df.loc[race_results_df["driver_id"] == driver_id, "GridPosition"].iloc[0]
        if grid_position <= 0 or np.isnan(grid_position):
            self.log.warning(f"Driver {driver_id} has grid position {grid_position}")
            # starting from pitlane probably
            grid_position = float(race_results_df["GridPosition"].max() + 2) if not np.isnan(grid_position) else np.nan

        return grid_position

    def compute_rolling_dnf_rate(
        self, history_before: pd.DataFrame, team_id: str, window: int = ROLLING_DNF_WINDOW
    ) -> float:
        team_history = history_before.loc[history_before["team_id"] == team_id].tail(window * 2).copy()
        if team_history.empty:
            return np.nan

        is_dns = team_history["status_raw"] == "Did not start"
        is_retired = team_history["status_raw"] == "Retired"
        is_first_lap_crash = is_retired & (team_history["laps_completed"] <= 1)
        is_dnf = is_dns | (is_retired & ~is_first_lap_crash)
        # rate = is_dnf.mean()
        rate = is_dnf.ewm(span=window * 2).mean().iloc[-1]
        return float(rate)

    def compute_car_age_proxy(
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
            if _is_reset_event(race["grid_position"], race["quali_position"], race["status_raw"]):
                cumulative_laps = 0
            cumulative_laps += race["laps_completed"]

        # Reset also applies to the race being predicted right now,
        # if its own grid shows a drop relative to quali (known pre-race)
        # TODO: not sure it is right, but consider adding a decrement factor instead of hard reset to 0
        if _is_reset_event(current_grid_position, current_quali_position, ""):
            cumulative_laps = 0

        return float(cumulative_laps)

    def compute_track_affinity(self, history_before: pd.DataFrame, driver_id: str, circuit_id: str) -> float:
        """Driver's average position at this circuit in past seasons."""
        driver_history = history_before[history_before["driver_id"] == driver_id]
        if driver_history.empty:
            return np.nan  # rookie o nessuno storico ancora disponibile

        career_avg = driver_history["race_position"].mean()
        circuit_avg = driver_history.loc[driver_history["circuit_id"] == circuit_id, "race_position"].mean()

        if np.isnan(career_avg) or np.isnan(circuit_avg):
            return np.nan

        # Using a difference instead of a division cause its more stable
        # Positive means the driver is worse at that track than average
        return float(circuit_avg - career_avg)

    def compute_race_current_form(self, history_before: pd.DataFrame, driver_id: str, races: int = CURRENT_FORM_RACES):
        driver_history = history_before[history_before["driver_id"] == driver_id]
        if driver_history.empty:
            return np.nan
        # recent_form = driver_history.sort_values("race_date")["race_position"].tail(races).mean()
        recent_form = (
            driver_history.sort_values("race_date")["race_position"].tail(races).ewm(span=races).mean().iloc[-1]
        )
        return float(recent_form)

    def compute_quali_current_form(self, history_before: pd.DataFrame, driver_id: str, races: int = CURRENT_FORM_RACES):
        driver_history = history_before[history_before["driver_id"] == driver_id]
        if driver_history.empty:
            return np.nan
        # recent_form = driver_history.sort_values("race_date")["quali_position"].tail(races).mean()
        recent_form = (
            driver_history.sort_values("race_date")["quali_position"].tail(races).ewm(span=races).mean().iloc[-1]
        )
        return float(recent_form)

    def compute_team_current_form(self, history_before: pd.DataFrame, team_id: str, races: int = CURRENT_FORM_RACES):
        """Team's average position in recent races."""
        team_history = history_before[history_before["team_id"] == team_id]
        if team_history.empty:
            return np.nan

        team_avg_position = (
            team_history.groupby("race_date")["race_position"]
            .mean()  # mean position between the two drivers
            .tail(races)
            .ewm(span=races)
            .mean()  # mean between races
            .iloc[-1]
        )

        return float(team_avg_position)

    def compute_driver_consistency(self, history_before: pd.DataFrame, driver_id: str, races: int = CONSISTENCY_WINDOW):
        """Standard deviation of the driver's recent positions."""
        driver_history = history_before[history_before["driver_id"] == driver_id]
        if driver_history.empty:
            return np.nan
        recent_positions = driver_history.sort_values("race_date")["race_position"].tail(races).std()
        return float(recent_positions)

    def compute_avg_positions_gained(
        self, history_before: pd.DataFrame, driver_id: str, races: int = CONSISTENCY_WINDOW
    ):
        driver_history = history_before[history_before["driver_id"] == driver_id].copy()
        if driver_history.empty:
            return np.nan

        driver_history = driver_history.sort_values("race_date").reset_index(drop=True)

        valid_history = driver_history.dropna(subset=["grid_position", "race_position"]).copy()
        if valid_history.empty:
            return np.nan

        race_counts = (
            history_before.sort_values("race_date")
            .groupby("race_date")["driver_id"]
            .count()
            .reindex(valid_history["race_date"])
        )
        denominator = race_counts.to_numpy(dtype=float) - 1.0

        positions_gained_raw = valid_history["grid_position"] - valid_history["race_position"]
        positions_gained = pd.Series(
            positions_gained_raw.to_numpy(dtype=float) / denominator, index=valid_history.index
        ).replace([np.inf, -np.inf], np.nan)

        recent_positions_gained = positions_gained.dropna().tail(races)
        if recent_positions_gained.empty:
            return np.nan

        span = min(races, len(recent_positions_gained))
        avg_positions_gained = recent_positions_gained.ewm(span=span).mean().iloc[-1]

        return float(avg_positions_gained)

    def compute_lap1_avg_positions_gained(
        self, history_before: pd.DataFrame, driver_id: str, races: int = CONSISTENCY_WINDOW
    ):
        driver_history = history_before[history_before["driver_id"] == driver_id].copy()
        if driver_history.empty:
            return np.nan

        driver_history = driver_history.sort_values("race_date").reset_index(drop=True)

        valid_history = driver_history.dropna(subset=["grid_position", "lap_1_position"]).copy()
        if valid_history.empty:
            return np.nan

        race_counts = (
            history_before.sort_values("race_date")
            .groupby("race_date")["driver_id"]
            .count()
            .reindex(valid_history["race_date"])
        )
        denominator = race_counts.to_numpy(dtype=float) - 1.0

        positions_gained_raw = valid_history["grid_position"] - valid_history["lap_1_position"]
        positions_gained = pd.Series(
            positions_gained_raw.to_numpy(dtype=float) / denominator, index=valid_history.index
        ).replace([np.inf, -np.inf], np.nan)

        recent_positions_gained = positions_gained.dropna().tail(races)
        if recent_positions_gained.empty:
            return np.nan

        span = min(races, len(recent_positions_gained))
        avg_positions_gained = recent_positions_gained.ewm(span=span).mean().iloc[-1]

        return float(avg_positions_gained)

    def compute_overtaking_difficulty(self, history_before: pd.DataFrame, circuit_location: str):
        """Variance of positions a driver typically gains at this track."""
        track_history = history_before[history_before["circuit_id"] == circuit_location].copy()
        if track_history.empty:
            return np.nan

        track_history = track_history.sort_values("race_date").reset_index(drop=True)

        valid_history = track_history.dropna(subset=["grid_position", "race_position"]).copy()
        if valid_history.empty:
            return np.nan

        race_counts = (
            history_before.sort_values("race_date")
            .groupby("race_date")["driver_id"]
            .count()
            .reindex(valid_history["race_date"])
        )
        denominator = race_counts.to_numpy(dtype=float) - 1.0

        positions_gained_raw = valid_history["grid_position"] - valid_history["race_position"]
        positions_gained = pd.Series(
            positions_gained_raw.to_numpy(dtype=float) / denominator, index=valid_history.index
        ).replace([np.inf, -np.inf], np.nan)

        valid_positions_gained = positions_gained.dropna()
        if valid_positions_gained.empty:
            return np.nan

        std_positions_gained = valid_positions_gained.std()

        return float(std_positions_gained)

    def compute_team_development_trend(
        self, history_before: pd.DataFrame, team_id: str, year: int, min_races: int = MIN_DEV_RACES
    ):
        """
        Trend di sviluppo del team, calcolato sulla pendenza del 'quali_time'
        nel corso della stagione.

        Aggregazione: per ogni GP si prende il minimo (il piu' veloce) tra i
        due piloti del team, per isolare il potenziale della vettura da un
        eventuale errore/incidente del singolo pilota.

        Reset annuale: filtrato solo su 'year' corrente. Un trend che scavalca
        il cambio di stagione (o peggio un cambio di regs era) rischierebbe di
        scambiare un reset di telaio invernale per una regressione di sviluppo
        -- stessa logica di car_age_proxy/regulation_era.

        Segno: la pendenza grezza (delta vs indice gara) e' negativa se il team
        migliora (delta che si riduce). Restituiamo -slope: un valore POSITIVO
        indica sviluppo positivo (macchina che diventa piu' veloce). Verifica
        che questa convenzione ti torni comoda rispetto alle altre feature.
        """
        team_history = history_before[(history_before["team_id"] == team_id) & (history_before["year"] == year)]
        if team_history.empty:
            return np.nan

        per_race = team_history.groupby("race_date")["quali_time"].min().sort_index()
        fastest_per_race = (
            history_before[history_before["year"] == year].groupby("race_date")["quali_time"].min().sort_index()
        )
        delta = (per_race - fastest_per_race).dropna()  # allinea per indice, poi dropna
        if len(delta) < min_races:
            return np.nan  # troppo pochi punti in stagione, il trend sarebbe rumore

        X = np.arange(len(delta)).reshape(-1, 1)
        y = delta.values

        model = HuberRegressor(epsilon=1.35, max_iter=1000)
        model.fit(X, y)

        return float(-model.coef_[0])

    def compute_wet_affinity(self, history_before: pd.DataFrame, driver_id: str) -> float:
        """Driver's average position in wet races."""
        driver_history = history_before[history_before["driver_id"] == driver_id]
        if driver_history.empty:
            return np.nan

        # Filter for wet races (you might need to adjust this condition)
        wet_races = driver_history[driver_history["rain_probability"] > WET_WEATHER_THRESHOLD]
        if wet_races.empty:
            return np.nan

        wet_avg = wet_races["race_position"].mean()
        career_avg = driver_history["race_position"].mean()

        if np.isnan(wet_avg) or np.isnan(career_avg):
            return np.nan

        return float(wet_avg - career_avg)

    def get_rain_probability(self, year, race_number, session, circuit_location, race_date, force) -> float:
        """Rain probability for race day, pre-fetched and stored at Bronze time."""
        longitude, latitude = CIRCUIT_COORDS[circuit_location]
        future = race_date > pd.Timestamp.now(tz="UTC").tz_localize(None)
        weather_df = self.silver.get_clean_weather(
            year, race_number, session, latitude, longitude, race_date, future, force
        )
        rain_probability = weather_df["rain_probability"].iloc[0]
        return float(rain_probability)

    def is_unclassified_dnf(self, status: str) -> bool:
        """True if the driver did not finish or was DQD, DNS, ecc.
        and should be excluded from the target, not set to last place"""
        if status is None:
            return False  # fallback > no signal
        if status in ["Retired", "Accident", "Withdrew", "Did not start", "Disqualified"]:
            return True
        else:
            return False

    def get_regs_era(self, year: int):
        """Return the regulation era based on the year."""
        if year >= 2026:
            return 2026
        elif year >= 2022:
            return 2022
        else:
            return np.nan  # Should not happen due to assertion in build_features


@staticmethod
def _is_reset_event(
    grid_position: float, quali_position: float, status: str, penalty_threshold: int = COMPONENTS_PENALTY_THRESHOLD
) -> bool:
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
    if pd.isna(grid_position) or pd.isna(quali_position):
        return False  # defensive - no signal

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

    if status in ["Retired", "Did not start", "Accident"]:
        return True  # Possibly something broke, so it will be changed

    delta = grid_position - quali_position
    return delta > penalty_threshold
