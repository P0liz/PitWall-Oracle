import fastf1
import pandas as pd
from src.utils import setup_custom_logger
from src.data.gold_layer import GoldLayer
from src.data.history_builder import HistoryBuilder
import datetime

categorical_cols = ["driver_id", "team_id"]

STARTING_YEAR = 2024
STATIC_ENDING_YEAR = 2025
NEW_YEAR = 2026
TEST_SET_DIM = 8

# TODO: consider adding preprocessing pipeline via scikit learn if there is
# missing data that causes errors

# TODO: rework sprint weekends mapping system
# don't like the idea of having 10 ifs to decide which case i am
# have a dictionary with different cases and based on year and weekend type we get the correct sessions mapping
# same changes to apply to gold_layer


class DataLoader:
    def __init__(self):
        self.gold = GoldLayer()
        self.history_builder = HistoryBuilder(self.gold.silver)
        self.log = setup_custom_logger("DataLoader")
        self.history_df = self.history_builder.get_history()
        self.circuit_dtype = (
            pd.CategoricalDtype(
                categories=sorted(
                    self.history_df["circuit_id"].unique()
                ),  # tutti i circuiti conosciuti, non solo quelli in train
                ordered=False,
            )
            if self.history_df is not None
            else None
        )
        self.train_df = None
        self.test_df = None

    # Single access point to the data
    async def load(self, last_date: datetime = None, force=False):

        pre_schedule = fastf1.get_event_schedule(STARTING_YEAR - 1)
        pre_n_races = pre_schedule["RoundNumber"].max()

        # 1. Build precedent history
        # Load last 10 reaces of the year before STARTING_YEAR in the history
        self.log.info(f"Building history for {STARTING_YEAR - 1} season...")
        for i in range(pre_n_races - 9, pre_n_races + 1):
            self.log.info(f"Building history for round {i} of {STARTING_YEAR - 1} season...")
            event = self.gold.silver.get_clean_event_metadata(STARTING_YEAR - 1, i, force)
            location = event["Location"].iloc[0]
            # Sprint race
            if event["EventFormat"].iloc[0] != "conventional":
                self.log.info("Building also sprint race history")
                sprint_race_session = 4 if STARTING_YEAR - 1 <= 2023 else 3
                race_date = event[f"Session{sprint_race_session}Date"].iloc[0]
                quali_results = self.gold.silver.get_clean_results(STARTING_YEAR - 1, i, sprint_race_session - 1, force)
                race_results = self.gold.silver.get_clean_results(STARTING_YEAR - 1, i, sprint_race_session, force)
                if hasattr(race_date, "tzinfo") and race_date.tzinfo is not None:
                    race_date = race_date.tz_convert("UTC").tz_localize(None)
                self.history_builder.update_history(
                    STARTING_YEAR - 1, i, quali_results, race_results, location, race_date
                )
            # Actual race
            race_date = event["Session5Date"].iloc[0]
            quali_session = 2 if (STARTING_YEAR - 1 <= 2023) and (event["EventFormat"].iloc[0] != "conventional") else 4
            quali_results = self.gold.silver.get_clean_results(STARTING_YEAR - 1, i, quali_session, force)
            race_results = self.gold.silver.get_clean_results(STARTING_YEAR - 1, i, 5, force)
            if hasattr(race_date, "tzinfo") and race_date.tzinfo is not None:
                race_date = race_date.tz_convert("UTC").tz_localize(None)
            self.history_builder.update_history(STARTING_YEAR - 1, i, quali_results, race_results, location, race_date)

        # 2. Separate data and build features for static set
        all_races = []
        race_is_test = []
        for year in range(STARTING_YEAR, STATIC_ENDING_YEAR + 1):
            schedule = fastf1.get_event_schedule(year)
            n_races = schedule["RoundNumber"].max()
            self.log.info(f"Building features for {year} season...")
            for i in range(1, n_races + 1):
                results = self.gold.build_features(year, i, force)
                is_test = year == STATIC_ENDING_YEAR and i > n_races - TEST_SET_DIM
                for race_df in results:
                    all_races.append(race_df)
                    race_is_test.append(is_test)

        # 2.1. Build features for current year
        if last_date is not None:
            self.log.info(f"Building features for {NEW_YEAR} season...")
            schedule = fastf1.get_event_schedule(NEW_YEAR)
            n_races = schedule.loc[schedule["Session5DateUtc"] <= last_date, "Session5DateUtc"].count()
            for i in range(1, n_races + 1):
                results = self.gold.build_features(NEW_YEAR, i, force)
                for race_df in results:
                    all_races.append(race_df)
                    race_is_test.append(False)

        # 3. Apply target encoding
        # Encoding is applied for every single race separately,
        # so that there is no leakage between "past" and "future" races
        self.history_df = self.history_builder.get_history()
        train_parts, test_parts = [], []
        assert len(all_races) == len(
            race_is_test
        ), f"Mismatch: {len(all_races)} race_df vs {len(race_is_test)} flag is_test"

        for race_df, is_test in zip(all_races, race_is_test):
            cutoff_date = race_df["race_date"].iloc[0]
            encoded = race_df.copy()
            for col in categorical_cols:
                encoded[col] = self.apply_target_encoding(race_df, col, cutoff_date=cutoff_date)
            (test_parts if is_test else train_parts).append(encoded)

        self.train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
        self.test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

        # Drop target nan
        self.train_df = self.train_df.dropna(subset=["target"])
        self.test_df = self.test_df.dropna(subset=["target"])

        # Make circuit_id categorical
        self.train_df["circuit_id"] = self.train_df["circuit_id"].astype(self.circuit_dtype)
        self.test_df["circuit_id"] = self.test_df["circuit_id"].astype(self.circuit_dtype)

        return self.train_df, self.test_df

    # Target Encoding: useful for the model to know the general strength of a driver/team
    # driver_id, team_id, circuit_id
    # ma questi presentano problemi nei dati quidni ne devo creare di miei con una mappa:
    #   driver_id: Abbreviation + first name + last name
    #   team_id: mappare il TeamName con un mio dizionario da aggiornare per ogni stagione
    #   circuit_id: usare location precisa del circuito (non universale ma quasi)
    # Inoltre questi non vanno bene lasciati come stringe per XGBRanker
    # quindi o li sostituisce tramite target_encoding (driver_id e team_id) o li si tralascia in seguito (circuit_id)
    def compute_target_encoding_map(
        self,
        group_col: str,  # "driver_id" | "team_id"
        cutoff_date: pd.Timestamp,  # esclusivo: solo dati STRETTAMENTE precedenti
        smoothing: int = 5,  # forza dello shrinkage verso la media globale
    ) -> dict:
        """
        Calcola la mappa {categoria -> % storica di podi}, usando solo dati con race_date < cutoff_date.
        Applica Bayesian/Laplace smoothing per gestire categorie con poco storico
        (es. rookie, team nuovo) senza valori estremi (0% o 100% su 1 sola gara).
        """
        past = self.history_df.loc[self.history_df["race_date"] < cutoff_date]

        if past.empty:
            return {}  # cold start totale: nessuno storico ancora disponibile

        global_podium_rate = past["is_podium"].mean()

        stats = past.groupby(group_col)["is_podium"].agg(["sum", "count"])
        # Shrinkage: (podi_reali + k * media_globale) / (gare_reali + k)
        stats["encoded"] = (stats["sum"] + smoothing * global_podium_rate) / (stats["count"] + smoothing)

        return stats["encoded"].to_dict()

    def apply_target_encoding(self, df: pd.DataFrame, group_col: str, cutoff_date: pd.Timestamp) -> pd.Series:
        """Applica la mappa calcolata sopra a un DataFrame Gold, gestendo i mai-visti."""
        encoding_map = self.compute_target_encoding_map(group_col, cutoff_date)

        past = self.history_df.loc[self.history_df["race_date"] < cutoff_date]
        global_fallback = past["is_podium"].mean() if not past.empty else 0.0
        return df[group_col].map(encoding_map).fillna(global_fallback)
