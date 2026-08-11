import fastf1
import pandas as pd
from src.utils import setup_custom_logger, get_session_mapping, normalize_utc_timestamp
from src.data.gold_layer import GoldLayer
from src.config import STARTING_YEAR, STATIC_ENDING_YEAR, NEW_YEAR
import datetime
from pathlib import Path

CATEGORICAL_COLS = ["driver_id", "team_id"]
TEST_SET_DIM = 8

log = setup_custom_logger("DataLoader")


class DataLoader:
    def __init__(self):
        self.gold = GoldLayer()
        self.history_builder = self.gold.history_builder
        self.log = log
        self.history_df = self.history_builder.get_history()
        self.circuit_dtype = None
        self._refresh_circuit_dtype(self.history_df)
        self.train_df = None
        self.test_df = None
        self.dnf_df = None
        self.data_dir = Path("data_files")

    def _refresh_circuit_dtype(self, *dataframes: pd.DataFrame | None) -> None:
        """Build the categorical dtype from every currently known circuit."""
        circuit_series = [
            df["circuit_id"] for df in dataframes if df is not None and not df.empty and "circuit_id" in df.columns
        ]
        if not circuit_series:
            self.circuit_dtype = None
            return

        categories = sorted(pd.concat(circuit_series, ignore_index=True).dropna().astype(str).unique().tolist())
        self.circuit_dtype = pd.CategoricalDtype(categories=categories, ordered=False)

    # Single access point to the data
    async def load_data(
        self, last_date: datetime = None, static_df: pd.DataFrame = None, is_dynamic: bool = False, force=False
    ):
        train_filename = self.data_dir / "train_df.parquet"
        test_filename = self.data_dir / "test_df.parquet"
        dnf_filename = self.data_dir / "dnf_df.parquet"

        if is_dynamic:
            if static_df is None:
                raise ValueError("I dati statici sono necessari per il training dinamico")
            if last_date is None:
                raise ValueError("Il cutoff temporale è necessario per il training dinamico")
            self.log.info("Building dynamic data through cutoff...")
            self.train_df, self.test_df = await self.build_dynamic_data(static_df, last_date, force)
            return self.train_df, self.test_df

        cache_exists = train_filename.exists() and test_filename.exists() and dnf_filename.exists()
        if cache_exists and not force:
            self.log.info("Loading static data from parquet files...")
            self.train_df = pd.read_parquet(train_filename)
            self.test_df = pd.read_parquet(test_filename)
            self.dnf_df = pd.read_parquet(dnf_filename)
        else:
            self.log.info("Building static data from scratch...")
            self.train_df, self.test_df = await self.build_static_data(force)
            self.train_df.to_parquet(train_filename, index=False)
            self.test_df.to_parquet(test_filename, index=False)
            self.dnf_df.to_parquet(dnf_filename, index=False)

        return self.train_df, self.test_df

    async def build_static_data(self, force=False):
        # 1. Build precedent history
        self._build_history(STARTING_YEAR - 1, force)

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

        # 3. Apply target encoding to the data and save
        self._apply_encoding(all_races, race_is_test)

        return self.train_df, self.test_df

    # The call of this function is assumed after the build_static_data method
    # Which means that the history and the static data is already built
    async def build_dynamic_data(self, static_df: pd.DataFrame, last_date: datetime = None, force: bool = False):
        if static_df is None:
            self.log.error("Static data is required for dynamic training")
            raise ValueError("I dati statici sono necessari per il training dinamico")
        cutoff_date = normalize_utc_timestamp(last_date, "last_date")

        # Build features for current year
        races, race_is_test = [], []
        self.log.info(f"Building features for {NEW_YEAR} season...")
        schedule = fastf1.get_event_schedule(NEW_YEAR)
        schedule_dates = pd.to_datetime(schedule["Session5DateUtc"], errors="coerce", utc=True)
        n_races = schedule_dates.loc[schedule_dates <= cutoff_date].count()
        for i in range(1, n_races + 1):
            results = self.gold.build_features(NEW_YEAR, i, force)
            for race_df in results:
                races.append(race_df)
                # Dynamic training uses all races so no test set
                race_is_test.append(False)

        # 3. Apply target encoding to the data and save
        self._apply_encoding(races, race_is_test, static_df)

        # Validate that the dynamic dataset is correct (no future dates to avoid data leakage)
        race_dates = pd.to_datetime(self.train_df["race_date"], errors="coerce", utc=True, format="mixed")
        if race_dates.isna().any() or (race_dates > cutoff_date).any():
            self.log.error("Dynamic dataset contains invalid or future dates")
            raise ValueError("Il dataset dinamico contiene date non valide o successive al cutoff")

        return self.train_df, self.test_df

    def _apply_encoding(self, all_races, race_is_test, static_df: pd.DataFrame = None):
        """
        Apply target encoding
        Encoding is applied for every single race separately,
        so that there is no leakage between "past" and "future" races
        """
        self.history_df = self.history_builder.get_history()
        train_parts, test_parts, dnf_parts = [], [], []
        assert len(all_races) == len(
            race_is_test
        ), f"Mismatch: {len(all_races)} race_df vs {len(race_is_test)} flag is_test"

        for race_df, is_test in zip(all_races, race_is_test):
            cutoff_date = race_df["race_date"].iloc[0]
            # Il modello DNF e la baseline gerarchica devono conservare gli ID
            # reali dei team: il target encoding è specifico del ranker.
            dnf_parts.append(race_df.copy())
            encoded = race_df.copy()
            for col in CATEGORICAL_COLS:
                encoded[col] = self.apply_target_encoding(race_df, col, cutoff_date=cutoff_date)
            (test_parts if is_test else train_parts).append(encoded)

        # Building dataframes and saving them
        if train_parts:
            self.train_df = pd.concat(train_parts, ignore_index=True)
            self.train_df = self.train_df.dropna(subset=["target"])
            # Adding dynamic data to precomputed static data in dynamic data building
            if static_df is not None:
                self.train_df = pd.concat([static_df, self.train_df], ignore_index=True)
        elif static_df is not None:
            self.train_df = static_df.copy()
        else:
            self.train_df = pd.DataFrame()

        if test_parts:
            self.test_df = pd.concat(test_parts, ignore_index=True)
            self.test_df = self.test_df.dropna(subset=["target"])
        else:
            self.test_df = pd.DataFrame()

        self.dnf_df = pd.concat(dnf_parts, ignore_index=True) if dnf_parts else pd.DataFrame()

        # The history may not have existed when DataLoader was constructed and
        # may have been rebuilt during this load. Refresh the dtype now instead
        # of leaving it as None (astype(None) attempts a numeric conversion).
        self._refresh_circuit_dtype(self.history_df, self.train_df, self.test_df)
        if self.circuit_dtype is None:
            raise ValueError("Impossibile costruire le categorie: nessun circuit_id disponibile")
        if not self.train_df.empty:
            self.train_df["circuit_id"] = self.train_df["circuit_id"].astype(self.circuit_dtype)
        if not self.test_df.empty:
            self.test_df["circuit_id"] = self.test_df["circuit_id"].astype(self.circuit_dtype)

    def _build_history(self, year: int, force: bool = False):
        pre_schedule = fastf1.get_event_schedule(year)
        pre_n_races = pre_schedule["RoundNumber"].max()

        # Load races of the year before STARTING_YEAR in the history
        self.log.info(f"Building history for {year} season...")
        for i in range(1, pre_n_races + 1):
            event = self.gold.silver.get_clean_event_metadata(year, i, force)
            if event["EventFormat"].iloc[0] != "conventional":
                self._build_history_session(year, False, "sr", force, i, event)
                self._build_history_session(year, False, "gp", force, i, event)
            else:
                self._build_history_session(year, True, "gp", force, i, event)

    def _build_history_session(
        self, year, is_conventional, race_type: str, force: bool, race_number: int, event: pd.DataFrame
    ):
        self.log.info(f"Building history for {race_type} session of round {race_number} of {year} season...")

        location = event["Location"].iloc[0]
        quali_session = get_session_mapping(year, is_conventional, race_type, "quali")
        quali_results = self.gold.silver.get_clean_results(year, race_number, quali_session, force)
        race_session = get_session_mapping(year, is_conventional, race_type, "race")
        race_date = event[f"Session{race_session}Date"].iloc[0]
        race_results = self.gold.silver.get_clean_results(year, race_number, race_session, force)
        raw_race_laps = self.gold.silver.get_untouched_laps(year, race_number, race_session, force)
        clean_race_laps = self.gold.silver.get_clean_laps(year, race_number, race_session, force)
        if hasattr(race_date, "tzinfo") and race_date.tzinfo is not None:
            race_date = race_date.tz_convert("UTC").tz_localize(None)
        self.history_builder.update_history(
            year,
            race_number,
            race_session,
            quali_results,
            race_results,
            raw_race_laps,
            clean_race_laps,
            location,
            race_date,
            force=force,
        )

    # Target Encoding: useful for the model to know the general strength of a driver/team
    # Invece di farli diventare categorical, li trasformiamo in numerici, con un encoding basato sullo storico dei podi.
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
