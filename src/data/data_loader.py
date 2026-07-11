import pandas as pd
from src.data.gold_layer import GoldLayer

# TODO: add preprocessing pipeline (kaggle scikit-learn pipeline)
# but only after calling train_test_split, otherwise we will have data leakage
# even better doing it inside the cross validation pipeline
# Steps:
# 1. Load data splitting into train, val, set test by year and race number
# For now exclude 2026 that will be integrated later with a more dynamic approach
# 2. Preprocess data (scaling, target-encoding, imputation, ecc), via sklearn pipeline

# TODO: setup logger for all warnings and or errors during data loading


class F1DataLoader:
    def __init__(self):
        self.gold = GoldLayer()

    # TODO: load all races starting from 2024
    # Single access point to the data
    def load(self, year: int, id: int):

        gold_df = self.gold.build_features(year, id)
        return gold_df

    # Target Encoding: useful for the model to know the general strength of a driver/team
    # TODO: inserire anche i vari identificativi, da calcolare SOLO dopo lo split del dataset (evitare leakage)
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

    def apply_target_encoding(self, gold_df: pd.DataFrame, group_col: str, cutoff_date: pd.Timestamp) -> pd.Series:
        """Applica la mappa calcolata sopra a un DataFrame Gold, gestendo i mai-visti."""
        encoding_map = self.compute_target_encoding_map(group_col, cutoff_date)
        if not self.history_df.empty:
            global_fallback = self.history_df.loc[self.history_df["race_date"] < cutoff_date, "is_podium"].mean()
        else:
            global_fallback = 0.0

        return gold_df[group_col].map(encoding_map).fillna(global_fallback)
