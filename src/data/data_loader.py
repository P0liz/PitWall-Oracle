import pandas as pd

# TODO: create the other files and import them here

class F1DataLoader:
    def __init__(self):
        self.bronze = BronzeLayer()
        self.silver = SilverLayer()
        self.gold = GoldLayer()

    def load(self, year: int, race_id: str) -> pd.DataFrame:
        """Punto di accesso unico. Ritorna sempre il Gold Layer."""
        bronze_df = self.bronze.get(year, race_id)
        silver_df = self.silver.get(year, race_id, bronze_df)
        gold_df   = self.gold.get(year, race_id, silver_df)
        return gold_df