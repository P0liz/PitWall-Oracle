import pandas as pd
from src.data.gold_layer import GoldLayer


class F1DataLoader:
    def __init__(self):
        self.gold = GoldLayer()

    # Single access point to the data
    def load(self, year: int, id: int):
    
        gold_df   = self.gold.build_features(year, id)
        return gold_df