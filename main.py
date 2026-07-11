from src.data.silver_layer import SilverLayer
from src.data.bronze_layer import BronzeLayer
from src.data.gold_layer import GoldLayer
import pandas as pd
from src.config import CIRCUIT_COORDS

if __name__ == "__main__":
    bronze = BronzeLayer()
    silver = SilverLayer()
    gold = GoldLayer()
    # df = silver.get_clean_laps(2026, 1, 3)
    # df = bronze.get_raw_laps(2026, 1, 5)
    # df = bronze.get_event_metadata(2025, 2)
    df = gold.build_features(2026, 9, True)
    print(df)
