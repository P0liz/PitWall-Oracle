from src.data.silver_layer import SilverLayer
from src.data.bronze_layer import BronzeLayer

if __name__ == "__main__":
    bronze = BronzeLayer()
    silver = SilverLayer()
    #df = silver.get_clean_laps(2026, 1, "FP1")
    df = bronze.get_raw_results(2026, 1, 5)
    print(df)