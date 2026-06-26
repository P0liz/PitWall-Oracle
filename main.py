from src.data.silver_layer import SilverLayer
from src.data.bronze_layer import BronzeLayer

if __name__ == "__main__":
    bronze = BronzeLayer()
    silver = SilverLayer()
    #df = silver.get_clean_results(2026, 1, 4)
    #df = bronze.get_raw_laps(2026, 1, 5)
    df = bronze.get_event_metadata(2026, 1)
    print(df)