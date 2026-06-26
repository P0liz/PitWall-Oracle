import fastf1
import pandas as pd
import os
from pathlib import Path
from src.config import DATA_DIR

class BronzeLayer:
    data_dir = Path(DATA_DIR) / "bronze"
    
    def get_raw_laps(self, year: int, race_number: int, session: int):
        filename = f"{year}_{race_number}_{session}_raw_laps.parquet"
        if filename in os.listdir(self.data_dir):
            # Load from file
            df = pd.read_parquet(self.data_dir / filename)
        else:
            # Load from API
            data = fastf1.get_session(year, race_number, session)
            data.load()
            laps = data.laps
            df = pd.DataFrame(data=laps).reset_index(drop=True)
            df.to_parquet(self.data_dir / filename)
        return df
    
    def get_raw_results(self, year: int, race_number: int, session: int):
        filename = f"{year}_{race_number}_{session}_raw_results.parquet"
        if filename in os.listdir(self.data_dir):
            # Load from file
            df = pd.read_parquet(self.data_dir / filename)
        else:
            # Load from API
            data = fastf1.get_session(year, race_number, session)
            data.load()
            results = data.results
            df = pd.DataFrame(data=results).reset_index(drop=True)
            df.to_parquet(self.data_dir / filename)
        return df
    
    def get_event_metadata(self, year: int, race_number: int):
        filename = f"{year}_{race_number}_event.parquet"
        if filename in os.listdir(self.data_dir):
            # Load from file
            df = pd.read_parquet(self.data_dir / filename)
        else:
            # Load from API
            data = fastf1.get_session(year, race_number, 1)
            data.load()
            df = pd.DataFrame([data.event]).reset_index(drop=True)
            df.to_parquet(self.data_dir / filename)
        return df
    
    def get_raw_weather(self, year: int, race_number: int, session: int):
        filename = f"{year}_{race_number}_{session}_raw_weather.parquet"
        if filename in os.listdir(self.data_dir):
            # Load from file
            df = pd.read_parquet(self.data_dir / filename)
        else:
            # Load from API
            data = fastf1.get_session(year, race_number, session)
            data.load()
            event = data.weather_data
            df = pd.DataFrame(data=event).reset_index(drop=True)
            df.to_parquet(self.data_dir / filename)
        return df
        
    
    

