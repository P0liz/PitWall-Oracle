import os
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR
from .bronze_layer import BronzeLayer

TIME_THRESHOLD = 1.04

class SilverLayer:
    data_dir = Path(DATA_DIR) / "silver"
    bronze = BronzeLayer()
    
    def get_clean_laps(self, year: int, race_number: int, session: int):
        assert (session >= 1) & (session <= 5), "Session number not valid: 1 <= session <= 5"
        
        filename = f"{year}_{race_number}_{session}_clean_laps.parquet"
        if filename in os.listdir(self.data_dir):
            # Load from file
            df = pd.read_parquet(self.data_dir / filename)
        else:
            # Load raw data
            df = self.bronze.get_raw_laps(year, race_number, session)
            
            time_reference = 0
            driver_ref = ""
            to_drop = []
            deleted_counter = 0
            deleted_idxs = []
            print("Tot: ", df.shape[0])
            for i in range(0, df.shape[0]):
                # Default data cleaning
                if df.iloc[i]["Deleted"] == True:
                    to_drop.append(i)
                    continue
                if df.iloc[i]["IsAccurate"] == False:
                    to_drop.append(i)
                    continue
                
                # Get reference for new driver
                if df.iloc[i]["Driver"] != driver_ref:
                    time_reference = df.iloc[i]["LapTime"]
                    driver_ref = df.iloc[i]["Driver"]
                    continue
                
                # Keep only laps that are within 4% of the driver's last lap time
                if df.iloc[i]["LapTime"] > time_reference * TIME_THRESHOLD:
                    # Check if the times switch from quali to race sim
                    if (i < df.shape[0] - 3):
                        new_ref_first = df.iloc[i]["LapTime"]
                        new_ref_second = df.iloc[i+1]["LapTime"]
                        new_ref_third = df.iloc[i+2]["LapTime"]
                        # If the next 3 laps are all within 4% of each other, we assume they are valid
                        if (new_ref_second < new_ref_first * TIME_THRESHOLD and new_ref_third < new_ref_second * TIME_THRESHOLD):
                            time_reference = new_ref_first
                            continue
                    to_drop.append(i)
                    deleted_counter += 1
                    deleted_idxs.append(i)
                else:
                    time_reference = df.iloc[i]["LapTime"]
            
            df = df.drop(to_drop).reset_index(drop=True)
            print("Deleted number: ",deleted_counter)
            print("Deleted: ", deleted_idxs)
            df.to_parquet(self.data_dir / filename)
        return df
    
    def get_clean_results(self, year: int, race_number: int, session: int):
        assert (session >= 1) & (session <= 5), "Session number not valid: 1 <= session <= 5"
        
        return self.bronze.get_raw_laps(year, race_number, session)
    
    def get_event_metadata(self, year: int, race_number: int):
        return self.bronze.get_event_metadata(year, race_number)
    
    def get_clean_weather(self, year: int, race_number: int, session: int):
        assert (session >= 1) & (session <= 5), "Session number not valid: 1 <= session <= 5"
        
        # TODO: possible transformations to the weather data
        return self.bronze.get_raw_weather(year, race_number, session)
    
