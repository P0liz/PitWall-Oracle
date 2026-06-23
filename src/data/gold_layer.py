import os
import warnings
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR, TEAM_ID_MAPPING
from .silver_layer import SilverLayer

class GoldLayer:
    
    def __init__(self):
        # Path where all the parquet files with features are saved
        self.data_dir = Path(DATA_DIR) / "gold"
        # File representing the state: containing data from all the weekends to have history on teams and drivers
        self.history_path = self.data_dir / "driver_team_history.parquet"
        self.silver = None
        self.history_df = None
        self.gold_df = None
    
    # Get features to parquet: one parquet for each weekend, with one row for each driver
    # The rows are the groups used by XGBRanker to calculate the ranking
    """ Main function to call for the data_loader """
    def build_features(self, year: int, race_number: int, force: bool = False):
        assert year >= 2022, "Year not supported: must be >= 2022"
        assert (race_number <= 24) & (race_number >= 1), "Race number {id} does not exist: max 24 races"
        
        filename = f"{year}_{race_number}_features.parquet"
        if filename in os.listdir(self.data_dir) and not force:
            # Load from file
            self.gold_df = pd.read_parquet(self.data_dir / filename)
        else:
            # Load data files
            self.silver = SilverLayer()
            self.history_df = pd.read_parquet(self.history_path)
            # Compute from features and save
            self.gold_df = self.get_gp_features(year, race_number)
            self.gold_df.to_parquet(self.data_dir / filename, index=False)
        return self.gold_df
            
    
    # Just calculate features and return a dataframe
    # TODO: feature engineering
    def get_gp_features(self, year: int, race_number: int):
        # Load raw data
        silver = self.silver
        
        laps_df = silver.get_clean_laps(year, race_number, 3)
        race_results_df = silver.get_clean_results(year, race_number, 5)
        quali_results_df = silver.get_clean_results(year, race_number, 4)
        
        event = silver.get_event_metadata(year, race_number)
        
        self.gold_df = pd.DataFrame()
            
        # Degradation rate 
        # Get from laps_df 
            
        # Race pace
        # Get from laps_df
            
        # Grid position
        # Get from quali_resultd_df
            
        # Teammate Delta pace
        # Calculate from race pace
            
        # Teammate Delta quali
        # Get from quali_resultd_df
        
        # Rolling Tech DNF Rate
        # Get from history: need to save Staus from the race results
        
        # TODO: rivedere la feature perchè i team cambiano spesso la PU, quindi è necessario sapere quale
        # viene usata, ed in quale gara, ma è possibile saperlo?
        # PU Age Proxy
        # Get from history: need to save laps from race results
        
        # Track affinity Score:
        # Get from history: need to save driver's finishing position from race results
        
        # Forecast Rain Probability
        # Need to access a weather API (open-meteo)
        # Calculate based on history and probability
        
        # Target (Y): l'obbiettivo di prediction del modello
        # Y = number of drivers - Position (from race results)
        # Da notare che se un pilota fa DNF allora sarebbe meglio escluderlo invece di piazzarlo ultimo
        
        # TODO: inserire anche i vari identificativi
        # driver_id, team_id, circuit_id
        # ma questi presentano problemi nei dati quidni ne devo creare di miei con una mappa:
        #   driver_id: Abbreviation + first name + last name
        #   team_id: mappare il TeamName con un mio dizionario da aggiornare per ogni stagione
        #   circuit_id: usare location precisa del circuito (non universale ma quasi)
        # Inoltre questi non vanno bene lasciati come stringe per XGBRanker
        # quindi o li sostituisce tramite target_encoding (driver_id e team_id) o li si tralascia in seguito (circuit_id)
        race_date = event["Session5Date"]
        circuit_location = event["Location"]
        for col in ["driver_id", "team_id"]:
            self.apply_target_encoding(group_col=col, cutoff_date=race_date)
        
        # Update file containig the state with the new data
        self.update_history(year, race_number, circuit_location, race_date)
        
        return self.gold_df
        
    # TODO: spostare la logica della history in un file a parte altrimenti mi incasino troppo la vita qui
    def update_history(self, year: int, race_number: str, circuit_location: str, race_date: pd.Timestamp):
        quali_results = self.silver.get_clean_results(race_number, "4")   
        race_results = self.silver.get_clean_results(race_number, "5")    

        new_history_rows = build_history_rows(
            quali_results=quali_results,
            race_results=race_results,
            race_id=race_number,
            race_date=race_date,
            season=year,
            circuit_location=circuit_location,
        )

        if self.history_path.exists():
            history_df = pd.read_parquet(self.history_path)
            history_df = history_df.loc[history_df["race_id"] != race_number]  # upsert: rimuovi se già presente
            updated_history = pd.concat([history_df, new_history_rows], ignore_index=True)
        else:
            updated_history = new_history_rows

        updated_history.to_parquet(self.history_path, index=False)
    
    
    # Target Encoding: useful for the model to know the general strength of a driver/team
    # TODO: rivedere per i podi dei circuiti che dovrebbero essere specifici per il pilota/team e non globali
    def compute_target_encoding_map(
        self,
        group_col: str,             # "driver_id" | "team_id" | "circuit_id"
        cutoff_date: pd.Timestamp,  # esclusivo: solo dati STRETTAMENTE precedenti
        smoothing: int = 5,         # forza dello shrinkage verso la media globale
    ) -> dict:
        """
        Calcola la mappa {categoria -> % storica di podi}, usando solo
        dati con race_date < cutoff_date (anti-leakage by construction).
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

    def apply_target_encoding(
        self,
        group_col: str,
        cutoff_date: pd.Timestamp,
    ) -> pd.Series:
        """Applica la mappa calcolata sopra a un DataFrame Gold, gestendo i mai-visti."""
        encoding_map = self.compute_target_encoding_map(group_col, cutoff_date)
        if not self.history_df.empty:
            global_fallback = self.history_df.loc[self.history_df["race_date"] < cutoff_date, "is_podium"].mean()
        else:
            global_fallback = 0.0

        return self.gold_df[group_col].map(encoding_map).fillna(global_fallback)
    
# Helpers
def build_driver_id(abbreviation: str, first_name: str, last_name: str) -> str:
    raw = f"{abbreviation}_{first_name}_{last_name}"
    return raw.strip().lower().replace(" ", "_")

def map_team_id(team_name: str) -> str:
    if team_name not in TEAM_ID_MAPPING:
        warnings.warn(f"TeamName '{team_name}' non presente in TEAM_ID_MAPPING — "
                       f"fallback al nome grezzo, aggiungilo manualmente al dizionario")
        return team_name  # fallback visibile, non silenzioso
    return TEAM_ID_MAPPING[team_name]

def build_history_rows(
    quali_results: pd.DataFrame,
    race_results: pd.DataFrame,
    race_id: str,
    race_date: pd.Timestamp,
    season: int,
    circuit_location: str,
) -> pd.DataFrame:
    MECHANICAL_DNF_KEYWORDS = ["Engine", "Gearbox", "Hydraulics", "Electrical",
                                "Brakes", "Suspension", "Power Unit", "Turbo"]

    def is_mechanical(status: str) -> bool:
        return any(kw.lower() in str(status).lower() for kw in MECHANICAL_DNF_KEYWORDS)

    rows = []
    for _, race_row in race_results.iterrows():
        driver_id = build_driver_id(
            race_row["Abbreviation"], race_row["FirstName"], race_row["LastName"]
        )

        quali_row = quali_results.loc[quali_results["Abbreviation"] == race_row["Abbreviation"]]
        grid_position = quali_row["GridPosition"].iloc[0] if not quali_row.empty else race_row.get("GridPosition", np.nan)

        classified_pos = race_row["ClassifiedPosition"]
        is_podium = classified_pos in ("1", "2", "3")

        rows.append({
            "race_id": race_id,
            "race_date": race_date,
            "season": season,
            "driver_id": driver_id,
            "team_id": map_team_id(race_row["TeamName"]),
            "circuit_id": circuit_location,
            "grid_position": grid_position,
            "points_scored": race_row["Points"],
            "laps_completed": race_row["Laps"],
            "status_raw": race_row["Status"],
            "is_mechanical_dnf": is_mechanical(race_row["Status"]),
            "is_podium": is_podium,
        })

    return pd.DataFrame(rows)
