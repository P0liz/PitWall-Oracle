GLOBAL_SEED = 2026
DATA_DIR = "data_files"

# Gold layer
MIN_VALID_LAPS = 5
ROLLING_DNF_WINDOW = 10
CURRENT_FORM_RACES = 3
CONSISTENCY_WINDOW = 10
COMPONENTS_PENALTY_THRESHOLD = 3

# Optimized training
TIME_PENALTY_THRESHOLD = 2
TOLLERANCE = 0.003

to_log = {
    "min_valid_laps": MIN_VALID_LAPS,
    "rolling_dnf_window": ROLLING_DNF_WINDOW,
    "current_form_races": CURRENT_FORM_RACES,
    "consistency_window": CONSISTENCY_WINDOW,
    "components_penalty_threshold": COMPONENTS_PENALTY_THRESHOLD,
    "time_penalty_threshold": TIME_PENALTY_THRESHOLD,
    "tollerance": TOLLERANCE,
}

TEAM_ID_MAPPING = {
    "Alfa Romeo": "audi_lineage",
    "Kick Sauber": "audi_lineage",
    "Racing Point": "aston_martin_lineage",
    "Aston Martin": "aston_martin_lineage",
    "Renault": "alpine_lineage",
    "Alpine": "alpine_lineage",
    "Red Bull": "red_bull_lineage",
    "Red Bull Racing": "red_bull_lineage",
    "Racing Bulls": "vcarb_lineage",
    "RB": "vcarb_lineage",
    "AlphaTauri": "vcarb_lineage",
    "Scuderia Ferrari": "ferrari_lineage",
    "Ferrari": "ferrari_lineage",
    "Mercedes": "mercedes_lineage",
    "McLaren": "mclaren_lineage",
    "Williams": "williams_lineage",
    "Haas F1 Team": "haas_lineage",
    "Audi": "audi_lineage",
    "Cadillac": "cadillac_lineage",
    "": "unknown",
}

# "Location": (longitude, latitude)
CIRCUIT_COORDS = {
    "Austin": (-97.633, 30.135),
    "Baku": (49.842, 40.369),
    "Barcelona": (2.259, 41.569),
    "Buenos Aires": (-58.459, -34.694),
    "Budapest": (19.250, 47.583),
    "Dix": (-76.927, 42.337),
    "Estoril": (-9.394, 38.751),
    "Hockenheim": (8.572, 49.330),
    "Imola": (11.713, 44.341),
    "Indianapolis": (-86.236, 39.795),
    "Istanbul": (29.412, 40.958),
    "Jacarepaguá": (-43.395, -22.976),
    "Jeddah": (39.104, 21.632),
    "Johannesburg": (28.069, -25.998),
    "Las Vegas": (-115.168, 36.116),
    "Le Castellet": (5.791, 43.253),
    "Lusail": (51.454, 25.490),
    "Madrid": (-3.620, 40.471),
    "Magny-Cours": (3.164, 46.863),
    "Melbourne": (144.970, -37.846),
    "Mexico City": (-99.091, 19.402),
    "Miami": (-80.239, 25.958),
    "Monaco": (7.429, 43.737),
    "Montréal": (-73.525, 45.506),
    "Monza": (9.290, 45.621),
    "Nürburg": (6.943, 50.334),
    "Portimão": (-8.628, 37.232),
    "Sakhir": (50.512, 26.031),
    "São Paulo": (-46.698, -23.702),
    "Scarperia e San Piero": (11.372, 43.998),
    "Sepang": (101.738, 2.761),
    "Shanghai": (121.221, 31.340),
    "Silverstone": (-1.017, 52.072),
    "Marina Bay": (103.859, 1.291),
    "Sochi": (39.960, 43.407),
    "Spa-Francorchamps": (5.971, 50.436),
    "Spielberg": (14.761, 47.223),
    "Suzuka": (136.534, 34.844),
    "Yas Marina": (54.601, 24.471),
    "Yas Island": (24.4672, 54.6031),
    "Zandvoort": (4.541, 52.389),
}

IS_STREET_CIRCUIT = {
    "Austin": False,
    "Baku": True,
    "Barcelona": False,
    "Buenos Aires": False,
    "Budapest": False,
    "Dix": False,
    "Estoril": False,
    "Hockenheim": False,
    "Imola": False,
    "Indianapolis": False,
    "Istanbul": False,
    "Jacarepaguá": False,
    "Jeddah": True,
    "Johannesburg": False,
    "Las Vegas": True,
    "Le Castellet": False,
    "Lusail": False,
    "Madrid": True,
    "Magny-Cours": False,
    "Melbourne": True,
    "Mexico City": False,
    "Miami": True,
    "Monaco": True,
    "Montréal": True,
    "Monza": False,
    "Nürburg": False,
    "Portimão": False,
    "Sakhir": False,
    "São Paulo": False,
    "Scarperia e San Piero": False,
    "Sepang": False,
    "Shanghai": False,
    "Silverstone": False,
    "Marina Bay": True,
    "Sochi": True,
    "Spa-Francorchamps": False,
    "Spielberg": False,
    "Suzuka": False,
    "Yas Marina": False,
    "Yas Island": False,
    "Zandvoort": False,
}


# Open-meteo API
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
