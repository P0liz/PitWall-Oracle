GLOBAL_SEED = 2003
DATA_DIR = "data_files"

# Gold layer
MIN_VALID_LAPS = 5
MIN_DEV_RACES = 5
ROLLING_DNF_WINDOW = 15
CURRENT_FORM_RACES = 3
CONSISTENCY_WINDOW = 10
COMPONENTS_PENALTY_THRESHOLD = 3
WET_WEATHER_THRESHOLD = 0.3

# Optimized training
RANKER_OPTUNA_TRIALS = 15
DEFAULT_DECAY_RATE = 0.0006189898976496755
TIME_PENALTY_THRESHOLD = 2
TOLLERANCE = 0.003
TARGET_MULTIPLIER = 2

to_log_ranker = {
    "min_valid_laps": MIN_VALID_LAPS,
    "min_dev_races": MIN_DEV_RACES,
    "rolling_dnf_window": ROLLING_DNF_WINDOW,
    "current_form_races": CURRENT_FORM_RACES,
    "consistency_window": CONSISTENCY_WINDOW,
    "components_penalty_threshold": COMPONENTS_PENALTY_THRESHOLD,
    "wet_weather_threshold": WET_WEATHER_THRESHOLD,
    "ranker_optuna_trials": RANKER_OPTUNA_TRIALS,
    "default_decay_rate": DEFAULT_DECAY_RATE,
    "time_penalty_threshold": TIME_PENALTY_THRESHOLD,
    "tollerance": TOLLERANCE,
    "target_multiplier": TARGET_MULTIPLIER,
}

# Optimized dnf
DNF_OPTUNA_TRIALS = 40
DNF_TARGET = "dnf_target"
MAX_DNF_PROB = 0.30

to_log_dnf = {"dnf_optuna_trials": DNF_OPTUNA_TRIALS, "dnf_target": DNF_TARGET, "max_dnf_prob": MAX_DNF_PROB}

STARTING_YEAR = 2024
STATIC_ENDING_YEAR = 2025
NEW_YEAR = 2026

# Simulator
FALLBACK_SIGMA = 0.5

# Session configuration
# (year, is_conventional, race_type, data) => (session_number)
SESSION_MAPPING = {
    (2023, False, "sr", "practice_laps"): 1,
    (2023, False, "sr", "practice_results"): 1,
    (2023, False, "sr", "quali"): 3,
    (2023, False, "sr", "race"): 4,
    (2023, False, "gp", "practice_laps"): 4,
    (2023, False, "gp", "practice_results"): 4,
    (2023, False, "gp", "quali"): 2,
    (2023, False, "gp", "race"): 5,
    (2024, False, "sr", "practice_laps"): 1,
    (2024, False, "sr", "practice_results"): 1,
    (2024, False, "sr", "quali"): 2,
    (2024, False, "sr", "race"): 3,
    (2024, False, "gp", "practice_laps"): 3,
    (2024, False, "gp", "practice_results"): 3,
    (2024, False, "gp", "quali"): 4,
    (2024, False, "gp", "race"): 5,
    (2025, False, "sr", "practice_laps"): 1,
    (2025, False, "sr", "practice_results"): 1,
    (2025, False, "sr", "quali"): 2,
    (2025, False, "sr", "race"): 3,
    (2025, False, "gp", "practice_laps"): 3,
    (2025, False, "gp", "practice_results"): 3,
    (2025, False, "gp", "quali"): 4,
    (2025, False, "gp", "race"): 5,
    (2026, False, "sr", "practice_laps"): 1,
    (2026, False, "sr", "practice_results"): 1,
    (2026, False, "sr", "quali"): 2,
    (2026, False, "sr", "race"): 3,
    (2026, False, "gp", "practice_laps"): 3,
    (2026, False, "gp", "practice_results"): 3,
    (2026, False, "gp", "quali"): 4,
    (2026, False, "gp", "race"): 5,
    (0, True, "gp", "practice_laps"): 2,
    (0, True, "gp", "practice_results"): 3,
    (0, True, "gp", "quali"): 4,
    (0, True, "gp", "race"): 5,
}

TEAM_ID_MAPPING = {
    "Alfa Romeo": "audi_lineage",
    "Kick Sauber": "audi_lineage",
    "Racing Point": "aston_martin_lineage",
    "Aston Martin": "aston_martin_lineage",
    "Renault": "alpine_lineage",
    "Alpine": "alpine_lineage",
    "Alpine F1 Team": "alpine_lineage",
    "Red Bull": "red_bull_lineage",
    "Red Bull Racing": "red_bull_lineage",
    "Racing Bulls": "vcarb_lineage",
    "RB": "vcarb_lineage",
    "RB F1 Team": "vcarb_lineage",
    "AlphaTauri": "vcarb_lineage",
    "Scuderia Ferrari": "ferrari_lineage",
    "Ferrari": "ferrari_lineage",
    "Mercedes": "mercedes_lineage",
    "McLaren": "mclaren_lineage",
    "Williams": "williams_lineage",
    "Haas F1 Team": "haas_lineage",
    "Audi": "audi_lineage",
    "Cadillac": "cadillac_lineage",
    "Cadillac F1 Team": "cadillac_lineage",
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
