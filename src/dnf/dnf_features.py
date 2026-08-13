"""Explicit feature registry for the all-cause DNF model."""

DNF_HISTORY_FEATURES: tuple[str, ...] = (
    "driver_dnf_free_streak",
    "smoothed_circuit_dnf_rate",
    "driver_wet_dnf_risk",
    "smoothed_driver_lap1_dnf_rate",
    "is_sprint",
)

DNF_CANDIDATE_FEATURES: tuple[str, ...] = (
    "grid_position",
    "team_dnf_rate",
    "driver_dnf_rate",
    "car_age_proxy",
    "avg_positions_gained",
    "lap1_avg_pos_gained",
    "wet_affinity",
    "forecast_rain_probability",
    "overtaking_difficulty",
    "is_street_circuit",
    *DNF_HISTORY_FEATURES,
)

if len(DNF_CANDIDATE_FEATURES) != len(set(DNF_CANDIDATE_FEATURES)):
    raise ValueError("DNF_CANDIDATE_FEATURES contiene duplicati")
