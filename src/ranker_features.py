MANDATORY_FEATURES: tuple[str, ...] = (
    "grid_position",
    "quali_pace",
    "teammate_delta_quali",
    "team_id",
    "driver_id",
    "circuit_id",
    "driver_current_form",
    "forecast_rain_probability",
    "driver_track_affinity",
    "wet_affinity",
    "year",
    "regulation_era",
)

# combo pace: tutte insieme funzionano ma singolarmente no
# ma la cosa strana è che comunque non migliorano il modello senza race pace features...
OPTIONAL_FEATURES: tuple[str, ...] = (
    "teammate_delta_grid_position",
    "degradation_rate",
    "teammate_delta_deg",
    "practice_position",  # combo pace
    "mean_race_pace",  # combo pace
    "late_stint_dropoff",  # combo pace
    "team_race_pace",  # combo pace
    "teammate_delta_pace",
    "teammate_recent_race_h2h",
    "teammate_delta_race_form",
    "driver_consistency",
    "teammate_delta_consistency",
    "team_current_form",
    "quali_current_form",
    "teammate_recent_quali_h2h",
    "avg_positions_gained",
    "teammate_delta_pos_gained",
    "lap1_avg_pos_gained",
    "teammate_delta_lap1_pos_gained",
    "driver_recent_race_pace",
    "teammate_delta_recent_pace",
    "overtaking_difficulty",
    "team_track_affinity",
    "team_development",
    "team_pit_execution_index",
    "team_strategy_aggressiveness_score",
    "teammate_delta_wet_affinity",
    "is_street_circuit",
)


ALL_RANKER_FEATURES: tuple[str, ...] = MANDATORY_FEATURES + OPTIONAL_FEATURES


# Fixed production input for the ranker.  Feature experiments must change this
# tuple intentionally and are evaluated by the walk-forward training pipeline;
# columns are never inferred from whatever happens to be present in Gold.
# The order matches the current champion models so existing artifacts remain a
# valid comparison baseline during the migration away from feature selection.

PRODUCTION_FEATURES: tuple[str, ...] = (
    "grid_position",
    "teammate_delta_grid_position",
    "quali_pace",
    "teammate_delta_quali",
    "team_id",
    "driver_id",
    "circuit_id",
    "driver_current_form",
    "teammate_delta_race_form",
    "forecast_rain_probability",
    "driver_track_affinity",
    "wet_affinity",
    "year",
    "regulation_era",
    "degradation_rate",
    "teammate_delta_deg",
    "practice_position",
    "mean_race_pace",
    "late_stint_dropoff",
    "team_race_pace",
    "teammate_delta_pace",
    "driver_consistency",
    "quali_current_form",
    "avg_positions_gained",
    "teammate_delta_pos_gained",
    "lap1_avg_pos_gained",
    "teammate_delta_lap1_pos_gained",
    "driver_recent_race_pace",
    "teammate_delta_recent_pace",
    "team_development",
    "team_strategy_aggressiveness_score",
    "teammate_delta_wet_affinity",
)


# Known non-ranker columns that may coexist with the registered features in the
# prepared dataframe.  This is documentation as well as a useful allow-list for
# diagnostics; preparation does not drop these columns because folds need them.
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "target",
        "dnf_target",
        "technical_dnf_target",
        "race_number",
        "race_date",
        "session_type",
        "raw_driver_id",
        "raw_team_id",
        "team_dnf_rate",
        "driver_dnf_rate",
        "car_age_proxy",
        "driver_dnf_free_streak",
        "smoothed_circuit_dnf_rate",
        "driver_wet_dnf_risk",
        "smoothed_driver_lap1_dnf_rate",
        "is_sprint",
        "source_file",
    }
)


def validate_feature_registry() -> None:
    """Fail fast when the explicit registry contains duplicates or overlap."""

    mandatory = set(MANDATORY_FEATURES)
    optional = set(OPTIONAL_FEATURES)
    overlap = mandatory & optional
    if overlap:
        raise ValueError(f"Feature presenti sia tra obbligatorie sia tra opzionali: {sorted(overlap)}")
    if len(mandatory) != len(MANDATORY_FEATURES):
        raise ValueError("MANDATORY_FEATURES contiene duplicati")
    if len(optional) != len(OPTIONAL_FEATURES):
        raise ValueError("OPTIONAL_FEATURES contiene duplicati")
    unknown_production = set(PRODUCTION_FEATURES) - (mandatory | optional)
    if unknown_production:
        raise ValueError(f"PRODUCTION_FEATURES contiene feature non registrate: {sorted(unknown_production)}")
    if len(PRODUCTION_FEATURES) != len(set(PRODUCTION_FEATURES)):
        raise ValueError("PRODUCTION_FEATURES contiene duplicati")


validate_feature_registry()
