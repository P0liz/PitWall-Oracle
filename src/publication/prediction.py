from datetime import datetime

import pandas as pd

from src.monte_carlo_simulator import SimulationResult, build_head_to_head_matrix
from src.utils import normalize_utc_timestamp
from webapp.api.schemas import DriverPrediction, PredictionDocument, PublicationInfo, RaceInfo, SessionType


def _display_name(row: pd.Series) -> str:
    full_name = row.get("FullName")
    if pd.notna(full_name) and str(full_name).strip():
        return str(full_name).strip()
    return f"{row['FirstName']} {row['LastName']}".strip()


def build_prediction_document(
    result: SimulationResult,
    event: pd.DataFrame,
    qualifying: pd.DataFrame,
    *,
    season: int,
    round_number: int,
    session_type: SessionType,
    session_number: int,
    generated_at: datetime,
    simulations: int,
    seed: int,
) -> PredictionDocument:
    if event.empty:
        raise ValueError("Event metadata is empty")
    required_qualifying = {"driver_id", "FirstName", "LastName", "Abbreviation", "team_id", "TeamName"}
    missing = required_qualifying - set(qualifying.columns)
    if missing:
        raise ValueError(f"Qualifying metadata is missing columns: {sorted(missing)}")
    metadata = qualifying.drop_duplicates("driver_id").set_index("driver_id")

    drivers: list[DriverPrediction] = []
    for predicted_position, summary_row in enumerate(result.summary.itertuples(index=False), start=1):
        driver_id = str(summary_row.driver_id)
        if driver_id not in metadata.index:
            raise ValueError(f"Driver '{driver_id}' is missing from qualifying metadata")
        driver_row = metadata.loc[driver_id]
        drivers.append(
            DriverPrediction(
                driver_id=driver_id,
                display_name=_display_name(driver_row),
                abbreviation=str(driver_row["Abbreviation"]),
                team_id=str(driver_row["team_id"]),
                team_name=str(driver_row["TeamName"]),
                predicted_position=predicted_position,
                expected_position=float(summary_row.expected_position),
                win_probability=float(summary_row.win_probability),
                podium_probability=float(summary_row.podium_probability),
                points_probability=float(summary_row.points_probability),
                dnf_probability=float(summary_row.dnf_probability),
                finish_probability=float(summary_row.finish_probability),
            )
        )

    event_row = event.iloc[0]
    return PredictionDocument(
        schema_version="1.0",
        race=RaceInfo(
            season=season,
            round=round_number,
            session_type=session_type,
            name=str(event_row["EventName"]),
            circuit=str(event_row["Location"]),
            start_time=normalize_utc_timestamp(
                event_row[f"Session{session_number}DateUtc"], f"{session_type} start time"
            ).to_pydatetime(),
        ),
        publication=PublicationInfo(
            type="live",
            generated_at=generated_at,
            data_cutoff=generated_at,
            model_artifact=result.ranker_model_path.name,
            dnf_strategy="logistic",
            simulations=simulations,
            seed=seed,
        ),
        drivers=drivers,
        head_to_head=build_head_to_head_matrix(result.driver_ids, result.simulated_positions),
    )
