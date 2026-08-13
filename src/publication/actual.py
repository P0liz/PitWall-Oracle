from datetime import datetime

import numpy as np
import pandas as pd

from src.utils import POST_RACE_EXCLUSION_STATUSES, is_race_dnf
from webapp.api.schemas import (
    ActualDriverResult,
    ComparisonSummary,
    DriverComparison,
    HistoryDocument,
    PredictionDocument,
)


class ResultsNotReadyError(RuntimeError):
    pass


def _classified_position(position, status: str) -> int | None:
    if is_race_dnf(status) or status in POST_RACE_EXCLUSION_STATUSES or pd.isna(position):
        return None
    numeric = int(float(position))
    return numeric if numeric > 0 else None


def build_history_document(
    prediction: PredictionDocument, official_results: pd.DataFrame, *, generated_at: datetime, prediction_path: str
) -> HistoryDocument:
    required = {"driver_id", "Position", "Status"}
    missing = required - set(official_results.columns)
    if missing or official_results.empty:
        raise ResultsNotReadyError(f"Official results are incomplete: missing {sorted(missing)}")
    if official_results["driver_id"].duplicated().any() or official_results["Status"].isna().any():
        raise ResultsNotReadyError("Official results contain duplicate drivers or missing statuses")

    predicted = {driver.driver_id: driver for driver in prediction.drivers}
    result_ids = set(official_results["driver_id"].astype(str))
    if result_ids != set(predicted):
        raise ResultsNotReadyError("Official result drivers do not match the prediction")

    result_map = official_results.set_index("driver_id")
    actual_results: list[ActualDriverResult] = []
    comparisons: list[DriverComparison] = []
    classified_differences: list[int] = []
    actual_positions: dict[str, int] = {}
    for driver in prediction.drivers:
        row = result_map.loc[driver.driver_id]
        status = str(row["Status"])
        actual_position = _classified_position(row["Position"], status)
        difference = actual_position - driver.predicted_position if actual_position is not None else None
        if difference is not None:
            classified_differences.append(difference)
            actual_positions[driver.driver_id] = actual_position
        actual_results.append(
            ActualDriverResult(driver_id=driver.driver_id, actual_position=actual_position, status=status)
        )
        comparisons.append(
            DriverComparison(
                driver_id=driver.driver_id,
                display_name=driver.display_name,
                predicted_position=driver.predicted_position,
                actual_position=actual_position,
                position_difference=difference,
                status=status,
            )
        )

    predicted_podium = {driver.driver_id for driver in prediction.drivers if driver.predicted_position <= 3}
    predicted_top_five = {driver.driver_id for driver in prediction.drivers if driver.predicted_position <= 5}
    actual_podium = {driver_id for driver_id, position in actual_positions.items() if position <= 3}
    actual_top_five = {driver_id for driver_id, position in actual_positions.items() if position <= 5}
    mae = float(np.mean(np.abs(classified_differences))) if classified_differences else None

    return HistoryDocument(
        schema_version="1.0",
        race=prediction.race,
        prediction_path=prediction_path,
        publication=prediction.publication,
        actual_results=actual_results,
        comparisons=comparisons,
        summary=ComparisonSummary(
            mean_absolute_position_error=mae,
            podium_hits=len(predicted_podium & actual_podium),
            top_five_hits=len(predicted_top_five & actual_top_five),
        ),
    )
