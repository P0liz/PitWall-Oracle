"""Read-only access to published prediction and history JSON documents."""

from itertools import combinations
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .paths import history_relative_path
from .schemas import (
    CurrentPredictionPointer,
    GlobalHistoryStatistics,
    HeadToHeadResponse,
    HistoryDocument,
    HistoryIndex,
    HistoryStatisticsPoint,
    PredictionDocument,
    SessionType,
)


class ResultNotFound(Exception):
    """Raised when the requested published result is unavailable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class InvalidPublishedData(Exception):
    """Raised when published result data cannot be trusted or validated."""


ModelType = TypeVar("ModelType", bound=BaseModel)


def _event_pairwise_accuracy(history: HistoryDocument) -> float:
    correct_pairs = 0
    eligible_pairs = 0
    for driver_a, driver_b in combinations(history.comparisons, 2):
        position_a = driver_a.actual_position
        position_b = driver_b.actual_position
        if position_a is None and position_b is None:
            continue

        eligible_pairs += 1
        predicted_a_ahead = driver_a.predicted_position < driver_b.predicted_position
        if position_a is None:
            actual_a_ahead = False
        elif position_b is None:
            actual_a_ahead = True
        else:
            actual_a_ahead = position_a < position_b
        correct_pairs += predicted_a_ahead == actual_a_ahead

    if not eligible_pairs:
        raise InvalidPublishedData("History event has no eligible driver pairs")
    return correct_pairs / eligible_pairs


def _global_history_statistics(histories: list[HistoryDocument]) -> GlobalHistoryStatistics | None:
    if not histories:
        return None

    winner_hits = 0
    podium_hits = 0
    podium_total = 0
    pairwise_accuracy_total = 0.0
    mean_absolute_position_error_total = 0.0
    mean_absolute_position_error_count = 0
    timeline = []
    for event_count, history in enumerate(histories, start=1):
        winner_hits += any(
            comparison.predicted_position == 1 and comparison.actual_position == 1
            for comparison in history.comparisons
        )
        podium_hits += history.summary.podium_hits
        podium_total += history.summary.podium_total
        pairwise_accuracy_total += _event_pairwise_accuracy(history)
        if history.summary.mean_absolute_position_error is not None:
            mean_absolute_position_error_total += history.summary.mean_absolute_position_error
            mean_absolute_position_error_count += 1
        timeline.append(
            HistoryStatisticsPoint(
                round=history.race.round,
                session_type=history.race.session_type,
                winner_accuracy=winner_hits / event_count,
                podium_hit_rate=podium_hits / podium_total,
                pairwise_accuracy=pairwise_accuracy_total / event_count,
                mean_absolute_position_error=(
                    mean_absolute_position_error_total / mean_absolute_position_error_count
                    if mean_absolute_position_error_count
                    else None
                ),
            )
        )

    latest = timeline[-1]
    return GlobalHistoryStatistics(
        winner_accuracy=latest.winner_accuracy,
        podium_hit_rate=latest.podium_hit_rate,
        pairwise_accuracy=latest.pairwise_accuracy,
        timeline=timeline,
    )


class ResultsRepository:
    def __init__(self, data_root: Path):
        self.data_root = data_root.resolve()

    def _resolve_relative(self, relative_path: str) -> Path:
        candidate = (self.data_root / relative_path).resolve()
        if self.data_root not in candidate.parents:
            raise InvalidPublishedData("Published data path escapes the data root")
        return candidate

    def _load_model(self, path: Path, model_type: type[ModelType]) -> ModelType:
        if not path.is_file():
            raise ResultNotFound("race_not_found", f"No published data at {path.name}")
        try:
            return model_type.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as error:
            raise InvalidPublishedData(str(error)) from error

    def load_current(self) -> PredictionDocument:
        pointer = self._load_model(self._resolve_relative("predictions/current.json"), CurrentPredictionPointer)
        completed_race_path = self._resolve_relative(
            history_relative_path(pointer.season, pointer.round, pointer.session_type)
        )
        if completed_race_path.is_file():
            raise ResultNotFound("prediction_not_available", "No current prediction is available for a completed race")

        prediction = self._load_model(self._resolve_relative(pointer.prediction_path), PredictionDocument)
        if (prediction.race.season, prediction.race.round, prediction.race.session_type) != (
            pointer.season,
            pointer.round,
            pointer.session_type,
        ):
            raise InvalidPublishedData("Current prediction does not match its pointer")
        return prediction

    def load_head_to_head(self, driver_a: str, driver_b: str) -> HeadToHeadResponse:
        if driver_a == driver_b:
            raise ResultNotFound("head_to_head_not_found", "A driver cannot be compared with themselves")

        prediction = self.load_current()
        drivers = {driver.driver_id: driver for driver in prediction.drivers}
        if driver_a not in drivers or driver_b not in drivers:
            raise ResultNotFound("driver_not_found", "Driver is not in the current prediction")

        try:
            driver_a_probability = prediction.head_to_head[driver_a][driver_b]
            driver_b_probability = prediction.head_to_head[driver_b][driver_a]
        except KeyError as error:
            raise ResultNotFound("head_to_head_not_found", "No head-to-head result is available") from error

        return HeadToHeadResponse(
            driver_a_id=driver_a,
            driver_a_name=drivers[driver_a].display_name,
            driver_a_probability=driver_a_probability,
            driver_b_id=driver_b,
            driver_b_name=drivers[driver_b].display_name,
            driver_b_probability=driver_b_probability,
        )

    def list_history(self, season: int) -> HistoryIndex:
        self._validate_season(season)
        index = self._load_model(self._resolve_relative(f"history/{season}/index.json"), HistoryIndex)
        if index.season != season:
            raise InvalidPublishedData("History index season does not match its path")
        histories = [self.load_history(item.season, item.round, item.session_type) for item in index.races]
        return index.model_copy(update={"global_statistics": _global_history_statistics(histories)})

    def load_history(self, season: int, round_number: int, session_type: SessionType) -> HistoryDocument:
        self._validate_season(season)
        self._validate_round(round_number)
        history = self._load_model(
            self._resolve_relative(history_relative_path(season, round_number, session_type)), HistoryDocument
        )
        if (history.race.season, history.race.round, history.race.session_type) != (
            season,
            round_number,
            session_type,
        ):
            raise InvalidPublishedData("History document does not match its path")
        return history

    @staticmethod
    def _validate_season(season: int) -> None:
        if season < 2026:
            raise ResultNotFound("race_not_found", "Season must be 2026 or later")

    @staticmethod
    def _validate_round(round_number: int) -> None:
        if not 1 <= round_number <= 24:
            raise ResultNotFound("race_not_found", "Round must be between 1 and 24")
