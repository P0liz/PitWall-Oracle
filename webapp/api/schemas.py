"""Pydantic contracts for published PitWall Oracle results."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

PublicationType = Literal["live", "backtest"]
SessionType = Literal["sprint", "race"]


class RaceInfo(BaseModel):
    season: int = Field(ge=2026)
    round: int = Field(ge=1, le=24)
    session_type: SessionType
    name: str = Field(min_length=1)
    circuit: str = Field(min_length=1)
    start_time: datetime


class PublicationInfo(BaseModel):
    type: PublicationType
    generated_at: datetime
    data_cutoff: datetime
    model_artifact: str = Field(min_length=1)
    dnf_strategy: str = Field(min_length=1)
    simulations: int = Field(gt=0)
    seed: int


class DriverPrediction(BaseModel):
    driver_id: str
    display_name: str
    abbreviation: str
    team_id: str
    team_name: str
    predicted_position: int = Field(gt=0)
    expected_position: float = Field(gt=0)
    win_probability: float = Field(ge=0, le=1)
    podium_probability: float = Field(ge=0, le=1)
    points_probability: float = Field(ge=0, le=1)
    dnf_probability: float = Field(ge=0, le=1)
    finish_probability: float = Field(ge=0, le=1)


class PredictionDocument(BaseModel):
    schema_version: Literal["1.0"]
    race: RaceInfo
    publication: PublicationInfo
    drivers: list[DriverPrediction] = Field(min_length=2)
    head_to_head: dict[str, dict[str, float]]

    @model_validator(mode="after")
    def validate_prediction_matrix(self) -> "PredictionDocument":
        driver_ids = [driver.driver_id for driver in self.drivers]
        if len(driver_ids) != len(set(driver_ids)):
            raise ValueError("Driver IDs must be unique")

        positions = [driver.predicted_position for driver in self.drivers]
        if len(positions) != len(set(positions)):
            raise ValueError("Predicted positions must be unique")

        driver_id_set = set(driver_ids)
        if set(self.head_to_head) != driver_id_set:
            raise ValueError("Head-to-head matrix must include every driver exactly once")

        for driver_a in driver_ids:
            pairings = self.head_to_head[driver_a]
            expected_opponents = driver_id_set - {driver_a}
            if set(pairings) != expected_opponents:
                raise ValueError("Head-to-head matrix is incomplete")
            for driver_b, probability in pairings.items():
                if not 0 <= probability <= 1:
                    raise ValueError("Head-to-head probabilities must be between zero and one")
                reverse_probability = self.head_to_head[driver_b][driver_a]
                if abs(probability + reverse_probability - 1) > 1e-6:
                    raise ValueError("Head-to-head probabilities must be complementary")

        return self


class CurrentPredictionPointer(BaseModel):
    season: int = Field(ge=2026)
    round: int = Field(ge=1, le=24)
    session_type: SessionType
    prediction_path: str = Field(min_length=1)


class ActualDriverResult(BaseModel):
    driver_id: str
    actual_position: int | None = Field(default=None, gt=0)
    status: str


class DriverComparison(BaseModel):
    driver_id: str
    display_name: str
    predicted_position: int = Field(gt=0)
    actual_position: int | None = Field(default=None, gt=0)
    position_difference: int | None
    status: str


class ComparisonSummary(BaseModel):
    mean_absolute_position_error: float | None = Field(default=None, ge=0)
    podium_hits: int = Field(ge=0, le=3)
    podium_total: Literal[3] = 3
    top_five_hits: int = Field(ge=0, le=5)
    top_five_total: Literal[5] = 5


class HistoryDocument(BaseModel):
    schema_version: Literal["1.0"]
    race: RaceInfo
    prediction_path: str
    publication: PublicationInfo
    actual_results: list[ActualDriverResult]
    comparisons: list[DriverComparison]
    summary: ComparisonSummary


class HistoryIndexItem(BaseModel):
    season: int = Field(ge=2026)
    round: int = Field(ge=1, le=24)
    session_type: SessionType
    name: str
    publication_type: PublicationType


class HistoryStatisticsPoint(BaseModel):
    round: int = Field(ge=1, le=24)
    session_type: SessionType
    winner_accuracy: float = Field(ge=0, le=1)
    podium_hit_rate: float = Field(ge=0, le=1)
    pairwise_accuracy: float = Field(ge=0, le=1)
    mean_absolute_position_error: float | None = Field(default=None, ge=0)


class GlobalHistoryStatistics(BaseModel):
    winner_accuracy: float = Field(ge=0, le=1)
    podium_hit_rate: float = Field(ge=0, le=1)
    pairwise_accuracy: float = Field(ge=0, le=1)
    timeline: list[HistoryStatisticsPoint]


class HistoryIndex(BaseModel):
    season: int = Field(ge=2026)
    races: list[HistoryIndexItem]
    global_statistics: GlobalHistoryStatistics | None = None


class HeadToHeadResponse(BaseModel):
    driver_a_id: str
    driver_a_name: str
    driver_a_probability: float = Field(ge=0, le=1)
    driver_b_id: str
    driver_b_name: str
    driver_b_probability: float = Field(ge=0, le=1)
