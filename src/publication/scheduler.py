from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from webapp.api.paths import history_relative_path, prediction_relative_path
from src.utils import get_session_mapping
from webapp.api.schemas import SessionType

OperationName = Literal["publish-prediction", "publish-actual"]


@dataclass(frozen=True)
class DueOperation:
    operation: OperationName
    season: int
    round_number: int
    session_type: SessionType
    session_number: int


def _event_sessions(row, season: int) -> tuple[tuple[SessionType, int, pd.Timestamp], ...]:
    is_conventional = str(row.EventFormat).lower() == "conventional"
    requested: list[tuple[SessionType, int]] = []
    if not is_conventional:
        requested.append(("sprint", get_session_mapping(season, False, "sr", "race")))
    requested.append(("race", get_session_mapping(season, is_conventional, "gp", "race")))

    sessions = []
    for session_type, session_number in requested:
        start = pd.to_datetime(getattr(row, f"Session{session_number}DateUtc", None), errors="coerce", utc=True)
        if pd.notna(start):
            sessions.append((session_type, session_number, start))
    return tuple(sessions)


def choose_due_operations(
    schedule: pd.DataFrame, now: datetime, data_root: Path, season: int, operation: OperationName | None = None
) -> tuple[DueOperation, ...]:
    if operation not in (None, "publish-prediction", "publish-actual"):
        raise ValueError(f"Unsupported publication operation filter: '{operation}'")
    data_root = Path(data_root)
    now_utc = pd.Timestamp(now)
    now_utc = now_utc.tz_localize("UTC") if now_utc.tzinfo is None else now_utc.tz_convert("UTC")
    due_actuals: list[DueOperation] = []
    due_predictions: list[DueOperation] = []

    for row in schedule.itertuples(index=False):
        round_number = int(row.RoundNumber)
        if round_number < 1:
            continue
        for session_type, session_number, start in _event_sessions(row, season):
            prediction = data_root / prediction_relative_path(season, round_number, session_type)
            history = data_root / history_relative_path(season, round_number, session_type)
            until_start = start - now_utc

            if prediction.is_file() and not history.is_file() and now_utc >= start + pd.Timedelta(hours=24):
                due_actuals.append(DueOperation("publish-actual", season, round_number, session_type, session_number))
            elif not prediction.exists() and pd.Timedelta(0) <= until_start <= pd.Timedelta(hours=12):
                due_predictions.append(
                    DueOperation("publish-prediction", season, round_number, session_type, session_number)
                )

    operations = tuple(due_actuals + due_predictions)
    if operation is None:
        return operations
    return tuple(item for item in operations if item.operation == operation)
