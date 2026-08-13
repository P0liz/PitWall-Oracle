from pathlib import Path

from .schemas import SessionType


def event_archive_name(round_number: int, session_type: SessionType) -> str:
    return f"round-{round_number:02d}-{session_type}.json"


def prediction_relative_path(season: int, round_number: int, session_type: SessionType) -> str:
    return str(Path("predictions") / str(season) / event_archive_name(round_number, session_type)).replace("\\", "/")


def history_relative_path(season: int, round_number: int, session_type: SessionType) -> str:
    return str(Path("history") / str(season) / event_archive_name(round_number, session_type)).replace("\\", "/")
