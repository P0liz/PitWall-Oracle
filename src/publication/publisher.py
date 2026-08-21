from dataclasses import asdict, dataclass
from pathlib import Path

from webapp.api.schemas import (
    CurrentPredictionPointer,
    HistoryDocument,
    HistoryIndex,
    HistoryIndexItem,
    PredictionDocument,
    SessionType,
)
from webapp.api.paths import history_relative_path, prediction_relative_path


class PublicationError(RuntimeError):
    pass


@dataclass(frozen=True)
class PublicationSummary:
    status: str
    operation: str
    season: int
    round_number: int
    session_type: SessionType
    changed_paths: tuple[str, ...]
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def _serialized(model) -> bytes:
    return (model.model_dump_json(indent=2) + "\n").encode("utf-8")


def _replace_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_suffix(path.suffix + ".tmp")
    candidate.write_bytes(content)
    candidate.replace(path)


def publish_prediction_document(
    document: PredictionDocument, data_root: Path, *, dry_run: bool, allow_replace: bool = False
) -> PublicationSummary:
    data_root = Path(data_root)
    season, round_number = document.race.season, document.race.round
    session_type = document.race.session_type
    relative_archive = prediction_relative_path(season, round_number, session_type)
    relative_pointer = "predictions/current.json"
    archive = data_root / relative_archive
    pointer = data_root / relative_pointer
    if archive.exists() and not allow_replace:
        raise PublicationError(f"Prediction archive already exists: '{relative_archive}'")

    archive_content = _serialized(document)
    PredictionDocument.model_validate_json(archive_content)
    pointer_model = CurrentPredictionPointer(
        season=season, round=round_number, session_type=session_type, prediction_path=relative_archive
    )
    pointer_content = _serialized(pointer_model)
    CurrentPredictionPointer.model_validate_json(pointer_content)
    summary = PublicationSummary(
        status="validated" if dry_run else "published",
        operation="publish-prediction",
        season=season,
        round_number=round_number,
        session_type=session_type,
        changed_paths=(relative_archive, relative_pointer),
        message="Prediction validated" if dry_run else "Prediction published",
    )
    if dry_run:
        return summary

    previous_archive = archive.read_bytes() if archive.exists() else None
    previous_pointer = pointer.read_bytes() if pointer.exists() else None
    try:
        _replace_bytes(archive, archive_content)
        _replace_bytes(pointer, pointer_content)
    except Exception as error:
        if previous_archive is None:
            archive.unlink(missing_ok=True)
        else:
            _replace_bytes(archive, previous_archive)
        if previous_pointer is None:
            pointer.unlink(missing_ok=True)
        else:
            _replace_bytes(pointer, previous_pointer)
        raise PublicationError("Prediction publication failed and was rolled back") from error
    return summary


def publish_history_document(document: HistoryDocument, data_root: Path, *, dry_run: bool) -> PublicationSummary:
    data_root = Path(data_root)
    season, round_number = document.race.season, document.race.round
    session_type = document.race.session_type
    relative_history = history_relative_path(season, round_number, session_type)
    relative_index = f"history/{season}/index.json"
    history_path = data_root / relative_history
    index_path = data_root / relative_index
    if history_path.exists():
        raise PublicationError(f"History archive already exists: '{relative_history}'")

    if index_path.exists():
        try:
            index = HistoryIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PublicationError("Existing history index is invalid") from error
    else:
        index = HistoryIndex(season=season, races=[])
    if index.season != season:
        raise PublicationError("History index season does not match publication")
    races = [item for item in index.races if (item.round, item.session_type) != (round_number, session_type)]
    races.append(
        HistoryIndexItem(
            season=season,
            round=round_number,
            session_type=session_type,
            name=document.race.name,
            publication_type="live",
        )
    )
    session_order = {"sprint": 0, "race": 1}
    updated_index = HistoryIndex(
        season=season, races=sorted(races, key=lambda item: (item.round, session_order[item.session_type]))
    )
    history_content = _serialized(document)
    index_content = _serialized(updated_index)
    HistoryDocument.model_validate_json(history_content)
    HistoryIndex.model_validate_json(index_content)
    summary = PublicationSummary(
        status="validated" if dry_run else "published",
        operation="publish-actual",
        season=season,
        round_number=round_number,
        session_type=session_type,
        changed_paths=(relative_history, relative_index),
        message="Actual result validated" if dry_run else "Actual result published",
    )
    if dry_run:
        return summary

    previous_index = index_path.read_bytes() if index_path.exists() else None
    try:
        _replace_bytes(history_path, history_content)
        _replace_bytes(index_path, index_content)
    except Exception as error:
        history_path.unlink(missing_ok=True)
        if previous_index is None:
            index_path.unlink(missing_ok=True)
        else:
            _replace_bytes(index_path, previous_index)
        raise PublicationError("Actual-result publication failed and was rolled back") from error
    return summary
