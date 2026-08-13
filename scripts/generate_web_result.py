import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import fastf1

from src.monte_carlo_simulator import simulate_race
from src.data.gold_layer import GoldLayer
from src.publication.actual import ResultsNotReadyError, build_history_document
from webapp.api.paths import prediction_relative_path
from src.publication.prediction import build_prediction_document
from src.publication.publisher import PublicationSummary, publish_history_document, publish_prediction_document
from src.publication.scheduler import DueOperation, choose_due_operations
from src.utils import get_session_mapping
from webapp.api.schemas import PredictionDocument, SessionType


def _session_number(event, season: int, session_type: SessionType) -> int:
    is_conventional = event["EventFormat"].iloc[0] == "conventional"
    if session_type == "sprint" and is_conventional:
        raise ValueError(f"Round does not contain a Sprint: season={season}")
    race_type = "sr" if session_type == "sprint" else "gp"
    return get_session_mapping(season, is_conventional, race_type, "race")


def _prediction_document(
    season: int,
    round_number: int,
    session_type: SessionType,
    generated_at: datetime,
    simulations: int,
    seed: int,
    force_refresh: bool,
) -> PredictionDocument:
    gold = GoldLayer()
    event = gold.silver.get_clean_event_metadata(season, round_number, force_refresh)
    session_number = _session_number(event, season, session_type)
    result = simulate_race(
        season, round_number, session_number, force=force_refresh, n_simulations=simulations, seed=seed
    )
    is_conventional = event["EventFormat"].iloc[0] == "conventional"
    race_type = "sr" if session_type == "sprint" else "gp"
    qualifying_session = get_session_mapping(season, is_conventional, race_type, "quali")
    qualifying = gold.silver.get_clean_results(season, round_number, qualifying_session, force_refresh)
    return build_prediction_document(
        result,
        event,
        qualifying,
        season=season,
        round_number=round_number,
        session_type=session_type,
        session_number=session_number,
        generated_at=generated_at,
        simulations=simulations,
        seed=seed,
    )


def _history_document(
    season: int,
    round_number: int,
    session_type: SessionType,
    session_number: int,
    generated_at: datetime,
    data_root: Path,
    force_refresh: bool,
):
    relative_prediction = prediction_relative_path(season, round_number, session_type)
    prediction_path = data_root / relative_prediction
    if not prediction_path.is_file():
        raise ResultsNotReadyError(f"Prediction archive is missing: '{relative_prediction}'")
    prediction = PredictionDocument.model_validate_json(prediction_path.read_text(encoding="utf-8"))
    gold = GoldLayer()
    official_results = gold.silver.get_clean_results(season, round_number, session_number, force_refresh)
    return build_history_document(
        prediction, official_results, generated_at=generated_at, prediction_path=relative_prediction
    )


def _execute(
    operation: DueOperation,
    *,
    generated_at: datetime,
    data_root: Path,
    dry_run: bool,
    simulations: int,
    seed: int,
    force_refresh: bool,
) -> PublicationSummary:
    if operation.operation == "publish-prediction":
        document = _prediction_document(
            operation.season,
            operation.round_number,
            operation.session_type,
            generated_at,
            simulations,
            seed,
            force_refresh,
        )
        return publish_prediction_document(document, data_root, dry_run=dry_run)
    if operation.operation == "publish-actual":
        document = _history_document(
            operation.season,
            operation.round_number,
            operation.session_type,
            operation.session_number,
            generated_at,
            data_root,
            force_refresh,
        )
        return publish_history_document(document, data_root, dry_run=dry_run)
    raise ValueError(f"Unsupported publication operation: '{operation.operation}'")


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _batch_summary(summaries: list[PublicationSummary]) -> dict:
    if not summaries:
        return {"status": "no-op", "operations": [], "changed_paths": []}
    statuses = {summary.status for summary in summaries}
    status = "published" if "published" in statuses else "validated" if "validated" in statuses else "deferred"
    changed_paths = sorted({path for summary in summaries for path in summary.changed_paths})
    return {"status": status, "operations": [summary.to_dict() for summary in summaries], "changed_paths": changed_paths}


def main(
    argv: list[str] | None = None, *, now: datetime | None = None, schedule_loader=fastf1.get_event_schedule
) -> int:
    parser = argparse.ArgumentParser(description="Generate validated PitWall Oracle web results")
    parser.add_argument("--operation", choices=("auto", "publish-prediction", "publish-actual"), default="auto")
    parser.add_argument("--season", type=int)
    parser.add_argument("--round", dest="round_number", type=int)
    parser.add_argument("--session-type", choices=("sprint", "race"), default="race")
    publication_mode = parser.add_mutually_exclusive_group()
    publication_mode.add_argument("--dry-run", dest="dry_run", action="store_true")
    publication_mode.add_argument("--publish", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=True)
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("webapp/api/data"))
    parser.add_argument("--summary-path", type=Path, default=Path("results/publication-summary.json"))
    args = parser.parse_args(argv)

    generated_at = now or datetime.now(timezone.utc)
    season = args.season or generated_at.year
    if args.simulations < 1:
        parser.error("--simulations must be greater than zero")

    automatic = args.operation == "auto"
    if automatic:
        schedule = schedule_loader(season)
        due = choose_due_operations(schedule, generated_at, args.data_root, season)
        if not due:
            _write_summary(args.summary_path, _batch_summary([]))
            return 0
        operations = due
    else:
        if args.season is None or args.round_number is None:
            parser.error("manual publication requires --season and --round")
        event = GoldLayer().silver.get_clean_event_metadata(args.season, args.round_number, args.force_refresh)
        session_number = _session_number(event, args.season, args.session_type)
        operations = (DueOperation(args.operation, args.season, args.round_number, args.session_type, session_number),)

    summaries: list[PublicationSummary] = []
    for operation in operations:
        try:
            summaries.append(
                _execute(
                    operation,
                    generated_at=generated_at,
                    data_root=args.data_root,
                    dry_run=args.dry_run,
                    simulations=args.simulations,
                    seed=args.seed,
                    force_refresh=args.force_refresh,
                )
            )
        except ResultsNotReadyError as error:
            if not automatic:
                raise
            summaries.append(
                PublicationSummary(
                    "deferred",
                    operation.operation,
                    operation.season,
                    operation.round_number,
                    operation.session_type,
                    (),
                    str(error),
                )
            )
    _write_summary(args.summary_path, _batch_summary(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
