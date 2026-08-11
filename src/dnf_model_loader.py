import re
import warnings
from pathlib import Path
from typing import Iterable

import mlflow

from .ranker_model_loader import MLFLOW_CACHE_DIR, MLFLOW_TRACKING_URI

MODEL_DIR = Path("models")
BASE_MODEL_PATH = MODEL_DIR / "dnf_logistic_base.joblib"
DNF_EXPERIMENT = "PitWall_Oracle_DNF"


def dnf_artifact_candidates(artifact_paths: Iterable[str], year: int, race_number: int) -> list[str]:
    if race_number < 1:
        raise ValueError("race_number deve essere maggiore o uguale a 1")

    base_path = None
    promoted_pattern = re.compile(rf"dnf_logistic_{year}_(\d+)\.joblib")
    promoted: list[tuple[int, str]] = []

    for artifact_path in artifact_paths:
        artifact_name = Path(artifact_path).name
        if artifact_name == "dnf_logistic_base.joblib":
            base_path = artifact_path
            continue
        match = promoted_pattern.fullmatch(artifact_name)
        if match and int(match.group(1)) <= race_number:
            promoted.append((int(match.group(1)), artifact_path))

    candidates = [artifact_path for _, artifact_path in sorted(promoted, reverse=True)]
    if base_path is not None:
        candidates.append(base_path)
    return candidates


def select_dnf_artifact_path(artifact_paths: Iterable[str], year: int, race_number: int) -> str | None:
    candidates = dnf_artifact_candidates(artifact_paths, year, race_number)
    return candidates[0] if candidates else None


def _runs_for_year(client, experiment_id: str, year: int):
    runs = []
    page_token = None
    while True:
        page = client.search_runs(
            experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=100, page_token=page_token
        )
        runs.extend(page)
        page_token = getattr(page, "token", None)
        if not page_token:
            break

    expected_run_name = f"DNF_Season_Simulation_{year}"
    return [
        run
        for run in runs
        if run.info.status == "FINISHED"
        and (
            run.data.params.get("simulation_year") == str(year)
            or run.data.tags.get("mlflow.runName") == expected_run_name
        )
    ]


def resolve_dnf_model_path(
    year: int, race_number: int, client=None, cache_dir: Path = MLFLOW_CACHE_DIR, local_model_dir: Path = MODEL_DIR
) -> Path:
    if race_number < 1:
        raise ValueError("race_number deve essere maggiore o uguale a 1")

    try:
        if client is None:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()

        experiment = client.get_experiment_by_name(DNF_EXPERIMENT)
        if experiment is None:
            raise ValueError(f"Esperimento '{DNF_EXPERIMENT}' non trovato")

        runs = _runs_for_year(client, experiment.experiment_id, year)
        if not runs:
            raise ValueError(f"Nessuna run DNF completata trovata per la stagione {year}")
        latest_run = max(runs, key=lambda run: run.info.start_time or 0)
        artifacts = client.list_artifacts(latest_run.info.run_id, "models")
        candidates = dnf_artifact_candidates([artifact.path for artifact in artifacts], year, race_number)
        if not candidates:
            raise ValueError(f"Nessun modello DNF causale trovato per {year} gara {race_number}")

        run_cache_dir = Path(cache_dir) / latest_run.info.run_id
        run_cache_dir.mkdir(parents=True, exist_ok=True)
        download_errors = []
        for artifact_path in candidates:
            try:
                return Path(
                    client.download_artifacts(
                        latest_run.info.run_id,
                        artifact_path,
                        dst_path=str(run_cache_dir),
                    )
                )
            except Exception as error:
                download_errors.append(f"{artifact_path}: {error}")

        raise RuntimeError("; ".join(download_errors))
    except Exception as error:
        local_model_dir = Path(local_model_dir)
        fallback = local_model_dir / BASE_MODEL_PATH.name
        warnings.warn(
            f"MLflow non disponibile o senza artefatti DNF utilizzabili per {year} gara {race_number} "
            f"({error}). Uso il modello locale '{fallback}'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return fallback
