import re
import warnings
from pathlib import Path

import mlflow

MODEL_DIR = Path("models")
BASE_MODEL_PATH = MODEL_DIR / "pitwall_oracle_base.json"
MLFLOW_CACHE_DIR = Path("data_files") / "mlflow_artifacts"
RANKING_EXPERIMENT = "PitWall_Oracle_Ranking"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # stesso backend store usato da ml_flow_auto.py


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

    expected_run_name = f"F1_Season_Simulation_{year}"
    return [
        run
        for run in runs
        if run.info.status == "FINISHED"
        and (
            run.data.params.get("simulation_year") == str(year)
            or run.data.tags.get("mlflow.runName") == expected_run_name
        )
    ]


def resolve_ranker_model_path(
    year: int, race_number: int, client=None, cache_dir: Path = MLFLOW_CACHE_DIR, local_base: Path = BASE_MODEL_PATH
) -> Path:
    """Carica il Champion causale dalla run MLflow conclusa più recente."""
    if race_number < 1:
        raise ValueError("race_number deve essere maggiore o uguale a 1")

    try:
        if client is None:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()

        experiment = client.get_experiment_by_name(RANKING_EXPERIMENT)
        if experiment is None:
            raise ValueError(f"Esperimento '{RANKING_EXPERIMENT}' non trovato")

        runs = _runs_for_year(client, experiment.experiment_id, year)
        if not runs:
            raise ValueError(f"Nessuna run MLflow completata trovata per la stagione {year}")

        latest_run = max(runs, key=lambda run: run.info.start_time or 0)
        promoted_pattern = re.compile(rf"pitwall_oracle_{year}_(\d+)\.json")
        candidates = []

        try:
            artifacts = client.list_artifacts(latest_run.info.run_id, "models")
        except Exception as error:
            raise RuntimeError(
                f"Impossibile elencare gli artefatti della run MLflow più recente " f"{latest_run.info.run_id}: {error}"
            ) from error

        for artifact in artifacts:
            artifact_name = Path(artifact.path).name
            match = promoted_pattern.fullmatch(artifact_name)
            if match and int(match.group(1)) <= race_number:
                candidates.append(
                    (int(match.group(1)), latest_run.info.start_time or 0, latest_run.info.run_id, artifact.path)
                )
            elif artifact_name == "pitwall_oracle_base.json":
                candidates.append((-1, latest_run.info.start_time or 0, latest_run.info.run_id, artifact.path))

        if not candidates:
            raise ValueError(f"Nessun modello Ranker trovato in MLflow per la stagione {year}")

        cache_dir = Path(cache_dir)
        download_errors = []
        for model_race, _, run_id, artifact_path in sorted(candidates, reverse=True):
            try:
                run_cache_dir = cache_dir / run_id
                run_cache_dir.mkdir(parents=True, exist_ok=True)
                downloaded = client.download_artifacts(run_id, artifact_path, dst_path=str(run_cache_dir))
                selected_path = Path(downloaded)
                model_label = "base" if model_race == -1 else f"promosso per gara {model_race}"
                print(f"Loading ranker {selected_path.name} da MLflow ({model_label}, run {run_id})")
                return selected_path
            except Exception as error:
                download_errors.append(f"{artifact_path} dalla run {run_id}: {error}")

        raise RuntimeError("; ".join(download_errors))
    except Exception as error:
        warnings.warn(
            f"MLflow non disponibile o senza artefatti utilizzabili per {year} gara {race_number} "
            f"({error}). Uso il modello base locale '{local_base}'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return Path(local_base)
