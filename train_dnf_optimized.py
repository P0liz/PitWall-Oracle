import asyncio
from dataclasses import dataclass
import hashlib
import shutil
import sys
import traceback
from pathlib import Path

import fastf1
import mlflow
import numpy as np
import pandas as pd

from ml_flow_auto import MLFLOW_HOST, MLFLOW_PORT, launch_mlflow_server
from src.config import DNF_OPTUNA_TRIALS, to_log_dnf
from src.data.data_loader import DataLoader, NEW_YEAR
from src.dnf_metrics import DNFProbabilityMetrics, DNFPromotionDecision, decide_dnf_promotion, score_dnf_probabilities
from src.dnf_model import (
    DNFFittedModel,
    DNFModelConfig,
    DNF_TARGET,
    DNFTrainingResult,
    config_from_training_result,
    compute_probabilities,
    fit_dnf_logistic,
    load_dnf_artifact,
    save_fitted_dnf_artifact,
    save_dnf_artifact,
    train_dnf_logistic,
)
from src.utils import normalize_utc_timestamp

DNF_EXPERIMENT = "PitWall_Oracle_DNF"
FORCE = False
MODEL_DIR = Path("models")


@dataclass(frozen=True)
class DNFRaceEvaluation:
    champion: DNFProbabilityMetrics
    challenger: DNFProbabilityMetrics | None
    decision: DNFPromotionDecision | None


def gp_evaluation_frame(weekend_results: list[pd.DataFrame]) -> pd.DataFrame:
    if not weekend_results:
        raise ValueError("Nessun risultato del weekend disponibile per la valutazione DNF")
    gp_df = weekend_results[-1].dropna(subset=[DNF_TARGET]).copy()
    if gp_df.empty or "session_type" not in gp_df or not gp_df["session_type"].eq("race").all():
        raise ValueError("La valutazione DNF richiede esclusivamente righe GP valide")
    return gp_df


def append_weekend_history(history: pd.DataFrame, weekend_results: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([history, *weekend_results], ignore_index=True)


def evaluate_dnf_artifact(artifact: dict, gp_df: pd.DataFrame) -> DNFProbabilityMetrics:
    if "cutoff_date" not in artifact or "race_date" not in gp_df:
        raise ValueError("Cutoff artefatto o data GP mancanti per la valutazione DNF")
    cutoff_date = normalize_utc_timestamp(artifact["cutoff_date"], "cutoff artefatto DNF")
    race_date = normalize_utc_timestamp(gp_df["race_date"].iloc[0], "data GP DNF")
    if cutoff_date >= race_date:
        raise ValueError(f"Leakage DNF: cutoff artefatto {cutoff_date} non precedente al GP {race_date}")

    probabilities = compute_probabilities(gp_df, artifact=artifact)
    return score_dnf_probabilities(gp_df[DNF_TARGET].astype(int).to_numpy(), probabilities)


def evaluate_dnf_duel(
    champion_artifact: dict, challenger_artifact: dict | None, gp_df: pd.DataFrame
) -> DNFRaceEvaluation:
    champion_metrics = evaluate_dnf_artifact(champion_artifact, gp_df)
    if challenger_artifact is None:
        return DNFRaceEvaluation(champion_metrics, None, None)

    challenger_metrics = evaluate_dnf_artifact(challenger_artifact, gp_df)
    decision = decide_dnf_promotion(champion_metrics, challenger_metrics)
    return DNFRaceEvaluation(champion_metrics, challenger_metrics, decision)


def fit_and_log_challenger(
    history: pd.DataFrame, config: DNFModelConfig, cutoff_date: pd.Timestamp, step: int
) -> DNFFittedModel:
    fitted = fit_dnf_logistic(history, config)
    pending_path = MODEL_DIR / "dnf_logistic_pending.joblib"
    save_fitted_dnf_artifact(fitted, pending_path, cutoff_date)
    mlflow.log_artifact(str(pending_path), artifact_path="models")
    mlflow.log_metric("challenger_training_rows", fitted.training_rows, step=step)
    mlflow.log_metric("challenger_training_races", fitted.training_races, step=step)
    mlflow.log_param(f"challenger_cutoff_race_{step}", pd.Timestamp(cutoff_date).isoformat())
    return fitted


def save_champion_snapshot(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def update_champion_after_duel(
    champion_path: Path,
    challenger_path: Path,
    decision: DNFPromotionDecision,
    year: int,
    race_number: int,
) -> Path:
    if not decision.promote:
        return champion_path
    promoted_path = MODEL_DIR / f"dnf_logistic_{year}_{race_number}.joblib"
    return save_champion_snapshot(challenger_path, promoted_path)


def dataset_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def prepare_base_history(cached_history: pd.DataFrame, year: int, first_race_date) -> pd.DataFrame:
    """Isola la history pre-stagione e ne verifica il cutoff causale."""
    if "race_date" not in cached_history or DNF_TARGET not in cached_history:
        raise ValueError("Impossibile determinare il cutoff della history DNF")

    normalized_dates = pd.to_datetime(cached_history["race_date"], errors="coerce", utc=True, format="mixed")
    history = cached_history.loc[normalized_dates.dt.year < year].copy()
    cutoff_dates = normalized_dates.loc[history.index[history[DNF_TARGET].notna()]].dropna()
    if cutoff_dates.empty:
        raise ValueError("cutoff della history DNF non valido")

    cutoff_date = normalize_utc_timestamp(cutoff_dates.max(), "cutoff della history DNF")
    first_race_timestamp = normalize_utc_timestamp(first_race_date, "data della prima gara")
    if cutoff_date >= first_race_timestamp:
        raise ValueError(
            f"cutoff della history DNF {cutoff_date} non precedente alla prima gara {first_race_timestamp}"
        )
    return history.reset_index(drop=True)


def train_and_log_dnf_model(
    dnf_df: pd.DataFrame, year: int, model_race: int, n_trials: int = DNF_OPTUNA_TRIALS
) -> tuple[DNFTrainingResult, pd.Timestamp]:
    """Addestra il solo logistic DNF con la history disponibile prima della gara."""
    if model_race != 1:
        raise ValueError("L'HPO DNF può creare esclusivamente il modello base pre-stagione")
    valid_rows = dnf_df[dnf_df[DNF_TARGET].notna()]
    if valid_rows.empty:
        raise ValueError(f"Nessuna riga con '{DNF_TARGET}' disponibile per il cutoff DNF")
    cutoff_date = pd.Timestamp(valid_rows["race_date"].max())
    result = train_dnf_logistic(dnf_df, n_trials=n_trials)

    artifact_path = MODEL_DIR / "dnf_logistic_base.joblib"
    save_dnf_artifact(result, artifact_path, cutoff_date)

    metric_values = {
        "oos_brier_score": result.oos_brier_score,
        "oos_log_loss": result.oos_log_loss,
        "selection_brier_score": result.selection_brier_score,
    }
    for metric_name, value in metric_values.items():
        if np.isfinite(value):
            mlflow.log_metric(metric_name, value, step=model_race)
    mlflow.log_metric("training_rows", result.training_rows, step=model_race)
    mlflow.log_metric("training_races", result.training_races, step=model_race)
    mlflow.log_metric("evaluation_races", result.evaluation_races, step=model_race)
    mlflow.log_param(f"logistic_C_race_{model_race}", result.model_parameters["C"])
    mlflow.log_param(f"logistic_penalty_race_{model_race}", result.model_parameters["penalty"])
    mlflow.log_param(
        f"logistic_positive_class_weight_race_{model_race}", result.model_parameters["positive_class_weight"]
    )
    mlflow.log_param(f"logistic_features_race_{model_race}", ",".join(result.features))
    mlflow.log_param(f"logistic_optuna_trials_race_{model_race}", result.model_parameters["optuna_trials"])
    mlflow.log_artifact(str(artifact_path), artifact_path="models")
    mlflow.log_metric("training_end_race", model_race - 1, step=model_race)
    return result, cutoff_date


# -----------------#
# Main
# -----------------#
def run_pipeline():
    data_loader = DataLoader()
    asyncio.run(data_loader.load_data(force=FORCE))

    new_schedule = fastf1.get_event_schedule(NEW_YEAR)
    scheduled_race_dates = new_schedule["Session5DateUtc"].dropna()
    if scheduled_race_dates.empty:
        raise ValueError(f"Nessuna gara schedulata per il {NEW_YEAR}")
    dnf_history_df = prepare_base_history(data_loader.dnf_df, NEW_YEAR, scheduled_race_dates.min())
    completed_races = new_schedule.loc[new_schedule["Session5DateUtc"] <= pd.Timestamp.now(), "Session5DateUtc"]

    with mlflow.start_run(run_name=f"DNF_Season_Simulation_{NEW_YEAR}"):
        mlflow.set_tags({"model_type": "dnf", "simulation_year": str(NEW_YEAR)})
        mlflow.log_param("simulation_year", NEW_YEAR)
        mlflow.log_params(to_log_dnf)
        mlflow.log_param("historical_dataset_hash", dataset_hash(dnf_history_df))

        # Il Champion base e la sua configurazione usano esclusivamente le stagioni precedenti.
        base_result, _ = train_and_log_dnf_model(dnf_history_df, NEW_YEAR, model_race=1)
        frozen_config = config_from_training_result(base_result)
        mlflow.log_param("frozen_logistic_features", ",".join(frozen_config.features))
        mlflow.log_param("frozen_logistic_C", frozen_config.c)
        mlflow.log_param("frozen_logistic_penalty", frozen_config.penalty)
        mlflow.log_param("frozen_logistic_positive_class_weight", frozen_config.positive_class_weight)

        champion_path = MODEL_DIR / "dnf_logistic_base.joblib"
        challenger_path: Path | None = None
        champion_history: list[DNFProbabilityMetrics] = []

        for idx, _ in enumerate(completed_races):
            race_number = idx + 1
            weekend_results = data_loader.gold.build_features(NEW_YEAR, race_number, force=FORCE)
            gp_df = gp_evaluation_frame(weekend_results)

            champion_artifact = load_dnf_artifact(champion_path)
            challenger_artifact = load_dnf_artifact(challenger_path) if challenger_path is not None else None
            evaluation = evaluate_dnf_duel(champion_artifact, challenger_artifact, gp_df)

            champion_history.append(evaluation.champion)
            mlflow.log_metric("champion_brier_score", evaluation.champion.brier_score, step=race_number)
            mlflow.log_metric("champion_log_loss", evaluation.champion.log_loss, step=race_number)
            recent_champion = champion_history[-5:]
            mlflow.log_metric(
                "champion_moving_avg_brier_score",
                float(np.mean([metrics.brier_score for metrics in recent_champion])),
                step=race_number,
            )
            mlflow.log_metric(
                "champion_moving_avg_log_loss",
                float(np.mean([metrics.log_loss for metrics in recent_champion])),
                step=race_number,
            )

            if evaluation.challenger is not None and evaluation.decision is not None:
                mlflow.log_metric("challenger_brier_score", evaluation.challenger.brier_score, step=race_number)
                mlflow.log_metric("challenger_log_loss", evaluation.challenger.log_loss, step=race_number)
                mlflow.log_metric("delta_brier_score", evaluation.decision.delta_brier, step=race_number)
                mlflow.log_metric("delta_log_loss", evaluation.decision.delta_log_loss, step=race_number)
                mlflow.log_metric("challenger_promoted", float(evaluation.decision.promote), step=race_number)

                if evaluation.decision.promote:
                    champion_path = update_champion_after_duel(
                        champion_path, challenger_path, evaluation.decision, NEW_YEAR, race_number
                    )
                    mlflow.log_artifact(str(champion_path), artifact_path="models")

                print(
                    f"[{NEW_YEAR}_{race_number}] Challenger "
                    f"{'promosso' if evaluation.decision.promote else 'rifiutato'}: {evaluation.decision.reason} | "
                    f"delta_brier={evaluation.decision.delta_brier:+.6f} | "
                    f"delta_log_loss={evaluation.decision.delta_log_loss:+.6f}"
                )

            # Il risultato della gara N entra soltanto nel modello per N+1.
            dnf_history_df = append_weekend_history(dnf_history_df, weekend_results)
            valid_history = dnf_history_df.dropna(subset=[DNF_TARGET, "race_date"])
            challenger_cutoff = pd.Timestamp(valid_history["race_date"].max())
            fit_and_log_challenger(dnf_history_df, frozen_config, challenger_cutoff, step=race_number)
            challenger_path = MODEL_DIR / "dnf_logistic_pending.joblib"

        latest_path = MODEL_DIR / "dnf_logistic_latest.joblib"
        save_champion_snapshot(champion_path, latest_path)
        mlflow.log_artifact(str(latest_path), artifact_path="models")


if __name__ == "__main__":
    mlflow_process = launch_mlflow_server()
    tracking_uri = f"http://{MLFLOW_HOST}:{MLFLOW_PORT}"
    print(f"[*] [DNF] Associazione client MLflow a {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        mlflow.set_experiment(DNF_EXPERIMENT)
        run_pipeline()
    except Exception:
        error_details = traceback.format_exc()
        print(error_details)
        Path("dnf_crash_telemetry.log").write_text(
            "=== DNF TRAINING ERROR REPORT ===\n" + error_details, encoding="utf-8"
        )
        sys.exit(1)
    finally:
        if mlflow_process:
            print(f"\nSessione MLflow disponibile su http://{MLFLOW_HOST}:{MLFLOW_PORT}")
            try:
                input("Premi [INVIO] per spegnere il server MLflow... ")
            except KeyboardInterrupt:
                pass
            mlflow_process.terminate()
            mlflow_process.wait()
