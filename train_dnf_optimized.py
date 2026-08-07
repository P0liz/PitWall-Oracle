import asyncio
import hashlib
import sys
import traceback
from pathlib import Path

import fastf1
import mlflow
import numpy as np
import pandas as pd

from ml_flow_auto import MLFLOW_HOST, MLFLOW_PORT, launch_mlflow_server
from src.data.data_loader import DataLoader, NEW_YEAR
from src.dnf_model import DNF_TARGET, save_dnf_artifact, train_dnf_gradient_boosting, train_dnf_logistic

DNF_EXPERIMENT = "PitWall_Oracle_DNF"
FORCE = False
MODEL_DIR = Path("models")


def dataset_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def train_and_log_dnf_model(dnf_df: pd.DataFrame, year: int, model_race: int):
    """Addestra entrambi i modelli utilizzabili alla gara con soli dati precedenti."""
    valid_rows = dnf_df[dnf_df[DNF_TARGET].notna()]
    cutoff_date = valid_rows["race_date"].max()
    results = {"logistic": train_dnf_logistic(dnf_df), "gradient_boosting": train_dnf_gradient_boosting(dnf_df)}

    for model_type, result in results.items():
        artifact_prefix = f"dnf_{model_type}"
        artifact_name = (
            f"{artifact_prefix}_base.joblib" if model_race == 1 else f"{artifact_prefix}_{year}_{model_race}.joblib"
        )
        artifact_path = MODEL_DIR / artifact_name
        save_dnf_artifact(result, artifact_path, cutoff_date)

        metric_values = {
            "oos_brier_score": result.oos_brier_score,
            "oos_log_loss": result.oos_log_loss,
            "selection_brier_score": result.selection_brier_score,
            "global_rate_brier_score": result.global_rate_brier_score,
            "team_beta_brier_score": result.team_beta_brier_score,
            "heuristic_brier_score": result.heuristic_brier_score,
            "strongest_simple_baseline_brier_score": result.baseline_brier_score,
            "brier_skill_vs_strongest_baseline": result.brier_skill_score,
        }
        for metric_name, value in metric_values.items():
            if np.isfinite(value):
                mlflow.log_metric(f"{model_type}_{metric_name}", value, step=model_race)
        mlflow.log_metric(f"{model_type}_training_rows", result.training_rows, step=model_race)
        mlflow.log_metric(f"{model_type}_training_races", result.training_races, step=model_race)
        mlflow.log_metric(f"{model_type}_evaluation_races", result.evaluation_races, step=model_race)
        mlflow.set_tag(f"{model_type}_baseline_strategy_race_{model_race}", result.baseline_strategy)
        mlflow.log_artifact(str(artifact_path), artifact_path="models")

    logistic_result = results["logistic"]
    mlflow.log_metric("best_c", logistic_result.best_c, step=model_race)
    mlflow.log_metric("class_weight_balanced", float(logistic_result.best_class_weight == "balanced"), step=model_race)
    mlflow.log_metric("training_end_race", model_race - 1, step=model_race)
    return results, cutoff_date


def run_pipeline():
    data_loader = DataLoader()
    asyncio.run(data_loader.load_data(force=FORCE))
    dnf_history_df = data_loader.dnf_df.copy()

    schedule = fastf1.get_event_schedule(NEW_YEAR)
    completed_races = schedule.loc[schedule["Session5DateUtc"] <= pd.Timestamp.now(), "Session5DateUtc"]

    with mlflow.start_run(run_name=f"DNF_Season_Simulation_{NEW_YEAR}"):
        mlflow.set_tags({"model_type": "dnf", "simulation_year": str(NEW_YEAR)})
        mlflow.log_param("simulation_year", NEW_YEAR)
        mlflow.log_param("target", DNF_TARGET)
        mlflow.log_param("historical_dataset_hash", dataset_hash(dnf_history_df))

        # Modello per gara 1: usa esclusivamente le stagioni precedenti.
        latest_results, latest_cutoff_date = train_and_log_dnf_model(dnf_history_df, NEW_YEAR, model_race=1)

        for idx, _ in enumerate(completed_races):
            race_number = idx + 1
            race_results = data_loader.gold.build_features(NEW_YEAR, race_number, force=FORCE)

            # Il risultato della gara N entra soltanto nel modello per N+1.
            dnf_history_df = pd.concat([dnf_history_df, *race_results], ignore_index=True)
            latest_results, latest_cutoff_date = train_and_log_dnf_model(
                dnf_history_df, NEW_YEAR, model_race=race_number + 1
            )

        for model_type, result in latest_results.items():
            latest_path = MODEL_DIR / f"dnf_{model_type}_latest.joblib"
            save_dnf_artifact(result, latest_path, latest_cutoff_date)
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
