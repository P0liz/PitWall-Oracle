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
from src.dnf_model import DNF_TARGET, save_dnf_artifact, train_dnf_logistic

DNF_EXPERIMENT = "PitWall_Oracle_DNF"
FORCE = False
MODEL_DIR = Path("models")


def dataset_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def train_and_log_dnf_model(dnf_df: pd.DataFrame, year: int, model_race: int):
    """Addestra il modello utilizzabile alla gara model_race con soli dati precedenti."""
    result = train_dnf_logistic(dnf_df)
    valid_rows = dnf_df[dnf_df[DNF_TARGET].notna()]
    cutoff_date = valid_rows["race_date"].max()
    artifact_name = "dnf_logistic_base.joblib" if model_race == 1 else f"dnf_logistic_{year}_{model_race}.joblib"
    artifact_path = MODEL_DIR / artifact_name
    save_dnf_artifact(result, artifact_path, cutoff_date)

    if np.isfinite(result.oos_brier_score):
        mlflow.log_metric("oos_brier_score", result.oos_brier_score, step=model_race)
    if np.isfinite(result.oos_log_loss):
        mlflow.log_metric("oos_log_loss", result.oos_log_loss, step=model_race)
    if np.isfinite(result.baseline_brier_score):
        mlflow.log_metric("baseline_brier_score", result.baseline_brier_score, step=model_race)
    if np.isfinite(result.brier_skill_score):
        mlflow.log_metric("brier_skill_score", result.brier_skill_score, step=model_race)
    mlflow.log_metric("best_c", result.best_c, step=model_race)
    mlflow.log_metric("class_weight_balanced", float(result.best_class_weight == "balanced"), step=model_race)
    mlflow.log_metric("training_rows", result.training_rows, step=model_race)
    mlflow.log_metric("training_races", result.training_races, step=model_race)
    mlflow.log_metric("training_end_race", model_race - 1, step=model_race)
    mlflow.log_artifact(str(artifact_path), artifact_path="models")
    return result, cutoff_date


def run_pipeline():
    data_loader = DataLoader()
    asyncio.run(data_loader.load(force=FORCE))
    dnf_history_df = data_loader.dnf_df.copy()

    schedule = fastf1.get_event_schedule(NEW_YEAR)
    completed_races = schedule.loc[schedule["Session5DateUtc"] <= pd.Timestamp.now(), "Session5DateUtc"]

    with mlflow.start_run(run_name=f"DNF_Season_Simulation_{NEW_YEAR}"):
        mlflow.set_tags({"model_type": "dnf", "simulation_year": str(NEW_YEAR)})
        mlflow.log_param("simulation_year", NEW_YEAR)
        mlflow.log_param("target", DNF_TARGET)
        mlflow.log_param("historical_dataset_hash", dataset_hash(dnf_history_df))

        # Modello per gara 1: usa esclusivamente le stagioni precedenti.
        latest_result, latest_cutoff_date = train_and_log_dnf_model(dnf_history_df, NEW_YEAR, model_race=1)

        for idx, _ in enumerate(completed_races):
            race_number = idx + 1
            race_results = data_loader.gold.build_features(NEW_YEAR, race_number, force=FORCE)

            # Il risultato della gara N entra soltanto nel modello per N+1.
            dnf_history_df = pd.concat([dnf_history_df, *race_results], ignore_index=True)
            latest_result, latest_cutoff_date = train_and_log_dnf_model(
                dnf_history_df,
                NEW_YEAR,
                model_race=race_number + 1,
            )

        latest_path = MODEL_DIR / "dnf_logistic_latest.joblib"
        save_dnf_artifact(latest_result, latest_path, latest_cutoff_date)
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
