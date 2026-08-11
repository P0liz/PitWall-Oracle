from src.data.gold_layer import GoldLayer
from src.trainer import DynamicTraining, StaticTraining, select_model_feature_frame
from src.model_optimization import FIXED_PARAMS, run_hpo_optuna
from src.data.data_loader import DataLoader, NEW_YEAR, CATEGORICAL_COLS
from src.config import DEFAULT_DECAY_RATE, to_log_ranker, TARGET_MULTIPLIER, RANKER_OPTUNA_TRIALS
from src.ranker_features import PRODUCTION_FEATURES
from src.ranking_metrics import decide_promotion, evaluate_grouped_rankings, race_ranking_metrics
from ml_flow_auto import launch_mlflow_server, MLFLOW_HOST, MLFLOW_PORT
import asyncio
import os
import fastf1
import pandas as pd
import numpy as np
from xgboost import XGBRanker
import mlflow
import sys
import traceback
import hashlib

BASE_MODEL = "pitwall_oracle_base.json"
LATEST_MODEL = "pitwall_oracle_latest.json"
PENDING_MODEL = "pitwall_oracle_pending.json"
FORCE = False
RUN_HPO = False


def run_pipeline():
    data_loader = DataLoader()
    selected_features = PRODUCTION_FEATURES
    target_train_multiplier = TARGET_MULTIPLIER

    # Inizializziamo MLflow Parent Run per l'intero ciclo di addestramento/test
    with mlflow.start_run(run_name=f"F1_Season_Simulation_{NEW_YEAR}") as parent_run:
        mlflow.set_tag("model_type", "ranking")
        mlflow.log_param("simulation_year", NEW_YEAR)
        mlflow.log_param("ranker_feature_count", len(selected_features))
        mlflow.log_param("ranker_primary_metric", "mean_race_pairwise_accuracy")

        # ----------------- PHASE 1: STATIC TRAINING & HPO -----------------
        trainer_static = StaticTraining(
            data_loader,
            feature_names=selected_features,
            target_year=NEW_YEAR,
            target_train_multiplier=target_train_multiplier,
        )
        trainer_static.log.info("Inizio pipeline training")
        static_train_df, static_test_df = asyncio.run(trainer_static.prepare_data(force=FORCE))
        static_df = pd.concat([static_train_df, static_test_df], ignore_index=True)
        mlflow.log_params(to_log_ranker)

        # Default
        best_params = {
            **FIXED_PARAMS,
            "learning_rate": 0.14453683332426176,
            "max_depth": 3,
            "subsample": 0.8797458349145832,
            "colsample_bytree": 0.7026515011475697,
            "n_estimators": 23,
        }
        decay_rate = DEFAULT_DECAY_RATE

        if RUN_HPO:
            trainer_static.log.info("Avvio HPO con Optuna per il modello statico...")
            # Eseguiamo Optuna e tracciamo i parametri migliori su MLflow
            best_params, decay_rate = run_hpo_optuna(trainer_static, n_trials=RANKER_OPTUNA_TRIALS)
            mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items()})
            mlflow.log_param("optuna_decay_rate", decay_rate)

        trainer_static.log.info("Inizio training su dati statici con parametri ottimizzati...")
        # Applichiamo i parametri al trainer (assicurati che il tuo StaticTraining possa riceverli o istanziare XGBRanker con essi)
        trainer_static.decay_rate = decay_rate
        trainer_static.ranker = XGBRanker(**best_params)
        base_model = trainer_static.train()

        def dataset_hash(df: pd.DataFrame) -> str:
            return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

        ranking_columns = trainer_static.feature_frame(trainer_static.train_df)
        mlflow.log_param("train_dataset_hash", dataset_hash(ranking_columns))

        # Salvataggio e logging del modello base
        trainer_static.save_artifacts(BASE_MODEL)
        mlflow.log_artifact(str(os.path.join(trainer_static.model_dir, BASE_MODEL)), artifact_path="models")

        # Scorecard sul test statico, sempre aggregata gara per gara.
        trainer_static.test_df = trainer_static.test_df.sort_values("race_date").reset_index(drop=True)
        trainer_static.test_df["qid"] = pd.factorize(trainer_static.test_df["race_date"])[0]
        X_test_fixed = trainer_static.feature_frame(trainer_static.test_df)
        static_metrics = evaluate_grouped_rankings(trainer_static.test_df, base_model.predict(X_test_fixed))
        for metric in (
            "pairwise_accuracy",
            "teammate_pairwise_accuracy",
            "position_mae",
            "ndcg_full",
            "ndcg_at_5",
            "ndcg_at_10",
            "top_3_overlap",
            "top_5_overlap",
        ):
            mlflow.log_metric(f"static_{metric}", float(static_metrics[metric].mean()))

        # ----------------- PHASE 2: DYNAMIC PREQUENTIAL TESTING -----------------
        trainer_dynamic = DynamicTraining(
            data_loader,
            feature_names=selected_features,
            target_year=NEW_YEAR,
            target_train_multiplier=target_train_multiplier,
        )
        # Forziamo i parametri ottimali anche sul trainer dinamico per i futuri challenger
        trainer_dynamic.ranker_params = best_params
        trainer_dynamic.decay_rate = decay_rate

        gold = GoldLayer()
        champion_path = trainer_dynamic.model_dir / BASE_MODEL

        new_schedule = fastf1.get_event_schedule(NEW_YEAR)
        races = new_schedule.loc[new_schedule["Session5DateUtc"] <= pd.Timestamp.now(), "Session5DateUtc"]

        trainer_dynamic.log.info("Inizio testing su gare dinamiche...")
        challenger = None
        champion_history: list[dict[str, float]] = []
        for idx, date in enumerate(races):
            race_idx = idx + 1
            trainer_dynamic.log.info(f"Elaborazione gara del {date.date()}")

            # Delete possibly old weather parquet
            if race_idx == len(races):
                dir = data_loader.gold.data_dir
                (dir / "silver" / f"{NEW_YEAR}_{race_idx}_5_clean_weather.parquet").unlink(missing_ok=True)
                (dir / "silver" / f"{NEW_YEAR}_{race_idx}_3_clean_weather.parquet").unlink(missing_ok=True)
                (dir / "bronze" / f"{NEW_YEAR}_{race_idx}_5_raw_weather.parquet").unlink(missing_ok=True)
                (dir / "bronze" / f"{NEW_YEAR}_{race_idx}_3_raw_weather.parquet").unlink(missing_ok=True)

            # Carichiamo il Champion corrente
            champion_model = XGBRanker()
            champion_model.load_model(champion_path)

            # Generazione feature Gold per il GP corrente
            new_race_results = gold.build_features(NEW_YEAR, race_idx, force=FORCE)
            race_df = new_race_results[-1].dropna(subset=["target"]).copy()

            # Target encoding dinamico senza leakage
            cutoff_date = race_df["race_date"].iloc[0]
            for col in CATEGORICAL_COLS:
                race_df[col] = data_loader.apply_target_encoding(race_df, col, cutoff_date=cutoff_date)
            race_df["circuit_id"] = race_df["circuit_id"].astype(data_loader.circuit_dtype)

            # Predizione sul GP in corso con il Champion attuale
            X_new_champion = select_model_feature_frame(champion_model, race_df)
            scores_champion = champion_model.predict(X_new_champion)

            champion_metrics = race_ranking_metrics(
                race_df["target"].to_numpy(), scores_champion, race_df["raw_team_id"].to_numpy()
            )
            champion_history.append(champion_metrics)
            for metric, value in champion_metrics.items():
                mlflow.log_metric(f"champion_{metric}", value, step=race_idx)
                mlflow.log_metric(
                    f"champion_moving_avg_{metric}",
                    float(np.mean([row[metric] for row in champion_history[-5:]])),
                    step=race_idx,
                )

            trainer_dynamic.log.info(
                f"[{NEW_YEAR}_{race_idx}] pairwise={champion_metrics['pairwise_accuracy']:.4f} | "
                f"teammate={champion_metrics['teammate_pairwise_accuracy']:.4f} | "
                f"position_mae={champion_metrics['position_mae']:.3f}"
            )

            if challenger is not None:
                # Confronto appaiato Champion vs Challenger sullo stesso GP OOS.
                X_new_challenger = select_model_feature_frame(challenger, race_df)
                scores_challenger = challenger.predict(X_new_challenger)
                challenger_metrics = race_ranking_metrics(
                    race_df["target"].to_numpy(), scores_challenger, race_df["raw_team_id"].to_numpy()
                )
                duel_row = {
                    **{f"champion_{metric}": value for metric, value in champion_metrics.items()},
                    **{f"challenger_{metric}": value for metric, value in challenger_metrics.items()},
                }
                for metric in champion_metrics:
                    mlflow.log_metric(
                        f"delta_{metric}", challenger_metrics[metric] - champion_metrics[metric], step=race_idx
                    )

                assert champion_model is not challenger  # guardia esplicita, non deve mai fallire ora
                decision = decide_promotion(pd.DataFrame([duel_row]))

                # Decisione muretto box: Promozione o Rifiuto
                if decision.promote:
                    new_filename = f"pitwall_oracle_{NEW_YEAR}_{race_idx}.json"
                    assert trainer_dynamic.ranker is challenger
                    trainer_dynamic.save_artifacts(new_filename)

                    # Aggiorniamo il puntatore locale
                    champion_path = trainer_dynamic.model_dir / new_filename

                    # Salviamo il nuovo Champion su MLflow
                    mlflow.log_artifact(str(champion_path), artifact_path="models")
                    mlflow.log_metric("challenger_promoted", 1.0, step=race_idx)

                    trainer_dynamic.log.info(
                        f"[{NEW_YEAR}_{race_idx}] Challenger promosso: {decision.reason} | "
                        f"delta_pairwise={decision.mean_delta_pairwise:+.4f} | "
                        f"delta_teammate={decision.mean_delta_teammate:+.4f} | "
                        f"delta_mae={decision.mean_delta_position_mae:+.3f}"
                    )
                else:
                    mlflow.log_metric("challenger_promoted", 0.0, step=race_idx)
                    trainer_dynamic.log.info(
                        f"[{NEW_YEAR}_{race_idx}] Challenger rifiutato: {decision.reason} | "
                        f"delta_pairwise={decision.mean_delta_pairwise:+.4f} | "
                        f"delta_teammate={decision.mean_delta_teammate:+.4f} | "
                        f"delta_mae={decision.mean_delta_position_mae:+.3f}"
                    )

            # Addestramento del Challenger con i dati aggiornati fino alla gara corrente (per prossima iter)
            asyncio.run(trainer_dynamic.prepare_data(static_df, date, force=FORCE))
            # Istanziamo il challenger applicando i parametri ottimali trovati da Optuna
            trainer_dynamic.ranker = XGBRanker(**best_params)
            challenger = trainer_dynamic.train()
            trainer_dynamic.save_artifacts(PENDING_MODEL)

        # Pubblica esclusivamente il champion già promosso. Il modello allenato
        # dopo l'ultimo GP resta pending fino alla prossima gara out-of-sample.
        published_champion = XGBRanker()
        published_champion.load_model(champion_path)
        latest_path = trainer_dynamic.model_dir / LATEST_MODEL
        published_champion.save_model(latest_path)
        mlflow.log_artifact(str(latest_path), artifact_path="models")
        trainer_dynamic.log.info(f"Champion pubblicato in {latest_path}")


if __name__ == "__main__":
    # 1. Avvia il server di telemetria
    mlflow_process = launch_mlflow_server()

    # 2. Configura immediatamente il client MLflow
    tracking_uri = f"http://{MLFLOW_HOST}:{MLFLOW_PORT}"
    print(f"[*] [Client] Associazione del client a {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    # 3. Registra l'esperimento (ora che il server è garantito come attivo)
    try:
        mlflow.set_experiment("PitWall_Oracle_Ranking")
    except Exception as e:
        print(f"[❌] Connessione a MLflow fallita: {e}")
        if mlflow_process:
            mlflow_process.terminate()
        sys.exit(1)

    # 4. Esegui la pipeline
    try:
        run_pipeline()
    except Exception as e:
        # Estrae l'intero stack trace come stringa
        error_details = traceback.format_exc()

        print("\n" + "=" * 50)
        print("[❌] CRITICAL FAILURE: Pipeline interrotta!")
        print("=" * 50)

        # Stampa a video l'errore completo (ottimo per lo sviluppo locale)
        print(error_details)

        # Salva la telemetria del crash su file (fondamentale per la produzione)
        with open("crash_telemetry.log", "w") as crash_log:
            crash_log.write("=== FATAL ERROR REPORT ===\n")
            crash_log.write(error_details)
    finally:
        # 5. Gestione del teardown del server
        if mlflow_process:
            print("\n" + "=" * 60)
            print(f"SESSIONE DI TELEMETRIA ATTIVA: http://{MLFLOW_HOST}:{MLFLOW_PORT}")
            print("=" * 60)
            try:
                input("\nPremi [INVIO] per spegnere il server MLflow e chiudere i box... ")
            except KeyboardInterrupt:
                pass
            print("[*] Spegnimento del server MLflow...")
            mlflow_process.terminate()
            mlflow_process.wait()
            print("[+] Server spento. Sessione conclusa.")
