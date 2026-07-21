from src.data.gold_layer import GoldLayer
from src.trainer import *
from src.model_optimization import *
from src.data.data_loader import *
from src.config import TOLLERANCE, to_log
from ml_flow_auto import launch_mlflow_server, MLFLOW_HOST, MLFLOW_PORT
import asyncio
import fastf1
import pandas as pd
import numpy as np
from xgboost import XGBRanker
import mlflow
import sys
import traceback
import hashlib

BASE_MODEL = "pitwall_oracle_base.json"
FORCE = True
RUN_HPO = True  # Imposta a True se vuoi rieseguire la ricerca parametri con Optuna


def run_pipeline():
    data_loader = DataLoader()

    # Inizializziamo MLflow Parent Run per l'intero ciclo di addestramento/test
    with mlflow.start_run(run_name=f"F1_Season_Simulation_{NEW_YEAR}") as parent_run:

        # ----------------- PHASE 1: STATIC TRAINING & HPO -----------------
        trainer_static = StaticTraining(data_loader)
        trainer_static.log.info("Inizio pipeline training")
        asyncio.run(trainer_static.prepare_data(force=FORCE))
        mlflow.log_params(to_log)

        # Default
        best_params = {**FIXED_PARAMS, "learning_rate": 0.05, "max_depth": 4, "n_estimators": 200}

        if RUN_HPO:
            trainer_static.log.info("Avvio HPO con Optuna per il modello statico...")
            # Eseguiamo Optuna e tracciamo i parametri migliori su MLflow
            best_params, decay_rate = run_hpo_optuna(trainer_static, n_trials=15)
            mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items()})
            mlflow.log_param("optuna_decay_rate", decay_rate)

        trainer_static.log.info("Inizio training su dati statici con parametri ottimizzati...")
        # Applichiamo i parametri al trainer (assicurati che il tuo StaticTraining possa riceverli o istanziare XGBRanker con essi)
        trainer_static.decay_rate = decay_rate
        trainer_static.ranker = XGBRanker(**best_params)
        base_model = trainer_static.train()

        def dataset_hash(df: pd.DataFrame) -> str:
            return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

        mlflow.log_param("train_dataset_hash", dataset_hash(trainer_static.train_df))

        # Salvataggio e logging del modello base
        trainer_static.save_artifacts(BASE_MODEL)
        mlflow.log_artifact(str(os.path.join(trainer_static.model_dir, BASE_MODEL)), artifact_path="models")

        # Preparazione del test set fisso (usato solo per static)
        trainer_static.test_df = trainer_static.test_df.sort_values("race_date").reset_index(drop=True)
        trainer_static.test_df["qid"] = pd.factorize(trainer_static.test_df["race_date"])[0]
        X_test_fixed = trainer_static.test_df.drop(TO_DROP + ["qid"], axis=1)
        y_test_fixed = trainer_static.test_df["target"]
        test_group_sizes = trainer_static.test_df["qid"].value_counts().sort_index().to_numpy()
        # Calcolo metriche di partenza
        initial_ndcg = compute_ndcg(
            y_test_fixed.to_numpy(), base_model.predict(X_test_fixed), group_sizes=test_group_sizes
        )
        mlflow.log_metric("static_baseline_ndcg", initial_ndcg)

        # ----------------- PHASE 2: DYNAMIC PREQUENTIAL TESTING -----------------
        trainer_dynamic = DynamicTraining(data_loader)
        # Forziamo i parametri ottimali anche sul trainer dinamico per i futuri challenger
        trainer_dynamic.ranker_params = best_params
        trainer_dynamic.decay_rate = decay_rate

        tracker = PrequentialTracker()
        gold = GoldLayer()
        champion_path = trainer_dynamic.model_dir / BASE_MODEL

        new_schedule = fastf1.get_event_schedule(NEW_YEAR)
        races = new_schedule.loc[new_schedule["Session5DateUtc"] <= pd.Timestamp.now(), "Session5DateUtc"]

        trainer_dynamic.log.info("Inizio testing su gare dinamiche...")
        challenger = None
        # Metrics setup
        oos_residuals = []  # per sigma empirico
        duel_history = []  # cronologia duelli per regression moving average
        recent_ndcgs = []  # saving ndcgs to compute perquential moving average
        for idx, date in enumerate(races):
            race_idx = idx + 1
            trainer_dynamic.log.info(f"Elaborazione gara del {date.date()}")

            # Carichiamo il Champion corrente
            champion_model = XGBRanker()
            champion_model.load_model(champion_path)

            # Generazione feature Gold per il GP corrente
            new_race_results = gold.build_features(NEW_YEAR, race_idx, force=FORCE)
            race_df = new_race_results[-1].dropna(subset=["target"]).copy()

            # Target encoding dinamico senza leakage
            cutoff_date = race_df["race_date"].iloc[0]
            for col in categorical_cols:
                race_df[col] = data_loader.apply_target_encoding(race_df, col, cutoff_date=cutoff_date)
            race_df["circuit_id"] = race_df["circuit_id"].astype(data_loader.circuit_dtype)

            # Predizione sul GP in corso con il Champion attuale
            X_new = race_df.drop(TO_DROP, axis=1)
            scores_champion = champion_model.predict(X_new)

            # Valutazione della precision:
            precision3 = compute_precision_at_k(race_df["target"].to_numpy(), scores_champion, k=3)
            precision5 = compute_precision_at_k(race_df["target"].to_numpy(), scores_champion, k=5)

            mlflow.log_metric("precision_at_3", precision3, step=race_idx)
            mlflow.log_metric("precision_at_5", precision5, step=race_idx)

            # Valutazione performance prequenziale (Out-of-sample)
            ndcg_prequential = compute_ndcg(race_df["target"].to_numpy(), scores_champion)
            tracker.log((NEW_YEAR, race_idx), ndcg_prequential)

            # Loggiamo la telemetria di gara su MLflow usando l'indice progressivo come step
            recent_ndcgs.append(ndcg_prequential)
            recent_ndcgs = recent_ndcgs[-5:]
            moving_avg = np.mean(recent_ndcgs)
            mlflow.log_metric("moving_avg_ndcg", moving_avg, step=race_idx)
            mlflow.log_metric("prequential_ndcg", ndcg_prequential, step=race_idx)
            mlflow.log_metric("cumulative_mean_ndcg", tracker.cumulative_mean(), step=race_idx)

            trainer_dynamic.log.info(
                f"[{NEW_YEAR}_{race_idx}] NDCG prequenziale: {ndcg_prequential:.4f} | "
                f"media cumulativa: {tracker.cumulative_mean():.4f}"
            )

            # per il calcolo del sigma alla fine
            race_residuals = scores_champion - race_df["target"].to_numpy()
            oos_residuals.extend(race_residuals.tolist())
            mlflow.log_metric("race_residual_std", np.std(race_residuals), step=race_idx)

            if challenger is not None:
                # Confronto Champion vs Challenger su dataset di test fisso (Shadow Test)
                scores_challenger = challenger.predict(X_new)
                ndcg_champion_new = compute_ndcg(race_df["target"].to_numpy(), scores_champion)
                ndcg_challenger_new = compute_ndcg(race_df["target"].to_numpy(), scores_challenger)
                delta_ndcg = ndcg_challenger_new - ndcg_champion_new
                # Log del duello su MLflow
                mlflow.log_metric("delta_ndcg", delta_ndcg, step=race_idx)

                assert champion_model is not challenger  # guardia esplicita, non deve mai fallire ora
                # Reduce noise
                duel_history.append((ndcg_champion_new, ndcg_challenger_new))
                k = max(3, race_idx)
                duel_history = duel_history[-k:]

                mean_champ = np.mean([c for c, _ in duel_history])
                mean_chall = np.mean([ch for _, ch in duel_history])
                regressed = mean_chall < mean_champ - TOLLERANCE

                # Decisione muretto box: Promozione o Rifiuto
                if not regressed:
                    new_filename = f"pitwall_oracle_{NEW_YEAR}_{race_idx}.json"
                    trainer_dynamic.save_artifacts(new_filename)

                    # Aggiorniamo il puntatore locale
                    champion_path = trainer_dynamic.model_dir / new_filename

                    # Salviamo il nuovo Champion su MLflow
                    mlflow.log_artifact(str(champion_path), artifact_path=f"models/race_{race_idx}")
                    mlflow.log_metric("challenger_promoted", 1.0, step=race_idx)

                    trainer_dynamic.log.info(
                        f"[{NEW_YEAR}_{race_idx}] Challenger promosso! "
                        f"Champ: {ndcg_champion_new:.4f} vs Chall: {ndcg_challenger_new:.4f}"
                    )
                else:
                    mlflow.log_metric("challenger_promoted", 0.0, step=race_idx)
                    trainer_dynamic.log.info(
                        f"[{NEW_YEAR}_{race_idx}] Challenger RIFIUTATO per regressione. "
                        f"Champ: {ndcg_champion_new:.4f} vs Chall: {ndcg_challenger_new:.4f}"
                    )

            # Addestramento del Challenger con i dati aggiornati fino alla gara corrente (per prossima iter)
            asyncio.run(trainer_dynamic.prepare_data(date, force=FORCE))
            # Istanziamo il challenger applicando i parametri ottimali trovati da Optuna
            trainer_dynamic.ranker = XGBRanker(**best_params)
            challenger = trainer_dynamic.train()

        trainer_dynamic.save_artifacts("pitwall_oracle_latest.json")
        # Calcolo finale delle metriche residue (Fattore sigma empirico per Monte Carlo)
        sigma_empirico = np.std(np.array(oos_residuals))
        mlflow.log_metric("final_sigma_empirico", sigma_empirico)
        trainer_dynamic.log.info(f"Simulazione completata. Sigma empirico calcolato: {sigma_empirico:.4f}")


if __name__ == "__main__":
    # 1. Avvia il server di telemetria
    mlflow_process = launch_mlflow_server()

    # 2. Configura immediatamente il client MLflow
    tracking_uri = f"http://{MLFLOW_HOST}:{MLFLOW_PORT}"
    print(f"[*] [Client] Associazione del client a {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    # 3. Registra l'esperimento (ora che il server è garantito come attivo)
    try:
        mlflow.set_experiment("PitWall_Oracle_Prequential_Pipeline")
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
