from src.data.gold_layer import GoldLayer
from src.data.trainer import *
from src.data.data_loader import *
from ml_flow_auto import launch_mlflow_server, MLFLOW_HOST, MLFLOW_PORT
import asyncio
import fastf1
import pandas as pd
import numpy as np
from xgboost import XGBRanker
import optuna
import mlflow
import sys
import traceback

BASE_MODEL = "pitwall_oracle_base.json"
TOLLERANCE = 0.001
FORCE = True
RUN_HPO = True  # Imposta a True se vuoi rieseguire la ricerca parametri con Optuna


def run_hpo_optuna(trainer, n_trials=20, val_size=8):
    """
    Esegue l'ottimizzazione degli iperparametri garantendo uno split temporale pulito.
    """
    trainer.log.info("Avvio ottimizzazione iperparametri con Optuna (Temporal Split)...")

    # 1. Recuperiamo il dataset statico dal trainer e lo ordiniamo temporalmente
    df = trainer.train_df.sort_values("race_date").reset_index(drop=True)

    # 2. Calcoliamo il qid esattamente come hai proposto tu (un intero incrementale per GP)
    df["qid"] = pd.factorize(df["race_date"])[0]

    # 3. Prepariamo le feature e i target
    X_full = df.drop(TO_DROP + ["qid"], axis=1)
    y_full = df["target"]
    qid_full = df["qid"]

    # 4. Split Temporale basato sui QID
    unique_qids = qid_full.unique()
    val_qids = unique_qids[-val_size:]  # Gli ID delle ultime gare

    # Maschere booleane per dividere i dati senza rompere le gare
    is_val = qid_full.isin(val_qids)
    is_train = ~is_val

    # Dataset di addestramento per Optuna
    X_tr = X_full[is_train]
    y_tr = y_full[is_train]
    qid_tr = qid_full[is_train]

    # Dataset di validazione per Optuna
    X_va = X_full[is_val]
    y_va = y_full[is_val]
    qid_va = qid_full[is_val]

    # 5. La magia di Pandas: ricalcoliamo all'istante la dimensione dei gruppi per la validazione!
    # value_counts() conta quanti piloti ci sono in ogni gara del validation set.
    # sort_index() garantisce che l'ordine dei gruppi rispetti la sequenza temporale di qid_va.
    val_group_sizes = qid_va.value_counts().sort_index().to_numpy()

    # 6. Prepariamo i pesi per l'addestramento (un peso per gruppo/gara)
    df_tr = df[is_train]
    weights_tr = make_weights(df_tr.groupby("qid")["race_date"].first())

    def objective(trial):
        params = {
            "objective": "rank:ndcg",
            "tree_method": "hist",
            "missing": np.nan,
            "enable_categorical": True,
            # Parametri da ottimizzare
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        }

        model = XGBRanker(**params)

        # 1. Il modello impara SOLO sulla prima parte del dataset statico
        model.fit(X_tr, y_tr, qid=qid_tr, sample_weight=weights_tr, verbose=False)

        # 2. Predice su dati del "futuro" che non ha mai visto durante il fit
        preds = model.predict(X_va)

        # 3. Calcola l'NDCG reale di generalizzazione
        ndcg_score = compute_ndcg(y_va.to_numpy(), preds, group_sizes=val_group_sizes)

        return ndcg_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    trainer.log.info(f"Migliori parametri trovati da Optuna: {study.best_params}")
    return study.best_params


def run_pipeline():
    data_loader = DataLoader()

    # Inizializziamo MLflow Parent Run per l'intero ciclo di addestramento/test
    with mlflow.start_run(run_name=f"F1_Season_Simulation_{NEW_YEAR}") as parent_run:

        # ----------------- PHASE 1: STATIC TRAINING & HPO -----------------
        trainer_static = StaticTraining(data_loader)
        asyncio.run(trainer_static.prepare_data(force=FORCE))

        # Default
        best_params = {
            "objective": "rank:ndcg",
            "tree_method": "hist",
            "missing": np.nan,
            "enable_categorical": True,
            "learning_rate": 0.05,
            "max_depth": 4,
            "n_estimators": 200,
        }

        if RUN_HPO:
            # Eseguiamo Optuna e tracciamo i parametri migliori su MLflow
            best_params = run_hpo_optuna(trainer_static, n_trials=15)
            mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items()})

        trainer_static.log.info("Inizio training su dati statici con parametri ottimizzati...")
        # Applichiamo i parametri al trainer (assicurati che il tuo StaticTraining possa riceverli o istanziare XGBRanker con essi)
        trainer_static.ranker = XGBRanker(**best_params)
        base_model = trainer_static.train()

        # Salvataggio e logging del modello base
        trainer_static.save_artifacts(BASE_MODEL)
        mlflow.log_artifact(str(os.path.join(trainer_static.model_dir, BASE_MODEL)), artifact_path="models")

        # Preparazione del test set fisso
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

        tracker = PrequentialTracker()
        gold = GoldLayer()
        champion_path = trainer_dynamic.model_dir / BASE_MODEL

        old_schedule = fastf1.get_event_schedule(STATIC_ENDING_YEAR)
        previous_date = old_schedule.loc[old_schedule["Session5DateUtc"] <= pd.Timestamp.now(), "Session5DateUtc"].max()
        new_schedule = fastf1.get_event_schedule(NEW_YEAR)
        races = new_schedule.loc[new_schedule["Session5DateUtc"] <= pd.Timestamp.now(), "Session5DateUtc"]

        trainer_dynamic.log.info("Inizio testing su gare dinamiche...")

        for idx, date in enumerate(races):
            race_idx = idx + 1
            trainer_dynamic.log.info(f"Elaborazione gara del {date.date()}")

            # Carichiamo il Champion corrente
            champion_model = XGBRanker()
            champion_model.load_model(champion_path)

            # Generazione feature Gold per il GP corrente
            new_race_results = gold.build_features(NEW_YEAR, race_idx, force=FORCE)
            race_df = new_race_results[-1].dropna(subset=["target"])

            # Target encoding dinamico senza leakage
            cutoff_date = race_df["race_date"].iloc[0]
            for col in categorical_cols:
                race_df[col] = data_loader.apply_target_encoding(race_df, col, cutoff_date=cutoff_date)
            race_df["circuit_id"] = race_df["circuit_id"].astype(data_loader.circuit_dtype)

            # Predizione sul GP in corso con il Champion attuale
            X_new = race_df.drop(TO_DROP, axis=1)
            scores_pred = champion_model.predict(X_new)

            # Valutazione performance prequenziale (Out-of-sample)
            ndcg_prequential = compute_ndcg(race_df["target"].to_numpy(), scores_pred)
            tracker.log((NEW_YEAR, race_idx), ndcg_prequential)

            # Loggiamo la telemetria di gara su MLflow usando l'indice progressivo come step
            mlflow.log_metric("prequential_ndcg", ndcg_prequential, step=race_idx)
            mlflow.log_metric("cumulative_mean_ndcg", tracker.cumulative_mean(), step=race_idx)

            trainer_dynamic.log.info(
                f"[{NEW_YEAR}_{race_idx}] NDCG prequenziale: {ndcg_prequential:.4f} | "
                f"media cumulativa: {tracker.cumulative_mean():.4f}"
            )

            # Addestramento del Challenger con i dati aggiornati fino alla gara corrente
            asyncio.run(trainer_dynamic.prepare_data(previous_date, force=FORCE))

            # Istanziamo il challenger applicando i parametri ottimali trovati da Optuna
            trainer_dynamic.ranker = XGBRanker(**best_params)
            challenger = trainer_dynamic.train()

            # Confronto Champion vs Challenger su dataset di test fisso (Shadow Test)
            scores_champion = champion_model.predict(X_new)
            scores_challenger = challenger.predict(X_new)
            ndcg_champion_new = compute_ndcg(race_df["target"].to_numpy(), scores_champion)
            ndcg_challenger_new = compute_ndcg(race_df["target"].to_numpy(), scores_challenger)

            # Log del duello su MLflow
            mlflow.log_metric("champion_new_ndcg", ndcg_champion_new, step=race_idx)
            mlflow.log_metric("challenger_new_ndcg", ndcg_challenger_new, step=race_idx)

            assert champion_model is not challenger  # guardia esplicita, non deve mai fallire ora
            regressed = ndcg_challenger_new < ndcg_champion_new - TOLLERANCE

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

            previous_date = date  # Aggiornamento per la prossima iterazione

        # Calcolo finale delle metriche residue (Fattore sigma empirico per Monte Carlo)
        # Usiamo l'ultimo champion per calcolare i residui storici sul test set fisso
        final_champion = XGBRanker()
        final_champion.load_model(champion_path)
        final_preds = final_champion.predict(X_test_fixed)

        residuals = final_preds - y_test_fixed.to_numpy()
        sigma_empirico = np.std(residuals)

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
