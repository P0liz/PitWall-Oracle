import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRanker
from .config import GLOBAL_SEED
from .ranker_model import make_weights
from .ranking_metrics import evaluate_grouped_rankings

FIXED_PARAMS = {
    "objective": "rank:ndcg",
    "tree_method": "hist",
    "random_state": GLOBAL_SEED,
    "missing": np.nan,
    "enable_categorical": True,
    "ndcg_exp_gain": False,
}


def temporal_folds(unique_qids, n_folds=4, min_train_races=15):
    """
    Genera fold in expanding window: ogni fold aggiunge blocchi di gare al training
    e valida sul blocco successivo, senza mai validare su gare precedenti al training.
    """
    unique_qids = sorted(unique_qids)
    n_races = len(unique_qids)
    remaining = n_races - min_train_races

    if remaining < n_folds:
        raise ValueError(
            f"Dataset troppo piccolo per {n_folds} fold con min_train_races={min_train_races} "
            f"({n_races} gare totali disponibili)."
        )

    fold_size = remaining // n_folds

    folds = []
    for i in range(n_folds):
        train_end = min_train_races + i * fold_size
        val_end = train_end + fold_size if i < n_folds - 1 else n_races
        train_qids = unique_qids[:train_end]
        val_qids = unique_qids[train_end:val_end]
        if len(val_qids) > 0:
            folds.append((train_qids, val_qids))
    return folds


def run_hpo_optuna(trainer, n_trials=20, n_folds=4, min_train_races=15):
    """
    Esegue l'ottimizzazione degli iperparametri con walk-forward CV (expanding window),
    includendo decay_rate nello spazio di ricerca ed early stopping per il numero di alberi.
    """
    trainer.log.info("Avvio ottimizzazione iperparametri con Optuna (Walk-Forward CV)...")

    # 1. Recuperiamo il dataset statico dal trainer e lo ordiniamo temporalmente
    df = trainer.train_df.sort_values("race_date").reset_index(drop=True)

    # 2. Calcoliamo il qid esattamente come un intero incrementale per GP
    df["qid"] = pd.factorize(df["race_date"])[0]

    # 3. Prepariamo le feature e i target
    X_full = trainer.feature_frame(df)
    y_full = df["target"]
    qid_full = df["qid"]
    unique_qids = qid_full.unique()

    # 4. Generiamo i fold in expanding window una sola volta, fuori dall'objective
    folds = temporal_folds(unique_qids, n_folds=n_folds, min_train_races=min_train_races)
    trainer.log.info(f"Generati {len(folds)} fold walk-forward (min_train_races={min_train_races}).")

    def objective(trial):
        params = {
            **FIXED_PARAMS,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "n_estimators": 500,  # tetto alto, il numero reale lo decide l'early stopping
            "early_stopping_rounds": 20,
        }
        decay_rate = trial.suggest_float("decay_rate", 0.0005, 0.015, log=True)

        fold_scores = []
        fold_teammate_scores = []
        fold_position_maes = []
        best_iterations = []

        for train_qids, val_qids in folds:
            is_tr = qid_full.isin(train_qids)
            is_va = qid_full.isin(val_qids)

            X_tr_f, y_tr_f, qid_tr_f = X_full[is_tr], y_full[is_tr], qid_full[is_tr]
            X_va_f, y_va_f, qid_va_f = X_full[is_va], y_full[is_va], qid_full[is_va]

            # reference_date esplicito = prima gara del fold di validazione corrente
            reference_date = df.loc[is_va, "race_date"].min()
            weights_tr_f = make_weights(df[is_tr].groupby("qid")["race_date"].first(), decay_rate, reference_date)

            model = XGBRanker(**params)
            model.fit(
                X_tr_f,
                y_tr_f,
                qid=qid_tr_f,
                sample_weight=weights_tr_f,
                eval_set=[(X_va_f, y_va_f)],
                eval_qid=[qid_va_f],
                verbose=False,
            )

            preds = model.predict(X_va_f)
            validation_metrics = evaluate_grouped_rankings(df.loc[is_va], preds)
            fold_scores.append(float(validation_metrics["pairwise_accuracy"].mean()))
            fold_teammate_scores.append(float(validation_metrics["teammate_pairwise_accuracy"].mean()))
            fold_position_maes.append(float(validation_metrics["position_mae"].mean()))
            best_iterations.append(model.best_iteration)

        # Diagnostica: se i best_iteration tra fold sono molto dispersi, il dataset
        # è probabilmente ancora troppo piccolo/rumoroso perché questo valore sia stabile.
        trial.set_user_attr("mean_best_iteration", int(np.rint(np.mean(best_iterations))) + 1)
        trial.set_user_attr("std_best_iteration", float(np.std(best_iterations)))
        trial.set_user_attr("mean_teammate_pairwise_accuracy", float(np.mean(fold_teammate_scores)))
        trial.set_user_attr("mean_position_mae", float(np.mean(fold_position_maes)))

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED))
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    tuned = best_trial.params.copy()
    decay_rate = tuned.pop("decay_rate")

    best_params = {
        **FIXED_PARAMS,
        **tuned,  # learning_rate, max_depth, subsample, colsample_bytree
        "n_estimators": best_trial.user_attrs["mean_best_iteration"],
        # niente early_stopping_rounds qui: il modello finale (statico o dinamico)
        # allena su tutti i dati disponibili, senza eval_set
    }

    trainer.log.info(
        f"Migliori parametri trovati da Optuna: {best_params} | decay_rate={decay_rate:.5f} | "
        f"std_best_iteration={best_trial.user_attrs['std_best_iteration']:.1f}"
    )

    return best_params, decay_rate
