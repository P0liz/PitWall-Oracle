import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRanker
from .config import GLOBAL_SEED
from .trainer import TO_DROP, make_weights

FIXED_PARAMS = {
    "objective": "rank:ndcg",
    "tree_method": "hist",
    "random_state": GLOBAL_SEED,
    "missing": np.nan,
    "enable_categorical": True,
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
    X_full = df.drop(TO_DROP + ["qid"], axis=1)
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
            val_group_sizes_f = qid_va_f.value_counts().sort_index().to_numpy()
            fold_scores.append(compute_ndcg(y_va_f.to_numpy(), preds, group_sizes=val_group_sizes_f))
            best_iterations.append(model.best_iteration)

        # Diagnostica: se i best_iteration tra fold sono molto dispersi, il dataset
        # è probabilmente ancora troppo piccolo/rumoroso perché questo valore sia stabile.
        trial.set_user_attr("mean_best_iteration", int(np.mean(best_iterations)))
        trial.set_user_attr("std_best_iteration", float(np.std(best_iterations)))

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


def compute_precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 3) -> float:
    """
    Precision@K per ranking.

    Parameters
    ----------
    y_true : array-like
        Target (20=primo, 19=secondo, ...)
    y_score : array-like
        Score predetti dal modello.
    k : int
        Numero di posizioni da considerare.

    Returns
    -------
    float
    """

    # Top-k reali
    true_topk = np.argsort(-y_true)[:k]

    # Top-k predetti
    pred_topk = np.argsort(-y_score)[:k]

    hits = len(set(true_topk) & set(pred_topk))

    return hits / k


def compute_ndcg(
    y_true_rel: np.ndarray, y_pred_score: np.ndarray, group_sizes: np.ndarray | None = None, k: int = 22
) -> float:
    """
    NDCG@k coerente con l'obiettivo rank:ndcg di XGBRanker.

    y_true_rel: le label di rilevanza (Y = n_drivers - posizione + 1), NON le posizioni grezze --
                sklearn.ndcg_score assume "valore alto = piu' rilevante", stessa convenzione
                gia' discussa per il target del ranker.
    group_sizes: dimensione di ogni gruppo/gara, in ordine. Se None, si assume un'unica gara
                 (caso tipico della valutazione prequenziale: una gara alla volta).
    k: normalmente coincide col numero di piloti in griglia (~20); troncarlo piu' in basso
       avrebbe poco senso qui, a differenza dei sistemi di recommendation con liste lunghissime.

    NB: sklearn.metrics.ndcg_score vuole array 2D shape (n_queries, n_docs_per_query) --
    per gruppi di dimensione diversa (es. weekend con ritiri, DNS) non si puo' fare un unico
    array rettangolare: si itera gara per gara e si fa la media (stessa convenzione usata
    nella walk-forward CV su 2024-2025 -- media semplice tra le gare del fold, non pesata
    per numero di piloti).
    """
    from sklearn.metrics import ndcg_score

    y_true_rel = np.asarray(y_true_rel, dtype=float)
    y_pred_score = np.asarray(y_pred_score, dtype=float)

    if group_sizes is None:
        group_sizes = np.array([len(y_true_rel)])

    if group_sizes.sum() != len(y_true_rel):
        raise ValueError("group_sizes non copre tutte le righe passate a compute_ndcg")

    ndcgs = []
    start = 0
    for size in group_sizes:
        end = start + size
        true_slice = y_true_rel[start:end]
        pred_slice = y_pred_score[start:end]

        # ndcg_score richiede almeno 2 elementi rilevanti distinti per essere non-degenere;
        # con un solo pilota nel gruppo (caso limite, es. dati corrotti/singolo DNS) si salta.
        if size >= 2:
            ndcgs.append(ndcg_score(true_slice.reshape(1, -1), pred_slice.reshape(1, -1), k=min(k, size)))
        start = end

    if not ndcgs:
        return float("nan")

    return float(np.mean(ndcgs))
