"""
Demo: walk-forward fold con target encoding leak-free + pesi a decadimento
esponenziale + fit di XGBRanker. Usa 2 gare giocattolo solo per mostrare
la meccanica del loop; con 2 sole gare la CV non è ovviamente significativa.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRanker

LAMBDA_DECAY = 0.01  # da tarare in walk-forward CV, come da tue note


def make_weights(race_dates: pd.Series) -> np.ndarray:
    max_date = race_dates.max()
    days_elapsed = (max_date - race_dates).dt.days
    return np.exp(-LAMBDA_DECAY * days_elapsed)


def compute_target_encoding_map(
    group_col: str,  # "driver_id" | "team_id"
    cutoff_date: pd.Timestamp,  # esclusivo: solo dati STRETTAMENTE precedenti
    smoothing: int = 5,  # forza dello shrinkage verso la media globale
) -> dict:
    """
    Calcola la mappa {categoria -> % storica di podi}, usando solo dati con race_date < cutoff_date.
    Applica Bayesian/Laplace smoothing per gestire categorie con poco storico
    (es. rookie, team nuovo) senza valori estremi (0% o 100% su 1 sola gara).
    """
    history_df = pd.read_parquet("data_files/gold/driver_team_history.parquet")
    past = history_df.loc[history_df["race_date"] < cutoff_date]

    if past.empty:
        return {}  # cold start totale: nessuno storico ancora disponibile

    global_podium_rate = past["is_podium"].mean()

    stats = past.groupby(group_col)["is_podium"].agg(["sum", "count"])
    # Shrinkage: (podi_reali + k * media_globale) / (gare_reali + k)
    stats["encoded"] = (stats["sum"] + smoothing * global_podium_rate) / (stats["count"] + smoothing)

    return stats["encoded"].to_dict()


def apply_target_encoding(df: pd.DataFrame, group_col: str, cutoff_date: pd.Timestamp) -> pd.Series:
    """Applica la mappa calcolata sopra a un DataFrame Gold, gestendo i mai-visti."""
    encoding_map = compute_target_encoding_map(group_col, cutoff_date)

    history_df = pd.read_parquet("data_files/gold/driver_team_history.parquet")
    past = history_df.loc[history_df["race_date"] < cutoff_date]
    global_fallback = past["is_podium"].mean() if not past.empty else 0.0
    return df[group_col].map(encoding_map).fillna(global_fallback)


def train():
    df = pd.read_parquet("data_files/test_df.parquet").sort_values("race_number").reset_index(drop=True)

    categorical_cols = ["driver_id", "team_id"]

    # --- un singolo "fold" walk-forward: train = gara 1, val = gara 2 ---
    train_mask = df["race_number"] == 1
    val_mask = df["race_number"] == 2

    train_df, val_df = df[train_mask].copy(), df[val_mask].copy()

    # Target encoding rifittato SOLO su train di questo fold (mai sull'intero dataset)
    for col in categorical_cols:
        train_df[col] = apply_target_encoding(train_df, col, cutoff_date=train_df["race_date"].min())
        val_df[col] = apply_target_encoding(val_df, col, cutoff_date=val_df["race_date"].min())

    X_train = train_df.drop(["target", "race_number", "race_date"], axis=1)
    X_val = val_df.drop(["target", "race_number", "race_date"], axis=1)
    y_train, y_val = train_df["target"], val_df["target"]
    qid_train, qid_val = train_df["race_number"], val_df["race_number"]

    # ATTENZIONE: per rank:ndcg, sample_weight vuole UN PESO PER GRUPPO (per gara),
    # non uno per riga. Un peso per-riga fa fallire xgboost con un errore poco
    # leggibile ("group_weights.size() == group_ptr.size() - 1"). Dato che il tuo
    # peso dipende solo da race_date (uguale per tutti i piloti della stessa gara),
    # è comunque costante nel gruppo: basta un valore per gara.
    w_per_race = make_weights(train_df.groupby("race_number")["race_date"].first())

    ranker = XGBRanker(
        objective="rank:ndcg",
        tree_method="hist",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        eval_metric="ndcg@20",
        missing=np.nan,
        early_stopping_rounds=20,
    )

    ranker.fit(
        X_train,
        y_train,
        qid=qid_train,
        sample_weight=w_per_race,
        eval_set=[(X_val, y_val)],
        eval_qid=[qid_val],
        verbose=False,
    )

    print("Miglior iterazione:", ranker.best_iteration)
    print("Best NDCG@20 su val:", ranker.best_score)
    return ranker
