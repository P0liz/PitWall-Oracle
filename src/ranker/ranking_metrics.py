from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def _ranking_arrays(y_true: Sequence[float], y_score: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_score, dtype=float)
    if truth.ndim != 1 or scores.ndim != 1 or truth.shape != scores.shape:
        raise ValueError("Target e score devono essere vettori monodimensionali della stessa lunghezza")
    if not np.isfinite(truth).all() or not np.isfinite(scores).all():
        raise ValueError("Target e score devono contenere solo valori finiti")
    return truth, scores


def _pair_indices(size: int, groups: Sequence[object] | None = None) -> tuple[np.ndarray, np.ndarray]:
    first, second = np.triu_indices(size, k=1)
    if groups is None:
        return first, second
    group_values = np.asarray(groups, dtype=object)
    if group_values.shape != (size,):
        raise ValueError("I gruppi devono avere la stessa lunghezza del ranking")
    valid = pd.notna(group_values[first]) & pd.notna(group_values[second])
    same_group = group_values[first] == group_values[second]
    return first[valid & same_group], second[valid & same_group]


def pairwise_accuracy(
    y_true: Sequence[float], y_score: Sequence[float], groups: Sequence[object] | None = None
) -> float:
    """Return concordant pair share; predicted ties receive half credit."""

    truth, scores = _ranking_arrays(y_true, y_score)
    first, second = _pair_indices(len(truth), groups)
    if not len(first):
        return float("nan")
    truth_delta = truth[first] - truth[second]
    comparable = truth_delta != 0.0
    if not comparable.any():
        return float("nan")
    predicted_delta = scores[first] - scores[second]
    products = truth_delta[comparable] * predicted_delta[comparable]
    return float((np.count_nonzero(products > 0.0) + 0.5 * np.count_nonzero(products == 0.0)) / len(products))


def mean_absolute_position_error(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    """Mean absolute error between true and predicted ranks, with average ranks for ties."""

    truth, scores = _ranking_arrays(y_true, y_score)
    if not len(truth):
        return float("nan")
    true_positions = pd.Series(truth).rank(method="average", ascending=False).to_numpy()
    predicted_positions = pd.Series(scores).rank(method="average", ascending=False).to_numpy()
    return float(np.mean(np.abs(true_positions - predicted_positions)))


def top_k_overlap(y_true: Sequence[float], y_score: Sequence[float], k: int) -> float:
    """Share of the true top-k set present in the predicted top-k set."""

    if k < 1:
        raise ValueError("k deve essere positivo")
    truth, scores = _ranking_arrays(y_true, y_score)
    effective_k = min(k, len(truth))
    if effective_k == 0:
        return float("nan")
    true_top = np.argsort(-truth, kind="stable")[:effective_k]
    predicted_top = np.argsort(-scores, kind="stable")[:effective_k]
    return float(len(set(true_top) & set(predicted_top)) / effective_k)


def ndcg(y_true: Sequence[float], y_score: Sequence[float], k: int | None = None) -> float:
    """Linear-gain NDCG, paired with XGBoost ``ndcg_exp_gain=False``."""

    truth, scores = _ranking_arrays(y_true, y_score)
    if len(truth) < 2:
        return float("nan")
    effective_k = None if k is None else min(k, len(truth))
    return float(ndcg_score(truth.reshape(1, -1), scores.reshape(1, -1), k=effective_k))


def race_ranking_metrics(
    y_true: Sequence[float], y_score: Sequence[float], raw_team_ids: Sequence[object]
) -> dict[str, float]:
    """Compute the complete ranker scorecard for one race query."""

    return {
        "pairwise_accuracy": pairwise_accuracy(y_true, y_score),
        "teammate_pairwise_accuracy": pairwise_accuracy(y_true, y_score, groups=raw_team_ids),
        "position_mae": mean_absolute_position_error(y_true, y_score),
        "ndcg_full": ndcg(y_true, y_score),
        "ndcg_at_5": ndcg(y_true, y_score, k=5),
        "ndcg_at_10": ndcg(y_true, y_score, k=10),
        "top_3_overlap": top_k_overlap(y_true, y_score, k=3),
        "top_5_overlap": top_k_overlap(y_true, y_score, k=5),
    }


def evaluate_grouped_rankings(
    frame: pd.DataFrame, scores: Sequence[float], *, query_column: str = "race_date", team_column: str = "raw_team_id"
) -> pd.DataFrame:
    """Evaluate predictions race by race without flattening different queries."""

    required = {query_column, team_column, "target"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Colonne necessarie alla valutazione mancanti: {missing}")
    predictions = np.asarray(scores, dtype=float)
    if predictions.shape != (len(frame),):
        raise ValueError("Il numero di score non coincide con le righe del dataframe")

    working = frame.copy()
    working["_ranking_score"] = predictions
    rows = []
    for query, race in working.groupby(query_column, sort=True):
        row = {query_column: query, "drivers": int(len(race))}
        for metadata in ("year", "race_number", "session_type"):
            if metadata in race:
                row[metadata] = race[metadata].iloc[0]
        row.update(race_ranking_metrics(race["target"], race["_ranking_score"], race[team_column]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(query_column, kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    mean_delta_pairwise: float
    mean_delta_teammate: float
    mean_delta_position_mae: float
    reason: str


def decide_promotion(duels: pd.DataFrame) -> PromotionDecision:
    """Conservative ranker-only promotion rule over paired race-level duels."""

    required = {
        "champion_pairwise_accuracy",
        "challenger_pairwise_accuracy",
        "champion_teammate_pairwise_accuracy",
        "challenger_teammate_pairwise_accuracy",
        "champion_position_mae",
        "challenger_position_mae",
    }
    missing = sorted(required - set(duels.columns))
    if missing or duels.empty:
        raise ValueError(f"Duelli insufficienti per la promozione; colonne mancanti: {missing}")

    delta_pairwise = float((duels["challenger_pairwise_accuracy"] - duels["champion_pairwise_accuracy"]).mean())
    delta_teammate = float(
        (duels["challenger_teammate_pairwise_accuracy"] - duels["champion_teammate_pairwise_accuracy"]).mean()
    )
    delta_mae = float((duels["challenger_position_mae"] - duels["champion_position_mae"]).mean())
    finite = np.isfinite([delta_pairwise, delta_teammate, delta_mae]).all()
    no_regression = finite and delta_pairwise >= 0.0 and delta_teammate >= 0.0 and delta_mae <= 0.0
    improved = delta_pairwise > 0.0 or delta_teammate > 0.0 or delta_mae < 0.0
    promote = bool(no_regression and improved)
    reason = (
        "migliora almeno una metrica senza regressioni ranker"
        if promote
        else "regressione o assenza di miglioramento nella scorecard ranker"
    )
    return PromotionDecision(promote, delta_pairwise, delta_teammate, delta_mae, reason)
