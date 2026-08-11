from dataclasses import dataclass

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


@dataclass(frozen=True)
class DNFProbabilityMetrics:
    brier_score: float
    log_loss: float


@dataclass(frozen=True)
class DNFPromotionDecision:
    promote: bool
    delta_brier: float
    delta_log_loss: float
    reason: str


def score_dnf_probabilities(targets, probabilities) -> DNFProbabilityMetrics:
    y_true = np.asarray(targets, dtype=int)
    predicted = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or predicted.shape != y_true.shape or y_true.size == 0:
        raise ValueError("Target e probabilità DNF devono essere vettori non vuoti della stessa dimensione")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("I target DNF devono essere binari")
    if not np.all(np.isfinite(predicted)) or np.any((predicted < 0.0) | (predicted > 1.0)):
        raise ValueError("Le probabilità DNF devono essere finite e comprese tra 0 e 1")

    clipped = np.clip(predicted, 1e-8, 1.0 - 1e-8)
    return DNFProbabilityMetrics(
        brier_score=float(brier_score_loss(y_true, clipped)), log_loss=float(log_loss(y_true, clipped, labels=[0, 1]))
    )


def decide_dnf_promotion(champion: DNFProbabilityMetrics, challenger: DNFProbabilityMetrics) -> DNFPromotionDecision:
    delta_brier = float(challenger.brier_score - champion.brier_score)
    delta_log_loss = float(challenger.log_loss - champion.log_loss)
    finite = np.isfinite([delta_brier, delta_log_loss]).all()
    promote = bool(finite and delta_brier < 0.0 and delta_log_loss <= 0.0)
    reason = (
        "Brier migliore senza regressione nella log-loss"
        if promote
        else "Brier non migliore o regressione nella log-loss"
    )
    return DNFPromotionDecision(promote, delta_brier, delta_log_loss, reason)
