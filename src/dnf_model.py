from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import *

DNF_TARGET = "technical_dnf_target"
DNF_FEATURES = ["rolling_tech_dnf_rate", "car_age_proxy"]


@dataclass(frozen=True)
class DNFTrainingResult:
    model: Pipeline
    best_c: float
    best_class_weight: str | None
    oos_brier_score: float
    oos_log_loss: float
    baseline_brier_score: float
    brier_skill_score: float
    training_rows: int
    training_races: int


def _make_pipeline(c: float, class_weight: str | None) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=c, class_weight=class_weight, max_iter=1000, random_state=GLOBAL_SEED)),
        ]
    )


def _validated_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = [*DNF_FEATURES, DNF_TARGET, "race_date"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti per il modello DNF: {missing}")

    clean = df.dropna(subset=[DNF_TARGET, "race_date"]).copy()
    clean[DNF_TARGET] = clean[DNF_TARGET].astype(int)
    invalid_targets = set(clean[DNF_TARGET].unique()) - {0, 1}
    if invalid_targets:
        raise ValueError(f"Valori non binari in {DNF_TARGET}: {sorted(invalid_targets)}")
    if clean[DNF_TARGET].nunique() < 2:
        raise ValueError("Servono esempi di entrambe le classi per addestrare il modello DNF")
    return clean.sort_values("race_date").reset_index(drop=True)


def train_dnf_logistic(
    df: pd.DataFrame, c_values=DEFAULT_C_VALUES, class_weights=DEFAULT_CLASS_WEIGHTS, min_training_races: int = 5
) -> DNFTrainingResult:
    """Ottimizza la regressione logistica con validation expanding-window per gara."""
    clean = _validated_training_frame(df)
    race_dates = pd.Index(clean["race_date"].drop_duplicates().sort_values())
    candidates = []

    for c in c_values:
        for class_weight in class_weights:
            y_true_parts = []
            probability_parts = []
            baseline_probability_parts = []

            for split_idx in range(min_training_races, len(race_dates)):
                train_dates = race_dates[:split_idx]
                validation_date = race_dates[split_idx]
                train_part = clean[clean["race_date"].isin(train_dates)]
                validation_part = clean[clean["race_date"] == validation_date]

                if train_part[DNF_TARGET].nunique() < 2 or validation_part.empty:
                    continue

                model = _make_pipeline(float(c), class_weight)
                model.fit(train_part[DNF_FEATURES], train_part[DNF_TARGET])
                probabilities = model.predict_proba(validation_part[DNF_FEATURES])[:, 1]
                y_true_parts.append(validation_part[DNF_TARGET].to_numpy())
                probability_parts.append(probabilities)
                baseline_probability_parts.append(
                    np.full(len(validation_part), train_part[DNF_TARGET].mean(), dtype=np.float64)
                )

            if y_true_parts:
                y_true = np.concatenate(y_true_parts)
                probabilities = np.concatenate(probability_parts)
                baseline_probabilities = np.concatenate(baseline_probability_parts)
                candidates.append(
                    (
                        brier_score_loss(y_true, probabilities),
                        log_loss(y_true, probabilities, labels=[0, 1]),
                        brier_score_loss(y_true, baseline_probabilities),
                        float(c),
                        class_weight,
                    )
                )

    if candidates:
        best_brier, best_log_loss, baseline_brier, best_c, best_class_weight = min(
            candidates, key=lambda candidate: (candidate[0], candidate[1])
        )
    else:
        best_c, best_class_weight = 1.0, None
        best_brier, best_log_loss, baseline_brier = float("nan"), float("nan"), float("nan")

    brier_skill_score = (
        1.0 - best_brier / baseline_brier
        if np.isfinite(best_brier) and np.isfinite(baseline_brier) and baseline_brier > 0
        else float("nan")
    )

    model = _make_pipeline(best_c, best_class_weight)
    model.fit(clean[DNF_FEATURES], clean[DNF_TARGET])

    return DNFTrainingResult(
        model=model,
        best_c=best_c,
        best_class_weight=best_class_weight,
        oos_brier_score=float(best_brier),
        oos_log_loss=float(best_log_loss),
        baseline_brier_score=float(baseline_brier),
        brier_skill_score=float(brier_skill_score),
        training_rows=len(clean),
        training_races=len(race_dates),
    )


def save_dnf_artifact(result: DNFTrainingResult, path: Path, cutoff_date: pd.Timestamp) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": result.model,
        "features": DNF_FEATURES,
        "target": DNF_TARGET,
        "cutoff_date": pd.Timestamp(cutoff_date).isoformat(),
        "best_c": result.best_c,
        "best_class_weight": result.best_class_weight,
        "oos_brier_score": result.oos_brier_score,
        "oos_log_loss": result.oos_log_loss,
        "baseline_brier_score": result.baseline_brier_score,
        "brier_skill_score": result.brier_skill_score,
        "training_rows": result.training_rows,
        "training_races": result.training_races,
    }
    joblib.dump(artifact, path)
    return path


def load_dnf_artifact(path: Path) -> dict:
    artifact = joblib.load(path)
    required = {"model", "features", "target", "cutoff_date"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Artefatto DNF non valido, chiavi mancanti: {sorted(missing)}")
    return artifact
