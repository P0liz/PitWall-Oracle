from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.utils import normalize_utc_timestamp

from .config import DNF_OPTUNA_TRIALS, GLOBAL_SEED, DNF_TARGET, MAX_DNF_PROB
from .dnf_features import DNF_CANDIDATE_FEATURES


# --------------------------#
# DNF Model
# --------------------------#
@dataclass(frozen=True)
class DNFModelConfig:
    features: tuple[str, ...]
    c: float
    penalty: str
    positive_class_weight: float

    def model_parameters(self) -> dict:
        return {"C": self.c, "penalty": self.penalty, "positive_class_weight": self.positive_class_weight}


@dataclass(frozen=True)
class DNFFittedModel:
    model: Pipeline
    config: DNFModelConfig
    training_rows: int
    training_races: int


@dataclass(frozen=True)
class DNFTrainingResult:
    model: Pipeline
    model_type: str
    features: list[str]
    model_parameters: dict
    selection_brier_score: float
    oos_brier_score: float
    oos_log_loss: float
    training_rows: int
    training_races: int
    evaluation_races: int


def make_pipeline(c: float, penalty: str, positive_class_weight: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(c),
                    l1_ratio=1.0 if penalty == "l1" else 0.0,
                    class_weight={0: 1.0, 1: float(positive_class_weight)},
                    solver="liblinear",
                    max_iter=1000,
                    random_state=GLOBAL_SEED,
                ),
            ),
        ]
    )


def config_from_training_result(result: DNFTrainingResult) -> DNFModelConfig:
    return DNFModelConfig(
        features=tuple(result.features),
        c=float(result.model_parameters["C"]),
        penalty=str(result.model_parameters["penalty"]),
        positive_class_weight=float(result.model_parameters["positive_class_weight"]),
    )


def fit_dnf_logistic(df: pd.DataFrame, config: DNFModelConfig) -> DNFFittedModel:
    clean = _validated_training_frame(df, config.features)
    model = make_pipeline(config.c, config.penalty, config.positive_class_weight)
    model.fit(clean[list(config.features)], clean[DNF_TARGET])
    return DNFFittedModel(
        model=model, config=config, training_rows=len(clean), training_races=clean["race_date"].nunique()
    )


def gp_evaluation_dates(df: pd.DataFrame) -> pd.Index:
    required = ["race_date", "session_type"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti per identificare i GP: {missing}")
    gp_rows = df[df["session_type"].astype(str).str.lower() == "race"]
    return pd.Index(gp_rows["race_date"].dropna().drop_duplicates().sort_values())


def history_before_gp(df: pd.DataFrame, gp_date: pd.Timestamp) -> pd.DataFrame:
    required = ["race_date", "year", "race_number", "session_type"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti per costruire lo storico DNF: {missing}")

    gp_rows = df[(df["race_date"] == gp_date) & (df["session_type"].astype(str).str.lower() == "race")]
    weekends = gp_rows[["year", "race_number"]].drop_duplicates()
    if len(weekends) != 1:
        raise ValueError(f"Il cutoff {gp_date} non identifica un unico GP")

    gp_year = weekends.iloc[0]["year"]
    gp_race_number = weekends.iloc[0]["race_number"]
    same_weekend = (df["year"] == gp_year) & (df["race_number"] == gp_race_number)
    return df[(df["race_date"] < gp_date) & ~same_weekend].copy()


# --------------------------#
# Probability computation
# --------------------------#
def _validated_training_frame(df: pd.DataFrame, features: Iterable[str] = DNF_CANDIDATE_FEATURES) -> pd.DataFrame:
    required = [*features, DNF_TARGET, "race_date", "year", "race_number", "session_type"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti per il modello DNF: {missing}")

    df = df.dropna(subset=[DNF_TARGET, "race_date"]).copy()
    df[DNF_TARGET] = df[DNF_TARGET].astype(int)
    invalid_targets = set(df[DNF_TARGET].unique()) - {0, 1}
    if invalid_targets:
        raise ValueError(f"Valori non binari in {DNF_TARGET}: {sorted(invalid_targets)}")
    if df[DNF_TARGET].nunique() < 2:
        raise ValueError("Servono esempi di entrambe le classi per addestrare il modello DNF")
    return df.sort_values("race_date").reset_index(drop=True)


def clip_dnf_probabilities(probabilities, max_probability: float = MAX_DNF_PROB) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Il modello DNF ha prodotto probabilità non finite")
    return np.clip(probabilities, 0.0, max_probability)


def compute_probabilities(prediction_df: pd.DataFrame, artifact: dict | None = None) -> np.ndarray:
    if artifact is None or "model" not in artifact:
        raise ValueError("La strategia logistic richiede un artefatto con il modello")

    features = list(artifact.get("features", DNF_CANDIDATE_FEATURES))
    model_frame = prediction_df.copy()
    missing = [feature for feature in features if feature not in model_frame.columns]
    if missing:
        raise ValueError(f"Feature DNF mancanti in prediction per logistic: {missing}")

    probabilities = artifact["model"].predict_proba(model_frame[features])[:, 1]
    return clip_dnf_probabilities(probabilities)


def _score_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    return (float(brier_score_loss(y_true, clipped)), float(log_loss(y_true, clipped, labels=[0, 1])))


# --------------------------------------------------#
# Hyperparameter optimization and training
# --------------------------------------------------#
def _selection_score(
    development: pd.DataFrame,
    features: list[str],
    c: float,
    penalty: str,
    positive_class_weight: float,
    min_training_races: int,
) -> float:
    race_dates = gp_evaluation_dates(development)
    y_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []

    for split_idx in range(min_training_races, len(race_dates)):
        validation_date = race_dates[split_idx]
        train = history_before_gp(development, validation_date)
        validation = development[
            (development["race_date"] == validation_date)
            & (development["session_type"].astype(str).str.lower() == "race")
        ]
        if train[DNF_TARGET].nunique() < 2 or validation.empty:
            continue
        model = make_pipeline(c, penalty, positive_class_weight)
        model.fit(train[features], train[DNF_TARGET])
        y_parts.append(validation[DNF_TARGET].to_numpy())
        probability_parts.append(clip_dnf_probabilities(model.predict_proba(validation[features])[:, 1]))

    if not y_parts:
        return float("nan")
    return _score_probabilities(np.concatenate(y_parts), np.concatenate(probability_parts))[0]


def _best_trial_parameters(study: optuna.Study) -> DNFModelConfig:
    parameters = study.best_trial.params
    features = [feature for feature in DNF_CANDIDATE_FEATURES if parameters[f"use_{feature}"]]
    return DNFModelConfig(
        features=tuple(features),
        c=float(parameters["C"]),
        penalty=str(parameters["penalty"]),
        positive_class_weight=float(parameters["positive_class_weight"]),
    )


def train_dnf_logistic(
    df: pd.DataFrame, n_trials: int = DNF_OPTUNA_TRIALS, min_training_races: int = 5, evaluation_races: int = 8
) -> DNFTrainingResult:
    """Ottimizza logistic su development e valuta una sola volta l'holdout temporale."""
    if n_trials < 1:
        raise ValueError("n_trials deve essere almeno 1")

    clean = _validated_training_frame(df)
    race_dates = gp_evaluation_dates(clean)
    if len(race_dates) <= min_training_races + 1:
        raise ValueError("Numero di gare insufficiente per separare sviluppo e valutazione")

    evaluation_races = min(evaluation_races, len(race_dates) - min_training_races - 1)
    evaluation_races = max(1, evaluation_races)
    outer_start = len(race_dates) - evaluation_races
    development = history_before_gp(clean, race_dates[outer_start])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED))
    study.enqueue_trial(
        {
            **{f"use_{feature}": True for feature in DNF_CANDIDATE_FEATURES},
            "C": 0.003,
            "penalty": "l2",
            "positive_class_weight": 1.0,
        }
    )

    def objective(trial: optuna.Trial) -> float:
        features = [
            feature for feature in DNF_CANDIDATE_FEATURES if trial.suggest_categorical(f"use_{feature}", [False, True])
        ]
        if not features:
            raise optuna.TrialPruned("Almeno una feature DNF deve essere selezionata")
        c = trial.suggest_float("C", 1e-4, 100.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        positive_class_weight = trial.suggest_float("positive_class_weight", 1.0, 3.0)
        score = _selection_score(development, features, c, penalty, positive_class_weight, min_training_races)
        if not np.isfinite(score):
            raise optuna.TrialPruned("Nessun fold di sviluppo valido")
        return score

    study.optimize(objective, n_trials=n_trials)
    if not study.best_trials:
        raise ValueError("Optuna non ha prodotto una configurazione DNF valida")
    config = _best_trial_parameters(study)
    features = list(config.features)

    y_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    for split_idx in range(outer_start, len(race_dates)):
        validation_date = race_dates[split_idx]
        train = history_before_gp(clean, validation_date)
        validation = clean[
            (clean["race_date"] == validation_date) & (clean["session_type"].astype(str).str.lower() == "race")
        ]
        if train[DNF_TARGET].nunique() < 2 or validation.empty:
            continue
        model = make_pipeline(config.c, config.penalty, config.positive_class_weight)
        model.fit(train[features], train[DNF_TARGET])
        probability_parts.append(clip_dnf_probabilities(model.predict_proba(validation[features])[:, 1]))
        y_parts.append(validation[DNF_TARGET].to_numpy())

    if not y_parts:
        raise ValueError("Nessuna previsione OOS prodotta nel periodo di valutazione")
    oos_brier_score, oos_log_loss = _score_probabilities(np.concatenate(y_parts), np.concatenate(probability_parts))

    fitted = fit_dnf_logistic(clean, config)
    return DNFTrainingResult(
        model=fitted.model,
        model_type="logistic",
        features=features,
        model_parameters={**config.model_parameters(), "optuna_trials": n_trials},
        selection_brier_score=float(study.best_value),
        oos_brier_score=oos_brier_score,
        oos_log_loss=oos_log_loss,
        training_rows=len(clean),
        training_races=len(race_dates),
        evaluation_races=evaluation_races,
    )


def save_dnf_artifact(result: DNFTrainingResult, path: Path, cutoff_date: pd.Timestamp) -> Path:
    if result.model_type != "logistic":
        raise ValueError("Solo artefatti DNF logistic sono supportati")
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_version": 4,
        "model": result.model,
        "model_type": "logistic",
        "model_parameters": result.model_parameters,
        "features": result.features,
        "target": DNF_TARGET,
        "cutoff_date": pd.Timestamp(cutoff_date).isoformat(),
        "selection_brier_score": result.selection_brier_score,
        "oos_brier_score": result.oos_brier_score,
        "oos_log_loss": result.oos_log_loss,
        "training_rows": result.training_rows,
        "training_races": result.training_races,
        "evaluation_races": result.evaluation_races,
    }
    joblib.dump(artifact, path)
    return path


def save_fitted_dnf_artifact(fitted: DNFFittedModel, path: Path, cutoff_date: pd.Timestamp) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_version": 5,
        "model": fitted.model,
        "model_type": "logistic",
        "model_parameters": fitted.config.model_parameters(),
        "features": list(fitted.config.features),
        "target": DNF_TARGET,
        "cutoff_date": pd.Timestamp(cutoff_date).isoformat(),
        "training_rows": fitted.training_rows,
        "training_races": fitted.training_races,
    }
    joblib.dump(artifact, path)
    return path


def load_dnf_artifact(path: Path) -> dict:
    artifact = joblib.load(path)
    required = {"features", "target", "cutoff_date", "model"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Artefatto DNF non valido, chiavi mancanti: {sorted(missing)}")
    model_type = artifact.get("model_type", "logistic")
    if model_type != "logistic":
        raise ValueError(f"Artefatto DNF non supportato: '{model_type}'")
    artifact["model_type"] = "logistic"
    artifact["cutoff_date"] = normalize_utc_timestamp(artifact["cutoff_date"], "cutoff_date")
    return artifact
