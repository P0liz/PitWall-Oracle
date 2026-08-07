from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_CLASS_WEIGHTS, DEFAULT_C_VALUES, GLOBAL_SEED
from .utils import is_race_dnf

DNF_TARGET = "dnf_target"
LEGACY_DNF_TARGET = "technical_dnf_target"
DNF_FEATURES = ["rolling_tech_dnf_rate", "car_age_proxy"]
BENCHMARK_DNF_FEATURES = ["grid_position", "team_dnf_rate", "driver_dnf_rate", "car_age_proxy", "wet_affinity"]
DNF_STRATEGIES = ("none", "global_rate", "team_beta", "heuristic", "logistic", "gradient_boosting")
DEFAULT_DNF_STRATEGY = "heuristic"

GRADIENT_BOOSTING_PARAMS = {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 10}

MAX_DNF_PROB = 0.30
FALLBACK_DNF_RATE = 0.12
DEFAULT_BETA_PRIOR_STRENGTH = 20.0
W_DNF_RATE = 0.55
W_AGE_PROXY = 0.45


@dataclass(frozen=True)
class DNFTrainingResult:
    model: Pipeline
    model_type: str
    features: list[str]
    model_parameters: dict
    best_c: float
    best_class_weight: str | None
    selection_brier_score: float
    oos_brier_score: float
    oos_log_loss: float
    global_rate_brier_score: float
    team_beta_brier_score: float
    heuristic_brier_score: float
    baseline_strategy: str
    baseline_brier_score: float
    brier_skill_score: float
    training_rows: int
    training_races: int
    evaluation_races: int
    global_rate: float
    team_beta_state: dict


def ensure_dnf_target(df: pd.DataFrame) -> pd.DataFrame:
    """Restituisce una copia con il target canonico, migrando input storici."""
    if DNF_TARGET in df.columns:
        return df.copy()

    migrated = df.copy()
    if LEGACY_DNF_TARGET in migrated.columns:
        migrated[DNF_TARGET] = migrated[LEGACY_DNF_TARGET]
        return migrated
    if "status_raw" in migrated.columns:
        migrated[DNF_TARGET] = migrated["status_raw"].map(is_race_dnf).astype(int)
        return migrated

    raise ValueError(f"Impossibile costruire '{DNF_TARGET}': servono '{LEGACY_DNF_TARGET}' oppure 'status_raw'.")


def _make_pipeline(
    c: float, class_weight: str | dict | None, features: Iterable[str] = DNF_FEATURES, penalty: str = "l2"
) -> Pipeline:
    # ``features`` è accettato per rendere esplicito il contratto dei modelli
    # usati nell'ablation; la selezione delle colonne avviene nel chiamante.
    del features
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c,
                    class_weight=class_weight,
                    l1_ratio=1.0 if penalty == "l1" else 0.0,
                    solver="liblinear" if penalty == "l1" else "lbfgs",
                    max_iter=1000,
                    random_state=GLOBAL_SEED,
                ),
            ),
        ]
    )


def make_logistic_pipeline(
    c: float = 0.003,
    class_weight: str | dict | None = None,
    penalty: str = "l2",
    positive_class_weight: float | None = None,
) -> Pipeline:
    """Factory pubblica usata dagli script di ablation."""
    if positive_class_weight is not None:
        class_weight = {0: 1.0, 1: float(positive_class_weight)}
    return _make_pipeline(c, class_weight, penalty=penalty)


def make_gradient_boosting_pipeline(parameters: dict | None = None) -> Pipeline:
    """Replica il classificatore LightGBM del benchmark con preprocessing causale."""
    model_parameters = {**GRADIENT_BOOSTING_PARAMS, **(parameters or {})}
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", LGBMClassifier(**model_parameters, random_state=GLOBAL_SEED, n_jobs=-1, verbosity=-1)),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def _validated_training_frame(df: pd.DataFrame, features: Iterable[str] = DNF_FEATURES) -> pd.DataFrame:
    clean = ensure_dnf_target(df)
    required = [*features, DNF_TARGET, "race_date"]
    missing = [column for column in required if column not in clean.columns]
    if missing:
        raise ValueError(f"Colonne mancanti per il modello DNF: {missing}")

    clean = clean.dropna(subset=[DNF_TARGET, "race_date"]).copy()
    clean[DNF_TARGET] = clean[DNF_TARGET].astype(int)
    invalid_targets = set(clean[DNF_TARGET].unique()) - {0, 1}
    if invalid_targets:
        raise ValueError(f"Valori non binari in {DNF_TARGET}: {sorted(invalid_targets)}")
    if clean[DNF_TARGET].nunique() < 2:
        raise ValueError("Servono esempi di entrambe le classi per addestrare il modello DNF")
    return clean.sort_values("race_date").reset_index(drop=True)


def clip_dnf_probabilities(probabilities, max_probability: float = MAX_DNF_PROB) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Il modello DNF ha prodotto probabilità non finite")
    return np.clip(probabilities, 0.0, max_probability)


def compute_global_rate(history_df: pd.DataFrame | None) -> float:
    if history_df is None or history_df.empty:
        return FALLBACK_DNF_RATE
    history = ensure_dnf_target(history_df).dropna(subset=[DNF_TARGET])
    if history.empty:
        return FALLBACK_DNF_RATE
    return float(history[DNF_TARGET].mean())


def compute_global_rate_probabilities(history_df: pd.DataFrame | None, prediction_df: pd.DataFrame) -> np.ndarray:
    probability = compute_global_rate(history_df)
    return clip_dnf_probabilities(np.full(len(prediction_df), probability, dtype=np.float64))


def fit_team_beta_state(
    history_df: pd.DataFrame,
    prior_strength: float = DEFAULT_BETA_PRIOR_STRENGTH,
    team_column: str = "team_id",
    lookback_races: int | None = None,
    half_life_races: float | None = None,
) -> dict:
    if prior_strength <= 0:
        raise ValueError("prior_strength deve essere positivo")

    history = ensure_dnf_target(history_df).dropna(subset=[DNF_TARGET])
    if team_column not in history.columns:
        raise ValueError(f"Colonna team mancante per Beta-Binomial: '{team_column}'")

    needs_race_dates = lookback_races is not None or half_life_races is not None
    if needs_race_dates and "race_date" not in history.columns:
        raise ValueError("race_date è necessaria per lookback o decadimento team_beta")
    race_dates = (
        pd.Index(pd.to_datetime(history["race_date"]).drop_duplicates().sort_values())
        if needs_race_dates
        else pd.Index([])
    )
    if lookback_races is not None and len(race_dates) > lookback_races:
        history = history[pd.to_datetime(history["race_date"]).isin(race_dates[-lookback_races:])].copy()
        race_dates = race_dates[-lookback_races:]

    global_rate = compute_global_rate(history)
    alpha = global_rate * prior_strength
    weighted = history.dropna(subset=[team_column]).copy()
    if half_life_races is None:
        weighted["_weight"] = 1.0
    else:
        date_rank = {date: rank for rank, date in enumerate(race_dates)}
        latest_rank = len(race_dates) - 1
        ages = pd.to_datetime(weighted["race_date"]).map(lambda date: latest_rank - date_rank[pd.Timestamp(date)])
        weighted["_weight"] = np.power(0.5, ages / float(half_life_races))
    weighted["_weighted_dnf"] = weighted[DNF_TARGET] * weighted["_weight"]
    grouped = weighted.groupby(team_column, observed=True).agg(
        weighted_dnf=("_weighted_dnf", "sum"), effective_count=("_weight", "sum")
    )
    probabilities = ((grouped["weighted_dnf"] + alpha) / (grouped["effective_count"] + prior_strength)).to_dict()

    return {
        "global_rate": global_rate,
        "prior_strength": float(prior_strength),
        "team_column": team_column,
        "team_probabilities": probabilities,
        "lookback_races": lookback_races,
        "half_life_races": half_life_races,
        "training_rows": int(len(history)),
    }


def predict_team_beta(state: dict, prediction_df: pd.DataFrame, max_probability: float = MAX_DNF_PROB) -> np.ndarray:
    team_column = state.get("team_column", "team_id")
    if team_column not in prediction_df.columns:
        raise ValueError(f"Colonna team mancante in prediction: '{team_column}'")

    global_rate = float(state.get("global_rate", FALLBACK_DNF_RATE))
    probabilities = prediction_df[team_column].map(state.get("team_probabilities", {})).fillna(global_rate)
    return clip_dnf_probabilities(probabilities.to_numpy(dtype=np.float64), max_probability=max_probability)


def compute_team_beta_probabilities(
    history_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    prior_strength: float = DEFAULT_BETA_PRIOR_STRENGTH,
    lookback_races: int | None = None,
    half_life_races: float | None = None,
) -> np.ndarray:
    state = fit_team_beta_state(
        history_df, prior_strength, lookback_races=lookback_races, half_life_races=half_life_races
    )
    return predict_team_beta(state, prediction_df)


def compute_heuristic_dnf_probabilities(
    race_df: pd.DataFrame,
    fallback_rate: float = FALLBACK_DNF_RATE,
    team_weight: float = W_DNF_RATE,
    driver_weight: float = 0.0,
    age_weight: float = W_AGE_PROXY,
    max_probability: float = MAX_DNF_PROB,
) -> np.ndarray:
    team_rate_column = "rolling_tech_dnf_rate" if "rolling_tech_dnf_rate" in race_df.columns else "team_dnf_rate"
    required = {team_rate_column, "car_age_proxy"}
    if driver_weight > 0:
        required.add("driver_dnf_rate")
    missing = required - set(race_df.columns)
    if missing:
        raise ValueError(f"Feature mancanti per l'euristica DNF: {sorted(missing)}")

    base_rate = race_df[team_rate_column].fillna(fallback_rate)
    age = race_df["car_age_proxy"]
    if age.notna().sum() >= 2 and age.max() > age.min():
        normalized_age = (age - age.min()) / (age.max() - age.min())
    else:
        normalized_age = pd.Series(0.0, index=race_df.index)

    total_weight = team_weight + driver_weight + age_weight
    if total_weight <= 0:
        raise ValueError("Almeno un peso dell'euristica DNF deve essere positivo")
    driver_rate = (
        race_df["driver_dnf_rate"].fillna(fallback_rate) if driver_weight > 0 else pd.Series(0.0, index=race_df.index)
    )
    probabilities = (
        team_weight * base_rate + driver_weight * driver_rate + age_weight * normalized_age.fillna(0.0)
    ) / total_weight
    return clip_dnf_probabilities(probabilities.to_numpy(dtype=np.float64), max_probability=max_probability)


def apply_probability_postprocessing(probabilities, artifact: dict | None = None) -> np.ndarray:
    artifact = artifact or {}
    values = np.asarray(probabilities, dtype=np.float64)
    calibrator = artifact.get("calibrator")
    if calibrator is not None:
        if artifact.get("calibration_method") == "sigmoid":
            logits = np.log(np.clip(values, 1e-8, 1 - 1e-8) / np.clip(1 - values, 1e-8, 1))
            values = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
        elif isinstance(calibrator, IsotonicRegression):
            values = calibrator.predict(values)
        else:
            values = calibrator.predict(values)
    return clip_dnf_probabilities(values, max_probability=float(artifact.get("max_dnf_probability", MAX_DNF_PROB)))


def compute_strategy_probabilities(
    strategy: str, prediction_df: pd.DataFrame, history_df: pd.DataFrame | None = None, artifact: dict | None = None
) -> np.ndarray:
    if strategy not in DNF_STRATEGIES:
        raise ValueError(f"Strategia DNF non valida: '{strategy}'. Valori ammessi: {DNF_STRATEGIES}")
    if strategy == "none":
        return np.zeros(len(prediction_df), dtype=np.float64)
    if strategy == "global_rate":
        if artifact and "global_rate" in artifact:
            raw = np.full(len(prediction_df), artifact["global_rate"])
        else:
            raw = np.full(len(prediction_df), compute_global_rate(history_df))
        return apply_probability_postprocessing(raw, artifact)
    if strategy == "team_beta":
        if artifact and artifact.get("team_beta_state"):
            return apply_probability_postprocessing(
                predict_team_beta(artifact["team_beta_state"], prediction_df, max_probability=1.0), artifact
            )
        if history_df is None:
            raise ValueError("La strategia team_beta richiede history_df o un artefatto compatibile")
        return compute_team_beta_probabilities(history_df, prediction_df)
    if strategy == "heuristic":
        fallback = compute_global_rate(history_df)
        parameters = (artifact or {}).get("model_parameters", {})
        raw = compute_heuristic_dnf_probabilities(
            prediction_df,
            fallback,
            team_weight=float(parameters.get("team_weight", W_DNF_RATE)),
            driver_weight=float(parameters.get("driver_weight", 0.0)),
            age_weight=float(parameters.get("age_weight", W_AGE_PROXY)),
            max_probability=1.0,
        )
        return apply_probability_postprocessing(raw, artifact)

    if artifact is None or "model" not in artifact:
        raise ValueError(f"La strategia {strategy} richiede un artefatto con il modello")
    features = artifact.get("features", DNF_FEATURES)
    model_frame = prediction_df.copy()
    if "rolling_tech_dnf_rate" in features and "rolling_tech_dnf_rate" not in model_frame:
        if "team_dnf_rate" in model_frame:
            model_frame["rolling_tech_dnf_rate"] = model_frame["team_dnf_rate"]
    missing = [feature for feature in features if feature not in model_frame.columns]
    if missing:
        raise ValueError(f"Feature DNF mancanti in prediction per {strategy}: {missing}")
    probabilities = artifact["model"].predict_proba(model_frame[features])[:, 1]
    return apply_probability_postprocessing(probabilities, artifact)


def _score_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    return (float(brier_score_loss(y_true, clipped)), float(log_loss(y_true, clipped, labels=[0, 1])))


def _candidate_selection_score(
    development: pd.DataFrame, c: float, class_weight: str | None, min_training_races: int
) -> float:
    race_dates = pd.Index(development["race_date"].drop_duplicates().sort_values())
    y_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []

    for split_idx in range(min_training_races, len(race_dates)):
        train = development[development["race_date"].isin(race_dates[:split_idx])]
        validation = development[development["race_date"] == race_dates[split_idx]]
        if train[DNF_TARGET].nunique() < 2 or validation.empty:
            continue
        model = _make_pipeline(c, class_weight)
        model.fit(train[DNF_FEATURES], train[DNF_TARGET])
        y_parts.append(validation[DNF_TARGET].to_numpy())
        probability_parts.append(model.predict_proba(validation[DNF_FEATURES])[:, 1])

    if not y_parts:
        return float("nan")
    return _score_probabilities(np.concatenate(y_parts), np.concatenate(probability_parts))[0]


def train_dnf_logistic(
    df: pd.DataFrame,
    c_values=DEFAULT_C_VALUES,
    class_weights=DEFAULT_CLASS_WEIGHTS,
    min_training_races: int = 5,
    evaluation_races: int = 8,
    beta_prior_strength: float = DEFAULT_BETA_PRIOR_STRENGTH,
) -> DNFTrainingResult:
    """
    Seleziona gli iperparametri sul periodo di sviluppo e valuta una sola volta
    sulle ultime gare tramite expanding-window. Le metriche OOS non riutilizzano
    quindi gli stessi fold impiegati per scegliere ``C`` e ``class_weight``.
    """
    clean = _validated_training_frame(df)
    race_dates = pd.Index(clean["race_date"].drop_duplicates().sort_values())
    if len(race_dates) <= min_training_races + 1:
        raise ValueError("Numero di gare insufficiente per separare sviluppo e valutazione")

    evaluation_races = min(evaluation_races, len(race_dates) - min_training_races - 1)
    evaluation_races = max(1, evaluation_races)
    outer_start = len(race_dates) - evaluation_races
    development_dates = race_dates[:outer_start]
    development = clean[clean["race_date"].isin(development_dates)]

    candidates = []
    for c in c_values:
        for class_weight in class_weights:
            score = _candidate_selection_score(development, float(c), class_weight, min_training_races)
            if np.isfinite(score):
                candidates.append((score, float(c), class_weight))

    if candidates:
        selection_brier, best_c, best_class_weight = min(candidates, key=lambda candidate: candidate[0])
    else:
        selection_brier, best_c, best_class_weight = float("nan"), 1.0, None

    prediction_parts: dict[str, list[np.ndarray]] = {
        "logistic": [],
        "global_rate": [],
        "team_beta": [],
        "heuristic": [],
    }
    y_parts: list[np.ndarray] = []

    for split_idx in range(outer_start, len(race_dates)):
        train_dates = race_dates[:split_idx]
        validation_date = race_dates[split_idx]
        train = clean[clean["race_date"].isin(train_dates)]
        validation = clean[clean["race_date"] == validation_date]
        if train[DNF_TARGET].nunique() < 2 or validation.empty:
            continue

        model = _make_pipeline(best_c, best_class_weight)
        model.fit(train[DNF_FEATURES], train[DNF_TARGET])
        prediction_parts["logistic"].append(clip_dnf_probabilities(model.predict_proba(validation[DNF_FEATURES])[:, 1]))
        prediction_parts["global_rate"].append(compute_global_rate_probabilities(train, validation))
        prediction_parts["team_beta"].append(compute_team_beta_probabilities(train, validation, beta_prior_strength))
        prediction_parts["heuristic"].append(
            compute_heuristic_dnf_probabilities(validation, compute_global_rate(train))
        )
        y_parts.append(validation[DNF_TARGET].to_numpy())

    if not y_parts:
        raise ValueError("Nessuna previsione OOS prodotta nel periodo di valutazione")

    y_true = np.concatenate(y_parts)
    scores = {name: _score_probabilities(y_true, np.concatenate(parts)) for name, parts in prediction_parts.items()}
    baseline_strategy = min(("global_rate", "team_beta", "heuristic"), key=lambda name: scores[name][0])
    baseline_brier = scores[baseline_strategy][0]
    logistic_brier, logistic_log_loss = scores["logistic"]
    brier_skill_score = 1.0 - logistic_brier / baseline_brier if baseline_brier > 0 else float("nan")

    model = _make_pipeline(best_c, best_class_weight)
    model.fit(clean[DNF_FEATURES], clean[DNF_TARGET])
    team_beta_state = fit_team_beta_state(clean, beta_prior_strength)

    return DNFTrainingResult(
        model=model,
        model_type="logistic",
        features=list(DNF_FEATURES),
        model_parameters={"C": best_c, "class_weight": best_class_weight},
        best_c=best_c,
        best_class_weight=best_class_weight,
        selection_brier_score=float(selection_brier),
        oos_brier_score=logistic_brier,
        oos_log_loss=logistic_log_loss,
        global_rate_brier_score=scores["global_rate"][0],
        team_beta_brier_score=scores["team_beta"][0],
        heuristic_brier_score=scores["heuristic"][0],
        baseline_strategy=baseline_strategy,
        baseline_brier_score=baseline_brier,
        brier_skill_score=float(brier_skill_score),
        training_rows=len(clean),
        training_races=len(race_dates),
        evaluation_races=evaluation_races,
        global_rate=compute_global_rate(clean),
        team_beta_state=team_beta_state,
    )


def train_dnf_gradient_boosting(
    df: pd.DataFrame,
    min_training_races: int = 5,
    evaluation_races: int = 8,
    beta_prior_strength: float = DEFAULT_BETA_PRIOR_STRENGTH,
) -> DNFTrainingResult:
    """Addestra e valuta il LightGBM benchmark con expanding-window temporale."""
    clean = _validated_training_frame(df, BENCHMARK_DNF_FEATURES)
    race_dates = pd.Index(clean["race_date"].drop_duplicates().sort_values())
    if len(race_dates) <= min_training_races + 1:
        raise ValueError("Numero di gare insufficiente per separare sviluppo e valutazione")

    evaluation_races = min(evaluation_races, len(race_dates) - min_training_races - 1)
    evaluation_races = max(1, evaluation_races)
    outer_start = len(race_dates) - evaluation_races
    prediction_parts = {"gradient_boosting": [], "global_rate": [], "team_beta": [], "heuristic": []}
    y_parts: list[np.ndarray] = []

    for split_idx in range(outer_start, len(race_dates)):
        train = clean[clean["race_date"].isin(race_dates[:split_idx])]
        validation = clean[clean["race_date"] == race_dates[split_idx]]
        if train[DNF_TARGET].nunique() < 2 or validation.empty:
            continue

        model = make_gradient_boosting_pipeline()
        model.fit(train[BENCHMARK_DNF_FEATURES], train[DNF_TARGET])
        prediction_parts["gradient_boosting"].append(
            clip_dnf_probabilities(model.predict_proba(validation[BENCHMARK_DNF_FEATURES])[:, 1])
        )
        prediction_parts["global_rate"].append(compute_global_rate_probabilities(train, validation))
        prediction_parts["team_beta"].append(compute_team_beta_probabilities(train, validation, beta_prior_strength))
        prediction_parts["heuristic"].append(
            compute_heuristic_dnf_probabilities(validation, compute_global_rate(train))
        )
        y_parts.append(validation[DNF_TARGET].to_numpy())

    if not y_parts:
        raise ValueError("Nessuna previsione OOS prodotta nel periodo di valutazione")

    y_true = np.concatenate(y_parts)
    scores = {name: _score_probabilities(y_true, np.concatenate(parts)) for name, parts in prediction_parts.items()}
    baseline_strategy = min(("global_rate", "team_beta", "heuristic"), key=lambda name: scores[name][0])
    baseline_brier = scores[baseline_strategy][0]
    model_brier, model_log_loss = scores["gradient_boosting"]

    model = make_gradient_boosting_pipeline()
    model.fit(clean[BENCHMARK_DNF_FEATURES], clean[DNF_TARGET])
    team_beta_state = fit_team_beta_state(clean, beta_prior_strength)
    return DNFTrainingResult(
        model=model,
        model_type="gradient_boosting",
        features=list(BENCHMARK_DNF_FEATURES),
        model_parameters=dict(GRADIENT_BOOSTING_PARAMS),
        best_c=float("nan"),
        best_class_weight=None,
        selection_brier_score=float("nan"),
        oos_brier_score=model_brier,
        oos_log_loss=model_log_loss,
        global_rate_brier_score=scores["global_rate"][0],
        team_beta_brier_score=scores["team_beta"][0],
        heuristic_brier_score=scores["heuristic"][0],
        baseline_strategy=baseline_strategy,
        baseline_brier_score=baseline_brier,
        brier_skill_score=1.0 - model_brier / baseline_brier,
        training_rows=len(clean),
        training_races=len(race_dates),
        evaluation_races=evaluation_races,
        global_rate=compute_global_rate(clean),
        team_beta_state=team_beta_state,
    )


def save_dnf_artifact(result: DNFTrainingResult, path: Path, cutoff_date: pd.Timestamp) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_version": 3,
        "model": result.model,
        "model_type": result.model_type,
        "model_parameters": result.model_parameters,
        "features": result.features,
        "target": DNF_TARGET,
        "cutoff_date": pd.Timestamp(cutoff_date).isoformat(),
        "max_dnf_probability": MAX_DNF_PROB,
        "best_c": result.best_c,
        "best_class_weight": result.best_class_weight,
        "selection_brier_score": result.selection_brier_score,
        "oos_brier_score": result.oos_brier_score,
        "oos_log_loss": result.oos_log_loss,
        "global_rate_brier_score": result.global_rate_brier_score,
        "team_beta_brier_score": result.team_beta_brier_score,
        "heuristic_brier_score": result.heuristic_brier_score,
        "baseline_strategy": result.baseline_strategy,
        "baseline_brier_score": result.baseline_brier_score,
        "brier_skill_score": result.brier_skill_score,
        "global_rate": result.global_rate,
        "team_beta_state": result.team_beta_state,
        "training_rows": result.training_rows,
        "training_races": result.training_races,
        "evaluation_races": result.evaluation_races,
    }
    joblib.dump(artifact, path)
    return path


def load_dnf_artifact(path: Path) -> dict:
    artifact = joblib.load(path)
    required = {"features", "target", "cutoff_date"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Artefatto DNF non valido, chiavi mancanti: {sorted(missing)}")
    model_type = artifact.get("model_type", "logistic")
    required_payload = {
        "logistic": "model",
        "gradient_boosting": "model",
        "team_beta": "team_beta_state",
        "global_rate": "global_rate",
    }.get(model_type)
    if required_payload and required_payload not in artifact:
        raise ValueError(f"Artefatto DNF '{model_type}' privo del payload '{required_payload}'")
    return artifact
