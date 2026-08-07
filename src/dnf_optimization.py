from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .config import GLOBAL_SEED
from .dnf_evaluation import paired_race_bootstrap
from .dnf_model import (
    BENCHMARK_DNF_FEATURES,
    DEFAULT_BETA_PRIOR_STRENGTH,
    DNF_FEATURES,
    DNF_TARGET,
    GRADIENT_BOOSTING_PARAMS,
    MAX_DNF_PROB,
    W_AGE_PROXY,
    W_DNF_RATE,
    compute_global_rate,
    compute_heuristic_dnf_probabilities,
    ensure_dnf_target,
    fit_team_beta_state,
    make_gradient_boosting_pipeline,
    make_logistic_pipeline,
    predict_team_beta,
)

# TODO: semplificare l'ottimizzazione per lavorare solo sui parametri del loistic regressor
# Oltre all' Ablation study capire se la scelta del regressor rispetto al gradient boosting ha senso

TUNED_FAMILIES = (
    "team_beta_tuned",
    "heuristic_tuned",
    "logistic_legacy_tuned",
    "logistic_benchmark_tuned",
    "gradient_boosting_tuned",
)
REFERENCE_FAMILIES = (
    "global_rate",
    "team_beta_current",
    "heuristic_current",
    "logistic_current",
    "logistic_benchmark_current",
    "gradient_boosting_benchmark",
)
ALL_FAMILIES = (*REFERENCE_FAMILIES, *TUNED_FAMILIES)


@dataclass
class PostProcessor:
    method: str
    cap: float
    calibrator: object | None = None

    def transform(self, probabilities) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float)
        if self.method == "sigmoid" and self.calibrator is not None:
            logits = np.log(np.clip(values, 1e-8, 1 - 1e-8) / np.clip(1 - values, 1e-8, 1))
            values = self.calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic" and self.calibrator is not None:
            values = self.calibrator.predict(values)
        return np.clip(values, 0.0, self.cap)


def _features_for_family(family: str) -> list[str]:
    if family in {"logistic_current", "logistic_legacy_tuned"}:
        return list(DNF_FEATURES)
    if family in {
        "logistic_benchmark_current",
        "logistic_benchmark_tuned",
        "gradient_boosting_benchmark",
        "gradient_boosting_tuned",
    }:
        return list(BENCHMARK_DNF_FEATURES)
    return []


def _gradient_parameters(parameters: dict) -> dict:
    return {key: value for key, value in parameters.items() if key != "features"}


def _clean_data(data: pd.DataFrame) -> pd.DataFrame:
    clean = ensure_dnf_target(data)
    clean["race_date"] = pd.to_datetime(clean["race_date"])
    required = {"race_date", "team_id", DNF_TARGET, *DNF_FEATURES, *BENCHMARK_DNF_FEATURES}
    missing = required - set(clean.columns)
    if missing:
        raise ValueError(f"Colonne mancanti per l'ottimizzazione DNF: {sorted(missing)}")
    clean = clean.dropna(subset=["race_date", DNF_TARGET]).copy()
    clean[DNF_TARGET] = clean[DNF_TARGET].astype(int)
    return clean.sort_values(["race_date", "driver_id"]).reset_index(drop=True)


def _raw_probabilities(
    family: str, train: pd.DataFrame, validation: pd.DataFrame, parameters: dict, features: list[str] | None = None
) -> np.ndarray:
    if family == "global_rate":
        return np.full(len(validation), compute_global_rate(train), dtype=float)

    if family in {"team_beta_current", "team_beta_tuned"}:
        state = fit_team_beta_state(
            train,
            prior_strength=float(parameters.get("prior_strength", DEFAULT_BETA_PRIOR_STRENGTH)),
            lookback_races=parameters.get("lookback_races"),
            half_life_races=parameters.get("half_life_races"),
        )
        return predict_team_beta(state, validation, max_probability=1.0)

    if family in {"heuristic_current", "heuristic_tuned"}:
        return compute_heuristic_dnf_probabilities(
            validation,
            fallback_rate=compute_global_rate(train),
            team_weight=float(parameters["team_weight"]),
            driver_weight=float(parameters["driver_weight"]),
            age_weight=float(parameters["age_weight"]),
            max_probability=1.0,
        )

    selected_features = features or parameters.get("features") or _features_for_family(family)
    if family in {
        "logistic_current",
        "logistic_benchmark_current",
        "logistic_legacy_tuned",
        "logistic_benchmark_tuned",
    }:
        model = make_logistic_pipeline(
            c=float(parameters.get("C", 0.003)),
            penalty=str(parameters.get("penalty", "l2")),
            positive_class_weight=float(parameters.get("positive_class_weight", 1.0)),
        )
    elif family in {"gradient_boosting_benchmark", "gradient_boosting_tuned"}:
        model = make_gradient_boosting_pipeline(_gradient_parameters(parameters))
    else:
        raise ValueError(f"Famiglia DNF sconosciuta: {family}")

    model.fit(train[selected_features], train[DNF_TARGET])
    return model.predict_proba(validation[selected_features])[:, 1]


def prequential_predictions(
    data: pd.DataFrame,
    family: str,
    parameters: dict,
    validation_dates: pd.Index,
    min_training_races: int,
    features: list[str] | None = None,
    trial: optuna.Trial | None = None,
) -> pd.DataFrame:
    race_dates = pd.Index(data["race_date"].drop_duplicates().sort_values())
    rows = []
    running_losses = []
    for step, validation_date in enumerate(validation_dates):
        train_dates = race_dates[race_dates < validation_date]
        if len(train_dates) < min_training_races:
            continue
        train = data[data["race_date"].isin(train_dates)]
        validation = data[data["race_date"] == validation_date]
        if train[DNF_TARGET].nunique() < 2 or validation.empty:
            continue
        probabilities = _raw_probabilities(family, train, validation, parameters, features)
        y_true = validation[DNF_TARGET].to_numpy()
        running_losses.extend((y_true - probabilities) ** 2)
        rows.append(pd.DataFrame({"race_date": validation_date, DNF_TARGET: y_true, "raw_probability": probabilities}))
        if trial is not None:
            trial.report(float(np.mean(running_losses)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    if not rows:
        raise ValueError(f"Nessuna previsione prequenziale prodotta per {family}")
    return pd.concat(rows, ignore_index=True)


def _suggest_parameters(trial: optuna.Trial, family: str) -> dict:
    if family == "team_beta_tuned":
        lookback = trial.suggest_categorical("lookback_races", [0, 10, 20, 40])
        half_life = trial.suggest_categorical("half_life_races", [0, 10, 20, 40])
        return {
            "prior_strength": trial.suggest_float("prior_strength", 1.0, 200.0, log=True),
            "lookback_races": None if lookback == 0 else int(lookback),
            "half_life_races": None if half_life == 0 else float(half_life),
        }
    if family == "heuristic_tuned":
        return {
            "team_weight": trial.suggest_float("team_weight", 0.0, 1.0),
            "driver_weight": trial.suggest_float("driver_weight", 0.0, 1.0),
            "age_weight": trial.suggest_float("age_weight", 0.0, 1.0),
        }
    if family in {"logistic_legacy_tuned", "logistic_benchmark_tuned"}:
        return {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "positive_class_weight": trial.suggest_float("positive_class_weight", 1.0, 3.0),
        }
    if family == "gradient_boosting_tuned":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 25, 300, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 3, 31),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100, step=5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 3.0),
        }
    raise ValueError(f"Nessuno spazio di ricerca per {family}")


def tune_family(
    data: pd.DataFrame, family: str, validation_dates: pd.Index, min_training_races: int, n_trials: int
) -> tuple[dict, optuna.Study]:
    sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(10, max(2, n_trials // 4)))
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    def objective(trial: optuna.Trial) -> float:
        parameters = _suggest_parameters(trial, family)
        predictions = prequential_predictions(
            data, family, parameters, validation_dates, min_training_races, trial=trial
        )
        return float(brier_score_loss(predictions[DNF_TARGET], predictions["raw_probability"]))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    parameters = _suggested_to_runtime_parameters(family, study.best_params)
    return parameters, study


def _suggested_to_runtime_parameters(family: str, parameters: dict) -> dict:
    converted = dict(parameters)
    if family == "team_beta_tuned":
        converted["lookback_races"] = None if converted["lookback_races"] == 0 else int(converted["lookback_races"])
        converted["half_life_races"] = (
            None if converted["half_life_races"] == 0 else float(converted["half_life_races"])
        )
    if family == "gradient_boosting_tuned":
        converted["subsample_freq"] = 1
    return converted


def _fit_calibrator(method: str, probabilities, y_true):
    probabilities = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if method == "none":
        return None
    if method == "sigmoid":
        logits = np.log(np.clip(probabilities, 1e-8, 1 - 1e-8) / np.clip(1 - probabilities, 1e-8, 1))
        calibrator = LogisticRegression(C=1e6, random_state=GLOBAL_SEED)
        calibrator.fit(logits.reshape(-1, 1), y_true)
        return calibrator
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probabilities, y_true)
    return calibrator


def select_postprocessor(
    development_predictions: pd.DataFrame, calibration_start: pd.Timestamp
) -> tuple[PostProcessor, pd.DataFrame]:
    calibration_train = development_predictions[development_predictions["race_date"] < calibration_start]
    selection = development_predictions[development_predictions["race_date"] >= calibration_start]
    if calibration_train[DNF_TARGET].nunique() < 2 or selection.empty:
        return PostProcessor("none", MAX_DNF_PROB), pd.DataFrame()

    candidates = []
    for method in ("none", "sigmoid", "isotonic"):
        calibrator = _fit_calibrator(method, calibration_train["raw_probability"], calibration_train[DNF_TARGET])
        for cap in (0.30, 0.50, 1.00):
            postprocessor = PostProcessor(method, cap, calibrator)
            probabilities = postprocessor.transform(selection["raw_probability"])
            candidates.append(
                {
                    "method": method,
                    "cap": cap,
                    "brier_score": float(brier_score_loss(selection[DNF_TARGET], probabilities)),
                    "log_loss": float(
                        log_loss(selection[DNF_TARGET], np.clip(probabilities, 1e-8, 1 - 1e-8), labels=[0, 1])
                    ),
                }
            )
    table = pd.DataFrame(candidates).sort_values(["brier_score", "log_loss"])
    best = table.iloc[0]
    final_calibrator = _fit_calibrator(
        str(best["method"]), development_predictions["raw_probability"], development_predictions[DNF_TARGET]
    )
    return PostProcessor(str(best["method"]), float(best["cap"]), final_calibrator), table


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, frame in predictions.groupby("strategy"):
        probabilities = np.clip(frame["probability"].to_numpy(), 1e-8, 1 - 1e-8)
        y_true = frame[DNF_TARGET].to_numpy()
        rows.append(
            {
                "strategy": family,
                "brier_score": float(brier_score_loss(y_true, probabilities)),
                "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
                "mean_probability": float(probabilities.mean()),
                "observed_rate": float(y_true.mean()),
                "max_probability": float(probabilities.max()),
                "rows": len(frame),
                "races": frame["race_date"].nunique(),
            }
        )
    return pd.DataFrame(rows).sort_values("brier_score").reset_index(drop=True)


def _study_rows(family: str, study: optuna.Study) -> list[dict]:
    rows = []
    for trial in study.trials:
        rows.append(
            {
                "family": family,
                "trial_number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                **trial.params,
            }
        )
    return rows


def _feature_ablation(
    data: pd.DataFrame, family: str, parameters: dict, validation_dates: pd.Index, min_training_races: int
) -> pd.DataFrame:
    rows = []
    for removed in [None, *BENCHMARK_DNF_FEATURES]:
        features = [feature for feature in BENCHMARK_DNF_FEATURES if feature != removed]
        predictions = prequential_predictions(
            data, family, parameters, validation_dates, min_training_races, features=features
        )
        rows.append(
            {
                "family": family,
                "removed_feature": removed or "none",
                "features": ",".join(features),
                "brier_score": float(brier_score_loss(predictions[DNF_TARGET], predictions["raw_probability"])),
            }
        )
    return pd.DataFrame(rows)


def _build_artifact(
    data: pd.DataFrame, family: str, parameters: dict, postprocessor: PostProcessor, metrics: dict
) -> dict:
    features = list(parameters.get("features") or _features_for_family(family))
    artifact = {
        "artifact_version": 4,
        "strategy": family,
        "target": DNF_TARGET,
        "features": features,
        "model_parameters": parameters,
        "cutoff_date": pd.Timestamp(data["race_date"].max()).isoformat(),
        "calibration_method": postprocessor.method,
        "calibrator": postprocessor.calibrator,
        "max_dnf_probability": postprocessor.cap,
        "optimization_metrics": metrics,
    }
    if family in {"team_beta_current", "team_beta_tuned"}:
        artifact["model_type"] = "team_beta"
        artifact["team_beta_state"] = fit_team_beta_state(
            data,
            prior_strength=float(parameters.get("prior_strength", DEFAULT_BETA_PRIOR_STRENGTH)),
            lookback_races=parameters.get("lookback_races"),
            half_life_races=parameters.get("half_life_races"),
        )
    elif family == "heuristic_tuned":
        artifact["model_type"] = "heuristic"
    elif family == "global_rate":
        artifact["model_type"] = "global_rate"
        artifact["global_rate"] = compute_global_rate(data)
    else:
        if family in {"logistic_current", "logistic_legacy_tuned", "logistic_benchmark_tuned"}:
            model = make_logistic_pipeline(
                c=float(parameters.get("C", 0.003)),
                penalty=str(parameters.get("penalty", "l2")),
                positive_class_weight=float(parameters.get("positive_class_weight", 1.0)),
            )
            artifact["model_type"] = "logistic"
        else:
            model = make_gradient_boosting_pipeline(_gradient_parameters(parameters))
            artifact["model_type"] = "gradient_boosting"
        model.fit(data[features], data[DNF_TARGET])
        artifact["model"] = model
    return artifact


def fit_optimized_artifact_for_history(template: dict, history: pd.DataFrame) -> tuple[str, dict]:
    """Rifitta causalmente il candidato ottimizzato conservando calibrazione e cap."""
    family = str(template["strategy"])
    parameters = dict(template.get("model_parameters", {}))
    artifact = {
        key: value
        for key, value in template.items()
        if key not in {"model", "team_beta_state", "global_rate", "cutoff_date"}
    }
    artifact["cutoff_date"] = pd.Timestamp(history["race_date"].max()).isoformat()

    if family in {"team_beta_current", "team_beta_tuned"}:
        artifact["team_beta_state"] = fit_team_beta_state(
            history,
            prior_strength=float(parameters.get("prior_strength", DEFAULT_BETA_PRIOR_STRENGTH)),
            lookback_races=parameters.get("lookback_races"),
            half_life_races=parameters.get("half_life_races"),
        )
        return "team_beta", artifact
    if family == "heuristic_tuned":
        return "heuristic", artifact
    if family == "global_rate":
        artifact["global_rate"] = compute_global_rate(history)
        return "global_rate", artifact

    features = list(template["features"])
    if family in {"logistic_current", "logistic_legacy_tuned", "logistic_benchmark_tuned"}:
        model = make_logistic_pipeline(
            c=float(parameters.get("C", 0.003)),
            penalty=str(parameters.get("penalty", "l2")),
            positive_class_weight=float(parameters.get("positive_class_weight", 1.0)),
        )
        runtime_strategy = "logistic"
    else:
        model = make_gradient_boosting_pipeline(_gradient_parameters(parameters))
        runtime_strategy = "gradient_boosting"
    model.fit(history[features], history[DNF_TARGET])
    artifact["model"] = model
    return runtime_strategy, artifact


def optimize_dnf_configurations(
    data: pd.DataFrame,
    output_dir: Path,
    min_training_races: int = 5,
    holdout_races: int = 12,
    calibration_races: int = 10,
    trials: dict[str, int] | None = None,
    bootstrap_samples: int = 2000,
) -> dict:
    clean = _clean_data(data)
    race_dates = pd.Index(clean["race_date"].drop_duplicates().sort_values())
    if len(race_dates) <= min_training_races + holdout_races + calibration_races:
        raise ValueError("Numero di gare insufficiente per tuning, calibrazione e holdout")

    development_dates = race_dates[:-holdout_races]
    holdout_dates = race_dates[-holdout_races:]
    calibration_dates = development_dates[-calibration_races:]
    tuning_dates = development_dates[:-calibration_races]
    tuning_validation_dates = tuning_dates[min_training_races:]
    development_validation_dates = development_dates[min_training_races:]

    trial_budget = {
        "team_beta_tuned": 20,
        "heuristic_tuned": 30,
        "logistic_legacy_tuned": 40,
        "logistic_benchmark_tuned": 40,
        "gradient_boosting_tuned": 60,
        **(trials or {}),
    }
    parameters = {
        "global_rate": {},
        "team_beta_current": {"prior_strength": DEFAULT_BETA_PRIOR_STRENGTH},
        "logistic_current": {"C": 0.003, "penalty": "l2", "positive_class_weight": 1.0},
        "logistic_benchmark_current": {"C": 0.003, "penalty": "l2", "positive_class_weight": 1.0},
        "gradient_boosting_benchmark": dict(GRADIENT_BOOSTING_PARAMS),
        "heuristic_current": {"team_weight": W_DNF_RATE, "driver_weight": 0.0, "age_weight": W_AGE_PROXY},
    }
    studies = {}
    trial_rows = []
    for family in TUNED_FAMILIES:
        parameters[family], studies[family] = tune_family(
            clean, family, tuning_validation_dates, min_training_races, trial_budget[family]
        )
        trial_rows.extend(_study_rows(family, studies[family]))

    feature_ablation = pd.concat(
        [
            _feature_ablation(clean, family, parameters[family], tuning_validation_dates, min_training_races)
            for family in ("logistic_benchmark_tuned", "gradient_boosting_tuned")
        ],
        ignore_index=True,
    )
    for family in ("logistic_benchmark_tuned", "gradient_boosting_tuned"):
        best_ablation = feature_ablation[feature_ablation["family"] == family].sort_values("brier_score").iloc[0]
        parameters[family]["features"] = str(best_ablation["features"]).split(",")

    postprocessors = {family: PostProcessor("none", MAX_DNF_PROB) for family in REFERENCE_FAMILIES}
    calibration_rows = []
    for family in TUNED_FAMILIES:
        development_predictions = prequential_predictions(
            clean[clean["race_date"].isin(development_dates)],
            family,
            parameters[family],
            development_validation_dates,
            min_training_races,
        )
        postprocessors[family], table = select_postprocessor(
            development_predictions, pd.Timestamp(calibration_dates[0])
        )
        if not table.empty:
            table.insert(0, "family", family)
            calibration_rows.append(table)

    outer_rows = []
    for family in ALL_FAMILIES:
        raw = prequential_predictions(clean, family, parameters[family], holdout_dates, min_training_races)
        raw["probability"] = postprocessors[family].transform(raw["raw_probability"])
        raw["strategy"] = family
        outer_rows.append(raw)
    holdout_predictions = pd.concat(outer_rows, ignore_index=True)
    metrics = _metrics(holdout_predictions)

    tuned_metrics = metrics[metrics["strategy"].isin(TUNED_FAMILIES)]
    candidate = str(tuned_metrics.iloc[0]["strategy"])
    baseline = "team_beta_current"
    candidate_row = metrics[metrics["strategy"] == candidate].iloc[0]
    baseline_row = metrics[metrics["strategy"] == baseline].iloc[0]
    relative_skill = 1.0 - candidate_row["brier_score"] / baseline_row["brier_score"]
    candidate_predictions = holdout_predictions[holdout_predictions["strategy"] == candidate].reset_index(drop=True)
    baseline_predictions = holdout_predictions[holdout_predictions["strategy"] == baseline].reset_index(drop=True)
    paired = candidate_predictions[["race_date", DNF_TARGET]].copy()
    paired[candidate] = candidate_predictions["probability"]
    paired[baseline] = baseline_predictions["probability"]
    bootstrap = paired_race_bootstrap(paired, candidate, baseline, n_bootstrap=bootstrap_samples)
    checks = {
        "brier_skill_at_least_1pct": bool(relative_skill >= 0.01),
        "log_loss_not_worse": bool(candidate_row["log_loss"] <= baseline_row["log_loss"]),
        "bootstrap_ci_below_zero": bool(bootstrap["ci_upper"] < 0.0),
    }
    gate = {
        "passed": all(checks.values()),
        "candidate_strategy": candidate,
        "baseline_strategy": baseline,
        "recommended_strategy": candidate if all(checks.values()) else "team_beta",
        "relative_brier_skill": float(relative_skill),
        "checks": checks,
        "bootstrap": bootstrap,
        "tuning_races": int(len(tuning_dates)),
        "calibration_races": int(len(calibration_dates)),
        "holdout_races": int(len(holdout_dates)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trial_rows).to_csv(output_dir / "dnf_optimization_trials.csv", index=False)
    if calibration_rows:
        pd.concat(calibration_rows, ignore_index=True).to_csv(output_dir / "dnf_calibration_search.csv", index=False)
    feature_ablation.to_csv(output_dir / "dnf_feature_ablation.csv", index=False)
    holdout_predictions.to_csv(output_dir / "dnf_optimization_holdout_predictions.csv", index=False)
    metrics.to_csv(output_dir / "dnf_optimization_metrics.csv", index=False)

    import json

    configuration_payload = {
        family: {
            "parameters": parameters[family],
            "calibration_method": postprocessors[family].method,
            "max_dnf_probability": postprocessors[family].cap,
        }
        for family in ALL_FAMILIES
    }
    (output_dir / "dnf_optimized_parameters.json").write_text(
        json.dumps(configuration_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "dnf_optimization_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    artifact = _build_artifact(
        clean, candidate, parameters[candidate], postprocessors[candidate], candidate_row.to_dict()
    )
    joblib.dump(artifact, output_dir / "dnf_optimized_candidate.joblib")
    return {
        "gate": gate,
        "metrics": metrics,
        "parameters": configuration_payload,
        "candidate_artifact": output_dir / "dnf_optimized_candidate.joblib",
    }
