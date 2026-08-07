from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .config import GLOBAL_SEED, ROLLING_DNF_WINDOW
from .dnf_model import (
    BENCHMARK_DNF_FEATURES,
    DNF_TARGET,
    compute_global_rate_probabilities,
    compute_heuristic_dnf_probabilities,
    compute_team_beta_probabilities,
    ensure_dnf_target,
    make_gradient_boosting_pipeline,
    make_logistic_pipeline,
    clip_dnf_probabilities,
)
from .utils import is_race_dnf

# TODO: semplificare come per dnf_optimization.py

SIMPLE_STRATEGIES = ("global_rate", "team_beta", "heuristic")
LOGISTIC_VARIANTS = {
    "logistic": ["rolling_tech_dnf_rate", "car_age_proxy"],
    "logistic_benchmark": BENCHMARK_DNF_FEATURES,
}
MODEL_STRATEGIES = (*SIMPLE_STRATEGIES, *LOGISTIC_VARIANTS, "gradient_boosting")


def load_cached_gold(data_dir: Path = Path("data_files/gold"), main_races_only: bool = False) -> pd.DataFrame:
    files = sorted(data_dir.glob("[0-9]*_features.parquet"))
    if main_races_only:
        files = [path for path in files if path.stem.endswith("_5_features")]
    if not files:
        raise FileNotFoundError(f"Nessun parquet Gold trovato in '{data_dir}'")

    parts = []
    for path in files:
        frame = pd.read_parquet(path)
        frame["source_file"] = path.name
        parts.append(frame)

    data = pd.concat(parts, ignore_index=True)
    data["race_date"] = pd.to_datetime(data["race_date"])

    # I parquet creati prima della migrazione conservano il vecchio target e
    # la vecchia rolling. Li riallineiamo offline dalla history grezza, senza
    # richiedere download o una rigenerazione distruttiva della cache.
    history_path = data_dir / "driver_team_history.parquet"
    if history_path.exists():
        history = pd.read_parquet(history_path)
        history["race_date"] = pd.to_datetime(history["race_date"])
        status_lookup = history[["race_date", "driver_id", "status_raw"]].drop_duplicates(
            ["race_date", "driver_id"], keep="last"
        )
        data = data.merge(status_lookup, on=["race_date", "driver_id"], how="left")
        migrated = ensure_dnf_target(data)
        canonical_target = data["status_raw"].map(is_race_dnf)
        data[DNF_TARGET] = canonical_target.where(data["status_raw"].notna(), migrated[DNF_TARGET]).astype(int)

        rolling_values = {}
        for race_date, team_id in data[["race_date", "team_id"]].drop_duplicates().itertuples(index=False):
            team_history = history[(history["race_date"] < race_date) & (history["team_id"] == team_id)].tail(
                ROLLING_DNF_WINDOW * 2
            )
            if team_history.empty:
                rolling_values[(race_date, team_id)] = np.nan
            else:
                is_dnf = team_history["status_raw"].map(is_race_dnf)
                rolling_values[(race_date, team_id)] = float(is_dnf.ewm(span=ROLLING_DNF_WINDOW * 2).mean().iloc[-1])
        data["rolling_tech_dnf_rate"] = [
            rolling_values[(race_date, team_id)]
            for race_date, team_id in data[["race_date", "team_id"]].itertuples(index=False)
        ]
        data = data.drop(columns=["status_raw"])
    else:
        data = ensure_dnf_target(data)

    data = data.dropna(subset=[DNF_TARGET, "race_date"]).copy()
    data[DNF_TARGET] = data[DNF_TARGET].astype(int)
    return data.sort_values(["race_date", "driver_id"]).reset_index(drop=True)


def generate_dnf_prequential_predictions(
    data: pd.DataFrame, min_training_races: int = 5, logistic_c: float = 0.003
) -> pd.DataFrame:
    clean = ensure_dnf_target(data).dropna(subset=[DNF_TARGET, "race_date"]).copy()
    clean["race_date"] = pd.to_datetime(clean["race_date"])
    clean = clean.sort_values("race_date").reset_index(drop=True)
    race_dates = pd.Index(clean["race_date"].drop_duplicates().sort_values())
    rows = []

    for split_idx in range(min_training_races, len(race_dates)):
        train = clean[clean["race_date"].isin(race_dates[:split_idx])]
        validation = clean[clean["race_date"] == race_dates[split_idx]].copy()
        if train[DNF_TARGET].nunique() < 2 or validation.empty:
            continue

        predictions = {
            "global_rate": compute_global_rate_probabilities(train, validation),
            "team_beta": compute_team_beta_probabilities(train, validation),
            "heuristic": compute_heuristic_dnf_probabilities(validation),
        }
        for name, features in LOGISTIC_VARIANTS.items():
            model = make_logistic_pipeline(logistic_c)
            model.fit(train[features], train[DNF_TARGET])
            predictions[name] = clip_dnf_probabilities(model.predict_proba(validation[features])[:, 1])
        gradient_boosting = make_gradient_boosting_pipeline()
        gradient_boosting.fit(train[BENCHMARK_DNF_FEATURES], train[DNF_TARGET])
        predictions["gradient_boosting"] = clip_dnf_probabilities(
            gradient_boosting.predict_proba(validation[BENCHMARK_DNF_FEATURES])[:, 1]
        )

        for row_idx, (_, validation_row) in enumerate(validation.iterrows()):
            result = {
                "race_date": validation_row["race_date"],
                "year": int(pd.Timestamp(validation_row["race_date"]).year),
                "driver_id": validation_row.get("driver_id"),
                "team_id": validation_row.get("team_id"),
                DNF_TARGET: int(validation_row[DNF_TARGET]),
            }
            result.update({name: float(values[row_idx]) for name, values in predictions.items()})
            rows.append(result)

    if not rows:
        raise ValueError("Nessuna previsione prequenziale prodotta")
    return pd.DataFrame(rows)


def summarize_probability_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    y_true = predictions[DNF_TARGET].to_numpy()
    probability_columns = [column for column in MODEL_STRATEGIES if column in predictions.columns]
    rows = []
    for strategy in probability_columns:
        probabilities = np.clip(predictions[strategy].to_numpy(dtype=float), 1e-8, 1.0 - 1e-8)
        rows.append(
            {
                "strategy": strategy,
                "brier_score": float(brier_score_loss(y_true, probabilities)),
                "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
                "mean_probability": float(probabilities.mean()),
                "observed_rate": float(y_true.mean()),
                "max_probability": float(probabilities.max()),
                "rows": int(len(predictions)),
                "races": int(predictions["race_date"].nunique()),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("brier_score").reset_index(drop=True)
    team_beta_brier = float(metrics.loc[metrics["strategy"] == "team_beta", "brier_score"].iloc[0])
    metrics["brier_skill_vs_team_beta"] = 1.0 - metrics["brier_score"] / team_beta_brier
    return metrics


def build_reliability_table(predictions: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    probability_columns = [column for column in MODEL_STRATEGIES if column in predictions.columns]
    parts = []
    for strategy in probability_columns:
        frame = predictions[[strategy, DNF_TARGET]].copy()
        frame["bin"] = pd.qcut(frame[strategy], q=n_bins, duplicates="drop")
        reliability = (
            frame.groupby("bin", observed=True)
            .agg(mean_probability=(strategy, "mean"), observed_rate=(DNF_TARGET, "mean"), count=(DNF_TARGET, "size"))
            .reset_index(drop=True)
        )
        reliability.insert(0, "strategy", strategy)
        parts.append(reliability)
    return pd.concat(parts, ignore_index=True)


def paired_race_bootstrap(
    predictions: pd.DataFrame, strategy: str, baseline: str, n_bootstrap: int = 2000, seed: int = GLOBAL_SEED
) -> dict:
    squared_error_delta = (predictions[DNF_TARGET] - predictions[strategy]) ** 2 - (
        predictions[DNF_TARGET] - predictions[baseline]
    ) ** 2
    race_delta = squared_error_delta.groupby(predictions["race_date"]).mean().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(race_delta, size=(n_bootstrap, len(race_delta)), replace=True).mean(axis=1)
    return {
        "mean_delta_brier": float(race_delta.mean()),
        "ci_lower": float(np.quantile(sampled, 0.025)),
        "ci_upper": float(np.quantile(sampled, 0.975)),
        "races": int(len(race_delta)),
        "bootstrap_samples": int(n_bootstrap),
    }


def evaluate_gradient_boosting_gate(
    predictions: pd.DataFrame, metrics: pd.DataFrame, min_relative_brier_skill: float = 0.01, n_bootstrap: int = 2000
) -> dict:
    baseline = "logistic"
    baseline_row = metrics.loc[metrics["strategy"] == baseline].iloc[0]
    candidate_row = metrics.loc[metrics["strategy"] == "gradient_boosting"].iloc[0]
    relative_skill = 1.0 - float(candidate_row["brier_score"]) / float(baseline_row["brier_score"])
    bootstrap = paired_race_bootstrap(
        predictions, strategy="gradient_boosting", baseline=baseline, n_bootstrap=n_bootstrap
    )
    checks = {
        "brier_skill_at_least_1pct": relative_skill >= min_relative_brier_skill,
        "log_loss_not_worse": float(candidate_row["log_loss"]) <= float(baseline_row["log_loss"]),
        "bootstrap_ci_below_zero": bootstrap["ci_upper"] < 0.0,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "recommended_strategy": "gradient_boosting" if passed else baseline,
        "candidate_strategy": "gradient_boosting",
        "baseline_strategy": baseline,
        "relative_brier_skill": relative_skill,
        "threshold": min_relative_brier_skill,
        "checks": checks,
        "bootstrap": bootstrap,
    }
