import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from src.dnf_features import DNF_HISTORY_FEATURES
from src.dnf_model import (
    DNF_CANDIDATE_FEATURES,
    DNFModelConfig,
    DNF_TARGET,
    DNFTrainingResult,
    fit_dnf_logistic,
    gp_evaluation_dates,
    history_before_gp,
    compute_probabilities,
    load_dnf_artifact,
    save_dnf_artifact,
    train_dnf_logistic,
)
from src.ranker_features import NON_FEATURE_COLUMNS


class ColumnCapturingModel:
    def __init__(self):
        self.columns = None

    def predict_proba(self, frame):
        self.columns = frame.columns.tolist()
        return np.column_stack((np.full(len(frame), 0.8), np.full(len(frame), 0.2)))


def make_history() -> pd.DataFrame:
    rows = []
    for race in range(14):
        for driver in range(4):
            row = {
                "race_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * race),
                "year": 2024,
                "race_number": race + 1,
                "session_type": "race",
                "grid_position": driver + 1,
                "team_dnf_rate": 0.03 + 0.05 * (driver // 2),
                "driver_dnf_rate": 0.01 * driver + 0.005 * race,
                "car_age_proxy": float(race * 50 + driver),
                "wet_affinity": 0.1 * driver,
                DNF_TARGET: int((race + driver) % 5 == 0),
            }
            for feature_index, feature in enumerate(DNF_CANDIDATE_FEATURES):
                row.setdefault(feature, float((race + driver + feature_index) % 7) / 7.0)
            rows.append(row)
    return pd.DataFrame(rows)


class DNFModelTests(unittest.TestCase):
    def test_gp_evaluation_excludes_sprints_but_later_training_retains_prior_sprints(self):
        frame = pd.DataFrame(
            {
                "race_date": pd.to_datetime(["2025-03-01", "2025-03-02", "2025-03-15", "2025-03-16"]),
                "year": [2025] * 4,
                "race_number": [1, 1, 2, 2],
                "session_type": ["sprint", "race", "sprint", "race"],
            }
        )

        evaluation_dates = gp_evaluation_dates(frame)
        training = history_before_gp(frame, pd.Timestamp("2025-03-16"))

        self.assertEqual(evaluation_dates.tolist(), [pd.Timestamp("2025-03-02"), pd.Timestamp("2025-03-16")])
        self.assertEqual(training["session_type"].tolist(), ["sprint", "race"])

    def test_fixed_config_refit_preserves_feature_order_and_parameters(self):
        history = make_history()
        config = DNFModelConfig(
            features=("team_dnf_rate", "grid_position"), c=0.25, penalty="l2", positive_class_weight=1.4
        )

        fitted = fit_dnf_logistic(history, config)

        self.assertEqual(fitted.config, config)
        self.assertEqual(fitted.training_rows, len(history))
        self.assertEqual(fitted.training_races, history["race_date"].nunique())
        model = fitted.model.named_steps["model"]
        self.assertEqual(model.C, 0.25)
        self.assertEqual(model.class_weight, {0: 1.0, 1: 1.4})

    def test_dnf_history_features_do_not_leak_into_ranker_inputs(self):
        self.assertEqual(len(DNF_CANDIDATE_FEATURES), len(set(DNF_CANDIDATE_FEATURES)))
        self.assertTrue(set(DNF_HISTORY_FEATURES).issubset(NON_FEATURE_COLUMNS))

    def test_logistic_inference_requires_a_model_artifact(self):
        prediction = pd.DataFrame({"driver_id": ["a", "b"]})
        with self.assertRaises(ValueError):
            compute_probabilities(prediction)

    def test_training_uses_nonempty_registry_subset_and_keeps_holdout_out_of_selection(self):
        history = make_history()
        changed_holdout = history.copy()
        holdout_dates = changed_holdout["race_date"].drop_duplicates().sort_values().iloc[-3:]
        changed_holdout.loc[changed_holdout["race_date"].isin(holdout_dates), DNF_TARGET] = 1

        result = train_dnf_logistic(history, n_trials=3, min_training_races=4, evaluation_races=3)
        changed_result = train_dnf_logistic(changed_holdout, n_trials=3, min_training_races=4, evaluation_races=3)

        self.assertTrue(result.features)
        self.assertTrue(set(result.features).issubset(DNF_CANDIDATE_FEATURES))
        self.assertEqual(result.evaluation_races, 3)
        self.assertEqual(result.features, changed_result.features)
        self.assertEqual(result.model_parameters, changed_result.model_parameters)
        self.assertEqual(result.selection_brier_score, changed_result.selection_brier_score)
        self.assertNotEqual(result.oos_brier_score, changed_result.oos_brier_score)

    def test_logistic_inference_uses_exact_feature_order_saved_in_artifact(self):
        model = ColumnCapturingModel()
        artifact = {
            "model": model,
            "model_type": "logistic",
            "features": ["wet_affinity", "team_dnf_rate", "grid_position"],
        }
        prediction = pd.DataFrame(
            {"grid_position": [5], "team_dnf_rate": [0.1], "wet_affinity": [0.3], "car_age_proxy": [120.0]}
        )

        probabilities = compute_probabilities(prediction, artifact=artifact)

        self.assertEqual(model.columns, ["wet_affinity", "team_dnf_rate", "grid_position"])
        self.assertAlmostEqual(probabilities[0], 0.2)

    def test_saved_logistic_artifact_round_trips_metadata_and_inference(self):
        result = DNFTrainingResult(
            model=ColumnCapturingModel(),
            model_type="logistic",
            features=["wet_affinity", "grid_position"],
            model_parameters={"C": 0.003, "penalty": "l2", "positive_class_weight": 1.0, "optuna_trials": 40},
            selection_brier_score=0.1,
            oos_brier_score=0.2,
            oos_log_loss=0.3,
            training_rows=20,
            training_races=5,
            evaluation_races=2,
        )
        cutoff = pd.Timestamp("2025-12-07")
        prediction = pd.DataFrame({"grid_position": [4], "wet_affinity": [0.6]})

        with TemporaryDirectory() as temporary_directory:
            artifact = load_dnf_artifact(save_dnf_artifact(result, Path(temporary_directory) / "dnf.joblib", cutoff))

        self.assertEqual(artifact["model_type"], "logistic")
        self.assertEqual(artifact["features"], ["wet_affinity", "grid_position"])
        self.assertEqual(artifact["model_parameters"], result.model_parameters)
        self.assertEqual(artifact["cutoff_date"], cutoff.tz_localize("UTC"))
        np.testing.assert_array_equal(compute_probabilities(prediction, artifact=artifact), np.array([0.2]))
        self.assertEqual(artifact["model"].columns, ["wet_affinity", "grid_position"])

    def test_load_accepts_legacy_logistic_artifact_without_model_type(self):
        legacy_artifact = {
            "model": ColumnCapturingModel(),
            "features": ["team_dnf_rate", "car_age_proxy"],
            "target": DNF_TARGET,
            "cutoff_date": "2025-12-07T00:00:00",
        }

        with TemporaryDirectory() as temporary_directory:
            legacy_path = Path(temporary_directory) / "legacy.joblib"
            joblib.dump(legacy_artifact, legacy_path)
            loaded = load_dnf_artifact(legacy_path)

        self.assertEqual(loaded["model_type"], "logistic")
        np.testing.assert_array_equal(
            compute_probabilities(pd.DataFrame({"team_dnf_rate": [0.1], "car_age_proxy": [200.0]}), artifact=loaded),
            np.array([0.2]),
        )

    def test_load_rejects_non_logistic_artifact(self):
        unsupported_artifact = {
            "model": ColumnCapturingModel(),
            "model_type": "gradient_boosting",
            "features": ["grid_position"],
            "target": DNF_TARGET,
            "cutoff_date": "2025-12-07T00:00:00",
        }

        with TemporaryDirectory() as temporary_directory:
            artifact_path = Path(temporary_directory) / "unsupported.joblib"
            joblib.dump(unsupported_artifact, artifact_path)
            with self.assertRaisesRegex(ValueError, "non supportato"):
                load_dnf_artifact(artifact_path)

    def test_load_rejects_nat_artifact_cutoff(self):
        artifact = {
            "model": ColumnCapturingModel(),
            "model_type": "logistic",
            "features": ["grid_position"],
            "target": DNF_TARGET,
            "cutoff_date": pd.NaT,
        }

        with TemporaryDirectory() as temporary_directory:
            artifact_path = Path(temporary_directory) / "nat-cutoff.joblib"
            joblib.dump(artifact, artifact_path)
            with self.assertRaisesRegex(ValueError, "cutoff_date"):
                load_dnf_artifact(artifact_path)


if __name__ == "__main__":
    unittest.main()
