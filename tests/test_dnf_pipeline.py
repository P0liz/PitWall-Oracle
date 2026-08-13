import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import train_dnf_optimized
from src.dnf.dnf_features import DNF_HISTORY_FEATURES
from src.dnf.dnf_metrics import DNFProbabilityMetrics, DNFPromotionDecision, decide_dnf_promotion
from src.dnf.dnf_model import (
    DNF_CANDIDATE_FEATURES,
    DNF_TARGET,
    DNFTrainingResult,
    compute_probabilities,
    gp_evaluation_dates,
    history_before_gp,
    load_dnf_artifact,
    save_dnf_artifact,
    train_dnf_logistic,
)
from src.ranker.ranker_features import NON_FEATURE_COLUMNS


class ColumnCapturingModel:
    def __init__(self, probabilities=(0.2,)):
        self.columns = None
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, frame):
        self.columns = frame.columns.tolist()
        probabilities = np.resize(self.probabilities, len(frame))
        return np.column_stack((1.0 - probabilities, probabilities))


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
                DNF_TARGET: int((race + driver) % 5 == 0),
            }
            for index, feature in enumerate(DNF_CANDIDATE_FEATURES):
                row.setdefault(feature, float((race + driver + index) % 7) / 7.0)
            rows.append(row)
    return pd.DataFrame(rows)


class DNFPipelineTests(unittest.TestCase):
    def test_gp_evaluation_excludes_sprints_but_training_keeps_prior_sprints(self):
        frame = pd.DataFrame(
            {
                "race_date": pd.to_datetime(["2025-03-01", "2025-03-02", "2025-03-15", "2025-03-16"]),
                "year": [2025] * 4,
                "race_number": [1, 1, 2, 2],
                "session_type": ["sprint", "race", "sprint", "race"],
            }
        )

        self.assertEqual(gp_evaluation_dates(frame).tolist(), [pd.Timestamp("2025-03-02"), pd.Timestamp("2025-03-16")])
        self.assertEqual(
            history_before_gp(frame, pd.Timestamp("2025-03-16"))["session_type"].tolist(), ["sprint", "race"]
        )

    def test_dnf_features_stay_out_of_ranker_and_inference_uses_artifact_order(self):
        self.assertTrue(set(DNF_HISTORY_FEATURES).issubset(NON_FEATURE_COLUMNS))
        model = ColumnCapturingModel()
        artifact = {
            "model": model,
            "model_type": "logistic",
            "features": ["wet_affinity", "team_dnf_rate", "grid_position"],
        }
        frame = pd.DataFrame(
            {"grid_position": [5], "team_dnf_rate": [0.1], "wet_affinity": [0.3], "car_age_proxy": [120.0]}
        )

        probabilities = compute_probabilities(frame, artifact=artifact)

        self.assertEqual(model.columns, artifact["features"])
        self.assertAlmostEqual(probabilities[0], 0.2)

    def test_feature_selection_does_not_use_the_holdout(self):
        history = make_history()
        changed = history.copy()
        holdout_dates = changed["race_date"].drop_duplicates().sort_values().iloc[-3:]
        changed.loc[changed["race_date"].isin(holdout_dates), DNF_TARGET] = 1

        result = train_dnf_logistic(history, n_trials=3, min_training_races=4, evaluation_races=3)
        changed_result = train_dnf_logistic(changed, n_trials=3, min_training_races=4, evaluation_races=3)

        self.assertTrue(set(result.features).issubset(DNF_CANDIDATE_FEATURES))
        self.assertEqual(result.features, changed_result.features)
        self.assertEqual(result.model_parameters, changed_result.model_parameters)
        self.assertEqual(result.selection_brier_score, changed_result.selection_brier_score)
        self.assertNotEqual(result.oos_brier_score, changed_result.oos_brier_score)

    def test_artifact_round_trip_preserves_cutoff_features_and_inference(self):
        result = DNFTrainingResult(
            model=ColumnCapturingModel(),
            model_type="logistic",
            features=["wet_affinity", "grid_position"],
            model_parameters={"C": 0.003, "penalty": "l2", "positive_class_weight": 1.0},
            selection_brier_score=0.1,
            oos_brier_score=0.2,
            oos_log_loss=0.3,
            training_rows=20,
            training_races=5,
            evaluation_races=2,
        )
        with TemporaryDirectory() as temp_dir:
            path = save_dnf_artifact(result, Path(temp_dir) / "dnf.joblib", pd.Timestamp("2025-12-07"))
            artifact = load_dnf_artifact(path)

        self.assertEqual(artifact["features"], result.features)
        self.assertEqual(artifact["cutoff_date"], pd.Timestamp("2025-12-07", tz="UTC"))
        np.testing.assert_array_equal(
            compute_probabilities(pd.DataFrame({"grid_position": [4], "wet_affinity": [0.6]}), artifact=artifact),
            np.array([0.2]),
        )

    def test_promotion_requires_brier_improvement_without_log_loss_regression(self):
        champion = DNFProbabilityMetrics(0.10, 0.40)
        cases = (
            (DNFProbabilityMetrics(0.09, 0.39), True),
            (DNFProbabilityMetrics(0.10, 0.39), False),
            (DNFProbabilityMetrics(0.09, 0.41), False),
        )
        for challenger, expected in cases:
            with self.subTest(challenger=challenger):
                self.assertEqual(decide_dnf_promotion(champion, challenger).promote, expected)

    def test_duel_rejects_noncausal_artifact(self):
        gp = pd.DataFrame({"race_date": [pd.Timestamp("2026-03-08")], "grid_position": [1.0], DNF_TARGET: [0]})
        artifact = {
            "model": ColumnCapturingModel(),
            "features": ["grid_position"],
            "target": DNF_TARGET,
            "cutoff_date": pd.Timestamp("2026-03-08"),
        }
        with self.assertRaisesRegex(ValueError, "Leakage"):
            train_dnf_optimized.evaluate_dnf_artifact(artifact, gp)

    def test_promoted_challenger_is_versioned_by_evaluation_gp(self):
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            champion = model_dir / "dnf_logistic_base.joblib"
            challenger = model_dir / "dnf_logistic_pending.joblib"
            champion.write_bytes(b"champion")
            challenger.write_bytes(b"challenger")
            decision = DNFPromotionDecision(True, -0.01, -0.02, "promoted")

            with patch.object(train_dnf_optimized, "MODEL_DIR", model_dir):
                promoted = train_dnf_optimized.update_champion_after_duel(
                    champion, challenger, decision, year=2026, race_number=2
                )

            self.assertEqual(promoted.name, "dnf_logistic_2026_2.joblib")
            self.assertEqual(promoted.read_bytes(), b"challenger")


if __name__ == "__main__":
    unittest.main()
