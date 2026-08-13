import unittest

import numpy as np
import pandas as pd

from src.ranker.ranker_model import Training, select_model_feature_frame
from src.ranker.ranking_metrics import (
    decide_promotion,
    evaluate_grouped_rankings,
    pairwise_accuracy,
    race_ranking_metrics,
)


class RankingPipelineTests(unittest.TestCase):
    def test_pairwise_metrics_cover_full_grid_and_teammates_separately(self):
        truth = np.array([4.0, 3.0, 2.0, 1.0])
        scores = np.array([3.0, 4.0, 2.0, 1.0])
        teams = np.array(["team_a", "team_a", "team_b", "team_b"])

        self.assertAlmostEqual(pairwise_accuracy(truth, scores), 5 / 6)
        self.assertAlmostEqual(pairwise_accuracy(truth, scores, groups=teams), 0.5)

    def test_grouped_evaluation_never_flattens_races(self):
        frame = pd.DataFrame(
            {
                "race_date": pd.to_datetime(["2026-01-01"] * 4 + ["2026-01-15"] * 4),
                "race_number": [1] * 4 + [2] * 4,
                "year": [2026] * 8,
                "session_type": ["race"] * 8,
                "raw_team_id": ["a", "a", "b", "b"] * 2,
                "target": [4.0, 3.0, 2.0, 1.0] * 2,
            }
        )
        metrics = evaluate_grouped_rankings(frame, np.array([4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0]))

        self.assertEqual(metrics["pairwise_accuracy"].tolist(), [1.0, 0.0])

    def test_scorecard_keeps_required_diagnostics(self):
        metrics = race_ranking_metrics(
            np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
            np.array([5.0, 4.0, 2.0, 3.0, 1.0]),
            np.array(["a", "a", "b", "b", "c"]),
        )

        self.assertAlmostEqual(metrics["pairwise_accuracy"], 0.9)
        self.assertIn("ndcg_full", metrics)
        self.assertIn("top_3_overlap", metrics)

    def test_promotion_rejects_regression_in_any_primary_metric(self):
        improving = pd.DataFrame(
            {
                "champion_pairwise_accuracy": [0.7],
                "challenger_pairwise_accuracy": [0.8],
                "champion_teammate_pairwise_accuracy": [0.6],
                "challenger_teammate_pairwise_accuracy": [0.7],
                "champion_position_mae": [2.0],
                "challenger_position_mae": [1.9],
            }
        )
        regressing = improving.copy()
        regressing["challenger_teammate_pairwise_accuracy"] = [0.5]

        self.assertTrue(decide_promotion(improving).promote)
        self.assertFalse(decide_promotion(regressing).promote)

    def test_inference_uses_feature_order_stored_by_model(self):
        class Booster:
            feature_names = ["feature_b", "feature_a"]

        class Model:
            @staticmethod
            def get_booster():
                return Booster()

        selected = select_model_feature_frame(
            Model(), pd.DataFrame({"feature_a": [1.0], "ignored": [9.0], "feature_b": [2.0]})
        )

        self.assertEqual(selected.columns.tolist(), ["feature_b", "feature_a"])

    def test_group_weights_have_one_value_per_sorted_qid(self):
        trainer = Training.__new__(Training)
        trainer.decay_rate = 0.0
        trainer.target_year = 2026
        trainer.target_train_multiplier = 2.0
        frame = pd.DataFrame(
            {
                "qid": [1, 0, 1, 0],
                "race_date": pd.to_datetime(["2026-03-08", "2025-12-07", "2026-03-08", "2025-12-07"]),
                "year": [2026, 2025, 2026, 2025],
            }
        )

        np.testing.assert_array_equal(trainer.group_weights(frame, pd.Timestamp("2026-03-08")), np.array([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
