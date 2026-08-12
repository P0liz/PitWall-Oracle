import unittest

import numpy as np
import pandas as pd

from src.ranker_model import Training, select_model_feature_frame
from src.ranking_metrics import (
    decide_promotion,
    evaluate_grouped_rankings,
    mean_absolute_position_error,
    pairwise_accuracy,
    race_ranking_metrics,
    top_k_overlap,
)


class RankingMetricTests(unittest.TestCase):
    def test_pairwise_accuracy_covers_the_full_grid(self):
        truth = np.array([4.0, 3.0, 2.0, 1.0])

        self.assertEqual(pairwise_accuracy(truth, truth), 1.0)
        self.assertEqual(pairwise_accuracy(truth, -truth), 0.0)

    def test_teammate_accuracy_is_not_diluted_by_cross_team_pairs(self):
        truth = np.array([4.0, 3.0, 2.0, 1.0])
        scores = np.array([3.0, 4.0, 2.0, 1.0])
        teams = np.array(["team_a", "team_a", "team_b", "team_b"])

        self.assertAlmostEqual(pairwise_accuracy(truth, scores), 5 / 6)
        self.assertAlmostEqual(pairwise_accuracy(truth, scores, groups=teams), 0.5)

    def test_position_mae_measures_the_size_of_rank_errors(self):
        truth = np.array([4.0, 3.0, 2.0, 1.0])
        scores = np.array([3.0, 4.0, 2.0, 1.0])

        self.assertEqual(mean_absolute_position_error(truth, scores), 0.5)

    def test_scorecard_keeps_top_k_as_diagnostics(self):
        truth = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        scores = np.array([5.0, 4.0, 2.0, 3.0, 1.0])
        teams = np.array(["a", "a", "b", "b", "c"])

        metrics = race_ranking_metrics(truth, scores, teams)

        self.assertAlmostEqual(metrics["pairwise_accuracy"], 0.9)
        self.assertEqual(metrics["top_3_overlap"], top_k_overlap(truth, scores, 3))
        self.assertIn("ndcg_full", metrics)

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
        scores = np.array([4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0])

        metrics = evaluate_grouped_rankings(frame, scores)

        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics.loc[0, "pairwise_accuracy"], 1.0)
        self.assertEqual(metrics.loc[1, "pairwise_accuracy"], 0.0)


class PromotionTests(unittest.TestCase):
    def test_candidate_can_be_promoted_after_one_paired_race(self):
        duels = pd.DataFrame(
            {
                "champion_pairwise_accuracy": [0.7],
                "challenger_pairwise_accuracy": [0.8],
                "champion_teammate_pairwise_accuracy": [0.6],
                "challenger_teammate_pairwise_accuracy": [0.7],
                "champion_position_mae": [2.0],
                "challenger_position_mae": [1.8],
            }
        )

        decision = decide_promotion(duels)

        self.assertTrue(decision.promote)

    def test_candidate_must_not_regress_any_primary_ranker_metric(self):
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
        teammate_regression = improving.copy()
        teammate_regression["challenger_teammate_pairwise_accuracy"] = [0.5]

        self.assertTrue(decide_promotion(improving).promote)
        self.assertFalse(decide_promotion(teammate_regression).promote)


class ProductionFeatureTests(unittest.TestCase):
    def test_inference_uses_feature_order_stored_by_model(self):
        class Booster:
            feature_names = ["feature_b", "feature_a"]

        class Model:
            @staticmethod
            def get_booster():
                return Booster()

        frame = pd.DataFrame({"feature_a": [1.0], "ignored": [9.0], "feature_b": [2.0]})

        selected = select_model_feature_frame(Model(), frame)

        self.assertEqual(selected.columns.tolist(), ["feature_b", "feature_a"])

    def test_group_weights_have_one_value_per_qid(self):
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

        weights = trainer.group_weights(frame, pd.Timestamp("2026-03-08"))

        np.testing.assert_array_equal(weights, np.array([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
