import math
import unittest

from src.dnf_metrics import DNFProbabilityMetrics, decide_dnf_promotion, score_dnf_probabilities


class DNFMetricsTests(unittest.TestCase):
    def test_probability_metrics_use_driver_level_brier_and_log_loss(self):
        metrics = score_dnf_probabilities([0, 1], [0.2, 0.7])

        self.assertAlmostEqual(metrics.brier_score, 0.065)
        self.assertAlmostEqual(metrics.log_loss, -(math.log(0.8) + math.log(0.7)) / 2.0)

    def test_promotion_requires_strict_brier_improvement_without_log_loss_regression(self):
        champion = DNFProbabilityMetrics(brier_score=0.10, log_loss=0.40)
        cases = (
            (DNFProbabilityMetrics(0.09, 0.39), True),
            (DNFProbabilityMetrics(0.10, 0.39), False),
            (DNFProbabilityMetrics(0.09, 0.41), False),
        )

        for challenger, expected in cases:
            with self.subTest(challenger=challenger):
                decision = decide_dnf_promotion(champion, challenger)
                self.assertEqual(decision.promote, expected)
                self.assertAlmostEqual(decision.delta_brier, challenger.brier_score - champion.brier_score)
                self.assertAlmostEqual(decision.delta_log_loss, challenger.log_loss - champion.log_loss)


if __name__ == "__main__":
    unittest.main()
