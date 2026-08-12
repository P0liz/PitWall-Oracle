import sys
import unittest
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.modules.setdefault("ranker_model", import_module("src.ranker_model"))
sys.modules.setdefault("ranker_optimization", import_module("src.ranker_optimization"))

from monte_carlo_simulator import fetch_relative_sigma
from train_ranker_optimized import compute_sigma_calibration_statistics, zscore_within_race


class RankerSigmaCalibrationTests(unittest.TestCase):
    def test_zscore_within_race_standardizes_values(self):
        standardized = zscore_within_race(np.array([1.0, 2.0, 3.0]))

        np.testing.assert_allclose(standardized, np.array([-1.22474487, 0.0, 1.22474487]))

    def test_zscore_within_race_returns_zeroes_for_constant_values(self):
        standardized = zscore_within_race(np.array([4.0, 4.0, 4.0]))

        np.testing.assert_array_equal(standardized, np.zeros(3))

    def test_sigma_statistics_accumulate_residuals_across_races(self):
        first_race = np.array([-1.0, 0.0, 1.0])
        second_race = np.array([-2.0, 0.0, 2.0])

        statistics = compute_sigma_calibration_statistics([first_race, second_race], [0.5, 1.5])

        self.assertAlmostEqual(statistics["race_sigma_relative"], np.std(second_race, ddof=0))
        self.assertAlmostEqual(
            statistics["cumulative_sigma_relative"], np.std(np.array([-1.0, 0.0, 1.0, -2.0, 0.0, 2.0]), ddof=0)
        )
        self.assertEqual(statistics["race_score_std"], 1.5)
        self.assertEqual(statistics["mean_score_std"], 1.0)
        self.assertEqual(statistics["median_score_std"], 1.0)

    def test_monte_carlo_fetches_previous_race_cumulative_sigma(self):
        metric_history = [
            SimpleNamespace(step=1, value=0.6, timestamp=10),
            SimpleNamespace(step=2, value=0.8, timestamp=20),
        ]
        client = SimpleNamespace(
            get_experiment_by_name=lambda _: SimpleNamespace(experiment_id="1"),
            get_metric_history=lambda _run_id, key: metric_history if key == "cumulative_sigma_relativo" else [],
        )
        run = SimpleNamespace(info=SimpleNamespace(run_id="run-2026"))

        with (
            patch("monte_carlo_simulator.mlflow.set_tracking_uri"),
            patch("monte_carlo_simulator.mlflow.tracking.MlflowClient", return_value=client),
            patch("monte_carlo_simulator._runs_for_year", return_value=[run]),
        ):
            sigma = fetch_relative_sigma(year=2026, race_number=3)

        self.assertEqual(sigma, 0.8)


if __name__ == "__main__":
    unittest.main()
