import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluate_monte_carlo_results import evaluate_race
from src.monte_carlo_simulator import (
    build_head_to_head_matrix,
    compute_dnf_probabilities,
    run_monte_carlo,
    summarize_results,
)


class SimulationTests(unittest.TestCase):
    def test_sprint_points_probability_uses_top_eight(self):
        positions = np.arange(1, 11, dtype=int).reshape(1, 10)
        dnf_draws = np.zeros_like(positions, dtype=bool)
        driver_ids = np.array([f"driver_{position}" for position in range(1, 11)])

        sprint = summarize_results(driver_ids, positions, dnf_draws, np.arange(10), points_cutoff=8)
        race = summarize_results(driver_ids, positions, dnf_draws, np.arange(10), points_cutoff=10)

        self.assertEqual(sprint.set_index("driver_id").loc["driver_9", "points_probability"], 0.0)
        self.assertEqual(race.set_index("driver_id").loc["driver_9", "points_probability"], 1.0)

    def test_seed_is_reproducible_and_zero_probability_disables_dnf(self):
        scores = np.array([1.0, 0.0, -1.0])
        probabilities = np.zeros(3)

        first = run_monte_carlo(scores, probabilities, 0.5, n_simulations=100, seed=7)
        second = run_monte_carlo(scores, probabilities, 0.5, n_simulations=100, seed=7)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertFalse(first[1].any())

    def test_logistic_dnf_artifact_must_strictly_precede_race(self):
        race = pd.DataFrame({"race_date": [pd.Timestamp("2026-03-08")]})
        for cutoff in ("2026-03-08T00:00:00Z", "2026-03-09T01:00:00+01:00"):
            with self.subTest(cutoff=cutoff):
                with (
                    patch("src.monte_carlo_simulator.resolve_dnf_model_path", return_value=Path(__file__)),
                    patch(
                        "src.monte_carlo_simulator.load_dnf_artifact",
                        return_value={"model_type": "logistic", "cutoff_date": cutoff},
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, "Leakage temporale"):
                        compute_dnf_probabilities(race, 2026, 1, strategy="logistic")

    def test_head_to_head_matrix_is_complete_and_complementary(self):
        driver_ids = np.array(["a", "b", "c"])
        positions = np.array([[1, 2, 3], [2, 1, 3], [1, 3, 2], [3, 2, 1]])

        matrix = build_head_to_head_matrix(driver_ids, positions)

        self.assertEqual(set(matrix), set(driver_ids))
        self.assertEqual(set(matrix["a"]), {"b", "c"})
        for first in driver_ids:
            for second in driver_ids:
                if first != second:
                    self.assertAlmostEqual(matrix[first][second] + matrix[second][first], 1.0)

    def test_result_evaluation_aligns_drivers_and_scores_official_positions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            summary_path = directory / "summary.csv"
            silver_path = directory / "silver.parquet"
            pd.DataFrame(
                {
                    "driver_id ": ["driver_3 ", "driver_1 ", "driver_4 ", "driver_2 "],
                    " base_position": [3, 1, 4, 2],
                    " expected_position": [4, 2, 3, 1],
                    " expected_position_if_finished": [2, 4, 1, 3],
                }
            ).to_csv(summary_path, index=False)
            pd.DataFrame(
                {
                    "driver_id": ["driver_1", "driver_2", "driver_3", "driver_4"],
                    "team_id": ["team_a", "team_b", "team_a", "team_b"],
                    "Position": [1.0, 2.0, 3.0, 4.0],
                    "Status": ["Finished", "Finished", "Lapped", "Retired"],
                }
            ).to_parquet(silver_path, index=False)

            metrics = evaluate_race(summary_path, silver_path)

        self.assertEqual(metrics["drivers"].tolist(), [4, 4, 4])
        self.assertEqual(metrics["pairwise_accuracy"].tolist(), [1.0, 2 / 3, 0.0])
        self.assertEqual(metrics["position_mae"].tolist(), [0.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
