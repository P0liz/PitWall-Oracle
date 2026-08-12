import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    from evaluate_monte_carlo_results import evaluate_race, evaluate_season
except ImportError:
    evaluate_race = None
    evaluate_season = None


class MonteCarloResultsEvaluationTests(unittest.TestCase):
    def test_evaluate_race_aligns_drivers_and_scores_all_official_positions(self):
        if evaluate_race is None:
            self.fail("evaluate_monte_carlo_results.evaluate_race non esiste")

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

        self.assertEqual(
            metrics["ranking"].tolist(), ["base_position", "expected_position", "expected_position_if_finished"]
        )
        self.assertEqual(metrics["drivers"].tolist(), [4, 4, 4])
        self.assertEqual(metrics["pairwise_accuracy"].tolist(), [1.0, 2 / 3, 0.0])
        self.assertEqual(metrics["teammate_pairwise_accuracy"].tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(metrics["position_mae"].tolist(), [0.0, 1.0, 2.0])

    def test_evaluate_season_skips_failed_simulations_when_averaging(self):
        if evaluate_season is None:
            self.fail("evaluate_monte_carlo_results.evaluate_season non esiste")

        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            results_dir = directory / "results"
            silver_dir = directory / "silver"
            results_dir.mkdir()
            silver_dir.mkdir()
            pd.DataFrame(
                {
                    "driver_id": ["driver_1", "driver_2"],
                    "base_position": [1, 2],
                    "expected_position": [1, 2],
                    "expected_position_if_finished": [1, 2],
                }
            ).to_csv(results_dir / "summary_2026_1.csv", index=False)
            pd.DataFrame(
                {"driver_id": ["driver_1", "driver_2"], "team_id": ["team_a", "team_a"], "Position": [1.0, 2.0]}
            ).to_parquet(silver_dir / "2026_1_5_clean_results.parquet", index=False)
            attempted = []

            def failing_simulator(year, race_number):
                attempted.append((year, race_number))
                raise RuntimeError("modello assente")

            details, summary, failures = evaluate_season(
                2026, [1, 2], results_dir=results_dir, silver_dir=silver_dir, simulator=failing_simulator
            )

        self.assertEqual(attempted, [(2026, 2)])
        self.assertEqual(details["race_number"].unique().tolist(), [1])
        self.assertEqual(summary["races_evaluated"].tolist(), [1, 1, 1])
        self.assertEqual(summary["pairwise_accuracy"].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(failures, {2: "modello assente"})


if __name__ == "__main__":
    unittest.main()
