import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from monte_carlo_simulator import compute_dnf_probabilities, run_monte_carlo


class MonteCarloTests(unittest.TestCase):
    def test_same_seed_produces_common_random_numbers(self):
        scores = np.array([1.0, 0.0, -1.0])
        probabilities = np.array([0.1, 0.2, 0.3])

        first = run_monte_carlo(scores, probabilities, 0.5, n_simulations=100, seed=7)
        second = run_monte_carlo(scores, probabilities, 0.5, n_simulations=100, seed=7)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_zero_probability_disables_dnf(self):
        _, dnf_draws = run_monte_carlo(np.array([1.0, 0.0]), np.zeros(2), 0.2, n_simulations=100, seed=7)
        self.assertFalse(dnf_draws.any())

    def test_none_returns_zeroes_without_accessing_history(self):
        race = pd.DataFrame({"race_date": [pd.Timestamp("2026-03-01")]})
        probabilities = compute_dnf_probabilities(race, year=2026, race_number=1, strategy="none", history_df=object())

        np.testing.assert_array_equal(probabilities, np.zeros(1, dtype=np.float64))

    def test_logistic_missing_artifact_raises_file_not_found(self):
        race = pd.DataFrame({"race_date": [pd.Timestamp("2026-03-01")]})
        missing_path = Path("tests") / "missing_dnf_logistic_2026_2.joblib"

        with patch("monte_carlo_simulator.resolve_dnf_model_path", return_value=missing_path):
            with self.assertRaisesRegex(FileNotFoundError, r"missing_dnf_logistic_2026_2\.joblib.*2026.*2"):
                compute_dnf_probabilities(race, year=2026, race_number=2, strategy="logistic")

    def test_logistic_rejects_nat_race_date(self):
        race = pd.DataFrame({"race_date": [pd.NaT]})
        artifact = {"model_type": "logistic", "cutoff_date": "2025-12-07T00:00:00Z"}

        with (
            patch("monte_carlo_simulator.resolve_dnf_model_path", return_value=Path(__file__)),
            patch("monte_carlo_simulator.load_dnf_artifact", return_value=artifact),
        ):
            with self.assertRaisesRegex(ValueError, "race_date"):
                compute_dnf_probabilities(race, year=2026, race_number=1, strategy="logistic")

    def test_logistic_rejects_equal_or_later_cutoff_with_mixed_timezones(self):
        race = pd.DataFrame({"race_date": [pd.Timestamp("2026-03-08")]})

        for cutoff in ("2026-03-08T00:00:00Z", "2026-03-09T01:00:00+01:00"):
            artifact = {"model_type": "logistic", "cutoff_date": cutoff}
            with self.subTest(cutoff=cutoff):
                with (
                    patch("monte_carlo_simulator.resolve_dnf_model_path", return_value=Path(__file__)),
                    patch("monte_carlo_simulator.load_dnf_artifact", return_value=artifact),
                ):
                    with self.assertRaisesRegex(ValueError, "Leakage temporale"):
                        compute_dnf_probabilities(race, year=2026, race_number=1, strategy="logistic")

    def test_obsolete_strategy_raises_value_error(self):
        race = pd.DataFrame({"race_date": [pd.Timestamp("2026-03-01")]})

        with self.assertRaises(ValueError):
            compute_dnf_probabilities(race, year=2026, race_number=1, strategy="team_beta")


if __name__ == "__main__":
    unittest.main()
