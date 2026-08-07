import unittest

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
        _, dnf_draws = run_monte_carlo(
            np.array([1.0, 0.0]),
            np.zeros(2),
            0.2,
            n_simulations=100,
            seed=7,
        )
        self.assertFalse(dnf_draws.any())

    def test_team_beta_uses_only_history_before_race(self):
        race = pd.DataFrame(
            {
                "race_date": [pd.Timestamp("2026-03-01")],
                "team_id": ["team"],
                "rolling_tech_dnf_rate": [0.1],
                "car_age_proxy": [100.0],
            }
        )
        history = pd.DataFrame(
            {
                "race_date": [
                    pd.Timestamp("2026-02-01"),
                    pd.Timestamp("2026-02-08"),
                    pd.Timestamp("2026-04-01"),
                ],
                "team_id": ["team", "team", "team"],
                "status_raw": ["Finished", "Finished", "Retired"],
            }
        )

        probability = compute_dnf_probabilities(
            race,
            year=2026,
            race_number=1,
            strategy="team_beta",
            history_df=history,
        )[0]

        self.assertLess(probability, 0.12)


if __name__ == "__main__":
    unittest.main()
