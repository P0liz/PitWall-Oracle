import unittest

import pandas as pd

from src.data.feature_engineer import FeatureEngineering
from src.utils import is_race_dnf


class RaceDNFClassificationTests(unittest.TestCase):
    def test_classified_finishes_are_not_dnf(self):
        self.assertFalse(is_race_dnf("Finished"))
        self.assertFalse(is_race_dnf("Lapped"))

    def test_post_race_exclusions_are_not_dnf(self):
        self.assertFalse(is_race_dnf("Disqualified"))
        self.assertFalse(is_race_dnf("Excluded"))

    def test_all_cause_retirements_are_dnf(self):
        dnf_statuses = ("Retired", "Accident", "Collision damage", "Did not start", "Brakes", "Undertray")
        for status in dnf_statuses:
            with self.subTest(status=status):
                self.assertTrue(is_race_dnf(status))

    def test_missing_status_is_not_treated_as_signal(self):
        self.assertFalse(is_race_dnf(None))
        self.assertFalse(is_race_dnf(float("nan")))
        self.assertFalse(is_race_dnf(""))

    def test_rolling_rate_uses_the_canonical_definition(self):
        history = pd.DataFrame(
            {
                "year": [2025] * 4,
                "team_id": ["team"] * 4,
                "status_raw": ["Finished", "Retired", "Disqualified", "Brakes"],
            }
        )
        expected = pd.Series([False, True, False, True]).ewm(span=30).mean().iloc[-1]

        actual = FeatureEngineering(silver=None).compute_team_dnf_rate(history, 2025, "team")

        self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
