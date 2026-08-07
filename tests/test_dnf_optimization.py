import unittest

import numpy as np
import pandas as pd

from src.dnf_model import DNF_TARGET
from src.dnf_optimization import (
    PostProcessor,
    fit_optimized_artifact_for_history,
    prequential_predictions,
)


class DNFOptimizationTests(unittest.TestCase):
    def test_postprocessor_applies_configured_cap(self):
        postprocessor = PostProcessor(method="none", cap=0.4)
        np.testing.assert_allclose(
            postprocessor.transform([0.1, 0.5]),
            np.array([0.1, 0.4]),
        )

    def test_team_beta_candidate_is_refit_on_supplied_history(self):
        history = pd.DataFrame(
            {
                "race_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"]
                ),
                "team_id": ["a", "a", "b", "b"],
                DNF_TARGET: [1, 0, 0, 0],
            }
        )
        template = {
            "strategy": "team_beta_tuned",
            "model_parameters": {
                "prior_strength": 5.0,
                "lookback_races": None,
                "half_life_races": None,
            },
            "calibration_method": "none",
            "calibrator": None,
            "max_dnf_probability": 0.3,
            "features": [],
        }
        runtime_strategy, artifact = fit_optimized_artifact_for_history(template, history)

        self.assertEqual(runtime_strategy, "team_beta")
        self.assertEqual(artifact["team_beta_state"]["training_rows"], len(history))
        self.assertEqual(
            pd.Timestamp(artifact["cutoff_date"]),
            history["race_date"].max(),
        )

    def test_prequential_global_rate_does_not_use_future_races(self):
        data = pd.DataFrame(
            {
                "race_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"]
                ),
                DNF_TARGET: [0, 1, 1, 1],
            }
        )
        predictions = prequential_predictions(
            data,
            family="global_rate",
            parameters={},
            validation_dates=pd.Index([pd.Timestamp("2024-01-15")]),
            min_training_races=2,
        )
        self.assertEqual(predictions["raw_probability"].iloc[0], 0.5)


if __name__ == "__main__":
    unittest.main()
