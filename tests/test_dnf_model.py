import unittest

import numpy as np
import pandas as pd

from src.dnf_model import (
    BENCHMARK_DNF_FEATURES,
    DNF_TARGET,
    MAX_DNF_PROB,
    compute_strategy_probabilities,
    ensure_dnf_target,
    fit_team_beta_state,
    predict_team_beta,
    train_dnf_gradient_boosting,
    train_dnf_logistic,
)


class DNFModelTests(unittest.TestCase):
    def test_legacy_target_is_migrated(self):
        legacy = pd.DataFrame({"technical_dnf_target": [0, 1]})
        migrated = ensure_dnf_target(legacy)
        self.assertEqual(migrated[DNF_TARGET].tolist(), [0, 1])

    def test_team_beta_shrinks_and_handles_unseen_team(self):
        history = pd.DataFrame(
            {
                "team_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
                DNF_TARGET: [1, 1, 0, 0, 0, 0, 0, 0],
            }
        )
        state = fit_team_beta_state(history, prior_strength=4)
        predictions = predict_team_beta(state, pd.DataFrame({"team_id": ["a", "b", "new"]}))

        self.assertGreater(predictions[0], predictions[2])
        self.assertLess(predictions[1], predictions[2])
        self.assertAlmostEqual(predictions[2], 0.25)
        self.assertTrue(np.all(predictions <= MAX_DNF_PROB))

    def test_none_strategy_is_deterministic_zero(self):
        prediction = pd.DataFrame({"driver_id": ["a", "b"]})
        probabilities = compute_strategy_probabilities("none", prediction)
        np.testing.assert_array_equal(probabilities, np.zeros(2))

    def test_legacy_logistic_feature_accepts_canonical_team_rate(self):
        class FixedModel:
            def predict_proba(self, frame):
                self.columns = frame.columns.tolist()
                return np.array([[0.8, 0.2]])

        model = FixedModel()
        artifact = {
            "model": model,
            "features": ["rolling_tech_dnf_rate", "car_age_proxy"],
        }
        probabilities = compute_strategy_probabilities(
            "logistic",
            pd.DataFrame({"team_dnf_rate": [0.1], "car_age_proxy": [200.0]}),
            artifact=artifact,
        )
        self.assertEqual(model.columns, ["rolling_tech_dnf_rate", "car_age_proxy"])
        self.assertAlmostEqual(probabilities[0], 0.2)

    def test_training_reports_independent_evaluation_window(self):
        rows = []
        for race in range(14):
            for driver in range(4):
                rows.append(
                    {
                        "race_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * race),
                        "team_id": f"team_{driver // 2}",
                        "rolling_tech_dnf_rate": 0.05 + 0.04 * (driver // 2),
                        "car_age_proxy": float(race * 50 + driver),
                        DNF_TARGET: int((race + driver) % 7 == 0),
                    }
                )
        result = train_dnf_logistic(
            pd.DataFrame(rows),
            c_values=(0.1,),
            class_weights=(None,),
            min_training_races=4,
            evaluation_races=3,
        )

        self.assertEqual(result.evaluation_races, 3)
        self.assertIn(result.baseline_strategy, {"global_rate", "team_beta", "heuristic"})
        self.assertEqual(result.team_beta_state["training_rows"], len(rows))

    def test_gradient_boosting_uses_benchmark_features_and_returns_probabilities(self):
        rows = []
        for race in range(14):
            for driver in range(6):
                rows.append(
                    {
                        "race_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * race),
                        "team_id": f"team_{driver // 2}",
                        "rolling_tech_dnf_rate": 0.05 + 0.02 * (driver // 2),
                        "grid_position": driver + 1,
                        "team_dnf_rate": 0.05 + 0.02 * (driver // 2),
                        "driver_dnf_rate": 0.03 + 0.01 * driver,
                        "car_age_proxy": float(race * 50 + driver),
                        "wet_affinity": np.nan if race < 2 else 0.1 * driver,
                        DNF_TARGET: int((race + driver) % 8 == 0),
                    }
                )

        result = train_dnf_gradient_boosting(
            pd.DataFrame(rows),
            min_training_races=4,
            evaluation_races=3,
        )
        self.assertEqual(result.model_type, "gradient_boosting")
        self.assertEqual(result.features, BENCHMARK_DNF_FEATURES)
        probabilities = result.model.predict_proba(pd.DataFrame(rows)[BENCHMARK_DNF_FEATURES])[:, 1]
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())


if __name__ == "__main__":
    unittest.main()
