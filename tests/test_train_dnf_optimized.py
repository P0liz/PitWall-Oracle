import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

import train_dnf_optimized
from src.dnf_metrics import DNFPromotionDecision
from src.dnf_model import DNF_TARGET


class FixedProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, frame):
        probabilities = self.probabilities[: len(frame)]
        return np.column_stack((1.0 - probabilities, probabilities))


class TrainDNFOptimizedTests(unittest.TestCase):
    @staticmethod
    def artifact(probabilities):
        return {
            "model": FixedProbabilityModel(probabilities),
            "features": ["grid_position"],
            "target": DNF_TARGET,
            "cutoff_date": pd.Timestamp("2025-12-07"),
        }

    def test_weekend_selection_evaluates_only_gp_but_retains_sprint_for_training(self):
        sprint = pd.DataFrame(
            {
                "session_type": ["sprint"],
                "race_date": [pd.Timestamp("2026-03-07")],
                "grid_position": [1.0],
                DNF_TARGET: [1],
            }
        )
        gp = pd.DataFrame(
            {
                "session_type": ["race", "race"],
                "race_date": [pd.Timestamp("2026-03-08")] * 2,
                "grid_position": [1.0, 2.0],
                DNF_TARGET: [0, 1],
            }
        )
        history = pd.DataFrame(
            {
                "session_type": ["race"],
                "race_date": [pd.Timestamp("2025-12-07")],
                "grid_position": [1.0],
                DNF_TARGET: [0],
            }
        )

        evaluation = train_dnf_optimized.gp_evaluation_frame([sprint, gp])
        updated = train_dnf_optimized.append_weekend_history(history, [sprint, gp])

        self.assertEqual(evaluation["session_type"].tolist(), ["race", "race"])
        self.assertEqual(updated["session_type"].tolist(), ["race", "sprint", "race", "race"])

    def test_duel_is_champion_only_without_pending_and_decides_when_pending_exists(self):
        gp = pd.DataFrame(
            {"race_date": [pd.Timestamp("2026-03-08")] * 2, "grid_position": [1.0, 2.0], DNF_TARGET: [0, 1]}
        )
        champion = self.artifact([0.2, 0.6])
        challenger = self.artifact([0.1, 0.8])

        first_race = train_dnf_optimized.evaluate_dnf_duel(champion, None, gp)
        second_race = train_dnf_optimized.evaluate_dnf_duel(champion, challenger, gp)

        self.assertIsNone(first_race.challenger)
        self.assertIsNone(first_race.decision)
        self.assertIsNotNone(second_race.challenger)
        self.assertTrue(second_race.decision.promote)

    def test_duel_rejects_artifact_not_strictly_before_gp(self):
        gp = pd.DataFrame({"race_date": [pd.Timestamp("2026-03-08")], "grid_position": [1.0], DNF_TARGET: [0]})
        artifact = self.artifact([0.2])
        artifact["cutoff_date"] = pd.Timestamp("2026-03-08")

        with self.assertRaisesRegex(ValueError, "Leakage"):
            train_dnf_optimized.evaluate_dnf_artifact(artifact, gp)

    def test_rejected_challenger_keeps_current_champion_without_saving(self):
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            champion = model_dir / "dnf_logistic_base.joblib"
            challenger = model_dir / "dnf_logistic_pending.joblib"
            champion.write_bytes(b"champion")
            challenger.write_bytes(b"challenger")
            decision = DNFPromotionDecision(False, 0.01, 0.02, "rejected")

            with patch.object(train_dnf_optimized, "MODEL_DIR", model_dir):
                actual = train_dnf_optimized.update_champion_after_duel(
                    champion, challenger, decision, year=2026, race_number=2
                )

            self.assertEqual(actual, champion)
            self.assertFalse((model_dir / "dnf_logistic_2026_2.joblib").exists())

    def test_promoted_challenger_uses_the_gp_where_it_was_evaluated(self):
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            champion = model_dir / "dnf_logistic_base.joblib"
            challenger = model_dir / "dnf_logistic_pending.joblib"
            champion.write_bytes(b"champion")
            challenger.write_bytes(b"challenger trained through GP1")
            decision = DNFPromotionDecision(True, -0.01, -0.02, "promoted")

            with patch.object(train_dnf_optimized, "MODEL_DIR", model_dir):
                actual = train_dnf_optimized.update_champion_after_duel(
                    champion, challenger, decision, year=2026, race_number=2
                )

            self.assertEqual(actual, model_dir / "dnf_logistic_2026_2.joblib")
            self.assertEqual(actual.read_bytes(), b"challenger trained through GP1")

    def test_prepare_base_history_excludes_cached_current_year_before_append(self):
        cached_history = pd.DataFrame(
            {"race_date": [pd.Timestamp("2025-12-07"), pd.Timestamp("2026-03-08", tz="UTC")], DNF_TARGET: [0, 1]}
        )

        base_history = train_dnf_optimized.prepare_base_history(
            cached_history, year=2026, first_race_date=pd.Timestamp("2026-03-08")
        )
        updated_history = pd.concat([base_history, cached_history.iloc[[1]]], ignore_index=True)

        self.assertEqual(base_history[DNF_TARGET].tolist(), [0])
        self.assertEqual(pd.to_datetime(updated_history["race_date"], utc=True).dt.year.tolist(), [2025, 2026])

    def test_prepare_base_history_rejects_invalid_or_nonpreceding_cutoff(self):
        cases = (
            (pd.DataFrame({"race_date": [pd.NaT], DNF_TARGET: [0]}), pd.Timestamp("2026-03-08")),
            (
                pd.DataFrame({"race_date": [pd.Timestamp("2025-12-07")], DNF_TARGET: [0]}),
                pd.Timestamp("2025-12-07", tz="UTC"),
            ),
        )

        for history, first_race_date in cases:
            with self.subTest(first_race_date=first_race_date):
                with self.assertRaisesRegex(ValueError, "cutoff"):
                    train_dnf_optimized.prepare_base_history(history, 2026, first_race_date)

    def test_preseason_training_cannot_create_a_numbered_champion(self):
        with self.assertRaisesRegex(ValueError, "base"):
            train_dnf_optimized.train_and_log_dnf_model(pd.DataFrame(), year=2026, model_race=2, n_trials=1)


if __name__ == "__main__":
    unittest.main()
