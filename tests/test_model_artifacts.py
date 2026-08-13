import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.monte_carlo_simulator import fetch_relative_sigma
from src.config import NEW_YEAR
from src.dnf.dnf_model_loader import resolve_dnf_model_path, select_dnf_artifact_path
from src.model_lifecycle import ModelLifecycleError, publish_staged_models, write_calibration
from src.ranker.ranker_model_loader import resolve_ranker_model_path, select_ranker_artifact_path
from src.train_ranker_optimized import compute_sigma_calibration_statistics


class ModelArtifactTests(unittest.TestCase):
    @staticmethod
    def populate_staging(staging: Path, year: int = 2026) -> None:
        staging.mkdir(parents=True, exist_ok=True)
        for name in (
            f"pitwall_oracle_{year}_base.json",
            f"pitwall_oracle_{year}_2.json",
            "pitwall_oracle_latest.json",
            f"dnf_logistic_{year}_base.joblib",
            f"dnf_logistic_{year}_3.joblib",
            "dnf_logistic_latest.joblib",
        ):
            (staging / name).write_bytes(f"new:{name}".encode())
        write_calibration(staging / "monte_carlo_calibration.json", year, {0: 0.7, 1: 0.6, 2: 0.8})

    def test_model_selectors_use_only_promotions_strictly_before_the_gp(self):
        cases = (
            (
                select_ranker_artifact_path,
                [
                    "models/pitwall_oracle_2026_base.json",
                    "models/pitwall_oracle_2026_2.json",
                    "models/pitwall_oracle_2026_4.json",
                ],
                "models/pitwall_oracle_2026_2.json",
            ),
            (
                select_dnf_artifact_path,
                [
                    "models/dnf_logistic_2026_base.joblib",
                    "models/dnf_logistic_2026_2.joblib",
                    "models/dnf_logistic_2026_4.joblib",
                ],
                "models/dnf_logistic_2026_2.joblib",
            ),
        )
        for selector, artifacts, expected in cases:
            with self.subTest(selector=selector.__name__):
                self.assertEqual(selector(artifacts, 2026, 4), expected)

    def test_legacy_bases_are_not_used_for_historical_seasons(self):
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            ranker = model_dir / "pitwall_oracle_base.json"
            dnf = model_dir / "dnf_logistic_base.joblib"
            ranker.write_bytes(b"ranker")
            dnf.write_bytes(b"dnf")

            self.assertEqual(resolve_ranker_model_path(NEW_YEAR, 1, local_model_dir=model_dir), ranker)
            self.assertEqual(resolve_dnf_model_path(NEW_YEAR, 1, local_model_dir=model_dir), dnf)
            with self.assertRaises(FileNotFoundError):
                resolve_ranker_model_path(NEW_YEAR - 1, 1, local_model_dir=model_dir)
            with self.assertRaises(FileNotFoundError):
                resolve_dnf_model_path(NEW_YEAR - 1, 1, local_model_dir=model_dir)

    def test_invalid_staging_leaves_live_models_unchanged(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            live = root / "models"
            staging = root / "staging"
            live.mkdir()
            (live / "pitwall_oracle_2026_2.json").write_bytes(b"old")
            self.populate_staging(staging)
            (staging / "dnf_logistic_latest.joblib").unlink()
            before = {path.name: path.read_bytes() for path in live.iterdir()}

            with self.assertRaises(ModelLifecycleError):
                publish_staged_models(staging, live, 2026)

            self.assertEqual({path.name: path.read_bytes() for path in live.iterdir()}, before)

    def test_successful_publish_replaces_current_season_only(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            live = root / "models"
            staging = root / "staging"
            live.mkdir()
            (live / "pitwall_oracle_2025_20.json").write_bytes(b"previous")
            (live / "pitwall_oracle_2026_9.json").write_bytes(b"obsolete")
            (live / "dnf_logistic_2026_8.joblib").write_bytes(b"obsolete")
            (live / "pitwall_oracle_pending.json").write_bytes(b"pending")
            self.populate_staging(staging)

            publish_staged_models(staging, live, 2026)

            self.assertTrue((live / "pitwall_oracle_2025_20.json").exists())
            self.assertFalse((live / "pitwall_oracle_2026_9.json").exists())
            self.assertFalse((live / "dnf_logistic_2026_8.joblib").exists())
            self.assertFalse((live / "pitwall_oracle_pending.json").exists())
            self.assertTrue((live / "pitwall_oracle_2026_2.json").exists())

    def test_training_failure_cannot_publish_partial_output(self):
        from scripts.run_post_race_pipeline import run_training_pipeline

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            live = root / "models"
            staging = root / "staging"
            live.mkdir()
            current = live / "pitwall_oracle_2026_2.json"
            current.write_bytes(b"current")

            def failing_runner(command, **kwargs):
                if command[-1] == "train_ranker_optimized.py":
                    staging.mkdir(parents=True, exist_ok=True)
                    (staging / "pitwall_oracle_2026_base.json").write_bytes(b"candidate")
                    return subprocess.CompletedProcess(command, 0)
                raise subprocess.CalledProcessError(1, command)

            with self.assertRaises(subprocess.CalledProcessError):
                run_training_pipeline(staging, live, 2026, repository_root=Path.cwd(), command_runner=failing_runner)

            self.assertEqual(current.read_bytes(), b"current")

    def test_sigma_is_cumulative_and_resolved_from_previous_round(self):
        statistics = compute_sigma_calibration_statistics(
            [np.array([-1.0, 0.0, 1.0]), np.array([-2.0, 0.0, 2.0])], [0.5, 1.5]
        )
        self.assertAlmostEqual(
            statistics["cumulative_sigma_relative"], np.std(np.array([-1.0, 0.0, 1.0, -2.0, 0.0, 2.0]), ddof=0)
        )

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "monte_carlo_calibration.json"
            write_calibration(path, 2026, {0: 0.7, 1: 0.6, 2: 0.8})
            self.assertEqual(fetch_relative_sigma(2026, 3, calibration_path=path), 0.8)


if __name__ == "__main__":
    unittest.main()
