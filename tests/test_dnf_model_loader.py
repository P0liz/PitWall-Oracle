import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from src.dnf_model_loader import resolve_dnf_model_path, select_dnf_artifact_path


class UnavailableClient:
    def get_experiment_by_name(self, name):
        raise RuntimeError("MLflow unavailable")


class DNFModelLoaderTests(unittest.TestCase):
    def test_historical_lookup_uses_only_promotions_from_previous_gps(self):
        artifacts = [
            "models/dnf_logistic_base.joblib",
            "models/dnf_logistic_2026_2.joblib",
            "models/dnf_logistic_2026_4.joblib",
        ]

        expected = {
            1: "models/dnf_logistic_base.joblib",
            2: "models/dnf_logistic_base.joblib",
            3: "models/dnf_logistic_2026_2.joblib",
            4: "models/dnf_logistic_2026_2.joblib",
            5: "models/dnf_logistic_2026_4.joblib",
        }

        for race_number, artifact_path in expected.items():
            with self.subTest(race_number=race_number):
                self.assertEqual(select_dnf_artifact_path(artifacts, 2026, race_number), artifact_path)

    def test_local_fallback_selects_the_latest_causally_eligible_promotion(self):
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            for name in ("dnf_logistic_base.joblib", "dnf_logistic_2026_2.joblib", "dnf_logistic_2026_4.joblib"):
                (model_dir / name).write_bytes(name.encode())

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                selected = resolve_dnf_model_path(
                    year=2026, race_number=4, client=UnavailableClient(), local_model_dir=model_dir
                )

        self.assertEqual(selected.name, "dnf_logistic_2026_2.joblib")


if __name__ == "__main__":
    unittest.main()
