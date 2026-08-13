import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd

from src.data.data_loader import DataLoader
from src.dnf.dnf_features import DNF_CANDIDATE_FEATURES


class DataPipelineTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def make_loader(data_dir: Path) -> DataLoader:
        loader = DataLoader.__new__(DataLoader)
        loader.data_dir = data_dir
        loader.train_df = None
        loader.test_df = None
        loader.dnf_df = None
        loader.log = Mock()
        return loader

    @staticmethod
    def valid_dnf_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "race_date": pd.to_datetime(["2025-01-01"]),
                "dnf_target": [False],
                **{feature: [0.0] for feature in DNF_CANDIDATE_FEATURES},
            }
        )

    async def test_stale_static_cache_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            for name in ("train_df.parquet", "test_df.parquet", "dnf_df.parquet"):
                (data_dir / name).write_bytes(b"cached")
            loader = self.make_loader(data_dir)
            cached = {
                "train_df.parquet": pd.DataFrame({"race_date": pd.to_datetime(["2025-01-01"])}),
                "test_df.parquet": pd.DataFrame({"race_date": pd.to_datetime(["2025-02-01"])}),
                "dnf_df.parquet": pd.DataFrame({"race_date": pd.to_datetime(["2025-01-01"]), "dnf_target": [False]}),
            }
            expected_train = pd.DataFrame({"race_date": pd.to_datetime(["2025-03-01"]), "target": [1.0]})
            expected_test = pd.DataFrame({"race_date": pd.to_datetime(["2025-04-01"]), "target": [2.0]})

            async def rebuild(force):
                loader.dnf_df = self.valid_dnf_frame()
                return expected_train, expected_test

            loader.build_static_data = AsyncMock(side_effect=rebuild)
            with patch("src.data.data_loader.pd.read_parquet", side_effect=lambda path: cached[Path(path).name].copy()):
                train, test = await loader.load_data(is_dynamic=False)

            pd.testing.assert_frame_equal(train, expected_train)
            pd.testing.assert_frame_equal(test, expected_test)
            loader.build_static_data.assert_awaited_once()

    async def test_dynamic_build_rejects_rows_after_cutoff(self):
        loader = self.make_loader(Path("unused"))
        loader.gold = Mock()
        history = pd.DataFrame({"race_date": pd.to_datetime(["2025-01-01"]), "target": [1.0]})
        loader.gold.build_features.return_value = [
            pd.DataFrame({"race_date": pd.to_datetime(["2026-03-09"]), "target": [2.0]})
        ]

        def assemble(races, race_is_test, historical):
            loader.train_df = pd.concat([historical, *races], ignore_index=True)
            loader.test_df = pd.DataFrame()

        loader._apply_encoding = assemble
        schedule = pd.DataFrame({"Session5DateUtc": pd.to_datetime(["2026-03-08"], utc=True)})
        with patch("src.data.data_loader.fastf1.get_event_schedule", return_value=schedule):
            with self.assertRaises(ValueError):
                await loader.build_dynamic_data(history, pd.Timestamp("2026-03-08", tz="UTC"))

    def test_target_encoding_uses_only_strictly_previous_history(self):
        loader = self.make_loader(Path("unused"))
        loader.history_df = pd.DataFrame(
            {
                "race_date": pd.to_datetime(["2025-01-01", "2025-01-08", "2025-01-15", "2025-01-22"]),
                "driver_id": ["driver_a", "driver_b", "driver_a", "driver_b"],
                "is_podium": [1, 0, 0, 1],
            }
        )

        encoding = loader.compute_target_encoding_map("driver_id", pd.Timestamp("2025-01-15"), smoothing=2)

        self.assertAlmostEqual(encoding["driver_a"], 2 / 3)
        self.assertAlmostEqual(encoding["driver_b"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
