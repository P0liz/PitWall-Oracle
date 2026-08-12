import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd

from src.data.data_loader import DataLoader
from src.dnf_features import DNF_CANDIDATE_FEATURES


class DataLoaderCacheTests(unittest.IsolatedAsyncioTestCase):
    def make_loader(self, data_dir: Path) -> DataLoader:
        loader = DataLoader.__new__(DataLoader)
        loader.data_dir = data_dir
        loader.train_df = None
        loader.test_df = None
        loader.dnf_df = None
        loader.log = Mock()
        return loader

    @staticmethod
    def create_cache_files(data_dir: Path) -> list[Path]:
        paths = [data_dir / "train_df.parquet", data_dir / "test_df.parquet", data_dir / "dnf_df.parquet"]
        for path in paths:
            path.write_bytes(b"unchanged")
        return paths

    @staticmethod
    def valid_dnf_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "race_date": pd.to_datetime(["2025-01-01"]),
                "dnf_target": [False],
                **{feature: [0.0] for feature in DNF_CANDIDATE_FEATURES},
            }
        )

    async def test_static_mode_loads_all_caches_without_a_cutoff(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            self.create_cache_files(data_dir)
            loader = self.make_loader(data_dir)
            frames = {
                "train_df.parquet": pd.DataFrame({"race_date": pd.to_datetime(["2025-01-01"])}),
                "test_df.parquet": pd.DataFrame({"race_date": pd.to_datetime(["2025-02-01"])}),
                "dnf_df.parquet": self.valid_dnf_frame(),
            }

            with patch("src.data.data_loader.pd.read_parquet", side_effect=lambda path: frames[Path(path).name].copy()):
                train_df, test_df = await loader.load_data(is_dynamic=False)

            pd.testing.assert_frame_equal(train_df, frames["train_df.parquet"])
            pd.testing.assert_frame_equal(test_df, frames["test_df.parquet"])
            pd.testing.assert_frame_equal(loader.dnf_df, frames["dnf_df.parquet"])

    async def test_static_mode_rebuilds_when_dnf_cache_schema_is_stale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            self.create_cache_files(data_dir)
            loader = self.make_loader(data_dir)
            cached_frames = {
                "train_df.parquet": pd.DataFrame({"race_date": pd.to_datetime(["2025-01-01"])}),
                "test_df.parquet": pd.DataFrame({"race_date": pd.to_datetime(["2025-02-01"])}),
                "dnf_df.parquet": pd.DataFrame({"race_date": pd.to_datetime(["2025-01-01"]), "dnf_target": [False]}),
            }
            rebuilt_train = pd.DataFrame({"race_date": pd.to_datetime(["2025-03-01"]), "target": [1.0]})
            rebuilt_test = pd.DataFrame({"race_date": pd.to_datetime(["2025-04-01"]), "target": [2.0]})
            rebuilt_dnf = self.valid_dnf_frame()

            async def build_static_data(force):
                loader.dnf_df = rebuilt_dnf
                return rebuilt_train, rebuilt_test

            loader.build_static_data = AsyncMock(side_effect=build_static_data)
            with patch(
                "src.data.data_loader.pd.read_parquet", side_effect=lambda path: cached_frames[Path(path).name].copy()
            ):
                train_df, test_df = await loader.load_data(is_dynamic=False)

            pd.testing.assert_frame_equal(train_df, rebuilt_train)
            pd.testing.assert_frame_equal(test_df, rebuilt_test)
            pd.testing.assert_frame_equal(loader.dnf_df, rebuilt_dnf)

    async def test_static_rebuild_persists_ranker_and_dnf_caches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            loader = self.make_loader(data_dir)
            train_df = pd.DataFrame({"race_date": pd.to_datetime(["2025-01-01"]), "target": [1.0]})
            test_df = pd.DataFrame({"race_date": pd.to_datetime(["2025-02-01"]), "target": [2.0]})
            dnf_df = pd.DataFrame({"race_date": pd.to_datetime(["2025-01-01"]), "dnf_target": [False]})

            async def build_static_data(force):
                loader.dnf_df = dnf_df
                return train_df, test_df

            loader.build_static_data = build_static_data

            await loader.load_data(is_dynamic=False, force=True)

            self.assertTrue((data_dir / "train_df.parquet").exists())
            self.assertTrue((data_dir / "test_df.parquet").exists())
            self.assertTrue((data_dir / "dnf_df.parquet").exists())

    async def test_dynamic_mode_builds_in_memory_without_touching_static_caches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            cache_paths = self.create_cache_files(data_dir)
            loader = self.make_loader(data_dir)
            static_df = pd.DataFrame({"race_date": pd.to_datetime(["2025-01-01"]), "target": [1.0]})
            cutoff = pd.Timestamp("2026-03-08", tz="UTC")
            expected_train = pd.DataFrame(
                {"race_date": pd.to_datetime(["2025-01-01", "2026-03-08"]), "target": [1.0, 2.0]}
            )
            expected_test = pd.DataFrame()
            loader.build_dynamic_data = AsyncMock(return_value=(expected_train, expected_test))

            with patch("src.data.data_loader.pd.read_parquet", return_value=static_df.copy()):
                train_df, test_df = await loader.load_data(
                    last_date=cutoff, static_df=static_df, is_dynamic=True, force=False
                )

            pd.testing.assert_frame_equal(train_df, expected_train)
            pd.testing.assert_frame_equal(test_df, expected_test)
            self.assertTrue(all(path.read_bytes() == b"unchanged" for path in cache_paths))

    async def test_dynamic_mode_requires_history_and_cutoff(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = self.make_loader(Path(temp_dir))
            static_df = pd.DataFrame({"race_date": pd.to_datetime(["2025-01-01"]), "target": [1.0]})

            with self.assertRaises(ValueError):
                await loader.load_data(last_date=pd.Timestamp("2026-03-08", tz="UTC"), is_dynamic=True)
            with self.assertRaises(ValueError):
                await loader.load_data(static_df=static_df, is_dynamic=True)

    async def test_dynamic_build_rejects_rows_after_the_cutoff(self):
        loader = self.make_loader(Path("unused"))
        loader.gold = Mock()
        static_df = pd.DataFrame({"race_date": pd.to_datetime(["2025-01-01"]), "target": [1.0]})
        future_race = pd.DataFrame({"race_date": pd.to_datetime(["2026-03-09"]), "target": [2.0]})
        loader.gold.build_features.return_value = [future_race]

        def assemble_dynamic_data(races, race_is_test, historical):
            loader.train_df = pd.concat([historical, *races], ignore_index=True)
            loader.test_df = pd.DataFrame()

        loader._apply_encoding = assemble_dynamic_data
        schedule = pd.DataFrame({"Session5DateUtc": pd.to_datetime(["2026-03-08"], utc=True)})

        with patch("src.data.data_loader.fastf1.get_event_schedule", return_value=schedule):
            with self.assertRaises(ValueError):
                await loader.build_dynamic_data(static_df, pd.Timestamp("2026-03-08", tz="UTC"))

    def test_target_encoding_uses_only_history_strictly_before_cutoff(self):
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
