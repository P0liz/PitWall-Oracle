"""Integration tests for the versioned demo result documents."""

from pathlib import Path
import unittest

from webapp.api.repository import ResultsRepository
from webapp.api.schemas import HistoryDocument, HistoryIndex, PredictionDocument


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


class DemoPublishedDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = ResultsRepository(DATA_ROOT)

    def test_current_demo_prediction_is_live_round_twelve(self) -> None:
        prediction = self.repository.load_current()

        self.assertEqual(prediction.race.round, 12)
        self.assertEqual(prediction.race.name, "Demo Grand Prix")
        self.assertEqual(prediction.publication.type, "live")
        self.assertEqual(len(prediction.drivers), 6)
        self.assertEqual(
            prediction.head_to_head["nor_lando_norris"]["ver_max_verstappen"], 0.55
        )

    def test_demo_history_index_lists_the_backtest_round(self) -> None:
        history_index = self.repository.list_history(2026)

        self.assertEqual(len(history_index.races), 1)
        self.assertEqual(history_index.races[0].round, 1)
        self.assertEqual(history_index.races[0].publication_type, "backtest")

    def test_demo_history_contains_expected_round_one_summary(self) -> None:
        history = self.repository.load_history(2026, 1)
        hamilton = next(
            result
            for result in history.actual_results
            if result.driver_id == "ham_lewis_hamilton"
        )

        self.assertEqual(history.summary.mean_absolute_position_error, 0.8)
        self.assertEqual(history.summary.podium_hits, 2)
        self.assertEqual(history.summary.top_five_hits, 5)
        self.assertIsNone(hamilton.actual_position)
        self.assertEqual(hamilton.status, "Retired")

    def test_demo_json_documents_validate_against_public_contracts(self) -> None:
        for prediction_path in (
            DATA_ROOT / "predictions" / "2026" / "round-01.json",
            DATA_ROOT / "predictions" / "2026" / "round-12.json",
        ):
            PredictionDocument.model_validate_json(prediction_path.read_text(encoding="utf-8"))

        HistoryDocument.model_validate_json(
            (DATA_ROOT / "history" / "2026" / "round-01.json").read_text(
                encoding="utf-8"
            )
        )
        HistoryIndex.model_validate_json(
            (DATA_ROOT / "history" / "2026" / "index.json").read_text(encoding="utf-8")
        )


if __name__ == "__main__":
    unittest.main()
