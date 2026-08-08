import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from webapp.api.repository import InvalidPublishedData, ResultNotFound, ResultsRepository
from webapp.api.schemas import PredictionDocument


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def valid_prediction() -> dict:
    return {
        "schema_version": "1.0",
        "race": {
            "season": 2026,
            "round": 1,
            "name": "Australian Grand Prix",
            "circuit": "Melbourne",
            "start_time": "2026-03-08T04:00:00Z",
        },
        "publication": {
            "type": "live",
            "generated_at": "2026-03-07T07:00:00Z",
            "data_cutoff": "2026-03-07T06:59:59Z",
            "model_artifact": "pitwall_oracle_latest.json",
            "dnf_strategy": "team_beta",
            "simulations": 10000,
            "seed": 2003,
        },
        "drivers": [
            {
                "driver_id": "nor_lando_norris",
                "display_name": "Lando Norris",
                "abbreviation": "NOR",
                "team_id": "mclaren",
                "team_name": "McLaren",
                "predicted_position": 1,
                "expected_position": 2.1,
                "win_probability": 0.42,
                "podium_probability": 0.75,
                "points_probability": 0.96,
                "dnf_probability": 0.08,
                "finish_probability": 0.92,
            },
            {
                "driver_id": "ver_max_verstappen",
                "display_name": "Max Verstappen",
                "abbreviation": "VER",
                "team_id": "red_bull_lineage",
                "team_name": "Red Bull Racing",
                "predicted_position": 2,
                "expected_position": 2.7,
                "win_probability": 0.31,
                "podium_probability": 0.68,
                "points_probability": 0.94,
                "dnf_probability": 0.07,
                "finish_probability": 0.93,
            },
        ],
        "head_to_head": {
            "nor_lando_norris": {"ver_max_verstappen": 0.58},
            "ver_max_verstappen": {"nor_lando_norris": 0.42},
        },
    }


class ResultsRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temporary_directory.name)
        self.repository = ResultsRepository(self.data_root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_current_prediction(self) -> None:
        write_json(
            self.data_root / "predictions" / "current.json",
            {
                "season": 2026,
                "round": 1,
                "prediction_path": "predictions/2026/round-01.json",
            },
        )
        write_json(
            self.data_root / "predictions" / "2026" / "round-01.json",
            valid_prediction(),
        )

    def test_load_current_returns_pointed_prediction(self) -> None:
        self.write_current_prediction()

        self.assertEqual(self.repository.load_current().race.round, 1)

    def test_load_head_to_head_returns_requested_direction(self) -> None:
        self.write_current_prediction()

        response = self.repository.load_head_to_head(
            "nor_lando_norris", "ver_max_verstappen"
        )

        self.assertEqual(response.driver_a_probability, 0.58)
        self.assertEqual(response.driver_a_name, "Lando Norris")
        self.assertEqual(response.driver_b_name, "Max Verstappen")

    def test_load_current_raises_when_no_pointer_is_published(self) -> None:
        with self.assertRaises(ResultNotFound):
            self.repository.load_current()

    def test_prediction_rejects_probability_above_one(self) -> None:
        payload_with_probability_above_one = valid_prediction()
        payload_with_probability_above_one["drivers"][0]["win_probability"] = 1.01

        with self.assertRaises(ValidationError):
            PredictionDocument.model_validate(payload_with_probability_above_one)

    def test_load_current_rejects_pointer_escaping_data_root(self) -> None:
        write_json(
            self.data_root / "predictions" / "current.json",
            {
                "season": 2026,
                "round": 1,
                "prediction_path": "../../outside.json",
            },
        )

        with self.assertRaises(InvalidPublishedData):
            self.repository.load_current()

    def test_prediction_rejects_duplicate_driver_ids(self) -> None:
        payload = valid_prediction()
        payload["drivers"][1]["driver_id"] = "nor_lando_norris"

        with self.assertRaises(ValidationError):
            PredictionDocument.model_validate(payload)

    def test_prediction_rejects_duplicate_predicted_positions(self) -> None:
        payload = valid_prediction()
        payload["drivers"][1]["predicted_position"] = 1

        with self.assertRaises(ValidationError):
            PredictionDocument.model_validate(payload)

    def test_prediction_rejects_incomplete_head_to_head_directions(self) -> None:
        payload = valid_prediction()
        del payload["head_to_head"]["ver_max_verstappen"]

        with self.assertRaises(ValidationError):
            PredictionDocument.model_validate(payload)

    def test_prediction_rejects_non_complementary_head_to_head_probabilities(self) -> None:
        payload = valid_prediction()
        payload["head_to_head"]["ver_max_verstappen"]["nor_lando_norris"] = 0.43

        with self.assertRaises(ValidationError):
            PredictionDocument.model_validate(payload)

    def test_load_current_rejects_completed_race(self) -> None:
        self.write_current_prediction()
        write_json(self.data_root / "history" / "2026" / "round-01.json", {})

        with self.assertRaises(ResultNotFound) as error:
            self.repository.load_current()

        self.assertEqual(error.exception.code, "prediction_not_available")

    def test_history_methods_load_index_and_race_document(self) -> None:
        history_document = {
            "schema_version": "1.0",
            "race": valid_prediction()["race"],
            "prediction_path": "predictions/2026/round-01.json",
            "publication": {**valid_prediction()["publication"], "type": "backtest"},
            "actual_results": [
                {"driver_id": "nor_lando_norris", "actual_position": 1, "status": "Finished"}
            ],
            "comparisons": [
                {
                    "driver_id": "nor_lando_norris",
                    "display_name": "Lando Norris",
                    "predicted_position": 1,
                    "actual_position": 1,
                    "position_difference": 0,
                    "status": "Finished",
                }
            ],
            "summary": {
                "mean_absolute_position_error": 0.0,
                "podium_hits": 1,
                "podium_total": 3,
                "top_five_hits": 1,
                "top_five_total": 5,
            },
        }
        write_json(
            self.data_root / "history" / "2026" / "index.json",
            {
                "season": 2026,
                "races": [
                    {
                        "season": 2026,
                        "round": 1,
                        "name": "Australian Grand Prix",
                        "publication_type": "backtest",
                    }
                ],
            },
        )
        write_json(self.data_root / "history" / "2026" / "round-01.json", history_document)

        self.assertEqual(self.repository.list_history(2026).races[0].round, 1)
        self.assertEqual(self.repository.load_history(2026, 1).summary.podium_hits, 1)


if __name__ == "__main__":
    unittest.main()
