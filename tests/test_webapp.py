import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import requests
from fastapi import HTTPException
from pydantic import ValidationError

from webapp.api.app import get_current_prediction, get_head_to_head, get_history, get_history_race
from webapp.api.repository import InvalidPublishedData, ResultNotFound, ResultsRepository
from webapp.api.schemas import HistoryDocument, HistoryIndex, PredictionDocument
from webapp.ui.api_client import ApiUnavailable, PitWallApiClient, PredictionUnavailable
from webapp.ui.current_prediction import fetch_head_to_head, head_to_head_options
from webapp.ui.formatting import prediction_rows
from webapp.ui.history import _display_history_rows, history_options


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def valid_prediction(session_type: str = "race") -> dict:
    drivers = [
        {
            "driver_id": "nor",
            "display_name": "Lando Norris",
            "abbreviation": "NOR",
            "team_id": "mclaren",
            "team_name": "McLaren",
            "predicted_position": 1,
            "expected_position": 1.5,
            "win_probability": 0.6,
            "podium_probability": 0.8,
            "points_probability": 0.95,
            "dnf_probability": 0.05,
            "finish_probability": 0.95,
        },
        {
            "driver_id": "ver",
            "display_name": "Max Verstappen",
            "abbreviation": "VER",
            "team_id": "red_bull",
            "team_name": "Red Bull Racing",
            "predicted_position": 2,
            "expected_position": 2.2,
            "win_probability": 0.3,
            "podium_probability": 0.7,
            "points_probability": 0.92,
            "dnf_probability": 0.08,
            "finish_probability": 0.92,
        },
    ]
    return {
        "schema_version": "1.0",
        "race": {
            "season": 2026,
            "round": 1,
            "session_type": session_type,
            "name": "Test Grand Prix",
            "circuit": "Test Circuit",
            "start_time": "2026-03-08T13:00:00Z",
        },
        "publication": {
            "type": "live",
            "generated_at": "2026-03-08T01:00:00Z",
            "data_cutoff": "2026-03-08T00:59:59Z",
            "model_artifact": "pitwall_oracle_2026_base.json",
            "dnf_strategy": "logistic",
            "simulations": 10000,
            "seed": 2003,
        },
        "drivers": drivers,
        "head_to_head": {"nor": {"ver": 0.6}, "ver": {"nor": 0.4}},
    }


class WebappContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temporary_directory.name)
        self.repository = ResultsRepository(self.data_root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def publish_prediction(self) -> None:
        write_json(
            self.data_root / "predictions" / "current.json",
            {
                "season": 2026,
                "round": 1,
                "session_type": "race",
                "prediction_path": "predictions/2026/round-01-race.json",
            },
        )
        write_json(self.data_root / "predictions" / "2026" / "round-01-race.json", valid_prediction())

    def test_repository_loads_current_prediction_and_head_to_head(self):
        self.publish_prediction()

        prediction = self.repository.load_current()
        duel = self.repository.load_head_to_head("nor", "ver")

        self.assertEqual(prediction.race.round, 1)
        self.assertEqual(duel.driver_a_probability, 0.6)
        self.assertEqual(duel.driver_b_probability, 0.4)

    def test_schema_rejects_broken_driver_and_matrix_invariants(self):
        cases = []
        duplicate_driver = valid_prediction()
        duplicate_driver["drivers"][1]["driver_id"] = "nor"
        cases.append(duplicate_driver)
        duplicate_position = valid_prediction()
        duplicate_position["drivers"][1]["predicted_position"] = 1
        cases.append(duplicate_position)
        non_complementary = valid_prediction()
        non_complementary["head_to_head"]["ver"]["nor"] = 0.5
        cases.append(non_complementary)

        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    PredictionDocument.model_validate(payload)

    def test_repository_blocks_path_escape_and_completed_race_pointer(self):
        write_json(
            self.data_root / "predictions" / "current.json",
            {"season": 2026, "round": 1, "session_type": "race", "prediction_path": "../../outside.json"},
        )
        with self.assertRaises(InvalidPublishedData):
            self.repository.load_current()

        self.publish_prediction()
        write_json(self.data_root / "history" / "2026" / "round-01-race.json", {})
        with self.assertRaises(ResultNotFound):
            self.repository.load_current()

    def test_api_handlers_return_models_and_translate_not_found(self):
        self.publish_prediction()
        self.assertEqual(get_current_prediction(self.repository).race.round, 1)
        self.assertEqual(get_head_to_head("nor", "ver", self.repository).driver_a_probability, 0.6)

        with self.assertRaises(HTTPException) as error:
            get_history(2026, ResultsRepository(self.data_root / "missing"))
        self.assertEqual(error.exception.status_code, 404)

    def test_versioned_demo_documents_match_public_contracts(self):
        data_root = Path(__file__).resolve().parents[1] / "webapp" / "api" / "data"
        for path in (data_root / "predictions" / "2026").glob("round-*.json"):
            PredictionDocument.model_validate_json(path.read_text(encoding="utf-8"))
        HistoryDocument.model_validate_json(
            (data_root / "history" / "2026" / "round-01-race.json").read_text(encoding="utf-8")
        )
        HistoryIndex.model_validate_json((data_root / "history" / "2026" / "index.json").read_text(encoding="utf-8"))

    def test_history_distinguishes_sprint_and_race_in_the_same_round(self):
        index = {
            "season": 2026,
            "races": [
                {"season": 2026, "round": 1, "session_type": "sprint", "name": "Test GP", "publication_type": "live"},
                {"season": 2026, "round": 1, "session_type": "race", "name": "Test GP", "publication_type": "live"},
            ],
        }
        write_json(self.data_root / "history" / "2026" / "index.json", index)
        for session_type in ("sprint", "race"):
            prediction = valid_prediction(session_type)
            history = {
                "schema_version": "1.0",
                "race": prediction["race"],
                "prediction_path": f"predictions/2026/round-01-{session_type}.json",
                "publication": prediction["publication"],
                "actual_results": [{"driver_id": "nor", "actual_position": 1, "status": "Finished"}],
                "comparisons": [
                    {
                        "driver_id": "nor",
                        "display_name": "Lando Norris",
                        "predicted_position": 1,
                        "actual_position": 1,
                        "position_difference": 0,
                        "status": "Finished",
                    }
                ],
                "summary": {"mean_absolute_position_error": 0.0, "podium_hits": 1, "top_five_hits": 1},
            }
            write_json(self.data_root / "history" / "2026" / f"round-01-{session_type}.json", history)

        self.assertEqual(self.repository.load_history(2026, 1, "sprint").race.session_type, "sprint")
        self.assertEqual(get_history_race(2026, 1, self.repository, "race").race.session_type, "race")
        self.assertEqual(list(history_options(index).values()), [(2026, 1, "sprint"), (2026, 1, "race")])

        session = Mock()
        session.get.return_value = self.response(200, {"race": {"session_type": "sprint"}})
        client = PitWallApiClient("https://api.example.test", session=session)
        client.get_history(2026, 1, "sprint")
        self.assertTrue(session.get.call_args.args[0].endswith("/api/v1/history/2026/1?session_type=sprint"))

    @staticmethod
    def response(status_code: int, payload: dict) -> Mock:
        response = Mock()
        response.status_code = status_code
        response.json.return_value = payload
        return response

    def test_api_client_maps_success_not_available_and_transport_errors(self):
        session = Mock()
        client = PitWallApiClient("https://api.example.test/", session=session)
        session.get.return_value = self.response(200, {"schema_version": "1.0"})
        self.assertEqual(client.get_current_prediction(), {"schema_version": "1.0"})
        self.assertEqual(session.get.call_args.args[0], "https://api.example.test/api/v1/predictions/current")

        session.get.return_value = self.response(
            404, {"detail": {"code": "prediction_not_available", "message": "No prediction available"}}
        )
        with self.assertRaises(PredictionUnavailable):
            client.get_current_prediction()

        session.get.side_effect = requests.Timeout()
        with self.assertRaises(ApiUnavailable):
            client.get_current_prediction()

    def test_head_to_head_ui_uses_stable_distinct_driver_ids(self):
        document = {
            "drivers": [
                {"display_name": "Lando Norris", "driver_id": "nor"},
                {"display_name": "Max Verstappen", "driver_id": "ver"},
            ]
        }
        client = Mock()
        client.get_head_to_head.return_value = {"driver_a_id": "nor", "driver_b_id": "ver"}

        self.assertEqual(head_to_head_options(document), {"Lando Norris": "nor", "Max Verstappen": "ver"})
        self.assertIsNone(fetch_head_to_head(client, "nor", "nor"))
        self.assertEqual(fetch_head_to_head(client, "nor", "ver")["driver_b_id"], "ver")
        client.get_head_to_head.assert_called_once_with("nor", "ver")

    def test_history_ui_shows_status_instead_of_fake_position(self):
        rows = _display_history_rows(
            {
                "comparisons": [
                    {
                        "display_name": "Lewis Hamilton",
                        "predicted_position": 6,
                        "actual_position": None,
                        "position_difference": None,
                        "status": "Retired",
                    }
                ]
            }
        )

        self.assertEqual(rows[0]["Actual"], "Retired")
        self.assertEqual(rows[0]["Difference"], "Not classified")

    def test_prediction_and_history_tables_use_requested_column_order(self):
        prediction = prediction_rows(valid_prediction())[0]
        history = _display_history_rows(
            {
                "comparisons": [
                    {
                        "display_name": "Lewis Hamilton",
                        "predicted_position": 6,
                        "actual_position": 4,
                        "position_difference": -2,
                        "status": "Finished",
                    }
                ]
            }
        )[0]

        self.assertEqual(list(prediction), ["Position", "Driver", "Team", "Win", "Podium", "Points", "DNF"])
        self.assertEqual(list(history), ["Driver", "Predicted", "Actual", "Difference"])
        self.assertEqual(history["Predicted"], "6")


if __name__ == "__main__":
    unittest.main()
