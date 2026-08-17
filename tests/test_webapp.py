import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import requests
from fastapi import HTTPException
from pydantic import ValidationError

import webapp.ui.history as history_ui
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


def history_payload(
    session_type: str,
    predicted_order: list[str],
    actual_positions: dict[str, int | None],
    podium_hits: int,
    mean_error: float,
) -> dict:
    comparisons = []
    actual_results = []
    for predicted_position, driver_id in enumerate(predicted_order, start=1):
        actual_position = actual_positions[driver_id]
        status = "Retired" if actual_position is None else "Finished"
        comparisons.append(
            {
                "driver_id": driver_id,
                "display_name": driver_id.upper(),
                "predicted_position": predicted_position,
                "actual_position": actual_position,
                "position_difference": (
                    None if actual_position is None else actual_position - predicted_position
                ),
                "status": status,
            }
        )
        actual_results.append(
            {"driver_id": driver_id, "actual_position": actual_position, "status": status}
        )
    prediction = valid_prediction(session_type)
    return {
        "schema_version": "1.0",
        "race": prediction["race"],
        "prediction_path": f"predictions/2026/round-01-{session_type}.json",
        "publication": prediction["publication"],
        "actual_results": actual_results,
        "comparisons": comparisons,
        "summary": {
            "mean_absolute_position_error": mean_error,
            "podium_hits": podium_hits,
            "top_five_hits": 4,
        },
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

    def test_history_index_aggregates_global_statistics_across_gp_and_sprint(self):
        index = {
            "season": 2026,
            "races": [
                {
                    "season": 2026,
                    "round": 1,
                    "session_type": "sprint",
                    "name": "Test GP",
                    "publication_type": "live",
                },
                {
                    "season": 2026,
                    "round": 1,
                    "session_type": "race",
                    "name": "Test GP",
                    "publication_type": "live",
                },
            ],
        }
        write_json(self.data_root / "history" / "2026" / "index.json", index)
        write_json(
            self.data_root / "history" / "2026" / "round-01-sprint.json",
            history_payload(
                "sprint",
                ["a", "b", "c", "d"],
                {"a": 1, "b": None, "c": 2, "d": None},
                podium_hits=2,
                mean_error=2.0,
            ),
        )
        write_json(
            self.data_root / "history" / "2026" / "round-01-race.json",
            history_payload(
                "race",
                ["a", "b", "c", "d"],
                {"a": 2, "b": 1, "c": None, "d": 3},
                podium_hits=3,
                mean_error=4.0,
            ),
        )

        statistics = getattr(self.repository.list_history(2026), "global_statistics", None)

        self.assertIsNotNone(statistics)
        self.assertAlmostEqual(statistics.winner_accuracy, 0.5)
        self.assertAlmostEqual(statistics.podium_hit_rate, 5 / 6)
        self.assertAlmostEqual(statistics.pairwise_accuracy, ((4 / 5) + (4 / 6)) / 2)
        timeline = getattr(statistics, "timeline", None)
        self.assertIsNotNone(timeline)
        self.assertEqual(
            [point.model_dump() for point in timeline],
            [
                {
                    "round": 1,
                    "session_type": "sprint",
                    "winner_accuracy": 1.0,
                    "podium_hit_rate": 2 / 3,
                    "pairwise_accuracy": 4 / 5,
                    "mean_absolute_position_error": 2.0,
                },
                {
                    "round": 1,
                    "session_type": "race",
                    "winner_accuracy": 0.5,
                    "podium_hit_rate": 5 / 6,
                    "pairwise_accuracy": ((4 / 5) + (4 / 6)) / 2,
                    "mean_absolute_position_error": 3.0,
                },
            ],
        )

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

    def test_history_ui_formats_global_statistics_as_percentages(self):
        formatter = getattr(history_ui, "global_stat_metrics", None)

        self.assertIsNotNone(formatter)
        self.assertEqual(
            formatter(
                {
                    "global_statistics": {
                        "winner_accuracy": 0.5,
                        "podium_hit_rate": 5 / 6,
                        "pairwise_accuracy": 11 / 15,
                    }
                }
            ),
            [
                ("Winner accuracy", "50.0%"),
                ("Podium hit rate", "83.3%"),
                ("Pairwise accuracy", "73.3%"),
            ],
        )

    def test_history_ui_formats_cumulative_timeline_for_chart(self):
        formatter = getattr(history_ui, "global_trend_rows", None)

        self.assertIsNotNone(formatter)
        self.assertEqual(
            formatter(
                {
                    "global_statistics": {
                        "timeline": [
                            {
                                "round": 1,
                                "session_type": "sprint",
                                "winner_accuracy": 1.0,
                                "podium_hit_rate": 2 / 3,
                                "pairwise_accuracy": 4 / 5,
                                "mean_absolute_position_error": 2.0,
                            },
                            {
                                "round": 1,
                                "session_type": "race",
                                "winner_accuracy": 0.5,
                                "podium_hit_rate": 5 / 6,
                                "pairwise_accuracy": 11 / 15,
                                "mean_absolute_position_error": 3.0,
                            },
                        ]
                    }
                }
            ),
            [
                {
                    "Event": "R1 Sprint",
                    "Event order": 0,
                    "Winner accuracy": 100.0,
                    "Podium hit rate": 66.7,
                    "Pairwise accuracy": 80.0,
                    "Mean absolute position error": 2.0,
                },
                {
                    "Event": "R1 Race",
                    "Event order": 1,
                    "Winner accuracy": 50.0,
                    "Podium hit rate": 83.3,
                    "Pairwise accuracy": 73.3,
                    "Mean absolute position error": 3.0,
                },
            ],
        )

    def test_history_accuracy_chart_preserves_chronological_event_order(self):
        chart_builder = getattr(history_ui, "accuracy_chart_spec", None)
        rows = [
            {
                "Event": label,
                "Event order": event_order,
                "Winner accuracy": 50.0,
                "Podium hit rate": 60.0,
                "Pairwise accuracy": 70.0,
                "Mean absolute position error": 2.0,
            }
            for event_order, label in enumerate(("R1 Race", "R2 Race", "R10 Race"))
        ]

        self.assertIsNotNone(chart_builder)
        chart_spec = chart_builder(rows)
        self.assertEqual(chart_spec["encoding"]["x"]["sort"], ["R1 Race", "R2 Race", "R10 Race"])
        self.assertEqual(chart_spec["encoding"]["y"]["scale"]["domain"], [0, 100])

    def test_history_mae_chart_uses_a_separate_position_scale(self):
        chart_builder = getattr(history_ui, "mae_chart_spec", None)
        rows = [
            {"Event": "R1 Race", "Event order": 0, "Mean absolute position error": 2.0},
            {"Event": "R10 Race", "Event order": 1, "Mean absolute position error": 3.0},
        ]

        self.assertIsNotNone(chart_builder)
        chart_spec = chart_builder(rows)
        self.assertEqual(chart_spec["encoding"]["x"]["sort"], ["R1 Race", "R10 Race"])
        self.assertEqual(chart_spec["encoding"]["y"]["field"], "Mean absolute position error")
        self.assertNotIn("scale", chart_spec["encoding"]["y"])

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

    def test_streamlit_entrypoint_exposes_repository_package_to_cloud_pages(self):
        repository_root = Path(__file__).resolve().parents[1]
        entrypoint = repository_root / "webapp" / "ui" / "streamlit_app.py"
        cloud_import = f"""
import runpy
import sys
from pathlib import Path

root = Path({str(repository_root)!r}).resolve()
ui = root / "webapp" / "ui"
sys.path = [str(ui)] + [
    path
    for path in sys.path
    if path and not str(path).startswith("__editable__") and Path(path).resolve() != root
]
sys.meta_path = [finder for finder in sys.meta_path if "__editable__" not in repr(finder)]
runpy.run_path({str(entrypoint)!r}, run_name="streamlit_cloud_entrypoint")
import webapp.ui.current_prediction
import webapp.ui.history
"""

        completed = subprocess.run(
            [sys.executable, "-c", cloud_import], cwd=repository_root, capture_output=True, text=True
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
