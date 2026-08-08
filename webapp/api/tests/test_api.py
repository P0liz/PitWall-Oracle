import json
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from webapp.api.app import (
    app,
    get_current_prediction,
    get_head_to_head,
    get_history,
    get_history_race,
    health,
)
from webapp.api.repository import ResultsRepository


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def prediction_document() -> dict:
    return {
        "schema_version": "1.0",
        "race": {
            "season": 2026,
            "round": 12,
            "name": "Demo Grand Prix",
            "circuit": "Demo Circuit",
            "start_time": "2026-06-14T12:00:00Z",
        },
        "publication": {
            "type": "live",
            "generated_at": "2026-06-13T12:00:00Z",
            "data_cutoff": "2026-06-13T11:59:59Z",
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
                "expected_position": 1.5,
                "win_probability": 0.5,
                "podium_probability": 0.8,
                "points_probability": 0.95,
                "dnf_probability": 0.05,
                "finish_probability": 0.95,
            },
            {
                "driver_id": "ver_max_verstappen",
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
        ],
        "head_to_head": {
            "nor_lando_norris": {"ver_max_verstappen": 0.6},
            "ver_max_verstappen": {"nor_lando_norris": 0.4},
        },
    }


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temporary_directory.name)
        self.repository = ResultsRepository(self.data_root)
        write_json(
            self.data_root / "predictions" / "current.json",
            {
                "season": 2026,
                "round": 12,
                "prediction_path": "predictions/2026/round-12.json",
            },
        )
        write_json(
            self.data_root / "predictions" / "2026" / "round-12.json",
            prediction_document(),
        )
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
        write_json(
            self.data_root / "history" / "2026" / "round-01.json",
            {
                "schema_version": "1.0",
                "race": {
                    **prediction_document()["race"],
                    "round": 1,
                    "name": "Australian Grand Prix",
                },
                "prediction_path": "predictions/2026/round-01.json",
                "publication": {**prediction_document()["publication"], "type": "backtest"},
                "actual_results": [
                    {
                        "driver_id": "nor_lando_norris",
                        "actual_position": 1,
                        "status": "Finished",
                    }
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
            },
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_public_routes_are_exactly_registered(self) -> None:
        public_paths = {
            route.path for route in app.routes if route.path.startswith("/api/v1")
        }

        self.assertEqual(
            public_paths,
            {
                "/api/v1/health",
                "/api/v1/predictions/current",
                "/api/v1/predictions/current/head-to-head",
                "/api/v1/history",
                "/api/v1/history/{season}/{round_number}",
            },
        )

    def test_handlers_return_repository_response_models(self) -> None:
        self.assertEqual(health()["status"], "ok")
        self.assertEqual(get_current_prediction(self.repository).race.round, 12)
        self.assertEqual(
            get_head_to_head(
                "nor_lando_norris", "ver_max_verstappen", self.repository
            ).driver_a_probability,
            0.6,
        )
        self.assertEqual(get_history(2026, self.repository).races[0].round, 1)
        self.assertEqual(get_history_race(2026, 1, self.repository).race.name, "Australian Grand Prix")

    def test_not_found_repository_error_maps_to_404(self) -> None:
        missing_repository = ResultsRepository(self.data_root / "missing")

        with self.assertRaises(HTTPException) as error:
            get_current_prediction(missing_repository)

        self.assertEqual(error.exception.status_code, 404)

    def test_invalid_published_data_maps_to_422(self) -> None:
        invalid_root = self.data_root / "invalid"
        write_json(
            invalid_root / "history" / "2026" / "index.json",
            {"season": 2026, "races": "not-a-list"},
        )

        with self.assertRaises(HTTPException) as error:
            get_history(2026, ResultsRepository(invalid_root))

        self.assertEqual(error.exception.status_code, 422)

    def test_duplicate_head_to_head_drivers_map_to_400(self) -> None:
        with self.assertRaises(HTTPException) as error:
            get_head_to_head("nor_lando_norris", "nor_lando_norris", self.repository)

        self.assertEqual(error.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
