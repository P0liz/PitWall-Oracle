import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.monte_carlo_simulator import SimulationResult
from scripts.generate_web_result import main as publication_main
from src.publication.actual import ResultsNotReadyError, build_history_document
from src.publication.prediction import build_prediction_document
from src.publication.publisher import PublicationError, publish_history_document, publish_prediction_document
from src.publication.scheduler import choose_due_operations
from webapp.api.schemas import PredictionDocument


def prediction_document(session_type: str = "race") -> PredictionDocument:
    drivers = []
    matrix = {}
    for index in range(1, 7):
        driver_id = f"driver_{index}"
        drivers.append(
            {
                "driver_id": driver_id,
                "display_name": f"Driver {index}",
                "abbreviation": f"D{index}",
                "team_id": f"team_{index}",
                "team_name": f"Team {index}",
                "predicted_position": index,
                "expected_position": float(index),
                "win_probability": 1.0 if index == 1 else 0.0,
                "podium_probability": 1.0 if index <= 3 else 0.0,
                "points_probability": 1.0,
                "dnf_probability": 0.0,
                "finish_probability": 1.0,
            }
        )
        matrix[driver_id] = {
            f"driver_{other}": 1.0 if index < other else 0.0 for other in range(1, 7) if index != other
        }
    return PredictionDocument.model_validate(
        {
            "schema_version": "1.0",
            "race": {
                "season": 2026,
                "round": 1,
                "session_type": session_type,
                "name": "Test GP",
                "circuit": "Test",
                "start_time": "2026-03-08T13:00:00Z",
            },
            "publication": {
                "type": "live",
                "generated_at": "2026-03-08T01:00:00Z",
                "data_cutoff": "2026-03-08T01:00:00Z",
                "model_artifact": "pitwall_oracle_2026_base.json",
                "dnf_strategy": "logistic",
                "simulations": 100,
                "seed": 2003,
            },
            "drivers": drivers,
            "head_to_head": matrix,
        }
    )


def official_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_id": [f"driver_{index}" for index in range(1, 7)],
            "Position": [2, 1, 3, 5, 4, 6],
            "Status": ["Finished", "Finished", "Finished", "Finished", "Finished", "Retired"],
        }
    )


class PublicationPipelineTests(unittest.TestCase):
    def test_simulation_is_exported_as_valid_prediction_with_complete_matrix(self):
        driver_ids = np.array(["driver_a", "driver_b", "driver_c"])
        positions = np.array([[1, 2, 3], [2, 1, 3], [1, 3, 2], [3, 2, 1]])
        result = SimulationResult(
            summary=pd.DataFrame(
                {
                    "driver_id": driver_ids,
                    "expected_position": [1.75, 2.0, 2.25],
                    "win_probability": [0.5, 0.25, 0.25],
                    "podium_probability": [1.0, 1.0, 1.0],
                    "points_probability": [1.0, 1.0, 1.0],
                    "dnf_probability": [0.1, 0.2, 0.3],
                    "finish_probability": [0.9, 0.8, 0.7],
                }
            ),
            simulated_positions=positions,
            dnf_draws=np.zeros_like(positions, dtype=bool),
            driver_ids=driver_ids,
            race_frame=pd.DataFrame({"race_date": [pd.Timestamp("2026-03-08")] * 3}),
            ranker_model_path=Path("models/pitwall_oracle_2026_2.json"),
            sigma_relative=0.5,
            sigma_absolute=0.1,
        )
        document = build_prediction_document(
            result,
            pd.DataFrame(
                {
                    "EventName": ["Test Grand Prix"],
                    "Location": ["Test Circuit"],
                    "Session5DateUtc": [pd.Timestamp("2026-03-08T13:00:00Z")],
                }
            ),
            pd.DataFrame(
                {
                    "driver_id": driver_ids,
                    "FirstName": ["A", "B", "C"],
                    "LastName": ["Driver"] * 3,
                    "Abbreviation": ["AAA", "BBB", "CCC"],
                    "team_id": ["team_a", "team_b", "team_c"],
                    "TeamName": ["Team A", "Team B", "Team C"],
                }
            ),
            season=2026,
            round_number=1,
            session_type="race",
            session_number=5,
            generated_at=datetime(2026, 3, 8, 1, tzinfo=timezone.utc),
            simulations=4,
            seed=2003,
        )

        self.assertEqual([driver.predicted_position for driver in document.drivers], [1, 2, 3])
        self.assertEqual(document.publication.model_artifact, "pitwall_oracle_2026_2.json")
        self.assertEqual(document.race.session_type, "race")
        self.assertAlmostEqual(
            document.head_to_head["driver_a"]["driver_b"] + document.head_to_head["driver_b"]["driver_a"], 1.0
        )

    def test_actual_export_handles_dnf_and_rejects_incomplete_results(self):
        prediction = prediction_document()
        history = build_history_document(
            prediction,
            official_results(),
            generated_at=datetime(2026, 3, 9, 13, tzinfo=timezone.utc),
            prediction_path="predictions/2026/round-01-race.json",
        )

        self.assertIsNone(history.actual_results[-1].actual_position)
        self.assertEqual(history.summary.mean_absolute_position_error, 0.8)
        self.assertEqual(history.summary.podium_hits, 3)
        with self.assertRaises(ResultsNotReadyError):
            build_history_document(
                prediction,
                official_results().iloc[:1],
                generated_at=datetime.now(timezone.utc),
                prediction_path="predictions/2026/round-01-race.json",
            )

    def test_scheduler_enforces_prediction_and_actual_boundaries(self):
        schedule = pd.DataFrame(
            {
                "RoundNumber": [1],
                "EventFormat": ["sprint"],
                "Session3DateUtc": [pd.Timestamp("2026-03-07T13:00:00Z")],
                "Session5DateUtc": [pd.Timestamp("2026-03-08T13:00:00Z")],
            }
        )
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            cases = (
                (datetime(2026, 3, 7, 0, 59, 59, tzinfo=timezone.utc), ()),
                (datetime(2026, 3, 7, 1, tzinfo=timezone.utc), ("sprint",)),
                (datetime(2026, 3, 8, 1, tzinfo=timezone.utc), ("race",)),
            )
            for now, expected in cases:
                with self.subTest(now=now):
                    operations = choose_due_operations(schedule, now, data_root, 2026)
                    self.assertEqual(tuple(item.session_type for item in operations), expected)

            archive = data_root / "predictions" / "2026" / "round-01-race.json"
            archive.parent.mkdir(parents=True)
            archive.write_text("{}", encoding="utf-8")
            self.assertEqual(
                choose_due_operations(schedule, datetime(2026, 3, 9, 12, 59, 59, tzinfo=timezone.utc), data_root, 2026),
                (),
            )
            operations = choose_due_operations(schedule, datetime(2026, 3, 9, 13, tzinfo=timezone.utc), data_root, 2026)
            self.assertEqual(
                tuple((item.operation, item.session_type) for item in operations), (("publish-actual", "race"),)
            )

    def test_publication_is_dry_run_safe_and_archives_are_immutable(self):
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            prediction = prediction_document()
            dry_run = publish_prediction_document(prediction, data_root, dry_run=True)
            self.assertEqual(dry_run.status, "validated")
            self.assertFalse((data_root / "predictions").exists())

            publish_prediction_document(prediction, data_root, dry_run=False)
            prediction_path = data_root / "predictions" / "2026" / "round-01-race.json"
            before = prediction_path.read_bytes()
            with self.assertRaises(PublicationError):
                publish_prediction_document(prediction, data_root, dry_run=False)

            history = build_history_document(
                prediction,
                official_results(),
                generated_at=datetime(2026, 3, 9, 13, tzinfo=timezone.utc),
                prediction_path="predictions/2026/round-01-race.json",
            )
            publish_history_document(history, data_root, dry_run=False)
            self.assertEqual(prediction_path.read_bytes(), before)

            sprint = prediction_document("sprint")
            publish_prediction_document(sprint, data_root, dry_run=False)
            sprint_history = build_history_document(
                sprint,
                official_results(),
                generated_at=datetime(2026, 3, 8, 13, tzinfo=timezone.utc),
                prediction_path="predictions/2026/round-01-sprint.json",
            )
            publish_history_document(sprint_history, data_root, dry_run=False)
            index = json.loads((data_root / "history" / "2026" / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [(item["round"], item["session_type"]) for item in index["races"]], [(1, "sprint"), (1, "race")]
            )

    def test_auto_noop_writes_machine_readable_summary(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_path = root / "summary.json"
            exit_code = publication_main(
                [
                    "--operation",
                    "auto",
                    "--season",
                    "2026",
                    "--data-root",
                    str(root / "data"),
                    "--summary-path",
                    str(summary_path),
                ],
                now=datetime(2026, 1, 1, tzinfo=timezone.utc),
                schedule_loader=lambda _: pd.DataFrame(columns=["RoundNumber", "Session5DateUtc"]),
            )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["status"], "no-op")
            self.assertEqual(summary["operations"], [])
            self.assertEqual(summary["changed_paths"], [])


if __name__ == "__main__":
    unittest.main()
