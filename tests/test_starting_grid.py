import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from src.data.starting_grid import StartingGridResolver, find_provisional_grid_pdf, parse_grid_words
from src.data.gold_layer import GoldLayer


class StartingGridDocumentTests(unittest.TestCase):
    def test_finds_active_gp_document_and_ignores_recalled_document(self):
        html = """
        <li class="document-row">
          <div class="recalled-document">
            <div class="title">Recalled - Doc 40 - Provisional Starting Grid</div>
          </div>
        </li>
        <li class="document-row">
          <a href="/system/files/active-grid.pdf">
            <div class="title">Doc 41 - Provisional Starting Grid</div>
            <div class="published">Published on <span>25.07.26 20:00</span> CET</div>
          </a>
        </li>
        <li class="document-row">
          <a href="/system/files/sprint-grid.pdf">
            <div class="title">Doc 12 - Provisional Sprint Starting Grid</div>
          </a>
        </li>
        """

        self.assertEqual(
            find_provisional_grid_pdf(html, sprint=False), "https://www.fia.com/system/files/active-grid.pdf"
        )

    def test_parses_both_pdf_columns_in_grid_order(self):
        words = [
            {"text": "Provisional", "x0": 210.0, "x1": 280.0, "top": 154.0},
            {"text": "Grid", "x0": 350.0, "x1": 375.0, "top": 154.0},
            {"text": "1", "x0": 52.0, "x1": 56.0, "top": 190.0},
            {"text": "4", "x0": 75.0, "x1": 79.0, "top": 188.0},
            {"text": "Lando", "x0": 86.0, "x1": 108.0, "top": 188.0},
            {"text": "2", "x0": 317.0, "x1": 321.0, "top": 202.0},
            {"text": "81", "x0": 335.0, "x1": 344.0, "top": 200.0},
            {"text": "Oscar", "x0": 350.0, "x1": 372.0, "top": 200.0},
            {"text": "3", "x0": 52.0, "x1": 56.0, "top": 220.0},
            {"text": "1", "x0": 70.0, "x1": 74.0, "top": 218.0},
            {"text": "Max", "x0": 86.0, "x1": 103.0, "top": 218.0},
            {"text": "PENALTIES", "x0": 276.0, "x1": 325.0, "top": 545.0},
            {"text": "44", "x0": 34.0, "x1": 43.0, "top": 557.0},
        ]

        self.assertEqual(parse_grid_words(words, page_width=595.0), {1: 4, 2: 81, 3: 1})


class StartingGridResolverTests(unittest.TestCase):
    @staticmethod
    def qualifying() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "DriverNumber": ["4", "81", "1"],
                "driver_id": ["nor_lando_norris", "pia_oscar_piastri", "ver_max_verstappen"],
                "Position": [1.0, 2.0, 3.0],
            }
        )

    def test_manual_override_has_priority_over_fia(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            override_path = data_dir / "2026_12_5_override.json"
            override_path.write_text(
                json.dumps({"grid": {"nor_lando_norris": 2, "pia_oscar_piastri": 1, "ver_max_verstappen": 3}}),
                encoding="utf-8",
            )

            def unexpected_get(*args, **kwargs):
                raise AssertionError("FIA must not be called when an override exists")

            resolver = StartingGridResolver(data_dir=data_dir, http_get=unexpected_get)
            result = resolver.resolve(2026, 12, 5, "Hungarian Grand Prix", self.qualifying())

            self.assertEqual(result.source, "manual_override")
            self.assertEqual(result.positions["nor_lando_norris"], 2.0)
            self.assertEqual(result.positions["pia_oscar_piastri"], 1.0)

    def test_uses_fia_grid_and_maps_car_numbers_to_driver_ids(self):
        class Response:
            def __init__(self, *, text="", content=b""):
                self.text = text
                self.content = content

            def raise_for_status(self):
                return None

        html = '<a href="/system/files/grid.pdf"><div>Doc 46 - Provisional Starting Grid</div></a>'
        responses = [Response(text=html), Response(content=b"pdf")]

        resolver = StartingGridResolver(
            data_dir=Path("unused"),
            http_get=lambda *args, **kwargs: responses.pop(0),
            pdf_parser=lambda content: {1: 81, 2: 4, 3: 1},
        )
        result = resolver.resolve(2026, 12, 5, "Hungarian Grand Prix", self.qualifying(), cache=False)

        self.assertEqual(result.source, "fia_provisional")
        self.assertEqual(
            result.positions, {"nor_lando_norris": 2.0, "pia_oscar_piastri": 1.0, "ver_max_verstappen": 3.0}
        )
        self.assertEqual(result.source_url, "https://www.fia.com/system/files/grid.pdf")


class GoldPredictionGridTests(unittest.TestCase):
    def test_prediction_features_use_the_resolved_grid(self):
        event = pd.DataFrame({"EventFormat": ["conventional"], "EventName": ["Hungarian Grand Prix"]})
        qualifying = StartingGridResolverTests.qualifying()
        resolved_positions = {"nor_lando_norris": 2.0, "pia_oscar_piastri": 1.0, "ver_max_verstappen": 3.0}
        gold = GoldLayer()
        gold.data_dir = Path("unused")
        gold.silver = Mock()
        gold.silver.get_clean_event_metadata.return_value = event
        gold.silver.get_clean_results.return_value = qualifying
        gold.starting_grid_resolver = Mock()
        gold.starting_grid_resolver.resolve.return_value = SimpleNamespace(
            positions=resolved_positions, source="fia_provisional", source_url="https://www.fia.com/grid.pdf"
        )

        def build_with_grid(event, year, race_number, session, force, prediction_mode, prediction_grid):
            return pd.DataFrame({"driver_id": list(prediction_grid), "grid_position": list(prediction_grid.values())})

        gold.get_gp_features = build_with_grid

        with tempfile.TemporaryDirectory() as temp_dir:
            gold.data_dir = Path(temp_dir)
            result = gold.build_prediction_features(2026, 12, 5)

        self.assertEqual(result.set_index("driver_id")["grid_position"].to_dict(), resolved_positions)
        self.assertEqual(result.attrs["grid_source"], "fia_provisional")
        self.assertEqual(result.attrs["grid_source_url"], "https://www.fia.com/grid.pdf")


if __name__ == "__main__":
    unittest.main()
