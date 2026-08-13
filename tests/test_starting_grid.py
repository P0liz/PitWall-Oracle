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

    def test_finds_sprint_document_without_confusing_it_with_gp(self):
        html = """
        <a href="/system/files/gp-grid.pdf"><div>Doc 41 - Provisional Starting Grid</div></a>
        <a href="/system/files/sprint-grid.pdf"><div>Doc 12 - Provisional Sprint Starting Grid</div></a>
        """

        self.assertEqual(
            find_provisional_grid_pdf(html, sprint=True), "https://www.fia.com/system/files/sprint-grid.pdf"
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

    def test_parses_grid_page_when_title_is_not_in_the_text_layer(self):
        words = [
            {"text": "1", "x0": 52.0, "x1": 56.0, "top": 190.0},
            {"text": "44", "x0": 70.0, "x1": 79.0, "top": 188.0},
            {"text": "Lewis", "x0": 86.0, "x1": 108.0, "top": 188.0},
            {"text": "2", "x0": 317.0, "x1": 321.0, "top": 202.0},
            {"text": "12", "x0": 335.0, "x1": 344.0, "top": 200.0},
            {"text": "Kimi", "x0": 350.0, "x1": 370.0, "top": 200.0},
        ]

        self.assertEqual(parse_grid_words(words, page_width=595.0), {1: 44, 2: 12})

    def test_ignores_grid_word_in_penalty_note_below_the_grid(self):
        words = [
            {"text": "1", "x0": 52.3, "x1": 59.0, "top": 189.7},
            {"text": "12", "x0": 70.4, "x1": 79.4, "top": 187.6},
            {"text": "Kimi", "x0": 85.6, "x1": 101.2, "top": 187.6},
            {"text": "2", "x0": 317.2, "x1": 323.9, "top": 201.5},
            {"text": "16", "x0": 335.2, "x1": 344.1, "top": 199.3},
            {"text": "Charles", "x0": 350.4, "x1": 376.3, "top": 199.3},
            {"text": "PENALTIES", "x0": 276.0, "x1": 325.0, "top": 544.7},
            {"text": "grid", "x0": 75.3, "x1": 88.0, "top": 556.8},
        ]

        self.assertEqual(parse_grid_words(words, page_width=595.0), {1: 12, 2: 16})


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
                json.dumps(
                    {
                        "grid": {
                            "nor_lando_norris": 2,
                            "pia_oscar_piastri": 1,
                            "ver_max_verstappen": 3,
                        }
                    }
                ),
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
            result.positions,
            {"nor_lando_norris": 2.0, "pia_oscar_piastri": 1.0, "ver_max_verstappen": 3.0},
        )
        self.assertEqual(result.source_url, "https://www.fia.com/system/files/grid.pdf")

    def test_falls_back_to_qualifying_when_fia_document_is_unavailable(self):
        class Response:
            text = "<html><body>No grid yet</body></html>"
            content = b""

            def raise_for_status(self):
                return None

        resolver = StartingGridResolver(data_dir=Path("unused"), http_get=lambda *args, **kwargs: Response())
        with self.assertWarnsRegex(
            RuntimeWarning,
            "FIA provisional starting grid unavailable for 2026 round 12 session 5 "
            r"\(Hungarian Grand Prix\); using qualifying order",
        ):
            result = resolver.resolve(2026, 12, 5, "Hungarian Grand Prix", self.qualifying(), cache=False)

        self.assertEqual(result.source, "qualifying_fallback")
        self.assertEqual(
            result.positions,
            {"nor_lando_norris": 1.0, "pia_oscar_piastri": 2.0, "ver_max_verstappen": 3.0},
        )


class GoldPredictionGridTests(unittest.TestCase):
    def test_prediction_features_use_the_resolved_grid(self):
        event = pd.DataFrame({"EventFormat": ["conventional"], "EventName": ["Hungarian Grand Prix"]})
        qualifying = StartingGridResolverTests.qualifying()
        resolved_positions = {
            "nor_lando_norris": 2.0,
            "pia_oscar_piastri": 1.0,
            "ver_max_verstappen": 3.0,
        }
        gold = GoldLayer.__new__(GoldLayer)
        gold.data_dir = Path("unused")
        gold.silver = Mock()
        gold.silver.get_clean_event_metadata.return_value = event
        gold.silver.get_clean_results.return_value = qualifying
        gold.starting_grid_resolver = Mock()
        gold.starting_grid_resolver.resolve.return_value = SimpleNamespace(positions=resolved_positions)

        def build_with_grid(event, year, race_number, session, force, prediction_mode, prediction_grid):
            return pd.DataFrame(
                {
                    "driver_id": list(prediction_grid),
                    "grid_position": list(prediction_grid.values()),
                }
            )

        gold.get_gp_features = build_with_grid

        with tempfile.TemporaryDirectory() as temp_dir:
            gold.data_dir = Path(temp_dir)
            result = gold.build_prediction_features(2026, 12, 5)

        self.assertEqual(result.set_index("driver_id")["grid_position"].to_dict(), resolved_positions)


if __name__ == "__main__":
    unittest.main()
