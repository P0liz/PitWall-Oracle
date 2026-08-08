"""Tests for pure current-race page view-model helpers."""

import copy
import unittest

from webapp.ui.current_prediction import (
    fetch_head_to_head,
    head_to_head_options,
    selected_head_to_head_pair,
)


class CurrentPredictionViewModelTests(unittest.TestCase):
    def test_head_to_head_options_map_names_to_stable_driver_ids_without_mutating_document(self) -> None:
        document = {
            "drivers": [
                {"display_name": "Lando Norris", "driver_id": "nor"},
                {"display_name": "Max Verstappen", "driver_id": "ver"},
            ]
        }
        original_document = copy.deepcopy(document)

        options = head_to_head_options(document)

        self.assertEqual(options, {"Lando Norris": "nor", "Max Verstappen": "ver"})
        self.assertEqual(document, original_document)

    def test_fetch_head_to_head_rejects_the_same_driver_before_calling_api(self) -> None:
        client = RecordingClient()

        result = fetch_head_to_head(client, "nor", "nor")

        self.assertIsNone(result)
        self.assertEqual(client.calls, [])

    def test_selected_head_to_head_pair_keeps_distinct_stable_driver_ids(self) -> None:
        self.assertEqual(selected_head_to_head_pair("nor", "ver"), ("nor", "ver"))

    def test_fetch_head_to_head_uses_distinct_stable_driver_ids(self) -> None:
        client = RecordingClient()

        result = fetch_head_to_head(client, "nor", "ver")

        self.assertEqual(result, {"driver_a_id": "nor", "driver_b_id": "ver"})
        self.assertEqual(client.calls, [("nor", "ver")])


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_head_to_head(self, driver_a: str, driver_b: str) -> dict:
        self.calls.append((driver_a, driver_b))
        return {"driver_a_id": driver_a, "driver_b_id": driver_b}


if __name__ == "__main__":
    unittest.main()
