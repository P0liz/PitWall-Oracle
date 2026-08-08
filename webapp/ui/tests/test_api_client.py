"""Contract tests for the Streamlit API client."""

import unittest
from unittest.mock import Mock

import requests

from webapp.ui.api_client import (
    ApiDataError,
    ApiUnavailable,
    PitWallApiClient,
    PredictionUnavailable,
)


class PitWallApiClientTests(unittest.TestCase):
    def make_response(self, status_code: int, payload: dict) -> Mock:
        response = Mock()
        response.status_code = status_code
        response.json.return_value = payload
        return response

    def test_current_prediction_returns_the_json_document(self) -> None:
        document = {"schema_version": "1.0", "drivers": []}
        session = Mock()
        session.get.return_value = self.make_response(200, document)
        client = PitWallApiClient("https://api.example.test", session=session)

        self.assertEqual(client.get_current_prediction(), document)
        session.get.assert_called_once_with(
            "https://api.example.test/api/v1/predictions/current",
            timeout=8.0,
        )

    def test_trailing_base_url_slash_does_not_double_up_request_path(self) -> None:
        session = Mock()
        session.get.return_value = self.make_response(200, {"schema_version": "1.0"})
        client = PitWallApiClient("https://api.example.test/", session=session)

        client.get_current_prediction()

        session.get.assert_called_once_with(
            "https://api.example.test/api/v1/predictions/current",
            timeout=8.0,
        )

    def test_prediction_not_available_response_raises_specific_error(self) -> None:
        session = Mock()
        session.get.return_value = self.make_response(
            404,
            {"detail": {"code": "prediction_not_available", "message": "Nessuna previsione"}},
        )
        client = PitWallApiClient("https://api.example.test", session=session)

        with self.assertRaisesRegex(PredictionUnavailable, "Nessuna previsione"):
            client.get_current_prediction()

    def test_unprocessable_response_uses_server_message(self) -> None:
        session = Mock()
        session.get.return_value = self.make_response(
            422,
            {"detail": {"code": "invalid_published_data", "message": "Dati non validi"}},
        )
        client = PitWallApiClient("https://api.example.test", session=session)

        with self.assertRaisesRegex(ApiDataError, "Dati non validi"):
            client.get_current_prediction()

    def test_timeout_raises_api_unavailable(self) -> None:
        session = Mock()
        session.get.side_effect = requests.Timeout()
        client = PitWallApiClient("https://api.example.test", session=session)

        with self.assertRaisesRegex(ApiUnavailable, "temporaneamente non raggiungibile"):
            client.get_current_prediction()

    def test_connection_error_raises_api_unavailable(self) -> None:
        session = Mock()
        session.get.side_effect = requests.ConnectionError()
        client = PitWallApiClient("https://api.example.test", session=session)

        with self.assertRaisesRegex(ApiUnavailable, "temporaneamente non raggiungibile"):
            client.get_current_prediction()

    def test_other_public_methods_use_their_api_paths(self) -> None:
        session = Mock()
        session.get.side_effect = [
            self.make_response(200, {"driver_a_id": "nor", "driver_b_id": "ver"}),
            self.make_response(200, {"season": 2026, "races": []}),
            self.make_response(200, {"race": {"round": 1}}),
        ]
        client = PitWallApiClient("https://api.example.test", timeout_seconds=4.0, session=session)

        self.assertEqual(client.get_head_to_head("nor", "ver")["driver_a_id"], "nor")
        self.assertEqual(client.list_history(2026)["season"], 2026)
        self.assertEqual(client.get_history(2026, 1)["race"]["round"], 1)
        self.assertEqual(
            session.get.call_args_list[0].args,
            ("https://api.example.test/api/v1/predictions/current/head-to-head?driver_a=nor&driver_b=ver",),
        )
        self.assertEqual(
            session.get.call_args_list[1].args,
            ("https://api.example.test/api/v1/history?season=2026",),
        )
        self.assertEqual(
            session.get.call_args_list[2].args,
            ("https://api.example.test/api/v1/history/2026/1",),
        )
        self.assertTrue(all(call.kwargs == {"timeout": 4.0} for call in session.get.call_args_list))


if __name__ == "__main__":
    unittest.main()
