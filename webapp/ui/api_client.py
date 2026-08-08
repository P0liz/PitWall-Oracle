"""Small HTTP client for the published PitWall Oracle API."""

from urllib.parse import urlencode

import requests


class ApiUnavailable(Exception):
    """Raised when the PitWall API cannot be reached."""


class PredictionUnavailable(Exception):
    """Raised when no current prediction has been published."""


class ApiDataError(Exception):
    """Raised when the API reports a data or request error."""


class PitWallApiClient:
    """Retrieve prediction documents from the read-only PitWall API."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 8.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def _get(self, path: str) -> dict:
        try:
            response = self.session.get(f"{self.base_url}{path}", timeout=self.timeout_seconds)
        except (requests.Timeout, requests.ConnectionError) as error:
            raise ApiUnavailable("API temporaneamente non raggiungibile") from error

        if response.status_code == 404:
            detail = response.json().get("detail", {})
            if detail.get("code") == "prediction_not_available":
                raise PredictionUnavailable(detail.get("message", "Previsione non disponibile"))
        if response.status_code >= 400:
            detail = response.json().get("detail", {})
            raise ApiDataError(detail.get("message", f"Errore API {response.status_code}"))
        return response.json()

    def get_current_prediction(self) -> dict:
        return self._get("/api/v1/predictions/current")

    def get_head_to_head(self, driver_a: str, driver_b: str) -> dict:
        query = urlencode({"driver_a": driver_a, "driver_b": driver_b})
        return self._get(f"/api/v1/predictions/current/head-to-head?{query}")

    def list_history(self, season: int) -> dict:
        return self._get(f"/api/v1/history?{urlencode({'season': season})}")

    def get_history(self, season: int, round_number: int) -> dict:
        return self._get(f"/api/v1/history/{season}/{round_number}")
