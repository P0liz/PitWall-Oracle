import io
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urljoin
from src.utils import setup_custom_logger

import pandas as pd
import pdfplumber
import requests

from src.config import DATA_DIR

log = setup_custom_logger("DataLoader")
FIA_BASE_URL = "https://www.fia.com"
FIA_CHAMPIONSHIP_URL = f"{FIA_BASE_URL}/documents/championships/fia-formula-one-world-championship-14/event"


@dataclass(frozen=True)
class ResolvedGrid:
    positions: dict[str, float]
    source: str
    source_url: str | None = None


class _DocumentLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._href: str | None = None
        self._text: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag != "a" or self._href is not None:
            return
        attributes = dict(attrs)
        href = attributes.get("href")
        if href:
            self._href = href
            self._text = []

    def handle_data(self, data: str):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str):
        if tag == "a" and self._href is not None:
            text = " ".join(" ".join(self._text).split())
            self.links.append((self._href, text))
            self._href = None
            self._text = []


def find_provisional_grid_pdf(html: str, sprint: bool) -> str | None:
    parser = _DocumentLinkParser()
    parser.feed(html)
    expected_title = "Provisional Sprint Starting Grid" if sprint else "Provisional Starting Grid"
    title_pattern = re.compile(rf"^(?:Doc\s+\d+\s+-\s+)?{re.escape(expected_title)}(?:\s+Published on\b.*)?$")

    for href, text in parser.links:
        if title_pattern.fullmatch(text):
            return urljoin(FIA_BASE_URL, href)
    return None


def _integer_word(word: dict) -> int | None:
    text = str(word.get("text", "")).strip()
    return int(text) if text.isdigit() else None


def parse_grid_words(words: list[dict], page_width: float) -> dict[int, int]:
    penalty_tops = [float(word["top"]) for word in words if str(word.get("text", "")).upper() == "PENALTIES"]
    grid_bottom = min(penalty_tops) if penalty_tops else math.inf
    title_tops = [
        float(word["top"])
        for word in words
        if str(word.get("text", "")).lower() == "grid" and float(word["top"]) < grid_bottom
    ]
    grid_top = min(title_tops) + 15 if title_tops else 0

    candidates = [
        word for word in words if grid_top < float(word["top"]) < grid_bottom and float(word["x0"]) < page_width
    ]
    entries: dict[int, int] = {}
    for position_word in candidates:
        position = _integer_word(position_word)
        if position is None or not 1 <= position <= 30:
            continue

        same_half_limit = page_width / 2 if float(position_word["x0"]) < page_width / 2 else page_width
        car_words = [
            word
            for word in candidates
            if _integer_word(word) is not None
            and float(position_word["x1"]) + 5 <= float(word["x0"]) <= float(position_word["x1"]) + 30
            and abs(float(word["top"]) - float(position_word["top"])) <= 4
            and float(word["x0"]) < same_half_limit
        ]
        if not car_words:
            continue
        car_word = min(car_words, key=lambda word: float(word["x0"]))
        car_number = _integer_word(car_word)
        name_follows = any(
            str(word.get("text", "")).isalpha()
            and float(car_word["x1"]) < float(word["x0"]) <= float(car_word["x1"]) + 80
            and abs(float(word["top"]) - float(car_word["top"])) <= 2
            for word in candidates
        )
        if car_number is not None and name_follows:
            entries[position] = car_number
    return entries


def parse_grid_pdf(content: bytes) -> dict[int, int]:
    entries: dict[int, int] = {}
    with pdfplumber.open(io.BytesIO(content)) as document:
        for page in document.pages:
            entries.update(parse_grid_words(page.extract_words(), float(page.width)))
    return entries


class StartingGridResolver:
    def __init__(
        self,
        data_dir: Path | None = None,
        http_get: Callable = requests.get,
        pdf_parser: Callable[[bytes], dict[int, int]] = parse_grid_pdf,
    ):
        self.data_dir = data_dir or Path(DATA_DIR) / "starting_grids"
        self.http_get = http_get
        self.pdf_parser = pdf_parser

    def resolve(
        self,
        year: int,
        race_number: int,
        session: int,
        event_name: str,
        qualifying: pd.DataFrame,
        force: bool = False,
        cache: bool = True,
    ) -> ResolvedGrid:
        if session not in (3, 5):
            raise ValueError("Starting grids are supported only for Sprint and Race sessions")

        override_path = self.data_dir / f"{year}_{race_number}_{session}_override.json"
        if override_path.exists():
            payload = json.loads(override_path.read_text(encoding="utf-8"))
            positions = payload.get("grid", payload)
            return ResolvedGrid(self._validate_driver_grid(positions, qualifying), "manual_override")

        cache_path = self.data_dir / f"{year}_{race_number}_{session}.json"
        if cache and cache_path.exists() and not force:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("source") == "fia_provisional":
                positions = self._validate_driver_grid(payload["grid"], qualifying)
                return ResolvedGrid(positions, payload["source"], payload.get("source_url"))

        resolved = self._resolve_fia(event_name, session == 3, qualifying)
        if resolved is None:
            log.warning(
                f"FIA provisional starting grid unavailable for {year} round {race_number} session {session} "
                f"({event_name}); using qualifying order"
            )
            resolved = ResolvedGrid(self._qualifying_grid(qualifying), "qualifying_fallback")

        if cache:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "source": resolved.source,
                        "source_url": resolved.source_url,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "grid": resolved.positions,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return resolved

    def _resolve_fia(self, event_name: str, sprint: bool, qualifying: pd.DataFrame) -> ResolvedGrid | None:
        event_url = f"{FIA_CHAMPIONSHIP_URL}/{quote(event_name)}"
        try:
            response = self.http_get(event_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            pdf_url = find_provisional_grid_pdf(response.text, sprint=sprint)
            if pdf_url is None:
                return None
            pdf_response = self.http_get(pdf_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            pdf_response.raise_for_status()
            numbered_grid = self.pdf_parser(pdf_response.content)
            positions = self._map_car_numbers(numbered_grid, qualifying)
            return ResolvedGrid(positions, "fia_provisional", pdf_url)
        except (requests.RequestException, KeyError, TypeError, ValueError) as exception:
            log.error(type(exception).__name__)
            log.error(exception.response.status_code)
            return None

    @staticmethod
    def _map_car_numbers(numbered_grid: dict[int, int], qualifying: pd.DataFrame) -> dict[str, float]:
        expected_positions = set(range(1, len(qualifying) + 1))
        if set(numbered_grid) != expected_positions:
            raise ValueError("FIA grid positions are incomplete or duplicated")

        number_to_driver = {
            int(number): str(driver_id)
            for number, driver_id in qualifying.loc[:, ["DriverNumber", "driver_id"]].itertuples(index=False)
        }
        if set(numbered_grid.values()) != set(number_to_driver):
            raise ValueError("FIA grid drivers do not match qualifying")
        return {number_to_driver[car_number]: float(position) for position, car_number in numbered_grid.items()}

    @staticmethod
    def _validate_driver_grid(positions: dict, qualifying: pd.DataFrame) -> dict[str, float]:
        expected_drivers = set(qualifying["driver_id"].astype(str))
        if set(positions) != expected_drivers:
            raise ValueError("Grid drivers do not match qualifying")
        normalized = {str(driver_id): float(position) for driver_id, position in positions.items()}
        if any(not math.isfinite(position) or position < 0 for position in normalized.values()):
            raise ValueError("Grid positions must be finite and non-negative")
        positive_positions = [position for position in normalized.values() if position > 0]
        if len(positive_positions) != len(set(positive_positions)):
            raise ValueError("Positive grid positions must be unique")
        return normalized

    @staticmethod
    def _qualifying_grid(qualifying: pd.DataFrame) -> dict[str, float]:
        ordered = qualifying.sort_values("Position", kind="stable", na_position="last")
        return {str(driver_id): float(position) for position, driver_id in enumerate(ordered["driver_id"], start=1)}
