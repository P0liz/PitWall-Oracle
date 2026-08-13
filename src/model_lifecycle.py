import json
import math
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path


class ModelLifecycleError(RuntimeError):
    pass


RANKER_PATTERN = re.compile(r"pitwall_oracle_(\d{4})_(\d+)\.json")
DNF_PATTERN = re.compile(r"dnf_logistic_(\d{4})_(\d+)\.joblib")
CALIBRATION_NAME = "monte_carlo_calibration.json"


def _validate_sigma(value: object, label: str) -> float:
    try:
        sigma = float(value)
    except (TypeError, ValueError) as error:
        raise ModelLifecycleError(f"Invalid sigma_relative for {label}") from error
    if not math.isfinite(sigma) or sigma <= 0:
        raise ModelLifecycleError(f"Invalid sigma_relative for {label}")
    return sigma


def read_calibration(path: Path) -> dict[int, dict[int, float]]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelLifecycleError(f"Cannot read Monte Carlo calibration '{path}'") from error

    if payload.get("schema_version") != 1 or not isinstance(payload.get("seasons"), dict):
        raise ModelLifecycleError("Invalid Monte Carlo calibration schema")

    parsed: dict[int, dict[int, float]] = {}
    for raw_year, records in payload["seasons"].items():
        try:
            year = int(raw_year)
        except (TypeError, ValueError) as error:
            raise ModelLifecycleError(f"Invalid calibration season '{raw_year}'") from error
        if not isinstance(records, list):
            raise ModelLifecycleError(f"Calibration records for {year} must be a list")

        season: dict[int, float] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ModelLifecycleError(f"Invalid calibration record for {year}")
            round_value = record.get("selected_after_round")
            if not isinstance(round_value, int) or isinstance(round_value, bool) or round_value < 0:
                raise ModelLifecycleError(f"Invalid calibration round for {year}")
            if round_value in season:
                raise ModelLifecycleError(f"Duplicate calibration round {round_value} for {year}")
            season[round_value] = _validate_sigma(record.get("sigma_relative"), f"{year} round {round_value}")
        if not season or 0 not in season:
            raise ModelLifecycleError(f"Calibration season {year} requires a round-zero baseline")
        parsed[year] = season
    return parsed


def write_calibration(path: Path, year: int, values: Mapping[int, float]) -> Path:
    path = Path(path)
    seasons: dict[int, dict[int, float]] = {}
    if path.exists():
        seasons.update(read_calibration(path))

    normalized: dict[int, float] = {}
    for round_value, sigma in values.items():
        if not isinstance(round_value, int) or isinstance(round_value, bool) or round_value < 0:
            raise ModelLifecycleError(f"Invalid calibration round for {year}")
        normalized[round_value] = _validate_sigma(sigma, f"{year} round {round_value}")
    if 0 not in normalized:
        raise ModelLifecycleError(f"Calibration season {year} requires a round-zero baseline")
    seasons[int(year)] = normalized

    payload = {
        "schema_version": 1,
        "seasons": {
            str(season_year): [
                {"selected_after_round": round_value, "sigma_relative": sigma}
                for round_value, sigma in sorted(records.items())
            ]
            for season_year, records in sorted(seasons.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_suffix(path.suffix + ".tmp")
    candidate.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    read_calibration(candidate)
    candidate.replace(path)
    return path


def resolve_calibration(path: Path, year: int, race_number: int) -> float:
    if race_number < 1:
        raise ValueError("race_number deve essere maggiore o uguale a 1")
    records = read_calibration(path).get(year)
    if records is None:
        raise ModelLifecycleError(f"No Monte Carlo calibration for season {year}")
    eligible = [round_value for round_value in records if round_value < race_number]
    if not eligible:
        raise ModelLifecycleError(f"No causal Monte Carlo calibration for {year} round {race_number}")
    return records[max(eligible)]


def _candidate_names(staging_dir: Path, year: int) -> tuple[set[str], set[str]]:
    ranker_base = f"pitwall_oracle_{year}_base.json"
    dnf_base = f"dnf_logistic_{year}_base.joblib"
    required = {ranker_base, "pitwall_oracle_latest.json", dnf_base, "dnf_logistic_latest.joblib", CALIBRATION_NAME}
    missing = sorted(name for name in required if not (staging_dir / name).is_file())
    if missing:
        raise ModelLifecycleError(f"Incomplete staged model set; missing: {missing}")

    publishable = set(required)
    transient: set[str] = set()
    for path in staging_dir.iterdir():
        name = path.name
        ranker_match = RANKER_PATTERN.fullmatch(name)
        dnf_match = DNF_PATTERN.fullmatch(name)
        if ranker_match and int(ranker_match.group(1)) == year:
            publishable.add(name)
        elif dnf_match and int(dnf_match.group(1)) == year:
            publishable.add(name)
        elif name in {"pitwall_oracle_pending.json", "dnf_logistic_pending.joblib"}:
            transient.add(name)
    return publishable, transient


def _managed_live_names(live_dir: Path, year: int) -> set[str]:
    managed = {
        f"pitwall_oracle_{year}_base.json",
        f"dnf_logistic_{year}_base.joblib",
        "pitwall_oracle_base.json",
        "dnf_logistic_base.joblib",
        "pitwall_oracle_latest.json",
        "dnf_logistic_latest.joblib",
        "pitwall_oracle_pending.json",
        "dnf_logistic_pending.joblib",
        CALIBRATION_NAME,
    }
    for path in live_dir.iterdir():
        for pattern in (RANKER_PATTERN, DNF_PATTERN):
            match = pattern.fullmatch(path.name)
            if match and int(match.group(1)) == year:
                managed.add(path.name)
    return managed


def publish_staged_models(
    staging_dir: Path,
    live_dir: Path,
    year: int,
    *,
    validate_ranker: Callable[[Path], None] | None = None,
    validate_dnf: Callable[[Path], None] | None = None,
) -> list[str]:
    staging_dir = Path(staging_dir)
    live_dir = Path(live_dir)
    if not staging_dir.is_dir():
        raise ModelLifecycleError(f"Staging directory does not exist: '{staging_dir}'")

    publishable, _ = _candidate_names(staging_dir, year)
    calibration = read_calibration(staging_dir / CALIBRATION_NAME)
    if year not in calibration:
        raise ModelLifecycleError(f"Staged calibration does not contain season {year}")

    for name in sorted(publishable):
        path = staging_dir / name
        if path.stat().st_size == 0:
            raise ModelLifecycleError(f"Empty staged artifact: '{name}'")
        if validate_ranker is not None and name.endswith(".json") and name != CALIBRATION_NAME:
            validate_ranker(path)
        if validate_dnf is not None and name.endswith(".joblib"):
            validate_dnf(path)

    live_dir.mkdir(parents=True, exist_ok=True)
    managed = _managed_live_names(live_dir, year)
    with tempfile.TemporaryDirectory(prefix="pitwall-model-backup-", dir=live_dir.parent) as backup_text:
        backup_dir = Path(backup_text)
        existing = {name for name in managed if (live_dir / name).is_file()}
        for name in existing:
            shutil.copy2(live_dir / name, backup_dir / name)

        try:
            for name in managed:
                (live_dir / name).unlink(missing_ok=True)
            for name in sorted(publishable):
                shutil.copy2(staging_dir / name, live_dir / name)
        except Exception as error:
            for name in managed | publishable:
                (live_dir / name).unlink(missing_ok=True)
            for name in existing:
                shutil.copy2(backup_dir / name, live_dir / name)
            raise ModelLifecycleError("Failed to publish staged models; live set restored") from error

    return sorted(publishable)
