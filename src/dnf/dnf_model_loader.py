import re
from pathlib import Path
from typing import Iterable

from ..config import NEW_YEAR

MODEL_DIR = Path("models")
BASE_MODEL_PATH = MODEL_DIR / "dnf_logistic_base.joblib"


def dnf_artifact_candidates(
    artifact_paths: Iterable[str | Path], year: int, race_number: int, *, allow_legacy_base: bool = False
) -> list[str]:
    """Return local DNF artifacts in causal preference order."""
    if race_number < 1:
        raise ValueError("race_number deve essere maggiore o uguale a 1")

    season_base: str | None = None
    legacy_base: str | None = None
    promoted_pattern = re.compile(rf"dnf_logistic_{year}_(\d+)\.joblib")
    promoted: list[tuple[int, str]] = []

    for artifact_path in artifact_paths:
        path = Path(artifact_path)
        artifact_text = str(artifact_path)
        if path.name == f"dnf_logistic_{year}_base.joblib":
            season_base = artifact_text
            continue
        if allow_legacy_base and path.name == BASE_MODEL_PATH.name:
            legacy_base = artifact_text
            continue

        match = promoted_pattern.fullmatch(path.name)
        if match and int(match.group(1)) < race_number:
            promoted.append((int(match.group(1)), artifact_text))

    candidates = [path for _, path in sorted(promoted, reverse=True)]
    if season_base is not None:
        candidates.append(season_base)
    elif legacy_base is not None:
        candidates.append(legacy_base)
    return candidates


def select_dnf_artifact_path(
    artifact_paths: Iterable[str | Path], year: int, race_number: int, *, allow_legacy_base: bool = False
) -> str | None:
    candidates = dnf_artifact_candidates(artifact_paths, year, race_number, allow_legacy_base=allow_legacy_base)
    return str(candidates[0]) if candidates else None


def resolve_dnf_model_path(year: int, race_number: int, local_model_dir: Path = MODEL_DIR) -> Path:
    """Resolve the newest local DNF model promoted strictly before the race."""
    local_model_dir = Path(local_model_dir)
    artifacts = list(local_model_dir.iterdir()) if local_model_dir.is_dir() else []
    selected = select_dnf_artifact_path(artifacts, year, race_number, allow_legacy_base=year == NEW_YEAR)
    if selected is None:
        raise FileNotFoundError(
            f"No causally eligible season-specific DNF model found for year={year}, "
            f"race_number={race_number} in '{local_model_dir}'"
        )
    return Path(selected)
