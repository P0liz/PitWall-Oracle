import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from xgboost import XGBRanker

from src.dnf.dnf_model import load_dnf_artifact
from src.model_lifecycle import CALIBRATION_NAME, publish_staged_models


def validate_ranker(path: Path) -> None:
    model = XGBRanker()
    model.load_model(path)


def validate_dnf(path: Path) -> None:
    load_dnf_artifact(path)


def run_training_pipeline(
    staging_dir: Path,
    live_model_dir: Path,
    year: int,
    *,
    repository_root: Path,
    command_runner: Callable = subprocess.run,
    ranker_validator: Callable[[Path], None] | None = None,
    dnf_validator: Callable[[Path], None] | None = None,
) -> list[str]:
    staging_dir = Path(staging_dir)
    live_model_dir = Path(live_model_dir)
    repository_root = Path(repository_root)
    staging_dir.mkdir(parents=True, exist_ok=True)

    current_calibration = live_model_dir / CALIBRATION_NAME
    if current_calibration.is_file():
        shutil.copy2(current_calibration, staging_dir / CALIBRATION_NAME)

    environment = os.environ.copy()
    environment["PITWALL_MODEL_DIR"] = str(staging_dir.resolve())
    environment["PITWALL_NON_INTERACTIVE"] = "1"
    commands = ([sys.executable, "-m", "src.train_ranker_optimized"], [sys.executable, "-m", "src.train_dnf_optimized"])
    for command in commands:
        command_runner(command, cwd=repository_root, env=environment, check=True)

    return publish_staged_models(
        staging_dir, live_model_dir, year, validate_ranker=ranker_validator, validate_dnf=dnf_validator
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run and atomically publish both post-race model trainings")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args(argv)

    repository_root = REPOSITORY_ROOT
    results_dir = repository_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model-staging-", dir=results_dir) as staging_text:
        published = run_training_pipeline(
            Path(staging_text),
            args.model_dir,
            args.year,
            repository_root=repository_root,
            ranker_validator=validate_ranker,
            dnf_validator=validate_dnf,
        )
    print("Published model artifacts:")
    for name in published:
        print(f"- {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
