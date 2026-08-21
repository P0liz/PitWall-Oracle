import argparse
import json
import os
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath


class WorkflowPolicyError(RuntimeError):
    pass


def _normalized_path(raw_path: object) -> str:
    if not isinstance(raw_path, str) or not raw_path or raw_path != raw_path.strip():
        raise WorkflowPolicyError(f"Invalid changed path: {raw_path!r}")
    path_text = raw_path.replace("\\", "/")
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:/", path_text):
        raise WorkflowPolicyError(f"Unsafe changed path: '{raw_path}'")
    normalized = str(path)
    if normalized in ("", ".") or "\x00" in normalized:
        raise WorkflowPolicyError(f"Invalid changed path: {raw_path!r}")
    return normalized


def _published_operations(summary: Mapping) -> tuple[Mapping, ...]:
    operations = summary.get("operations", ())
    if not isinstance(operations, list):
        raise WorkflowPolicyError("Publication summary operations must be a list")
    published = []
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise WorkflowPolicyError("Publication summary contains an invalid operation")
        if operation.get("status") == "published":
            published.append(operation)
    return tuple(published)


def published_current_gp(summary: Mapping, current_year: int, enabled: bool) -> bool:
    if not enabled:
        return False
    return any(
        operation.get("operation") == "publish-actual"
        and operation.get("session_type") == "race"
        and operation.get("season") == current_year
        for operation in _published_operations(summary)
    )


def _expected_public_paths(mode: str, summary: Mapping) -> set[str]:
    expected_operation = "publish-prediction" if mode == "prediction" else "publish-actual"
    expected_root = "predictions/" if mode == "prediction" else "history/"
    paths: set[str] = set()
    for operation in _published_operations(summary):
        if operation.get("operation") != expected_operation:
            raise WorkflowPolicyError(
                f"Summary operation '{operation.get('operation')}' is not allowed in {mode} workflow"
            )
        changed_paths = operation.get("changed_paths", ())
        if not isinstance(changed_paths, (list, tuple)):
            raise WorkflowPolicyError("Publication changed_paths must be a list")
        for raw_path in changed_paths:
            relative_path = _normalized_path(raw_path)
            if not relative_path.startswith(expected_root):
                raise WorkflowPolicyError(f"Unexpected public path for {mode}: '{relative_path}'")
            paths.add(f"webapp/api/data/{relative_path}")
    return paths


def _managed_model_path(path: str, year: int) -> bool:
    year_text = re.escape(str(year))
    patterns = (
        rf"models/pitwall_oracle_(?:{year_text}_(?:base|\d+)|base|latest)\.json",
        rf"models/dnf_logistic_(?:{year_text}_(?:base|\d+)|base|latest)\.joblib",
    )
    return path == "models/monte_carlo_calibration.json" or any(re.fullmatch(pattern, path) for pattern in patterns)


def validate_changed_paths(
    mode: str, git_paths: Iterable[str], summary: Mapping, year: int, training_ran: bool
) -> tuple[str, ...]:
    if mode not in ("prediction", "post-race"):
        raise WorkflowPolicyError(f"Unsupported workflow mode: '{mode}'")
    expected_public = _expected_public_paths(mode, summary)
    changed = {_normalized_path(path) for path in git_paths}
    # missing = sorted(expected_public - changed)
    # if missing:
    # raise WorkflowPolicyError(f"Summary paths are missing from Git changes: {missing}")

    for path in sorted(changed):
        if path in expected_public:
            continue
        if path.startswith("data_files/") and not path.startswith("data_files/mlflow_artifacts/"):
            continue
        if mode == "post-race" and training_ran and _managed_model_path(path, year):
            continue
        raise WorkflowPolicyError(f"Unexpected changed path: '{path}'")
    return tuple(sorted(changed))


def collect_git_paths(repository_root: Path) -> tuple[str, ...]:
    commands = (
        ("git", "diff", "--name-only", "-z", "--no-renames", "HEAD", "--"),
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
    )
    paths: set[str] = set()
    for command in commands:
        result = subprocess.run(command, cwd=repository_root, check=True, stdout=subprocess.PIPE)
        paths.update(os.fsdecode(item) for item in result.stdout.split(b"\0") if item)
    return tuple(sorted(paths))


def _read_summary(path: Path) -> Mapping:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowPolicyError(f"Cannot read publication summary '{path}'") from error
    if not isinstance(summary, Mapping):
        raise WorkflowPolicyError("Publication summary must be a JSON object")
    return summary


def _boolean(value: str) -> bool:
    return value.lower() == "true"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate GitHub Actions publication changes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    training_parser = subparsers.add_parser("training-required")
    training_parser.add_argument("--summary-path", type=Path, required=True)
    training_parser.add_argument("--current-year", type=int, required=True)
    training_parser.add_argument("--enabled", choices=("true", "false"), default="true")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--mode", choices=("prediction", "post-race"), required=True)
    validate_parser.add_argument("--summary-path", type=Path, required=True)
    validate_parser.add_argument("--year", type=int, required=True)
    validate_parser.add_argument("--training-ran", choices=("true", "false"), default="false")
    validate_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    validate_parser.add_argument("--output-path", type=Path, default=Path("results/commit-paths.txt"))
    validate_parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)

    summary = _read_summary(args.summary_path)
    if args.command == "training-required":
        required = published_current_gp(summary, args.current_year, _boolean(args.enabled))
        print(str(required).lower())
        return 0

    approved = validate_changed_paths(
        args.mode, collect_git_paths(args.repository_root), summary, args.year, _boolean(args.training_ran)
    )
    has_published_operation = bool(_published_operations(summary))
    commit_paths = approved if has_published_operation else ()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text("".join(f"{path}\n" for path in commit_paths), encoding="utf-8")
    has_changes = bool(commit_paths)
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"has_changes={str(has_changes).lower()}\n")
    print(f"Validated {len(approved)} changed path(s); commit paths: {len(commit_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
