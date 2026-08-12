from pathlib import Path
from collections.abc import Callable, Sequence

import fastf1
import numpy as np
import pandas as pd

from src.ranking_metrics import mean_absolute_position_error, pairwise_accuracy

YEAR = 2026
RACE_NUMBER = None

RANKING_LABELS = {
    "base_position": "Base position",
    "expected_position": "Expected position",
    "expected_position_if_finished": "Expected position if finished",
}


def _load_and_align(summary_path: Path, silver_path: Path) -> pd.DataFrame:
    if not summary_path.is_file():
        raise FileNotFoundError(f"Risultati Monte Carlo non trovati: {summary_path}")
    if not silver_path.is_file():
        raise FileNotFoundError(f"Risultati Silver non trovati: {silver_path}")

    summary = pd.read_csv(summary_path)
    summary.columns = summary.columns.str.strip()
    silver = pd.read_parquet(silver_path)
    silver.columns = silver.columns.str.strip()

    prediction_columns = ["base_position", "expected_position", "expected_position_if_finished"]
    summary_required = {"driver_id", *prediction_columns}
    silver_required = {"driver_id", "team_id", "Position"}
    missing_summary = sorted(summary_required - set(summary.columns))
    missing_silver = sorted(silver_required - set(silver.columns))
    if missing_summary:
        raise ValueError(f"Colonne mancanti nel riepilogo Monte Carlo: {missing_summary}")
    if missing_silver:
        raise ValueError(f"Colonne mancanti nei risultati Silver: {missing_silver}")

    summary = summary[["driver_id", *prediction_columns]].copy()
    silver = silver[["driver_id", "team_id", "Position"]].copy()
    for frame in (summary, silver):
        frame["driver_id"] = frame["driver_id"].astype("string").str.strip()
    silver["team_id"] = silver["team_id"].astype("string").str.strip()

    if summary["driver_id"].duplicated().any():
        raise ValueError("Il riepilogo Monte Carlo contiene driver_id duplicati")
    if silver["driver_id"].duplicated().any():
        raise ValueError("I risultati Silver contengono driver_id duplicati")

    summary_drivers = set(summary["driver_id"])
    silver_drivers = set(silver["driver_id"])
    if summary_drivers != silver_drivers:
        only_summary = sorted(summary_drivers - silver_drivers)
        only_silver = sorted(silver_drivers - summary_drivers)
        raise ValueError(
            "Piloti non allineati tra Monte Carlo e Silver: "
            f"solo Monte Carlo={only_summary}, solo Silver={only_silver}"
        )

    aligned = silver.merge(summary, on="driver_id", how="inner", validate="one_to_one")
    numeric_columns = ["Position", *prediction_columns]
    for column in numeric_columns:
        aligned[column] = pd.to_numeric(aligned[column], errors="raise")
    if not np.isfinite(aligned[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Le posizioni reali e previste devono contenere solo valori finiti")
    if aligned["team_id"].isna().any() or aligned["team_id"].eq("").any():
        raise ValueError("team_id mancante nei risultati Silver")

    return aligned


def evaluate_race(summary_path: Path, silver_path: Path) -> pd.DataFrame:
    aligned = _load_and_align(Path(summary_path), Path(silver_path))
    true_scores = -aligned["Position"].to_numpy(dtype=float)
    teams = aligned["team_id"].to_numpy()
    rows = []

    for column in ["base_position", "expected_position", "expected_position_if_finished"]:
        predicted_scores = -aligned[column].to_numpy(dtype=float)
        rows.append(
            {
                "ranking": column,
                "drivers": len(aligned),
                "pairwise_accuracy": pairwise_accuracy(true_scores, predicted_scores),
                "teammate_pairwise_accuracy": pairwise_accuracy(true_scores, predicted_scores, groups=teams),
                "position_mae": mean_absolute_position_error(true_scores, predicted_scores),
            }
        )

    return pd.DataFrame(rows)


def _display_race_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    table = metrics.rename(
        columns={
            "ranking": "ordine previsto",
            "drivers": "piloti",
            "pairwise_accuracy": "pairwise accuracy",
            "teammate_pairwise_accuracy": "teammate pairwise accuracy",
            "position_mae": "position MAE",
        }
    ).copy()
    table["ordine previsto"] = table["ordine previsto"].map(RANKING_LABELS)
    return table


def _display_season_summary(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.rename(
        columns={
            "ranking": "ordine previsto",
            "races_evaluated": "gare valutate",
            "pairwise_accuracy": "avg pairwise accuracy",
            "teammate_pairwise_accuracy": "avg teammate pairwise accuracy",
            "position_mae": "avg position MAE",
        }
    ).copy()
    table["ordine previsto"] = table["ordine previsto"].map(RANKING_LABELS)
    return table


def _run_simulator(year: int, race_number: int) -> None:
    from monte_carlo_simulator import main as simulator_main

    simulator_main(year, race_number)


def evaluate_configured_race(
    year: int,
    race_number: int,
    *,
    results_dir: Path = Path("results"),
    silver_dir: Path = Path("data_files/silver"),
    simulator: Callable[[int, int], object] = _run_simulator,
) -> pd.DataFrame:
    summary_path = results_dir / f"summary_{year}_{race_number}.csv"
    if not summary_path.is_file():
        print(f"Summary GP #{race_number} assente: avvio simulazione Monte Carlo.")
        simulator(year, race_number)
    silver_path = silver_dir / f"{year}_{race_number}_5_clean_results.parquet"
    metrics = evaluate_race(summary_path, silver_path)
    table = _display_race_metrics(metrics)
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / f"evaluation_{year}_{race_number}.csv", float_format="%.4f", index=False)

    driver_count = int(metrics["drivers"].iloc[0])
    print(f"\nConfronto Monte Carlo — {year} GP #{race_number} ({driver_count} piloti)\n")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return metrics


def completed_race_numbers(year: int, cutoff: pd.Timestamp | None = None) -> list[int]:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    session_dates = pd.to_datetime(schedule["Session5DateUtc"], errors="coerce", utc=True)
    cutoff_utc = pd.Timestamp.now(tz="UTC") if cutoff is None else pd.Timestamp(cutoff)
    if cutoff_utc.tzinfo is None:
        cutoff_utc = cutoff_utc.tz_localize("UTC")
    else:
        cutoff_utc = cutoff_utc.tz_convert("UTC")
    rounds = pd.to_numeric(schedule.loc[session_dates <= cutoff_utc, "RoundNumber"], errors="coerce")
    return sorted(rounds.dropna().astype(int).loc[lambda values: values > 0].unique().tolist())


def evaluate_season(
    year: int,
    race_numbers: Sequence[int],
    *,
    results_dir: Path = Path("results"),
    silver_dir: Path = Path("data_files/silver"),
    simulator: Callable[[int, int], object] = _run_simulator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    evaluated = []
    failures = {}
    for race_number in race_numbers:
        try:
            metrics = evaluate_configured_race(
                year,
                race_number,
                results_dir=results_dir,
                silver_dir=silver_dir,
                simulator=simulator,
            )
            metrics.insert(0, "race_number", race_number)
            evaluated.append(metrics)
        except Exception as error:
            failures[race_number] = str(error)
            print(f"\n[ERRORE] GP #{race_number} non valutato: {error}")

    if not evaluated:
        raise RuntimeError(f"Nessuna gara {year} è stata valutata correttamente")

    details = pd.concat(evaluated, ignore_index=True)
    summary = (
        details.groupby("ranking", sort=False)
        .agg(
            races_evaluated=("race_number", "nunique"),
            pairwise_accuracy=("pairwise_accuracy", "mean"),
            teammate_pairwise_accuracy=("teammate_pairwise_accuracy", "mean"),
            position_mae=("position_mae", "mean"),
        )
        .reset_index()
    )
    summary_table = _display_season_summary(summary)
    summary_table.to_csv(results_dir / f"evaluation_{year}_summary.csv", float_format="%.4f", index=False)

    print(f"\nRiepilogo {year} — {summary['races_evaluated'].max()} gare valutate\n")
    print(summary_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if failures:
        failed_rounds = ", ".join(f"#{race}" for race in failures)
        print(f"\nGare non valutate: {failed_rounds}")

    return details, summary, failures


def main() -> None:
    if RACE_NUMBER is None:
        races = completed_race_numbers(YEAR)
        if not races:
            raise RuntimeError(f"Nessuna gara {YEAR} conclusa alla data corrente")
        print(f"Gare {YEAR} concluse secondo FastF1: {', '.join(f'#{race}' for race in races)}")
        evaluate_season(YEAR, races)
        return

    if not isinstance(RACE_NUMBER, int) or isinstance(RACE_NUMBER, bool) or RACE_NUMBER < 1:
        raise ValueError("RACE_NUMBER deve essere None oppure un intero positivo")
    evaluate_configured_race(YEAR, RACE_NUMBER)


if __name__ == "__main__":
    main()
