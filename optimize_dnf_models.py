import argparse
from pathlib import Path

import optuna

from src.dnf_evaluation import load_cached_gold
from src.dnf_optimization import optimize_dnf_configurations

# TODO: semplifiacre l'ottimizzazione solo sui parametri del loistic regressor
# magari eliminare direttamente questo file e mettere tutto in train_dnf_optimized.py


def main(
    data_dir: Path,
    output_dir: Path,
    holdout_races: int,
    calibration_races: int,
    min_training_races: int,
    team_beta_trials: int,
    heuristic_trials: int,
    logistic_trials: int,
    gradient_boosting_trials: int,
    bootstrap_samples: int,
) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    result = optimize_dnf_configurations(
        load_cached_gold(data_dir),
        output_dir=output_dir,
        min_training_races=min_training_races,
        holdout_races=holdout_races,
        calibration_races=calibration_races,
        trials={
            "team_beta_tuned": team_beta_trials,
            "heuristic_tuned": heuristic_trials,
            "logistic_legacy_tuned": logistic_trials,
            "logistic_benchmark_tuned": logistic_trials,
            "gradient_boosting_tuned": gradient_boosting_trials,
        },
        bootstrap_samples=bootstrap_samples,
    )
    print(result["metrics"].to_string(index=False))
    gate = result["gate"]
    print(
        f"\nGate ottimizzazione: {'GO' if gate['passed'] else 'NO-GO'} | "
        f"candidato={gate['candidate_strategy']} | "
        f"raccomandazione={gate['recommended_strategy']}"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ottimizzazione temporale delle configurazioni DNF")
    parser.add_argument("--data-dir", type=Path, default=Path("data_files/gold"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/dnf_optimization"))
    parser.add_argument("--holdout-races", type=int, default=12)
    parser.add_argument("--calibration-races", type=int, default=10)
    parser.add_argument("--min-training-races", type=int, default=5)
    parser.add_argument("--team-beta-trials", type=int, default=20)
    parser.add_argument("--heuristic-trials", type=int, default=30)
    parser.add_argument("--logistic-trials", type=int, default=40)
    parser.add_argument("--gradient-boosting-trials", type=int, default=60)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        holdout_races=args.holdout_races,
        calibration_races=args.calibration_races,
        min_training_races=args.min_training_races,
        team_beta_trials=args.team_beta_trials,
        heuristic_trials=args.heuristic_trials,
        logistic_trials=args.logistic_trials,
        gradient_boosting_trials=args.gradient_boosting_trials,
        bootstrap_samples=args.bootstrap_samples,
    )
