import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from xgboost import XGBRanker

from src.data.data_loader import DataLoader
from src.data.gold_layer import GoldLayer
from src.trainer import select_model_feature_frame
from src.dnf_model import (
    DNF_STRATEGIES,
    DEFAULT_DNF_STRATEGY,
    compute_heuristic_dnf_probabilities,
    compute_strategy_probabilities,
    ensure_dnf_target,
    load_dnf_artifact,
)
from src.utils import setup_custom_logger

log = setup_custom_logger("MonteCarloSimulator")

# TODO: reworkare così che sia automatico
PREDICTION_MODE = False

MODEL_DIR = Path("models")
BASE_MODEL_PATH = MODEL_DIR / "pitwall_oracle_base.json"
MLFLOW_CACHE_DIR = Path("data_files") / "mlflow_artifacts"
RANKING_EXPERIMENT = "PitWall_Oracle_Ranking"
DNF_EXPERIMENT = "PitWall_Oracle_DNF"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # stesso backend store usato da ml_flow_auto.py

N_SIMULATIONS = 10000
ORDER_BY = "expected_position_if_finished"  # or "expected_position" "expected_position_if_finished"

# Sigma RELATIVO (adimensionale): frazione dello spread di score DELLA GARA
# CORRENTE che rappresenta rumore/incertezza del modello. Va moltiplicato per
# scores.std() della gara specifica per ottenere il sigma assoluto (vedi main()).
# Placeholder arbitrario, usato SOLO come fallback se MLflow non ha nessuna run
# con la metrica 'final_sigma_relativo_empirico'.
FALLBACK_SIGMA_RELATIVE = 0.7


def _runs_for_year(client: mlflow.tracking.MlflowClient, experiment_id: str, year: int):
    """Restituisce le run della stagione richiesta, dalla più recente."""
    runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=100)
    expected_run_name = f"F1_Season_Simulation_{year}"
    return [
        run
        for run in runs
        if run.data.params.get("simulation_year") == str(year)
        or run.data.tags.get("mlflow.runName") == expected_run_name
    ]


def fetch_relative_sigma(year: int, race_number: int) -> float:
    """
    Recupera il sigma disponibile prima della gara richiesta.

    Per la gara N usa cumulative_sigma_relativo allo step N-1. Per la prima
    gara usa il sigma finale della stagione precedente. Non usa mai il sigma
    finale della stagione corrente per una gara storica.
    """
    if race_number < 1:
        raise ValueError("race_number deve essere maggiore o uguale a 1")

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(RANKING_EXPERIMENT)
        if experiment is None:
            raise ValueError(f"Esperimento '{RANKING_EXPERIMENT}' non trovato")

        if race_number > 1:
            target_step = race_number - 1
            for run in _runs_for_year(client, experiment.experiment_id, year):
                history = client.get_metric_history(run.info.run_id, "cumulative_sigma_relativo")
                matching_values = [metric for metric in history if metric.step == target_step]
                if matching_values:
                    metric = max(matching_values, key=lambda item: item.timestamp)
                    sigma_rel = float(metric.value)
                    log.info(
                        f"Sigma cumulativo MLflow per {year} gara {race_number}: "
                        f"step {target_step}, run {run.info.run_id}, valore={sigma_rel:.4f}"
                    )
                    return sigma_rel

            raise ValueError(f"Nessun 'cumulative_sigma_relativo' trovato per {year} allo step {target_step}")

        previous_year = year - 1
        for run in _runs_for_year(client, experiment.experiment_id, previous_year):
            if "final_sigma_relativo_empirico" in run.data.metrics:
                sigma_rel = float(run.data.metrics["final_sigma_relativo_empirico"])
                log.info(
                    f"Sigma finale {previous_year} usato per la prima gara {year} "
                    f"(run {run.info.run_id}): {sigma_rel:.4f}"
                )
                return sigma_rel

        raise ValueError(f"Nessun sigma finale trovato per la stagione precedente ({previous_year})")

    except Exception as e:
        log.warning(
            f"Impossibile recuperare un sigma temporalmente valido per {year} gara "
            f"{race_number} da MLflow ({e}). Uso fallback fisso = {FALLBACK_SIGMA_RELATIVE}."
        )
        return FALLBACK_SIGMA_RELATIVE


def model_path_for_race(year: int, race_number: int) -> Path:
    """Seleziona il modello addestrato con dati fino alla gara precedente."""
    if race_number < 1:
        raise ValueError("race_number deve essere maggiore o uguale a 1")
    if not PREDICTION_MODE:
        local_path = BASE_MODEL_PATH if race_number == 1 else MODEL_DIR / f"pitwall_oracle_{year}_{race_number}.json"
    else:
        local_path = MODEL_DIR / "pitwall_oracle_latest.json"
    print(f"Loading ranker {local_path.name}")
    return resolve_mlflow_artifact(
        experiment_name=RANKING_EXPERIMENT,
        year=year,
        artifact_path=f"models/{local_path.name}",
        local_fallback=local_path,
    )


def resolve_mlflow_artifact(experiment_name: str, year: int, artifact_path: str, local_fallback: Path) -> Path:
    """Scarica l'artefatto dalla run stagionale; usa il locale solo come fallback."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Esperimento '{experiment_name}' non trovato")

        MLFLOW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for run in _runs_for_year(client, experiment.experiment_id, year):
            try:
                downloaded = client.download_artifacts(run.info.run_id, artifact_path, dst_path=str(MLFLOW_CACHE_DIR))
                path = Path(downloaded)
                log.info(
                    f"Artefatto MLflow recuperato da '{experiment_name}', " f"run {run.info.run_id}: {artifact_path}"
                )
                return path
            except Exception:
                continue
    except Exception as error:
        log.warning(f"Recupero MLflow fallito in '{experiment_name}' per '{artifact_path}': {error}")

    return local_fallback


def compute_dnf_probabilities(
    race_df: pd.DataFrame,
    year: int,
    race_number: int,
    strategy: str = "logistic",
    history_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Calcola probabilità DNF pre-gara con una strategia esplicita."""
    if strategy not in DNF_STRATEGIES:
        raise ValueError(f"Strategia DNF non valida: '{strategy}'")

    race_date = pd.Timestamp(race_df["race_date"].iloc[0])
    causal_history = None
    if history_df is not None:
        causal_history = history_df.loc[pd.to_datetime(history_df["race_date"]) < race_date]
        causal_history = ensure_dnf_target(causal_history)

    model_strategies = {"logistic", "gradient_boosting"}
    if strategy not in model_strategies:
        probabilities = compute_strategy_probabilities(strategy, prediction_df=race_df, history_df=causal_history)
        log.info(f"Strategia DNF '{strategy}' applicata per {year} gara {race_number}")
        return probabilities

    artifact_prefix = "dnf_logistic" if strategy == "logistic" else "dnf_gradient_boosting"
    if not PREDICTION_MODE:
        artifact_name = (
            f"{artifact_prefix}_base.joblib" if race_number == 1 else f"{artifact_prefix}_{year}_{race_number}.joblib"
        )
    else:
        artifact_name = f"{artifact_prefix}_latest.joblib"
    print(f"Loading DNF model {artifact_name}")
    artifact_path = resolve_mlflow_artifact(
        experiment_name=DNF_EXPERIMENT,
        year=year,
        artifact_path=f"models/{artifact_name}",
        local_fallback=MODEL_DIR / artifact_name,
    )
    if not artifact_path.exists():
        log.warning(f"Modello DNF pre-gara non trovato in '{artifact_path}'. Uso fallback euristico esplicito.")
        return compute_heuristic_dnf_probabilities(race_df)

    dnf_artifact = load_dnf_artifact(artifact_path)
    expected_model_type = dnf_artifact.get("model_type", "logistic")
    if expected_model_type != strategy:
        raise ValueError(f"Tipo modello DNF non compatibile: artifact={expected_model_type}, richiesto={strategy}")

    cutoff_date = pd.Timestamp(dnf_artifact["cutoff_date"])
    if cutoff_date >= race_date:
        raise ValueError(
            f"Leakage temporale nel modello DNF '{artifact_path}': cutoff {cutoff_date} "
            f"non precedente alla gara {race_date}"
        )

    probabilities = compute_strategy_probabilities(
        strategy, prediction_df=race_df, history_df=causal_history, artifact=dnf_artifact
    )
    log.info(f"Modello DNF caricato per {year} gara {race_number}: {artifact_path} " f"(cutoff={cutoff_date})")
    return probabilities


def run_monte_carlo(
    scores: np.ndarray,
    dnf_probabilities: np.ndarray,
    sigma_absolute: float,
    n_simulations: int = N_SIMULATIONS,
    seed: int = 42,
):
    """
    Per ogni simulazione:
      1. Aggiunge rumore gaussiano N(0, sigma_absolute) agli score del ranker.
         ATTENZIONE: sigma_absolute deve già essere nella stessa scala/unità
         degli 'scores' passati qui — è responsabilità del chiamante (main())
         riscalare il sigma relativo sullo spread reale della gara corrente.
      2. Estrae DNF indipendenti per ogni pilota secondo la propria probabilità.
      3. I piloti ritirati vengono spinti in fondo alla classifica simulata.
      4. Ordina gli score residui per ottenere la posizione simulata di ognuno.

    I piloti ritirati restano nel gruppo con uno score inferiore a ogni finisher.
    Il loro ordine viene randomizzato per approssimare un diverso momento del
    ritiro ed evitare che i pareggi siano risolti dall'ordine delle colonne.
    """
    rng = np.random.default_rng(seed)
    n_drivers = len(scores)

    noisy_scores = scores[None, :] + rng.normal(0.0, sigma_absolute, size=(n_simulations, n_drivers))
    dnf_draws = rng.random(size=(n_simulations, n_drivers)) < dnf_probabilities[None, :]

    penalty = noisy_scores.min(axis=1, keepdims=True) - 100.0
    randomized_dnf_scores = penalty + rng.random(size=(n_simulations, n_drivers))
    simulated_scores = np.where(dnf_draws, randomized_dnf_scores, noisy_scores)

    # Rank decrescente per riga (trick argsort-of-argsort): 1 = primo classificato
    simulated_positions = np.argsort(np.argsort(-simulated_scores, axis=1), axis=1) + 1

    return simulated_positions, dnf_draws


def summarize_results(
    driver_ids: np.ndarray, simulated_positions: np.ndarray, dnf_draws: np.ndarray, base_scores: np.ndarray
) -> pd.DataFrame:
    base_positions = np.argsort(np.argsort(-base_scores)) + 1
    summary = pd.DataFrame({"driver_id": driver_ids, "base_score": base_scores, "base_position": base_positions})
    finished = ~dnf_draws
    summary["win_probability"] = ((simulated_positions == 1) & finished).mean(axis=0)
    summary["podium_probability"] = ((simulated_positions <= 3) & finished).mean(axis=0)
    summary["points_probability"] = ((simulated_positions <= 10) & finished).mean(axis=0)
    summary["dnf_probability"] = dnf_draws.mean(axis=0)
    summary["finish_probability"] = finished.mean(axis=0)
    summary["expected_position"] = simulated_positions.mean(axis=0)
    summary["expected_position_if_finished"] = [
        float(simulated_positions[finished[:, idx], idx].mean()) if finished[:, idx].any() else float("nan")
        for idx in range(simulated_positions.shape[1])
    ]
    return summary.sort_values(ORDER_BY, ascending=True).reset_index(drop=True)


def head_to_head(driver_a: str, driver_b: str, driver_ids: np.ndarray, simulated_positions: np.ndarray) -> dict:
    """
    Confronto diretto A vs B sulle STESSE simulazioni (stesso scenario di gara
    condiviso, stesso rumore in ogni run). E' l'unico modo statisticamente
    corretto per rispondere a "chi arriva davanti tra i due": confrontare le
    medie marginali (es. expected_position) ignora la correlazione tra i due
    piloti dentro ogni singola simulazione e puo' essere fuorviante quando le
    distribuzioni non sono simmetriche.
    """
    idx_a = np.where(driver_ids == driver_a)[0]
    idx_b = np.where(driver_ids == driver_b)[0]
    if len(idx_a) == 0 or len(idx_b) == 0:
        raise ValueError(
            f"Driver non trovato in questa gara: verifica '{driver_a}' e '{driver_b}' "
            f"contro i driver_id disponibili: {sorted(driver_ids.tolist())}"
        )
    idx_a, idx_b = idx_a[0], idx_b[0]

    pos_a = simulated_positions[:, idx_a]
    pos_b = simulated_positions[:, idx_b]

    return {
        driver_a: float((pos_a < pos_b).mean()),
        driver_b: float((pos_b < pos_a).mean()),
        "tie": float((pos_a == pos_b).mean()),  # atteso ~0: rumore continuo, pareggi rarissimi
    }


def main(
    year: int,
    race_number: int,
    force: bool = False,
    dnf_strategy: str = DEFAULT_DNF_STRATEGY,
    n_simulations: int = N_SIMULATIONS,
    seed: int = 42,
):
    sigma_relative = fetch_relative_sigma(year, race_number)
    model_path = model_path_for_race(year, race_number)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modello temporalmente corretto non trovato in '{model_path}'. "
            f"Per la gara {race_number} serve un modello addestrato al massimo "
            f"fino alla gara {race_number - 1}."
        )
    log.info(f"Caricamento modello per {year} gara {race_number}: {model_path}")
    champion = XGBRanker()
    champion.load_model(model_path)

    gold = GoldLayer()
    data_loader = DataLoader()

    race_df = gold.build_prediction_features(year, race_number, 5, force=True)
    # race_df = race_results[-1].copy()  # solo gara principale, come in predict.py

    race_df["driver_id_raw"] = race_df["driver_id"].copy()  # salva prima del target encoding
    race_df["team_id_raw"] = race_df["team_id"].copy()
    cutoff_date = race_df["race_date"].iloc[0]

    dnf_probabilities = compute_dnf_probabilities(
        race_df, year, race_number, strategy=dnf_strategy, history_df=data_loader.history_df
    )

    for col in ["driver_id", "team_id"]:
        race_df[col] = data_loader.apply_target_encoding(race_df, col, cutoff_date=cutoff_date)
    race_df["circuit_id"] = race_df["circuit_id"].astype(data_loader.circuit_dtype)

    X = select_model_feature_frame(champion, race_df)

    scores = champion.predict(X)

    # Sigma ASSOLUTO per QUESTA gara: riscaliamo il sigma relativo (adimensionale,
    # stimato sul test set fisso 2025) sullo spread REALE degli score della gara
    # che stiamo simulando ora. E' il fix al bug diagnosticato: un sigma assoluto
    # fisso confrontava scale diverse (score del ranker vs target), gonfiandolo
    # di circa 6x e appiattendo tutte le probabilita' verso l'uniforme.
    score_spread = scores.std()
    if score_spread < 1e-6:
        log.warning("Spread degli score quasi nullo in questa gara: uso fallback assoluto conservativo.")
        sigma_absolute = FALLBACK_SIGMA_RELATIVE
    else:
        sigma_absolute = sigma_relative * score_spread

    log.info(
        f"Score spread gara corrente: {score_spread:.4f} | sigma relativo: {sigma_relative:.4f} "
        f"| sigma assoluto applicato: {sigma_absolute:.4f}"
    )

    simulated_positions, dnf_draws = run_monte_carlo(
        scores, dnf_probabilities, sigma_absolute, n_simulations=n_simulations, seed=seed
    )
    driver_ids_raw = race_df["driver_id_raw"].to_numpy()
    summary = summarize_results(driver_ids_raw, simulated_positions, dnf_draws, scores)
    summary["dnf_strategy"] = dnf_strategy

    summary_path = Path(f"results/summary_{year}_{race_number}.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    pd.set_option("display.float_format", "{:.1%}".format)
    print(
        f"\n=== Simulazione Monte Carlo — {year} GP #{race_number} "
        f"({n_simulations} run, DNF={dnf_strategy}, sigma assoluto={sigma_absolute:.4f}) ===\n"
    )
    print(summary.to_string(index=False))
    print(f"\nSummary salvato in: {summary_path.resolve()}")

    podium_sum = summary["podium_probability"].sum()
    print(f"\n[Sanity check] Somma probabilità di podio dei finisher: {podium_sum:.3f} ")

    return summary, simulated_positions, driver_ids_raw, race_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo simulazione Monte Carlo — PitWall Oracle")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--race", type=int, default=11, help="Numero di gara da simulare")
    parser.add_argument("--force", action="store_true", help="Forza il refresh delle feature Gold")
    parser.add_argument("--dnf-strategy", choices=DNF_STRATEGIES, default=DEFAULT_DNF_STRATEGY)
    parser.add_argument("--n-simulations", type=int, default=N_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--compare-a", type=str, default=None, help="driver_id per il confronto testa a testa")
    parser.add_argument("--compare-b", type=str, default=None, help="driver_id per il confronto testa a testa")
    args = parser.parse_args()

    summary, simulated_positions, driver_ids, race_df = main(
        args.year,
        args.race,
        force=args.force,
        dnf_strategy=args.dnf_strategy,
        n_simulations=args.n_simulations,
        seed=args.seed,
    )

    for team_id in race_df["team_id_raw"].unique():
        team_drivers = race_df[race_df["team_id_raw"] == team_id]
        if len(team_drivers) != 2:
            print(f"Team {team_id} non ha esattamente 2 piloti, saltato.")
            continue
        driver_a = team_drivers.iloc[0]["driver_id_raw"]  # First driver
        driver_b = team_drivers.iloc[1]["driver_id_raw"]  # Second driver
        result = head_to_head(driver_a, driver_b, driver_ids, simulated_positions)
        print(f"\n=== Testa a testa: {driver_a} vs {driver_b} ===")
        for k, v in result.items():
            print(f"  {k}: {v:.1%}")

    if args.compare_a and args.compare_b:
        result = head_to_head(args.compare_a, args.compare_b, driver_ids, simulated_positions)
        print(f"\n=== Testa a testa: {args.compare_a} vs {args.compare_b} ===")
        for k, v in result.items():
            print(f"  {k}: {v:.1%}")
