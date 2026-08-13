import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from src.data.data_loader import DataLoader
from src.data.gold_layer import GoldLayer
from src.dnf.dnf_model_loader import resolve_dnf_model_path
from src.model_lifecycle import CALIBRATION_NAME, ModelLifecycleError, resolve_calibration
from src.ranker.ranker_model import select_model_feature_frame
from src.dnf.dnf_model import compute_probabilities, load_dnf_artifact, normalize_utc_timestamp
from src.ranker.ranker_model_loader import resolve_ranker_model_path
from src.utils import setup_custom_logger
from src.config import FALLBACK_SIGMA

log = setup_custom_logger("MonteCarloSimulator")

MODEL_DIR = Path("models")
CALIBRATION_PATH = MODEL_DIR / CALIBRATION_NAME

N_SIMULATIONS = 10000
ORDER_BY = "base_position"  # or "expected_position" "expected_position_if_finished"


@dataclass(frozen=True)
class SimulationResult:
    summary: pd.DataFrame
    simulated_positions: np.ndarray
    dnf_draws: np.ndarray
    driver_ids: np.ndarray
    race_frame: pd.DataFrame
    ranker_model_path: Path
    sigma_relative: float
    sigma_absolute: float


# Sigma RELATIVO (adimensionale): frazione dello spread di score DELLA GARA
# CORRENTE che rappresenta rumore/incertezza del modello. Va moltiplicato per
# scores.std() della gara specifica per ottenere il sigma assoluto (vedi main()).
def fetch_relative_sigma(year: int, race_number: int, calibration_path: Path = CALIBRATION_PATH) -> float:
    """
    Recupera il sigma disponibile prima della gara richiesta.

    Per la gara N usa il valore con selected_after_round massimo ma
    strettamente minore di N.
    """
    if race_number < 1:
        raise ValueError("race_number deve essere maggiore o uguale a 1")

    try:
        sigma_relative = resolve_calibration(calibration_path, year, race_number)
    except ModelLifecycleError as error:
        raise RuntimeError(
            f"Monte Carlo calibration unavailable for year={year}, race_number={race_number}: {error}"
        ) from error
    log.info(f"Sigma relativo causale per {year} gara {race_number}: {sigma_relative:.4f}")
    return sigma_relative


def model_path_for_race(year: int, race_number: int, local_model_dir: Path = MODEL_DIR) -> Path:
    """Usa la policy Ranker condivisa con predict.py."""
    return resolve_ranker_model_path(year=year, race_number=race_number, local_model_dir=local_model_dir)


def dnf_model_path_for_race(year: int, race_number: int, local_model_dir: Path = MODEL_DIR) -> Path:
    """Usa per il DNF la stessa policy temporale locale del Ranker."""
    return resolve_dnf_model_path(year=year, race_number=race_number, local_model_dir=local_model_dir)


def compute_dnf_probabilities(
    race_df: pd.DataFrame,
    year: int,
    race_number: int,
    strategy: str = "logistic",
    history_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Calcola probabilità DNF pre-gara con una strategia esplicita."""
    del history_df
    if strategy == "none":
        return np.zeros(len(race_df), dtype=np.float64)
    if strategy != "logistic":
        raise ValueError(f"Strategia DNF non supportata: '{strategy}'")

    race_date = normalize_utc_timestamp(race_df["race_date"].iloc[0], "race_date")
    artifact_path = dnf_model_path_for_race(year, race_number)
    print(f"Loading DNF model {artifact_path.name}")
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Modello DNF logistico non trovato in '{artifact_path}' per year={year}, race_number={race_number}"
        )

    dnf_artifact = load_dnf_artifact(artifact_path)
    model_type = dnf_artifact.get("model_type")
    if model_type != "logistic":
        raise ValueError(f"Tipo modello DNF non compatibile: artifact={model_type}, richiesto=logistic")

    cutoff_date = normalize_utc_timestamp(dnf_artifact["cutoff_date"], "cutoff_date")
    if cutoff_date >= race_date:
        raise ValueError(
            f"Leakage temporale nel modello DNF '{artifact_path}': cutoff {cutoff_date} "
            f"non precedente alla gara {race_date}"
        )

    probabilities = compute_probabilities(prediction_df=race_df, artifact=dnf_artifact)
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
    driver_ids: np.ndarray,
    simulated_positions: np.ndarray,
    dnf_draws: np.ndarray,
    base_scores: np.ndarray,
    points_cutoff: int = 10,
) -> pd.DataFrame:
    if points_cutoff < 1:
        raise ValueError("points_cutoff deve essere maggiore di zero")
    base_positions = np.argsort(np.argsort(-base_scores)) + 1
    summary = pd.DataFrame({"driver_id": driver_ids, "base_score": base_scores, "base_position": base_positions})
    finished = ~dnf_draws
    summary["win_probability"] = ((simulated_positions == 1) & finished).mean(axis=0)
    summary["podium_probability"] = ((simulated_positions <= 3) & finished).mean(axis=0)
    summary["points_probability"] = ((simulated_positions <= points_cutoff) & finished).mean(axis=0)
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

    return {driver_a: float((pos_a < pos_b).mean()), driver_b: float((pos_b < pos_a).mean())}


def build_head_to_head_matrix(driver_ids: np.ndarray, simulated_positions: np.ndarray) -> dict[str, dict[str, float]]:
    if simulated_positions.ndim != 2 or simulated_positions.shape[1] != len(driver_ids):
        raise ValueError("Le posizioni simulate non corrispondono ai driver")
    normalized_ids = [str(driver_id) for driver_id in driver_ids]
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("I driver_id devono essere unici")

    matrix: dict[str, dict[str, float]] = {driver_id: {} for driver_id in normalized_ids}
    for first_index, driver_a in enumerate(normalized_ids):
        for second_index in range(first_index + 1, len(normalized_ids)):
            driver_b = normalized_ids[second_index]
            probability_a = float((simulated_positions[:, first_index] < simulated_positions[:, second_index]).mean())
            matrix[driver_a][driver_b] = probability_a
            matrix[driver_b][driver_a] = 1.0 - probability_a
    return matrix


def simulate_race(
    year: int, race_number: int, session: int, force: bool = False, n_simulations: int = N_SIMULATIONS, seed: int = 42
) -> SimulationResult:
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

    try:
        race_df = gold.build_prediction_features(year, race_number, session, force=force)
    except Exception as e:
        raise RuntimeError(
            f"Errore nel caricamento dei dati per season: {year}, race: {race_number}, session: {session}, {e}"
        )

    race_df["driver_id_raw"] = race_df["driver_id"].copy()  # salva prima del target encoding
    race_df["team_id_raw"] = race_df["team_id"].copy()
    cutoff_date = race_df["race_date"].iloc[0]

    dnf_probabilities = compute_dnf_probabilities(race_df, year, race_number)

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
        sigma_absolute = FALLBACK_SIGMA
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
    points_cutoff = 8 if session != 5 else 10
    summary = summarize_results(driver_ids_raw, simulated_positions, dnf_draws, scores, points_cutoff=points_cutoff)

    return SimulationResult(
        summary=summary,
        simulated_positions=simulated_positions,
        dnf_draws=dnf_draws,
        driver_ids=driver_ids_raw,
        race_frame=race_df,
        ranker_model_path=model_path,
        sigma_relative=sigma_relative,
        sigma_absolute=sigma_absolute,
    )


def main(
    year: int, race_number: int, session: int, force: bool = False, n_simulations: int = N_SIMULATIONS, seed: int = 42
):
    result = simulate_race(year, race_number, session, force, n_simulations, seed)
    summary = result.summary

    summary_path = Path(f"results/summary_{year}_{race_number}_{session}.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    pd.set_option("display.float_format", "{:.1%}".format)
    session_label = "Sprint" if session != 5 else "GP"
    print(
        f"\n=== Simulazione Monte Carlo — {year} {session_label} #{race_number} "
        f"({n_simulations} run, sigma assoluto={result.sigma_absolute:.4f}) ===\n"
    )
    print(summary.to_string(index=False))
    print(f"\nSummary salvato in: {summary_path.resolve()}")

    podium_sum = summary["podium_probability"].sum()
    print(f"\n[Sanity check] Somma probabilità di podio dei finisher: {podium_sum:.3f} ")

    return summary, result.simulated_positions, result.driver_ids, result.race_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo simulazione Monte Carlo — PitWall Oracle")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--race", type=int, default=11, help="Numero di gara da simulare")
    parser.add_argument("--session", type=int, default=5, help="Sessione da cui prendere le feature")
    parser.add_argument("--force", action="store_true", help="Forza il refresh delle feature Gold")
    parser.add_argument("--n-simulations", type=int, default=N_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--compare-a", type=str, default=None, help="driver_id per il confronto testa a testa")
    parser.add_argument("--compare-b", type=str, default=None, help="driver_id per il confronto testa a testa")
    args = parser.parse_args()

    summary, simulated_positions, driver_ids, race_df = main(
        args.year, args.race, args.session, force=args.force, n_simulations=args.n_simulations, seed=args.seed
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
