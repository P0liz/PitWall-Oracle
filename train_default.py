from src.data.gold_layer import GoldLayer
from src.data.history_builder import HistoryBuilder
from src.data.trainer import StaticTraining, DynamicTraining, PrequentialTracker, compute_ndcg
from src.data.data_loader import DataLoader, NEW_YEAR, STATIC_ENDING_YEAR, categorical_cols
import asyncio
import fastf1
import pandas as pd
from xgboost import XGBRanker

BASE_MODEL = "pitwall_oracle_base.json"
TOLLERANCE = 0.001
FORCE = True

if __name__ == "__main__":
    data_loader = DataLoader()

    # Static training
    trainer = StaticTraining(data_loader)
    asyncio.run(trainer.prepare_data(force=FORCE))
    trainer.log.info("Inizio training su dati statici...")
    challenger = trainer.train()
    trainer.save_artifacts(BASE_MODEL)
    X_test_fixed, y_test_fixed, test_group_sizes = trainer.get_test_data()

    # Dynamic training
    trainer = DynamicTraining(data_loader)
    tracker = PrequentialTracker()
    gold = GoldLayer()
    champion_path = trainer.model_dir / BASE_MODEL

    old_schedule = fastf1.get_event_schedule(STATIC_ENDING_YEAR)
    previous_date = old_schedule.loc[old_schedule["Session5DateUtc"] <= pd.Timestamp.now(), "Session5DateUtc"].max()
    new_schedule = fastf1.get_event_schedule(NEW_YEAR)
    races = new_schedule.loc[new_schedule["Session5DateUtc"] <= pd.Timestamp.now(), "Session5DateUtc"]

    trainer.log.info("Inizio testing su gare dinamiche...")
    for idx, date in enumerate(races):
        trainer.log.info(f"Elaborazione gara del {date.date()}")
        # Computing new race results
        idx += 1
        # Champion: istanza SEMPRE fresca, caricata da disco, mai condivisa con trainer.ranker
        champion_model = XGBRanker()
        champion_model.load_model(champion_path)

        new_race_results = gold.build_features(NEW_YEAR, idx, force=FORCE)
        race_df = new_race_results[-1]  # Only consider main race
        # Drop nan target rows
        race_df = race_df.dropna(subset=["target"])
        # Target encoding
        cutoff_date = race_df["race_date"].iloc[0]
        for col in categorical_cols:
            race_df[col] = data_loader.apply_target_encoding(race_df, col, cutoff_date=cutoff_date)
        # Categories
        race_df["circuit_id"] = race_df["circuit_id"].astype(data_loader.circuit_dtype)

        # Predict new race with actual champion
        X_new = race_df.drop(["target", "race_number", "race_date"], axis=1)
        scores_pred = champion_model.predict(X_new)

        # Evaluate and log metrics
        ndcg = compute_ndcg(race_df["target"].to_numpy(), scores_pred)
        tracker.log((NEW_YEAR, idx), ndcg)
        trainer.log.info(
            f"[{NEW_YEAR}_{idx}] NDCG prequenziale: {ndcg:.4f} | media cumulativa 2026: {tracker.cumulative_mean():.4f}"
        )

        # Train challenger
        asyncio.run(trainer.prepare_data(previous_date, force=FORCE))
        trainer.log.debug(
            f"[{NEW_YEAR}_{idx}] train shape: {trainer.train_df.shape}, "
            f"max race_date: {trainer.train_df['race_date'].max()}"
        )
        challenger = trainer.train()  # independent from champion

        # ora champion e challenger predicono ENTRAMBI sulla gara idx, mai vista da nessuno dei due
        scores_champion = champion_model.predict(X_new)
        scores_challenger = challenger.predict(X_new)
        ndcg_champion_new = compute_ndcg(race_df["target"].to_numpy(), scores_champion)
        ndcg_challenger_new = compute_ndcg(race_df["target"].to_numpy(), scores_challenger)

        assert champion_model is not challenger  # guardia esplicita, non deve mai fallire ora
        regressed = ndcg_challenger_new < ndcg_champion_new - TOLLERANCE

        # Promozione condizionata
        if not regressed:
            new_filename = f"pitwall_oracle_{NEW_YEAR}_{idx}.json"
            trainer.save_artifacts(new_filename)
            champion_path = trainer.model_dir / new_filename  # aggiorna solo il PUNTATORE al file
            trainer.log.info(
                f"[{NEW_YEAR}_{idx}] Challenger promosso a champion: "
                f"champion={ndcg_champion_new} vs challenger={ndcg_challenger_new}"
            )
        else:
            trainer.log.info(
                f"[{NEW_YEAR}_{idx}] Challenger RIFIUTATO (regressione su test fisso 2025): "
                f"champion={ndcg_champion_new} vs challenger={ndcg_challenger_new}"
            )

        previous_date = date  # update for new iter
    print("Fine training dinamico")
