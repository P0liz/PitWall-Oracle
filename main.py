from src.data.gold_layer import GoldLayer
from src.data.history_builder import HistoryBuilder
from src.data.trainer import StaticTraining, DynamicTraining, PrequentialTracker, compute_ndcg
from src.data.data_loader import DataLoader, NEW_YEAR, categorical_cols
import asyncio
import fastf1
import pandas as pd
import os
from xgboost import XGBRanker

BASE_MODEL = "pitwall_oracle_base.json"
TOLLERANCE = 0.001
FORCE = False

if __name__ == "__main__":
    data_loader = DataLoader()

    # Static training
    trainer = StaticTraining(data_loader)
    if (not os.path.exists(f"{trainer.model_dir}/{BASE_MODEL}")) and (not FORCE):
        asyncio.run(trainer.prepare_data())
        print("Inizio training su dati statici...")
        challenger = trainer.train()
        trainer.save_artifacts("pitwall_oracle_base.json")
    X_test_fixed, y_test_fixed, test_group_sizes = trainer.get_train_data()

    # Dynamic training
    trainer = DynamicTraining(data_loader)
    tracker = PrequentialTracker()
    gold = GoldLayer()
    champion_path = trainer.model_dir / BASE_MODEL

    schedule = fastf1.get_event_schedule(NEW_YEAR)
    races = schedule.loc[schedule["Session5DateUtc"] <= pd.Timestamp.now(), "Session5DateUtc"]

    print("Inizio testing su gare dinamiche...")
    for idx, date in enumerate(races):
        print(f"Elaborazione gara del {date.date()}")
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
        print(
            f"[{NEW_YEAR}_{idx}] NDCG prequenziale: {ndcg:.4f} | media cumulativa 2026: {tracker.cumulative_mean():.4f}"
        )

        # Train challenger
        asyncio.run(trainer.prepare_data(date, force=FORCE))
        print(
            f"[{NEW_YEAR}_{idx}] train shape: {trainer.train_df.shape}, "
            f"max race_date: {trainer.train_df['race_date'].max()}"
        )
        challenger = trainer.train()  # independent from champion
        # trainer.save_artifacts(f"pitwall_oracle_{date.date()}.json")

        ndcg_champion_fixed = compute_ndcg(
            y_test_fixed.to_numpy(), champion_model.predict(X_test_fixed), group_sizes=test_group_sizes
        )
        ndcg_challenger_fixed = compute_ndcg(
            y_test_fixed.to_numpy(), challenger.predict(X_test_fixed), group_sizes=test_group_sizes
        )

        assert champion_model is not challenger  # guardia esplicita, non deve mai fallire ora
        regressed = ndcg_challenger_fixed < ndcg_champion_fixed - TOLLERANCE

        # Promozione condizionata
        if not regressed:
            new_filename = f"pitwall_oracle_{NEW_YEAR}_{idx}.json"
            trainer.save_artifacts(new_filename)
            champion_path = trainer.model_dir / new_filename  # aggiorna solo il PUNTATORE al file
            print(
                f"[{NEW_YEAR}_{idx}] Challenger promosso a champion: "
                f"champion={ndcg_champion_fixed} vs challenger={ndcg_challenger_fixed}"
            )
            trainer.log.info(
                f"[{NEW_YEAR}_{idx}] Challenger promosso a champion: "
                f"champion={ndcg_champion_fixed} vs challenger={ndcg_challenger_fixed}"
            )
        else:
            print(
                f"[{NEW_YEAR}_{idx}] Challenger RIFIUTATO (regressione su test fisso 2025): "
                f"champion={ndcg_champion_fixed} vs challenger={ndcg_challenger_fixed}"
            )
            trainer.log.info(
                f"[{NEW_YEAR}_{idx}] Challenger RIFIUTATO (regressione su test fisso 2025): "
                f"champion={ndcg_champion_fixed} vs challenger={ndcg_challenger_fixed}"
            )
    print("Fine training dinamico")
