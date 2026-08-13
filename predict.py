from xgboost import XGBRanker

from src.ranker_model import select_model_feature_frame
from src.data.gold_layer import GoldLayer
from src.data.data_loader import DataLoader
from src.config import NEW_YEAR
from src.ranker_model_loader import resolve_ranker_model_path

PREDICT_RACE = 4  # numero gara da predire
RACE_SESSION = 5


def main():
    champion_path = resolve_ranker_model_path(NEW_YEAR, PREDICT_RACE)
    if not champion_path.exists():
        raise FileNotFoundError(f"Modello Ranker non trovato in '{champion_path}'")

    champion_model = XGBRanker()
    champion_model.load_model(champion_path)
    gold = GoldLayer()
    data_loader = DataLoader()

    race_df = gold.build_prediction_features(NEW_YEAR, PREDICT_RACE, RACE_SESSION, force=False)
    # results = gold.build_features(NEW_YEAR, PREDICT_RACE, force=True)
    # race_df = results[-1]
    cutoff_date = race_df["race_date"].iloc[0]
    race_df["driver_id_raw"] = race_df["driver_id"].copy()  # salva prima dell'encoding
    for col in ["driver_id", "team_id"]:
        race_df[col] = data_loader.apply_target_encoding(race_df, col, cutoff_date=cutoff_date)
    race_df["circuit_id"] = race_df["circuit_id"].astype(data_loader.circuit_dtype)

    X_new = select_model_feature_frame(champion_model, race_df)
    scores = champion_model.predict(X_new)

    predictions = race_df[["driver_id_raw"]].copy()
    predictions.columns = ["driver_id"]
    predictions["score"] = scores
    print(predictions.sort_values("score", ascending=False).reset_index(drop=True))


if __name__ == "__main__":
    main()
