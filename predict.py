from xgboost import XGBRanker
from src.data.gold_layer import GoldLayer
from src.data.data_loader import DataLoader

NEW_YEAR = 2026
PREDICT_RACE = 8  # numero gara da predire
champion_path = "models/pitwall_oracle_2026_7.json"

# Inference
champion_model = XGBRanker()
champion_model.load_model(champion_path)
gold = GoldLayer()
data_loader = DataLoader()

race_results = gold.build_features(NEW_YEAR, PREDICT_RACE, force=True)
race_df = race_results[-1]
cutoff_date = race_df["race_date"].iloc[0]
race_df["driver_id_raw"] = race_df["driver_id"].copy()  # salva prima dell'encoding
for col in ["driver_id", "team_id"]:
    race_df[col] = data_loader.apply_target_encoding(race_df, col, cutoff_date=cutoff_date)
race_df["circuit_id"] = race_df["circuit_id"].astype(data_loader.circuit_dtype)

X_new = race_df.drop(
    [col for col in ["target", "race_number", "race_date", "driver_id_raw"] if col in race_df.columns], axis=1
)
scores = champion_model.predict(X_new)

predictions = race_df[["driver_id_raw"]].copy()
predictions.columns = ["driver_id"]
predictions["score"] = scores
print(predictions.sort_values("score", ascending=False).reset_index(drop=True))

"""
predictions["actual_position"] = (race_df["target"].max() - race_df["target"] + 1).values
predictions["predicted_position"] = predictions["score"].rank(ascending=False).astype(int)


def topn_exact_accuracy(n):
    top_n = predictions.nsmallest(n, "actual_position")
    correct = (top_n["predicted_position"].values == top_n["actual_position"].values).sum()
    return correct / n


print(f"Podio exact accuracy: {topn_exact_accuracy(3):.0%}")
for n in [5, 10, 20]:
    print(f"Top-{n} exact accuracy: {topn_exact_accuracy(n):.0%}")
"""
