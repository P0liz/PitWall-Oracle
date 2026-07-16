import mlflow
from xgboost import XGBRanker
from exercises.phase2.demo_walkforward import train

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.xgboost.autolog()

ranker = train()
model_info = mlflow.xgboost.log_model(xgb_model=ranker, name="iris_model")
