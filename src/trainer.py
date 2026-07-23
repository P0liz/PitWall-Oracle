import pandas as pd
import numpy as np
from xgboost import XGBRanker
from src.data.data_loader import DataLoader
import os
from pathlib import Path
from src.utils import setup_custom_logger
from .config import GLOBAL_SEED, DEFAULT_DECAY_RATE

TO_DROP = ["target", "technical_dnf_target", "race_number", "race_date"]


def make_weights(race_dates: pd.Series, decay_rate: float, reference_date: pd.Timestamp) -> np.ndarray:
    days_elapsed = (reference_date - race_dates).dt.days
    return np.exp(-decay_rate * days_elapsed)


class Training:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.model_dir = Path("models")
        self.data_dir = "data_files"
        os.makedirs(self.model_dir, exist_ok=True)

        # Inizializziamo a None, lo stato si popolerà a comando
        self.train_df = None
        self.test_df = None
        self.ranker = XGBRanker(
            objective="rank:ndcg",
            tree_method="hist",
            random_state=GLOBAL_SEED,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            eval_metric="ndcg@20",
            missing=np.nan,
            early_stopping_rounds=20,
            enable_categorical=True,
        )

    def save_artifacts(self, filename: str = "pitwall_oracle_v1.json"):
        """Salva il modello addestrato nel formato nativo di XGBoost."""
        if not hasattr(self.ranker, "feature_importances_"):
            raise ValueError("Il modello non è ancora stato addestrato. Impossibile salvare.")
        filepath = os.path.join(self.model_dir, filename)

        # Salva in formato JSON nativo
        self.ranker.save_model(filepath)
        print(f"Mappa motore (Modello) salvata con successo in: {filepath}")

    def load_artifacts(self, filename: str = "pitwall_oracle_v1.json"):
        """Carica un modello pre-addestrato per l'inferenza immediata."""
        filepath = (self.model_dir / filename).resolve()
        if not str(filepath).startswith(str(self.model_dir.resolve())):
            raise ValueError(f"Path non valido: {filepath}")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Nessun modello trovato in {filepath}")

        # Carica i pesi e la struttura direttamente nell'istanza esistente
        self.ranker.load_model(filepath)
        print(f"Modello caricato! Pronto per simulare la griglia di partenza.")
        return self.ranker


class StaticTraining(Training):
    def __init__(self, data_loader: DataLoader):
        super().__init__(data_loader)
        self.train_df = None
        self.test_df = None
        self.decay_rate = DEFAULT_DECAY_RATE
        self.log = setup_custom_logger("StaticTraining")
        self.ranker = XGBRanker(
            objective="rank:ndcg",
            tree_method="hist",
            random_state=GLOBAL_SEED,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            eval_metric="ndcg@20",
            missing=np.nan,
            early_stopping_rounds=20,
            enable_categorical=True,
        )

    async def prepare_data(self, force: bool = False):
        """Metodo esplicito: i dati vengono caricati solo quando chiami questo metodo."""
        print("Inizio Ingestion dati F1...")
        self.train_df, self.test_df = await self.data_loader.load(force=force)
        self.train_df.to_parquet(Path(self.data_dir) / "train_df.parquet", index=False)
        self.test_df.to_parquet(Path(self.data_dir) / "test_df.parquet", index=False)
        self.test_df["circuit_id"] = pd.Categorical(
            self.test_df["circuit_id"], categories=self.train_df["circuit_id"].cat.categories
        )
        print("Dati pronti nel Paddock.")

    def train(self):
        if self.train_df is None:
            raise ValueError("Dati non trovati. Esegui prepare_data() prima di train().")

        # Setting model params for eval
        self.ranker.early_stopping_rounds = 20
        self.ranker.eval_metric = "ndcg@20"

        self.train_df = self.train_df.sort_values("race_date").reset_index(drop=True)
        self.train_df["qid"] = pd.factorize(self.train_df["race_date"])[0]
        self.test_df = self.test_df.sort_values("race_date").reset_index(drop=True)
        self.test_df["qid"] = pd.factorize(self.test_df["race_date"])[0]

        reference_date = self.test_df["race_date"].min()  # prima gara del test set
        weights_tr = make_weights(self.train_df.groupby("qid")["race_date"].first(), self.decay_rate, reference_date)

        X_train = self.train_df.drop(TO_DROP + ["qid"], axis=1)
        X_test = self.test_df.drop(TO_DROP + ["qid"], axis=1)
        y_train, y_test = self.train_df["target"], self.test_df["target"]
        qid_train, qid_test = self.train_df["qid"], self.test_df["qid"]

        # ATTENZIONE: per rank:ndcg, sample_weight vuole UN PESO PER GRUPPO (per gara),
        # non uno per riga. Un peso per-riga fa fallire xgboost con un errore poco
        # leggibile ("group_weights.size() == group_ptr.size() - 1"). Dato che il tuo
        # peso dipende solo da race_date (uguale per tutti i piloti della stessa gara),
        # è comunque costante nel gruppo: basta un valore per gara.
        self.ranker.fit(
            X_train,
            y_train,
            qid=qid_train,
            sample_weight=weights_tr,
            eval_set=[(X_test, y_test)],
            eval_qid=[qid_test],
            verbose=False,
        )

        if self.ranker.early_stopping_rounds is not None:
            print("Miglior iterazione:", self.ranker.best_iteration)
            print("Best NDCG@20 su val:", self.ranker.best_score)
        importances = pd.Series(self.ranker.feature_importances_, index=X_train.columns)
        print(importances.sort_values(ascending=False).head(10))
        return self.ranker


class DynamicTraining(Training):
    def __init__(self, data_loader: DataLoader):
        super().__init__(data_loader)
        self.train_df = None
        self.test_df = None
        self.decay_rate = DEFAULT_DECAY_RATE
        self.log = setup_custom_logger("DynamicTraining")
        self.ranker = XGBRanker(
            objective="rank:ndcg",
            tree_method="hist",
            random_state=GLOBAL_SEED,
            missing=np.nan,
            enable_categorical=True,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
        )

    async def prepare_data(self, last_date, force: bool = False):
        """Metodo esplicito: i dati vengono caricati solo quando chiami questo metodo."""
        print("Inizio Ingestion dati F1...")
        self.train_df, self.test_df = await self.data_loader.load(last_date, force=force)
        self.test_df["circuit_id"] = pd.Categorical(
            self.test_df["circuit_id"], categories=self.train_df["circuit_id"].cat.categories
        )
        print("Dati pronti nel Paddock.")

    def train(self):
        if self.train_df is None:
            raise ValueError("Dati non trovati. Esegui prepare_data() prima di train().")

        # Ordiniamo temporalmente
        self.train_df = self.train_df.sort_values("race_date").reset_index(drop=True)
        self.train_df["qid"] = pd.factorize(self.train_df["race_date"])[0]

        # 100% dei dati disponibili va nel train set!
        X_train = self.train_df.drop(TO_DROP + ["qid"], axis=1)
        y_train = self.train_df["target"]
        qid_train = self.train_df["qid"]

        reference_date = self.train_df["race_date"].max()
        weights_tr = make_weights(self.train_df.groupby("qid")["race_date"].first(), self.decay_rate, reference_date)

        # Fit pulito senza eval_set. È deterministico e ultra-veloce.
        self.ranker.fit(X_train, y_train, qid=qid_train, sample_weight=weights_tr, verbose=False)

        # Monitoriamo l'importanza delle feature per verificare che il modello
        # stia effettivamente dando peso ai trend del 2026 (es. Track Affinity o qualifiche)
        importances = pd.Series(self.ranker.feature_importances_, index=X_train.columns)
        self.log.debug(f"Top Feature Importances:\n{importances.sort_values(ascending=False)}")
        return self.ranker


class PrequentialTracker:
    """Accumula le metriche gara-per-gara, senza mai mischiarle col test fisso 2025."""

    ndcg_scores: list[float] = []
    race_ids: list[str] = []

    def log(self, race_id, ndcg: float) -> None:
        self.race_ids.append(race_id)
        self.ndcg_scores.append(ndcg)

    def cumulative_mean(self) -> float:
        return float(np.mean(self.ndcg_scores)) if self.ndcg_scores else float("nan")
