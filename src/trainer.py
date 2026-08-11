import pandas as pd
import numpy as np
from xgboost import XGBRanker
from src.data.data_loader import DataLoader
import os
from collections.abc import Sequence
from pathlib import Path
from src.utils import setup_custom_logger
from .config import GLOBAL_SEED, DEFAULT_DECAY_RATE, NEW_YEAR
from .ranker_features import PRODUCTION_FEATURES


def make_weights(race_dates: pd.Series, decay_rate: float, reference_date: pd.Timestamp) -> np.ndarray:
    days_elapsed = (reference_date - race_dates).dt.days
    return np.exp(-decay_rate * days_elapsed)


def select_model_feature_frame(model: XGBRanker, frame: pd.DataFrame) -> pd.DataFrame:
    """Align inference columns and order to the feature names stored by XGBoost."""

    feature_names = model.get_booster().feature_names
    if not feature_names:
        raise ValueError("Il modello non espone i nomi delle feature")
    missing = [feature for feature in feature_names if feature not in frame.columns]
    if missing:
        raise ValueError(f"Feature richieste dal modello ma assenti dal dataframe: {missing}")
    return frame.loc[:, feature_names]


class Training:
    def __init__(
        self,
        data_loader: DataLoader,
        feature_names: Sequence[str] | None = None,
        target_year: int = NEW_YEAR,
        target_train_multiplier: float = 1.0,
    ):
        self.data_loader = data_loader
        self.feature_names = tuple(feature_names) if feature_names is not None else PRODUCTION_FEATURES
        self.target_year = target_year
        self.target_train_multiplier = target_train_multiplier
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
            ndcg_exp_gain=False,
            missing=np.nan,
            early_stopping_rounds=20,
            enable_categorical=True,
        )

    def feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Select the explicit production feature set in deterministic order."""

        missing = [feature for feature in self.feature_names if feature not in frame.columns]
        if missing:
            raise ValueError(f"Feature configurate ma assenti dal dataframe: {missing}")
        return frame.loc[:, self.feature_names]

    def group_weights(self, frame: pd.DataFrame, reference_date: pd.Timestamp) -> np.ndarray:
        """Create one temporal/regime-aware weight per qid."""

        groups = frame.groupby("qid", sort=True).agg(race_date=("race_date", "first"), year=("year", "first"))
        weights = make_weights(groups["race_date"], self.decay_rate, reference_date)
        weights[groups["year"].to_numpy() == self.target_year] *= self.target_train_multiplier
        return weights

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
    def __init__(
        self,
        data_loader: DataLoader,
        feature_names: Sequence[str] | None = None,
        target_year: int = 2026,
        target_train_multiplier: float = 1.0,
    ):
        super().__init__(data_loader, feature_names, target_year, target_train_multiplier)
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
            ndcg_exp_gain=False,
            missing=np.nan,
            early_stopping_rounds=20,
            enable_categorical=True,
        )

    async def prepare_data(self, is_dynamic: bool = False, force: bool = False):
        """Metodo esplicito: i dati vengono caricati solo quando chiami questo metodo."""
        print("Inizio Ingestion dati F1...")
        self.train_df, self.test_df = await self.data_loader.load_data(is_dynamic=is_dynamic, force=force)
        self.test_df["circuit_id"] = pd.Categorical(
            self.test_df["circuit_id"], categories=self.train_df["circuit_id"].cat.categories
        )
        print("Dati pronti nel Paddock.")
        return self.train_df, self.test_df

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
        weights_tr = self.group_weights(self.train_df, reference_date)

        X_train = self.feature_frame(self.train_df)
        X_test = self.feature_frame(self.test_df)
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
    def __init__(
        self,
        data_loader: DataLoader,
        feature_names: Sequence[str] | None = None,
        target_year: int = 2026,
        target_train_multiplier: float = 1.0,
    ):
        super().__init__(data_loader, feature_names, target_year, target_train_multiplier)
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
            ndcg_exp_gain=False,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
        )

    async def prepare_data(self, static_df: pd.DataFrame, last_date, is_dynamic: bool = True, force: bool = False):
        """Metodo esplicito: i dati vengono caricati solo quando chiami questo metodo."""
        print("Inizio Ingestion dati F1...")
        self.train_df, self.test_df = await self.data_loader.load_data(
            last_date, static_df, is_dynamic=is_dynamic, force=force
        )
        print("Dati pronti nel Paddock.")

    def train(self):
        if self.train_df is None:
            raise ValueError("Dati non trovati. Esegui prepare_data() prima di train().")

        # Ordiniamo temporalmente
        self.train_df = self.train_df.sort_values("race_date").reset_index(drop=True)
        self.train_df["qid"] = pd.factorize(self.train_df["race_date"])[0]

        # 100% dei dati disponibili va nel train set!
        X_train = self.feature_frame(self.train_df)
        y_train = self.train_df["target"]
        qid_train = self.train_df["qid"]

        reference_date = self.train_df["race_date"].max()
        weights_tr = self.group_weights(self.train_df, reference_date)

        # Fit pulito senza eval_set. È deterministico e ultra-veloce.
        self.ranker.fit(X_train, y_train, qid=qid_train, sample_weight=weights_tr, verbose=False)

        # Monitoriamo l'importanza delle feature per verificare che il modello
        # stia effettivamente dando peso ai trend del 2026 (es. Track Affinity o qualifiche)
        importances = pd.Series(self.ranker.feature_importances_, index=X_train.columns)
        self.log.debug(f"Top Feature Importances:\n{importances.sort_values(ascending=False)}")
        return self.ranker
