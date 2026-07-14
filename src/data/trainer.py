import pandas as pd
import numpy as np
from xgboost import XGBRanker
from src.data.data_loader import DataLoader
import os
from pathlib import Path
from src.utils import setup_custom_logger
from dataclasses import field

LAMBDA_DECAY = 0.01  # da tarare in walk-forward CV, come da tue note
TO_DROP = ["target", "race_number", "race_date"]

# TODO correggere formato gara weekend sprint 2023 (era tipo il sabato dedicato solo a sprint)


def make_weights(race_dates: pd.Series) -> np.ndarray:
    max_date = race_dates.max()
    days_elapsed = (max_date - race_dates).dt.days
    return np.exp(-LAMBDA_DECAY * days_elapsed)


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
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            eval_metric="ndcg@20",
            missing=np.nan,
            early_stopping_rounds=20,
            enable_categorical=True,
        )

    def train(self):
        if self.train_df is None:
            raise ValueError("Dati non trovati. Esegui prepare_data() prima di train().")

        self.train_df = self.train_df.sort_values("race_date").reset_index(drop=True)
        self.train_df["qid"] = pd.factorize(self.train_df["race_date"])[0]
        self.test_df = self.test_df.sort_values("race_date").reset_index(drop=True)
        self.test_df["qid"] = pd.factorize(self.test_df["race_date"])[0]

        X_train = self.train_df.drop(TO_DROP + ["qid"], axis=1)
        X_test = self.test_df.drop(TO_DROP + ["qid"], axis=1)
        y_train, y_test = self.train_df["target"], self.test_df["target"]
        qid_train, qid_val = self.train_df["qid"], self.test_df["qid"]

        # ATTENZIONE: per rank:ndcg, sample_weight vuole UN PESO PER GRUPPO (per gara),
        # non uno per riga. Un peso per-riga fa fallire xgboost con un errore poco
        # leggibile ("group_weights.size() == group_ptr.size() - 1"). Dato che il tuo
        # peso dipende solo da race_date (uguale per tutti i piloti della stessa gara),
        # è comunque costante nel gruppo: basta un valore per gara.
        self.ranker.fit(
            X_train,
            y_train,
            qid=qid_train,
            sample_weight=make_weights(self.train_df.groupby("qid")["race_date"].first()),
            eval_set=[(X_test, y_test)],
            eval_qid=[qid_val],
            verbose=False,
        )

        print("Miglior iterazione:", self.ranker.best_iteration)
        print("Best NDCG@20 su val:", self.ranker.best_score)
        importances = pd.Series(self.ranker.feature_importances_, index=X_train.columns)
        print(importances.sort_values(ascending=False).head(10))
        return self.ranker

    def get_train_data(self, force: bool = False):
        if not force:
            self.test_df = pd.read_parquet(f"{self.data_dir}/test_df.parquet")
            X_test = self.test_df.drop(TO_DROP, axis=1)
            y_test = self.test_df["target"]
            test_group_sizes = self.test_df.groupby("race_date", sort=False).size().to_numpy()
        else:
            raise NotImplementedError
        return X_test, y_test, test_group_sizes

    def save_artifacts(self, filename: str = "pitwall_oracle_v1.json"):
        """Salva il modello addestrato nel formato nativo di XGBoost."""
        if not hasattr(self.ranker, "best_iteration"):
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


class DynamicTraining(Training):
    def __init__(self, data_loader: DataLoader):
        super().__init__(data_loader)
        self.train_df = None
        self.test_df = None
        self.log = setup_custom_logger("DynamicTraining")

    async def prepare_data(self, last_date, force: bool = False):
        """Metodo esplicito: i dati vengono caricati solo quando chiami questo metodo."""
        print("Inizio Ingestion dati F1...")
        self.train_df, self.test_df = await self.data_loader.load(last_date, force=force)
        self.test_df["circuit_id"] = pd.Categorical(
            self.test_df["circuit_id"], categories=self.train_df["circuit_id"].cat.categories
        )
        print("Dati pronti nel Paddock.")


class PrequentialTracker:
    """Accumula le metriche gara-per-gara, senza mai mischiarle col test fisso 2025."""

    ndcg_scores: list[float] = []
    race_ids: list[str] = []

    def log(self, race_id, ndcg: float) -> None:
        self.race_ids.append(race_id)
        self.ndcg_scores.append(ndcg)

    def cumulative_mean(self) -> float:
        return float(np.mean(self.ndcg_scores)) if self.ndcg_scores else float("nan")


def compute_ndcg(
    y_true_rel: np.ndarray, y_pred_score: np.ndarray, group_sizes: np.ndarray | None = None, k: int = 22
) -> float:
    """
    NDCG@k coerente con l'obiettivo rank:ndcg di XGBRanker.

    y_true_rel: le label di rilevanza (Y = n_drivers - posizione + 1), NON le posizioni grezze --
                sklearn.ndcg_score assume "valore alto = piu' rilevante", stessa convenzione
                gia' discussa per il target del ranker.
    group_sizes: dimensione di ogni gruppo/gara, in ordine. Se None, si assume un'unica gara
                 (caso tipico della valutazione prequenziale: una gara alla volta).
    k: normalmente coincide col numero di piloti in griglia (~20); troncarlo piu' in basso
       avrebbe poco senso qui, a differenza dei sistemi di recommendation con liste lunghissime.

    NB: sklearn.metrics.ndcg_score vuole array 2D shape (n_queries, n_docs_per_query) --
    per gruppi di dimensione diversa (es. weekend con ritiri, DNS) non si puo' fare un unico
    array rettangolare: si itera gara per gara e si fa la media (stessa convenzione usata
    nella walk-forward CV su 2024-2025 -- media semplice tra le gare del fold, non pesata
    per numero di piloti).
    """
    from sklearn.metrics import ndcg_score

    y_true_rel = np.asarray(y_true_rel, dtype=float)
    y_pred_score = np.asarray(y_pred_score, dtype=float)

    if group_sizes is None:
        group_sizes = np.array([len(y_true_rel)])

    if group_sizes.sum() != len(y_true_rel):
        raise ValueError("group_sizes non copre tutte le righe passate a compute_ndcg")

    ndcgs = []
    start = 0
    for size in group_sizes:
        end = start + size
        true_slice = y_true_rel[start:end]
        pred_slice = y_pred_score[start:end]

        # ndcg_score richiede almeno 2 elementi rilevanti distinti per essere non-degenere;
        # con un solo pilota nel gruppo (caso limite, es. dati corrotti/singolo DNS) si salta.
        if size >= 2:
            ndcgs.append(ndcg_score(true_slice.reshape(1, -1), pred_slice.reshape(1, -1), k=min(k, size)))
        start = end

    if not ndcgs:
        return float("nan")

    return float(np.mean(ndcgs))
