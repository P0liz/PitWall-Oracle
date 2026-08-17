# PitWall-Oracle

Custom trained ML model to predict Formula 1 race outcomes.

## Repository structure

```text
PitWall-Oracle/
|-- src/          Data pipeline, ranking and DNF models, simulation, and publication logic
|-- scripts/      Automation entry points for prediction and post-race workflows
|-- webapp/       FastAPI backend, Streamlit UI, and published JSON data
|-- data_files/   Cached datasets and starting grids
|-- models/       Ranker, DNF, and Monte Carlo calibration artifacts
|-- tests/        Unit and regression tests
|-- exercises/    Exploratory and learning scripts
```

## Web app

The public app serves precomputed PitWall Oracle results through a read-only
FastAPI API and presents them in Streamlit. The hosted services only read the
validated JSON published in [`webapp/api/data`](webapp/api/data).

The UI separates the next-race prediction and Head-to-Head tool from the
historical predicted-versus-real comparison.

GitHub Actions automatically publishes due Sprint and Grand Prix predictions,
adds completed-session results to the history, and retrains the Ranker and DNF
models after newly published Grands Prix.

Streamlit was chosen because the current interface is small and primarily
data-driven. It keeps the web layer in Python and avoids introducing a separate
frontend stack before the product requires deeper visual customization. The
FastAPI boundary leaves room to replace the UI independently in the future.

Install the small web-app dependencies in the local virtual environment:

```powershell
.\venv\Scripts\python.exe -m pip install -r webapp\requirements.txt
.\venv\Scripts\python.exe -m pip install -r webapp\ui\requirements.txt
```

Start the API in one PowerShell terminal:

```powershell
.\venv\Scripts\python.exe -m uvicorn webapp.api.app:app --host 127.0.0.1 --port 8000
```

Then start Streamlit from the repository root in a second terminal:

```powershell
$env:PITWALL_API_URL = "http://127.0.0.1:8000"
.\venv\Scripts\python.exe -m streamlit run webapp/ui/streamlit_app.py
```

The webapp is deployed via streamlit and accessible at [PitWall Oracle](https://pitwall-oracle.streamlit.app/)

It is possibile that the page is **sleeping** (cause of Streamlit Community Cloud policy), so just press the button to wake it up and wait a few seconds before refreshing the page.

<!--
For details, see the [web-app design](docs/superpowers/specs/2026-08-08-web-app-design.md)
and [implementation plan](docs/superpowers/plans/2026-08-08-web-app-mvp.md).
The implementation is organised around the [API app](webapp/api/app.py),
[data contracts](webapp/api/schemas.py), [JSON repository](webapp/api/repository.py),
and [Streamlit entry point](webapp/ui/streamlit_app.py).
-->


## Ranker training and evaluation

The ranker uses the explicit `PRODUCTION_FEATURES` tuple in
`src/ranker/ranker_features.py`; feature subsets are no longer selected by a separate
combinatorial pipeline.

Run the optimized training workflow with:

```powershell
.\venv\Scripts\python.exe -m src.train_ranker_optimized
```

Optuna tunes model parameters on expanding temporal folds. Its primary score is
mean race-level pairwise accuracy over the full classified grid. The dynamic
champion/challenger evaluation additionally requires no regression in teammate
pairwise accuracy or mean absolute position error. NDCG and top-k overlap remain
diagnostic metrics. Sprint queries are excluded from ranker training and primary
evaluation; DNF and Monte Carlo evaluation are separate workflows.

Promotion intentionally uses only the latest out-of-sample Grand Prix. Earlier
regressions are not accumulated over a rolling window, so they cannot prevent a
challenger that improves the latest race, without regressing any current
scorecard metric, from being promoted. 

## DNF model and simulation

`src/train_dnf_optimized.py` is the single DNF training and optimization workflow.
It uses `LogisticRegression`; Optuna jointly selects the feature subset and
hyperparameters on expanding temporal folds, followed by a separate
out-of-sample holdout evaluation.

Optimization intentionally differs between the two models. The nonlinear
`XGBRanker` uses a fixed production feature set because independently toggling
30+ correlated inputs would create a very large, unstable search space; ranker
feature changes are therefore evaluated as explicit ablation challengers on
the same temporal folds. The DNF model is a linear logistic regression with a
smaller candidate registry, so Optuna can reasonably treat feature inclusion
as a boolean hyperparameter alongside regularization and class weighting.
In both workflows, feature or parameter selection uses development folds only;
the out-of-sample period remains separate from selection.

Train the DNF model with:

```powershell
.\venv\Scripts\python.exe -m src.train_dnf_optimized
```

The Monte Carlo simulator combines the ranker with the logistic DNF model. A
missing or incompatible logistic artifact produces an explicit error; the
simulator does not fall back to an alternative probability method.

Run a simulation with:

```powershell
.\venv\Scripts\python.exe -m src.monte_carlo_simulator --year 2026 --race 10
```

## Future improvements:
 - Add some sort of comparision with chat gpt or gemini (maybe via api key), for the prediction, to see who did better.
 - In the next race window add when and where the next race will be, plus a countdown.
 - Display also the matrix for head to head in some way.
 - Improve frontend by switching to Vue to realize a proper webapp (deployment on Vercel)
