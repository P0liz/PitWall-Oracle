# PitWall-Oracle

Custom trained ML model to predict Formula 1 race outcomes.

## Web app

The public app serves precomputed PitWall Oracle results through a read-only
FastAPI API and presents them in Streamlit. The hosted services only read the
validated JSON published in [`webapp/api/data`](webapp/api/data).

The UI separates the next-race prediction and Head-to-Head tool from the
historical predicted-versus-real comparison.

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

For details, see the [web-app design](docs/superpowers/specs/2026-08-08-web-app-design.md)
and [implementation plan](docs/superpowers/plans/2026-08-08-web-app-mvp.md).
The implementation is organised around the [API app](webapp/api/app.py),
[data contracts](webapp/api/schemas.py), [JSON repository](webapp/api/repository.py),
and [Streamlit entry point](webapp/ui/streamlit_app.py).

## Ranker training and evaluation

The ranker uses the explicit `PRODUCTION_FEATURES` tuple in
`src/ranker_features.py`; feature subsets are no longer selected by a separate
combinatorial pipeline.

Run the optimized training workflow with:

```powershell
.\venv\Scripts\python.exe train_ranker_optimized.py
```

Optuna tunes model parameters on expanding temporal folds. Its primary score is
mean race-level pairwise accuracy over the full classified grid. The dynamic
champion/challenger evaluation additionally requires no regression in teammate
pairwise accuracy or mean absolute position error. NDCG and top-k overlap remain
diagnostic metrics. Sprint queries are excluded from ranker training and primary
evaluation; DNF and Monte Carlo evaluation are separate workflows.

## DNF strategies

The Monte Carlo simulator supports interchangeable DNF probability providers:

```text
none | global_rate | team_beta | heuristic | logistic
```

`team_beta` is the default. It estimates constructor reliability with a
Beta-Binomial prior, shrinking teams with little history towards the causal
global DNF rate.

Run a simulation with:

```powershell
python monte_carlo_simulator.py --year 2026 --race 10 --dnf-strategy team_beta
```

The logistic model remains available for experiments:

```powershell
python monte_carlo_simulator.py --year 2026 --race 10 --dnf-strategy logistic
```

## Ablation

Evaluate the DNF probability providers:

```powershell
python evaluate_dnf_ablation.py
```

Evaluate their end-to-end effect on win, podium and points probabilities:

```powershell
python evaluate_simulator_ablation.py --year 2026
```

Both commands write reproducible predictions, metrics, reliability data and
GO/NO-GO gate results under `results/ablation`.
