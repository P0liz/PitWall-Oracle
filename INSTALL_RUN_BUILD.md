# Install, Run, and Model Workflows

## Prerequisites

> Python 3.12 or later

> A local virtual environment at `venv`

## Install

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the web application

Start the FastAPI service from the repository root:

```powershell
.\venv\Scripts\python.exe -m uvicorn webapp.api.app:app --host 127.0.0.1 --port 8000
```

Then, in a separate PowerShell terminal, start Streamlit:

```powershell
$env:PITWALL_API_URL = "http://127.0.0.1:8000"
.\venv\Scripts\python.exe -m streamlit run webapp\ui\streamlit_app.py
```

## Train the models

Run the ranker workflow:

```powershell
.\venv\Scripts\python.exe -m src.train_ranker_optimized
```

Run the DNF workflow:

```powershell
.\venv\Scripts\python.exe -m src.train_dnf_optimized
```

## Run a simulation

```powershell
.\venv\Scripts\python.exe -m src.monte_carlo_simulator --year 2026 --race 10
```

## Notes

- Gold data is cached under `data_files/gold`; rebuild affected data after a Gold-feature change before retraining.
- The web application reads published JSON files from `webapp/api/data`.
