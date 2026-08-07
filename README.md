# PitWall-Oracle

Custom trained ML model to predict Formula 1 race outcomes.

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
