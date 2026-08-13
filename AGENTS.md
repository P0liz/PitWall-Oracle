# PitWall Oracle — agent instructions

## Project overview

PitWall Oracle predicts Formula 1 finishing order with an `XGBRanker`, models
DNF probability separately, and feeds both into a Monte Carlo race simulator.
The repository is developed primarily on Windows with PowerShell and the local
virtual environment under `venv`.

All public-facing webapp content must be written exclusively in English. Do not
add a translation or localization system unless explicitly requested.

Prefer:

```powershell
.\venv\Scripts\python.exe <script>
```

The repository `venv` is valid even when a sandboxed shell cannot reach its
`base_prefix` under `AppData\Local\Python`. If that access fails, rerun the
same `venv` command with the required sandbox escalation. Do not diagnose the
environment as broken, switch to `uv`, or recreate the virtual environment
based only on that sandbox error.

Do not assume the working tree is clean. Preserve unrelated user changes,
generated models, cached data, and experiment results.

## Data architecture and causal safety

The data pipeline is layered:

- `src/data/bronze_layer.py`: raw external data;
- `src/data/silver_layer.py`: cleaning and normalization;
- `src/data/history_builder.py`: historical state;
- `src/data/feature_engineer.py`: causal feature calculations;
- `src/data/gold_layer.py`: one model-ready row per driver and race query;
- `src/data/data_loader.py`: temporal assembly, target encoding and dtypes.

Historical features for a race must use only information strictly preceding
that race. Driver/team target encoding must always use `race_date < cutoff`.
Never use the current race result, future races, or post-race information in a
pre-race feature.

Gold parquet files are cached under `data_files/gold`. After adding or changing
a Gold feature, rebuild the affected parquet data before training or feature
selection. A forced full rebuild can be expensive and may refresh lower data
layers, so do it only when required or explicitly requested.

Keep `circuit_id` as a pandas categorical with consistent categories. The
ranker uses encoded numeric `driver_id` and `team_id`; raw string identifiers
must not be passed directly to XGBoost.

## Ranking model invariants

- A query/group is one race, identified by `race_date`/`qid`.
- Rows must be sorted so `qid` is nondecreasing before fitting.
- `target` is relevance, where a larger value means a better finishing result.
- Drivers without a valid classified finishing target are excluded from ranker
  training.
- XGBoost ranker `sample_weight` must contain one value per race/query, not one
  value per driver row.
- NDCG comparisons must be computed race by race and then aggregated; do not
  flatten different races into a single ranking query.
- Candidate and reference models must be compared on the same temporal folds
  with paired per-race deltas.

DNF modelling is separate from ranking. Changes to DNF features, targets, or
calibration should not silently alter the ranker feature matrix.

## Feature registry

`src/ranker_features.py` is the source of truth for ranker inputs:

- `MANDATORY_FEATURES` and `OPTIONAL_FEATURES` document the complete registry;
- `PRODUCTION_FEATURES` is the fixed, ordered input used by training;
- `ALL_RANKER_FEATURES` must exactly match the model-input columns produced by
  Gold after excluding metadata and targets.

Do not return to the old pattern of treating every column outside `TO_DROP` as
an automatic model feature. New features must be deliberately added to Gold
and to the explicit registry. `year` and `regulation_era` are intentional
mandatory features even if a particular trained model assigns them zero gain.

At inference, select and order columns using the feature names stored by the
XGBoost model. Feature changes are intentional ablation experiments against the
fixed production set, not an automated subset search.

## Ranker evaluation workflow

`train_ranker_optimized.py` is the only ranker tuning, evaluation and training
orchestrator. There is no separate feature-selection pipeline or production
manifest.

Ranker evaluation is GP-only. Sprint results must not be mixed into the primary
training/evaluation population; they may be used only through explicitly causal
historical features or reported as a separate diagnostic segment.

The primary HPO metric is mean race-level pairwise accuracy over the full
classified grid. Report separately:

- full-grid pairwise accuracy;
- teammate pairwise accuracy using the raw team metadata, never encoded IDs;
- mean absolute position error;
- full-grid, top-5 and top-10 NDCG as diagnostics;
- top-3 and top-5 set overlap as diagnostics.

XGBoost and Python metrics must use the same linear-gain convention
(`ndcg_exp_gain=False`). Control and challenger are compared on exactly the same
future GP with paired per-race deltas. A challenger remains pending until enough
out-of-sample GP comparisons exist; `pitwall_oracle_latest.json` must always be
the last promoted champion.

DNF and Monte Carlo metrics are outside ranker evaluation. Do not use them in
ranker HPO or champion/challenger promotion gates.

Typical command:

```powershell
.\venv\Scripts\python.exe train_ranker_optimized.py
```

## Coding policy

- Never create commits or push changes. Implement what is asked and then let
the user decide what to commit.
- If the requested changes only concern a small portion of code (few lines in one or two files), it is not necessary to write a full plan or specs document; instead write a preview of the corrections in chat and ask for the user's permission to procede.

## Testing policy

Tests of any kind must go inside the tests folder.

This is a compact experimental ML project. Do not automatically create or run
a broad test suite for every intermediate feature addition, screening step, or
small pipeline phase.

- During intermediate development, prefer compilation/import checks and one
  focused smoke check only when needed to verify that changed components connect
  correctly.
- Add a targeted regression test only when a high-risk invariant—especially
  temporal leakage, qid grouping, group-weight cardinality, schema matching, or
  feature-order alignment—cannot be verified reliably with a simpler check.
- Reserve comprehensive tests, full backtests, simulator validation, and broad
  metric review for the final selected-model build or when the user explicitly
  requests them.
- Do not remove existing tests merely because an intermediate phase does not
  require new ones.

When reporting verification, distinguish focused development checks from the
final model evaluation. Never describe a screening/HPO validation score as an
untouched test result.
