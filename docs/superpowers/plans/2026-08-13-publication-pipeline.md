# Publication and Retraining Pipeline Implementation Plan

**Goal:** Build a repo-only pipeline that publishes race JSON and performs the
full Ranker and DNF walk-forward retraining in GitHub Actions after every GP.

**Architecture:** `data_files/` is committed training state. Both model families
train into staging, their existing loaders apply a strict local filename policy,
and two thin GitHub Actions workflows handle pre-race prediction and day-after
result publication/retraining.

**Spec:** `docs/superpowers/specs/2026-08-13-publication-pipeline-design.md`

## Constraints

- Preserve unrelated dirty-worktree changes.
- Never commit or push from the development agent.
- Use `apply_patch` for text changes and keep tests under `tests/`.
- Do not add dependencies without approval.
- Use the existing `venv` for focused checks.
- Keep public-facing content in English.
- Preserve causal feature construction and strict pre-race model eligibility.
- Do not run routine HPO in the post-race workflow.
- Never use broad model deletion or broad Git staging.
- `pending` is training-only and must not be committed.

## Task 1: Versioned data and local model resolvers

**Files:**

- Modify: `.gitignore`
- Modify: `src/ranker_model_loader.py`
- Modify: `src/dnf_model_loader.py`
- Create: `tests/test_ranker_model_loader.py`
- Modify: `tests/test_dnf_model_loader.py`

Steps:

1. Keep all `data_files/` trackable; keep `results/`, MLflow state, and crash
   telemetry ignored.
2. Write failing tests for strict `selected_after_round < requested_round`
   selection in both model families.
3. Remove inference-time MLflow lookup and select only local, season-matching
   filenames.
4. Ignore `latest`, `pending`, malformed, other-season, and future artifacts.
5. Prefer season-specific bases and temporarily accept the legacy base only for
   `NEW_YEAR`.
6. Verify round-before, round-at, and round-after selection without running a
   training backtest.

## Task 2: Shared historical resolver

**Files:**

- Modify: `src/ranker_model_loader.py`
- Modify: `src/dnf_model_loader.py`
- Modify: `predict.py`
- Modify: `monte_carlo_simulator.py`
- Modify focused existing tests

Steps:

1. Make `predict.py` and the simulator call the strict local loaders without
   MLflow client/cache parameters.
2. Resolve Ranker and DNF once per prediction request.
3. Make `predict.py` accept explicit season, round, session, and refresh flags.
4. Hard-fail on missing, corrupt, or causally ineligible artifacts.
5. Verify feature ordering remains model-driven and historical races select the
   intended earlier champion. Monte Carlo calibration is implemented separately
   as causal per-round output, not as a model registry.

## Task 3: Atomic model training lifecycle

**Files:**

- Create: `src/model_lifecycle.py`
- Modify: `train_ranker_optimized.py`
- Modify: `train_dnf_optimized.py`
- Create: `scripts/run_post_race_pipeline.py`
- Create: `tests/test_model_lifecycle.py`
- Modify focused training tests

Steps:

1. Write failure-first tests proving invalid staging never changes live models
   or calibration.
2. Make each trainer accept an output directory rather than assuming `models/`.
3. Run Ranker and DNF sequentially into a unique staging directory.
4. Treat `pending` as an internal walk-forward buffer only.
5. Validate model loading, promoted-round filenames, and causal per-round sigma
   output.
6. Replace only current-season managed artifacts plus `base` and `latest`, one
   file at a time; preserve earlier seasons and unrelated files.
7. Delete staging on success. On failure retain the published model set and
   emit diagnostics without a Git commit.

## Task 4: Prediction and actual-result exporters

**Files:**

- Create: `src/publication/__init__.py`
- Create: `src/publication/prediction.py`
- Create: `src/publication/actual.py`
- Create: `src/publication/publisher.py`
- Create: `src/publication/scheduler.py`
- Create: `scripts/generate_web_result.py`
- Modify: `monte_carlo_simulator.py`
- Modify: `src/data/gold_layer.py`
- Modify only affected public schemas and fixtures
- Create focused publication tests

Steps:

1. Extract a reusable deterministic `simulate_race` result without CSV side
   effects.
2. Preserve starting-grid provenance in dataframe metadata, not model columns.
3. Build ordered prediction summaries and the complete Head-to-Head matrix.
4. Validate immutable prediction archives and update `current.json` last.
5. Validate official result completeness and build comparison metrics/history.
6. Return machine-readable summaries with exact changed paths.
7. Implement schedule decisions at the exact 12-hour pre-race and 24-hour
   post-race boundaries.

## Task 5: GitHub Actions workflows

**Files:**

- Create: `.github/workflows/publish-prediction.yml`
- Create: `.github/workflows/post-race-pipeline.yml`
- Create: `docs/github-actions-publication-guide.md`
- Modify: `README.md`
- Reuse the repository's existing dependency declaration

### Prediction workflow

1. Add `workflow_dispatch` inputs for season, round, dry run, simulations, seed,
   and refresh.
2. Wake at minutes 17 and 47; let Python decide whether the next GP is within
   12 hours.
3. Run focused checks and the prediction exporter.
4. Allow only prediction JSON and explicitly reported `data_files/` updates.

### Post-race workflow

1. Add `workflow_dispatch` inputs for season, round, dry run, and refresh.
2. Wake daily at a non-peak UTC minute; let Python enforce the 24-hour boundary
   and official-result readiness.
3. Update the completed weekend data, publish actual JSON, run both full
   walk-forward trainers, validate, and promote the staged models.
4. Allow only `data_files/`, managed `models/`, calibration, and public history
   JSON.
5. Optionally upload ignored MLflow telemetry for diagnosis; never use it as
   production state.

### Shared workflow safety

1. Use Python 3.12 on `ubuntu-latest`, pip caching, a shared concurrency group,
   and `contents: write` only.
2. Parse and validate changed paths before staging each one explicitly.
3. Use the GitHub Actions bot identity and push `HEAD:main` without force,
   rebase, or a PAT.
4. Produce a readable job summary for `published`, `deferred`, `no-op`, and
   failure outcomes.

## Task 6: Verification and guided rollout

1. Run loader, lifecycle, simulator, scheduler, publication, and affected API
   focused tests.
2. Compile every modified Python entry point.
3. Inspect workflow YAML and the explicit Git allowlists.
4. Have the user commit and push the implementation.
5. Enable repository workflow write permission.
6. Run each workflow manually with dry-run enabled and inspect the steps
   together.
7. Run one controlled non-dry-run prediction, then one controlled post-race
   retraining.
8. Enable schedules only after the corresponding manual run succeeds.
9. Inspect the first automatic commits and verify they contain only reported
   paths.

## Focused verification commands

```powershell
.\venv\Scripts\python.exe -m unittest tests.test_ranker_model_loader tests.test_dnf_model_loader tests.test_model_lifecycle tests.test_monte_carlo tests.test_starting_grid tests.test_publication_prediction tests.test_publication_actual tests.test_publication_scheduler tests.test_publication_pipeline webapp.api.tests.test_api webapp.api.tests.test_repository webapp.api.tests.test_demo_data -v
.\venv\Scripts\python.exe -m py_compile src\ranker_model_loader.py src\dnf_model_loader.py src\model_lifecycle.py src\publication\scheduler.py src\publication\prediction.py src\publication\actual.py src\publication\publisher.py scripts\generate_web_result.py scripts\run_post_race_pipeline.py predict.py monte_carlo_simulator.py train_ranker_optimized.py train_dnf_optimized.py
```

These are development checks for the pipeline contract, not model-quality or
untouched backtest results.
