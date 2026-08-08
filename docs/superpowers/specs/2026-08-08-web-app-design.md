# PitWall Oracle Web App — Design

**Date:** 2026-08-08
**Status:** Approved
**Scope:** Public portfolio MVP for the 2026 Formula 1 season and later

## Summary

PitWall Oracle will expose precomputed race predictions through a public FastAPI API hosted on Vercel and a Streamlit UI hosted on Streamlit Community Cloud. GitHub Actions will run the existing model and Monte Carlo code after qualifying, validate the output, and commit versioned JSON results to the repository.

The web services will never train the model, build Gold data, or run simulations. They will only read validated JSON. This keeps the public application fast, inexpensive, reproducible, and suitable for free hosting.

The product has two deliberately separate experiences:

1. **Current prediction:** the next race for which a post-qualifying prediction has been published, including Head-to-Head comparisons.
2. **History:** completed races with a user-friendly comparison between the original prediction and the real result.

Historical backtests are allowed, but must be labelled clearly and must never appear as live pre-race predictions.

## Goals

- Present the predicted finishing order and win, podium, points, DNF, and expected-position probabilities.
- Let a visitor compare any two drivers through a precomputed Head-to-Head probability.
- Keep the current prediction separate from completed-race comparisons.
- Preserve every published prediction as an immutable, reviewable repository artifact.
- Explain model performance in language understandable without an ML background.
- Deploy automatically from a public GitHub repository using free hosting tiers.
- Give recruiters clear entry points for exploring the architecture and implementation.

## Non-goals

- Running inference, Monte Carlo simulation, training, or FastF1 ingestion in Vercel or Streamlit.
- Supporting seasons before 2026.
- Authentication, user accounts, editing, or administrative controls in the web app.
- A database or external object store for the MVP.
- Automatic detection of qualifying completion in the first release.
- Replacing the existing ranker, DNF, Monte Carlo, or MLflow workflows.

## Architecture

```text
Existing model and Monte Carlo code
                 |
                 v
       GitHub Action (manual MVP)
       - publish prediction
       - publish actual result
       - historical backtest
                 |
                 v
       Validated, versioned JSON
                 |
                 v
         FastAPI on Vercel
                 |
                 v
 Streamlit Community Cloud UI
                 |
                 v
              Visitor
```

### Component responsibilities

**Prediction generator**

- Reuses the existing causal feature pipeline, champion/race-specific ranker, DNF strategy, and Monte Carlo functions.
- Converts simulator output into the public JSON schema.
- Precomputes the complete pairwise Head-to-Head matrix from the shared simulations.
- Does not alter model-training behavior.

**GitHub Action**

- Provides an initial manual `workflow_dispatch` interface.
- Validates all generated data before changing published files.
- Commits only complete, valid results.
- Prevents concurrent publication runs for the same race.

**FastAPI**

- Is a read-only adapter over versioned JSON files.
- Validates stored data with Pydantic response models.
- Exposes stable, versioned endpoints and machine-readable errors.
- Contains no dependency on XGBoost, FastF1, pandas, MLflow, or the simulator.

**Streamlit**

- Calls only the public FastAPI endpoints.
- Presents percentages and comparison summaries in plain language.
- Does not read model artifacts or repository JSON directly.
- Keeps the current prediction and historical comparison on separate pages.

## Proposed repository layout

```text
webapp/
  api/
    app.py
    repository.py
    schemas.py
    requirements.txt
    data/
      predictions/
        current.json
        2026/
          round-01.json
      history/
        2026/
          index.json
          round-01.json
  ui/
    streamlit_app.py
    api_client.py
    requirements.txt
scripts/
  generate_web_result.py
.github/workflows/
  publish-race-result.yml
docs/superpowers/specs/
  2026-08-08-web-app-design.md
```

The Vercel project root will be `webapp`. The FastAPI entry point will be `webapp/api/app.py` and the Streamlit entry point will be `webapp/ui/streamlit_app.py`. Each deployed component will have a minimal dependency file so the public services do not install the ML training stack.

## Publication lifecycle

### Publish a current prediction

The workflow accepts:

- `season`, restricted to 2026 or later;
- `round`;
- operation `publish-prediction`;
- DNF strategy;
- simulation count and deterministic seed.

The job performs these steps:

1. Confirm that the race, required model artifact, qualifying data, and grid data exist.
2. Build prediction features using only data available at the recorded cutoff.
3. Run ranker inference, DNF probability calculation, and Monte Carlo simulation.
4. Produce driver summaries and the complete Head-to-Head matrix.
5. Validate the candidate JSON.
6. Write the immutable archive file under `predictions/<season>/round-<round>.json`.
7. Update `predictions/current.json` only after the archive file is valid.
8. Commit both files so Vercel and Streamlit redeploy from the successful publication.

The operation fails if the archived prediction already exists. Correcting an exceptional publication error requires an explicit repository change whose history remains visible; the normal workflow never overwrites it.

### Publish a completed-race comparison

The operation `publish-actual`:

1. Requires an existing immutable prediction archive.
2. Loads the official race result.
3. Calculates per-driver position differences and friendly summary metrics.
4. Creates a separate file under `history/<season>/round-<round>.json`.
5. Updates the season history index.
6. Never changes the archived prediction.

Publishing the actual result does not automatically remove `current.json`. The next successful post-qualifying prediction replaces the current pointer. If the pointed race is already complete, the current endpoint reports that no upcoming prediction is available instead of showing stale content.

### Historical backfill

The operation `historical-backtest` is available only for completed races. It must use the race-specific, temporally correct model and causal data cutoff. Its archive metadata uses `publication_type: "backtest"`; the UI displays this label prominently. A missing time-correct model or unavailable causal input causes the race to be skipped rather than evaluated with the latest model.

## Data contracts

### Prediction archive

Each prediction archive contains:

- `schema_version`;
- race identity, display name, circuit, start time, season, and round;
- publication type: `live` or `backtest`;
- generation time and data cutoff;
- model artifact name, DNF strategy, simulation count, and seed;
- ordered driver results;
- complete Head-to-Head matrix.

Each driver result contains:

- stable driver ID;
- display name and abbreviation;
- stable team ID and display name;
- predicted position;
- expected position;
- win, podium, points, DNF, and finish probabilities.

Raw XGBoost scores may be retained as internal provenance but are not part of the default UI.

The Head-to-Head matrix stores `P(driver A finishes ahead of driver B)` for every distinct pair. Both directions must be present and sum to approximately one. The diagonal is omitted. The matrix is derived from the same Monte Carlo simulations used for the race summary; individual simulation arrays are not stored.

### Current prediction pointer

`predictions/current.json` contains only the season, round, and relative path of the active archive. FastAPI resolves and validates the target. An absent pointer, invalid target, or already-completed pointed race produces a `404 prediction_not_available` response.

### Historical comparison

Each history file contains:

- race identity;
- a reference to the immutable prediction archive;
- prediction type, preserving the live/backtest distinction;
- official results with classified position and status;
- per-driver predicted position, actual position, and signed difference;
- mean absolute position error;
- podium hits out of three;
- top-five hits out of five.

The API may expose technical metric names, but the UI translates them into sentences such as “errore medio di 2,1 posizioni” and “4 piloti della top 5 individuati”. DNF, DNS, DSQ, and unclassified outcomes retain their official status and are not presented as ordinary numeric finishing errors without an explicit rule in the result schema.

## API design

The public API uses the `/api/v1` prefix.

- `GET /api/v1/health` — service and data-schema status.
- `GET /api/v1/predictions/current` — complete current prediction.
- `GET /api/v1/predictions/current/head-to-head?driver_a=<id>&driver_b=<id>` — one pairwise comparison.
- `GET /api/v1/history?season=2026` — completed races available for the season.
- `GET /api/v1/history/{season}/{round}` — prediction-versus-result comparison for one race.

The Head-to-Head endpoint returns both probabilities and display metadata, so Streamlit does not need to join driver records itself.

Expected error codes include:

- `400 invalid_request` for invalid seasons, rounds, or duplicate driver selections;
- `404 prediction_not_available`, `race_not_found`, or `driver_not_found`;
- `422 invalid_published_data` if a committed result violates the schema;
- `503 data_unavailable` for a temporary repository/data-loading problem.

CORS allows the configured Streamlit deployment origin and localhost development origins. The API remains publicly callable without authentication.

## User experience

### Current prediction page

The default page shows:

- Grand Prix name, circuit, date, data cutoff, and a “generated after qualifying” notice;
- predicted finishing order;
- driver and team names;
- win, podium, points, DNF, and expected-position values;
- a dedicated Head-to-Head section with two driver selectors and a simple percentage bar.

Raw model scores and unexplained ML metrics are hidden from the primary view. If no current prediction exists, the page states that it will be available after qualifying and never falls back to the previous race.

### History page

The second page shows only completed races. It provides:

- a race selector;
- an obvious `LIVE PREDICTION` or `HISTORICAL BACKTEST` badge;
- plain-language summary cards;
- a single table with predicted position, driver, actual position, and difference;
- arrows and restrained colour cues for better- or worse-than-predicted results;
- an optional collapsed section for technical provenance.

The navigation contains only “Prossima gara” and “Storico”. A short “Come funziona” explanation lives in the sidebar.

## Error handling and integrity

Before publication, the generator validates:

- one unique row per driver;
- required race and provenance metadata;
- probabilities in the inclusive `[0, 1]` interval;
- unique predicted positions;
- a complete Head-to-Head matrix;
- complementary Head-to-Head directions within a documented floating-point tolerance;
- valid references between current, prediction, and history files;
- no attempt to overwrite an archived prediction.

Candidate files are generated separately and moved into their published paths only after validation. `current.json` is always the last file updated. A failed workflow therefore leaves the last valid publication intact.

FastAPI returns structured errors. Streamlit converts them to friendly messages and provides a retry action for transient API failures. It distinguishes “not published yet” from service failure.

## Verification strategy

Verification is intentionally focused and does not duplicate the model evaluation suite.

- Schema validation tests for prediction and history fixtures.
- API tests for success, missing prediction, missing race, invalid drivers, and corrupt published data.
- Head-to-Head tests for completeness, complementarity, and correct lookup.
- Publication test proving an archived prediction cannot be overwritten.
- Historical test proving actual-result publication leaves the prediction bytes unchanged.
- One generator smoke check using a small deterministic simulation count.
- One local Streamlit smoke check against the local FastAPI service.
- One GitHub Actions dry run that validates output without committing it.

These are development checks. Ranker quality, DNF calibration, and Monte Carlo evaluation remain governed by the existing ML workflows and are not redefined by the web app.

## Deployment

- Vercel deploys FastAPI from the public GitHub repository using the Hobby plan and the `webapp` root.
- Streamlit Community Cloud deploys `webapp/ui/streamlit_app.py` from the same repository.
- Streamlit receives the API base URL through deployment configuration.
- GitHub Actions uses the repository-provided token with narrowly scoped `contents: write` permission only for the publication job.
- No application secrets or persistent runtime filesystem are required for the MVP.

## README requirement

The root README receives a concise web-app section rather than duplicating this specification. It will contain:

- a short description of the public application;
- the model-to-publication flow in one compact diagram or paragraph;
- deployed UI and API links once available;
- minimal local start commands;
- links to this design, `webapp/api`, `webapp/ui/streamlit_app.py`, the publication workflow, and the public result schema for readers who want implementation detail.

## Acceptance criteria

The MVP is complete when:

1. A visitor can open the Streamlit app without authentication.
2. The current page either shows the post-qualifying prediction or clearly says it is not available.
3. The visitor can compare any two current-race drivers without starting a simulation.
4. The history page clearly separates live predictions from backtests and compares predicted and real results in plain language.
5. FastAPI serves all data from validated repository JSON and exposes no model execution endpoint.
6. Publishing an actual result cannot modify its archived prediction.
7. The GitHub Action can validate a run without publishing and can publish a valid result through an intentional manual run.
8. The README explains the architecture briefly and points to the detailed implementation locations.
