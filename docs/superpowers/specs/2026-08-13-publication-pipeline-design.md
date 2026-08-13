# PitWall Oracle Publication and Retraining Pipeline Design

**Date:** 2026-08-13

**Status:** Approved for implementation

**Related design:** [2026-08-08-web-app-design.md](2026-08-08-web-app-design.md)

This document supersedes the publication-lifecycle and GitHub Actions portions
of the related web-app design. Publication and history contracts identify an
event by season, round, and session type (`sprint` or `race`).

## Goals

The pipeline must:

- generate validated prediction and historical-comparison JSON without manual
  assembly;
- support manual runs and automatic runs;
- publish a live prediction during the first successful check in the 12 hours
  before every Sprint and GP;
- publish each completed-session comparison 24 hours after its start, retrying
  while FastF1 results are incomplete;
- retrain both the Ranker and DNF model after every completed GP using their
  existing full walk-forward procedures;
- version `data_files/` so a fresh runner has the complete training and causal
  feature state;
- replace current-season model artifacts only after the complete candidate set
  has passed validation;
- resolve the most recent promoted model that was already eligible before the
  requested race;
- use a GitHub-hosted Linux runner and commit validated results back to `main`;
- document the workflow so it can be operated and debugged without treating it
  as a black box.

## Non-goals

The first version will not:

- host MLflow remotely;
- use GitHub Actions cache or artifacts as the source of truth for training
  data or models;
- run HPO after every race (`RUN_HPO` remains disabled for routine retraining);
- alter the FIA starting-grid resolver;
- force-push, automatically rebase, or stage unrelated repository changes.

## Architecture

The implementation has five boundaries:

1. **Versioned data state** - `data_files/` contains the complete checked-in
   Bronze, Silver, Gold, history, and assembled training data used by the next
   workflow run.
2. **Model lifecycle** - training writes candidate artifacts outside the live
   model set, validates them, and only then promotes the complete set.
3. **Local model resolvers** - the existing Ranker and DNF loaders select
   season/round-named artifacts with one shared causal filename policy.
4. **Publication service** - builds and validates immutable prediction and
   completed-race JSON documents.
5. **GitHub Actions orchestration** - decides when work is due, runs Python,
   verifies the changed-path allowlist, and creates an explicit bot commit.

MLflow may run locally inside the training job because the current training
programs use it for telemetry. Its database and artifact directory are
ephemeral diagnostics, not an inference source. They may be uploaded as a
short-lived workflow artifact, but they are never committed.

## Repository layout

```text
data_files/                         # versioned training and causal data state
models/
  pitwall_oracle_<season>_base.json
  pitwall_oracle_<season>_<round>.json
  pitwall_oracle_latest.json
  dnf_logistic_<season>_base.joblib
  dnf_logistic_<season>_<round>.joblib
  dnf_logistic_latest.joblib
  monte_carlo_calibration.json
src/
  model_lifecycle.py
  publication/
    __init__.py
    scheduler.py
    prediction.py
    actual.py
    publisher.py
scripts/
  generate_web_result.py
  run_post_race_pipeline.py
.github/workflows/
  publish-prediction.yml
  post-race-pipeline.yml
docs/
  github-actions-publication-guide.md
tests/
  test_ranker_model_loader.py
  test_dnf_model_loader.py
  test_model_lifecycle.py
  test_publication_scheduler.py
  test_publication_pipeline.py
```

Public prediction and history documents remain under `webapp/api/data`.

## Versioned data state

The complete `data_files/` directory is committed. A workflow checkout
therefore starts from the exact same Parquet state as the previous successful
run. No second copy of `driver_team_history.parquet` is maintained; the
canonical path remains:

```text
data_files/gold/driver_team_history.parquet
```

Routine retraining uses the existing Gold and assembled Parquet files and only
downloads or calculates missing data for the new weekend. A forced feature
rebuild may refresh more files, but it remains an explicit manual operation.

GitHub cache is useful only for the Python package cache. It is not used as the
only copy of project data because cache entries may be evicted. Workflow
artifacts are similarly limited to optional MLflow diagnostics.

## Local model resolution and causal selection

The filenames are the inference contract:

```text
pitwall_oracle_<season>_base.json
pitwall_oracle_<season>_<selected_after_round>.json
dnf_logistic_<season>_base.joblib
dnf_logistic_<season>_<selected_after_round>.joblib
```

For a Sprint or GP prediction in round `N`, a promoted model is eligible only
when:

```text
artifact.season == requested season
artifact.selected_after_round < N
```

Each loader selects the matching artifact with the greatest promotion round. If
none is eligible, it selects the season-specific base. A promotion decided
using round `N` cannot be used to predict round `N`; it becomes valid from
round `N + 1`.

`predict.py`, `monte_carlo_simulator.py`, and the publication exporter call the
same two loader functions. Missing, corrupt, or causally ineligible artifacts
are hard errors. `latest` and `pending` aliases are never selection candidates,
and there is no inference-time MLflow lookup.

Promoted champions and season bases from previous seasons remain available so
historical races can be resolved. The existing unqualified base names are
accepted only as a transition for `NEW_YEAR`; the next successful training
creates season-qualified bases. A new training run replaces only the managed
artifacts for the current season and the `latest` aliases.

Monte Carlo sigma is not a model-selection registry. Training writes its
per-round causal calibration history to `monte_carlo_calibration.json`, and the
simulator selects the most recent calibration produced strictly before the
requested race.

## Atomic training lifecycle

The post-race orchestrator runs the existing Ranker and DNF full walk-forward
training against the updated dataset. Model output is redirected to a unique
temporary staging directory.

During training, `pending` is the challenger buffer used between consecutive
walk-forward rounds. It is not a production artifact and is not committed.

The staged candidate must contain:

- a loadable pre-season base model for each family;
- one artifact for every champion promotion reproduced by the current-season
  walk-forward;
- a loadable `latest` snapshot for each family;
- valid Monte Carlo calibration;
- a valid per-round Monte Carlo calibration file.

Only after both families and calibration validate does the orchestrator:

1. remove the old managed artifacts for the current season one file at a time;
2. move the staged current-season champion set, base aliases, and latest aliases
   into `models/`;
3. atomically replace `monte_carlo_calibration.json`;
4. discard staged `pending` files.

It never recursively deletes `models/`. If either training program, model load,
filename-policy check, or calibration validation fails, the previously
published model set and calibration remain unchanged.

## Publication operations

### Live prediction

The prediction exporter:

1. validates the event, qualifying data, and resolved starting grid;
2. resolves the causally eligible Ranker, DNF model, and sigma;
3. builds prediction features from the versioned history plus current weekend;
4. runs a deterministic Monte Carlo batch;
5. builds driver summaries and the complete Head-to-Head matrix;
6. validates the candidate through the public Pydantic schema;
7. creates the immutable round archive;
8. updates `predictions/current.json` last.

Prediction archives use `round-<NN>-sprint.json` and
`round-<NN>-race.json`. An existing session archive is never overwritten by
the normal workflow. Sprint points probability uses the top eight; GP points
probability uses the top ten.

### Completed session and retraining

The post-race operation:

1. requires the immutable prediction archive for the Sprint or GP;
2. verifies that complete FastF1 results are available for that session;
3. builds the actual-result comparison and updated history index;
4. refreshes all affected `data_files/` layers for the completed weekend;
5. for a GP only, runs the full Ranker and DNF walk-forward training into
   staging;
6. for a GP only, validates and promotes the complete model set and
   calibration;
7. validates the public JSON documents;
8. reports the exact files eligible for the bot commit.

Sprint publication never triggers training or promotion. Gold already loads
the completed Sprint while constructing the later GP prediction features, so
the public-history delay does not affect causal feature availability.

The Git commit is the externally visible transaction. A failed runner is
discarded, so no partial data, model, or JSON update reaches `main`. An
automatic run with incomplete official results reports `deferred` and changes
nothing; a later scheduled run retries.

## Scheduling

GitHub cron cannot calculate a trigger from a FastF1 event time. Each workflow
therefore wakes on a fixed UTC schedule and Python decides whether work is due.

### Prediction workflow

- wakes at minutes 17 and 47;
- selects every future Sprint or GP with no matching session archive and at
  most 12 hours remaining;
- publishes at the first successful eligible check;
- also exposes `workflow_dispatch` for a specific season, round, and session
  type.

### Post-race workflow

- wakes once per day at a non-peak minute;
- selects every completed Sprint or GP whose result has not been published and
  whose start was at least 24 hours earlier;
- retries daily until FastF1 supplies complete results;
- processes all due sessions in one run; if a GP result is newly published,
  retraining runs once after all JSON publications;
- also exposes `workflow_dispatch` for recovery and controlled backfills.

Both workflows use a shared concurrency group with
`cancel-in-progress: false`, so prediction and retraining cannot commit over one
another.

## GitHub Actions jobs

Both Linux jobs:

1. check out `main` with full enough history to push safely;
2. install Python 3.12 and restore the pip dependency cache;
3. install the locked repository dependencies;
4. run focused contract checks;
5. invoke one thin Python orchestration command;
6. write a machine-readable result and human-readable job summary;
7. reject unexpected changed paths;
8. configure `github-actions[bot]`, stage explicit allowed paths, commit, and
   push `HEAD:main` without force.

The prediction allowlist contains only public prediction JSON and current-
weekend `data_files/` changes required to build them. The post-race allowlist
contains `data_files/`, managed `models/`, calibration, and public history JSON.
Neither workflow uses `git add .` or `git add -A`.

Permissions are limited to `contents: write`. The repository setting must
allow workflows to write contents. The ephemeral `GITHUB_TOKEN` is sufficient;
no personal access token is required. A non-fast-forward push fails visibly and
is recovered by rerunning the workflow.

## Model and data retention

- All committed `data_files/` are retained and updated incrementally.
- Current-season transient or rejected challengers are absent from the final
  model set.
- `pending` exists only in training staging.
- Exactly one artifact is retained for each actual promotion point.
- Re-running the same walk-forward replaces the same deterministic promotion
  filenames rather than creating duplicate versions.
- Previous-season promoted champions remain available for historical
  resolution.
- Local MLflow databases, `mlruns/`, crash logs, and `results/` remain ignored.

## Error handling

Hard failures produce no commit when there is:

- incomplete or inconsistent model staging output;
- invalid model filename, season, causal round, or calibration history;
- no causally eligible Ranker, DNF model, or sigma;
- incomplete qualifying/grid input for an explicit prediction;
- invalid JSON or Head-to-Head data;
- an attempted immutable archive overwrite;
- a changed file outside the operation allowlist;
- a non-fast-forward push.

Incomplete official post-race data during an automatic run is an expected
`deferred` outcome. Network failures that cannot be identified as data latency
fail visibly.

## Verification

Focused tests cover:

- strict pre-race model eligibility for both local loaders;
- identical resolver behavior in both inference entry points;
- staging validation before model replacement;
- scoped current-season cleanup that preserves previous seasons and unrelated
  files;
- absence of committed pending models;
- scheduler boundaries at 12 hours before and 24 hours after a GP;
- publication immutability and pointer-last updates;
- incomplete-results deferral and retry;
- explicit changed-path allowlists;
- one deterministic low-simulation publication smoke run.

The final development verification uses focused tests and compilation/import
checks. A real post-race training workflow remains the model-quality evaluation;
publication checks are not reported as backtest results.

## Guided rollout

1. Build and inspect both historical model resolvers locally.
2. Run both workflows manually in dry-run mode and inspect every YAML step.
3. Enable `contents: write` and run one manual prediction publication.
4. Run one manual post-race retraining and inspect the data/model bot commit.
5. Enable the schedules and observe at least one `no-op` or `deferred` run.
6. Inspect the first automatic publication and retraining commits together.

`docs/github-actions-publication-guide.md` explains triggers, permissions,
inputs, steps, conditions, summaries, commits, failure recovery, and how the
fixed cron cooperates with FastF1 event times.

## Acceptance criteria

The pipeline is ready when:

1. `data_files/` is available in a fresh checkout and advances only in a
   successful bot commit.
2. The shared resolver returns the most recent causally eligible promoted model
   for current and historical races.
3. Failed training leaves the last valid model set and calibration unchanged.
4. Successful training leaves only base, actual promoted champions, and latest;
   no pending artifact is committed.
5. Manual and automatic prediction runs generate validated immutable JSON.
6. The day-after workflow publishes actual results, retrains both models, and
   commits only allowlisted data, models, calibration, and public JSON.
7. Scheduled runs safely report `no-op` or `deferred` when no work is ready.
8. The repository owner can explain and operate every workflow section using
   the guide.
