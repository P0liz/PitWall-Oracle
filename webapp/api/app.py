"""Read-only HTTP API for published PitWall Oracle results."""

import os
from pathlib import Path
from typing import NoReturn

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .repository import (
    InvalidPublishedData,
    ResultNotFound,
    ResultsRepository,
)
from .schemas import (
    HeadToHeadResponse,
    HistoryDocument,
    HistoryIndex,
    PredictionDocument,
)


repository = ResultsRepository(Path(__file__).parent / "data")

app = FastAPI(
    title="PitWall Oracle API",
    version="1.0.0",
    description="Read-only Formula 1 prediction results.",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("PITWALL_ALLOWED_ORIGINS", "http://localhost:8501").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def get_repository() -> ResultsRepository:
    return repository


def translate_repository_error(error: Exception) -> NoReturn:
    if isinstance(error, ResultNotFound):
        raise HTTPException(
            status_code=404,
            detail={"code": error.code, "message": error.message},
        ) from error
    if isinstance(error, InvalidPublishedData):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_published_data", "message": str(error)},
        ) from error
    raise error


@app.get("/api/v1/health", response_model=dict[str, str])
def health() -> dict[str, str]:
    return {"status": "ok", "schema_version": "1.0"}


@app.get("/api/v1/predictions/current", response_model=PredictionDocument)
def get_current_prediction(
    results_repository: ResultsRepository = Depends(get_repository),
) -> PredictionDocument:
    try:
        return results_repository.load_current()
    except (ResultNotFound, InvalidPublishedData) as error:
        translate_repository_error(error)


@app.get(
    "/api/v1/predictions/current/head-to-head",
    response_model=HeadToHeadResponse,
)
def get_head_to_head(
    driver_a: str,
    driver_b: str,
    results_repository: ResultsRepository = Depends(get_repository),
) -> HeadToHeadResponse:
    if driver_a == driver_b:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "duplicate_drivers",
                "message": "A driver cannot be compared with themselves",
            },
        )
    try:
        return results_repository.load_head_to_head(driver_a, driver_b)
    except (ResultNotFound, InvalidPublishedData) as error:
        translate_repository_error(error)


@app.get("/api/v1/history", response_model=HistoryIndex)
def get_history(
    season: int,
    results_repository: ResultsRepository = Depends(get_repository),
) -> HistoryIndex:
    try:
        return results_repository.list_history(season)
    except (ResultNotFound, InvalidPublishedData) as error:
        translate_repository_error(error)


@app.get("/api/v1/history/{season}/{round_number}", response_model=HistoryDocument)
def get_history_race(
    season: int,
    round_number: int,
    results_repository: ResultsRepository = Depends(get_repository),
) -> HistoryDocument:
    try:
        return results_repository.load_history(season, round_number)
    except (ResultNotFound, InvalidPublishedData) as error:
        translate_repository_error(error)
