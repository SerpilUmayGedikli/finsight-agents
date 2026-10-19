from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel, Field

from apps.api.errors import NotFoundError
from memory.cli import ingest_tickers

router = APIRouter()
JobStatus = Literal["queued", "running", "done", "failed"]


class IngestRequest(BaseModel):
    tickers: list[str] = Field(min_length=1)
    form_type: str = "10-K"
    limit: int = Field(default=2, ge=1)


class IngestAccepted(BaseModel):
    job_id: str


class IngestJobStatus(BaseModel):
    job_id: str
    status: JobStatus
    progress: int
    error: str | None = None


def _run_ingest_job(
    jobs: dict[str, dict],
    job_id: str,
    tickers: list[str],
    form_type: str,
    limit: int,
) -> None:
    job = jobs[job_id]
    job["status"] = "running"

    def update_progress(completed: int, total: int) -> None:
        job["progress"] = round(completed * 100 / total) if total else 100

    try:
        ingest_tickers(
            tickers,
            form=form_type,
            limit=limit,
            progress_callback=update_progress,
        )
        job["status"] = "done"
        job["progress"] = 100
    except Exception as exc:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = str(exc)


@router.post("/ingest", response_model=IngestAccepted, status_code=202)
def ingest(
    payload: IngestRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> IngestAccepted:
    job_id = str(uuid4())
    jobs = request.app.state.ingest_jobs
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "error": None,
    }
    background_tasks.add_task(
        _run_ingest_job,
        jobs,
        job_id,
        payload.tickers,
        payload.form_type,
        payload.limit,
    )
    return IngestAccepted(job_id=job_id)


@router.get("/ingest/{job_id}", response_model=IngestJobStatus)
def get_ingest_job(job_id: str, request: Request) -> IngestJobStatus:
    job = request.app.state.ingest_jobs.get(job_id)
    if job is None:
        raise NotFoundError("Ingest job not found")
    return IngestJobStatus(**job)
