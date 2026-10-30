"""Asynchronous bulk registration jobs for named similarity indexes.

``POST /similarity/index/add`` is synchronous: the client holds a connection
open for the whole embedding run and gets no progress, no per-file failure
classification and no way to stop. That does not scale to the hundreds or
thousands of files a shared server is expected to ingest, so registration also
has a job form.

These endpoints are not demo-gated: bulk registration is a production feature.
"""
import logging
from typing import List, Optional

import core
import jobs
import index_registry
import storage
import jobstore
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/similarity", tags=["CAD Similarity Search"])

_MAX_FILE_IDS_PER_JOB = 100000


class IndexAddJobRequest(BaseModel):
    file_ids: Optional[List[str]] = Field(
        None, description="file_ids of CAD files already in the server's CAD store."
    )
    zip_file_id: Optional[str] = Field(
        None, description="file_id of an already-uploaded ZIP archive to expand."
    )
    server_paths: Optional[List[str]] = Field(
        None,
        description=(
            "Paths on the server machine. Administrator seeding only; returns "
            "403 unless HOOPS_AI_ALLOW_SERVER_PATHS is enabled."
        ),
    )
    workers: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Parallel embedding workers. Omit to size automatically from CPU "
            "cores and free RAM (capped by HOOPS_AI_MAX_WORKERS, default 12). "
            "For a retry job over heavy assemblies, use a small number."
        ),
    )
    time_limit: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Per-file embedding budget in seconds. Omit to use "
            "HOOPS_AI_EMBED_TIME_LIMIT, or the SDK's own 120 s when that is "
            "unset. Files that exceed it are reported with retryable=true; "
            "resubmit them as a second job with a longer budget."
        ),
    )


class JobProgress(BaseModel):
    # phase is "embedding" while the batch runs, "heavy" while the SDK works
    # through its single-worker RAM fallback, then "done". done/errors/heavy
    # advance live: they are parsed from the SDK's own progress bar.
    phase: str
    done: int
    total: int
    errors: int
    heavy: int = 0


class JobAcceptedResponse(BaseModel):
    job_id: str
    status: str
    accepted: int
    errors: list[dict]


class JobStatusResponse(BaseModel):
    job_id: str
    kind: str
    index_name: Optional[str] = None
    status: str
    progress: JobProgress
    summary: dict
    errors: list[dict]
    errors_truncated: bool = False
    timings: dict
    detail: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    cancel_requested: bool = False
    report_url: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]
    count: int
    total: int


def _public_job(record: dict) -> JobStatusResponse:
    """Project a stored record onto the response schema.

    Explicitly field-by-field so internal bookkeeping can be added to the
    record later without leaking into the API.
    """
    progress = record.get("progress") or {}
    summary = record.get("summary") or {}
    # Only advertise the report once the job has actually written one.
    report_url = record.get("report_url")
    if report_url is None and summary.get("report") and record.get("index_name"):
        report_url = (
            f"/similarity/index/{record['index_name']}/jobs/{record['job_id']}/report"
        )
    return JobStatusResponse(
        job_id=record["job_id"],
        kind=record.get("kind", ""),
        index_name=record.get("index_name"),
        status=record.get("status", ""),
        progress=JobProgress(
            phase=str(progress.get("phase", "")),
            done=int(progress.get("done", 0)),
            total=int(progress.get("total", 0)),
            errors=int(progress.get("errors", 0)),
            heavy=int(progress.get("heavy", 0)),
        ),
        summary=record.get("summary") or {},
        errors=record.get("errors") or [],
        errors_truncated=bool(record.get("errors_truncated", False)),
        timings=record.get("timings") or {},
        detail=record.get("detail"),
        created_at=record.get("created_at"),
        started_at=record.get("started_at"),
        finished_at=record.get("finished_at"),
        cancel_requested=bool(record.get("cancel_requested", False)),
        report_url=report_url,
    )


def _validate_name_or_raise(name: str) -> None:
    try:
        core._validate_index_name(name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _require_writable_index(name: str) -> None:
    """Reject unknown and legacy-schema indexes before a job is created.

    Doing this up front means a client never gets a 202 for a job that is
    guaranteed to fail on its first batch.
    """
    if not index_registry._index_faiss_path(name).exists():
        raise HTTPException(status_code=404, detail=f"Index '{name}' not found.")
    try:
        schema = index_registry._load_index_schema(name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read index '{name}': {exc}")
    if schema["schema_version"] < core._INDEX_SCHEMA_VERSION:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Index '{name}' uses legacy schema v{schema['schema_version']} "
                f"and must be rebuilt as schema v{core._INDEX_SCHEMA_VERSION} "
                "before adding parts."
            ),
        )


def _resolve_inputs(payload: IndexAddJobRequest) -> tuple[list[str], list[dict]]:
    """Turn the request's three input forms into a list of file_ids.

    Every rejected input produces an error entry, so the count of accepted
    files plus the count of errors always equals the number of inputs.
    """
    file_ids: list[str] = []
    errors: list[dict] = []

    for fid in payload.file_ids or []:
        fid = (fid or "").strip()
        if not fid:
            continue
        try:
            storage.find_persistent_CAD_file(fid)
            file_ids.append(fid)
        except RuntimeError as exc:
            errors.append({"file_id": fid, "detail": str(exc)})

    if payload.zip_file_id:
        try:
            zip_path = storage.find_persistent_CAD_file(payload.zip_file_id.strip())
            resolved, zip_errors = core.extract_zip_cad_files(zip_path.read_bytes())
            file_ids.extend(fid for fid, _ in resolved)
            errors.extend(zip_errors)
        except RuntimeError as exc:
            errors.append({"file_id": payload.zip_file_id, "detail": str(exc)})
        except core.ZipSlipError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except (core.ZipSizeLimitError, core.ZipFileLimitError) as exc:
            raise HTTPException(status_code=413, detail=str(exc))
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to process ZIP archive: {exc}"
            ) from exc

    if payload.server_paths:
        try:
            resolved, path_errors = core.resolve_server_paths(payload.server_paths)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        file_ids.extend(resolved)
        errors.extend(path_errors)

    return file_ids, errors


@router.post(
    "/index/{name}/jobs/add",
    response_model=JobAcceptedResponse,
    status_code=202,
)
def create_index_add_job(
    name: str,
    payload: IndexAddJobRequest = Body(...),
):
    """Queue a bulk registration job and return immediately.

    Supply one or more of ``file_ids`` (the standard route: upload first with
    ``POST /files/upload-batch``, skipping what ``POST /files/exists`` already
    knows), ``zip_file_id``, or ``server_paths``.

    Poll ``GET /similarity/index/{name}/jobs/{job_id}`` for progress. Jobs run
    one at a time by default, so a job may sit in ``queued`` for a while.

    One job is one embedding pass. To register a large corpus efficiently, run
    the documented two-pass client loop: submit everything with many ``workers``
    and a short ``time_limit``, then resubmit the failures whose ``retryable``
    flag is true as a second job with few workers and a long ``time_limit``.
    Failures that are not retryable are deterministic CAD errors and will never
    succeed, however long the budget. When the job finishes, ``report_url``
    points at a plain-text breakdown.

    **Cancellation only takes effect before a job starts embedding.** The whole
    input goes to one ``embed_shape_batch`` call, which cannot be interrupted,
    so cancelling a running job leaves it to finish. Split a large corpus into
    several jobs to keep both the commit granularity and the cancellation
    granularity under your control.
    """
    _validate_name_or_raise(name)
    _require_writable_index(name)

    file_ids, errors = _resolve_inputs(payload)

    # Drop duplicates but keep order: re-embedding the same file inside one job
    # is pure waste, and the second write would only overwrite the first.
    seen: set[str] = set()
    unique_ids: list[str] = []
    duplicates = 0
    for fid in file_ids:
        if fid in seen:
            duplicates += 1
            continue
        seen.add(fid)
        unique_ids.append(fid)

    if not unique_ids:
        raise HTTPException(
            status_code=422,
            detail=(
                "No registrable files resolved. Supply file_ids, zip_file_id or "
                f"server_paths. Rejected inputs: {len(errors)}."
            ),
        )
    if len(unique_ids) > _MAX_FILE_IDS_PER_JOB:
        raise HTTPException(
            status_code=422,
            detail=f"A job accepts at most {_MAX_FILE_IDS_PER_JOB} files.",
        )

    store = jobs.get_index_job_store()
    record = store.create(
        kind=core.JOB_KIND_INDEX_ADD,
        params={
            "files": len(unique_ids),
            "duplicates_dropped": duplicates,
            "workers": payload.workers,
            "time_limit": payload.time_limit,
        },
        index_name=name,
    )
    if errors:
        store.append_errors(record["job_id"], errors)
        store.update(record["job_id"], progress={"errors": len(errors)})

    store.submit(
        record["job_id"],
        lambda ctx: core.run_index_add_job(
            ctx,
            name,
            unique_ids,
            num_workers=payload.workers,
            time_limit=payload.time_limit,
        ),
    )
    logger.info(
        "[JOB %s] queued: index '%s', %d file(s), %d rejected",
        record["job_id"], name, len(unique_ids), len(errors),
    )
    return JobAcceptedResponse(
        job_id=record["job_id"],
        status=record["status"],
        accepted=len(unique_ids),
        errors=errors,
    )


class EmbeddingPassPlan(BaseModel):
    workers: int
    time_limit: int
    basis: dict


class EmbeddingPlanResponse(BaseModel):
    file_count: Optional[int] = None
    first_pass: EmbeddingPassPlan
    retry_pass: EmbeddingPassPlan


@router.get("/embedding-plan", response_model=EmbeddingPlanResponse)
def get_embedding_plan(
    file_count: Optional[int] = Query(
        None, ge=0, le=_MAX_FILE_IDS_PER_JOB,
        description=(
            "How many files the run would submit. Omit it to plan for the host "
            "alone, before a selection exists."
        ),
    ),
):
    """Resolve the worker count and per-file budget a job would use.

    Exactly what ``POST /similarity/index/{name}/jobs/add`` does when ``workers``
    and ``time_limit`` are omitted, answered before the job is created so a UI
    can show the real numbers rather than the word "auto". Also resolves the
    retry pass -- fewer workers, a longer budget -- which is the documented
    second half of the registration loop.

    ``file_count`` is optional because it is the only input that depends on a
    selection, and it can only lower the answer. Omitted, the reply is the
    ceiling the cap, the cores and the free RAM allow, and ``file_count`` and
    ``basis.files`` come back ``null`` to say the count was not in the
    minimum -- which lets a dialog fill its fields the moment it opens and ask
    again once a folder is chosen.

    It lives here, beside the job endpoint whose two optional fields it is
    answering for, rather than in ``/client-config``: that is a one-shot startup
    document, while this depends on the selection and is re-asked as it changes.
    It takes no index name because none of it depends on an index -- the numbers
    come from the host's cores and free RAM, the embedding knobs, and the file
    count.

    Read-only and side-effect free: it resolves nothing, loads no index and
    creates no job. ``time_limit`` is always the effective budget, never the
    raw ``0`` that means "leave the SDK's own default alone".
    """
    return EmbeddingPlanResponse(**core.plan_embedding_run(file_count))


@router.get("/index/{name}/jobs", response_model=JobListResponse)
def list_index_jobs(
    name: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List registration jobs for an index, newest first."""
    _validate_name_or_raise(name)
    page, total = jobs.get_index_job_store().list(
        kind=core.JOB_KIND_INDEX_ADD, index_name=name, limit=limit, offset=offset
    )
    job_list = [_public_job(record) for record in page]
    return JobListResponse(jobs=job_list, count=len(job_list), total=total)


@router.get("/index/{name}/jobs/{job_id}", response_model=JobStatusResponse)
def get_index_job(name: str, job_id: str):
    """Return the current state of a registration job."""
    _validate_name_or_raise(name)
    record = _load_job_or_404(name, job_id)
    return _public_job(record)


@router.get(
    "/index/{name}/jobs/{job_id}/report",
    response_class=PlainTextResponse,
    responses={200: {"content": {"text/plain": {}}}},
)
def get_index_job_report(name: str, job_id: str):
    """Return the job's registration report as plain text.

    Three groups: files embedded into the index (ADDED), files that are not in
    the index with their reason and whether they are worth resubmitting
    (FAILED), and files hoops_ai deferred to its single-worker RAM fallback
    (HEAVY-FLAGGED).

    Returns 404 until the job has finished and written one.
    """
    _validate_name_or_raise(name)
    _load_job_or_404(name, job_id)
    try:
        path = jobs.get_index_job_store().artifacts_dir(job_id) / "report.txt"
        text = path.read_text(encoding="utf-8")
    except (OSError, jobstore.JobNotFound) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"No report available for job '{job_id}'.",
        ) from exc
    return PlainTextResponse(text)


@router.delete("/index/{name}/jobs/{job_id}", response_model=JobStatusResponse)
def cancel_index_job(name: str, job_id: str):
    """Request cancellation of a registration job.

    A queued job is canceled at once. A running job stops after the batch
    currently inside ``embed_shape_batch`` finishes, so the response may still
    report ``running``; poll the status endpoint to see it settle. Files
    already registered stay registered.
    """
    _validate_name_or_raise(name)
    _load_job_or_404(name, job_id)
    record = jobs.get_index_job_store().request_cancel(job_id)
    return _public_job(record)


def _load_job_or_404(name: str, job_id: str) -> dict:
    try:
        record = jobs.get_index_job_store().get(job_id)
    except jobstore.JobNotFound as exc:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.") from exc
    if record.get("index_name") != name:
        raise HTTPException(
            status_code=404, detail=f"Job '{job_id}' does not belong to index '{name}'."
        )
    return record
