"""Persistent, bounded job store for long-running server work.

Written for Phase 3 bulk registration, but deliberately generic so other
long-running endpoints can adopt it later. It exists because a naive in-memory
job store (a bare dict guarded by a lock) meets none of the requirements of
a shared server:

* it is memory-only, so a restart loses every job and its result;
* nothing is ever evicted, so the process leaks one entry per job forever;
* concurrency is unbounded, so N clients start N embedding runs at once.

This store fixes all three: records are JSON files under ``jobs/``, finished
records are swept by TTL and capped in number, and execution goes through a
bounded thread pool.

Concurrency defaults to **1**. hoops_ai writes ``error_summary.json`` to the
process working directory and offers no way to redirect it -- ``flowdir`` is a
Flow-API attribute that ``embed_shape_batch`` never populates, and there is no
environment override (verified against the compiled SDK). Two registration jobs
running at once would therefore read each other's failure summary. Raise
HOOPS_AI_JOB_MAX_CONCURRENCY only for job kinds that do not read that file.
"""
import json
import logging
import os
import pathlib
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELED = "canceled"

TERMINAL_STATUSES = frozenset({STATUS_DONE, STATUS_FAILED, STATUS_CANCELED})


class JobCanceled(Exception):
    """Raised inside a job body when a cancellation request is observed."""


class JobNotFound(KeyError):
    """Raised when a job id is unknown to the store."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class JobContext:
    """Handle a running job uses to report progress and observe cancellation.

    Every mutator persists the record, so a client polling the status endpoint
    sees progress even though the job runs in another thread, and a crash
    leaves the last reported state on disk rather than nothing.
    """

    def __init__(self, store: "JobStore", job_id: str) -> None:
        self.store = store
        self.job_id = job_id

    def check_canceled(self) -> None:
        """Raise :class:`JobCanceled` if cancellation has been requested."""
        if self.store.is_cancel_requested(self.job_id):
            raise JobCanceled(f"Job '{self.job_id}' was canceled.")

    def set_phase(self, phase: str) -> None:
        self.store.update(self.job_id, progress={"phase": phase})

    def set_progress(self, **fields: Any) -> None:
        """Merge arbitrary counters into the progress section."""
        self.store.update(self.job_id, progress=fields)

    def set_total(self, total: int) -> None:
        self.store.update(self.job_id, progress={"total": int(total)})

    def advance(self, done: int = 0, errors: int = 0) -> None:
        self.store.increment_progress(self.job_id, done=done, errors=errors)

    def set_summary(self, **fields: Any) -> None:
        self.store.update(self.job_id, summary=fields)

    def set_timings(self, **fields: float) -> None:
        self.store.update(self.job_id, timings=fields)

    def add_errors(self, errors: list[dict]) -> None:
        self.store.append_errors(self.job_id, errors)


class JobStore:
    """A file-backed job store with a bounded worker pool.

    *root* is the directory holding ``<job_id>.json``. *max_workers* caps how
    many jobs run at once; the rest sit in ``queued``. *ttl_seconds* and
    *max_records* bound how long finished records are kept.
    """

    def __init__(
        self,
        root: pathlib.Path,
        max_workers: int = 1,
        ttl_seconds: int = 7 * 86400,
        max_records: int = 1000,
        max_errors_per_job: int = 1000,
    ) -> None:
        self.root = pathlib.Path(root)
        self.max_workers = max(1, int(max_workers))
        self.ttl_seconds = int(ttl_seconds)
        self.max_records = max(1, int(max_records))
        self.max_errors_per_job = max(1, int(max_errors_per_job))
        self._lock = threading.RLock()
        self._cancel_requested: set[str] = set()
        self._executor: Optional[ThreadPoolExecutor] = None

    # -- storage -----------------------------------------------------------

    def _path(self, job_id: str) -> pathlib.Path:
        if not job_id or "/" in job_id or "\\" in job_id or job_id.startswith("."):
            raise JobNotFound(f"Invalid job id '{job_id}'.")
        return self.root / f"{job_id}.json"

    def artifacts_dir(self, job_id: str) -> pathlib.Path:
        """Directory for a job's on-disk side files (logs, reports).

        Kept next to the record as ``<root>/<job_id>/`` so that garbage
        collecting the record also removes its artefacts; nothing else may write
        into ``root`` with that name because job ids are UUIDs.
        """
        self._path(job_id)  # validates the id
        return self.root / job_id

    def _remove_record(self, path: pathlib.Path) -> None:
        """Delete a record file together with its artefacts directory."""
        path.unlink(missing_ok=True)
        artifacts = path.with_suffix("")
        if artifacts.is_dir():
            shutil.rmtree(artifacts, ignore_errors=True)

    def _write(self, record: dict) -> None:
        """Persist *record* atomically so a crash never leaves half a record."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(record["job_id"])
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _load(self, path: pathlib.Path) -> dict:
        """Read one record file, retrying past transient sharing violations.

        On Windows the atomic ``os.replace`` in :meth:`_write` briefly makes the
        target unopenable, so a client polling a running job can hit
        ``PermissionError`` on a perfectly healthy record. Retrying turns that
        into a short wait instead of a spurious 404 -- and, more importantly,
        stops :meth:`collect_garbage` from mistaking a live record for a corrupt
        one and deleting it.
        """
        last: Optional[OSError] = None
        for attempt in range(5):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except FileNotFoundError:
                raise
            except OSError as exc:
                last = exc
                time.sleep(0.01 * (attempt + 1))
        raise last  # type: ignore[misc]

    def _read(self, job_id: str) -> Optional[dict]:
        path = self._path(job_id)
        # Under the store lock, so an in-process writer is never observed
        # mid-update. The retry in _load covers other processes and Windows.
        with self._lock:
            try:
                return self._load(path)
            except FileNotFoundError:
                return None
            except (OSError, ValueError) as exc:
                logger.warning("Unreadable job record %s: %s", path, exc)
                return None

    # -- lifecycle ---------------------------------------------------------

    def create(self, kind: str, params: Optional[dict] = None, **extra: Any) -> dict:
        """Create a ``queued`` record and return it."""
        job_id = uuid.uuid4().hex
        record = {
            "job_id": job_id,
            "kind": kind,
            "status": STATUS_QUEUED,
            "params": params or {},
            "progress": {"phase": "queued", "done": 0, "total": 0, "errors": 0},
            "summary": {},
            "errors": [],
            "errors_truncated": False,
            "timings": {},
            "detail": None,
            "created_at": _utc_now(),
            "started_at": None,
            "finished_at": None,
            **extra,
        }
        with self._lock:
            self._write(record)
        self.collect_garbage()
        return record

    def get(self, job_id: str) -> dict:
        record = self._read(job_id)
        if record is None:
            raise JobNotFound(f"Job '{job_id}' not found.")
        return record

    def update(self, job_id: str, **sections: Any) -> dict:
        """Merge *sections* into the record. Dict values merge key-by-key."""
        with self._lock:
            record = self.get(job_id)
            for key, value in sections.items():
                if isinstance(value, dict) and isinstance(record.get(key), dict):
                    record[key].update(value)
                else:
                    record[key] = value
            self._write(record)
            return record

    def increment_progress(self, job_id: str, done: int = 0, errors: int = 0) -> None:
        with self._lock:
            record = self.get(job_id)
            progress = record["progress"]
            progress["done"] = int(progress.get("done", 0)) + int(done)
            progress["errors"] = int(progress.get("errors", 0)) + int(errors)
            self._write(record)

    def append_errors(self, job_id: str, errors: list[dict]) -> None:
        """Append per-item errors, capped so one bad batch cannot blow up disk."""
        if not errors:
            return
        with self._lock:
            record = self.get(job_id)
            room = self.max_errors_per_job - len(record["errors"])
            if room <= 0:
                record["errors_truncated"] = True
            else:
                record["errors"].extend(errors[:room])
                if len(errors) > room:
                    record["errors_truncated"] = True
            self._write(record)

    def list(
        self,
        kind: Optional[str] = None,
        index_name: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """Return ``(page, total)``, newest first."""
        records = []
        if self.root.exists():
            for path in self.root.glob("*.json"):
                record = self._read(path.stem)
                if record is None:
                    continue
                if kind is not None and record.get("kind") != kind:
                    continue
                if index_name is not None and record.get("index_name") != index_name:
                    continue
                records.append(record)
        records.sort(key=lambda r: (r.get("created_at") or "", r["job_id"]), reverse=True)
        return records[offset:offset + limit], len(records)

    # -- cancellation ------------------------------------------------------

    def request_cancel(self, job_id: str) -> dict:
        """Flag a job for cancellation.

        A ``queued`` job is canceled immediately. A ``running`` job stops at the
        next checkpoint its body observes; work already in flight inside the SDK
        cannot be interrupted. Terminal jobs are returned unchanged.
        """
        with self._lock:
            record = self.get(job_id)
            if record["status"] in TERMINAL_STATUSES:
                return record
            self._cancel_requested.add(job_id)
            record["cancel_requested"] = True
            if record["status"] == STATUS_QUEUED:
                record["status"] = STATUS_CANCELED
                record["finished_at"] = _utc_now()
                record["progress"]["phase"] = "canceled"
                # Nothing will ever run for this id, so drop the flag now
                # rather than leaving it to accumulate for the process lifetime.
                self._cancel_requested.discard(job_id)
            self._write(record)
            return record

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._cancel_requested:
                return True
            record = self._read(job_id)
            return bool(record and record.get("cancel_requested"))

    # -- execution ---------------------------------------------------------

    def _pool(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="jobstore",
                )
            return self._executor

    def submit(self, job_id: str, fn: Callable[[JobContext], Any]) -> None:
        """Queue *fn* for execution with a :class:`JobContext`."""
        self._pool().submit(self._run, job_id, fn)

    def _run(self, job_id: str, fn: Callable[[JobContext], Any]) -> None:
        started = time.perf_counter()
        with self._lock:
            record = self._read(job_id)
            # A job canceled while queued must not start.
            if record is None or record["status"] in TERMINAL_STATUSES:
                return
            record["status"] = STATUS_RUNNING
            record["started_at"] = _utc_now()
            record["progress"]["phase"] = "running"
            self._write(record)

        ctx = JobContext(self, job_id)
        try:
            fn(ctx)
            status, detail = STATUS_DONE, None
        except JobCanceled as exc:
            status, detail = STATUS_CANCELED, str(exc)
        except Exception as exc:
            status, detail = STATUS_FAILED, f"{type(exc).__name__}: {exc}"
            logger.exception("[JOB %s] failed", job_id)

        elapsed = time.perf_counter() - started
        with self._lock:
            record = self._read(job_id)
            if record is not None:
                record["status"] = status
                record["detail"] = detail
                record["finished_at"] = _utc_now()
                record["progress"]["phase"] = status
                record["timings"]["total_seconds"] = round(elapsed, 3)
                self._write(record)
            self._cancel_requested.discard(job_id)
        logger.info("[JOB %s] %s in %.2fs", job_id, status, elapsed)

    def shutdown(self, wait: bool = False) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=wait)

    def recover_interrupted(self) -> int:
        """Mark jobs left ``queued``/``running`` by a previous process as failed.

        Nothing resumes them: the worker threads died with the process. Without
        this the records would claim to be in progress forever and a client
        would poll a job that can never finish.
        """
        recovered = 0
        with self._lock:
            if not self.root.exists():
                return 0
            for path in sorted(self.root.glob("*.json")):
                record = self._read(path.stem)
                if record is None or record.get("status") in TERMINAL_STATUSES:
                    continue
                record["status"] = STATUS_FAILED
                record["detail"] = (
                    "Interrupted: the server stopped while this job was in progress. "
                    "Files already registered were kept; resubmit the rest."
                )
                record["finished_at"] = _utc_now()
                record["progress"]["phase"] = STATUS_FAILED
                self._write(record)
                recovered += 1
        if recovered:
            logger.warning("[JOB] marked %d interrupted job(s) as failed", recovered)
        return recovered

    # -- garbage collection ------------------------------------------------

    def collect_garbage(self, now: Optional[float] = None) -> dict[str, int]:
        """Delete terminal records past the TTL, then enforce *max_records*.

        Running and queued jobs are never collected, however old: losing the
        record of work still in flight would leave a client polling forever.
        """
        removed = 0
        cutoff = (time.time() if now is None else now) - self.ttl_seconds
        with self._lock:
            if not self.root.exists():
                return {"removed": 0, "kept": 0}
            survivors: list[tuple[float, pathlib.Path, dict]] = []
            for path in self.root.glob("*.json"):
                try:
                    record = self._load(path)
                except FileNotFoundError:
                    continue
                except ValueError:
                    # Malformed JSON can never be served, so drop it.
                    self._remove_record(path)
                    removed += 1
                    continue
                except OSError as exc:
                    # Locked or otherwise temporarily unreadable: keep it. A
                    # transient sharing violation must not delete live work.
                    logger.warning("Skipping job record %s during GC: %s", path, exc)
                    continue
                terminal = record.get("status") in TERMINAL_STATUSES
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = time.time()
                if terminal and self.ttl_seconds > 0 and mtime < cutoff:
                    self._remove_record(path)
                    removed += 1
                    continue
                survivors.append((mtime, path, record))

            # Cap the total, oldest terminal records first.
            overflow = len(survivors) - self.max_records
            if overflow > 0:
                terminal = [s for s in survivors if s[2].get("status") in TERMINAL_STATUSES]
                terminal.sort(key=lambda s: s[0])
                for _, path, _ in terminal[:overflow]:
                    self._remove_record(path)
                    removed += 1
                survivors = [s for s in survivors if s[1].exists()]

            return {"removed": removed, "kept": len(survivors)}
