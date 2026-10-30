"""Background index-registration jobs.

Owns the process-wide job store and the job body that registers CAD files into
a FAISS index in a single SDK ``embed_shape_batch`` call, plus the retention
knobs, the retryable-failure annotation and the on-disk registration report.

Dependency direction is one-way: ``core`` imports this module; this module
reaches its dependencies through their owning modules -- ``index_registry``
(``add_to_index``), ``assembly`` (``prewarm_assembly_matcher``), ``storage``
and ``shape_map`` -- and imports :mod:`config` symbols directly. It never
imports ``core``. ``jobstore`` is imported lazily inside
:func:`get_index_job_store` so importing this module stays cheap.

Two names are deliberately **not** re-exported from ``core``:

* :func:`get_index_job_store` -- an (A) name. The router and the job tests
  swap it (``jobs.get_index_job_store = ...`` in the tests), so it must be
  reached as ``jobs.get_index_job_store`` to keep a single source of truth.
* :data:`_index_job_store` -- both an (A) name (the tests assign it directly)
  and a (B) name (``get_index_job_store`` rebinds it under ``global`` as a lazy
  singleton). The ``global`` statement stays inside this module, so the
  singleton keeps working after the move. Re-exporting either name would create
  the double-truth trap the layout test guards against.

The remaining twelve names are stable (C) names re-exported from ``core`` by
identity.
"""


import datetime
import logging
# Referenced only from string annotations ("pathlib.Path"). Kept so that
# typing.get_type_hints() can still resolve them.
import pathlib
import threading
import time
from typing import Any, Optional

import assembly
import config
import index_registry
import storage
from config import _env_int
from index_registry import _EMBED_ERROR_DEFAULT, compute_auto_workers, embed_time_limit
from storage import map_persistent_CAD_files
from shape_map import prewarm_shape_map


logger = logging.getLogger(__name__)


# --- Background registration jobs -------------------------------------------

JOB_KIND_INDEX_ADD = "index_add"

_JOB_MAX_CONCURRENCY_DEFAULT = 1
_JOB_TTL_DAYS_DEFAULT = 7
_JOB_MAX_RECORDS_DEFAULT = 1000

_index_job_store: Any = None
_index_job_store_lock = threading.Lock()


def job_max_concurrency() -> int:
    """How many jobs may run at once, from HOOPS_AI_JOB_MAX_CONCURRENCY.

    Defaults to 1, i.e. registration jobs are serialised across all indexes, for
    three reasons:

    * Memory. Each job spawns its own pool of embedding workers, every one of
      which loads the ~2 GB model checkpoint, so two concurrent jobs multiply
      peak memory and push the SDK into its single-worker RAM fallback --
      slower in total than running them one after the other.
    * ``error_summary.json``. hoops_ai writes it to the process working
      directory and overwrites it on every ``embed_shape_batch`` call, with no
      way to redirect it (it logs ``Undefined flowdir ... Using path [.]``), so
      concurrent jobs would read each other's failure reasons. The mtime guard
      in :func:`read_error_summary_reasons` catches a stale file, not a
      concurrent one.
    * Live progress. :func:`capture_embed_progress` replaces the process-wide
      ``sys.stderr``; a second job reports nothing rather than the wrong thing.

    ``too_heavy_files.log`` is kept apart per job via
    ``specifications['log_dir']`` -- a documented key (hoops_ai 1.1.0,
    HOOPSEmbeddings._embed_shape_batch_parallel) whose effect is nonetheless
    still unverified live, so concurrency would rest on it holding.
    """
    return _env_int("HOOPS_AI_JOB_MAX_CONCURRENCY", _JOB_MAX_CONCURRENCY_DEFAULT, minimum=1)


def job_ttl_seconds() -> int:
    """Retention of finished job records, from HOOPS_AI_JOB_TTL_DAYS."""
    return _env_int("HOOPS_AI_JOB_TTL_DAYS", _JOB_TTL_DAYS_DEFAULT, minimum=0) * 86400


def job_max_records() -> int:
    """Hard cap on retained job records, from HOOPS_AI_JOB_MAX_RECORDS."""
    return _env_int("HOOPS_AI_JOB_MAX_RECORDS", _JOB_MAX_RECORDS_DEFAULT, minimum=1)


def get_index_job_store():
    """Process-wide job store for index registration jobs."""
    global _index_job_store
    with _index_job_store_lock:
        if _index_job_store is None:
            import jobstore

            _index_job_store = jobstore.JobStore(
                root=config.JOBS_DIR,
                max_workers=job_max_concurrency(),
                ttl_seconds=job_ttl_seconds(),
                max_records=job_max_records(),
            )
        return _index_job_store


_RETRY_REASON_MARKERS = ("timeout", "timed out")


def annotate_failures(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach a ``retryable`` flag to each embedding failure.

    A file is marked retryable when it timed out, or when no reason could be
    determined at all. Everything else -- a corrupt file, an unsupported format,
    a CAD kernel error -- will fail again however long the budget, so retrying it
    only wastes the budget. Treating an unknown reason as retryable is the safe
    direction. This is the rule hoops_ai_native_bridge's Qt front end applies.

    The reason itself is already resolved into each error's ``detail`` from
    ``batch.metadata['errors']``. That front end additionally parses
    ``error_summary.json`` out of its working directory, but only because the
    bridge's C ABI drops the reason on the way out; running the SDK in-process,
    this server gets it directly and needs no such fallback.

    The flag exists because only the server sees the reason. **The decision to
    retry stays with the client**, which resubmits the retryable file_ids as a
    second job with a longer ``time_limit`` and fewer ``workers``.

    Returns a new list; the input is not modified.
    """
    annotated: list[dict[str, Any]] = []
    for err in errors:
        item = dict(err)
        detail = str(item.get("detail") or "").strip()
        unknown = not detail or detail == _EMBED_ERROR_DEFAULT
        low = detail.lower()
        item["detail"] = detail or _EMBED_ERROR_DEFAULT
        item["retryable"] = bool(unknown or any(m in low for m in _RETRY_REASON_MARKERS))
        annotated.append(item)
    return annotated


def _write_add_job_report(
    path: "pathlib.Path",
    name: str,
    added: list[str],
    failed: list[dict[str, Any]],
    heavy_flagged: list[str],
    timings: dict[str, Any],
    duplicate_names: Optional[dict[str, list[str]]] = None,
) -> None:
    """Write the registration report (ADDED / FAILED / HEAVY-FLAGGED / DUPLICATES).

    The groups answer the operator's question directly: which parts are in the
    index, which are not and why (and whether a longer budget could still save
    them), which files made the run slow, and which of the submitted files were
    the same geometry arriving under more than one name. Written to disk rather
    than into the job record because a corpus-sized job would otherwise put tens
    of thousands of paths into a JSON document that is rewritten on every
    progress update.
    """
    def _fmt(seconds: Any) -> str:
        try:
            secs = float(seconds)
        except (TypeError, ValueError):
            return "-"
        total = int(secs + 0.5)
        h, m, s = total // 3600, (total % 3600) // 60, total % 60
        if h:
            hms = f"{h}h {m:02d}m {s:02d}s"
        elif m:
            hms = f"{m}m {s:02d}s"
        else:
            hms = f"{s}s"
        return f"{hms} ({secs:.1f}s)"

    retryable = [e for e in failed if e.get("retryable")]
    duplicates = sorted((duplicate_names or {}).items())
    # One geometry per entry, so the redundant names are all of them bar one.
    extra_copies = sum(len(names) for _, names in duplicates) - len(duplicates)
    lines: list[str] = []
    lines.append(
        "# Index registration report - "
        f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )
    lines.append(f"# Index: {name}")
    lines.append(
        f"# Embedding: num_workers={timings.get('num_workers', '-')} "
        f"time_limit={timings.get('time_limit', '-')}s"
    )
    lines.append(f"# Elapsed: {_fmt(timings.get('embed_seconds'))}")
    lines.append(
        f"# Totals: input={len(added) + len(failed)} added={len(added)} "
        f"failed={len(failed)} retryable={len(retryable)} "
        f"heavy_flagged={len(heavy_flagged)} duplicate_names={len(duplicates)}"
    )

    lines.append("")
    lines.append(f"[ADDED] embedded into the index ({len(added)})")
    lines.extend(added)

    lines.append("")
    lines.append(
        f"[FAILED] not in the index ({len(failed)}; {len(retryable)} worth "
        "resubmitting with a longer time_limit)"
    )
    for err in failed:
        mark = "RETRYABLE" if err.get("retryable") else "PERMANENT"
        label = err.get("filename") or err.get("file_id") or ""
        lines.append(f"{mark}\t{label}\t{err.get('detail', '')}")

    lines.append("")
    lines.append(
        "[HEAVY-FLAGGED] deferred by hoops_ai to its 1-worker RAM fallback "
        f"({len(heavy_flagged)})"
    )
    lines.extend(heavy_flagged)

    lines.append("")
    lines.append(
        "[DUPLICATES] one geometry registered under several original filenames "
        f"({len(duplicates)} geometr{'y' if len(duplicates) == 1 else 'ies'}, "
        f"{extra_copies} redundant copy/copies). The index holds ONE entry for "
        "each: identical bytes share a file_id, so nothing below is a duplicate "
        "index entry."
    )
    for fid, names in duplicates:
        lines.append(f"{fid}\t{', '.join(names)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_index_add_job(
    ctx,
    name: str,
    file_ids: list[str],
    num_workers: Optional[int] = None,
    time_limit: Optional[int] = None,
) -> None:
    """Job body: register *file_ids* into index *name* in a single SDK call.

    One job is one pass. The whole input goes to one ``embed_shape_batch``:
    splitting it into fixed-size chunks was measurably wrong, because each call
    builds a fresh spawn-based process pool whose every worker reloads the ~2 GB
    checkpoint, ~55-60 s per call.

    **The two-pass strategy is the client's to run, not this server's.** A
    caller registers a corpus with many workers and a short ``time_limit``,
    reads the ``retryable`` flag on each reported failure, and resubmits just
    those files as a second job with few workers and a long ``time_limit``.
    Splitting it this way keeps the commit granularity and the retry policy in
    the caller's hands; the server's part is to make the decision possible by
    reporting *why* each file failed (see :func:`annotate_failures`).

    Consequences, both accepted deliberately:
      * Cancellation is only observed before the call starts. ``embed_shape_batch``
        is a single blocking call that cannot be interrupted.
      * A crash mid-job loses that job's work. Callers control the commit
        granularity by splitting a large corpus across several jobs.

    Failures are recorded per file with the reason reported by the SDK, never
    dropped.
    """
    total = len(file_ids)
    ctx.set_total(total)
    ctx.set_phase("embedding")
    ctx.set_summary(added=0, updated=0, failed=0, skipped=0, heavy_flagged=0)
    ctx.check_canceled()

    store = get_index_job_store()
    try:
        artifacts = store.artifacts_dir(ctx.job_id)
    except Exception:  # a job store that predates artefact directories
        artifacts = config.JOBS_DIR / str(ctx.job_id)

    workers = num_workers if num_workers is not None else compute_auto_workers(total)
    limit = time_limit if time_limit is not None else embed_time_limit()
    timings: dict[str, Any] = {"num_workers": workers, "time_limit": limit}
    ctx.set_timings(**timings)

    # Resolved once, so the report can name files rather than hashes.
    path_by_fid: dict[str, str] = {
        fid: str(path.resolve())
        for fid, path in map_persistent_CAD_files(file_ids).items()
    }

    t0 = time.perf_counter()

    # Live counters parsed from the SDK's own progress bar, so a client polling
    # the job sees it advance instead of 0/N until the batch ends. The bar's own
    # total is ignored: progress.total is the number of files this job submitted,
    # which is the number the client asked about.
    seen_phase = {"value": "embedding"}

    def _on_progress(
        phase: str, done: int, _bar_total: int, err_count: int, heavy: int
    ) -> None:
        fields: dict[str, Any] = {"done": int(done)}
        if err_count >= 0:
            fields["errors"] = int(err_count)
        if heavy >= 0:
            fields["heavy"] = int(heavy)
        if phase != seen_phase["value"]:
            seen_phase["value"] = phase
            fields["phase"] = phase
        try:
            ctx.set_progress(**fields)
        except Exception:
            # Persisting progress is best-effort; never fail a run over it.
            logger.debug("progress update dropped", exc_info=True)

    result = index_registry.add_to_index(
        name, file_ids, num_workers=workers, time_limit=limit,
        log_dir=artifacts, progress_cb=_on_progress,
    )
    timings["embed_seconds"] = round(time.perf_counter() - t0, 3)

    # too_heavy_files.log must be read now: the next embedding call overwrites it.
    heavy_flagged = storage.read_too_heavy_files(artifacts)
    errors = annotate_failures(list(result.get("errors") or []))
    for err in errors:
        fid = str(err.get("file_id") or "")
        if fid in path_by_fid:
            err.setdefault("filename", path_by_fid[fid])

    added = int(result.get("added", 0))
    updated = int(result.get("updated", 0))
    index_count = int(result.get("index_count", 0))
    retryable = sum(1 for e in errors if e.get("retryable"))

    # Accounting guard: every submitted file must be added, updated or reported
    # as failed. A silent shortfall means the SDK contract moved.
    accounted = added + updated + len(errors)
    if accounted != total:
        logger.warning(
            "[JOB] index '%s': %d files submitted but only %d accounted for "
            "(added/updated/errors); some inputs were not reported",
            name, total, accounted,
        )
    if retryable:
        logger.info(
            "[JOB] index '%s': %d of %d failure(s) are retryable; resubmit them "
            "with a longer time_limit and fewer workers",
            name, retryable, len(errors),
        )

    ctx.add_errors(errors)
    # Absolute, not incremental: the live callback above has already been moving
    # progress.done, so incrementing here would count the batch twice.
    ctx.set_progress(done=total, errors=len(errors), heavy=len(heavy_flagged))
    ctx.set_summary(
        added=added, updated=updated, failed=len(errors), skipped=0,
        index_count=index_count, heavy_flagged=len(heavy_flagged),
        retryable=retryable,
    )
    ctx.set_timings(**timings)

    failed_ids = {str(e.get("file_id") or "") for e in errors}
    # Only this job's own inputs: the report is about this run, and the
    # index-wide view lives behind GET .../duplicates. One scan of the CAD
    # store, and never fatal -- a missing section must not cost a good report.
    try:
        duplicate_names = storage.map_duplicate_CAD_filenames(file_ids)
    except Exception as exc:
        logger.warning("[JOB] could not collect duplicate names: %s", exc)
        duplicate_names = {}
    try:
        _write_add_job_report(
            artifacts / "report.txt",
            name,
            [path_by_fid.get(f, f) for f in file_ids if f not in failed_ids],
            errors,
            heavy_flagged,
            timings,
            duplicate_names,
        )
        ctx.set_summary(report=True)
    except Exception as exc:  # a missing report must never fail a good job
        logger.warning("[JOB] could not write report for job %s: %s", ctx.job_id, exc)

    ctx.set_phase("done")

    # The job has just changed the .faiss mtime, invalidating any cached
    # assembly matcher. Rebuild it off the job thread so the job is reported
    # done immediately and the next searcher does not pay for the k-means.
    if added or updated:
        try:
            assembly.prewarm_assembly_matcher(name)
        except Exception as exc:  # never fail a completed job over a warm-up
            logger.warning("[JOB] could not start matcher prewarm: %s", exc)
        try:
            prewarm_shape_map(name)
        except Exception as exc:  # never fail a completed job over a warm-up
            logger.warning("[JOB] could not start shape-map prewarm: %s", exc)