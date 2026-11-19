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
import upload_paths
from config import _env_int
from index_registry import (
    _EMBED_ERROR_DEFAULT,
    _SDK_DEFAULT_TIME_LIMIT_SECONDS,
    compute_auto_workers,
    embed_time_limit,
)
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

    The flag exists because only the server sees the reason. The decision to
    retry is the caller's: a job with ``retry_timeouts`` acts on it itself in a
    second pass (see :func:`run_index_add_job`); otherwise the client resubmits
    the retryable file_ids as a second job with a longer ``time_limit`` and
    fewer ``workers``.

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
    collapsed_paths: Optional[dict[str, list[str]]] = None,
    selected_count: Optional[int] = None,
    passes: Optional[list[dict[str, Any]]] = None,
    recovered: Optional[list[str]] = None,
) -> None:
    """Write the registration report (ADDED / FAILED / HEAVY-FLAGGED / DUPLICATES).

    *passes* is given only when a second pass ran, and adds one header line per
    pass; *recovered* then names the files that pass saved, listed in
    ``[RECOVERED IN PASS 2]`` as well as in ``[ADDED]``. *added* and *failed*
    are the final state either way. A job that stopped after Pass 1 passes
    neither and gets the report it always did.

    The groups answer the operator's question directly: which parts are in the
    index, which are not and why (and whether a longer budget could still save
    them), which files made the run slow, and -- when the caller said how many
    files it picked -- why fewer went in than were chosen.

    ``selected_count`` and ``collapsed_paths`` describe **this submission and
    nothing else**: the paths come from the client that queued this job, so no
    other registration, index or stored file can contribute a name to them.
    Written to disk rather than into the job record because a corpus-sized job
    would otherwise put tens of thousands of paths into a JSON document that is
    rewritten on every progress update.
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
    collapsed = sorted((collapsed_paths or {}).items())
    # One file_id per entry, so the copies that never reached the index are all
    # the paths bar one.
    extra_copies = sum(len(paths) for _, paths in collapsed) - len(collapsed)
    submitted = len(added) + len(failed)
    lines: list[str] = []
    lines.append(
        "# Index registration report - "
        f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )
    lines.append(f"# Index: {name}")
    # A recorded limit of 0 means "nothing was passed to the SDK", which leaves
    # its own per-file budget in place. Printed raw it reads as no budget at
    # all, so the report says the number that was really in force and where it
    # came from.
    raw_limit = timings.get("time_limit", None)
    if raw_limit == 0:
        limit_text = (
            f"{_SDK_DEFAULT_TIME_LIMIT_SECONDS}s (SDK default; "
            "HOOPS_AI_EMBED_TIME_LIMIT unset)"
        )
    elif raw_limit is None:
        limit_text = "-"
    else:
        limit_text = f"{raw_limit}s"
    # hoops_ai does not apply a time limit when it runs a single worker: that
    # path is the same one it uses to finish heavy parts one at a time, and it
    # lets them run to completion. Measured on one file with time_limit=120:
    # 1,771 s and no timeout at num_workers=1, 120.2 s at num_workers=2. So
    # the limit is reported, but not as something that is in force.
    workers = timings.get("num_workers", "-")
    if workers == 1 and raw_limit is not None:
        note = "not enforced with num_workers=1"
        if limit_text.endswith(")"):
            limit_text = f"{limit_text[:-1]}; {note})"
        else:
            limit_text = f"{limit_text} ({note})"
    lines.append(f"# Embedding: num_workers={workers} time_limit={limit_text}")
    lines.append(f"# Elapsed: {_fmt(timings.get('embed_seconds'))}")
    for entry in passes or []:
        lines.append(
            f"# Pass {entry.get('pass')}: workers={entry.get('workers')} "
            f"time_limit={entry.get('time_limit')}s "
            f"submitted={entry.get('submitted')} added={entry.get('added')} "
            f"failed={entry.get('failed')} retryable={entry.get('retryable')} "
            f"elapsed={_fmt(entry.get('seconds'))}"
        )
    # "selected" is what the caller picked; "input" is what reached the SDK.
    # The two differ when the selection held byte-identical copies, and that
    # difference is the thing [DUPLICATES] below explains.
    if selected_count is not None:
        lines.append(
            f"# Selection: selected={selected_count} submitted={submitted} "
            f"collapsed={max(0, selected_count - submitted)}"
        )
    lines.append(
        f"# Totals: input={submitted} added={len(added)} "
        f"failed={len(failed)} retryable={len(retryable)} "
        f"heavy_flagged={len(heavy_flagged)} collapsed_copies={extra_copies}"
    )

    lines.append("")
    lines.append(f"[ADDED] embedded into the index ({len(added)})")
    lines.extend(added)

    if recovered is not None:
        lines.append("")
        lines.append(
            f"[RECOVERED IN PASS 2] timed out in Pass 1, embedded in Pass 2 "
            f"({len(recovered)})"
        )
        lines.extend(recovered)

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
        "[DUPLICATES] files in THIS selection that were byte-identical to each "
        f"other ({len(collapsed)} shape{'' if len(collapsed) == 1 else 's'}, "
        f"{extra_copies} copy/copies not submitted). Only one of each reached "
        "the index: a file_id is the SHA-256 of the bytes, so identical files "
        "are one entry. This covers this registration only -- no other job, "
        "index or previously stored file is consulted."
    )
    for fid, paths in collapsed:
        lines.append(f"{fid}\t{', '.join(paths)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_index_add_job(
    ctx,
    name: str,
    file_ids: list[str],
    num_workers: Optional[int] = None,
    time_limit: Optional[int] = None,
    selected_count: Optional[int] = None,
    collapsed_paths: Optional[dict[str, list[str]]] = None,
    retry_timeouts: bool = False,
    retry_workers: Optional[int] = None,
    retry_time_limit: Optional[int] = None,
) -> None:
    """Job body: register *file_ids* into index *name*, in one pass or two.

    The whole input goes to one ``embed_shape_batch`` (Pass 1): splitting it
    into fixed-size chunks was measurably wrong, because each call builds a
    fresh spawn-based process pool whose every worker reloads the ~2 GB
    checkpoint, ~55-60 s per call.

    **The second pass is optional and off by default.** With
    ``retry_timeouts`` the job goes on to retry, on the same index, the files
    whose Pass 1 failure was marked ``retryable`` (see
    :func:`annotate_failures`) -- with fewer workers and a longer budget, which
    is the trade the heavy assemblies that time out need. ``retry_workers`` and
    ``retry_time_limit`` override the defaults, which are the plan's
    ``retry_pass`` for the number of files being retried
    (:func:`index_registry.plan_embedding_run`). Permanent failures are never
    retried. Without ``retry_timeouts`` the job stops after Pass 1 and a client
    that wants a retry resubmits the retryable file_ids itself, as a second
    job; the report and the summary say which files are worth it.

    The summary and the report describe the *final* state: a file recovered in
    Pass 2 is no longer a failure, and ``added`` counts both passes. The
    per-pass figures are in ``summary['passes']``.

    Consequences, both accepted deliberately:
      * Cancellation is only observed before each pass starts.
        ``embed_shape_batch`` is a single blocking call that cannot be
        interrupted. A cancel that arrives during Pass 1 skips Pass 2, keeps
        what Pass 1 did, and ends the job as canceled.
      * A crash mid-pass loses that pass's work. Callers control the commit
        granularity by splitting a large corpus across several jobs.

    Failures are recorded per file with the reason reported by the SDK, never
    dropped.

    ``selected_count`` and ``collapsed_paths`` are what the caller picked and
    which of those files turned out to be the same bytes. Both are display
    data supplied by the caller and are reported back unchanged: this job
    explains its own submission, and scanning the store or another job's record
    for extra names is exactly what it must not do.
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

    # Resolved once, so the report can name files the way the client does.
    # The original filename, never the stored path: the report is read by an
    # operator through the browser, and neither this server's directory layout
    # nor its "<sha256>_<name>" rename rule is the client's business. The
    # client-supplied folder is preferred when there is one; everything
    # uploaded before that was recorded falls back to the filename on disk, and
    # an id with neither (a file deleted from the store mid-job) falls back to
    # the id itself rather than to an empty column.
    name_by_fid: dict[str, str] = {
        fid: storage.original_filename(path.name)
        for fid, path in map_persistent_CAD_files(file_ids).items()
    }
    try:
        relative_by_fid = upload_paths.paths_for(file_ids)
    except Exception:  # display data only; never fail a run over it
        logger.debug("upload paths unavailable", exc_info=True)
        relative_by_fid = {}

    def _display(fid: str) -> str:
        recorded = relative_by_fid.get(fid)
        if recorded:
            return recorded[0]
        return name_by_fid.get(fid, fid)

    display_by_fid: dict[str, str] = {fid: _display(fid) for fid in file_ids}

    t0 = time.perf_counter()

    # Live counters parsed from the SDK's own progress bar, so a client polling
    # the job sees it advance instead of 0/N until the batch ends. The bar's own
    # total is ignored: progress.total is the number of files this job submitted,
    # which is the number the client asked about.
    seen_phase = {"value": "embedding"}
    current_pass = {"value": 1}

    def _on_progress(
        phase: str, done: int, _bar_total: int, err_count: int, heavy: int
    ) -> None:
        fields: dict[str, Any] = {"done": int(done)}
        if err_count >= 0:
            fields["errors"] = int(err_count)
        if heavy >= 0:
            fields["heavy"] = int(heavy)
        # The second pass stays "retry" whatever phase the SDK's bar is in.
        if phase != seen_phase["value"] and current_pass["value"] == 1:
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
    # The SDK logs its own stored paths; the report shows the client's names.
    def _display_stored(line: Any) -> str:
        base = pathlib.PurePath(str(line or "").replace("\\", "/")).name
        if base[64:65] == "_" and base[:64].lower() in display_by_fid:
            return display_by_fid[base[:64].lower()]
        return storage.original_filename(line)

    heavy_flagged = [_display_stored(line) for line in heavy_flagged]

    def _resolve_errors(raw: dict[str, Any]) -> list[dict[str, Any]]:
        found = annotate_failures(list(raw.get("errors") or []))
        for err in found:
            fid = str(err.get("file_id") or "")
            if fid in display_by_fid:
                # Assigned, not set as a default: whatever the SDK put here is
                # one of its own stored paths, which is the thing being kept
                # off the report.
                err["filename"] = display_by_fid[fid]
            elif err.get("filename"):
                err["filename"] = storage.original_filename(err["filename"])
        return found

    errors = _resolve_errors(result)

    added = int(result.get("added", 0))
    updated = int(result.get("updated", 0))
    index_count = int(result.get("index_count", 0))
    retryable = sum(1 for e in errors if e.get("retryable"))
    passes: list[dict[str, Any]] = [{
        "pass": 1, "workers": workers,
        "time_limit": limit or _SDK_DEFAULT_TIME_LIMIT_SECONDS,
        "submitted": total,
        "added": added, "failed": len(errors), "retryable": retryable,
        "seconds": timings["embed_seconds"],
    }]

    # Accounting guard: every submitted file must be added, updated or reported
    # as failed. A silent shortfall means the SDK contract moved.
    accounted = added + updated + len(errors)
    if accounted != total:
        logger.warning(
            "[JOB] index '%s': %d files submitted but only %d accounted for "
            "(added/updated/errors); some inputs were not reported",
            name, total, accounted,
        )
    if retryable and not retry_timeouts:
        logger.info(
            "[JOB] index '%s': %d of %d failure(s) are retryable; resubmit them "
            "with a longer time_limit and fewer workers",
            name, retryable, len(errors),
        )

    # Pass 2: only the files whose failure was retryable, on the same index.
    recovered_ids: Optional[list[str]] = None
    canceled_before_retry = False
    retry_ids = list(dict.fromkeys(
        str(e.get("file_id") or "") for e in errors if e.get("retryable")
    ))
    retry_ids = [f for f in retry_ids if f in display_by_fid]
    if retry_timeouts and retry_ids:
        import jobstore  # lazily, like get_index_job_store

        try:
            ctx.check_canceled()
        except jobstore.JobCanceled:
            canceled_before_retry = True
            logger.info(
                "[JOB] index '%s': canceled before the retry pass; keeping the "
                "first pass's results", name,
            )
    if retry_timeouts and retry_ids and not canceled_before_retry:
        plan = index_registry.plan_embedding_run(len(retry_ids))["retry_pass"]
        r_workers = retry_workers if retry_workers is not None else plan["workers"]
        r_workers = max(1, min(int(r_workers), len(retry_ids)))
        r_limit = retry_time_limit if retry_time_limit is not None else plan["time_limit"]
        logger.info(
            "[JOB] index '%s': retry pass starting: %d timed-out file(s), "
            "workers=%d, time_limit=%ds",
            name, len(retry_ids), r_workers, r_limit,
        )
        current_pass["value"] = 2
        ctx.set_total(len(retry_ids))
        ctx.set_progress(**{"pass": 2, "phase": "retry", "done": 0, "errors": 0, "heavy": 0})
        t1 = time.perf_counter()
        result2: dict[str, Any] = {}
        pass_error: Optional[str] = None
        try:
            result2 = index_registry.add_to_index(
                name, retry_ids, num_workers=r_workers, time_limit=r_limit,
                log_dir=artifacts, progress_cb=_on_progress,
            )
        except Exception as exc:  # Pass 1's work is already in the index
            pass_error = f"{type(exc).__name__}: {exc}"
            logger.warning("[JOB] index '%s': retry pass failed: %s", name, pass_error)
        seconds2 = round(time.perf_counter() - t1, 3)
        timings["embed_seconds"] = round(timings["embed_seconds"] + seconds2, 3)

        # Read now as well: Pass 1's lines were taken before this call replaced
        # the log, and these are the heavy files of the retry.
        for line in storage.read_too_heavy_files(artifacts):
            shown = _display_stored(line)
            if shown not in heavy_flagged:
                heavy_flagged.append(shown)

        if pass_error is None:
            errors2 = _resolve_errors(result2)
            for err in errors2:
                err["detail"] = f"{err['detail']} (retry pass, time_limit={r_limit}s)"
            failed2 = {str(e.get("file_id") or ""): e for e in errors2}
            retry_set = set(retry_ids)
            recovered_ids = [f for f in retry_ids if f not in failed2]
            errors = [
                failed2.get(fid, e)
                for e in errors
                for fid in (str(e.get("file_id") or ""),)
                if fid not in retry_set or fid in failed2
            ]
            added2 = int(result2.get("added", 0))
            added += added2
            updated += int(result2.get("updated", 0))
            index_count = int(result2.get("index_count", index_count))
            retryable = sum(1 for e in errors if e.get("retryable"))
            entry = {
                "pass": 2, "workers": r_workers, "time_limit": r_limit,
                "submitted": len(retry_ids), "added": added2,
                "failed": len(errors2),
                "retryable": sum(1 for e in errors2 if e.get("retryable")),
                "seconds": seconds2,
            }
        else:
            entry = {
                "pass": 2, "workers": r_workers, "time_limit": r_limit,
                "submitted": len(retry_ids), "added": 0,
                "failed": len(retry_ids), "retryable": len(retry_ids),
                "seconds": seconds2, "error": pass_error,
            }
        passes.append(entry)
        logger.info(
            "[JOB] index '%s': retry pass finished in %.1fs: %d of %d file(s) "
            "recovered, %d still failing",
            name, seconds2, entry["added"], len(retry_ids), entry["failed"],
        )

    ctx.add_errors(errors)
    # Absolute, not incremental: the live callback above has already been moving
    # progress.done, so incrementing here would count the batch twice.
    ctx.set_total(total)
    ctx.set_progress(
        done=total, errors=len(errors), heavy=len(heavy_flagged),
        **{"pass": len(passes)},
    )
    ctx.set_summary(
        added=added, updated=updated, failed=len(errors), skipped=0,
        index_count=index_count, heavy_flagged=len(heavy_flagged),
        retryable=retryable, passes=passes,
    )
    ctx.set_timings(**timings)

    failed_ids = {str(e.get("file_id") or "") for e in errors}
    # This run's own accounting, from what the caller sent with the job. No
    # directory scan and no cross-job lookup: a store-wide scan used to supply
    # this and it mixed in names from registrations the reader never made.
    collapsed = {
        fid: paths for fid, paths in (collapsed_paths or {}).items()
        if fid in set(file_ids)
    }
    redundant = sum(len(p) for p in collapsed.values()) - len(collapsed)
    report_added = [display_by_fid.get(f, f) for f in file_ids if f not in failed_ids]
    if selected_count is not None:
        # The selection figures are published whole, from the same two lists
        # the report's "# Selection:" line counts, so the summary and the
        # report can never disagree. The client used to rebuild the gap from
        # added + failed, which answers a different question -- how embedding
        # went -- and reported 41 collapsed copies for a run that had none.
        submitted = len(report_added) + len(errors)
        ctx.set_summary(
            selected=int(selected_count),
            submitted=submitted,
            collapsed=max(0, int(selected_count) - submitted),
        )
    if collapsed:
        ctx.set_summary(collapsed_copies=redundant)
    try:
        _write_add_job_report(
            artifacts / "report.txt",
            name,
            report_added,
            errors,
            heavy_flagged,
            timings,
            collapsed,
            selected_count,
            passes=passes if len(passes) > 1 else None,
            recovered=(
                [display_by_fid.get(f, f) for f in recovered_ids]
                if recovered_ids is not None else None
            ),
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

    if canceled_before_retry:
        raise jobstore.JobCanceled(
            "Canceled before the retry pass; the first pass's results were kept."
        )