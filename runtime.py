"""Runtime directory lifecycle and startup configuration logging.

Owns the transient-directory management the server runs at startup: creating
the directories it writes to, TTL-sweeping the throw-away ones (``out/`` and,
optionally, ``embeddings_cache/``), and logging where each resolved setting
came from. The CAD store and ``indexes/`` are permanent and are never swept.

Dependency direction is one-way: ``core`` imports this module; this module
reaches its dependencies through their owning modules -- :mod:`config`,
:mod:`embeddings`, :mod:`storage`, :mod:`shape_map` (``clear_scs_map_cache``)
and :mod:`jobs`. It never imports ``core``.

The one cross-domain edge is deliberate: :func:`run_startup_maintenance` calls
``jobs.get_index_job_store()`` to recover interrupted jobs and collect garbage.
``get_index_job_store`` is an (A) name that is not re-exported from ``core``,
so it is reached on its owning module; the job tests swap it there. This is why
``jobs`` is extracted before ``runtime``.

``hoops_license_shim`` is imported lazily inside
:func:`log_resolved_configuration` (only to report which activation source is
in effect), so importing this module does not require the compiled shim.

All nine names are stable (C) names re-exported from ``core`` by identity;
``_json_safe`` is a private helper used only by
:func:`log_resolved_configuration`.
"""


import logging
import os
import pathlib
import time
from typing import Any, Optional

import config
import embeddings
import jobs
import storage
from config import _env_int
from shape_map import clear_scs_map_cache


logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def log_resolved_configuration() -> None:
    """Log where each runtime setting was resolved from, at startup.

    Emitted as INFO so an operator supporting a distributed deployment can see,
    without a debugger, which value came from where and which paths were chosen.
    Best-effort: never raises, so a logging problem cannot block startup.
    """
    try:
        config.load_env_file()
        sdk_set = bool(os.environ.get("HOOPS_AI_SDK_DIR"))
        try:
            import hoops_license_shim  # noqa: F401
            license_source = "compiled shim"
        except ImportError:
            license_source = ".env fallback (development)"
        logger.info(
            "[CONFIG] Setting precedence: os.environ > .env > derived default "
            "(HOOPS_AI_LICENSE excepted: .env only, never os.environ)."
        )
        logger.info(
            "[CONFIG] Data root: %s (source: %s).", config.DATA_ROOT, config._DATA_ROOT_SOURCE
        )
        logger.info(
            "[CONFIG] HOOPS_AI_SDK_DIR: %s (%s).",
            "set" if sdk_set else "unset",
            "available" if embeddings.sdk_dir_configured() else "disabled",
        )
        logger.info("[CONFIG] License activation source: %s.", license_source)
        name_env = os.environ.get("HOOPS_AI_EMBEDDINGS_MODEL_NAME")
        if name_env:
            logger.info(
                "[MODEL] Embedding checkpoint selected by name "
                "(HOOPS_AI_EMBEDDINGS_MODEL_NAME=%s).", name_env,
            )
        else:
            try:
                embeddings.resolve_bundled_checkpoint()  # logs the resolved path and size
            except embeddings.ModelResolutionError as exc:
                logger.warning("[MODEL] No embedding checkpoint resolved yet: %s", exc)
    except Exception:  # pragma: no cover - diagnostics must never break startup
        logger.debug("[CONFIG] Failed to log resolved configuration.", exc_info=True)


# --- Runtime directory lifecycle --------------------------------------------
#
# The CAD store (``uploads/``, or HOOPS_AI_CAD_SHARED_DIR when set) holds the
# only copy of every file registered in an index: index records store the
# SHA-256 file_id, never the bytes. It is therefore a permanent data directory
# on the same footing as ``indexes/`` and is never swept here.
#
# ``out/`` is the opposite: throw-away viewer artefacts (SCS streams, result
# images, shape maps) served over ``/out/`` and referenced only by URLs handed
# to a client moments earlier. It grows without bound, so it gets a TTL sweep.
# ``embeddings_cache/`` is a cache, but recomputing one entry costs a full CAD
# load, so it is kept indefinitely unless an operator opts into a TTL.
#
# Nothing is wiped at startup any more. The previous rmtree destroyed the CAD
# store -- taking the payload of every registered index with it -- and, when a
# second instance was started on another port, deleted the working files of the
# instance already running.

_OUT_TTL_HOURS_DEFAULT = 24
_EMBEDDINGS_CACHE_TTL_DAYS_DEFAULT = 0  # 0 = keep forever


def viewer_output_ttl_seconds() -> int:
    """TTL for ``out/``, from HOOPS_AI_OUT_TTL_HOURS. 0 disables the sweep."""
    return _env_int("HOOPS_AI_OUT_TTL_HOURS", _OUT_TTL_HOURS_DEFAULT, minimum=0) * 3600


def embeddings_cache_ttl_seconds() -> int:
    """TTL for ``embeddings_cache/``, from HOOPS_AI_EMBEDDINGS_CACHE_TTL_DAYS.

    0 (the default) disables the sweep.
    """
    return _env_int(
        "HOOPS_AI_EMBEDDINGS_CACHE_TTL_DAYS",
        _EMBEDDINGS_CACHE_TTL_DAYS_DEFAULT,
        minimum=0,
    ) * 86400


def sweep_directory(
    directory: pathlib.Path,
    max_age_seconds: int,
    now: Optional[float] = None,
) -> dict[str, int]:
    """Delete files under *directory* whose mtime is older than *max_age_seconds*.

    Returns ``{"removed": n, "kept": n, "failed": n}``. A *max_age_seconds* of 0
    or less disables the sweep and reports zeroes. Individual failures are
    logged and counted but never raised: a file another request is still
    streaming must not be able to take startup down.
    """
    stats = {"removed": 0, "kept": 0, "failed": 0}
    if max_age_seconds <= 0 or not directory.exists():
        return stats

    cutoff = (time.time() if now is None else now) - max_age_seconds
    for path in directory.rglob("*"):
        try:
            if not path.is_file():
                continue
            if path.stat().st_mtime >= cutoff:
                stats["kept"] += 1
                continue
            path.unlink()
            stats["removed"] += 1
        except OSError as exc:
            stats["failed"] += 1
            logger.warning("Failed to sweep %s: %s", path, exc)

    # Drop directories the sweep emptied, deepest first. Best effort only.
    for path in sorted(directory.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        except OSError:
            pass
    return stats


def ensure_runtime_dirs() -> None:
    """Create the directories the server writes to, without deleting anything."""
    for folder in (
        storage.get_CAD_shared_dir(),
        config.CAD_VIEWER_OUTPUT_DIR,
        config.EMBEDDINGS_CACHE_DIR,
        config.INDEXES_DIR,
        config.JOBS_DIR,
    ):
        folder.mkdir(parents=True, exist_ok=True)


def run_startup_maintenance() -> dict[str, dict[str, int]]:
    """Create runtime directories and TTL-sweep the transient ones."""
    config.ensure_data_root_writable()
    ensure_runtime_dirs()
    results = {
        "out": sweep_directory(config.CAD_VIEWER_OUTPUT_DIR, viewer_output_ttl_seconds()),
        "embeddings_cache": sweep_directory(
            config.EMBEDDINGS_CACHE_DIR, embeddings_cache_ttl_seconds()
        ),
    }
    for label, stats in results.items():
        if stats["removed"] or stats["failed"]:
            logger.info(
                "[GC] %s: removed %d, kept %d, failed %d",
                label, stats["removed"], stats["kept"], stats["failed"],
            )
    scs_removed = clear_scs_map_cache()
    if scs_removed:
        logger.info("[GC] scs_map: cleared %d stale point-cloud SCS file(s)", scs_removed)
    jobs.get_index_job_store().recover_interrupted()
    jobs.get_index_job_store().collect_garbage()
    return results