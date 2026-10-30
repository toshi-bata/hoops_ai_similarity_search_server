"""Named-index registry: incremental index management and the public index API.

This module owns the on-disk named-index registry -- the in-memory write-store
cache and the read-side CADSearch (searcher) cache, index creation/listing,
the body-level embedding pass that fills an index, per-part add/remove asset
bookkeeping, index schema/model side-cars, and the small helpers that flatten
FAISS hits for the routers. It is a leaf: it depends only on the standard
library, ``config``, ``storage``, ``embeddings`` and ``locks``. It must never
import ``core`` --
the dependency runs one way, from ``core`` down to ``index_registry``. The
registry cluster has no upward edges: nothing here calls tags, search,
assembly, shape-map or job code (verified for this batch).

Two operations that read like index calls stay in ``core`` on purpose:
``delete_index`` and ``remove_from_index``. Both orchestrate clean-up that
reaches into sibling domains -- ``delete_index`` evicts the assembly-matcher
and tags caches, ``remove_from_index`` prunes tags and context records. Those
domains sit *above* ``index_registry`` in the dependency DAG, so ``core`` keeps
the orchestration and calls ``index_registry.<name>`` downward for the
index-owned parts (see "The layering" in docs/architecture.md).

Reference convention (see "The re-export convention" in
docs/architecture.md). The names a test swaps must be reached as module
attributes -- ``index_registry._load_named_index``,
``index_registry._get_named_searcher``, ``index_registry._load_index_schema``,
``index_registry._resolve_index_asset``,
``index_registry.add_to_index``, ``index_registry._save_named_index_atomic``,
``index_registry._registered_index_ids``, ``index_registry._named_indexes``,
``index_registry._index_faiss_path``, ``index_registry._index_cache_max_bytes``,
``index_registry._index_entry_bytes``, ``index_registry.list_indexes``,
``index_registry._load_shape_index_pickle_compat`` and
``index_registry._embed_bodies_for_index``. Do NOT write ``from
index_registry import <name>``: a ``from`` import binds the name at import
time, so a test patching ``index_registry.<name>`` would never reach a module
that imported the old object. These fifteen are therefore NOT re-exported by
``core``. Everything else moved here is stable and is re-exported from ``core``
by explicit name for its own callers and for existing ``core.<name>``
references in routers, tools and tests.

Record-order invariant (issue #43). ``_embed_bodies_for_index`` writes one
FAISS record per body in ``file_ids`` order -- the order the caller listed the
files, not the order worker processes happen to finish. K-means over the
resulting vectors is order-sensitive, so a stable write order keeps the shape
map and any downstream clustering reproducible run to run. Preserve that
ordering exactly.

Cache/lock layout (cross-reference the LOCK ORDERING note in locks.py). The
write-store cache ``_named_indexes`` (an ``OrderedDict`` LRU bounded by memory,
not entry count) is guarded by the leaf ``_named_indexes_mutex``: never acquire
another lock while holding it. The read-side ``_named_searchers`` cache, keyed
by (faiss path, mtime_ns) so a rebuilt index is picked up automatically, is
mutated only while holding the per-index lock from ``locks._get_index_lock``.
Every write persists to disk under that per-index lock before returning, so a
cache entry can be evicted and transparently reloaded on the next access.
"""

import datetime
import json
import logging
import os
import pathlib
import re
import shutil
import sys
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Optional

import config
import storage
import upload_paths
import embeddings

# ``_env_int`` is a stdlib-only env helper owned by config.py; it is never
# swapped by a test, so a plain ``from`` import is safe and lets the bare
# ``_env_int(...)`` call sites moved here keep working unchanged.
from config import _env_int
# ``map_persistent_CAD_files`` is a stable storage helper (never swapped), read
# by the body-level embedding pass.
from storage import map_persistent_CAD_files
# Stable, never-rebound names owned by embeddings.py, reached bare by the code
# moved here (model-key constants, checkpoint hashing, the intra-op thread cap
# and the L2 normaliser). The swappable embedder names are reached as
# ``embeddings.<name>`` instead (see section 8.4), never imported here.
from embeddings import (
    _l2_normalize,
    ckpt_sha256,
    resolve_checkpoint_path,
    intraop_threads,
    intraop_thread_limit,
)
# Lock handles owned by locks.py.
from locks import _get_index_lock, _PROGRESS_STDERR_LOCK

logger = logging.getLogger(__name__)


# Named-index on-disk schema version.
#   1 = legacy: one averaged vector row per file (created before this migration)
#   2 = body-level: one vector row per body, with kind/bodies/thumbnail/scs/obb
# schema_version is recorded in indexes/<name>/index.json.
_INDEX_SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Named index registry (incremental index management)
# ---------------------------------------------------------------------------

# SDK verification notes (verified against hoops_ai.ml.embeddings):
#   - FaissVectorStore.load(path) is a CLASSMETHOD returning a new store instance.
#     Calling vs_instance.load(path) does NOT load data into the existing instance.
#   - upsert() accepts VectorRecord(id, embedding, metadata) where `embedding` must
#     be an Embedding(values=np.ndarray, model=str, dim=int) object, not a raw array.
#   - upsert() with a duplicate ID inserts a second entry in the FAISS index;
#     get_ids() deduplicates IDs but count() reflects raw FAISS entries.
#     Solution: always delete(id) before upserting an existing ID to avoid duplicates.
#   - Empty-index query() returns [] without error.
#   - HOOPSEmbeddings exposes an `embedding_dim` attribute for dimension discovery.

_INDEX_NAME_RE = re.compile(r"^[a-z0-9_-]{1,64}$")

# name -> FaissVectorStore, most-recently-used last. This is the WRITE store
# (updated on every create/add/remove); it is NOT read by search. Every write
# persists to disk under the per-index lock before returning, so an entry can be
# evicted and transparently reloaded on the next access. Bounded by MEMORY, not
# entry count, via HOOPS_AI_INDEX_CACHE_MAX_MB: index sizes vary wildly across
# deployments, so a fixed entry count would be meaningless.
_INDEX_CACHE_MAX_MB_DEFAULT = 1024
_named_indexes: "OrderedDict[str, Any]" = OrderedDict()
# Guards _named_indexes only. Leaf lock: never acquire another lock while
# holding it (see the LOCK ORDERING note in locks.py).
_named_indexes_mutex = threading.Lock()

# Cache of CADSearch instances for named indexes, keyed by (faiss path, mtime_ns).
# The mtime is part of the key so a rebuilt index is picked up automatically on
# the next request. Each instance loads the whole index into its own memory, so
# the cache is deliberately tiny; two entries let one index be searched while a
# second one is in use, without holding several copies of large indexes.
_NAMED_SEARCHER_CACHE_MAX = 2
_named_searchers: "OrderedDict[tuple[str, int], Any]" = OrderedDict()
def _validate_index_name(name: str) -> None:
    """Raise ValueError for invalid names."""
    if not _INDEX_NAME_RE.match(name):
        raise ValueError(
            f"Invalid index name '{name}'. "
            r"Must match ^[a-z0-9_-]{1,64}$ (lowercase alphanumerics, hyphens, underscores)."
        )


def _index_base_path(name: str) -> pathlib.Path:
    """Base path (no extension) passed to FaissVectorStore.save/load.
    Generates indexes/{name}/index.faiss and indexes/{name}/index.meta.
    """
    return config.INDEXES_DIR / name / "index"


def _index_faiss_path(name: str) -> pathlib.Path:
    return config.INDEXES_DIR / name / "index.faiss"


def _get_embedder_dim() -> int:
    """Return the embedding dimension from the lazy-loaded HOOPSEmbeddings model."""
    embedder = embeddings.get_embedder()
    dim = getattr(embedder, "embedding_dim", None)
    if dim is not None:
        return int(dim)
    raise RuntimeError(
        "Cannot determine embedding dimension from HOOPSEmbeddings. "
        "Ensure the embeddings model is correctly configured."
    )


def _index_cache_max_bytes() -> int:
    """Memory budget for the named-index write cache (HOOPS_AI_INDEX_CACHE_MAX_MB)."""
    return _env_int(
        "HOOPS_AI_INDEX_CACHE_MAX_MB", _INDEX_CACHE_MAX_MB_DEFAULT, minimum=1
    ) * 1024 * 1024


def _index_entry_bytes(name: str) -> int:
    """Approximate resident size of one cached index = its on-disk faiss+meta.

    The in-memory FaissVectorStore holds ``bodies x dim x 4`` bytes of vectors
    plus metadata, which is exactly what ``index.faiss`` + ``index.meta`` store
    on disk. Using the file sizes avoids any SDK call and is robust to schema
    changes. A missing file contributes 0.
    """
    total = 0
    for p in (_index_faiss_path(name), config.INDEXES_DIR / name / "index.meta"):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def _evict_named_indexes_to_cap(keep: str) -> None:
    """Drop least-recently-used stores until under the memory cap.

    Caller must hold ``_named_indexes_mutex``. Evicts oldest-first (insertion
    order is LRU order because reads ``move_to_end``). Never evicts *keep* (the
    entry just touched) and never empties the cache; a single index larger than
    the cap is kept resident with a warning rather than thrashing.
    """
    max_bytes = _index_cache_max_bytes()
    sizes = {n: _index_entry_bytes(n) for n in _named_indexes}
    cur = sum(sizes.values())
    for name in list(_named_indexes):  # oldest first
        if cur <= max_bytes:
            return
        if name == keep or len(_named_indexes) <= 1:
            continue
        _named_indexes.pop(name, None)
        cur -= sizes.get(name, 0)
        logger.info(
            "[CACHE] evicted named index '%s' (~%d MB); index cache now %d entries / ~%d MB",
            name, sizes.get(name, 0) // (1024 * 1024), len(_named_indexes), cur // (1024 * 1024),
        )
    if cur > max_bytes:
        logger.warning(
            "[CACHE] resident named indexes ~%d MB exceed HOOPS_AI_INDEX_CACHE_MAX_MB=%d MB; "
            "cannot evict the in-use index '%s'",
            cur // (1024 * 1024), max_bytes // (1024 * 1024), keep,
        )


def _store_named_index(name: str, vs: Any) -> None:
    """Publish *vs* as the cached store for *name* and enforce the memory cap."""
    with _named_indexes_mutex:
        _named_indexes[name] = vs
        _named_indexes.move_to_end(name)
        _evict_named_indexes_to_cap(keep=name)


def _load_named_index(name: str) -> Any:
    """Return the in-memory FaissVectorStore for `name`, loading from disk if needed.

    Must be called while holding the per-index lock (which serialises loads of
    the same name). Raises KeyError if the index does not exist on disk.
    """
    with _named_indexes_mutex:
        vs = _named_indexes.get(name)
        if vs is not None:
            _named_indexes.move_to_end(name)
            return vs

    faiss_file = _index_faiss_path(name)
    if not faiss_file.exists():
        raise KeyError(f"Index '{name}' does not exist.")

    from hoops_ai.ml.embeddings import FaissVectorStore

    vs = FaissVectorStore.load(str(_index_base_path(name)))
    _store_named_index(name, vs)
    return vs


def _save_named_index_atomic(name: str, vs: Any) -> None:
    """Persist *vs* to disk atomically using temp-file + replace pattern.

    Writes to ``_tmp_{uuid}.faiss`` / ``.meta`` in INDEXES_DIR and then calls
    ``Path.replace()`` to swap them in place.  replace() is atomic on POSIX and
    as close to atomic as Windows allows (non-atomic but crash-safe for our use case).
    """
    config.INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    index_dir = config.INDEXES_DIR / name
    index_dir.mkdir(parents=True, exist_ok=True)
    tmp_base = index_dir / f"_tmp_{uuid.uuid4().hex}"
    try:
        vs.save(str(tmp_base))
        pathlib.Path(str(tmp_base) + ".faiss").replace(_index_faiss_path(name))
        pathlib.Path(str(tmp_base) + ".meta").replace(config.INDEXES_DIR / name / "index.meta")
    except Exception:
        for suffix in (".faiss", ".meta"):
            tmp = pathlib.Path(str(tmp_base) + suffix)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        raise
def create_index(name: str) -> dict[str, Any]:
    """Create a new empty named index.

    The index uses the single embedding checkpoint the server resolves (see
    ``embeddings.resolve_checkpoint_path``). The checkpoint identity is recorded
    per record as each part is added, not on the empty index.

    Raises ValueError for invalid names, and FileExistsError if an index with
    that name already exists.
    """
    _validate_index_name(name)
    lock = _get_index_lock(name)
    with lock:
        if _index_faiss_path(name).exists():
            raise FileExistsError(f"Index '{name}' already exists.")

        from hoops_ai.ml.embeddings import FaissVectorStore

        dim = _get_embedder_dim()
        vs = FaissVectorStore(dim)
        _save_named_index_atomic(name, vs)
        _save_index_schema(name, _INDEX_SCHEMA_VERSION)
        _store_named_index(name, vs)
        return {"name": name, "count": 0, "dim": dim}


def list_indexes() -> list[dict[str, Any]]:
    """Return metadata for all known indexes."""
    result: list[dict[str, Any]] = []

    # Indexes named as startup-warmup targets (HOOPS_AI_WARMUP_INDEXES). "warm"
    # means "configured to be warmed at startup" -- the operator's signal for
    # which index matters -- not "already finished warming". Read from the raw
    # setting (no resolve/cap) to keep this a simple lookup and avoid recursion.
    warm_set = set(config.warmup_indexes_setting())

    # Named indexes from INDEXES_DIR -- each lives in its own subdirectory
    if config.INDEXES_DIR.exists():
        for faiss_file in sorted(config.INDEXES_DIR.glob("*/index.faiss")):
            idx_name = faiss_file.parent.name
            last_modified_idx: Optional[str] = None
            mtime = faiss_file.stat().st_mtime
            last_modified_idx = (
                datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            idx_count: Optional[int] = None
            lock = _get_index_lock(idx_name)
            with lock:
                try:
                    vs = _load_named_index(idx_name)
                    idx_count = len(vs.get_ids())
                except Exception:
                    # Listing indexes must not fail because one of them cannot
                    # be opened (corrupt, mid-write, built by another SDK
                    # version). That index is listed with count=None instead.
                    pass
            result.append(
                {
                    "name": idx_name,
                    "count": idx_count,
                    "last_modified": last_modified_idx,
                    "warm": idx_name in warm_set,
                }
            )

    return result


_EMBED_ERROR_DEFAULT = "embedding produced no rows (file skipped by embed_shape_batch)"


def _embed_error_detail(
    embed_errors: Any,
    fid: str,
    filename: str,
    norm_path: str,
) -> str:
    """Best-effort failure reason for *fid* from ``batch.metadata['errors']``.

    Observed on hoops_ai 1.1.0: a list of plain strings shaped like
    ``"Failed to compute embedding for <abs path>: <reason>"``. Other shapes
    (list of dicts, dict keyed by path) are still handled defensively because
    the contract is not documented. Always returns a non-empty string.

    Matching is path-first and filename matching is boundary-aware, so a file
    named ``0.stp`` never picks up the error belonging to ``100.stp``. Matched
    messages are returned verbatim -- they are short and already name the file.
    """
    default = _EMBED_ERROR_DEFAULT

    def _norm(p: str) -> str:
        try:
            return os.path.normcase(os.path.normpath(p))
        except Exception:
            return os.path.normcase(p)

    want_path = _norm(norm_path) if norm_path else ""
    want_file = os.path.normcase(filename) if filename else ""

    def _mentions(text: str) -> bool:
        if not text:
            return False
        low = _norm(text)
        if want_path and want_path in low:
            return True
        if fid and fid in text:
            return True
        if not want_file:
            return False
        # Boundary-aware filename search: reject "0.stp" inside "100.stp".
        start = 0
        while True:
            k = low.find(want_file, start)
            if k < 0:
                return False
            before_ok = k == 0 or low[k - 1] in "\\/\"' \t([<,"
            end = k + len(want_file)
            after_ok = end >= len(low) or not (
                low[end].isalnum() or low[end] in "._-"
            )
            if before_ok and after_ok:
                return True
            start = k + 1

    if isinstance(embed_errors, dict):
        for key, val in embed_errors.items():
            if _mentions(str(key)) and val:
                return str(val).strip() or default
    elif isinstance(embed_errors, (list, tuple)):
        for item in embed_errors:
            if isinstance(item, str):
                if _mentions(item):
                    return item.strip() or default
            elif isinstance(item, dict):
                text = " ".join(str(v) for v in item.values())
                if _mentions(text):
                    for k in ("detail", "message", "error", "reason"):
                        if item.get(k):
                            return str(item[k]).strip() or default
                    return text.strip() or default
    return default


# --- Embedding tuning: worker count and per-file time budget -----------------
#
# Ported from hoops_ai_native_bridge's ComputeAutoWorkers. Every worker is a
# spawned interpreter holding its own copy of the checkpoint (see
# _MODEL_FOOTPRINT_MB_DEFAULT for what that measures out at), so peak RSS
# grows roughly linearly with the worker count while throughput reaches a
# plateau and then declines once RAM-bound. Measured on the bridge: light
# single-body parts plateau near the physical core count (12 was the peak on a
# 14-core machine, 8..cores within +/-5%); heavy assemblies plateau lower (12
# peaked on a 16-core machine, and 16/20/24 were each slower than 8). So aim at
# the plateau rather than a single "best" value -- run-to-run noise exceeds the
# difference between neighbouring counts.
#
# The default cap of 12 is where both of those curves peaked: 12 on the 14-core
# machine with light parts, and 12 on the 16-core machine with heavy assemblies.
# It is a safety valve, not a target -- the physical core count and the free-RAM
# bound are what normally decide, and the cap only starts to bite above 12
# physical cores.
#
# The cap is not removed because "as many workers as cores" is measurably wrong
# on the heavy corpus: 16, 20 and 24 workers were each slower than 8 there. A
# host with more cores than that should be raised deliberately, after measuring
# its own corpus, via HOOPS_AI_MAX_WORKERS.
# The three variable names below are the ones hoops_ai and the bridge already
# use, deliberately kept identical so one environment configures both.
#
# HOOPS_AI_MIN_FILES_PARALLEL defaults to 1 here, i.e. disabled, unlike the
# bridge's 32. That threshold judges by file count and ignores how much work
# each file is: six heavy assemblies are measurably faster on three workers,
# and forcing them onto one would be a large regression. Its stated rationale
# -- that pool startup does not pay off for a few files -- is also weak, since
# the ~55-60 s startup measured here was for a 20-worker pool and scales with
# the worker count. Set it if your corpus is uniformly light.
_MAX_WORKERS_DEFAULT = 12

# What one worker of a normal pass is assumed to occupy, in MB. Measured, not
# estimated: a 14-physical-core / 32 GB host finished a 10,250-file
# registration on 12 workers at a peak of 25 GB (~79%). Twelve workers at the
# previous 2048 MB would have put 24 GB of baseline alone under that peak,
# which does not fit, so one worker costs about 1 GB and not two. The old 2048
# was a conservative guess, and its cost was concrete: it handed ram_cap = 6 to
# a machine that demonstrably completes on 12.
#
# This moves an estimate towards a measurement; it does not weaken the valve.
# The bound is still proportional to free RAM and still leaves
# _HOST_RAM_RESERVE_BYTES (2 GiB) to the host.
#
# Paired with _HEAVY_FOOTPRINT_FACTOR below: that factor is expressed as a
# multiple of THIS number, so the retry pass's per-worker budget is the product
# of the two. Changing either one alone changes that budget. See the factor's
# own comment for the arithmetic it has to keep true.
_MODEL_FOOTPRINT_MB_DEFAULT = 1024
_MIN_FILES_PARALLEL_DEFAULT = 1
_HOST_RAM_RESERVE_BYTES = 2 * 1024 * 1024 * 1024


def _physical_core_cap() -> tuple:
    """Return ``(cores, measured)`` -- the physical core count where known.

    ``logical // 2`` is only an approximation of the physical cores, and it is
    wrong in both directions on real hardware: a CPU without SMT has logical ==
    physical, so halving throws away half the machine, and on a P-core/E-core
    split the ratio is not two either. ``psutil.cpu_count(logical=False)``
    counts them instead, the same way hoops_ai_qt_sandbox's
    ``detectPhysicalCores()`` (SimilarityIndexPanel.cpp) does.

    This sharpens an approximation; it does not change the intent. psutil stays
    optional here (it is not a declared dependency), so the old halving remains
    the fallback when it is missing or reports nothing usable.
    """
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical and int(physical) > 0:
            return max(1, int(physical)), True
    except Exception:
        pass
    logical = os.cpu_count() or 2
    return max(1, logical // 2), False


# The retry pass registers only the files the first pass could not finish, which
# by definition are the heavy ones: large assemblies with far more bodies to
# tessellate and hold at once than the single-body parts that make up the bulk
# of a corpus. A worker on one of those occupies measurably more than the
# checkpoint HOOPS_AI_MODEL_FOOTPRINT_MB accounts for, so sizing the retry pass
# with the first pass's footprint oversubscribes RAM exactly where it hurts
# most.
#
# The number that matters is the per-worker budget, which is the PRODUCT of
# this factor and _MODEL_FOOTPRINT_MB_DEFAULT above: 4 x 1024 MB = 4 GB. That
# 4 GB is what hoops_ai_qt_sandbox's retry pass already allows for -- it sizes
# pass 2 as round(free RAM / 4 GiB) (SimilarityIndexPanel.cpp, lines 1783-1786)
# and measured 2..4 workers on real hosts. Same budget per worker, reached
# through this server's existing knob rather than a second hard-coded gigabyte
# figure, so HOOPS_AI_MODEL_FOOTPRINT_MB keeps moving both passes together.
#
# Hence the two constants move together. When the base was halved from 2048 to
# 1024 this factor was doubled from 2 to 4 in the same change, because halving
# the base alone would have quietly dropped the retry budget to 2 GB and
# falsified the sentence above. tests/test_embedding_plan.py pins the product.
_HEAVY_FOOTPRINT_FACTOR = 4

# The per-file budget for the retry pass, in seconds. Measured, not chosen:
# with it a retry over 276 timed-out files recovered 267 of them (9 still
# failed). It is also the value hoops_ai_qt_sandbox and the published write-up
# use for their second pass.
_RETRY_TIME_LIMIT_SECONDS = 1200

# What the SDK applies per file when no budget is passed to it. Mirrored here
# only so the server can SAY what will happen: HOOPS_AI_EMBED_TIME_LIMIT=0 means
# "do not pass one", and reporting that as a limit of "0 s" reads like "no
# time at all" or "unlimited" depending on the reader. Neither is true.
_SDK_DEFAULT_TIME_LIMIT_SECONDS = 120


def _size_workers(file_count, footprint_factor: int = 1) -> dict[str, Any]:
    """Size a worker pool for *file_count* files, and say what decided it.

    The single sizing rule, so that the number a run uses and the number the UI
    shows can never disagree. Returns the chosen count under ``workers`` plus
    the four figures it is the minimum of (``cap``, ``cores``, ``ram``,
    ``files``), which are what the ``[EMBED] auto workers=...`` log line
    reports; ``ram`` is ``None`` when there is no memory reading to report.

    *file_count* may be ``None``, meaning "not known yet". The file count only
    ever lowers the answer -- a pool never has more workers than files to give
    them -- so leaving it out of the minimum yields the ceiling the host alone
    allows, which is what a UI can show before a selection exists. ``files``
    comes back ``None`` to say so rather than guessing a number.

    *footprint_factor* is how much more memory one worker of this pass needs
    than HOOPS_AI_MODEL_FOOTPRINT_MB describes; 1 is the normal pass. It
    divides the memory bound directly. With no RAM reading available it divides
    the core bound instead, because that is then the only bound left on which
    "each worker now costs several times as much" can be stated at all --
    saying nothing would size a heavy retry exactly like a light first pass,
    which is the mistake the factor exists to prevent.
    """
    max_workers = _env_int("HOOPS_AI_MAX_WORKERS", _MAX_WORKERS_DEFAULT, minimum=1)
    model_mb = _env_int(
        "HOOPS_AI_MODEL_FOOTPRINT_MB", _MODEL_FOOTPRINT_MB_DEFAULT, minimum=1
    )
    factor = max(1, int(footprint_factor))
    core_cap, cores_measured = _physical_core_cap()

    ram_cap: Optional[int] = None
    available_gb = None
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
        available_gb = available / (1024 ** 3)
        usable = max(0, available - _HOST_RAM_RESERVE_BYTES)
        ram_cap = int(usable // (model_mb * factor * 1024 * 1024))
    except Exception:
        # No psutil (or an unsupported platform): fall back to the core cap
        # alone rather than guessing at memory.
        logger.debug("[EMBED] RAM probe unavailable; sizing workers by cores only")
        core_cap = max(1, core_cap // factor)

    bounds = [max_workers, core_cap]
    if file_count is not None:
        bounds.append(max(1, file_count))
    if ram_cap is not None:
        bounds.append(ram_cap)
    return {
        "workers": max(1, min(bounds)),
        "cap": max_workers,
        "cores": core_cap,
        "cores_measured": cores_measured,
        "ram": ram_cap,
        "available_gb": available_gb,
        "model_mb": model_mb * factor,
        "files": file_count,
    }


def compute_auto_workers(file_count: int) -> int:
    """Choose a worker count for *file_count* files.

    ``min(HOOPS_AI_MAX_WORKERS, physical_cores, usable_RAM / model_footprint,
    file_count)``. Physical cores rather than logical ones avoids E-core and SMT
    oversubscription; never exceeding the file count keeps idle workers from
    paying the checkpoint load for nothing.

    This is only consulted when the caller passes no explicit worker count. An
    explicit value is forwarded to the SDK unchanged, cap included -- the caps
    here exist to pick a safe default, not to overrule a deliberate choice.

    Deliberately never returns ``None`` (which would let the SDK auto-detect
    across all logical cores): that oversubscribes RAM, and on this corpus a
    20-worker run was ~2x slower per file than the bridge's 12.

    The physical core count is the measured throughput optimum once
    ``HOOPS_AI_INTRAOP_THREADS`` pins each worker to one intra-op thread (see
    :func:`intraop_threads`), so it is normally what decides -- together with the
    free-RAM bound. ``HOOPS_AI_MAX_WORKERS`` defaults to 12 because that is where
    both measured curves peaked (light parts on 14 cores, heavy assemblies on
    16); it is a safety valve against the heavy corpus getting slower past that
    point, and only bites on a host with more than 12 physical cores.
    """
    max_workers = _env_int("HOOPS_AI_MAX_WORKERS", _MAX_WORKERS_DEFAULT, minimum=1)
    model_mb = _env_int(
        "HOOPS_AI_MODEL_FOOTPRINT_MB", _MODEL_FOOTPRINT_MB_DEFAULT, minimum=1
    )
    min_parallel = _env_int(
        "HOOPS_AI_MIN_FILES_PARALLEL", _MIN_FILES_PARALLEL_DEFAULT, minimum=1
    )
    if file_count < min_parallel:
        return 1

    sizing = _size_workers(file_count)
    chosen = sizing["workers"]
    core_cap = sizing["cores"]
    ram_cap = sizing["ram"]

    # Free RAM is sampled at job start and fluctuates -- a preceding job's
    # workers may not have been reclaimed yet -- so a surprisingly low count
    # must be explainable from the log rather than looking arbitrary. The
    # cores= field says which of the two sources it came from, because a
    # psutil-less host silently halves its logical count and the resulting
    # worker count is otherwise indistinguishable from a measured one.
    logger.info(
        "[EMBED] auto workers=%d (cap=%d, cores=%d(%s), ram=%s, files=%d)",
        chosen, max_workers, core_cap,
        "physical" if sizing["cores_measured"] else "approx",
        ram_cap if ram_cap is not None else "n/a", file_count,
    )
    if ram_cap is not None and ram_cap <= 1 and min(max_workers, core_cap, file_count) > 1:
        logger.warning(
            "[EMBED] falling back to sequential embedding: only %.1f GB free, "
            "and each worker needs %d MB on top of a %.1f GB host reserve. "
            "Cores would allow %d. Wait for the previous job's workers to be "
            "reclaimed, or pass an explicit 'workers' to override.",
            sizing["available_gb"], model_mb,
            _HOST_RAM_RESERVE_BYTES / (1024 ** 3),
            min(max_workers, core_cap),
        )
    return chosen


def effective_embed_time_limit() -> int:
    """The per-file budget a run will really apply, in seconds.

    :func:`embed_time_limit` returns the raw setting, where ``0`` means "pass
    nothing to the SDK and let its own 120 s stand". That zero is correct as an
    argument and wrong as an answer: shown to a user or written into a report it
    reads as no budget at all. Everything that *reports* a budget uses this.
    """
    return embed_time_limit() or _SDK_DEFAULT_TIME_LIMIT_SECONDS


def plan_embedding_run(file_count=None) -> dict[str, Any]:
    """Resolve the worker count and per-file budget for both registration passes.

    The documented loop is two passes: a first one sized for throughput, then a
    retry of the files that ran out of time, sized for the heavy assemblies
    those always turn out to be. This resolves both with the same rules a job
    actually applies, so a client can show the real numbers instead of the word
    "auto" -- and so the rules stay in one place. A client that computes them
    itself goes stale the next time the sizing changes, which in this file has
    happened repeatedly.

    *file_count* may be omitted, which answers for the host alone: everything
    except the file count is settled before any files are chosen, and the file
    count can only lower the result, so a dialog can show real numbers from the
    moment it opens and refresh them when a selection arrives.

    Returns ``{"file_count", "first_pass": {...}, "retry_pass": {...}}`` where
    each pass carries ``workers``, ``time_limit`` and the ``basis`` the worker
    count is the minimum of (``cap`` / ``cores`` / ``ram`` / ``files``, plus
    whether the core count was measured or approximated). ``file_count`` and
    ``basis.files`` are ``None`` when none was given. Budgets are always
    effective values, never the raw ``0``.
    """
    count = None if file_count is None else max(0, int(file_count))
    first = _size_workers(count)
    retry = _size_workers(count, footprint_factor=_HEAVY_FOOTPRINT_FACTOR)

    def _pass(sizing: dict[str, Any], time_limit: int) -> dict[str, Any]:
        return {
            "workers": sizing["workers"],
            "time_limit": time_limit,
            "basis": {
                "cap": sizing["cap"],
                "cores": sizing["cores"],
                "cores_measured": bool(sizing["cores_measured"]),
                "ram": sizing["ram"],
                "files": sizing["files"],
                "model_mb": sizing["model_mb"],
            },
        }

    return {
        "file_count": count,
        "first_pass": _pass(first, effective_embed_time_limit()),
        # Longer per file and fewer at a time: the opposite trade from the first
        # pass, which is the whole point of resubmitting.
        "retry_pass": _pass(retry, _RETRY_TIME_LIMIT_SECONDS),
    }


def embed_time_limit() -> int:
    """Per-file embedding budget in seconds, from HOOPS_AI_EMBED_TIME_LIMIT.

    ``0`` (the default) leaves the SDK's own 120 s in place.
    """
    return _env_int("HOOPS_AI_EMBED_TIME_LIMIT", 0, minimum=0)


def read_error_summary_reasons(
    directory: Optional["pathlib.Path"] = None,
    newer_than: Optional[float] = None,
) -> dict[str, str]:
    """Per-file failure reasons from the SDK's ``error_summary.json``.

    This is the same source ``hoops_ai_qt_sandbox`` reads
    (``readErrorSummaryReasons`` in SimilarityIndexPanel.cpp): a JSON array of
    ``{"item_index", "item", "error"}`` objects where ``item`` is the absolute
    input path. Only the first line of ``error`` is kept.

    It is needed because ``batch.metadata['errors']`` is not always attributable.
    Measured on hoops_ai 1.1.0: a parallel run that times out reports bare reason
    strings with no path in them, so there is nothing to match a file against --
    and the list is in completion order, which is not the submission order, so
    zipping it positionally would mis-attribute reasons. ``error_summary.json``
    carries the path explicitly, so it is exact.

    The SDK logs ``Undefined flowdir ... Using path [.]`` and writes the file to
    the process working directory, overwriting it on every ``embed_shape_batch``
    call. *directory* is tried first in case a future SDK honours ``log_dir``.
    *newer_than* is a ``time.time()`` stamp taken before the batch started: an
    older file belongs to a previous run and is ignored rather than reported as
    if it described this one.

    Returns ``{normalised path: reason}``; empty when the file is absent,
    unreadable, stale or not in the expected shape.
    """
    import json

    candidates: list[pathlib.Path] = []
    if directory is not None:
        candidates.append(pathlib.Path(directory))
    candidates.append(pathlib.Path.cwd())

    for base in candidates:
        path = base / "error_summary.json"
        try:
            if newer_than is not None and path.stat().st_mtime < newer_than:
                logger.info(
                    "ignoring stale %s (written before this batch started)", path
                )
                continue
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                doc = json.load(fh)
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue

        if not isinstance(doc, list):
            logger.warning(
                "%s has unexpected shape %s (expected a list)",
                path,
                type(doc).__name__,
            )
            continue

        reasons: dict[str, str] = {}
        for entry in doc:
            if not isinstance(entry, dict):
                continue
            item = entry.get("item")
            if not item:
                continue
            reason = str(entry.get("error") or "").splitlines()
            reasons[_norm_path_key(str(item))] = reason[0].strip() if reason else ""
        return reasons
    return {}


def _norm_path_key(path: str) -> str:
    """Normalise a path for matching against the SDK's own log files.

    The SDK writes paths with either slash direction and, on Windows, in a case
    that need not match ours. Mirrors ``normPathKey`` in qt_sandbox.
    """
    try:
        return os.path.normcase(os.path.normpath(path))
    except Exception:
        return os.path.normcase(path)


def build_embed_specifications(
    images_dir: "pathlib.Path",
    time_limit: Optional[int] = None,
    log_dir: Optional["pathlib.Path"] = None,
) -> dict[str, Any]:
    """Build the ``specifications`` dict for ``embed_shape_batch``.

    All four time-limit keys are set to the same value. ``time_limit_overall``
    is the one that governs the per-item "Timeout (CUMULATIVE)" that drops heavy
    assemblies; setting only the size buckets leaves it at the 120 s default, so
    a nominally longer pass would still be killed at 120 s. This was verified
    empirically in hoops_ai_native_bridge, and a large per-item value does not
    abort the whole batch.
    """
    specs: dict[str, Any] = {
        "generate_images": True,
        "images_out_dir": str(images_dir),
    }
    if log_dir is not None:
        # Sends too_heavy_files.log to a per-pass directory so pass 2 does not
        # overwrite pass 1's list. 'log_dir' is a documented specification key
        # (hoops_ai 1.1.0, HOOPSEmbeddings._embed_shape_batch_parallel, default
        # '.'). Its effect is still unverified live: every run so far flagged no
        # heavy files at all, so readers keep the CWD fallback.
        specs["log_dir"] = str(log_dir)
    limit = time_limit if time_limit is not None else embed_time_limit()
    if limit and limit > 0:
        value = float(limit)
        specs["time_limit_overall"] = value
        specs["time_limit_small"] = value
        specs["time_limit_medium"] = value
        specs["time_limit_large"] = value
    return specs
class _TqdmProgressStderr:
    """``sys.stderr`` replacement that parses hoops_ai's tqdm bar.

    ``embed_shape_batch`` renders a tqdm bar on stderr that already carries
    everything a caller needs::

        Computing embeddings:  70%|####| 28/40 [01:06<00:22, 1.84s/it, pool=1, errors=18, heavy=0]

    Every write is mirrored to the real stderr and then parsed, so the server log
    is unchanged and the job record gains live counters. Ported from
    ``_BridgeProgressStderr`` in hoops_ai_native_bridge (hoops_ai_bridge.cpp),
    including ``isatty() -> True``: tqdm only emits incremental updates when it
    believes it is writing to a terminal, and the server's stderr may be a pipe
    or a service log.

    Parsing is strictly best-effort. Any failure is swallowed, because dropping a
    progress update must never break the embedding run.
    """

    _re_frac = re.compile(r"(\d+)\s*/\s*(\d+)")
    _re_err = re.compile(r"errors=(\d+)")
    _re_heavy = re.compile(r"heavy=(\d+)")

    def __init__(self, real: Any, report: Any) -> None:
        self._real = real
        self._report = report
        self._last: Optional[tuple] = None

    def write(self, s: str) -> int:
        try:
            if self._real is not None:
                self._real.write(s)
        except Exception:
            # This shim stands in for stderr while an embedding run is going.
            # Failing to pass a line through to the real stream is not worth
            # raising out of arbitrary library code that is only logging.
            pass
        try:
            if s and ("Computing embeddings" in s or "Heavy Files" in s):
                phase = "heavy" if "Heavy Files" in s else "embedding"
                m = self._re_frac.search(s)
                if m:
                    done, total = int(m.group(1)), int(m.group(2))
                    me = self._re_err.search(s)
                    mh = self._re_heavy.search(s)
                    key = (
                        phase,
                        done,
                        total,
                        int(me.group(1)) if me else -1,
                        int(mh.group(1)) if mh else -1,
                    )
                    # tqdm redraws the same line many times a second; only report
                    # when a counter actually moved, so the job record is not
                    # rewritten on every repaint.
                    if key != self._last:
                        self._last = key
                        self._report(*key)
        except Exception:
            pass
        return len(s) if s else 0

    def flush(self) -> None:
        try:
            if self._real is not None:
                self._real.flush()
        except Exception:
            # Same reason as write(): a stderr shim must never be the thing
            # that breaks the run it is only reporting on.
            pass

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        if self._real is not None:
            return self._real.fileno()
        raise OSError("no fileno")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


@contextmanager
def capture_embed_progress(report: Optional[Any]):
    """Install the tqdm-parsing stderr shim for the duration of the block.

    Yields ``True`` when the shim is active. Yields ``False`` when there is
    nothing to report to, or when another embedding run already holds the shim:
    ``sys.stderr`` is process-global, so two concurrent runs would interleave
    into one bar and each would report the other's counters. Registration jobs
    are serialised by default (see :func:`job_max_concurrency`), so the second
    case only arises if an operator raised the concurrency; progress is then
    dropped rather than reported wrongly.

    The original stderr is always restored, including on exceptions.
    """
    if report is None:
        yield False
        return
    if not _PROGRESS_STDERR_LOCK.acquire(blocking=False):
        logger.warning(
            "another embedding run holds the progress shim; "
            "live progress is disabled for this one"
        )
        yield False
        return
    real = sys.stderr
    try:
        sys.stderr = _TqdmProgressStderr(real, report)
        yield True
    finally:
        sys.stderr = real
        _PROGRESS_STDERR_LOCK.release()


def _embed_bodies_for_index(
    file_ids: list[str],
    index_name: str,
    num_workers: Optional[int] = None,
    time_limit: Optional[int] = None,
    log_dir: Optional["pathlib.Path"] = None,
    progress_cb: Optional[Any] = None,
) -> tuple[list, list[dict[str, Any]]]:
    """Embed *file_ids* at body granularity and build one VectorRecord per body.

    Calls ``embedder.embed_shape_batch()`` exactly once for the whole batch
    (performance-critical: never loop per file). Thumbnails (PNG) and stream
    caches (SCS) are produced by the same CAD load via ``generate_images`` and
    relocated to the canonical ``thumbnails/<file_id>.png`` / ``scs/<file_id>.scs``
    locations. Per-body vectors are L2-normalised individually (never averaged).

    Returns ``(records, errors)`` where *records* is a flat list of VectorRecord
    (multiple rows share the same file_id) and *errors* is a per-file error list.

    *num_workers* and *time_limit* are forwarded to the SDK; ``None`` means
    "decide here" (see ``compute_auto_workers`` and ``embed_time_limit``).
    *log_dir* redirects the SDK's ``too_heavy_files.log`` (see
    :func:`build_embed_specifications`).

    *progress_cb* is called as ``cb(phase, done, total, errors, heavy)`` while
    the batch runs, parsed from the SDK's tqdm bar (see
    :func:`capture_embed_progress`). ``None`` keeps the bar suppressed.
    """
    import json
    import shutil
    from collections import OrderedDict

    import numpy as np

    from hoops_ai.ml.embeddings import Embedding, VectorRecord

    embedder = embeddings.get_embedder()
    model_name = (
        getattr(embedder, "model_name", None)
        or getattr(embedder, "model", None)
        or "hoops_embeddings_model"
    )

    # Record the checkpoint identity on every body row (record-only). The digest
    # is cached per process, so this hashes the checkpoint at most once. Stored
    # under a leading-underscore key so _public_metadata() omits it from every
    # API response, keeping the response schema unchanged.
    ckpt_sha: Optional[str] = None
    try:
        ckpt_sha = ckpt_sha256(resolve_checkpoint_path())
    except Exception as exc:
        logger.warning(
            "Could not hash embedding checkpoint for index '%s': %s", index_name, exc
        )

    errors: list[dict[str, Any]] = []

    # Resolve real CAD paths; keep bidirectional maps keyed by file_id and by
    # the resolved path string (embed_shape_batch echoes input paths in .ids).
    fid_by_norm_path: dict[str, str] = {}
    norm_by_fid: dict[str, str] = {}
    filename_by_fid: dict[str, str] = {}
    ordered_paths: list[str] = []
    resolved_paths = map_persistent_CAD_files(file_ids)
    for fid in file_ids:
        cad_path = resolved_paths.get(fid)
        if cad_path is None:
            # Rare miss (e.g. removed after acceptance): reproduce the exact
            # per-file error message storage.find_persistent_CAD_file would have raised.
            try:
                cad_path = storage.find_persistent_CAD_file(fid)
            except Exception as exc:
                errors.append({"file_id": fid, "detail": str(exc)})
                continue
        norm = str(cad_path.resolve())
        fid_by_norm_path[norm] = fid
        norm_by_fid[fid] = norm
        parts = cad_path.name.split("_", 1)
        filename_by_fid[fid] = (
            parts[1] if len(parts) == 2 and len(parts[0]) == 64 else cad_path.name
        )
        ordered_paths.append(str(cad_path))

    if not ordered_paths:
        return [], errors

    def _fid_for_row(raw_id: str) -> Optional[str]:
        norm = str(pathlib.Path(str(raw_id)).resolve())
        if norm in fid_by_norm_path:
            return fid_by_norm_path[norm]
        # Single-input fallback: every row belongs to the only file.
        if len(fid_by_norm_path) == 1:
            return next(iter(fid_by_norm_path.values()))
        return None

    tmp_images = config.INDEXES_DIR / index_name / f"_tmp_images_{uuid.uuid4().hex}"
    tmp_images.mkdir(parents=True, exist_ok=True)

    records: list = []
    try:
        specs = build_embed_specifications(tmp_images, time_limit, log_dir=log_dir)
        workers = num_workers if num_workers is not None else compute_auto_workers(
            len(ordered_paths)
        )
        kwargs: dict[str, Any] = {"show_progress": False, "specifications": specs}
        if workers and workers > 0:
            kwargs["num_workers"] = int(workers)
        intraop = intraop_threads()
        logger.info(
            "[EMBED] index '%s': %d file(s), num_workers=%s, time_limit=%s, "
            "intraop_threads=%s",
            index_name, len(ordered_paths), kwargs.get("num_workers", "sdk-default"),
            specs.get("time_limit_overall", "sdk-default"),
            intraop if intraop > 0 else "sdk-default",
        )
        # Stamp taken before the batch so a leftover error_summary.json from an
        # earlier run is recognised as stale.
        batch_started = time.time()
        with intraop_thread_limit(intraop), capture_embed_progress(progress_cb) as live:
            # show_progress is deliberately left False. The bridge sets it True
            # to make the bar appear, but measured here on hoops_ai 1.1.0 the
            # parallel executor draws "Computing embeddings: 0/40 ..." on stderr
            # even with it False, so turning it on risks a second, outer bar
            # whose counters would interleave with the one we parse. If a future
            # SDK stops drawing it, progress simply stops advancing -- the run
            # itself is unaffected.
            _ = live
            batch = embedder.embed_shape_batch(ordered_paths, **kwargs)

        ids = list(getattr(batch, "ids", []) or [])
        values = np.asarray(getattr(batch, "values", []), dtype=np.float32)
        if values.ndim == 1 and values.size:
            values = values.reshape(1, -1)
        meta = getattr(batch, "metadata", None) or {}
        png_paths = meta.get("png_paths") if hasattr(meta, "get") else None
        kinds = meta.get("kind") if hasattr(meta, "get") else None
        obbs = meta.get("oriented_bounding_box") if hasattr(meta, "get") else None

        # Group row indices by file_id, preserving first-seen order.
        rows_by_fid: "OrderedDict[str, list[int]]" = OrderedDict()
        unmatched_rows = 0
        for i, raw_id in enumerate(ids):
            fid = _fid_for_row(raw_id)
            if fid is None:
                unmatched_rows += 1
                continue
            rows_by_fid.setdefault(fid, []).append(i)

        if unmatched_rows:
            logger.warning(
                "%d embedded row(s) could not be matched to an input file "
                "in index '%s'",
                unmatched_rows,
                index_name,
            )

        # Reorder rows_by_fid into the caller's *file_ids* order rather than
        # embed_shape_batch's completion order. embed_shape_batch spawns one
        # worker process per file (see the intraop_threads note above), so
        # batch.ids -- and therefore the first-seen order built just above --
        # reflects which worker finished first and varies from run to run. The
        # Shape Map clusters with FAISS k-means seeded from row numbers
        # (seed=42), so a different stored row order seats different vectors on
        # the seeded rows and converges to a different clustering even for the
        # identical file set. Writing rows in file_ids order makes the stored
        # order reproducible for the same file list submitted in the same order.
        # Per-file body rows keep their ascending row-number order (row_idx was
        # appended in increasing i). Any row that could not be attributed to a
        # file in file_ids (the single-input fallback in _fid_for_row) is kept,
        # after the ordered files, in its current relative order so nothing is
        # dropped. Both downstream loops (asset relocation and record building)
        # read this same OrderedDict, so they cannot drift apart.
        ordered_rows_by_fid: "OrderedDict[str, list[int]]" = OrderedDict()
        for fid in file_ids:
            if fid in rows_by_fid and fid not in ordered_rows_by_fid:
                ordered_rows_by_fid[fid] = rows_by_fid[fid]
        for fid, row_idx in rows_by_fid.items():
            if fid not in ordered_rows_by_fid:
                ordered_rows_by_fid[fid] = row_idx
        rows_by_fid = ordered_rows_by_fid

        # Surface files that embed_shape_batch silently skipped (CAD load error,
        # timeout, etc.): they never appear in batch.ids, so they are missing
        # from rows_by_fid. Report each one instead of dropping it silently.
        embed_errors = meta.get("errors") if hasattr(meta, "get") else None

        # A later phase routes retries on the reason text (e.g. "Timeout"), so a
        # silent fallback to the generic message must never go unnoticed. Warn
        # at most once per batch when the metadata stops matching the observed
        # contract (list[str] as of hoops_ai 1.1.0).
        if embed_errors is not None and not isinstance(embed_errors, list):
            logger.warning(
                "index '%s': batch metadata['errors'] has unexpected type %s "
                "(expected list[str]); embedding failure reasons may be unreliable",
                index_name,
                type(embed_errors).__name__,
            )

        submitted_fids = set(fid_by_norm_path.values())
        missing_fids = [
            fid for fid in file_ids if fid in submitted_fids and fid not in rows_by_fid
        ]
        # Fallback source for reasons the batch metadata cannot attribute. Read
        # once per batch, and only when there is something to explain.
        summary_reasons: dict[str, str] = {}
        if missing_fids:
            summary_reasons = read_error_summary_reasons(
                log_dir, newer_than=batch_started
            )
        matched_any_reason = False
        summary_hits = 0
        for fid in missing_fids:
            detail = _embed_error_detail(
                embed_errors,
                fid,
                filename_by_fid.get(fid, ""),
                norm_by_fid.get(fid, ""),
            )
            if detail != _EMBED_ERROR_DEFAULT:
                matched_any_reason = True
            elif summary_reasons:
                reason = summary_reasons.get(_norm_path_key(norm_by_fid.get(fid, "")))
                if reason:
                    detail = reason
                    summary_hits += 1
            errors.append({"file_id": fid, "detail": detail})

        if summary_hits:
            logger.info(
                "index '%s': recovered %d of %d failure reason(s) from "
                "error_summary.json",
                index_name,
                summary_hits,
                len(missing_fids),
            )

        if (
            missing_fids
            and isinstance(embed_errors, list)
            and embed_errors
            and not matched_any_reason
            and not summary_hits
        ):
            logger.warning(
                "index '%s': %d file(s) produced no rows but none matched any of "
                "the %d entries in batch metadata['errors'], and error_summary.json "
                "did not cover them either; embedding failure reasons are unreliable "
                "for this batch",
                index_name,
                len(missing_fids),
                len(embed_errors),
            )

        failed_count = meta.get("failed_count") if hasattr(meta, "get") else None
        logger.info(
            "index '%s' batch: input=%d rows=%d failed=%s",
            index_name,
            len(ordered_paths),
            len(ids),
            failed_count,
        )

        # Relocate PNG + SCS once per file (png_paths repeats per body row).
        thumb_dir = _index_thumbnails_dir(index_name)
        scs_dir = _index_scs_dir(index_name)
        thumbnail_rel: dict[str, str] = {}
        scs_rel: dict[str, str] = {}
        for fid, row_idx in rows_by_fid.items():
            src_png = None
            if png_paths:
                for i in row_idx:
                    if i < len(png_paths) and png_paths[i]:
                        p = pathlib.Path(png_paths[i])
                        # embed_shape_batch returns paths relative to images_out_dir.
                        if not p.is_absolute():
                            p = tmp_images / p
                        if p.exists():
                            src_png = p
                            break
            if src_png is None:
                continue
            try:
                thumb_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_png), str(thumb_dir / f"{fid}.png"))
                thumbnail_rel[fid] = f"thumbnails/{fid}.png"
                # SCS sibling shares the stem minus the "_white" suffix.
                stem = src_png.stem
                base_stem = stem[: -len("_white")] if stem.endswith("_white") else stem
                scs_src = src_png.parent / f"{base_stem}.scs"
                if scs_src.exists():
                    scs_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(scs_src), str(scs_dir / f"{fid}.scs"))
                    scs_rel[fid] = f"scs/{fid}.scs"
            except Exception as exc:  # non-fatal: search still works without assets
                logger.warning("Failed to relocate thumbnail/scs for %s: %s", fid, exc)

        # Build one record per body row.
        registered_at = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        obb_dropped_logged = False
        for fid, row_idx in rows_by_fid.items():
            bodies = len(row_idx)
            row_kind = None
            if kinds and row_idx[0] < len(kinds):
                row_kind = kinds[row_idx[0]]
            kind = row_kind or ("assembly" if bodies >= 2 else "part")

            base_meta: dict[str, Any] = {
                "file_id": fid,
                "filename": filename_by_fid.get(fid, ""),
                "registered_at": registered_at,
                "kind": kind,
                "bodies": bodies,
            }
            if ckpt_sha:
                base_meta["_ckpt_sha256"] = ckpt_sha
            if fid in thumbnail_rel:
                base_meta["thumbnail"] = thumbnail_rel[fid]
            if fid in scs_rel:
                base_meta["scs"] = scs_rel[fid]

            # oriented_bounding_box is file-level: hoops_ai returns it as a dict
            # keyed by the input path (== batch.ids entry), or as a per-row list.
            # It is the same for every body row of a file, so resolve it once.
            raw_id = ids[row_idx[0]]
            obb_val = None
            if isinstance(obbs, dict):
                obb_val = obbs.get(raw_id)
            elif isinstance(obbs, (list, tuple)) and row_idx[0] < len(obbs):
                obb_val = obbs[row_idx[0]]

            if obb_val is not None:
                obb_str = _obb_meta_value(obb_val)
                if obb_str is not None:
                    base_meta["obb"] = obb_str
                elif not obb_dropped_logged:
                    logger.warning(
                        "obb dropped for index '%s' (not serialisable or > 512 bytes)",
                        index_name,
                    )
                    obb_dropped_logged = True

            for i in row_idx:
                vec = _l2_normalize(values[i].astype(np.float32).flatten())
                emb_obj = Embedding(values=vec, model=model_name, dim=int(vec.shape[0]))
                records.append(
                    VectorRecord(id=fid, embedding=emb_obj, metadata=dict(base_meta))
                )
    finally:
        shutil.rmtree(tmp_images, ignore_errors=True)

    return records, errors


def add_to_index(
    name: str,
    file_ids: list[str],
    num_workers: Optional[int] = None,
    time_limit: Optional[int] = None,
    log_dir: Optional["pathlib.Path"] = None,
    progress_cb: Optional[Any] = None,
) -> dict[str, Any]:
    """Embed *file_ids* at body granularity and upsert one row per body.

    Re-inserting an existing file_id overwrites all of its previous body rows
    (delete-then-upsert). The single embedding checkpoint the server resolves is
    used, and its SHA-256 is stamped on every new record. If the index already
    holds records from a different checkpoint a WARNING is logged, because mixing
    checkpoints in one index makes similarity scores unreliable.

    Raises ``SchemaVersionError`` for legacy (schema v1) indexes, which must be
    rebuilt before body-level writes are allowed.

    *num_workers* and *time_limit* tune the SDK call; ``None`` keeps this
    module's defaults. *log_dir* redirects the SDK's ``too_heavy_files.log``.
    *progress_cb* receives live counters parsed from the SDK's progress bar.
    Existing callers that pass none of them are unaffected.

    Returns ``added``, ``updated``, ``index_count``, and per-file ``errors``.
    ``added``/``updated`` count distinct files (not body rows).
    """
    _validate_index_name(name)
    schema = _load_index_schema(name)
    lock = _get_index_lock(name)
    with lock:
        vs = _load_named_index(name)

        if schema["schema_version"] < _INDEX_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Index '{name}' uses legacy schema v{schema['schema_version']} "
                f"and must be rebuilt as schema v{_INDEX_SCHEMA_VERSION} "
                f"before adding parts."
            )

        # Warn when this write would mix checkpoints in one index. The digest is
        # process-cached, so hashing here is paid once and reused by the embed
        # pass below; this is a write path, so it never violates the "hash only
        # while writing an index" invariant.
        try:
            incoming = ckpt_sha256(resolve_checkpoint_path())
            existing_digests = {
                m.get("_ckpt_sha256")
                for m in vs.iter_metadata()
                if isinstance(m, dict) and m.get("_ckpt_sha256")
            }
            stale = existing_digests - {incoming}
            if stale:
                logger.warning(
                    "[MODEL] Index '%s' already holds vectors from %d other "
                    "checkpoint(s) (%s); the current checkpoint is %s. Mixing "
                    "checkpoints in one index makes similarity scores unreliable.",
                    name, len(stale),
                    ", ".join(sorted(str(s)[:12] for s in stale)), incoming[:12],
                )
        except Exception as exc:
            logger.warning(
                "Could not check checkpoint consistency for index '%s': %s", name, exc
            )

        records, errors = _embed_bodies_for_index(
            file_ids,
            name,
            num_workers=num_workers,
            time_limit=time_limit,
            log_dir=log_dir,
            progress_cb=progress_cb,
        )

        existing_ids: set[str] = set(vs.get_ids())
        distinct_fids: list[str] = []
        seen: set[str] = set()
        for rec in records:
            if rec.id not in seen:
                seen.add(rec.id)
                distinct_fids.append(rec.id)

        added = 0
        updated = 0
        for fid in distinct_fids:
            if fid in existing_ids:
                # delete first to prevent FAISS duplicate entries on re-insert
                vs.delete([fid])
                updated += 1
            else:
                added += 1

        if records:
            vs.upsert(records)

        _save_named_index_atomic(name, vs)
        result = {
            "name": name,
            "added": added,
            "updated": updated,
            "index_count": len(vs.get_ids()),
            "errors": errors,
        }

    return result
def _load_shape_index_pickle_compat(searcher: Any, faiss_index_path: pathlib.Path) -> Any:
    """Call ``searcher.load_shape_index()`` with cross-OS pickle compatibility.

    Index sidecars pickled on Windows contain WindowsPath objects (and vice
    versa). The foreign path class cannot be instantiated on the current OS
    (``WindowsPath`` on POSIX, ``PosixPath`` on Windows) and unpickling it fails
    with "cannot instantiate 'WindowsPath'/'PosixPath' on your system".
    Temporarily aliasing the foreign class to the concrete local one lets the
    stored path parts reconstruct so the same index file can be read anywhere.
    """
    if os.name == "nt":
        foreign_attr, local_cls = "PosixPath", pathlib.WindowsPath
    else:
        foreign_attr, local_cls = "WindowsPath", pathlib.PosixPath
    _orig = getattr(pathlib, foreign_attr)
    try:
        setattr(pathlib, foreign_attr, local_cls)
        return searcher.load_shape_index(path=str(faiss_index_path))
    finally:
        setattr(pathlib, foreign_attr, _orig)


def _evict_named_searchers(name: str) -> None:
    """Drop every cached CADSearch instance belonging to *name*."""
    path_key = str(_index_faiss_path(name))
    for stale in [k for k in _named_searchers if k[0] == path_key]:
        _named_searchers.pop(stale, None)


def _get_named_searcher(name: str) -> Any:
    """Return a CADSearch instance with the named index loaded, from cache.

    Creating a CADSearch and calling ``load_shape_index()`` is expensive, so
    instances are cached by (faiss path, mtime). A write to the index changes
    the mtime, which invalidates the entry automatically.

    Must be called while holding the per-index lock. The returned instance is
    safe to use *outside* the lock: ``load_shape_index()`` reads the index into
    the instance's own memory, so a concurrent ``_save_named_index_atomic()``
    replacing the file on disk cannot corrupt an already-loaded searcher.

    Raises ``KeyError`` when the index does not exist.
    """
    faiss_path = _index_faiss_path(name)
    if not faiss_path.exists():
        raise KeyError(f"Index '{name}' does not exist.")

    path_key = str(faiss_path)
    key = (path_key, faiss_path.stat().st_mtime_ns)
    cached = _named_searchers.get(key)
    if cached is not None:
        _named_searchers.move_to_end(key)
        return cached

    try:
        from hoops_ai.ml.embeddings import CADSearch
    except ImportError:  # older/newer layouts re-export it one level up
        from hoops_ai.ml import CADSearch

    searcher = CADSearch(shape_model=embeddings.get_embedder())
    _load_shape_index_pickle_compat(searcher, faiss_path)

    # Drop the previous generation of this index before inserting the new one.
    _evict_named_searchers(name)
    _named_searchers[key] = searcher
    while len(_named_searchers) > _NAMED_SEARCHER_CACHE_MAX:
        _named_searchers.popitem(last=False)
    return searcher


def _public_metadata(meta: Any) -> dict[str, Any]:
    """Return a copy of *meta* without SDK-internal keys.

    ``CADSearch.search_by_shape()`` decorates every hit with undocumented
    underscore-prefixed fields (``_query_body_rep_count``,
    ``_query_body_rep_indices``, ...). They are implementation details that may
    change between SDK releases, so they are stripped from the public response.
    The input dict is never mutated: hits may reference the store's own
    metadata objects.
    """
    if not isinstance(meta, dict):
        return {}
    return {k: v for k, v in meta.items() if not str(k).startswith("_")}


def _flatten_body_hit_objects(results: Any, top_k: int) -> list[Any]:
    """Flatten ``CADSearch.search_by_shape()`` output to VectorHit objects.

    *results* is ``List[List[VectorHit]]``: one already-ranked list per unique
    query body. Body lists are concatenated in order, de-duplicated by hit id,
    and capped at *top_k*.

    Scores are deliberately **not** re-sorted here: the per-body ranking
    produced by CADSearch is authoritative, and re-sorting on the raw score
    would diverge from the bridge implementation. The score CADSearch reports is
    the cosine similarity of the L2-normalised embeddings -- the same quantity
    :func:`compare_embeddings` computes. Measured on 2026-09-10 against a
    10,230-file index, three pairs matched to four decimal places (0.9028 /
    0.8748 / 0.8625) and all 301 hits came back in non-increasing score order.
    The 1.1 reranker refines the order of the top candidates without changing
    the score: the installed ``search_by_shape`` docstring states the matches
    "may be reranked with geometry signals ... hit.score is never modified", and
    ``scale_match_threshold=0.99`` adds an oriented-bounding-box size sort above
    that for high-confidence hits. Because the reranker reorders without
    touching the score, the returned order is not guaranteed to be monotonic in
    score, even though it was observed to be so for the 301 hits measured above.

    This returns the hit *objects* so callers that need more than the public
    ``{id, score, metadata}`` view (the Context Layer, which passes VectorHit
    objects straight to ``ContextPredictor.infer``) share the exact
    concatenate / de-duplicate / cap rules with :func:`_flatten_body_hits`.
    """
    k = int(top_k) if int(top_k) > 0 else 1
    seen: set[str] = set()
    out: list[Any] = []
    for body_hits in results or []:
        for h in body_hits or []:
            hid = str(getattr(h, "id", "") or "")
            if not hid or hid in seen:
                continue
            seen.add(hid)
            out.append(h)
            if len(out) >= k:
                return out
    return out


def _flatten_body_hits(results: Any, top_k: int) -> list[dict[str, Any]]:
    """Flatten ``CADSearch.search_by_shape()`` output, matching hoops_ai_native_bridge.

    Produces the public ``{id, score, metadata}`` view. The id de-duplication
    and top_k cap are delegated to :func:`_flatten_body_hit_objects` so the dict
    path and the object path cannot drift apart.
    """
    return [
        {
            "id": str(getattr(h, "id", "") or ""),
            "score": round(float(getattr(h, "score", 0.0) or 0.0), 6),
            "metadata": _public_metadata(getattr(h, "metadata", None)),
        }
        for h in _flatten_body_hit_objects(results, top_k)
    ]


# Tag filtering happens after the SDK has ranked, so a filtered search asks for
# more candidates than it needs. The multiplier is a compromise: high enough
# that a moderately common tag still fills top_k, low enough that the extra
# candidates cost little. The cap stops a large top_k from asking the SDK for
# an unbounded shortlist.
_TAG_FILTER_OVERFETCH = 5
_TAG_FILTER_OVERFETCH_MAX = 200


def _tag_filtered_fetch_k(k: int, tag_filter: Optional[dict[str, Any]]) -> int:
    """Return how many candidates to request for a *k*-hit tag-filtered search."""
    if not tag_filter:
        return k
    return min(max(k, k * _TAG_FILTER_OVERFETCH), _TAG_FILTER_OVERFETCH_MAX)
def get_index_stats(name: str) -> dict[str, Any]:
    """Return body-level statistics for the named index.

    Keys: ``name, files, bodies, assemblies, single_part, dim,
    schema_version, last_modified``. Works for both schema versions (a legacy
    index simply reports one body per file).

    Raises ``KeyError`` when the index does not exist.
    """
    _validate_index_name(name)
    schema = _load_index_schema(name)
    lock = _get_index_lock(name)
    with lock:
        vs = _load_named_index(name)
        stats = _compute_faiss_stats(vs)
        dim = getattr(vs, "dim", None)

    last_modified: Optional[str] = None
    faiss_file = _index_faiss_path(name)
    if faiss_file.exists():
        last_modified = (
            datetime.datetime.fromtimestamp(
                faiss_file.stat().st_mtime, tz=datetime.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        )

    return {
        "name": name,
        "files": stats["files"],
        "bodies": stats["bodies"],
        "assemblies": stats["assemblies"],
        "single_part": stats["single_part"],
        "dim": int(dim) if dim is not None else None,
        "schema_version": schema["schema_version"],
        "last_modified": last_modified,
    }


def check_index_model_consistency(name: str) -> dict[str, Any]:
    """Compare the checkpoint an index was built with against the current one.

    Every record stamped by ``_embed_bodies_for_index`` carries the SHA-256 of
    the checkpoint that produced it (``_ckpt_sha256``). This walks
    ``iter_metadata()`` once, collapses to one digest per file, and compares
    each against the digest of the checkpoint the server resolves right now.

    Hashing a multi-GB checkpoint takes a few seconds, so the current digest is
    computed here (once per process, cached by ``ckpt_sha256``). This is the
    only read path that hashes the checkpoint on purpose; search and startup
    never do. Callers should expect the first call to take a few seconds.

    Returns a dict with keys ``name``, ``current_ckpt_sha256``, ``total_files``,
    ``matching_files``, ``mismatching_files``, ``unknown_files`` (records with no
    stamped digest) and ``status``, one of:

      * ``"match"``            -- every file matches the current checkpoint.
      * ``"partial_mismatch"`` -- some match, some do not (the index was extended
        after the checkpoint was fine-tuned).
      * ``"mismatch"``         -- none match, or none carry a digest at all; the
        search results cannot be trusted.

    Raises ``KeyError`` when the index does not exist.
    """
    _validate_index_name(name)
    current = ckpt_sha256(resolve_checkpoint_path())
    lock = _get_index_lock(name)
    with lock:
        vs = _load_named_index(name)
        digest_by_fid: dict[str, Optional[str]] = {}
        for meta in vs.iter_metadata():
            if not isinstance(meta, dict):
                continue
            fid = meta.get("file_id")
            if fid and fid not in digest_by_fid:
                digest_by_fid[str(fid)] = meta.get("_ckpt_sha256")

    total = len(digest_by_fid)
    matching = sum(1 for d in digest_by_fid.values() if d == current)
    unknown = sum(1 for d in digest_by_fid.values() if not d)
    mismatching = total - matching - unknown

    if total > 0 and matching == total:
        status = "match"
    elif matching > 0:
        status = "partial_mismatch"
    else:
        status = "mismatch"

    return {
        "name": name,
        "current_ckpt_sha256": current,
        "total_files": total,
        "matching_files": matching,
        "mismatching_files": mismatching,
        "unknown_files": unknown,
        "status": status,
    }


def _require_index_exists(name: str) -> None:
    """Raise ``KeyError`` when the named index has no FAISS file on disk."""
    if not _index_faiss_path(name).exists():
        raise KeyError(f"Index '{name}' does not exist.")


def _registered_index_ids(name: str) -> set[str]:
    """Return the distinct file ids in the named index."""
    lock = _get_index_lock(name)
    with lock:
        vs = _load_named_index(name)
        return set(vs.get_ids())
def _index_thumbnails_dir(name: str) -> pathlib.Path:
    """Return the per-index thumbnail image directory path."""
    return config.INDEXES_DIR / name / "thumbnails"


def _index_model_sidecar_path(name: str) -> pathlib.Path:
    """Return the legacy per-index model sidecar JSON path (indexes/{name}/model.json).

    Only read for backward compatibility with schema-v1 indexes. New indexes
    record only their schema_version in ``index.json`` (see _index_json_path);
    the model is no longer recorded, since each record carries its own
    ``_ckpt_sha256`` checkpoint digest instead.
    """
    return config.INDEXES_DIR / name / "model.json"


def _index_json_path(name: str) -> pathlib.Path:
    """Return the per-index metadata JSON path (indexes/{name}/index.json)."""
    return config.INDEXES_DIR / name / "index.json"


def _index_scs_dir(name: str) -> pathlib.Path:
    """Return the per-index SCS stream-cache directory path."""
    return config.INDEXES_DIR / name / "scs"


def _load_index_schema(name: str) -> dict[str, Any]:
    """Return ``{"schema_version": <int>}`` for the named index.

    A legacy ``index.json`` may still carry a ``"model"`` field (``legacy`` /
    ``signal``); it is ignored -- the checkpoint identity now lives per record
    under ``_ckpt_sha256``. Only the schema version is read.

    Fallback order:
      1. ``index.json`` present  -> use it (body-level, schema 2 by default)
      2. only ``model.json``     -> ``{"schema_version": 1}``
      3. neither present         -> ``{"schema_version": 1}``
    """
    import json

    json_path = _index_json_path(name)
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return {
                "schema_version": int(data.get("schema_version", _INDEX_SCHEMA_VERSION)),
            }
        except Exception:
            # Corrupt index.json: treat as body-level.
            return {"schema_version": _INDEX_SCHEMA_VERSION}

    if _index_model_sidecar_path(name).exists():
        return {"schema_version": 1}

    return {"schema_version": 1}


def _save_index_schema(
    name: str,
    schema_version: int = _INDEX_SCHEMA_VERSION,
) -> None:
    """Persist ``index.json`` (``{"schema_version": ...}``).

    The checkpoint identity that lets a future release detect a model/index
    mismatch is stamped on each record's metadata under ``_ckpt_sha256`` (see
    ``_embed_bodies_for_index``); it is not duplicated here.
    """
    import json

    data: dict[str, Any] = {"schema_version": schema_version}
    path = _index_json_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def _obb_meta_value(value: Any) -> Optional[str]:
    """Return a JSON string for *value* suitable for storing in record metadata.

    Returns ``None`` (drop the key) when the value is not JSON-serialisable or
    when the serialised form exceeds 512 bytes, keeping index.meta compact.
    """
    import json

    try:
        serialized = json.dumps(value)
    except (TypeError, ValueError):
        return None
    if len(serialized.encode("utf-8")) > 512:
        return None
    return serialized


def _resolve_index_asset(name: str, part_id: str, asset: str) -> Optional[pathlib.Path]:
    """Resolve a per-part thumbnail/scs file, guarding against path traversal.

    *asset* is ``"thumbnail"`` (thumbnails/<id>.png) or ``"scs"`` (scs/<id>.scs).
    Returns the file Path only when it exists AND resolves inside indexes/<name>/;
    otherwise returns ``None``.
    """
    _validate_index_name(name)
    index_dir = (config.INDEXES_DIR / name).resolve()
    if asset == "thumbnail":
        candidate = (config.INDEXES_DIR / name / "thumbnails" / f"{part_id}.png").resolve()
    elif asset == "scs":
        candidate = (config.INDEXES_DIR / name / "scs" / f"{part_id}.scs").resolve()
    else:
        return None
    try:
        candidate.relative_to(index_dir)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _compute_faiss_stats(vs: Any) -> dict[str, int]:
    """Derive file/body/assembly/single-part counts from a FaissVectorStore.

    Reads the underlying IndexIDMap directly (verified against hoops_ai 1.1.0).
    """
    import faiss
    from collections import Counter

    index = vs._index
    idm = faiss.vector_to_array(index.id_map)
    bodies = int(index.ntotal)
    counts = Counter(int(x) for x in idm.tolist())
    files = len(vs._id_to_int)
    assemblies = sum(1 for v in counts.values() if v >= 2)
    single = sum(1 for v in counts.values() if v == 1)
    return {"files": files, "bodies": bodies, "assemblies": assemblies, "single_part": single}
class SchemaVersionError(RuntimeError):
    """Raised when a legacy (schema v1) index cannot serve the request.

    Mapped to HTTP 409 by the router. Legacy indexes store one averaged vector
    per file, so body-level writes and assembly search both require a rebuild.
    """
