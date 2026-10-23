"""Assembly-to-assembly search over the vendored AssemblyMatcher.

``search_assembly_index`` ranks whole assemblies against a query assembly by
running the vendored ``AssemblyMatcher`` (vendor/assembly_matcher.py) exactly
as hoops_ai_native_bridge does: a per-body Hungarian matching blended with a
bag-of-parts composition score over an index-backed candidate shortlist. It is
a two-stage search -- a plain FAISS shortlist, then the matcher's scoring --
and building a matcher is far more expensive than a plain searcher because it
also runs a FAISS k-means over every body vector (~19 s at 42k bodies).

Matcher cache. ``_assembly_matchers`` holds at most one AssemblyMatcher, keyed
by ``(faiss path, mtime_ns)`` just like the searcher cache, so re-registering
an index (which moves the ``.faiss`` mtime) invalidates it automatically. Only
one entry is kept because a matcher owns a CADSearch plus an L2-normalised copy
of every body vector, roughly twice the on-disk index size. ``prewarm_assembly_matcher``
rebuilds it off the request path after a registration job so the next searcher
does not pay the k-means; only indexes that already had a matcher are rebuilt.

Dependency direction. This module depends only on the standard library,
``storage`` (find_persistent_CAD_file), ``embeddings`` (get_embedder,
log_compute_environment), ``locks`` (_get_index_lock, _get_assembly_build_lock),
``config`` (_search_timing_enabled), ``diagnostics`` (_PhaseTimer),
``index_registry`` and ``tags``. The four registry helpers
``_index_faiss_path`` / ``_load_index_schema``
/ ``_load_named_index`` / ``_load_shape_index_pickle_compat`` are (A) names and
are reached as ``index_registry.<name>`` (module attribute); the stable (C)
names ``_INDEX_SCHEMA_VERSION`` / ``_public_metadata`` / ``_tag_filtered_fetch_k``
/ ``_validate_index_name`` / ``SchemaVersionError`` are imported by name. The
vendored ``AssemblyMatcher`` and the SDK ``CADSearch`` are imported lazily inside
``_get_assembly_matcher`` so importing this module never pulls in the SDK. It
must never import ``core``: the dependency runs one way, ``core`` -> ``assembly``
-> ``tags`` / ``index_registry`` / ``locks`` / ``storage`` / ``embeddings``.

Why the tag helpers are imported BY NAME, never ``import tags``.
``search_assembly_index`` has a parameter called ``tags``. Binding the ``tags``
module at module level would let that parameter shadow the module for the whole
function body, so any later ``tags.<attr>`` would hit the argument list instead
-- the exact UnboundLocalError-class footgun behind #61. The three helpers
``normalize_tag_filter`` / ``tag_filter_predicate`` / ``tag_filter_diagnostics``
are stable (C) names, so ``from tags import ...`` is both safe and sufficient.
For the same reason the local that receives ``tag_filter_diagnostics(...)`` is
named ``tag_diag`` (renamed from ``diagnostics`` in #61); do not rename it back.
That rule is now load-bearing twice over: ``diagnostics`` is bound at module
level for ``_PhaseTimer``, so a local of that name inside
``search_assembly_index`` would shadow the module for the whole function body
and every ``diagnostics._PhaseTimer`` would raise UnboundLocalError. No local
in that function may be called ``diagnostics``.

Reference convention (see "The re-export convention" in docs/architecture.md).
``_get_assembly_matcher``, ``prewarm_assembly_matcher`` and
``search_assembly_index`` are the three (A)
names: ``routers/similarity.py`` reaches ``assembly.search_assembly_index`` and
tests swap all three ``assembly.<name>`` attributes to install fakes and assert
the orchestrators drove them. They are therefore NOT re-exported by ``core``;
re-exporting would bind a second object under ``core`` while this module keeps
using its own, so a test swap and the live path would diverge -- the double-truth
trap. The core orchestrators ``_warmup_one`` and ``run_index_add_job`` call them
as ``assembly._get_assembly_matcher`` / ``assembly.prewarm_assembly_matcher``.
Every other name here is stable and is re-exported from ``core`` by explicit
name for its existing callers (including ``delete_index``, which calls
``_evict_assembly_matchers``).
"""

import logging
import os
# Referenced only from string annotations ("pathlib.Path"). Kept so that
# typing.get_type_hints() can still resolve them.
import pathlib
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

import config
import diagnostics
import embeddings
import index_registry
import storage

# The two per-index locks this module needs live in locks.py: the shared index
# lock (also held by part search) and the assembly-matcher build lock.
from locks import _get_assembly_build_lock, _get_index_lock
# Stable (C) names in index_registry. The (A) names (_index_faiss_path,
# _load_index_schema, _load_named_index, _load_shape_index_pickle_compat) are
# reached as ``index_registry.<name>`` (module attribute), never imported.
from index_registry import (
    _INDEX_SCHEMA_VERSION,
    _public_metadata,
    _tag_filtered_fetch_k,
    _validate_index_name,
    SchemaVersionError,
)
# Stable (C) tag-filter helpers imported BY NAME on purpose: search_assembly_index
# has a ``tags`` parameter, so ``import tags`` here would let that argument shadow
# the module inside the function (the #61 footgun). Never ``import tags`` here.
from tags import (
    normalize_tag_filter,
    tag_filter_diagnostics,
    tag_filter_predicate,
)


logger = logging.getLogger(__name__)


# --- Assembly-to-assembly search (vendor/assembly_matcher.py) --------------
# Accepted coverage denominators. "symmetric" = matched / larger side,
# "containment" = matched / query size, "jaccard" = matched / union.
_ASSEMBLY_COVERAGE_MODES = frozenset({"symmetric", "containment", "jaccard"})

# An id with at least this many body rows counts as an assembly. Matches both
# hoops_ai_native_bridge and _compute_faiss_stats().
_MIN_BODIES_FOR_ASSEMBLY = 2

# Cache of AssemblyMatcher instances, keyed by (faiss path, mtime_ns) exactly
# like _named_searchers, so an index write invalidates it automatically.
# Only ONE entry is kept (hoops_ai_native_bridge keeps one too): a matcher owns
# a CADSearch *plus* an L2-normalised copy of every body vector, roughly twice
# the on-disk index size, which is far too large to hold several of.
_ASSEMBLY_MATCHER_CACHE_MAX = 1
_assembly_matchers: "OrderedDict[tuple[str, int], Any]" = OrderedDict()

# Guards _assembly_matchers itself. Always the innermost lock: never acquire
# anything else while holding it.
_assembly_matchers_mutex = threading.Lock()

# The assembly matcher build lock (_assembly_build_locks) now lives in locks.py.

# Index names whose matcher is being rebuilt in the background, so a burst of
# jobs on one index cannot stack rebuilds of the same corpus.
_assembly_prewarm_in_flight: set[str] = set()
_assembly_prewarm_mutex = threading.Lock()

# Thread-pool size for the matcher's stage-2 scoring. The bridge hard-codes 8;
# here it is configurable because those threads compete with the server's own
# request workers.
_ASSEMBLY_SEARCH_JOBS_DEFAULT = 8

# Bounds for the tunable scoring parameters, enforced before reaching the SDK.
_ASSEMBLY_CANDIDATE_K_MAX = 1000


# ---------------------------------------------------------------------------
# Assembly-to-assembly search
# ---------------------------------------------------------------------------


def _assembly_search_jobs() -> int:
    """Thread-pool size for AssemblyMatcher stage-2 scoring.

    Read from HOOPS_AI_ASSEMBLY_SEARCH_JOBS on every call so the value can be
    tuned without restarting. Invalid or non-positive values fall back to the
    default rather than failing a search.
    """
    raw = os.getenv("HOOPS_AI_ASSEMBLY_SEARCH_JOBS", "").strip()
    if not raw:
        return _ASSEMBLY_SEARCH_JOBS_DEFAULT
    try:
        jobs = int(raw)
    except ValueError:
        logger.warning(
            "Invalid HOOPS_AI_ASSEMBLY_SEARCH_JOBS=%r; using %d.",
            raw, _ASSEMBLY_SEARCH_JOBS_DEFAULT,
        )
        return _ASSEMBLY_SEARCH_JOBS_DEFAULT
    if jobs < 1:
        logger.warning(
            "HOOPS_AI_ASSEMBLY_SEARCH_JOBS=%d is out of range; using %d.",
            jobs, _ASSEMBLY_SEARCH_JOBS_DEFAULT,
        )
        return _ASSEMBLY_SEARCH_JOBS_DEFAULT
    return jobs


def _evict_assembly_matchers_locked(name: str) -> None:
    """Drop every cached AssemblyMatcher for *name*. Caller holds the mutex."""
    path_key = str(index_registry._index_faiss_path(name))
    for stale in [k for k in _assembly_matchers if k[0] == path_key]:
        _assembly_matchers.pop(stale, None)


def _evict_assembly_matchers(name: str) -> None:
    """Drop every cached AssemblyMatcher belonging to *name*."""
    with _assembly_matchers_mutex:
        _evict_assembly_matchers_locked(name)


def _lookup_assembly_matcher(faiss_path: "pathlib.Path") -> tuple[Any, tuple[str, int]]:
    """Return ``(matcher_or_None, cache_key)`` for the index's current mtime."""
    key = (str(faiss_path), faiss_path.stat().st_mtime_ns)
    with _assembly_matchers_mutex:
        cached = _assembly_matchers.get(key)
        if cached is not None:
            _assembly_matchers.move_to_end(key)
    return cached, key


def _get_assembly_matcher(name: str) -> Any:
    """Return an AssemblyMatcher for the named index, from cache.

    Building one is far more expensive than a plain searcher: on top of loading
    the index it runs ``build_part_rarity_weights()``, a FAISS k-means over
    every body vector in the corpus -- 19.3 s at 42,098 bodies. The construction
    time is logged so the real cost stays visible.

    **Must not be called while holding the per-index lock**; this function takes
    that lock itself, for the disk read only. Splitting it that way is the point:
    ``_save_named_index_atomic()`` replaces ``.faiss`` and ``.meta`` in two
    steps, so a reader outside the lock could pair a new index with an old
    metadata file (and on Windows the replace can fail outright against an open
    handle) -- but the k-means afterwards touches nothing but memory, and holding
    the index lock across it would stall part search on an index nobody is
    writing to.

    Concurrent callers are serialised on a per-index build lock, so they produce
    one build rather than one each. The returned instance is safe to use outside
    every lock for the same reason as :func:`_get_named_searcher`: it holds its
    own copy of the corpus, so a later save cannot disturb an in-flight search.
    The next request sees a new mtime and rebuilds.

    Raises ``KeyError`` when the index does not exist.
    """
    faiss_path = index_registry._index_faiss_path(name)
    if not faiss_path.exists():
        raise KeyError(f"Index '{name}' does not exist.")

    cached, _ = _lookup_assembly_matcher(faiss_path)
    if cached is not None:
        return cached

    with _get_assembly_build_lock(name):
        # Re-check: whoever held the build lock has just published a matcher,
        # and it is current unless the index changed again in the meantime.
        if not faiss_path.exists():
            raise KeyError(f"Index '{name}' does not exist.")
        cached, _ = _lookup_assembly_matcher(faiss_path)
        if cached is not None:
            return cached

        try:
            from hoops_ai.ml.embeddings import CADSearch
        except ImportError:  # older/newer layouts re-export it one level up
            from hoops_ai.ml import CADSearch

        from vendor.assembly_matcher import AssemblyMatcher

        started = time.perf_counter()

        # Phase 1 (index lock): read the index off disk, paired with its meta.
        with _get_index_lock(name):
            if not faiss_path.exists():
                raise KeyError(f"Index '{name}' does not exist.")
            key = (str(faiss_path), faiss_path.stat().st_mtime_ns)
            searcher = CADSearch(shape_model=embeddings.get_embedder())
            batch = index_registry._load_shape_index_pickle_compat(searcher, faiss_path)

        # Phase 2 (unlocked): the k-means, which is the expensive part and works
        # only on the copy loaded above.
        matcher = AssemblyMatcher(
            searcher=searcher,
            embedding_batch=batch,
            min_bodies_for_assembly=_MIN_BODIES_FOR_ASSEMBLY,
        )
        matcher.build_part_rarity_weights()
        logger.info(
            "[ASSEMBLY] built matcher for index '%s' in %.2fs",
            name, time.perf_counter() - started,
        )

        # Keyed by the mtime read at load time: if the index was written while
        # the k-means ran, this entry is born stale and is simply never served.
        with _assembly_matchers_mutex:
            _evict_assembly_matchers_locked(name)
            _assembly_matchers[key] = matcher
            while len(_assembly_matchers) > _ASSEMBLY_MATCHER_CACHE_MAX:
                _assembly_matchers.popitem(last=False)
        return matcher


def _has_cached_assembly_matcher(name: str) -> bool:
    """Whether a matcher for *name* is held, current or stale.

    Presence is the signal that assembly search has actually been used on this
    index: nothing else puts an entry here. A stale entry (one whose mtime no
    longer matches) counts, because it is exactly what a registration job has
    just invalidated.
    """
    path_key = str(index_registry._index_faiss_path(name))
    with _assembly_matchers_mutex:
        return any(k[0] == path_key for k in _assembly_matchers)


def assembly_prewarm_enabled() -> bool:
    """Whether to rebuild the assembly matcher after a registration job.

    From HOOPS_AI_ASSEMBLY_PREWARM, default on. Turning it off costs the first
    searcher after each job the full build time instead.
    """
    raw = os.getenv("HOOPS_AI_ASSEMBLY_PREWARM", "").strip().lower()
    if not raw:
        return True
    return raw in ("1", "true", "yes", "on")


def _rebuild_assembly_matcher(name: str) -> None:
    """Body of the background rebuild. Never raises."""
    try:
        schema = index_registry._load_index_schema(name)
        if schema["schema_version"] < _INDEX_SCHEMA_VERSION:
            # Assembly search rejects legacy indexes, so a matcher would never
            # be used.
            return
        started = time.perf_counter()
        # No index lock here: _get_assembly_matcher takes it itself, for the
        # disk read only, so part search is not stalled for the k-means.
        _get_assembly_matcher(name)
        logger.info(
            "[ASSEMBLY] prewarmed matcher for index '%s' in %.2fs",
            name, time.perf_counter() - started,
        )
    except KeyError:
        # Index deleted between the job finishing and this thread running.
        logger.info("[ASSEMBLY] prewarm skipped: index '%s' no longer exists", name)
    except Exception as exc:
        # A failed prewarm only means the next searcher pays the build cost.
        logger.warning("[ASSEMBLY] prewarm failed for index '%s': %s", name, exc)
    finally:
        with _assembly_prewarm_mutex:
            _assembly_prewarm_in_flight.discard(name)


def prewarm_assembly_matcher(name: str) -> bool:
    """Rebuild the assembly matcher for *name* in the background.

    A registration job changes the ``.faiss`` mtime, which invalidates the
    cached matcher; rebuilding it costs a FAISS k-means over every body vector
    in the corpus -- 24 s at 42,098 bodies -- and that cost would otherwise land
    on whoever searches next. On a shared server that is a different person from
    the one who indexed, which is the whole reason to move it here.

    Only indexes that already had a matcher are rebuilt (see
    :func:`_has_cached_assembly_matcher`), so a deployment that never runs
    assembly search never pays for one. Only one rebuild per index is in flight
    at a time. Returns whether a rebuild was started.

    A searcher arriving mid-rebuild waits on the same per-index build lock and
    then finds the matcher cached, rather than starting a second build of the
    same corpus. No stale matcher can be served while it runs, because the cache
    is keyed by the index's mtime.

    **Only registration jobs call this, not the synchronous ``add_to_index``.**
    That path is used one file at a time (the MCP server adds singly), so
    rebuilding on every call would put a ~20 s k-means behind a one-file add and
    let prewarming dominate the cost of registering. Warming the synchronous
    path would need a debounce -- one rebuild once the adds have stopped -- not
    this hook.
    """
    if not assembly_prewarm_enabled():
        return False
    if not _has_cached_assembly_matcher(name):
        logger.debug(
            "[ASSEMBLY] prewarm skipped for index '%s': assembly search unused",
            name,
        )
        return False
    with _assembly_prewarm_mutex:
        if name in _assembly_prewarm_in_flight:
            return False
        _assembly_prewarm_in_flight.add(name)
    thread = threading.Thread(
        target=_rebuild_assembly_matcher,
        args=(name,),
        name=f"assembly-prewarm-{name}",
        daemon=True,
    )
    thread.start()
    return True


def _assembly_metadata_by_id(vs: Any) -> dict[str, dict[str, Any]]:
    """Map file_id -> stored metadata in a single pass over the index.

    ``iter_metadata()`` yields one entry per body row, so the first entry per
    file wins; per-file fields (filename, kind, bodies, ...) are identical
    across the rows of a file.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for meta in vs.iter_metadata():
        if not isinstance(meta, dict):
            continue
        fid = meta.get("file_id")
        if fid and fid not in by_id:
            by_id[str(fid)] = dict(meta)
    return by_id


def _assembly_hit(result: Any, meta_by_id: dict[str, dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Convert one AssemblyMatcher result dict into a response hit.

    The matcher's key names are kept verbatim on its side and renamed here:
    ``matched`` -> matched_parts, ``n_parts`` -> candidate_parts (bodies in the
    candidate) and ``M`` -> query_parts (bodies in the query). Returns None for
    anything unusable so one odd row cannot fail the whole search.
    """
    if not isinstance(result, dict):
        return None
    hid = str(result.get("assembly") or "")
    if not hid:
        return None

    def _num(key: str) -> float:
        try:
            return float(result.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _count(key: str) -> int:
        try:
            return int(result.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    return {
        "id": hid,
        "score": round(_num("score"), 6),
        "geom_score": round(_num("geom_score"), 6),
        "coverage": round(_num("coverage"), 6),
        "matched_parts": _count("matched"),
        "candidate_parts": _count("n_parts"),
        "query_parts": _count("M"),
        "metadata": _public_metadata(meta_by_id.get(hid)) or {"file_id": hid},
    }


def _assembly_self_hit(file_id: str, meta: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Build the synthetic self hit inserted when include_self is set.

    An assembly is a perfect match for itself, so the scores are 1.0 and every
    body counts as matched. The values are synthetic, never produced by the
    matcher, because the query is excluded from scoring (see
    :func:`search_assembly_index`).
    """
    public = _public_metadata(meta)
    try:
        bodies = int(public.get("bodies") or 0)
    except (TypeError, ValueError):
        bodies = 0
    return {
        "id": file_id,
        "score": 1.0,
        "geom_score": 1.0,
        "coverage": 1.0,
        "matched_parts": bodies,
        "candidate_parts": bodies,
        "query_parts": bodies,
        "metadata": public or {"file_id": file_id},
    }


def search_assembly_index(
    name: str,
    file_id: str,
    top_k: int,
    candidate_k: int = 30,
    sim_thresh: float = 0.80,
    bop_weight: float = 0.30,
    coverage_mode: str = "symmetric",
    use_idf: bool = True,
    include_self: bool = False,
    tags: Optional[list[str]] = None,
    tag_mode: str = "any",
    tagged: Optional[bool] = None,
) -> dict[str, Any]:
    """Search the named index for assemblies similar to the assembly *file_id*.

    Runs the vendored ``AssemblyMatcher`` (see vendor/assembly_matcher.py) with
    exactly the arguments hoops_ai_native_bridge uses, so both projects rank the
    same corpus identically: a per-body Hungarian matching, blended with a
    bag-of-parts composition score, over an index-backed candidate shortlist.

    *coverage_mode* is ``"symmetric"`` (matched mass over the larger side),
    ``"containment"`` (over the query, i.e. "which assemblies contain this
    one?") or ``"jaccard"``. Out-of-range parameters raise ``ValueError``.

    Like :func:`search_index`, the query cannot be dropped by the SDK itself --
    ``AssemblyMatcher.search()`` discards candidates equal to the query *path*
    while records are keyed by SHA-256 -- so the self hit is removed explicitly
    here and only re-inserted, at the front, when *include_self* is set. One
    extra result is requested to compensate, and the total never exceeds
    *top_k*.

    A query that is already indexed is passed to the matcher by its record id
    (the file_id) rather than by path, so ``reuse_index_vectors=True`` reuses
    the stored per-body vectors instead of re-embedding the CAD file.

    Returns ``{"hits": [...], "count": n, "image_url": None}``; the hits
    carry the matcher's diagnostics (geom_score, coverage, matched_parts,
    candidate_parts, query_parts) alongside the stored metadata. ``image_url``
    is always None; clients render their own gallery from the returned hits.

    *tags* / *tag_mode* / *tagged* narrow the hits by the tag sidecar, applied
    after the matcher has ranked and with the same over-fetch policy and the
    same caveat as :func:`search_index`: the result can fall short of *top_k*.

    Raises ``KeyError`` when the index does not exist.
    """
    _validate_index_name(name)
    if coverage_mode not in _ASSEMBLY_COVERAGE_MODES:
        raise ValueError(
            f"Invalid coverage_mode '{coverage_mode}'. "
            f"Must be one of: {sorted(_ASSEMBLY_COVERAGE_MODES)}."
        )
    k = int(top_k) if int(top_k) > 0 else 1
    candidate_k = int(candidate_k)
    if not 1 <= candidate_k <= _ASSEMBLY_CANDIDATE_K_MAX:
        raise ValueError(
            f"Invalid candidate_k {candidate_k}. Must be between 1 and "
            f"{_ASSEMBLY_CANDIDATE_K_MAX}."
        )
    sim_thresh = float(sim_thresh)
    if not 0.0 <= sim_thresh <= 1.0:
        raise ValueError(f"Invalid sim_thresh {sim_thresh}. Must be between 0.0 and 1.0.")
    bop_weight = float(bop_weight)
    if not 0.0 <= bop_weight <= 1.0:
        raise ValueError(f"Invalid bop_weight {bop_weight}. Must be between 0.0 and 1.0.")
    tag_filter = normalize_tag_filter(tags, tag_mode, tagged)
    fetch_k = _tag_filtered_fetch_k(k, tag_filter)
    timer = diagnostics._PhaseTimer() if config._search_timing_enabled() else None
    if timer:
        # Dump the compute environment (interpreter, torch/CUDA build, thread
        # settings) so a slow timed request can be explained before any change.
        embeddings.log_compute_environment("SEARCH_ENV")
    if timer:
        timer.lap("setup")

    schema = index_registry._load_index_schema(name)
    if timer:
        timer.lap("load_schema")
    if schema["schema_version"] < _INDEX_SCHEMA_VERSION:
        # A legacy index stores one averaged vector per file, so every entry
        # looks like a single-body part: assemblies_only would discard the whole
        # corpus and the search would silently return nothing useful.
        raise SchemaVersionError(
            f"Index '{name}' uses legacy schema v{schema['schema_version']} and "
            f"has no per-body vectors. Rebuild it as schema v{_INDEX_SCHEMA_VERSION} "
            f"to use assembly search."
        )
    lock = _get_index_lock(name)

    # Phase 1 (locked): read the metadata map. Resolving the matcher is left
    # outside the lock deliberately -- it takes the index lock itself for its
    # disk read, then runs the expensive k-means unlocked -- so a cold assembly
    # search does not stall part search on the same index for ~20 s.
    with lock:
        if timer:
            timer.lap("lock_acquire")
        vs = index_registry._load_named_index(name)
        if timer:
            timer.lap("load_named_index")
        registered_ids = set(vs.get_ids())
        if timer:
            timer.lap("get_ids")
        if not registered_ids:
            if timer:
                # Emit here too: without it the one case that returns before any
                # work -- an empty index -- would still leave an empty log, the
                # very silence this instrumentation exists to end.
                timer.emit(
                    "ASSEMBLY_TIMING",
                    {
                        "index": name,
                        "file_id": file_id,
                        "registered_ids": 0,
                        "top_k": k,
                        "empty_index": True,
                    },
                )
            return {"hits": [], "count": 0, "image_url": None}
        meta_by_id = _assembly_metadata_by_id(vs)
        if timer:
            timer.lap("metadata_map")
    matcher_cached = _has_cached_assembly_matcher(name) if timer else False
    matcher = _get_assembly_matcher(name)
    if timer:
        timer.lap("get_matcher")

    if file_id in registered_ids:
        # AssemblyMatcher keys its per-file rows by record id, which here is the
        # SHA-256 file_id. Passing the id lets reuse_index_vectors=True hit the
        # stored per-body vectors instead of re-reading and re-embedding the CAD
        # file (~1.16s -> ~0.16s warm). It also makes the matcher's own
        # candidates.discard(query_path) work, which it cannot do with a path
        # because our records are keyed by hash rather than by location.
        matcher_query = file_id
    else:
        # Not indexed: hand over the real path so the matcher falls back to
        # embedder.embed_shape().
        matcher_query = str(storage.find_persistent_CAD_file(file_id))
    if timer:
        timer.lap("resolve_query")

    # Phase 2 (unlocked): the fixed arguments mirror hoops_ai_native_bridge.
    results = matcher.search(
        query_path=matcher_query,
        top_k=fetch_k + 1,
        candidate_k=candidate_k,
        sim_thresh=sim_thresh,
        method="hungarian",
        use_idf=bool(use_idf),
        candidate_mode="search",
        assemblies_only=True,
        reuse_index_vectors=True,
        coverage_mode=coverage_mode,
        bop_weight=bop_weight,
        n_jobs=_assembly_search_jobs(),
    )
    if timer:
        timer.lap("matcher_search")

    hit_dicts: list[dict[str, Any]] = []
    for result in results or []:
        hit = _assembly_hit(result, meta_by_id)
        if hit is not None and hit["id"] != file_id:
            hit_dicts.append(hit)
    if timer:
        timer.lap("build_hits")

    candidates_examined = len(hit_dicts)

    if tag_filter:
        matches = tag_filter_predicate(name, tag_filter)
        hit_dicts = [h for h in hit_dicts if matches(h["id"])]
    if timer:
        timer.lap("tag_filter")

    if include_self and file_id in registered_ids:
        # The query was filtered out above, so this cannot duplicate a hit. It
        # is subject to the tag filter for the same reason as in search_index().
        if not tag_filter or tag_filter_predicate(name, tag_filter)(file_id):
            hit_dicts.insert(0, _assembly_self_hit(file_id, meta_by_id.get(file_id)))
    hit_dicts = hit_dicts[:k]

    response: dict[str, Any] = {
        "hits": hit_dicts,
        "count": len(hit_dicts),
        "image_url": None,
    }
    tag_diag = tag_filter_diagnostics(
        name,
        tag_filter,
        # Only assemblies are searchable here, so counting parts too would
        # overstate how many results the filter could ever have produced.
        [pid for pid, meta in meta_by_id.items() if (meta or {}).get("kind") == "assembly"],
        candidates_examined,
    )
    if tag_diag is not None:
        response["tag_filter"] = tag_diag
    if timer:
        timer.lap("finalize")
        timer.emit(
            "ASSEMBLY_TIMING",
            {
                "index": name,
                "file_id": file_id,
                "registered_ids": len(registered_ids),
                "top_k": k,
                "candidate_k": candidate_k,
                "coverage_mode": coverage_mode,
                "bop_weight": bop_weight,
                "query_in_index": file_id in registered_ids,
                "matcher_cached": matcher_cached,
                "candidates_examined": candidates_examined,
                "hits": len(hit_dicts),
            },
        )
    return response
