"""Part-search entry point and its finished-response cache.

``search_index`` is the public part-to-part similarity search behind
``POST /indexes/{name}/search``. A repeat search with identical parameters is
about 20 s of pure re-encode, so a finished response is cached (#42). The cache
key carries the ``.faiss`` file mtime AND the tag sidecar ETag, so the entry
invalidates itself automatically when the index is re-registered (mtime moves)
or a part is tagged (ETag changes); ``_search_cache_key`` reaches the sidecar as
``tags._read_tags_document``. Stored and returned values are deep copies, so a
caller mutating a response cannot poison the cache. ``_search_cache_mutex`` is a
leaf mutex: never acquire another lock while holding it.

Dependency direction. This module depends only on the standard library,
``config`` (_env_int, _search_timing_enabled), ``diagnostics`` (_PhaseTimer),
``storage`` (find_persistent_CAD_file), ``embeddings`` (log_compute_environment),
``locks`` (_get_index_lock), ``index_registry`` and ``tags``. The three
searcher/path helpers ``_get_named_searcher`` / ``_index_faiss_path`` /
``_load_named_index`` are (A) names in index_registry
and are reached as ``index_registry.<name>`` (module attribute); the stable (C)
names ``_flatten_body_hits`` / ``_public_metadata`` / ``_tag_filtered_fetch_k`` /
``_validate_index_name`` are imported by name. ``_read_tags_document`` is the (A)
name in tags and is reached as ``tags._read_tags_document``. It must never import
``core``: the dependency runs one way, ``core`` -> ``search`` -> ``tags`` ->
``index_registry``.

Reference convention (see "The re-export convention" in docs/architecture.md).
``search_index`` and ``_search_cache_enabled`` are the two (A) names:
``routers/similarity.py`` reaches
them as ``search.search_index`` / ``search._search_cache_enabled`` and tests swap
those same ``search.<name>`` attributes to install fakes and to force the cache
on/off. They are therefore
NOT re-exported by ``core``. Re-exporting them would bind a second object under
``core`` while this module keeps using its own, so a test swap and the live code
path would diverge -- the double-truth trap. Every other name here is stable and
is re-exported from ``core`` by explicit name for its existing callers.
"""

import copy
import os
import threading
from collections import OrderedDict
from typing import Any, Optional

import config
import diagnostics
import index_registry
import storage
import tags

# ``_env_int`` is a stdlib-only env helper owned by config.py; never swapped.
from config import _env_int
# ``log_compute_environment`` dumps the interpreter/torch/CUDA build for a slow
# timed request; a stable (C) name owned by embeddings.py.
from embeddings import log_compute_environment
# ``_get_index_lock`` is the per-index lock factory owned by locks.py.
from locks import _get_index_lock
# Stable (C) names in index_registry. The (A) names (_get_named_searcher,
# _index_faiss_path, _load_named_index) are reached as
# ``index_registry.<name>`` (module attribute), never imported.
from index_registry import (
    _flatten_body_hits,
    _public_metadata,
    _tag_filtered_fetch_k,
    _validate_index_name,
)
# Stable (C) tag-filter helpers. ``_read_tags_document`` is an (A) name in tags
# and is reached as ``tags._read_tags_document`` (module attribute), never here.
from tags import (
    normalize_tag_filter,
    tag_filter_diagnostics,
    tag_filter_predicate,
)


# Accepted values for the named-index search `kind` filter. "any" means no
# filter is passed to CADSearch at all (preserves pre-filter behaviour).
_SEARCH_KINDS = frozenset({"any", "part", "assembly"})

# Cache of finished part-search responses (see search_index / #42). A repeat
# search with identical parameters is ~20 s of pure re-encode today; caching
# the response makes it instant. The key carries the .faiss mtime AND the tag
# sidecar ETag so re-registering or tagging invalidates it automatically. The
# stored value is a deep copy, and a deep copy is returned, so a caller mutating
# the response cannot poison the cache. Leaf mutex: never acquire another lock
# while holding it. Bounded by entry count (HOOPS_AI_SEARCH_CACHE_MAX); each
# entry is a small list of hit dicts, so a count cap is enough.
_SEARCH_CACHE_MAX_DEFAULT = 256
_search_cache: "OrderedDict[tuple, dict[str, Any]]" = OrderedDict()
_search_cache_mutex = threading.Lock()


# ---------------------------------------------------------------------------
# Part-search response cache (#42)
# ---------------------------------------------------------------------------


def _search_cache_enabled() -> bool:
    """Whether the part-search response cache is on.

    Controlled by ``HOOPS_AI_SEARCH_CACHE`` (default on); set it to ``0`` to
    measure the uncached path in a benchmark.
    """
    return os.environ.get("HOOPS_AI_SEARCH_CACHE", "1").strip() != "0"


def _search_cache_capacity() -> int:
    """Maximum number of cached search responses (``HOOPS_AI_SEARCH_CACHE_MAX``)."""
    return _env_int("HOOPS_AI_SEARCH_CACHE_MAX", _SEARCH_CACHE_MAX_DEFAULT, minimum=1)


def clear_search_cache() -> None:
    """Drop every cached search response (used by the verification script)."""
    with _search_cache_mutex:
        _search_cache.clear()


def _tag_filter_key(tag_filter: Optional[dict[str, Any]]) -> tuple:
    """A stable, hashable representation of a normalized tag filter.

    Tags are sorted because tag order never changes the result, so ``["a","b"]``
    and ``["b","a"]`` should share a cache entry.
    """
    if not tag_filter:
        return ()
    return (
        tuple(sorted(tag_filter.get("tags") or [])),
        tag_filter.get("tag_mode"),
        tag_filter.get("tagged"),
    )


def _search_cache_key(
    name: str,
    file_id: str,
    k: int,
    kind: str,
    include_self: bool,
    tag_filter: Optional[dict[str, Any]],
) -> Optional[tuple]:
    """Build the cache key, or return None when the index has no .faiss yet.

    The ``.faiss`` mtime makes re-registering invalidate the entry (the same key
    the searcher cache uses). The tag sidecar ETag makes tagging invalidate it:
    ``_write_tags_document`` touches only the JSON sidecar, never the ``.faiss``,
    so without the ETag a tag-filtered response would go stale.
    """
    try:
        faiss_mtime = index_registry._index_faiss_path(name).stat().st_mtime_ns
    except OSError:
        return None
    try:
        _doc, tags_etag = tags._read_tags_document(name)
    except Exception:
        # An unreadable sidecar must not be cached under a shared key: skip the
        # cache so the request is computed fresh every time.
        return None
    return (
        name,
        faiss_mtime,
        tags_etag,
        file_id,
        int(k),
        kind,
        bool(include_self),
        _tag_filter_key(tag_filter),
    )


def _search_cache_get(key: tuple) -> Optional[dict[str, Any]]:
    """Return a deep copy of the cached response for *key*, or None."""
    with _search_cache_mutex:
        entry = _search_cache.get(key)
        if entry is None:
            return None
        _search_cache.move_to_end(key)
        return copy.deepcopy(entry)


def _search_cache_put(key: tuple, value: dict[str, Any]) -> None:
    """Store a deep copy of *value* under *key*, evicting the LRU past capacity."""
    with _search_cache_mutex:
        _search_cache[key] = copy.deepcopy(value)
        _search_cache.move_to_end(key)
        while len(_search_cache) > _search_cache_capacity():
            _search_cache.popitem(last=False)


def search_index(
    name: str,
    file_id: str,
    top_k: int,
    kind: str = "any",
    include_self: bool = False,
    tags: Optional[list[str]] = None,
    tag_mode: str = "any",
    tagged: Optional[bool] = None,
) -> dict[str, Any]:
    """Search the named index for the top-k most similar parts to *file_id*.

    Runs through ``CADSearch.search_by_shape()`` so that queries are handled at
    body granularity and go through the SDK's second-stage reranking, matching
    both the preset search and hoops_ai_native_bridge. The score returned is the
    plain embedding cosine (see :func:`_flatten_body_hit_objects`).
    Results are flattened with
    :func:`_flatten_body_hits` (concatenate per-body rankings, de-duplicate by
    file_id, cap at *top_k*) without re-sorting.

    *kind* is ``"any"`` (no filter), ``"part"`` or ``"assembly"``; anything else
    raises ``ValueError``.

    *tags* / *tag_mode* / *tagged* narrow the hits by the tag sidecar. The
    filter is applied **after** the SDK has ranked, because CADSearch has no
    knowledge of tags; scores and relative order are therefore untouched. To
    compensate, a tag-filtered search asks the SDK for several times *top_k*
    candidates (see ``_TAG_FILTER_OVERFETCH``). That makes a short result
    unlikely but not impossible: if fewer than *top_k* of the over-fetched
    candidates carry the tag, fewer than *top_k* hits are returned. Raising the
    over-fetch further would start to cost more than it buys.

    ``CADSearch.search_by_shape()`` does *not* drop the query from its own
    candidates here: records are keyed by SHA-256 while the query is passed as
    an ``uploads/<sha256>_<name>`` path, so the ids never match. The self hit is
    therefore removed explicitly unless *include_self* is set, in which case it
    is promoted to the front with score 1.0 instead. Either way the result never
    exceeds *top_k*.

    Returns an empty hits list when the index has no entries. ``image_url`` is
    always None; clients render their own gallery from the returned hits.
    """
    _validate_index_name(name)
    if kind not in _SEARCH_KINDS:
        raise ValueError(
            f"Invalid kind '{kind}'. Must be one of: {sorted(_SEARCH_KINDS)}."
        )
    tag_filter = normalize_tag_filter(tags, tag_mode, tagged)
    timer = diagnostics._PhaseTimer() if config._search_timing_enabled() else None
    if timer:
        # Dump the compute environment (interpreter, torch/CUDA build, thread
        # settings) so a slow timed request can be explained before any change.
        log_compute_environment("SEARCH_ENV")
    if timer:
        timer.lap("setup")

    k = int(top_k) if int(top_k) > 0 else 1

    # Result cache (before any index lock, so a hit costs nothing): a repeat
    # search with identical parameters and an unchanged index+tags returns the
    # exact response the code below would build, without re-encoding the query.
    cache_enabled = _search_cache_enabled()
    cache_key = None
    if cache_enabled:
        cache_key = _search_cache_key(
            name, file_id, k, kind, include_self, tag_filter
        )
        if cache_key is not None:
            cached = _search_cache_get(cache_key)
            if cached is not None:
                if timer:
                    timer.lap("cache_lookup")
                    timer.emit(
                        "SEARCH_TIMING",
                        {
                            "index": name,
                            "file_id": file_id,
                            "kind": kind,
                            "top_k": k,
                            "cache": "HIT",
                        },
                    )
                return cached

    lock = _get_index_lock(name)

    # Phase 1 (locked): resolve the cached searcher only. The expensive part --
    # embedding the query and searching -- runs unlocked below so that indexing
    # a batch no longer blocks every search against the same index.
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
            return {"hits": [], "count": 0, "image_url": None}
        self_meta: Optional[dict[str, Any]] = None
        if include_self and file_id in registered_ids:
            for meta in vs.iter_metadata():
                if meta.get("file_id") == file_id:
                    self_meta = dict(meta)
                    break
        if timer:
            timer.lap("self_meta_scan")
        searcher = index_registry._get_named_searcher(name)
        if timer:
            timer.lap("get_named_searcher")

    query_path = storage.find_persistent_CAD_file(file_id)
    if timer:
        timer.lap("storage.find_persistent_CAD_file")

    # Phase 2 (unlocked): search_by_shape() embeds the query internally, so the
    # embedding cannot be hoisted out of this call. Ask for one extra candidate
    # because the query itself usually occupies a slot and is dropped below.
    fetch_k = _tag_filtered_fetch_k(k, tag_filter)
    search_kwargs: dict[str, Any] = {"top_k": fetch_k + 1}
    if kind != "any":
        search_kwargs["filters"] = {"kind": kind}
    results = searcher.search_by_shape(str(query_path), **search_kwargs)
    if timer:
        timer.lap("search_by_shape")

    hit_dicts = _flatten_body_hits(results, fetch_k + 1)
    hit_dicts = [h for h in hit_dicts if h["id"] != file_id]
    if timer:
        timer.lap("flatten_body_hits")
    candidates_examined = len(hit_dicts)
    if tag_filter:
        matches = tag_filter_predicate(name, tag_filter)
        hit_dicts = [h for h in hit_dicts if matches(h["id"])]
    if timer:
        timer.lap("tag_filter")

    if include_self and file_id in registered_ids:
        # Put the query back at the front so clients can tag the whole displayed
        # cluster. It was removed above, so this cannot produce a duplicate. The
        # tag filter applies to it as well: a response must not contain a part
        # that fails the filter the caller asked for.
        if not tag_filter or tag_filter_predicate(name, tag_filter)(file_id):
            hit_dicts.insert(
                0,
                {
                    "id": file_id,
                    "score": 1.0,
                    "metadata": _public_metadata(self_meta) or {"file_id": file_id},
                },
            )
    hit_dicts = hit_dicts[:k]

    result: dict[str, Any] = {
        "hits": hit_dicts,
        "count": len(hit_dicts),
        "image_url": None,
    }
    tag_diag = tag_filter_diagnostics(
        name, tag_filter, registered_ids, candidates_examined
    )
    if tag_diag is not None:
        result["tag_filter"] = tag_diag
    if cache_enabled and cache_key is not None:
        _search_cache_put(cache_key, result)
    if timer:
        timer.lap("finalize")
        timer.emit(
            "SEARCH_TIMING",
            {
                "index": name,
                "file_id": file_id,
                "registered_ids": len(registered_ids),
                "kind": kind,
                "top_k": k,
                "query_in_index": file_id in registered_ids,
                "query_bodies": len(results or []),
                "cache": "MISS",
            },
        )
    return result
