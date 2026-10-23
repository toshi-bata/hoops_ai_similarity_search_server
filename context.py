"""Context Layer: predict production records from nearest neighbours.

The context sidecar (indexes/<name>/context.json) and all of its read/write
machinery live in ``context_store.py``, which imports neither this module nor
``hoops_ai``. Only ``predict_context`` needs the searcher -- it fetches the
nearest neighbours of a part and asks the SDK to score candidate field values --
so only the prediction and its thin sidecar wrappers live here.

Dependency direction. This module depends on the standard library, ``config``
(INDEXES_DIR, _env_int), ``context_store`` (the sidecar body, reached as
``context_store.<name>``), ``storage`` (find_persistent_CAD_file), ``locks``
(_get_index_lock), ``index_registry``, ``tags`` and ``search``. The (A) names
``_get_named_searcher`` / ``_load_named_index`` /
``_resolve_index_asset`` are reached as ``index_registry.<name>``; the stable (C)
names ``_require_index_exists`` / ``_validate_index_name`` / ``_public_metadata``
/ ``_flatten_body_hit_objects`` / ``_tag_filtered_fetch_k`` are imported by name.
It must never import ``core``: the dependency runs one way, ``core`` ->
``context`` -> ``search``/``tags``/``index_registry``.

``_SEARCH_KINDS`` is imported ``from search`` because ``predict_context`` accepts
the same ``kind`` filter as a search and must validate it against the one set of
accepted kinds; keeping a single owner (search.py) avoids two lists drifting
apart. This is the only edge from ``context`` to ``search``. All names here are
stable (C) names and are re-exported from ``core`` by explicit name for their
existing ``core.<name>`` callers.
"""

from typing import Any, Optional

import config
import context_store
import index_registry
import storage

# ``_env_int`` is a stdlib-only env helper owned by config.py; never swapped.
from config import _env_int
# ``_get_index_lock`` is the per-index lock factory owned by locks.py.
from locks import _get_index_lock
# Stable (C) names in index_registry. The (A) names (_get_named_searcher,
# _load_named_index, _resolve_index_asset) are reached as
# ``index_registry.<name>`` (module attribute), never imported.
from index_registry import (
    _flatten_body_hit_objects,
    _public_metadata,
    _require_index_exists,
    _tag_filtered_fetch_k,
    _validate_index_name,
)
# Stable (C) tag-filter helpers owned by tags.py.
from tags import normalize_tag_filter, tag_filter_predicate
# The accepted ``kind`` set is owned by search.py; see the module docstring.
from search import _SEARCH_KINDS


# A predicted key set is bounded so one request cannot ask the SDK to score an
# unbounded number of fields.
_PREDICT_KEYS_DEFAULT = 32


def max_predict_keys() -> int:
    """Upper bound on keys per prediction (``HOOPS_AI_MAX_CONTEXT_PREDICT_KEYS``)."""
    return _env_int(
        "HOOPS_AI_MAX_CONTEXT_PREDICT_KEYS",
        _PREDICT_KEYS_DEFAULT,
        minimum=1,
        maximum=1000,
    )


def _normalize_predict_keys(keys: Any) -> list[str]:
    """Validate the requested key list, preserving order and de-duplicating."""
    if isinstance(keys, str) or not isinstance(keys, (list, tuple)):
        raise ValueError("keys must be a list of strings.")
    out: list[str] = []
    for key in keys:
        cleaned = context_store.normalize_key(key)
        if cleaned not in out:
            out.append(cleaned)
    if not out:
        raise ValueError("At least one key to predict is required.")
    limit = max_predict_keys()
    if len(out) > limit:
        raise ValueError(
            f"{len(out)} keys requested; the maximum per prediction is {limit} "
            f"(HOOPS_AI_MAX_CONTEXT_PREDICT_KEYS)."
        )
    return out


def _context_jsonify(value: Any) -> Any:
    """Coerce a ContextPrediction attribute into JSON-serializable form.

    ``alternatives`` / ``reasons`` may hold SDK dataclasses or tuples whose
    shape is not part of the documented contract; this keeps the endpoint from
    failing to serialize a response while still surfacing the information.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    # numpy scalars (e.g. a predicted numeric value or confidence) expose
    # ``.item()`` and live in the ``numpy`` module. They are not JSON
    # serializable as-is -- and ``numpy.float64`` even passes the ``float``
    # check below -- so unwrap them to a native Python scalar first.
    item = getattr(value, "item", None)
    if callable(item) and type(value).__module__ == "numpy":
        try:
            return _context_jsonify(item())
        except Exception:
            return str(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _context_jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_context_jsonify(v) for v in value]
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return {str(k): _context_jsonify(v) for k, v in attrs.items()}
    return str(value)


def predict_context(
    name: str,
    file_id: str,
    keys: list[str],
    top_k: int = 100,
    kind: str = "part",
    tags: Optional[list[str]] = None,
    tag_mode: str = "any",
    tagged: Optional[bool] = None,
    query_context: Optional[dict[str, Any]] = None,
    status_policy: Optional[dict[str, Any]] = None,
    neighbor_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Predict *keys* for the query part from its nearest neighbours' records.

    The query part is embedded and searched exactly as :func:`search_index`
    does -- including the ``kind`` filter -- so the neighbours here are the same
    neighbours a similarity search would show. *kind* defaults to ``"part"`` so
    an estimate is built from parts and not polluted by assemblies; pass
    ``"any"`` to widen it. The self hit is dropped, an optional tag filter is
    applied, the *top_k* nearest are kept, and those that carry a context record
    are handed to ``ContextPredictor.infer`` together with the requested *keys*.

    *neighbor_ids* pins the neighbour set explicitly. When given, the search is
    run only to rebuild the authoritative ``VectorHit`` objects (with their real
    scores and metadata), and the considered neighbours are exactly those whose
    id is in *neighbor_ids*, in search-rank order -- the *top_k* cap and the tag
    filter are not applied on top, because the caller has already narrowed the
    set (e.g. the web client's Results tab, sliced by its similarity threshold
    and tag filter). Ids not present in the search pool for this query, and the
    self hit, are skipped. This makes the estimate use precisely the parts the
    user narrowed to, rather than a fresh top_k neighbourhood.

    *status_policy* tunes where ``infer`` draws the ready/needs-review line. It
    is merged over the index's rules file (the request wins); an empty result
    means the argument is omitted so the SDK keeps its own default.

    Returns a dict with ``candidates_examined`` (neighbours pulled and
    examined), ``hits_considered`` (nearest neighbours used as the pool),
    ``hits_with_context`` (of those, how many had a record and fed the
    prediction), ``coverage`` (their ratio), ``predictions`` (one entry per key)
    and ``hits`` (the evidence records, with only the values actually present).
    Raises ``KeyError`` when the index does not exist and ``ValueError`` for a
    bad key list, an unknown ``kind``, or an empty index.
    """
    _validate_index_name(name)
    if kind not in _SEARCH_KINDS:
        raise ValueError(
            f"Invalid kind '{kind}'. Must be one of: {sorted(_SEARCH_KINDS)}."
        )
    predict_keys = _normalize_predict_keys(keys)
    cleaned_query_context = (
        context_store.normalize_context(query_context) if query_context else None
    )
    tag_filter = normalize_tag_filter(tags, tag_mode, tagged)
    lock = _get_index_lock(name)

    with lock:
        vs = index_registry._load_named_index(name)
        registered_ids = set(vs.get_ids())
        if not registered_ids:
            raise ValueError(f"Index '{name}' has no registered parts to predict from.")
        searcher = index_registry._get_named_searcher(name)

    query_path = storage.find_persistent_CAD_file(file_id)

    # An explicit neighbour set from the caller (the Results-tab shown tiles):
    # the self hit is never valid evidence, so drop it here too.
    pinned_neighbors = neighbor_ids is not None
    neighbor_set = {
        str(nid) for nid in (neighbor_ids or []) if nid and str(nid) != file_id
    }

    k = int(top_k) if int(top_k) > 0 else 1
    if pinned_neighbors:
        # Fetch at least enough candidates to cover every pinned id (they are a
        # subset of the query's top_k neighbourhood, so the search pool contains
        # them when top_k matches the caller's search).
        fetch_k = max(k, len(neighbor_set))
    else:
        # Use the same candidate-count rule as a tag-filtered search so a tagged
        # prediction is built from the same neighbours a tagged search would show.
        fetch_k = _tag_filtered_fetch_k(k, tag_filter)
    search_kwargs: dict[str, Any] = {"top_k": fetch_k + 1}
    if kind != "any":
        search_kwargs["filters"] = {"kind": kind}
    results = searcher.search_by_shape(str(query_path), **search_kwargs)

    hit_objs = _flatten_body_hit_objects(results, fetch_k + 1)
    hit_objs = [h for h in hit_objs if str(getattr(h, "id", "") or "") != file_id]
    # Count what the search returned (after dropping the self hit) before any
    # narrowing, matching search_index's diagnostic semantics.
    candidates_examined = len(hit_objs)

    if pinned_neighbors:
        # Keep exactly the caller's neighbours, in search-rank order. No tag
        # filter or top_k cap: the caller already narrowed the set.
        considered = [
            h for h in hit_objs if str(getattr(h, "id", "") or "") in neighbor_set
        ]
    else:
        if tag_filter:
            matches = tag_filter_predicate(name, tag_filter)
            hit_objs = [h for h in hit_objs if matches(str(getattr(h, "id", "") or ""))]
        considered = hit_objs[:k]
    considered_ids = [str(getattr(h, "id", "") or "") for h in considered]


    index_dir = config.INDEXES_DIR / name
    # Read the records for the considered neighbours once. The predictor is then
    # built over this in-memory snapshot, so the evidence reported in ``hits``
    # and the records the prediction actually aggregates are the same data even
    # if a concurrent write changes context.json between the two.
    contexts = context_store.get_contexts(index_dir, considered_ids)
    with_context = [
        h for h in considered
        if str(getattr(h, "id", "") or "") in contexts
    ]

    predictions_out: dict[str, Any] = {}
    if with_context:
        numeric_override = context_store.load_rules_config(index_dir).get("numeric_keys")
        provider = context_store.make_memory_context_provider(
            contexts, numeric_override=numeric_override
        )
        predictor = context_store.build_context_predictor(index_dir, provider)
        infer_kwargs: dict[str, Any] = {}
        if cleaned_query_context is not None:
            infer_kwargs["query_context"] = cleaned_query_context
        # Merge the request's status_policy over the rules file; only pass it
        # when non-empty so the SDK keeps its own default otherwise (infer()
        # uses a sentinel default, so passing None would not be equivalent).
        resolved_policy = context_store.resolve_status_policy(index_dir, status_policy)
        if resolved_policy:
            infer_kwargs["status_policy"] = resolved_policy
        predictions = predictor.infer(with_context, predict_keys, **infer_kwargs)
        for key, pred in predictions.items():
            predictions_out[key] = {
                "value": _context_jsonify(getattr(pred, "value", None)),
                "confidence": _context_jsonify(getattr(pred, "confidence", None)),
                "evidence_count": _context_jsonify(getattr(pred, "evidence_count", None)),
                "status": getattr(pred, "status", None),
                "coverage": _context_jsonify(getattr(pred, "coverage", None)),
                "alternatives": _context_jsonify(getattr(pred, "alternatives", None)),
                "reasons": _context_jsonify(getattr(pred, "reasons", None)),
                "injected_context": _context_jsonify(
                    getattr(pred, "injected_context", None)
                ),
            }

    hits_out = []
    for h in with_context:
        hid = str(getattr(h, "id", "") or "")
        record = contexts.get(hid, {})
        values = {
            key: record[key]
            for key in predict_keys
            if key in record and record[key] is not None
        }
        meta = _public_metadata(getattr(h, "metadata", None))
        hits_out.append(
            {
                "id": hid,
                "score": round(float(getattr(h, "score", 0.0) or 0.0), 6),
                "filename": meta.get("filename"),
                "has_thumbnail": index_registry._resolve_index_asset(name, hid, "thumbnail") is not None,
                "values": values,
            }
        )

    hits_considered = len(considered)
    hits_with_context = len(with_context)
    coverage = (hits_with_context / hits_considered) if hits_considered else 0.0
    return {
        "keys": predict_keys,
        "candidates_examined": candidates_examined,
        "hits_considered": hits_considered,
        "hits_with_context": hits_with_context,
        "coverage": round(coverage, 6),
        "predictions": predictions_out,
        "hits": hits_out,
    }


def get_index_context_summary(name: str) -> tuple[dict[str, Any], str]:
    """Return ``({"keys", "parts_with_context"}, etag)`` for the named index."""
    _validate_index_name(name)
    _require_index_exists(name)
    return context_store.context_summary(config.INDEXES_DIR / name)


def get_part_context(name: str, part_id: str) -> tuple[dict[str, Any], str]:
    """Return ``({"part_id", "context"}, etag)`` for one part (``{}`` when none)."""
    _validate_index_name(name)
    _require_index_exists(name)
    return context_store.get_part_context(config.INDEXES_DIR / name, part_id)


def set_part_context(
    name: str,
    part_id: str,
    context: Any,
    if_match: Optional[str] = None,
    client_id: Optional[str] = None,
) -> tuple[dict[str, Any], str]:
    """Replace the whole context of *part_id*. An empty mapping clears it.

    Raises ``ValueError`` for an unregistered part or invalid values,
    ``KeyError`` when the index does not exist, and
    ``context_store.ContextPreconditionFailed`` on an If-Match mismatch.
    """
    _validate_index_name(name)
    _require_index_exists(name)
    # Hold the index lock across the registration check and the context write so
    # a concurrent part removal cannot slip between them and leave an orphan
    # record. Lock order is index lock -> context lock (never the reverse).
    lock = _get_index_lock(name)
    with lock:
        vs = index_registry._load_named_index(name)
        registered = set(vs.get_ids())
        return context_store.set_part_context(
            config.INDEXES_DIR / name,
            part_id,
            context,
            if_match=if_match,
            client_id=client_id,
            registered_ids=registered,
        )


def prune_context(name: str, part_ids: list[str]) -> int:
    """Drop context entries for ids no longer registered in the named index."""
    try:
        _validate_index_name(name)
    except ValueError:
        return 0
    return context_store.prune_context(config.INDEXES_DIR / name, part_ids)
