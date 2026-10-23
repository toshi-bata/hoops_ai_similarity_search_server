"""Per-index production-record store (indexes/<name>/context.json).

The Context Layer predicts missing production records -- material, process,
unit cost, lead time and so on -- for a query part from the records of its
nearest neighbours. Those records are held in a sidecar next to the index,
exactly like tags (see core.py's tags.json), and for the same reason: writing
them into the FaissVectorStore metadata would call save(), rewrite the .faiss
file, move its mtime and trigger a 19.3 s assembly-matcher rebuild on the next
search. A sidecar edit touches no vectors.

Everything in this module that reads or writes context.json is plain Python and
imports nothing from ``hoops_ai`` or ``core`` -- it is unit-testable without the
SDK. Only the two builders at the bottom (:func:`make_index_context_provider`
and :func:`build_context_predictor`) touch ``hoops_ai``, and they import it
lazily inside the function so importing this module never pulls the SDK (which
would import matplotlib/pandas) into an interpreter that does not need it.

The functions take an ``index_dir`` (``indexes/<name>``) rather than an index
name so this module does not have to know how the data root is resolved; core
passes ``INDEXES_DIR / name`` and the seed tool / tests pass a temp directory.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import logging
import math
import os
import pathlib
import threading
import uuid
from collections import OrderedDict
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger("context_store")

# Bump only for an incompatible on-disk change; a file written by a newer schema
# is refused rather than silently reinterpreted (see :func:`_parse_document`).
_CONTEXT_SCHEMA_VERSION = 1

# Value rules. Keys and string values are bounded so a runaway client cannot
# grow the sidecar without limit; the caps are generous for real production
# fields (Material, ProcessRoute, Supplier ...).
_KEY_MAX_LENGTH = 64
_STR_VALUE_MAX_LENGTH = 256
_KEYS_PER_PART_DEFAULT = 32

# Parsed context.json per index_dir, validated against the file's
# (mtime_ns, size) so an edit made outside the server is picked up. Bounded
# because a server may host many indexes.
_CONTEXT_CACHE_MAX = 8
_context_cache: "OrderedDict[str, tuple[tuple[int, int], dict[str, Any], str]]" = OrderedDict()
_context_cache_mutex = threading.Lock()

# Per-index lock held across the read-modify-write of context.json. Separate
# from the index lock on purpose: a context edit touches no vectors, so it must
# not block searches. Reads take no lock -- the file is swapped in with
# os.replace, so a reader sees either the whole old file or the whole new one.
_context_locks: dict[str, threading.Lock] = {}
_context_locks_mutex = threading.Lock()


class ContextFileError(RuntimeError):
    """Raised when context.json exists but cannot be read as a v1 document."""


class ContextPreconditionFailed(RuntimeError):
    """Raised when an If-Match header does not match the current ETag."""


# ---------------------------------------------------------------------------
# Paths, locks and small helpers
# ---------------------------------------------------------------------------


def context_path(index_dir: pathlib.Path) -> pathlib.Path:
    """Return the context sidecar path (indexes/<name>/context.json)."""
    return pathlib.Path(index_dir) / "context.json"


def rules_path(index_dir: pathlib.Path) -> pathlib.Path:
    """Return the optional rules path (indexes/<name>/context_rules.json)."""
    return pathlib.Path(index_dir) / "context_rules.json"


def _lock_key(index_dir: pathlib.Path) -> str:
    return str(pathlib.Path(index_dir))


def _get_context_lock(index_dir: pathlib.Path) -> threading.Lock:
    key = _lock_key(index_dir)
    with _context_locks_mutex:
        if key not in _context_locks:
            _context_locks[key] = threading.Lock()
        return _context_locks[key]


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d.", name, raw, default)
        return default
    if value < minimum or value > maximum:
        logger.warning(
            "%s=%d is out of range [%d, %d]; using %d.",
            name, value, minimum, maximum, default,
        )
        return default
    return value


def max_keys_per_part() -> int:
    """Upper bound on distinct keys for one part (HOOPS_AI_MAX_CONTEXT_KEYS)."""
    return _env_int(
        "HOOPS_AI_MAX_CONTEXT_KEYS", _KEYS_PER_PART_DEFAULT, minimum=1, maximum=1000
    )


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_number(value: Any) -> bool:
    # bool is a subclass of int but is categorical here, not numeric.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# Value validation
# ---------------------------------------------------------------------------


def normalize_key(key: Any) -> str:
    """Return the stored form of a context *key*, or raise ``ValueError``."""
    if not isinstance(key, str):
        raise ValueError(f"Context key must be a string, got {type(key).__name__}.")
    cleaned = key.strip()
    if not cleaned:
        raise ValueError("Context key must not be empty or whitespace only.")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in cleaned):
        raise ValueError(f"Context key '{key}' contains control characters.")
    if len(cleaned) > _KEY_MAX_LENGTH:
        raise ValueError(
            f"Context key is {len(cleaned)} characters; the maximum is "
            f"{_KEY_MAX_LENGTH}."
        )
    return cleaned


def normalize_value(key: str, value: Any) -> Any:
    """Return the stored form of one context *value*, or raise ``ValueError``.

    Only JSON scalars are accepted -- string, finite number, bool or None --
    because a production record is a flat set of fields and the Context Layer's
    rules operate on scalars. ``None`` is preserved so a caller can record "this
    field is known to be empty" distinctly from "this field is absent".
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if len(cleaned) > _STR_VALUE_MAX_LENGTH:
            raise ValueError(
                f"Value for '{key}' is {len(cleaned)} characters; the maximum "
                f"is {_STR_VALUE_MAX_LENGTH}."
            )
        if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in cleaned):
            raise ValueError(f"Value for '{key}' contains control characters.")
        return cleaned
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Value for '{key}' must be a finite number.")
        return value
    raise ValueError(
        f"Value for '{key}' must be a string, number, boolean or null, got "
        f"{type(value).__name__}."
    )


def normalize_context(context: Any) -> dict[str, Any]:
    """Validate a context mapping, returning a clean ``{key: value}`` dict.

    Duplicate keys after normalization, too many keys, or a non-scalar value
    raise ``ValueError``. An empty mapping is valid and means "clear".
    """
    if not isinstance(context, Mapping):
        raise ValueError("context must be an object of key/value pairs.")
    out: dict[str, Any] = {}
    for raw_key, raw_value in context.items():
        key = normalize_key(raw_key)
        if key in out:
            raise ValueError(f"Duplicate context key '{key}'.")
        out[key] = normalize_value(key, raw_value)
    limit = max_keys_per_part()
    if len(out) > limit:
        raise ValueError(
            f"{len(out)} context keys supplied; the maximum per part is {limit} "
            f"(HOOPS_AI_MAX_CONTEXT_KEYS)."
        )
    return out


# ---------------------------------------------------------------------------
# Document serialization and parsing
# ---------------------------------------------------------------------------


def _empty_document() -> dict[str, Any]:
    return {"version": _CONTEXT_SCHEMA_VERSION, "updated_at": None, "units": {}, "parts": {}}


def _clean_units(raw: Any) -> dict[str, str]:
    """Return a sanitized ``{field: unit_label}`` map (string -> string only)."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip():
            out[key] = value
    return out


def _serialize_document(doc: dict[str, Any]) -> bytes:
    """Serialize deterministically so the ETag depends only on the content."""
    return json.dumps(doc, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _etag(payload: bytes) -> str:
    """Return a strong ETag for the serialized document."""
    return '"' + hashlib.sha256(payload).hexdigest()[:32] + '"'


def _parse_document(raw: Any, path: pathlib.Path) -> dict[str, Any]:
    """Validate a decoded context.json and drop entries that cannot be used.

    A structurally broken file raises rather than being silently discarded:
    returning an empty document would make the next write erase every record in
    the file. Individual values that are not scalars are dropped (loudly) so one
    stray value cannot make a whole part unreadable.
    """
    if not isinstance(raw, dict):
        raise ContextFileError(f"{path} is not a JSON object.")
    version = raw.get("version")
    if version != _CONTEXT_SCHEMA_VERSION:
        raise ContextFileError(
            f"{path} has version {version!r}; this server understands "
            f"version {_CONTEXT_SCHEMA_VERSION}."
        )
    parts = raw.get("parts")
    if parts is None:
        parts = {}
    if not isinstance(parts, dict):
        raise ContextFileError(f"{path} has a 'parts' member that is not an object.")

    cleaned: dict[str, Any] = {}
    for part_id, entry in parts.items():
        if not isinstance(entry, dict):
            raise ContextFileError(f"{path}: entry for '{part_id}' is not an object.")
        context = entry.get("context")
        if not isinstance(context, dict):
            raise ContextFileError(
                f"{path}: 'context' for '{part_id}' is not an object."
            )
        kept: dict[str, Any] = {}
        for k, v in context.items():
            if not isinstance(k, str) or not k.strip():
                continue
            if v is None or isinstance(v, (str, int, float, bool)):
                if isinstance(v, float) and not math.isfinite(v):
                    continue
                kept[k] = v
            else:
                logger.warning(
                    "[CONTEXT] dropping non-scalar value for '%s'.'%s' in %s",
                    part_id, k, path,
                )
        if not kept:
            continue
        record: dict[str, Any] = {"context": kept}
        if entry.get("updated_at"):
            record["updated_at"] = entry["updated_at"]
        if entry.get("updated_by"):
            record["updated_by"] = entry["updated_by"]
        if entry.get("source"):
            record["source"] = entry["source"]
        cleaned[str(part_id)] = record

    return {
        "version": _CONTEXT_SCHEMA_VERSION,
        "updated_at": raw.get("updated_at"),
        "units": _clean_units(raw.get("units")),
        "parts": cleaned,
    }


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


def read_context_document(index_dir: pathlib.Path) -> tuple[dict[str, Any], str]:
    """Return ``(document, etag)`` for the index directory.

    Takes no lock: writers swap the file in with ``os.replace``, so a reader
    sees either the whole previous file or the whole new one. A missing file is
    an empty document, not an error.
    """
    path = context_path(index_dir)
    try:
        st = path.stat()
    except FileNotFoundError:
        doc = _empty_document()
        return doc, _etag(_serialize_document(doc))

    key = (st.st_mtime_ns, st.st_size)
    cache_key = _lock_key(index_dir)
    with _context_cache_mutex:
        cached = _context_cache.get(cache_key)
        if cached is not None and cached[0] == key:
            _context_cache.move_to_end(cache_key)
            return copy.deepcopy(cached[1]), cached[2]

    try:
        payload = path.read_bytes()
        raw = json.loads(payload.decode("utf-8"))
    except ContextFileError:
        raise
    except Exception as exc:
        raise ContextFileError(f"Could not read {path}: {exc}") from exc

    doc = _parse_document(raw, path)
    etag = _etag(_serialize_document(doc))

    with _context_cache_mutex:
        _context_cache[cache_key] = (key, copy.deepcopy(doc), etag)
        _context_cache.move_to_end(cache_key)
        while len(_context_cache) > _CONTEXT_CACHE_MAX:
            _context_cache.popitem(last=False)

    return doc, etag


def _write_context_document(index_dir: pathlib.Path, doc: dict[str, Any]) -> str:
    """Persist *doc* atomically and return its new ETag.

    Must be called while holding the context lock for *index_dir*.
    """
    doc["version"] = _CONTEXT_SCHEMA_VERSION
    doc["updated_at"] = _utcnow()
    payload = _serialize_document(doc)
    etag = _etag(payload)

    path = context_path(index_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"_tmp_context_{uuid.uuid4().hex}.json")
    try:
        tmp.write_bytes(payload)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    cache_key = _lock_key(index_dir)
    try:
        st = path.stat()
        with _context_cache_mutex:
            _context_cache[cache_key] = (
                (st.st_mtime_ns, st.st_size), copy.deepcopy(doc), etag,
            )
            _context_cache.move_to_end(cache_key)
            while len(_context_cache) > _CONTEXT_CACHE_MAX:
                _context_cache.popitem(last=False)
    except OSError:
        evict_cache(index_dir)

    return etag


def evict_cache(index_dir: pathlib.Path) -> None:
    with _context_cache_mutex:
        _context_cache.pop(_lock_key(index_dir), None)


def etag_matches(if_match: Optional[str], current: str) -> bool:
    """Return True when *if_match* permits a write against *current*.

    A missing or empty header always passes; ``*`` matches anything; otherwise
    the comma-separated list is compared after stripping the weak-validator
    prefix. Mirrors tags.etag_matches so the two sidecars behave identically.
    """
    if if_match is None:
        return True
    candidate = if_match.strip()
    if not candidate or candidate == "*":
        return True
    current_value = current.strip('"')
    for token in candidate.split(","):
        token = token.strip()
        if token.startswith("W/"):
            token = token[2:].strip()
        if token.strip('"') == current_value:
            return True
    return False


def _mutate(index_dir: pathlib.Path, if_match: Optional[str], mutator):
    """Run *mutator* against context.json under the context lock.

    *mutator* receives the document and returns ``(result, changed)``. The
    If-Match check happens inside the lock, so a concurrent writer cannot slip
    between the check and the write.
    """
    with _get_context_lock(index_dir):
        doc, etag = read_context_document(index_dir)
        if not etag_matches(if_match, etag):
            raise ContextPreconditionFailed(
                "The index context has changed since it was read "
                f"(current ETag {etag}). Re-read and retry."
            )
        result, changed = mutator(doc)
        if changed:
            etag = _write_context_document(index_dir, doc)
        return result, etag


def _stamp(entry: dict[str, Any], client_id: Optional[str]) -> None:
    entry["updated_at"] = _utcnow()
    if client_id:
        entry["updated_by"] = client_id
    else:
        entry.pop("updated_by", None)


# ---------------------------------------------------------------------------
# Public read/write API
# ---------------------------------------------------------------------------


def get_contexts(index_dir: pathlib.Path, ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Return ``{id: context}`` for the requested ids that have a record.

    Ids without a record are omitted (not returned as empty), matching the
    ContextProvider contract the predictor expects.
    """
    doc, _ = read_context_document(index_dir)
    parts = doc["parts"]
    out: dict[str, dict[str, Any]] = {}
    for raw_id in ids:
        pid = str(raw_id)
        entry = parts.get(pid)
        if entry:
            out[pid] = dict(entry["context"])
    return out


def get_part_context(
    index_dir: pathlib.Path, part_id: str
) -> tuple[dict[str, Any], str]:
    """Return ``({"part_id", "context"}, etag)`` for one part (``{}`` when none)."""
    doc, etag = read_context_document(index_dir)
    entry = doc["parts"].get(str(part_id))
    context = dict(entry["context"]) if entry else {}
    return {"part_id": str(part_id), "context": context}, etag


def set_part_context(
    index_dir: pathlib.Path,
    part_id: str,
    context: Any,
    if_match: Optional[str] = None,
    client_id: Optional[str] = None,
    registered_ids: Optional[set[str]] = None,
    source: Optional[str] = None,
) -> tuple[dict[str, Any], str]:
    """Replace the whole context of *part_id*. An empty mapping clears it.

    ``registered_ids``, when given, must contain *part_id* or ``ValueError`` is
    raised -- the server passes the index's id set so a record cannot be written
    for a part that is not in the index. The seed tool passes ``None`` to skip
    that check. Raises ``ContextPreconditionFailed`` on an If-Match mismatch.
    """
    pid = str(part_id)
    cleaned = normalize_context(context)
    if registered_ids is not None and pid not in registered_ids:
        raise ValueError(f"Part '{pid}' is not registered in this index.")

    def mutator(doc: dict[str, Any]):
        previous = doc["parts"].get(pid, {}).get("context", {})
        if previous == cleaned:
            return {"part_id": pid, "context": dict(cleaned)}, False
        if cleaned:
            entry = doc["parts"].setdefault(pid, {})
            entry["context"] = dict(cleaned)
            _stamp(entry, client_id)
            if source:
                entry["source"] = source
            else:
                entry.pop("source", None)
        else:
            doc["parts"].pop(pid, None)
        return {"part_id": pid, "context": dict(cleaned)}, True

    return _mutate(index_dir, if_match, mutator)


def context_summary(index_dir: pathlib.Path) -> tuple[dict[str, Any], str]:
    """Return ``({"keys": [...], "parts_with_context": n}, etag)``.

    Each key entry is ``{"key", "count", "type"}`` where ``type`` is
    ``numeric`` / ``string`` / ``boolean`` / ``mixed`` inferred from the stored
    values. Keys are ordered by descending count, then by name.
    """
    doc, etag = read_context_document(index_dir)
    counts: dict[str, int] = {}
    kinds: dict[str, set[str]] = {}
    for entry in doc["parts"].values():
        for key, value in entry["context"].items():
            counts[key] = counts.get(key, 0) + 1
            if value is None:
                kind = "null"
            elif isinstance(value, bool):
                kind = "boolean"
            elif _is_number(value):
                kind = "numeric"
            elif isinstance(value, str):
                kind = "string"
            else:
                kind = "mixed"
            kinds.setdefault(key, set()).add(kind)

    def _type_for(key: str) -> str:
        seen = kinds.get(key, set()) - {"null"}
        if len(seen) == 1:
            return next(iter(seen))
        if not seen:
            return "null"
        return "mixed"

    items = [
        {"key": key, "count": count, "type": _type_for(key)}
        for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return {
        "keys": items,
        "parts_with_context": len(doc["parts"]),
        "units": _clean_units(doc.get("units")),
    }, etag


def _numeric_keys_from_contexts(mapping: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Return the keys whose values across *mapping* are all numeric.

    A key qualifies when it has at least one numeric value and no non-numeric,
    non-null value.
    """
    numeric_seen: dict[str, bool] = {}
    disqualified: set[str] = set()
    for context in mapping.values():
        for key, value in context.items():
            if value is None:
                continue
            if _is_number(value):
                numeric_seen[key] = True
            else:
                disqualified.add(key)
    return sorted(k for k in numeric_seen if k not in disqualified)


def list_numeric_keys(index_dir: pathlib.Path) -> list[str]:
    """Return the keys whose stored values are all numeric.

    A key qualifies when it has at least one numeric value and no non-numeric,
    non-null value. Used by the ContextProvider so the predictor applies a
    numeric rule to cost / lead-time style fields. A rules file may override
    this list (see :func:`load_rules_config`).
    """
    doc, _ = read_context_document(index_dir)
    return _numeric_keys_from_contexts(
        {pid: entry["context"] for pid, entry in doc["parts"].items()}
    )


def prune_context(index_dir: pathlib.Path, part_ids: Iterable[str]) -> int:
    """Drop context entries for ids no longer registered in the index.

    Called after parts are removed. Without it the sidecar would grow forever.
    Never raises: a failure to prune must not fail the part removal.
    """
    targets = {str(p).strip() for p in part_ids if str(p).strip()}
    if not targets:
        return 0
    try:
        def mutator(doc: dict[str, Any]):
            removed = 0
            for pid in targets:
                if doc["parts"].pop(pid, None) is not None:
                    removed += 1
            return removed, removed > 0

        removed, _ = _mutate(index_dir, None, mutator)
    except Exception as exc:  # noqa: BLE001 - pruning is best-effort
        logger.warning("[CONTEXT] could not prune '%s': %s", index_dir, exc)
        return 0
    if removed:
        logger.info(
            "[CONTEXT] dropped %d context entries removed from '%s'", removed, index_dir
        )
    return removed


# ---------------------------------------------------------------------------
# Rules configuration (optional indexes/<name>/context_rules.json)
# ---------------------------------------------------------------------------
#
# The rules file only ever carries scalar constructor arguments and the *name*
# of a normalizer from a fixed registry. It never carries code: a sidecar is
# data, and evaluating code from it would be a remote-code-execution hole.

# Constructor arguments confirmed against hoops_ai 1.1.x (STEP 0 probe). Any key
# outside these sets is warned about and ignored rather than passed through, so
# a typo or a newer/older SDK cannot raise a TypeError deep inside infer().
_CATEGORICAL_PARAMS = {
    "temperature", "score_floor", "min_confidence", "min_margin",
    "top1_dominance_threshold", "top1_dominance_weight",
    "min_top1_margin_for_dominance",
}
_NUMERIC_PARAMS = {
    "log_scale", "min_evidence", "interval_sigmas", "auto_relevance_weight",
    "score_temperature", "nearest_neighbor_threshold",
}
_NEAREST_PARAMS = {"threshold"}

# ContextPredictor.infer(status_policy=...) tuning knobs. Only these numeric
# keys are read from the rules file or a request; anything else is ignored so a
# request cannot reach an undocumented argument. These move the boundary
# between ``ready_to_propose`` and ``needs_review`` -- the one demo-day dial.
_STATUS_POLICY_PARAMS = {
    "min_observed_hits", "min_observed_score_weight", "min_top_share", "min_margin",
}


def clean_status_policy(raw: Any) -> dict[str, Any]:
    """Return a status_policy dict with only whitelisted numeric keys.

    A non-object, or any key outside :data:`_STATUS_POLICY_PARAMS` or with a
    non-numeric value, is dropped (warned) rather than passed through to the SDK.
    """
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in _STATUS_POLICY_PARAMS:
            logger.warning("[CONTEXT] ignoring unknown status_policy key '%s'.", key)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            logger.warning("[CONTEXT] ignoring non-numeric status_policy '%s'.", key)
            continue
        out[key] = value
    return out


def resolve_status_policy(
    index_dir: pathlib.Path, override: Any = None
) -> dict[str, Any]:
    """Merge the rules-file status_policy with a per-request *override*.

    The request wins key by key. Returns ``{}`` when neither supplies anything,
    which the caller must translate to "omit the argument" so the SDK keeps its
    own default.
    """
    base = clean_status_policy(load_rules_config(index_dir).get("status_policy"))
    base.update(clean_status_policy(override))
    return base

# Named normalizers. Values are looked up by name only; a caller cannot supply
# a callable through the JSON sidecar.
_NORMALIZERS: dict[str, Any] = {
    "lower": lambda v: v.lower() if isinstance(v, str) else v,
    "upper": lambda v: v.upper() if isinstance(v, str) else v,
    "strip": lambda v: v.strip() if isinstance(v, str) else v,
    "lower_strip": lambda v: v.strip().lower() if isinstance(v, str) else v,
    "title": lambda v: v.title() if isinstance(v, str) else v,
}


def _filter_params(spec: Mapping[str, Any], allowed: set[str], where: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in spec.items():
        if key in ("type", "normalize"):
            continue
        if key not in allowed:
            logger.warning("[CONTEXT] ignoring unknown %s rule option '%s'.", where, key)
            continue
        out[key] = value
    return out


def load_rules_config(index_dir: pathlib.Path) -> dict[str, Any]:
    """Read and sanitize context_rules.json, returning a safe spec.

    Missing file -> ``{}`` (all defaults). A malformed file is warned about and
    treated as empty rather than failing prediction: the rules are a tuning
    convenience, not required for a prediction to run.

    Returns a dict with keys ``categorical`` / ``numeric`` / ``per_key`` /
    ``numeric_keys`` holding only whitelisted scalar options and normalizer
    *names* (never callables or code).
    """
    path = rules_path(index_dir)
    try:
        raw = json.loads(path.read_bytes().decode("utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 - rules are optional
        logger.warning("[CONTEXT] ignoring unreadable rules file '%s': %s", path, exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning("[CONTEXT] rules file '%s' is not an object; ignoring.", path)
        return {}

    out: dict[str, Any] = {}
    categorical = raw.get("categorical")
    if isinstance(categorical, dict):
        out["categorical"] = _filter_params(categorical, _CATEGORICAL_PARAMS, "categorical")
        norm = categorical.get("normalize")
        if isinstance(norm, str) and norm in _NORMALIZERS:
            out.setdefault("categorical", {})["normalize"] = norm
        elif norm is not None:
            logger.warning("[CONTEXT] ignoring unknown normalizer '%s'.", norm)

    numeric = raw.get("numeric")
    if isinstance(numeric, dict):
        out["numeric"] = _filter_params(numeric, _NUMERIC_PARAMS, "numeric")

    per_key = raw.get("per_key")
    if isinstance(per_key, dict):
        cleaned_per_key: dict[str, Any] = {}
        for key, spec in per_key.items():
            if not isinstance(spec, dict):
                logger.warning("[CONTEXT] ignoring non-object per-key rule for '%s'.", key)
                continue
            rule_type = spec.get("type")
            if rule_type not in ("categorical", "numeric", "nearest_neighbor"):
                logger.warning(
                    "[CONTEXT] per-key rule for '%s' has unknown type %r; ignoring.",
                    key, rule_type,
                )
                continue
            entry: dict[str, Any] = {"type": rule_type}
            if rule_type == "categorical":
                entry.update(_filter_params(spec, _CATEGORICAL_PARAMS, "categorical"))
                norm = spec.get("normalize")
                if isinstance(norm, str) and norm in _NORMALIZERS:
                    entry["normalize"] = norm
                elif norm is not None:
                    logger.warning("[CONTEXT] ignoring unknown normalizer '%s'.", norm)
            elif rule_type == "numeric":
                entry.update(_filter_params(spec, _NUMERIC_PARAMS, "numeric"))
            else:
                entry.update(_filter_params(spec, _NEAREST_PARAMS, "nearest_neighbor"))
            cleaned_per_key[str(key)] = entry
        if cleaned_per_key:
            out["per_key"] = cleaned_per_key

    numeric_keys = raw.get("numeric_keys")
    if isinstance(numeric_keys, list):
        keys = [str(k) for k in numeric_keys if isinstance(k, str) and k.strip()]
        if keys:
            out["numeric_keys"] = keys

    status_policy = clean_status_policy(raw.get("status_policy"))
    if status_policy:
        out["status_policy"] = status_policy

    return out


# ---------------------------------------------------------------------------
# SDK builders (hoops_ai imported lazily; nothing above this line needs it)
# ---------------------------------------------------------------------------


def make_index_context_provider(index_dir: pathlib.Path):
    """Build a ContextProvider backed by this index's context.json.

    ``hoops_ai`` is imported here, not at module load, so importing
    ``context_store`` never pulls the SDK (and matplotlib/pandas) into a process
    that only needs the plain read/write helpers above.
    """
    from hoops_ai.ml.context_layer import ContextProvider

    index_dir = pathlib.Path(index_dir)
    numeric_override = load_rules_config(index_dir).get("numeric_keys")

    class _IndexContextProvider(ContextProvider):
        def get_contexts(self, ids):
            return get_contexts(index_dir, ids)

        def set_contexts(self, contexts):
            # The predictor only reads during infer(); this is provided for
            # completeness so the provider honours the full ContextProvider
            # contract. It writes through the same validated, atomic path.
            for part_id, context in dict(contexts).items():
                set_part_context(index_dir, part_id, context, source="provider")

        def list_numeric_keys(self):
            if numeric_override is not None:
                return list(numeric_override)
            return list_numeric_keys(index_dir)

    return _IndexContextProvider()


def make_memory_context_provider(contexts, numeric_override=None):
    """Build a ContextProvider backed by an in-memory snapshot of records.

    Used by prediction so the records the predictor aggregates and the evidence
    reported to the caller are one immutable snapshot, unaffected by a
    concurrent write to context.json. ``hoops_ai`` is imported lazily here too.
    """
    from hoops_ai.ml.context_layer import ContextProvider

    snapshot = {str(k): dict(v) for k, v in dict(contexts).items()}
    if numeric_override is not None:
        numeric = list(numeric_override)
    else:
        numeric = _numeric_keys_from_contexts(snapshot)

    class _MemoryContextProvider(ContextProvider):
        def get_contexts(self, ids):
            return {
                str(i): dict(snapshot[str(i)])
                for i in ids
                if str(i) in snapshot
            }

        def set_contexts(self, contexts):
            for part_id, context in dict(contexts).items():
                snapshot[str(part_id)] = dict(context)

        def list_numeric_keys(self):
            return list(numeric)

    return _MemoryContextProvider()


def _build_rule(module, spec: Mapping[str, Any]):
    """Instantiate one rule from a sanitized per-key spec."""
    kwargs = {k: v for k, v in spec.items() if k not in ("type", "normalize")}
    rule_type = spec.get("type")
    if rule_type == "numeric":
        return module.NumericWeightedRule(**kwargs)
    if rule_type == "nearest_neighbor":
        return module.NearestNeighborRule(**kwargs)
    normalize = _NORMALIZERS.get(spec.get("normalize")) if spec.get("normalize") else None
    if normalize is not None:
        kwargs["normalize"] = normalize
    return module.CategoricalRule(**kwargs)


def build_context_predictor(index_dir: pathlib.Path, provider):
    """Build a ContextPredictor for *provider* using this index's rules file.

    Only constructor arguments whitelisted in :func:`load_rules_config` reach
    the SDK, so a bad rules file degrades to defaults instead of raising.
    """
    from hoops_ai.ml.context_layer import (
        CategoricalRule,
        ContextPredictor,
        NumericWeightedRule,
    )
    import hoops_ai.ml.context_layer as ctx_module

    config = load_rules_config(pathlib.Path(index_dir))

    default_categorical = None
    cat_cfg = dict(config.get("categorical", {}))
    if cat_cfg:
        norm = _NORMALIZERS.get(cat_cfg.pop("normalize", None))
        if norm is not None:
            cat_cfg["normalize"] = norm
        default_categorical = CategoricalRule(**cat_cfg)

    default_numeric = None
    num_cfg = dict(config.get("numeric", {}))
    if num_cfg:
        default_numeric = NumericWeightedRule(**num_cfg)

    per_key_rules = None
    per_key_cfg = config.get("per_key")
    if per_key_cfg:
        per_key_rules = {
            key: _build_rule(ctx_module, spec) for key, spec in per_key_cfg.items()
        }

    kwargs: dict[str, Any] = {}
    if default_categorical is not None:
        kwargs["default_categorical_rule"] = default_categorical
    if default_numeric is not None:
        kwargs["default_numerical_rule"] = default_numeric
    if per_key_rules is not None:
        kwargs["per_key_rules"] = per_key_rules

    return ContextPredictor(provider, **kwargs)
