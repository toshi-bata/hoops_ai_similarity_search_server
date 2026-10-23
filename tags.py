"""Index tag sidecars: the per-index tags.json, the application-wide tag
vocabulary, the tags cache, and the tag-filter API.

Tags live in a sidecar (indexes/<name>/tags.json) rather than in the record
metadata. FaissVectorStore does expose update_record_metadata(), and a probe
confirmed that an arbitrary key is accepted, survives save()/load(), and --
because the store keeps one metadata dict per record id rather than per vector
row -- covers every body of a file in a single call. It was still rejected, for
one measured reason: the only way to persist it is save(), which always
rewrites the .faiss file. That moves its mtime, and both the searcher cache and
the assembly matcher cache are keyed by (path, mtime_ns). A single tag edit
would therefore cost a 19.3 s matcher rebuild (measured at 42,098 bodies) plus
an 84 MB rewrite held under the index lock, for a change that does not touch a
single vector. The sidecar is keyed by file_id, which is the SHA-256 of the CAD
contents, so tags survive a full rebuild of the index: the ids do not change.

The tag vocabulary is the one exception to "per index". The names a user has
decided on (DATA_ROOT/tag_vocabulary.json) are shared by the whole
application, because the wording of a tag is a vocabulary choice, not a
property of one index: the same "Socket head cap screw" is the same name
whichever index it was first typed into, and an operator who rebuilds an index
under a new name should not have to retype the list. What that costs is that
names used in another index are offered here too. Clients are expected to put
the tags IN USE first and vocabulary-only names after, so the irrelevant ones
sink to the bottom of a picker rather than getting in the way; that ordering is
the assumption the shared file was chosen on.

Dependency direction. This module is a leaf that depends only on the standard
library, ``config`` (INDEXES_DIR, TAG_VOCABULARY_FILE, _env_int), ``locks``
(_get_tags_lock, _get_tag_vocabulary_lock) and
``index_registry`` (_validate_index_name, _require_index_exists, and
``index_registry._registered_index_ids`` reached as a module attribute because
it is an (A) name there). It must never import ``core``: the dependency runs one
way, ``core`` -> ``tags`` -> ``index_registry``/``locks``/``config``. The tag
cluster has no upward edges -- nothing here calls search, assembly, shape-map,
job or context code.

Reference convention (see "The re-export convention" in docs/architecture.md).
``_read_tags_document`` is the one (A) name: tests swap
``tags._read_tags_document`` to prove the search
cache key reads the sidecar exactly once per predicate, and ``core._search_cache_key``
reaches it as ``tags._read_tags_document``. It is therefore NOT re-exported by
``core``; a ``from tags import _read_tags_document`` (or a core re-export) would
bind the old object and hide the swap, silently returning stale search results
after a tag edit. Every other name here is stable and is re-exported from
``core`` by explicit name for its existing ``core.<name>`` callers.

Orchestrators that stay in ``core``. ``list_index_parts``, ``delete_index`` and
``remove_from_index`` are not part of this module even though they touch tags:
each also calls index-registry code (``index_registry._load_named_index`` /
``_resolve_index_asset``) and, for the two delete paths, assembly/context
clean-up. They are cross-domain orchestrators, so ``core`` keeps them and calls
``tags.<name>`` downward (tags_by_part, normalize_tag_filter, tag_filter_predicate,
prune_tags), the same position ``delete_index`` held after batch 3.
"""

import copy
import datetime
import hashlib
import json
import logging
import pathlib
import threading
import uuid
from collections import OrderedDict
from typing import Any, Optional

import config
import index_registry

# ``_env_int`` is a stdlib-only env helper owned by config.py; never swapped, so
# a plain ``from`` import is safe for the bare ``_env_int(...)`` call site here.
from config import _env_int
# ``_validate_index_name`` and ``_require_index_exists`` are stable (C) names in
# index_registry; ``_registered_index_ids`` is an (A) name there and is reached
# as ``index_registry._registered_index_ids`` (module attribute), never imported.
from index_registry import _require_index_exists, _validate_index_name
# ``_get_tags_lock`` is the per-index tag lock factory owned by locks.py;
# ``_get_tag_vocabulary_lock`` is the process-wide lock for the shared tag
# vocabulary, which is NOT per index (see the tag vocabulary section below).
from locks import _get_tag_vocabulary_lock, _get_tags_lock

logger = logging.getLogger(__name__)


_TAGS_SCHEMA_VERSION = 1

# Tag text rules. Case is significant ("Bracket" != "bracket") to match the Qt
# application; a case-only collision is logged rather than merged, because
# silently folding them would lose a distinction the user may have meant.
_TAG_MAX_LENGTH = 64
_TAGS_PER_PART_DEFAULT = 32

# Parsed tags.json per index, keyed by name and validated against the file's
# (mtime_ns, size) so an edit made outside the server is picked up. Bounded
# because a server may host many indexes.
_TAGS_CACHE_MAX = 8
_tags_cache: "OrderedDict[str, tuple[tuple[int, int], dict[str, Any], str]]" = OrderedDict()
_tags_cache_mutex = threading.Lock()

# The tags lock (_tags_locks) now lives in locks.py.


class TagsFileError(RuntimeError):
    """Raised when tags.json exists but cannot be read as a v1 document."""


class TagsPreconditionFailed(RuntimeError):
    """Raised when an If-Match header does not match the current ETag."""


def _index_tags_path(name: str) -> pathlib.Path:
    """Return the per-index tag sidecar path (indexes/{name}/tags.json)."""
    return config.INDEXES_DIR / name / "tags.json"


def max_tags_per_part() -> int:
    """Upper bound on tags for one part (``HOOPS_AI_MAX_TAGS_PER_PART``)."""
    return _env_int(
        "HOOPS_AI_MAX_TAGS_PER_PART", _TAGS_PER_PART_DEFAULT, minimum=1, maximum=1000
    )


def normalize_tag(tag: Any) -> str:
    """Return the stored form of *tag*, or raise ``ValueError``.

    Surrounding whitespace is trimmed; the result must be non-empty, free of
    control characters, and at most 64 characters. Case is preserved.
    """
    if not isinstance(tag, str):
        raise ValueError(f"Tag must be a string, got {type(tag).__name__}.")
    cleaned = tag.strip()
    if not cleaned:
        raise ValueError("Tag must not be empty or whitespace only.")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in cleaned):
        raise ValueError(f"Tag '{tag}' contains control characters.")
    if len(cleaned) > _TAG_MAX_LENGTH:
        raise ValueError(
            f"Tag is {len(cleaned)} characters; the maximum is {_TAG_MAX_LENGTH}."
        )
    # Tags appear as a path segment in /index/{name}/tags/{tag}/parts. Allowing
    # a separator would route those requests somewhere else and surface as a
    # confusing 404, so it is rejected at the point of entry instead.
    if "/" in cleaned or "\\" in cleaned:
        raise ValueError(f"Tag '{cleaned}' must not contain '/' or '\\'.")
    return cleaned


def _normalize_tag_list(tags: Any) -> list[str]:
    """Validate a list of tags, de-duplicating while preserving order."""
    if isinstance(tags, str) or not isinstance(tags, (list, tuple)):
        raise ValueError("tags must be a list of strings.")
    out: list[str] = []
    for tag in tags:
        cleaned = normalize_tag(tag)
        if cleaned not in out:
            out.append(cleaned)
    limit = max_tags_per_part()
    if len(out) > limit:
        raise ValueError(
            f"{len(out)} tags supplied; the maximum per part is {limit} "
            f"(HOOPS_AI_MAX_TAGS_PER_PART)."
        )
    return out


def _warn_on_case_collision(doc: dict[str, Any], tag: str) -> None:
    """Log when *tag* differs from an existing tag only by case.

    Tags are case-sensitive by design, so this is not an error -- but it is
    almost always a typo, and it is invisible in a tag list sorted by count.
    """
    lowered = tag.lower()
    for existing in _iter_all_tags(doc):
        if existing != tag and existing.lower() == lowered:
            logger.warning(
                "[TAGS] '%s' differs from the existing tag '%s' only by case; "
                "they are kept as separate tags",
                tag,
                existing,
            )
            return


def _iter_all_tags(doc: dict[str, Any]):
    seen: set[str] = set()
    for entry in doc.get("parts", {}).values():
        for tag in entry.get("tags", ()):
            if tag not in seen:
                seen.add(tag)
                yield tag


def _empty_tags_document() -> dict[str, Any]:
    return {"version": _TAGS_SCHEMA_VERSION, "updated_at": None, "parts": {}}


def _serialize_tags_document(doc: dict[str, Any]) -> bytes:
    """Serialize deterministically so the ETag depends only on the content."""
    import json

    return json.dumps(doc, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _tags_etag(payload: bytes) -> str:
    """Return a strong ETag for the serialized document."""
    return '"' + hashlib.sha256(payload).hexdigest()[:32] + '"'


def _parse_tags_document(raw: Any, path: pathlib.Path) -> dict[str, Any]:
    """Validate a decoded tags.json and drop entries that cannot be used.

    Anything structurally wrong raises rather than being silently discarded:
    quietly returning an empty document would make the next write erase every
    tag in the file.
    """
    if not isinstance(raw, dict):
        raise TagsFileError(f"{path} is not a JSON object.")
    version = raw.get("version")
    if version != _TAGS_SCHEMA_VERSION:
        raise TagsFileError(
            f"{path} has version {version!r}; this server understands "
            f"version {_TAGS_SCHEMA_VERSION}."
        )
    parts = raw.get("parts")
    if parts is None:
        parts = {}
    if not isinstance(parts, dict):
        raise TagsFileError(f"{path} has a 'parts' member that is not an object.")

    cleaned: dict[str, Any] = {}
    for part_id, entry in parts.items():
        if not isinstance(entry, dict):
            raise TagsFileError(f"{path}: entry for '{part_id}' is not an object.")
        tags = entry.get("tags")
        if not isinstance(tags, list):
            raise TagsFileError(f"{path}: 'tags' for '{part_id}' is not a list.")
        kept = [t for t in tags if isinstance(t, str) and t.strip()]
        if not kept:
            continue
        record = {"tags": kept}
        if entry.get("updated_at"):
            record["updated_at"] = entry["updated_at"]
        if entry.get("updated_by"):
            record["updated_by"] = entry["updated_by"]
        cleaned[str(part_id)] = record

    return {
        "version": _TAGS_SCHEMA_VERSION,
        "updated_at": raw.get("updated_at"),
        "parts": cleaned,
    }


def _read_tags_document(name: str) -> tuple[dict[str, Any], str]:
    """Return ``(document, etag)`` for the named index.

    Takes no lock: writers swap the file in with ``os.replace``, so a reader
    sees either the whole previous file or the whole new one. A missing file is
    an empty document, not an error -- an index simply has no tags yet.
    """
    import json

    path = _index_tags_path(name)
    try:
        st = path.stat()
    except FileNotFoundError:
        doc = _empty_tags_document()
        return doc, _tags_etag(_serialize_tags_document(doc))

    key = (st.st_mtime_ns, st.st_size)
    with _tags_cache_mutex:
        cached = _tags_cache.get(name)
        if cached is not None and cached[0] == key:
            _tags_cache.move_to_end(name)
            return copy.deepcopy(cached[1]), cached[2]

    try:
        payload = path.read_bytes()
        raw = json.loads(payload.decode("utf-8"))
    except TagsFileError:
        raise
    except Exception as exc:
        raise TagsFileError(f"Could not read {path}: {exc}") from exc

    doc = _parse_tags_document(raw, path)
    etag = _tags_etag(_serialize_tags_document(doc))

    with _tags_cache_mutex:
        _tags_cache[name] = (key, copy.deepcopy(doc), etag)
        _tags_cache.move_to_end(name)
        while len(_tags_cache) > _TAGS_CACHE_MAX:
            _tags_cache.popitem(last=False)

    return doc, etag


def _write_tags_document(name: str, doc: dict[str, Any]) -> str:
    """Persist *doc* atomically and return its new ETag.

    Must be called while holding the tags lock for *name*.
    """
    doc["version"] = _TAGS_SCHEMA_VERSION
    doc["updated_at"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    payload = _serialize_tags_document(doc)
    etag = _tags_etag(payload)

    path = _index_tags_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"_tmp_tags_{uuid.uuid4().hex}.json")
    try:
        tmp.write_bytes(payload)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    try:
        st = path.stat()
        with _tags_cache_mutex:
            _tags_cache[name] = ((st.st_mtime_ns, st.st_size), copy.deepcopy(doc), etag)
            _tags_cache.move_to_end(name)
            while len(_tags_cache) > _TAGS_CACHE_MAX:
                _tags_cache.popitem(last=False)
    except OSError:
        # The file is written; only the cache priming failed. Drop the entry so
        # the next read goes to disk rather than serving something stale.
        _evict_tags_cache(name)

    return etag


def _evict_tags_cache(name: str) -> None:
    with _tags_cache_mutex:
        _tags_cache.pop(name, None)


def etag_matches(if_match: Optional[str], current: str) -> bool:
    """Return True when *if_match* permits a write against *current*.

    A missing header always passes (documented as the backwards-compatible
    default). ``*`` matches anything. Otherwise the comma-separated list is
    compared after stripping the weak-validator prefix.
    """
    if if_match is None:
        return True
    candidate = if_match.strip()
    if not candidate:
        return True
    if candidate == "*":
        return True
    current_value = current.strip('"')
    for token in candidate.split(","):
        token = token.strip()
        if token.startswith("W/"):
            token = token[2:].strip()
        if token.strip('"') == current_value:
            return True
    return False


def _mutate_tags(
    name: str,
    if_match: Optional[str],
    mutator,
) -> tuple[Any, str]:
    """Run *mutator* against tags.json under the tags lock.

    *mutator* receives the document and returns ``(result, changed)``. The
    If-Match check happens inside the lock, so a concurrent writer cannot slip
    between the check and the write.
    """
    _validate_index_name(name)
    _require_index_exists(name)
    with _get_tags_lock(name):
        doc, etag = _read_tags_document(name)
        if not etag_matches(if_match, etag):
            raise TagsPreconditionFailed(
                "The index tags have changed since they were read "
                f"(current ETag {etag}). Re-read and retry."
            )
        result, changed = mutator(doc)
        if changed:
            etag = _write_tags_document(name, doc)
        return result, etag


def _stamp(entry: dict[str, Any], client_id: Optional[str]) -> None:
    entry["updated_at"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    if client_id:
        entry["updated_by"] = client_id
    else:
        entry.pop("updated_by", None)


def get_index_tags(name: str) -> tuple[dict[str, Any], str]:
    """Return ``({"tags": [{"tag", "count"}], "total": n}, etag)``.

    Ordered by descending count, then by tag name ascending. These are the tags
    IN USE in this index: a name that no part carries is not here, even if it
    stays in the application-wide vocabulary (see :func:`get_tag_vocabulary`).
    """
    _validate_index_name(name)
    _require_index_exists(name)
    doc, etag = _read_tags_document(name)
    counts: dict[str, int] = {}
    for entry in doc["parts"].values():
        for tag in entry["tags"]:
            counts[tag] = counts.get(tag, 0) + 1
    items = [
        {"tag": tag, "count": count}
        for tag, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return {"tags": items, "total": len(items)}, etag


def get_part_tags(name: str, part_id: str) -> tuple[dict[str, Any], str]:
    """Return ``({"part_id", "tags"}, etag)`` for one part.

    A part with no tags returns an empty list rather than 404: the caller is
    asking what is tagged, and "nothing" is a valid answer for any registered
    id. Whether the id exists in the index is not checked here, so that a
    client can read tags for an id it is about to register.
    """
    _validate_index_name(name)
    _require_index_exists(name)
    doc, etag = _read_tags_document(name)
    entry = doc["parts"].get(part_id)
    return {"part_id": part_id, "tags": list(entry["tags"]) if entry else []}, etag


def set_part_tags(
    name: str,
    part_id: str,
    tags: list[str],
    if_match: Optional[str] = None,
    client_id: Optional[str] = None,
) -> tuple[dict[str, Any], str]:
    """Replace the whole tag set of *part_id*. An empty list clears it.

    Raises ``ValueError`` for invalid tag text or an unregistered part_id,
    ``KeyError`` when the index does not exist, and ``TagsPreconditionFailed``
    on an If-Match mismatch.
    """
    _validate_index_name(name)
    _require_index_exists(name)
    cleaned = _normalize_tag_list(tags)
    if part_id not in index_registry._registered_index_ids(name):
        raise ValueError(f"Part '{part_id}' is not registered in index '{name}'.")

    def mutator(doc: dict[str, Any]):
        for tag in cleaned:
            _warn_on_case_collision(doc, tag)
        previous = doc["parts"].get(part_id, {}).get("tags", [])
        if previous == cleaned:
            return {"part_id": part_id, "tags": list(cleaned)}, False
        if cleaned:
            entry = doc["parts"].setdefault(part_id, {})
            entry["tags"] = list(cleaned)
            _stamp(entry, client_id)
        else:
            doc["parts"].pop(part_id, None)
        return {"part_id": part_id, "tags": list(cleaned)}, True

    result = _mutate_tags(name, if_match, mutator)
    # Assigning a name is what puts it in the vocabulary, whichever route the
    # assignment came in by and whichever index it was applied to. Clearing a
    # part's tags records nothing, and takes nothing away: the names stay
    # available to pick from.
    _record_tags_in_vocabulary(cleaned)
    return result


def _bulk_tag_update(
    name: str,
    tag: str,
    part_ids: list[str],
    if_match: Optional[str],
    client_id: Optional[str],
    *,
    assign: bool,
) -> tuple[dict[str, Any], str]:
    """Shared body of assign/unassign: apply *tag* to every valid part id.

    Ids that are not registered are skipped and listed with a reason instead of
    failing the whole request, so one stale id in a gallery selection cannot
    discard the rest of the work.
    """
    _validate_index_name(name)
    _require_index_exists(name)
    cleaned_tag = normalize_tag(tag)
    limit = max_tags_per_part()
    registered = index_registry._registered_index_ids(name)

    def mutator(doc: dict[str, Any]):
        if assign:
            _warn_on_case_collision(doc, cleaned_tag)
        updated = 0
        skipped: list[dict[str, str]] = []
        seen: set[str] = set()
        for part_id in part_ids:
            pid = str(part_id).strip()
            if not pid or pid in seen:
                continue
            seen.add(pid)
            if pid not in registered:
                skipped.append({"part_id": pid, "reason": "not_registered"})
                continue
            entry = doc["parts"].get(pid)
            current = list(entry["tags"]) if entry else []
            if assign:
                if cleaned_tag in current:
                    continue
                if len(current) >= limit:
                    skipped.append({"part_id": pid, "reason": "tag_limit_reached"})
                    continue
                current.append(cleaned_tag)
                entry = doc["parts"].setdefault(pid, {})
                entry["tags"] = current
                _stamp(entry, client_id)
            else:
                if cleaned_tag not in current:
                    continue
                current.remove(cleaned_tag)
                if current:
                    entry = doc["parts"][pid]
                    entry["tags"] = current
                    _stamp(entry, client_id)
                else:
                    doc["parts"].pop(pid, None)
            updated += 1
        result = {"tag": cleaned_tag, "updated": updated, "skipped": skipped}
        return result, updated > 0

    outcome = _mutate_tags(name, if_match, mutator)
    if assign:
        # Recorded even when every id was skipped: the user named the tag, and
        # that decision is the thing worth keeping.
        _record_tags_in_vocabulary([cleaned_tag])
    return outcome


def assign_tag(
    name: str,
    tag: str,
    part_ids: list[str],
    if_match: Optional[str] = None,
    client_id: Optional[str] = None,
) -> tuple[dict[str, Any], str]:
    """Add *tag* to every registered id in *part_ids*."""
    return _bulk_tag_update(
        name, tag, part_ids, if_match, client_id, assign=True
    )


def unassign_tag(
    name: str,
    tag: str,
    part_ids: list[str],
    if_match: Optional[str] = None,
    client_id: Optional[str] = None,
) -> tuple[dict[str, Any], str]:
    """Remove *tag* from every registered id in *part_ids*."""
    return _bulk_tag_update(
        name, tag, part_ids, if_match, client_id, assign=False
    )


def delete_tag(
    name: str,
    tag: str,
    if_match: Optional[str] = None,
    client_id: Optional[str] = None,
) -> tuple[dict[str, Any], str]:
    """Remove *tag* from every part in the index."""
    _validate_index_name(name)
    _require_index_exists(name)
    cleaned_tag = normalize_tag(tag)

    def mutator(doc: dict[str, Any]):
        removed = 0
        for part_id in list(doc["parts"].keys()):
            entry = doc["parts"][part_id]
            if cleaned_tag not in entry["tags"]:
                continue
            remaining = [t for t in entry["tags"] if t != cleaned_tag]
            if remaining:
                entry["tags"] = remaining
                _stamp(entry, client_id)
            else:
                doc["parts"].pop(part_id, None)
            removed += 1
        return {"tag": cleaned_tag, "removed_from": removed}, removed > 0

    return _mutate_tags(name, if_match, mutator)


def prune_tags(name: str, part_ids: list[str]) -> int:
    """Drop tag entries for ids that are no longer registered.

    Called after parts are removed from an index. Without it the sidecar would
    grow forever: an id removed from the index would keep its tags, count
    towards every tag total, and never be reachable again.
    """
    try:
        _validate_index_name(name)
        _require_index_exists(name)
    except (ValueError, KeyError):
        return 0

    targets = {str(p).strip() for p in part_ids if str(p).strip()}
    if not targets:
        return 0

    def mutator(doc: dict[str, Any]):
        removed = 0
        for part_id in targets:
            if doc["parts"].pop(part_id, None) is not None:
                removed += 1
        return removed, removed > 0

    removed, _ = _mutate_tags(name, None, mutator)
    if removed:
        logger.info("[TAGS] dropped %d tag entries removed from index '%s'", removed, name)
    return removed


def tags_by_part(name: str) -> dict[str, list[str]]:
    """Return ``{part_id: [tag, ...]}`` for the named index (empty when none)."""
    try:
        doc, _ = _read_tags_document(name)
    except TagsFileError as exc:
        # Filtering and listing must not fail because the sidecar is unreadable;
        # they degrade to "no tags", but loudly.
        logger.warning("[TAGS] ignoring unreadable sidecar for index '%s': %s", name, exc)
        return {}
    return {pid: list(entry["tags"]) for pid, entry in doc["parts"].items()}


def _shape_map_tags_snapshot(name: str) -> tuple[dict[str, list[str]], str]:
    """Return ``(tags_map, etag)`` for the named index from a single read.

    The Shape Map caches key partly on tag state, and the payload bakes the tag
    labels in. Deriving both the ETag (for the cache key) and the ``tags_map``
    (for the payload) from one ``_read_tags_document`` call keeps them consistent
    -- a tag write between two separate reads could otherwise cache a payload
    under an ETag that does not match the tags it was built from. An unreadable
    sidecar degrades to "no tags" with a sentinel ETag, exactly as
    :func:`tags_by_part` degrades, so a corrupt file does not break the map.
    """
    try:
        doc, etag = _read_tags_document(name)
    except TagsFileError as exc:
        logger.warning(
            "[TAGS] ignoring unreadable sidecar for index '%s' (shape map): %s",
            name, exc,
        )
        return {}, "tags-unreadable"
    tags_map = {pid: list(entry["tags"]) for pid, entry in doc["parts"].items()}
    return tags_map, etag


def normalize_tag_filter(
    tags: Optional[list[str]],
    tag_mode: str = "any",
    tagged: Optional[bool] = None,
) -> Optional[dict[str, Any]]:
    """Validate a tag filter, returning None when no filtering was requested."""
    if tag_mode not in ("any", "all"):
        raise ValueError(f"Invalid tag_mode '{tag_mode}'. Must be 'any' or 'all'.")
    cleaned = [normalize_tag(t) for t in tags] if tags else []
    if not cleaned and tagged is None:
        return None
    return {"tags": cleaned, "tag_mode": tag_mode, "tagged": tagged}


def tag_filter_predicate(name: str, tag_filter: Optional[dict[str, Any]]):
    """Return a callable ``(part_id) -> bool`` implementing *tag_filter*.

    The sidecar is read once and captured, so a filter applied to a long list
    of hits does not re-read the file per item.
    """
    if not tag_filter:
        return lambda _part_id: True

    by_part = tags_by_part(name)
    wanted = tag_filter["tags"]
    mode = tag_filter["tag_mode"]
    tagged = tag_filter["tagged"]
    wanted_set = set(wanted)

    def predicate(part_id: str) -> bool:
        current = by_part.get(part_id) or []
        if tagged is True and not current:
            return False
        if tagged is False and current:
            return False
        if not wanted_set:
            return True
        if mode == "all":
            return wanted_set.issubset(current)
        return bool(wanted_set.intersection(current))

    return predicate


def tag_filter_diagnostics(
    name: str,
    tag_filter: Optional[dict[str, Any]],
    registered_ids,
    candidates_examined: int,
) -> Optional[dict[str, Any]]:
    """Explain an apparently empty tag-filtered search.

    Filtering runs after ranking over a bounded candidate pool, so a rare tag
    can produce zero hits even though matching parts exist. Without these two
    numbers a caller cannot tell "nothing carries this tag" from "seven parts
    carry it but none of them ranked inside the pool", and the second case
    looks like a bug. Returns None when no filter was requested, so the key is
    absent from an ordinary search.
    """
    if not tag_filter:
        return None
    matches = tag_filter_predicate(name, tag_filter)
    return {
        "tags": list(tag_filter.get("tags") or []),
        "tag_mode": tag_filter.get("tag_mode"),
        "tagged": tag_filter.get("tagged"),
        "candidates_examined": int(candidates_examined),
        "matching_parts_in_index": sum(1 for pid in registered_ids if matches(pid)),
    }


# ---------------------------------------------------------------------------
# Tag vocabulary -- DATA_ROOT/tag_vocabulary.json (one per application)
# ---------------------------------------------------------------------------
#
# The names a user has decided to use, kept apart from which parts carry them.
# Untagging the last part that carries a name does NOT take the name out of the
# vocabulary: the work of deciding what to call something is worth more than
# the assignment, and retyping "Socket head cap screw" from memory is exactly
# where typos and near-duplicates come from. So the vocabulary only ever grows
# by assignment, and shrinks only when asked to.
#
# One file for the application, not one per index. A name is a vocabulary
# choice rather than a property of an index, and an index rebuilt under a new
# name would otherwise start with an empty picker. The cost is that names only
# ever used in another index are offered here as well; clients list the tags in
# use first and vocabulary-only names after, so those sink to the bottom.
#
# It lives in this module anyway, because it is read and written on the same
# paths as tags.json and shares its normalization rules; what it does NOT share
# is the lock. ``_get_tags_lock`` is per index, so two threads tagging parts in
# two different indexes would both enter and one of the two names would be lost
# in the read-modify-write. The shared file needs the shared lock,
# ``_get_tag_vocabulary_lock()``.
#
# The file is meant to be edited by hand, so its shape is the minimum that can
# carry the information: {"tags": [...]}, one array of normalized names, no
# version and no timestamps. Order is whatever the file says; a hand-written
# grouping survives a round trip through the server.

# Upper bound on names in the vocabulary, for the whole application. This is a
# list a person reads and picks from, so its real limit is human; 1,000 is far
# past the point where picking from a list beats typing, and is here only to
# stop a runaway client or a scripted import from writing a file that can never
# be used or edited. A constant rather than an env var on purpose: a knob would
# need documenting in .env.example, two READMEs and the config header table,
# for a number nobody should ever need to change.
_TAG_VOCABULARY_MAX = 1000


def _tag_vocabulary_path() -> pathlib.Path:
    """Return the application-wide vocabulary path.

    Takes no index name: there is one vocabulary for the process. Reached
    through ``config`` rather than rebuilt from DATA_ROOT here, so a test or a
    tool that swaps the config name redirects this file with the rest.
    """
    return config.TAG_VOCABULARY_FILE


def _read_tag_vocabulary() -> list[str]:
    """Return the stored names, or an empty list.

    Deliberately forgiving where ``_read_tags_document`` is strict: a missing,
    empty, malformed or hand-mangled file degrades to "no vocabulary" with a
    warning instead of raising. The two sidecars hold different things. Losing
    tags.json loses which parts are what, so a bad file must stop the write
    that would overwrite it; the vocabulary is a convenience list, and refusing
    to tag anything because a suggestion file is broken would be the worse
    failure. Individual entries that are not valid tags are dropped the same
    way, so one bad line does not cost the rest of the file.

    Takes no lock: writers swap the file in with ``os.replace``, so a reader
    sees either the whole previous file or the whole new one. Not cached -- it
    is read on a tag write and when a client opens a tag picker, which is far
    too rare to pay for invalidation.
    """
    path = _tag_vocabulary_path()
    try:
        raw = json.loads(path.read_bytes().decode("utf-8"))
    except FileNotFoundError:
        return []
    except Exception as exc:
        logger.warning("[TAGS] ignoring unreadable tag vocabulary: %s", exc)
        return []

    if not isinstance(raw, dict):
        logger.warning("[TAGS] tag vocabulary is not a JSON object; ignoring it")
        return []
    entries = raw.get("tags")
    if entries is None:
        return []
    if not isinstance(entries, list):
        logger.warning(
            "[TAGS] tag vocabulary has a 'tags' member that is not a list; "
            "ignoring it"
        )
        return []

    out: list[str] = []
    dropped = 0
    for entry in entries:
        try:
            cleaned = normalize_tag(entry)
        except ValueError:
            dropped += 1
            continue
        if cleaned not in out:
            out.append(cleaned)
    if dropped:
        logger.warning(
            "[TAGS] dropped %d unusable entries from the tag vocabulary", dropped
        )
    return out[:_TAG_VOCABULARY_MAX]


def _write_tag_vocabulary(names: list[str]) -> None:
    """Persist *names* atomically, in the same way tags.json is written.

    Must be called while holding the tag vocabulary lock.
    """
    payload = json.dumps(
        {"tags": list(names)}, ensure_ascii=False, indent=2
    ).encode("utf-8")
    path = _tag_vocabulary_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"_tmp_tag_vocabulary_{uuid.uuid4().hex}.json")
    try:
        tmp.write_bytes(payload)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def get_tag_vocabulary() -> dict[str, Any]:
    """Return ``{"tags": [...], "total": n}`` for the application.

    These are the names available to CHOOSE from, across every index. They are
    not the tags in use -- see :func:`get_index_tags` for those, which is per
    index. A client offers the union, in-use names first.
    """
    names = _read_tag_vocabulary()
    return {"tags": names, "total": len(names)}


def add_tag_vocabulary(tags: list[str]) -> dict[str, Any]:
    """Add names to the vocabulary, keeping what is already there.

    Returns ``{"tags", "total", "added", "rejected"}``. Names already present
    are neither added nor rejected -- they simply are not in ``added``.
    ``rejected`` names are the ones the limit left no room for; they are
    reported rather than dropped, because a client that believes it registered
    a name and did not would offer a list that quietly misses entries.

    Raises ``ValueError`` for invalid tag text, so the vocabulary can never
    hold a name that :func:`normalize_tag` would refuse at assignment.
    """
    if isinstance(tags, str) or not isinstance(tags, (list, tuple)):
        raise ValueError("tags must be a list of strings.")
    cleaned = []
    for tag in tags:
        candidate = normalize_tag(tag)
        if candidate not in cleaned:
            cleaned.append(candidate)

    with _get_tag_vocabulary_lock():
        current = _read_tag_vocabulary()
        existing = set(current)
        added: list[str] = []
        rejected: list[str] = []
        for candidate in cleaned:
            if candidate in existing:
                continue
            if len(current) >= _TAG_VOCABULARY_MAX:
                rejected.append(candidate)
                continue
            current.append(candidate)
            existing.add(candidate)
            added.append(candidate)
        if added:
            _write_tag_vocabulary(current)

    if rejected:
        logger.warning(
            "[TAGS] tag vocabulary is full at %d names; %d name(s) were not added",
            _TAG_VOCABULARY_MAX,
            len(rejected),
        )
    return {
        "tags": current,
        "total": len(current),
        "added": added,
        "rejected": rejected,
    }


def remove_tag_vocabulary(tag: str) -> dict[str, Any]:
    """Remove one name from the vocabulary.

    This touches the vocabulary ONLY. Parts carrying *tag* keep it, in every
    index, and the tag goes on appearing in :func:`get_index_tags` and in every
    filter -- the two are different things, and this one is for tidying away a
    name that was mistyped or decided against. To take a tag off the parts of
    one index, use :func:`delete_tag`.

    Returns ``{"tag", "removed", "tags", "total"}``; ``removed`` is False when
    the name was not in the vocabulary, which is not an error.
    """
    cleaned = normalize_tag(tag)
    with _get_tag_vocabulary_lock():
        current = _read_tag_vocabulary()
        removed = cleaned in current
        if removed:
            current = [t for t in current if t != cleaned]
            _write_tag_vocabulary(current)
    return {
        "tag": cleaned,
        "removed": removed,
        "tags": current,
        "total": len(current),
    }


def _record_tags_in_vocabulary(tags) -> None:
    """Remember *tags* as names in use. Never raises.

    Called after a successful assignment, from every path that can put a tag on
    a part, so the vocabulary keeps the name once the last part loses it. It
    takes no index name on purpose: whichever index the tag was applied to, the
    name goes into the one shared vocabulary.

    Called AFTER the tags.json write rather than inside its mutator: the tags
    lock is non-reentrant and the mutator holds it, and taking a second lock
    underneath it would put an ordering rule where none is needed. The write is
    therefore not atomic with the assignment, which costs nothing that matters
    -- the worst case is a name in the vocabulary that no part carries, which
    is the state this file exists to hold anyway.

    A vocabulary failure must never fail a tag assignment, so everything is
    caught and logged. A full vocabulary is logged by add_tag_vocabulary; the
    assignment itself still succeeds, because refusing to tag a part over a
    suggestion list would be the wrong trade.
    """
    try:
        wanted = [t for t in tags if t]
        if wanted:
            add_tag_vocabulary(wanted)
    except Exception as exc:
        logger.warning(
            "[TAGS] could not record %s in the tag vocabulary: %s", tags, exc
        )
