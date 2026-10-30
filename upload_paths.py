"""Where the client said each file came from.

A browser that uploads a folder knows the layout the operator sees; the server
does not, because a stored file is named ``<sha256>_<basename>`` and nothing
else. This module keeps the client's own relative path for a ``file_id`` so a
report can name a file the way the operator named it.

Nothing here ever reaches the filesystem as a path component. The values come
from ``webkitRelativePath``, which is attacker controlled, so a value that
could not have come from a folder picker is refused outright rather than
repaired, and the one thing it is used for is display.

The store is a single JSON file at ``DATA_ROOT/upload_paths.json``, following
``tag_vocabulary.json`` rather than the per-index sidecars: where a file came
from is a property of the upload, and one ``file_id`` can be registered into
several indexes.
"""
import json
import re
from pathlib import Path

import config
from locks import _get_upload_paths_lock

_SCHEMA_VERSION = 1

# A folder picker produces short, shallow paths. These bounds exist so a forged
# value cannot grow the store or a report line without limit; they are far
# above anything a real tree produces.
_MAX_PATH_LENGTH = 400
_MAX_SEGMENTS = 32

# One file can legitimately appear in several assemblies -- that is the case
# this store exists to show -- but a file seen in dozens of folders tells an
# operator nothing more than a handful does.
_MAX_PATHS_PER_ID = 8

_FILE_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def _store_path() -> Path:
    """Return the path of the JSON store (patched wholesale by the tests)."""
    return config.DATA_ROOT / "upload_paths.json"


def sanitize_relative_path(raw) -> str:
    """Return a safe relative path for display, or ``""`` to refuse the value.

    Refusal, not repair. Salvaging the basename out of ``..\\..\\evil.stp``
    would put a string the operator never typed in front of them as if it were
    their own folder layout, and it would leave two fallbacks that look alike.
    Returning ``""`` leaves exactly one: the original filename, which is
    already on disk.

    Refused: anything that is not a string; empty or blank; longer than
    ``_MAX_PATH_LENGTH``; containing a control character (a newline would forge
    a line in the line-oriented registration report); rooted at a separator;
    carrying a drive letter; containing a ``..`` segment; deeper than
    ``_MAX_SEGMENTS``.
    """
    if not isinstance(raw, str):
        return ""
    text = raw.strip()
    if not text or len(text) > _MAX_PATH_LENGTH:
        return ""
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in text):
        return ""
    if _DRIVE_RE.match(text):
        return ""
    if text[0] in ("/", "\\"):
        return ""
    segments = [s for s in text.replace("\\", "/").split("/") if s not in ("", ".")]
    if not segments or len(segments) > _MAX_SEGMENTS:
        return ""
    if any(s == ".." for s in segments):
        return ""
    return "/".join(segments)


def _parse_document(doc) -> dict:
    """Return ``{file_id: [path, ...]}`` from a parsed document.

    Every value is re-checked here, not only on the way in: the file is plain
    JSON on an operator's disk, and a value that never passed
    :func:`sanitize_relative_path` must not reach a report because someone
    edited it in.
    """
    if not isinstance(doc, dict):
        return {}
    raw = doc.get("paths")
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for fid, values in raw.items():
        if not isinstance(fid, str) or not _FILE_ID_RE.match(fid):
            continue
        if not isinstance(values, list):
            continue
        kept: list = []
        for value in values:
            safe = sanitize_relative_path(value)
            if safe and safe not in kept:
                kept.append(safe)
            if len(kept) >= _MAX_PATHS_PER_ID:
                break
        if kept:
            out[fid] = kept
    return out


def read_all() -> dict:
    """Return every recorded path. Missing or unreadable store means none."""
    path = _store_path()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return _parse_document(doc)


def _write_all(data: dict) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"schema_version": _SCHEMA_VERSION, "paths": data},
        ensure_ascii=False,
        indent=2,
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def record(pairs) -> int:
    """Record ``(file_id, client path)`` pairs. Return how many were new.

    One read-modify-write for the whole batch, under the process lock, so an
    upload of ten thousand files touches the file once. Never raises: this is
    display data, and losing it must not fail an upload that otherwise worked.
    """
    cleaned: list = []
    for fid, raw in pairs or ():
        if not isinstance(fid, str) or not _FILE_ID_RE.match(fid):
            continue
        safe = sanitize_relative_path(raw)
        if safe:
            cleaned.append((fid, safe))
    if not cleaned:
        return 0
    added = 0
    try:
        with _get_upload_paths_lock():
            data = read_all()
            for fid, safe in cleaned:
                values = data.setdefault(fid, [])
                if safe in values or len(values) >= _MAX_PATHS_PER_ID:
                    continue
                values.append(safe)
                added += 1
            if added:
                _write_all({k: v for k, v in data.items() if v})
    except OSError:
        return 0
    return added


def paths_for(file_ids) -> dict:
    """Return the recorded paths for the asked-for ids that have any."""
    wanted = {f for f in (file_ids or ()) if isinstance(f, str)}
    if not wanted:
        return {}
    return {fid: v for fid, v in read_all().items() if fid in wanted}


def decorate_names(recorded, names):
    """Return ``names`` with each one replaced by its recorded path, if any.

    Matching is by basename, because the recorded path and the stored name were
    produced from the same upload. A name with no recorded path is returned
    unchanged -- everything uploaded before this store existed is in that case,
    and an operator must see the filename rather than a blank or "unknown".
    """
    if not recorded:
        return list(names or [])
    by_base: dict = {}
    for value in recorded:
        by_base.setdefault(str(value).rsplit("/", 1)[-1], []).append(value)
    out: list = []
    for name in names or []:
        base = str(name).replace("\\", "/").rsplit("/", 1)[-1]
        candidates = by_base.get(base)
        out.append(candidates.pop(0) if candidates else name)
    return out
