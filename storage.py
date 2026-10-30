"""CAD file storage and ingest: the upload store, ZIP extraction, and lookups.

Owns the permanent CAD store (get_CAD_shared_dir) and every operation that puts
bytes into it or resolves them back out: single-file upload, ZIP extraction,
server-side path import, and the persistence lookups/listings. Index records
carry only the SHA-256 file_id, so this module holds the sole copy of every
registered CAD payload.

This module is the storage engine; ``routers/files.py`` is the HTTP layer that
exposes ``/files`` endpoints and delegates the actual work here. It is named
``storage`` rather than ``files`` on purpose: ``files`` collided with the
FastAPI upload parameter ``files: list[UploadFile]`` and with the
``routers/files.py`` basename, which forced ``import files as filestore``
aliases and made the code read ambiguously.

Reference the swappable names as module attributes -- ``storage.find_persistent_CAD_file``,
``storage.get_CAD_shared_dir``, ``storage.read_too_heavy_files``,
``storage.upload_CAD_file_persistent``. Do NOT write ``from storage import <one of
those>``: a ``from`` import binds the name at import time, so a test that swaps
``storage.<name>`` would never reach a module that imported the old object.

Everything else defined here is stable and is re-exported from ``core`` for
backwards compatibility, so existing ``core.<name>`` and ``except core.Zip*``
call sites keep working. See "The re-export convention" in
docs/architecture.md for the failure mode the attribute-reference rule avoids.

Depends only on the standard library, ``fastapi.UploadFile`` and ``config``; it
must never import ``core`` (the dependency runs one way, from core down to storage).
"""

import hashlib
import io
import logging
import os
import pathlib
import re
import tempfile
import uuid
import zipfile
from typing import Any, Optional

from fastapi import UploadFile

import config

logger = logging.getLogger(__name__)


# Allowed CAD file extensions for upload/ZIP extraction.
#
# Mirrors cadNameFilters() in hoops_ai_qt_sandbox/SimilarityIndexPanel.cpp, which
# in turn mirrors the "All Supported Files" list of HOOPS Exchange. Kept lower
# case; every comparison lower-cases the candidate suffix first, so files saved
# with upper-case extensions (part.SLDPRT is common with SolidWorks exports) are
# accepted on case-sensitive filesystems too.
CAD_ALLOWED_EXTENSIONS = frozenset({
    ".3ds", ".3mf", ".3dxml", ".sat", ".sab", ".dgn",
    ".dwg", ".dxf", ".dwf", ".dwfx", ".model", ".dlv",
    ".exp", ".session", ".catpart", ".catproduct", ".catshape",
    ".catdrawing", ".cgr", ".dae", ".prt", ".neu",
    ".asm", ".xas", ".xpr", ".fbx", ".gltf", ".glb",
    ".arc", ".unv", ".mf1", ".pkg", ".ifc", ".ifczip",
    ".igs", ".iges", ".ipt", ".iam", ".jt", ".kmz",
    ".nwd", ".prc", ".x_t", ".xmt", ".x_b", ".xmt_txt",
    ".rvt", ".3dm", ".stp", ".step", ".stpz", ".stl",
    ".par", ".pwd", ".psm", ".sldprt", ".sldasm", ".sldfpp",
    ".u3d", ".vda", ".wrl", ".vrml", ".obj", ".xv3",
    ".xv0", ".hsf",
})


def read_too_heavy_files(
    directory: Optional["pathlib.Path"] = None,
) -> list[str]:
    """Paths hoops_ai deferred to its built-in single-worker RAM fallback.

    ``too_heavy_files.log`` lists them: ``#`` comment lines, then one path per
    line. It is informational only -- these files are usually embedded
    successfully, just slowly -- but it is what tells an operator which parts
    made the run expensive.

    *directory* is the pass's ``log_dir``. The process working directory is
    tried as well: ``log_dir`` defaults to ``'.'``, so should a release ever stop
    honouring it, the file still lands in the CWD and is still found. That
    fallback is also what makes an empty result trustworthy -- a run reporting no
    heavy files has been checked in both places, not just the redirected one.
    """
    candidates: list[pathlib.Path] = []
    if directory is not None:
        candidates.append(pathlib.Path(directory))
    candidates.append(pathlib.Path.cwd())

    for base in candidates:
        path = base / "too_heavy_files.log"
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        return [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]
    return []


class ZipSlipError(RuntimeError):
    """Raised when a ZIP archive contains a Zip Slip path traversal attempt."""


class ZipSizeLimitError(RuntimeError):
    """Raised when a ZIP archive's uncompressed size exceeds the configured limit."""


class ZipFileLimitError(RuntimeError):
    """Raised when a ZIP archive contains more files than the configured limit."""


class UploadSizeLimitError(RuntimeError):
    """Raised when a single uploaded file exceeds the configured size limit."""


def get_CAD_shared_dir() -> pathlib.Path:
    """Return the CAD store: the single permanent home of every uploaded file.

    Index records carry only the SHA-256 file_id, so this directory holds the
    only copy of the CAD payload behind every registered index. It ranks with
    ``indexes/`` as permanent data and is never cleared by the server.
    ``CAD_UPLOAD_DIR`` is merely the default location; HOOPS_AI_CAD_SHARED_DIR
    overrides it. Every reader and writer must go through this function so the
    configured location and the location actually used can never diverge.
    """
    value = os.environ.get("HOOPS_AI_CAD_SHARED_DIR") or config.read_env_file().get(
        "HOOPS_AI_CAD_SHARED_DIR"
    )
    if value:
        return pathlib.Path(value).expanduser().resolve()
    return config.CAD_UPLOAD_DIR.resolve()


_UPLOAD_CHUNK_BYTES = 1024 * 1024


_MAX_UPLOAD_BYTES_DEFAULT = 4 * 1024 * 1024 * 1024  # 4 GB


_FILE_ID_RE = re.compile(r"\A[0-9a-f]{64}\Z")


_EXISTS_MAX_IDS_DEFAULT = 1000


def _max_upload_bytes() -> int:
    """Per-file upload cap, from HOOPS_AI_MAX_UPLOAD_BYTES."""
    return config._env_int("HOOPS_AI_MAX_UPLOAD_BYTES", _MAX_UPLOAD_BYTES_DEFAULT, minimum=1)


def exists_max_ids() -> int:
    """Maximum number of ids accepted by a single existence check."""
    return config._env_int("HOOPS_AI_EXISTS_MAX_IDS", _EXISTS_MAX_IDS_DEFAULT, minimum=1)


def upload_CAD_file_persistent(upload_file: UploadFile) -> tuple[str, pathlib.Path, bool]:
    """Upload a CAD file and store it persistently using SHA-256 content hash as file_id.

    Returns (file_id, path, already_existed). Uploading the same file twice is
    idempotent.

    The body is streamed to disk in chunks and hashed as it goes, so a large
    assembly never has to fit in memory and is read only once. The bytes land in
    a temporary ``.part`` file that is renamed into place only after the whole
    stream has been consumed, so an aborted upload can never leave a truncated
    file that would later be served under a hash it does not match.

    Raises ``UploadSizeLimitError`` when the body exceeds
    HOOPS_AI_MAX_UPLOAD_BYTES, and ``RuntimeError`` when the filename is missing.
    """
    if not upload_file.filename:
        raise RuntimeError("Uploaded CAD file must have a filename.")

    safe_filename = pathlib.PurePath(upload_file.filename).name
    CAD_shared_dir = get_CAD_shared_dir()
    CAD_shared_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = _max_upload_bytes()
    digest = hashlib.sha256()
    total = 0
    tmp_path = CAD_shared_dir / f".upload_{uuid.uuid4().hex}.part"
    try:
        with tmp_path.open("wb") as handle:
            while True:
                chunk = upload_file.file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise UploadSizeLimitError(
                        f"Uploaded file '{safe_filename}' exceeds the "
                        f"{max_bytes // (1024 * 1024)} MB per-file limit."
                    )
                digest.update(chunk)
                handle.write(chunk)

        file_id = digest.hexdigest()
        dest_path = CAD_shared_dir / f"{file_id}_{safe_filename}"
        existed = dest_path.exists()
        if not existed:
            os.replace(tmp_path, dest_path)
        return file_id, dest_path, existed
    finally:
        # No-op once the rename succeeded; cleans up on error or duplicate.
        tmp_path.unlink(missing_ok=True)


def known_persistent_CAD_file_ids() -> set[str]:
    """Return the set of file_ids currently held in the CAD store.

    Scans the directory once. Callers checking many ids must use this rather
    than calling find_persistent_CAD_file() in a loop, which globs per id.
    """
    CAD_shared_dir = get_CAD_shared_dir()
    if not CAD_shared_dir.exists():
        return set()
    known: set[str] = set()
    for path in CAD_shared_dir.iterdir():
        if not path.is_file():
            continue
        head, sep, _ = path.name.partition("_")
        if sep and _FILE_ID_RE.match(head):
            known.add(head)
    return known


def check_persistent_CAD_files(file_ids: list[str]) -> dict[str, list[str]]:
    """Split *file_ids* into known / unknown / invalid, preserving input order.

    Lets a client skip uploading files the server already has. Malformed ids are
    reported separately instead of being lumped in with "unknown", so a client
    that hashes incorrectly sees a distinct signal rather than silently
    re-uploading everything on every run.
    """
    known_ids = known_persistent_CAD_file_ids()
    known: list[str] = []
    unknown: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for raw in file_ids:
        candidate = str(raw).strip()
        if candidate in seen:
            continue
        seen.add(candidate)
        if not _FILE_ID_RE.match(candidate.lower()) or candidate != candidate.lower():
            invalid.append(candidate)
        elif candidate in known_ids:
            known.append(candidate)
        else:
            unknown.append(candidate)
    return {"known": known, "unknown": unknown, "invalid": invalid}


def find_persistent_CAD_file(file_id: str) -> pathlib.Path:
    """Look up a previously uploaded CAD file by its file_id (SHA-256 hash).

    Raises RuntimeError if no matching file is found.
    """
    CAD_shared_dir = get_CAD_shared_dir()
    matches = list(CAD_shared_dir.glob(f"{file_id}_*"))
    if not matches:
        raise RuntimeError(
            f"No uploaded file found for file_id '{file_id}'. "
            "Upload the file first via POST /files/upload."
        )
    return matches[0]


def map_persistent_CAD_files(file_ids: list[str]) -> dict[str, "pathlib.Path"]:
    """Resolve many file_ids to their persistent CAD paths in a single scan.

    Equivalent to calling :func:`find_persistent_CAD_file` per id, but reads the
    CAD store directory once instead of globbing it for every id. This turns a
    bulk resolve from O(N * store_size) into O(store_size + N); for a 10k-file
    corpus over a 10k-file store the difference is minutes of directory scanning
    before embedding can even start. Only ids present in the store appear in the
    result; missing ids are omitted so callers can decide how to report them.
    """
    wanted = set(file_ids)
    result: dict[str, pathlib.Path] = {}
    if not wanted:
        return result
    CAD_shared_dir = get_CAD_shared_dir()
    if not CAD_shared_dir.exists():
        return result
    with os.scandir(CAD_shared_dir) as it:
        for entry in it:
            name = entry.name
            # Stored files are "<sha256>_<originalname>": the id is the 64-char
            # prefix ending at the first underscore.
            if name[:64] not in wanted or name[64:65] != "_":
                continue
            if not entry.is_file():
                continue
            fid = name[:64]
            if fid not in result:
                # One path per id is all an embedding run needs, and the bytes
                # behind every path sharing an id are identical by construction
                # (the id IS their SHA-256). The extra names are not an error,
                # so this keeps the first and says nothing -- which is why the
                # fact that one geometry was uploaded under several original
                # filenames used to be invisible. See
                # :func:`map_duplicate_CAD_filenames` for that view; do not
                # change the rule here, callers depend on getting exactly one
                # path per id.
                result[fid] = pathlib.Path(entry.path)
    return result


def map_duplicate_CAD_filenames(file_ids: list[str]) -> dict[str, list[str]]:
    """Return the original filenames of every *file_id* stored under two or more.

    Same single-scan policy as :func:`map_persistent_CAD_files`, answering the
    other question: not "where is this file" but "was this one geometry
    uploaded under several names". Identical bytes hash to one file_id and are
    therefore one index entry, yet a client that uploads a folder sends every
    copy, so the store keeps ``<sha256>_a.stp`` and ``<sha256>_b.stp`` side by
    side. Nothing in the index is duplicated; only the original names are.

    Only ids carrying two or more names are returned, so the result is
    proportional to the answer (hundreds) rather than to the corpus (tens of
    thousands) -- a caller that wants the usual two headline figures still has
    them: the number of such geometries is ``len(result)`` and the number of
    redundant copies is ``sum(len(v) for v in result.values()) - len(result)``,
    since every omitted id contributes exactly zero of the latter.

    The values are the original filenames with the ``<sha256>_`` prefix
    removed, never server paths: this feeds a response that is read by a
    browser, and the layout of the CAD store is not the client's business.
    Each id's names are sorted so two calls read the same.
    """
    wanted = set(file_ids)
    names_by_id: dict[str, list[str]] = {}
    if not wanted:
        return {}
    CAD_shared_dir = get_CAD_shared_dir()
    if not CAD_shared_dir.exists():
        return {}
    with os.scandir(CAD_shared_dir) as it:
        for entry in it:
            name = entry.name
            # Stored files are "<sha256>_<originalname>", as above.
            if name[:64] not in wanted or name[64:65] != "_":
                continue
            if not entry.is_file():
                continue
            names_by_id.setdefault(name[:64], []).append(name[65:])
    return {
        fid: sorted(names)
        for fid, names in names_by_id.items()
        if len(names) > 1
    }


def original_filename(stored: Any) -> str:
    """The name a client uploaded a file under, from its stored name or path.

    Stored files are ``<sha256>_<originalname>``; that prefix is this server's
    bookkeeping, not something a client ever saw. Anything leaving the server
    for a report or a response goes through here, so neither the rename rule
    nor the directory the store happens to sit in is ever disclosed. A name
    that does not carry the prefix is returned as its own basename, which is
    what makes this safe to point at paths from the SDK's own logs.
    """
    name = pathlib.PurePath(str(stored or "").replace("\\", "/")).name
    if len(name) > 65 and name[64:65] == "_" and _FILE_ID_RE.match(name[:64].lower()):
        return name[65:]
    return name


def delete_persistent_CAD_file(file_id: str) -> bool:
    """Delete a persistent CAD file by file_id. Returns True if a file was deleted."""
    CAD_shared_dir = get_CAD_shared_dir()
    matches = list(CAD_shared_dir.glob(f"{file_id}_*"))
    for path in matches:
        path.unlink(missing_ok=True)
    return len(matches) > 0


def list_persistent_CAD_files() -> list[dict[str, Any]]:
    """Return metadata for all persistently uploaded CAD files."""
    CAD_shared_dir = get_CAD_shared_dir()
    if not CAD_shared_dir.exists():
        return []
    result = []
    for path in sorted(CAD_shared_dir.iterdir()):
        if not path.is_file():
            continue
        parts = path.name.split("_", 1)
        if len(parts) == 2 and len(parts[0]) == 64:  # SHA-256 hex = 64 chars
            result.append(
                {
                    "file_id": parts[0],
                    "filename": parts[1],
                    "size": path.stat().st_size,
                }
            )
    return result


def server_paths_enabled() -> bool:
    """Whether jobs may read CAD from arbitrary server-side paths.

    Off by default: it lets any client with API access read any file the server
    process can read, and it bypasses the upload path's size and extension
    checks. It exists only for an administrator seeding a fresh installation
    from a local corpus, so it must be switched on deliberately.
    """
    value = os.environ.get("HOOPS_AI_ALLOW_SERVER_PATHS") or config.read_env_file().get(
        "HOOPS_AI_ALLOW_SERVER_PATHS", ""
    )
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def resolve_server_paths(paths: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    """Import CAD files from server-side *paths* into the CAD store.

    Returns ``(file_ids, errors)``. Every rejected path yields an error entry,
    so a caller can always account for each input. Raises ``PermissionError``
    when the feature is disabled.
    """
    if not server_paths_enabled():
        raise PermissionError(
            "Server-side paths are disabled. Set HOOPS_AI_ALLOW_SERVER_PATHS=true "
            "to enable them (administrator seeding only)."
        )
    file_ids: list[str] = []
    errors: list[dict[str, Any]] = []
    for raw in paths:
        path = pathlib.Path(raw).expanduser()
        try:
            if not path.is_file():
                raise RuntimeError("not a file")
            if path.suffix.lower() not in CAD_ALLOWED_EXTENSIONS:
                raise RuntimeError(f"unsupported CAD extension '{path.suffix}'")
            with path.open("rb") as fh:
                fid, _, _ = upload_CAD_file_persistent(
                    _BytesUploadFile(fh.read(), path.name)
                )
            file_ids.append(fid)
        except Exception as exc:
            errors.append({"filename": str(raw), "detail": str(exc)})
    return file_ids, errors


# Default limits for ZIP extraction (also used by routers). Both can be raised
# for bulk registration via HOOPS_AI_ZIP_MAX_TOTAL_BYTES / HOOPS_AI_ZIP_MAX_FILES;
# the module-level values stay as the defaults so existing callers that pass
# nothing keep the historical behaviour.
ZIP_MAX_TOTAL_BYTES = 500 * 1024 * 1024  # 500 MB uncompressed


ZIP_MAX_FILES = 50


def zip_max_total_bytes() -> int:
    """Uncompressed-size cap for a ZIP archive, from HOOPS_AI_ZIP_MAX_TOTAL_BYTES."""
    return config._env_int("HOOPS_AI_ZIP_MAX_TOTAL_BYTES", ZIP_MAX_TOTAL_BYTES, minimum=1)


def zip_max_files() -> int:
    """CAD-file-count cap for a ZIP archive, from HOOPS_AI_ZIP_MAX_FILES."""
    return config._env_int("HOOPS_AI_ZIP_MAX_FILES", ZIP_MAX_FILES, minimum=1)


class _BytesUploadFile:
    """Minimal duck-typed UploadFile used to pass in-memory bytes to core helpers."""

    def __init__(self, data: bytes, filename: str) -> None:
        self.filename = filename
        self.file = io.BytesIO(data)


def extract_zip_cad_files(
    zip_data: bytes,
    max_total_bytes: Optional[int] = None,
    max_files: Optional[int] = None,
) -> tuple[list[tuple[str, str]], list[dict]]:
    """Extract a ZIP archive and upload each valid CAD file persistently.

    Returns ``(resolved, errors)`` where *resolved* is a list of
    ``(file_id, display_name)`` tuples for successfully processed files, and
    *errors* is a list of ``{"filename": ..., "detail": ...}`` dicts for files
    that could not be processed.

    Raises:
        ZipSlipError: if any member path resolves outside the extraction directory.
        ZipSizeLimitError: if the total uncompressed size exceeds *max_total_bytes*.
        ZipFileLimitError: if the number of CAD files exceeds *max_files*.
    """
    if max_total_bytes is None:
        max_total_bytes = zip_max_total_bytes()
    if max_files is None:
        max_files = zip_max_files()

    resolved: list[tuple[str, str]] = []
    errors: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir).resolve()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            total_size = 0
            file_count = 0

            for member in zf.infolist():
                if member.is_dir():
                    continue

                # Zip Slip guard: resolve the destination and verify it stays
                # inside the temporary directory.
                member_dest = (tmp_path / member.filename).resolve()
                try:
                    member_dest.relative_to(tmp_path)
                except ValueError:
                    raise ZipSlipError(
                        f"Zip Slip detected: '{member.filename}' "
                        "resolves outside the extraction directory."
                    )

                suffix = pathlib.Path(member.filename).suffix.lower()
                if suffix not in CAD_ALLOWED_EXTENSIONS:
                    continue  # skip non-CAD entries silently

                total_size += member.file_size
                if total_size > max_total_bytes:
                    raise ZipSizeLimitError(
                        f"ZIP archive total uncompressed size exceeds the "
                        f"{max_total_bytes // (1024 * 1024)} MB limit."
                    )

                file_count += 1
                if file_count > max_files:
                    raise ZipFileLimitError(
                        f"ZIP archive contains more than {max_files} CAD files."
                    )

                member_dest.parent.mkdir(parents=True, exist_ok=True)
                member_dest.write_bytes(zf.read(member.filename))

                display_name = pathlib.Path(member.filename).name
                try:
                    fake = _BytesUploadFile(member_dest.read_bytes(), display_name)
                    fid, _, _ = upload_CAD_file_persistent(fake)
                    resolved.append((fid, display_name))
                except Exception as exc:
                    errors.append({"filename": display_name, "detail": str(exc)})

    return resolved, errors


def get_shared_CAD_file(cad_file_path: str) -> pathlib.Path:
    requested_path = pathlib.Path(cad_file_path).expanduser()

    if requested_path.is_absolute():
        resolved_path = requested_path.resolve()
    else:
        CAD_shared_dir = get_CAD_shared_dir()
        resolved_path = CAD_shared_dir.joinpath(requested_path).resolve()
        try:
            resolved_path.relative_to(CAD_shared_dir)
        except ValueError as exc:
            raise RuntimeError(f"CAD file must be under shared folder: {CAD_shared_dir}") from exc

    if not resolved_path.exists() or not resolved_path.is_file():
        raise RuntimeError(f"CAD file not found: {resolved_path}")
    return resolved_path
