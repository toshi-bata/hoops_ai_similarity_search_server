import pathlib

import core
import storage
from fastapi import APIRouter, Body, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/files", tags=["File Management"])


class FileExistsRequest(BaseModel):
    file_ids: list[str] = Field(
        default_factory=list,
        description="Client-computed SHA-256 hex digests to check against the CAD store.",
    )


class FileExistsResponse(BaseModel):
    known: list[str] = []
    unknown: list[str] = []
    invalid: list[str] = []


@router.post("/upload")
def upload_file(file: UploadFile = File(...)):
    """Upload a CAD file to the server.

    Returns a ``file_id`` derived from the file's SHA-256 hash.
    Uploading the same file again returns the same ``file_id`` without re-storing
    the file (``already_existed`` will be ``true``).
    Pass the ``file_id`` to any processing endpoint instead of re-uploading the file.

    To send many files in one request, use ``POST /files/upload-batch``.
    """
    try:
        file_id, path, existed = storage.upload_CAD_file_persistent(file)
        return {
            "file_id": file_id,
            # The name the client sent, not the name on disk. Files are stored
            # as "<sha256>_<original name>", which is this server's bookkeeping
            # and no business of a caller's: a client that is handed it has to
            # know the rename rule to show the file the way its operator named
            # it.
            "filename": storage.original_filename(path.name),
            "already_existed": existed,
        }
    except core.UploadSizeLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/upload-batch")
def upload_files(
    files: list[UploadFile] = File(...),
    relative_paths: list[str] = Form(default=[]),
):
    """Upload several CAD files in one request.

    A separate endpoint rather than an extra parameter on ``POST /files/upload``:
    that endpoint returns a single object, and keeping the one-file and
    many-file shapes apart means neither has to be read conditionally.

    One failing file does not abort the request. Every input is reported in
    either ``files`` or ``errors``, so ``len(files) + len(errors)`` always equals
    the number of parts sent.

    ``relative_paths`` is optional and, when sent, is positional: the *i*-th
    value describes the *i*-th file, and it is the client's own folder path for
    it (a browser's ``webkitRelativePath``). It is carried in its own field
    rather than as the multipart filename for one reason: the multipart
    filename is what the stored name is derived from, and
    ``PurePath(name).name`` does not strip backslashes on POSIX, so a path sent
    that way would put ``..\\..\\x.stp`` into a filename on a Linux host. Kept
    apart, the save path is derived from exactly what it was derived from
    before, and the client's path is recorded as display data only -- refused
    outright if it is absolute, rooted or contains ``..`` (see
    :func:`upload_paths.sanitize_relative_path`).
    """
    uploaded: list[dict] = []
    errors: list[dict] = []
    recorded: list[tuple[str, str]] = []
    for position, item in enumerate(files):
        display_name = pathlib.PurePath(item.filename or "").name or "<unnamed>"
        try:
            file_id, path, existed = storage.upload_CAD_file_persistent(item)
            uploaded.append(
                {
                    "file_id": file_id,
                    # The name the client sent; see POST /files/upload.
                    "filename": storage.original_filename(path.name),
                    "already_existed": existed,
                }
            )
            if position < len(relative_paths):
                recorded.append((file_id, relative_paths[position]))
        except Exception as exc:
            errors.append({"filename": display_name, "detail": str(exc)})
    core.record_upload_paths(recorded)
    return {"files": uploaded, "errors": errors, "count": len(uploaded)}


@router.post("/exists", response_model=FileExistsResponse)
def check_files_exist(payload: FileExistsRequest = Body(...)):
    """Report which of the given file_ids the server already holds.

    Lets a bulk-registration client hash its folder locally and upload only what
    is missing, instead of re-sending hundreds of megabytes on every run.
    Duplicate ids are collapsed; malformed ones come back under ``invalid``
    rather than ``unknown``, so a client that hashes incorrectly gets a distinct
    signal instead of silently re-uploading everything each time.
    """
    limit = core.exists_max_ids()
    if len(payload.file_ids) > limit:
        raise HTTPException(
            status_code=422,
            detail=f"Too many file_ids: {len(payload.file_ids)} (limit {limit}).",
        )
    return core.check_persistent_CAD_files(payload.file_ids)


@router.post("/upload-from-path")
def upload_from_path(
    file_path: str = Query(
        ...,
        description=(
            "Server-side path to a CAD file or ZIP archive. "
            "Absolute paths (e.g. C:\\\\temp\\\\parts.zip) are accepted as-is. "
            "Relative paths are resolved under the configured shared folder "
            "(HOOPS_AI_CAD_SHARED_DIR). "
            "ZIP archives are extracted and every contained CAD file is uploaded "
            "individually. Single CAD files are uploaded directly. "
            "Returns a list of file_ids that can be passed to /similarity/* endpoints."
        ),
    ),
):
    """Upload a CAD file or ZIP archive by server-side file path.

    This endpoint is the recommended way for MCP / scripted clients to register
    files that already exist on the server's filesystem (e.g. a path the user
    dragged into the chat).  Pass the returned ``file_ids`` to
    ``POST /similarity/compare``,
    ``POST /similarity/index/add``, etc.

    * **Single CAD file** – uploaded and a single entry is returned.
    * **ZIP archive** – extracted, every recognised CAD file is uploaded, and all
      resulting ``file_id`` values are returned together.

    Supported CAD extensions: see ``core.CAD_ALLOWED_EXTENSIONS`` (mirrors the
    HOOPS Exchange "All Supported Files" list).

    ZIP limits default to 500 MB uncompressed and 50 CAD files per archive, and
    can be raised with HOOPS_AI_ZIP_MAX_TOTAL_BYTES / HOOPS_AI_ZIP_MAX_FILES.
    """
    try:
        resolved_path = core.get_shared_CAD_file(file_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    suffix = resolved_path.suffix.lower()

    if suffix == ".zip":
        try:
            zip_data = resolved_path.read_bytes()
            uploaded, errors = core.extract_zip_cad_files(zip_data)
        except core.ZipSlipError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (core.ZipSizeLimitError, core.ZipFileLimitError) as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to process ZIP archive: {exc}"
            ) from exc

        return {
            "files": [
                {"file_id": fid, "filename": name} for fid, name in uploaded
            ],
            "errors": errors,
        }

    if suffix in core.CAD_ALLOWED_EXTENSIONS:
        try:
            fake = core._BytesUploadFile(resolved_path.read_bytes(), resolved_path.name)
            file_id, dest, existed = storage.upload_CAD_file_persistent(fake)
        except core.UploadSizeLimitError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "files": [
                {
                    "file_id": file_id,
                    "filename": resolved_path.name,
                    "already_existed": existed,
                }
            ],
            "errors": [],
        }

    allowed = ", ".join(sorted(core.CAD_ALLOWED_EXTENSIONS | {".zip"}))
    raise HTTPException(
        status_code=422,
        detail=(
            f"Unsupported file type '{suffix}'. "
            f"Allowed extensions: {allowed}"
        ),
    )


@router.get("/")
def list_files():
    """List all CAD files currently stored on the server."""
    return {"files": core.list_persistent_CAD_files()}


@router.get("/{file_id}/download")
def download_file(file_id: str):
    """Download a stored CAD file by its file_id.

    Returns the original CAD file bytes so a client can render the model
    locally (for example in a WebViewer). This server does not display CAD;
    the gallery hands the file_id here and rendering is the client's
    responsibility.
    """
    try:
        path = storage.find_persistent_CAD_file(file_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Files are stored as "<sha256>_<original name>"; hand back the original name.
    parts = path.name.split("_", 1)
    download_name = (
        parts[1] if len(parts) == 2 and len(parts[0]) == 64 else path.name
    )
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=download_name,
    )


@router.post("/{file_id}/scs")
def export_file_scs(file_id: str):
    """Convert a stored CAD file to an SCS stream cache for viewing.

    Uploaded files are not part of any index and have no pre-authored SCS, so
    this produces one on demand from the persistent CAD file and returns its URL
    under ``/out/``. The web client uses this to show an uploaded query file in
    the 3D viewer; SCS authoring is done entirely by the HOOPS AI SDK.
    """
    try:
        scs_name = core.export_scs_for_part(file_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"file_id": file_id, "scs_url": f"/out/{scs_name}"}


@router.delete("/{file_id}")
def delete_file(file_id: str):
    """Delete an uploaded CAD file by its file_id."""
    try:
        deleted = core.delete_persistent_CAD_file(file_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
        return {"deleted": file_id}
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
