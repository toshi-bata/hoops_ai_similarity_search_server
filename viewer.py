"""SCS stream-cache export for the HOOPS Web Viewer.

Owns :func:`export_scs_for_part`, the thin wrapper that converts a persistent
CAD file into a self-contained ``.scs`` stream cache served under ``/out/``.
It is a leaf helper -- it does not touch any per-session cache and is not a
cross-domain orchestrator.

Dependency direction is one-way: ``core`` imports this module; this module
imports only :mod:`config` and :mod:`storage` (plus the standard library).
It never imports ``core``.

The HOOPS SDK (``HOOPSLoader`` / ``HOOPSTools``) is imported lazily inside the
function so that importing this module does not pull the SDK into memory --
mirroring :mod:`storage`, which has no module-level SDK import.

``export_scs_for_part`` is a stable (C) name: ``core`` re-exports it by
identity and routers reach it as ``viewer.export_scs_for_part``.
"""


import pathlib
import uuid

import config
import storage


# ---------------------------------------------------------------------------
# SCS stream-cache export (shared by /compare, /query and per-part assets)
# ---------------------------------------------------------------------------


def export_scs_for_part(file_id: str) -> str:
    """Convert a persistent CAD file to an SCS stream cache and return its filename.

    A thin wrapper around the SCS stream-cache conversion -- it does
    NOT touch any per-session cache.  Returns just the SCS filename (served
    under ``/out/``), not a full path or URL.
    """
    from hoops_ai.cadaccess import HOOPSLoader, HOOPSTools

    cad_file_path = storage.find_persistent_CAD_file(file_id)
    config.CAD_VIEWER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    unique_id = uuid.uuid4().hex[:12]
    scs_name = f"{unique_id}_{cad_file_path.stem}.scs"
    scs_path = config.CAD_VIEWER_OUTPUT_DIR / scs_name

    cad_loader = HOOPSLoader()
    model = cad_loader.create_from_file(str(cad_file_path))

    tools = HOOPSTools()
    _png_path, scs_path = tools.exportStreamCache(
        model,
        filename=str(scs_path),
        is_white_background=True,
        overwrite=True,
    )
    return pathlib.Path(scs_path).name