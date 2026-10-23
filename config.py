"""Server configuration: the data root, its derived paths, and .env helpers.

This module owns the whole path and environment surface in one place so the
data root is resolved exactly once (at import time) and every consumer sees the
same derived directories.

Reference these names as module attributes -- ``config.INDEXES_DIR``,
``config.read_env_file()``, and so on. Do NOT write ``from config import
INDEXES_DIR``. A ``from`` import binds the name at import time, so a test (or a
tool) that swaps ``config.INDEXES_DIR`` to redirect the data root would never
reach a module that imported the old object. Keeping every read as a
``config.<name>`` attribute lookup makes those swaps take effect everywhere.
See "The re-export convention" in docs/architecture.md for the failure mode
this avoids.

The only names re-exported elsewhere are the two exception classes
(``EnvConfigError`` / ``PathConfigError``): they are never swapped, so ``core``
re-exports them for backwards compatibility with existing ``except core.*``
handlers.

Depends only on the standard library; it must never import ``core`` (the
dependency runs one way, from ``core`` down to ``config``).
"""

import logging
import os
import pathlib
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable reference (single catalogue for the whole server)
# ---------------------------------------------------------------------------
#
# This is documentation, not configuration. Every knob the server reads is
# listed here so a reader has one place to look, but the DEFAULT VALUES ARE NOT
# repeated in this block: each row names the module that owns the knob, and the
# default lives there (as a named ``_*_DEFAULT`` constant or an inline literal
# next to the ``_env_int(...)`` call). Keeping the values in exactly one place
# means this catalogue can never drift out of sync with what the code actually
# does.
#
# Groups:
#   A = a user decides this to get started (README + .env.example document it).
#   B = an operator tunes this for the machine or the data size (README's
#       "Memory sizing" and "Startup warmup" sections document it; also in
#       .env.example, commented out).
#   C = internal knob, upper bound, TTL, diagnostic or managed value. Sensible
#       default; rarely changed. NOT in README or .env.example -- this block is
#       its documentation.
#
# --- Group A: user-facing, documented in README + .env.example --------------
#   HOOPS_AI_LICENSE                 licensing.py    HOOPS AI license key; read
#                                                    from .env only, never os.environ.
#   HOOPS_AI_SDK_DIR                 embeddings.py   Optional SDK install dir; lets
#                                                    a full SDK reuse its checkpoints.
#   HOOPS_AI_DATA_DIR                config.py       Data root (uploads/ indexes/ ...);
#                                                    defaults to <app>/data.
#   HOOPS_AI_EMBEDDINGS_MODEL_NAME   embeddings.py   Checkpoint *.ckpt file name to use.
#
# --- Group B: operator-tunable, documented in README + .env.example ---------
#   HOOPS_AI_INDEX_CACHE_MAX_MB      index_registry.py  Named-index write-store budget.
#   HOOPS_AI_EMBED_CACHE_MAX_MB      embeddings.py      Embedding cache budget (disk-backed).
#   HOOPS_AI_WARMUP_INDEXES          config.py          Index names to warm at startup.
#   HOOPS_AI_WARMUP_MAX              warmup.py          Cap on how many indexes are warmed.
#   HOOPS_AI_WARMUP_STAGES           config.py          Which warmup stages run (default: all).
#   HOOPS_AI_DEMO                    config.py          Show webinar-demo-only UI (off by default).
#
# --- Group C: internal; this block is the only documentation ----------------
#   HOOPS_AI_WARMUP_CLUSTERING       config.py       Clusterings the map stages warm.
#                                                    Set it to drop Ward from a large
#                                                    index's startup, or to fix which
#                                                    one the client's map opens on.
#   HOOPS_AI_WARMUP_WARD_MAX         config.py       One Ward cluster ceiling for every
#                                                    index, instead of the per-index
#                                                    value; set it only to pin the
#                                                    granularity fleet-wide.
#   HOOPS_AI_ASSEMBLY_SEARCH_JOBS    assembly.py     Assembly-scoring thread-pool size.
#   HOOPS_AI_ASSEMBLY_PREWARM        assembly.py     Rebuild assembly matcher after a job.
#   HOOPS_AI_SHAPE_MAP_PREWARM       shape_map.py    Recompute default Shape Map after a job.
#   HOOPS_AI_SHAPE_MAP_SCS_CACHE_MAX shape_map.py    Max cached Shape Map .scs files (LRU).
#   HOOPS_AI_SHAPE_MAP_KMEANS_MAX_POINTS_PER_CENTROID shape_map.py  FAISS k-means training-set
#                                                    subsample size per centroid.
#   HOOPS_AI_SHAPE_MAP_KMEANS_NREDO  shape_map.py    FAISS k-means restarts from new seeds.
#   HOOPS_AI_SHAPE_MAP_KMEANS_SPHERICAL shape_map.py Normalise k-means centroids each iteration.
#   HOOPS_AI_SHAPE_MAP_KMEANS_INIT   shape_map.py    FAISS k-means initialisation:
#                                                    random (default) | kmeans++ | afkmc2.
#   HOOPS_AI_SHAPE_MAP_UMAP_N_NEIGHBORS shape_map.py UMAP neighbourhood size (default 15).
#   HOOPS_AI_SHAPE_MAP_UMAP_MIN_DIST shape_map.py    UMAP minimum point separation (0.1).
#   HOOPS_AI_SHAPE_MAP_UMAP_SEED     shape_map.py    UMAP random_state (42; fixing it makes
#                                                    umap-learn single-threaded).
#   HOOPS_AI_SHAPE_MAP_WARD_MAX_POINTS shape_map.py  Point cap for Ward clustering
#                                                    (default 20000; it builds an
#                                                    N(N-1)/2 distance matrix).
#   HOOPS_AI_LOG_LEVEL               main.py         Root log level applied at startup.
#   HOOPS_AI_MAX_UPLOAD_BYTES        storage.py      Per-file upload cap (413 above it).
#   HOOPS_AI_EXISTS_MAX_IDS          storage.py      Max ids per /files/exists (422 above it).
#   HOOPS_AI_ZIP_MAX_TOTAL_BYTES     storage.py      Uncompressed-size cap for a ZIP.
#   HOOPS_AI_ZIP_MAX_FILES           storage.py      CAD-file-count cap for a ZIP.
#   HOOPS_AI_CAD_SHARED_DIR          runtime.py      CAD store location (default uploads/).
#   HOOPS_AI_OUT_TTL_HOURS           runtime.py      TTL sweep age for out/ (0 disables).
#   HOOPS_AI_EMBEDDINGS_CACHE_TTL_DAYS runtime.py    TTL sweep age for embeddings_cache/.
#   HOOPS_AI_JOB_MAX_CONCURRENCY     jobs.py         Concurrent registration jobs.
#   HOOPS_AI_JOB_TTL_DAYS            jobs.py         Retention of finished job records.
#   HOOPS_AI_JOB_MAX_RECORDS         jobs.py         Hard cap on retained job records.
#   HOOPS_AI_MAX_TAGS_PER_PART       tags.py         Max index tags one part may carry.
#   HOOPS_AI_MAX_CONTEXT_KEYS        context_store.py Max distinct context keys per part.
#   HOOPS_AI_MAX_CONTEXT_PREDICT_KEYS context.py     Max keys one predict-context request scores.
#   HOOPS_AI_MAX_WORKERS             index_registry.py  Cap on the auto worker count.
#   HOOPS_AI_MODEL_FOOTPRINT_MB      index_registry.py  Assumed per-worker RSS for RAM sizing.
#   HOOPS_AI_MIN_FILES_PARALLEL      index_registry.py  Force one worker below this file count.
#   HOOPS_AI_EMBED_TIME_LIMIT        index_registry.py  Per-file embedding budget (s); 0 = SDK's.
#   HOOPS_AI_INTRAOP_THREADS         embeddings.py   BLAS/OpenMP threads per embedding worker.
#   HOOPS_AI_ALLOW_SERVER_PATHS      storage.py      Allow jobs to read server_paths (403 gate).
#   HOOPS_AI_SEARCH_CACHE            search.py       In-process part-search cache (0 disables).
#   HOOPS_AI_SEARCH_CACHE_MAX        search.py       Max cached search responses (LRU).
#   HOOPS_SEARCH_TIMING              config.py       Log per-phase search timing (diagnostic).
#
# Managed by the server (set, not read as config): OMP_NUM_THREADS,
# MKL_NUM_THREADS, OPENBLAS_NUM_THREADS -- embeddings.py sets these on each
# spawned worker to enforce HOOPS_AI_INTRAOP_THREADS.
#
# Build/seed tooling only (never read by the running server): HOOPS_AI_HOME and
# HOOPS_AI_LICENSE_BUILD (tools/ build and seeding scripts), HOOPS_AI_VENV
# (start_server launcher).
# ---------------------------------------------------------------------------

APP_ROOT = pathlib.Path(__file__).resolve().parent
ENV_FILE_PATH = pathlib.Path(__file__).with_name(".env")


def _read_env_file_value(name: str) -> Optional[str]:
    """Read a single key directly from the .env file.

    Used during module import to resolve the data root before the normal
    ``load_env_file`` pass has run. Intentionally tiny; the full parser lives in
    ``read_env_file``. Reads with utf-8-sig so a BOM-prefixed .env still parses.
    """
    try:
        text = ENV_FILE_PATH.read_text(encoding="utf-8-sig")
    except OSError:
        return None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            resolved = value.strip().strip('"').strip("'")
            return resolved or None
    return None


def _dir_is_writable(path: pathlib.Path) -> bool:
    """Return True if *path* exists (or can be created) and accepts a write."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".write_probe_{os.getpid()}"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def _resolve_data_root() -> tuple[pathlib.Path, str]:
    """Resolve the data root and report where the choice came from.

    Order (first match wins):
      1. os.environ['HOOPS_AI_DATA_DIR']
      2. .env HOOPS_AI_DATA_DIR                (the installer's normal knob)
      3. APP_ROOT/data                         (default: uploads/, indexes/, ...
                                               live under <app>/data/)

    Kept deliberately flat -- just the two configuration sources plus a single
    derived default -- so there is no "magic" auto-detection to reason about.
    The default puts the runtime data under APP_ROOT/data so the source tree and
    the runtime directories (uploads/, indexes/, out/, ...) stay separate: a
    fresh checkout works with no configuration, and everything the server writes
    is confined to one data/ subtree.

    Never moves or deletes anything. Import must never fail, so this never
    raises; a startup writability check (ensure_data_root_writable) turns a
    read-only data root into a clear error at server start -- e.g. when the app
    is installed under Program Files, set HOOPS_AI_DATA_DIR to a writable path.
    """
    env_value = os.environ.get("HOOPS_AI_DATA_DIR")
    if env_value:
        return pathlib.Path(env_value).expanduser(), "os.environ[HOOPS_AI_DATA_DIR]"

    dotenv_value = _read_env_file_value("HOOPS_AI_DATA_DIR")
    if dotenv_value:
        return pathlib.Path(dotenv_value).expanduser(), ".env HOOPS_AI_DATA_DIR"

    return APP_ROOT / "data", "APP_ROOT/data (default)"


DATA_ROOT, _DATA_ROOT_SOURCE = _resolve_data_root()
CAD_UPLOAD_DIR = DATA_ROOT.joinpath("uploads")
CAD_VIEWER_OUTPUT_DIR = DATA_ROOT.joinpath("out")
EMBEDDINGS_CACHE_DIR = DATA_ROOT.joinpath("embeddings_cache")
INDEXES_DIR = DATA_ROOT.joinpath("indexes")
JOBS_DIR = DATA_ROOT.joinpath("jobs")
# Operator-supplied / bundled embedding checkpoints (see resolve_bundled_checkpoint).
MODELS_DIR = DATA_ROOT.joinpath("models")
# Point-cloud .scs cache for the index-based Shape Map (a DATA_ROOT-derived path;
# the SCS export code itself stays in core.py).
SCS_MAP_DIR = DATA_ROOT.joinpath("scs_map")
# The tag vocabulary: the one tag sidecar that is NOT per index. The names a
# user has decided on are shared by the whole application, so the file sits at
# the data root rather than under indexes/<name>/ (see the tag-vocabulary
# section of tags.py for why, and for what it costs).
TAG_VOCABULARY_FILE = DATA_ROOT.joinpath("tag_vocabulary.json")


def read_env_file(path: pathlib.Path = ENV_FILE_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            values[key] = value
    return values


def load_env_file(path: pathlib.Path = ENV_FILE_PATH) -> None:
    # Uses setdefault so a value already present in os.environ (from Docker,
    # systemd, the shell, etc.) is never overridden by the .env file. Together
    # with the os.environ.get() reads elsewhere this yields the standard
    # precedence: os.environ > .env > derived default.
    for key, value in read_env_file(path).items():
        os.environ.setdefault(key, value)


class EnvConfigError(Exception):
    """Raised when a required environment variable is missing or empty."""


class PathConfigError(Exception):
    """Raised when a path derived from an environment variable does not exist."""


def get_required_file_env(name: str) -> str:
    # Reads from the .env file ONLY, never from os.environ. This asymmetry is
    # intentional and applies solely to the HOOPS AI license variable: reading
    # the key from the process environment would expose it via /proc/<pid>/environ
    # (or the Windows process environment) to any local user. Every other setting
    # follows the normal os.environ > .env precedence via get_required_env().
    value = read_env_file().get(name)
    if not value:
        msg = f"[CONFIG] Required environment variable '{name}' is not set in {ENV_FILE_PATH.name}."
        logger.error(msg)
        print(msg, flush=True)
        raise EnvConfigError(msg)
    return value


def ensure_data_root_writable() -> None:
    """Fail fast at startup if the resolved data root cannot be written.

    Running this at server start (rather than mid-search) keeps operations
    simple: a read-only data root -- e.g. the app installed under Program Files
    with no HOOPS_AI_DATA_DIR set -- stops the server immediately with a clear
    message instead of failing on the first upload.
    """
    if not _dir_is_writable(DATA_ROOT):
        msg = (
            f"[CONFIG] Data root is not writable: {DATA_ROOT} "
            f"(source: {_DATA_ROOT_SOURCE}). Set HOOPS_AI_DATA_DIR to a writable "
            f"location."
        )
        logger.error(msg)
        print(msg, flush=True)
        raise EnvConfigError(msg)


def _env_int(name: str, default: int, minimum: int = 1, maximum: Optional[int] = None) -> int:
    """Read a positive integer setting from the environment.

    Read on every call so limits can be tuned without a restart. A missing,
    malformed or out-of-range value falls back to *default* with a warning
    rather than failing the request: an operator typo must not take the server
    down, but it must not be silent either.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d.", name, raw, default)
        return default
    if value < minimum or (maximum is not None and value > maximum):
        logger.warning(
            "%s=%d is out of range [%s, %s]; using default %d.",
            name, value, minimum, maximum if maximum is not None else "inf", default,
        )
        return default
    return value


def _search_timing_enabled() -> bool:
    """True when ``HOOPS_SEARCH_TIMING=1``.

    Read on every call (like the other env helpers) so timing can be turned on
    or off without restarting the server.
    """
    return os.environ.get("HOOPS_SEARCH_TIMING", "").strip() == "1"


def demo_enabled() -> bool:
    """Whether the webinar-demo-only parts of the UI are shown.

    From HOOPS_AI_DEMO, **default off**. The Estimate panel predicts production
    records from the Context Layer, which is only meaningful once an index has
    real records attached; the demo seeds fabricated ones (see
    tools/seed_context_store.py), so on a fresh clone the feature would look
    like it works while resting on invented data. Gating it means a new user
    never meets it by accident, and whoever wants it says so explicitly.

    Only the CLIENT reads this (through the /client-config endpoint): the
    context API stays enabled either way, because it is genuinely useful to
    anyone who attaches their own records.

    Read on every call, and accepting the same truthy spellings as the other
    boolean settings (HOOPS_AI_ASSEMBLY_PREWARM, HOOPS_AI_ALLOW_SERVER_PATHS),
    so there is one convention to remember. Unlike those two, the default is
    off, so anything unrecognised reads as off.
    """
    raw = os.environ.get("HOOPS_AI_DEMO") or read_env_file().get("HOOPS_AI_DEMO", "")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def warmup_indexes_setting() -> list[str]:
    """Index names to warm at startup, from HOOPS_AI_WARMUP_INDEXES (comma list).

    Empty/unset -> no warmup (the default for a redistributed product). Read on
    every call (like the other env helpers) so it can change without a restart.
    """
    raw = os.getenv("HOOPS_AI_WARMUP_INDEXES", "").strip()
    if not raw:
        return []
    return [n.strip() for n in raw.split(",") if n.strip()]


# The stages _warmup_one() knows how to run, in the order it runs them.
_WARMUP_STAGES = ("assembly", "map", "scs", "searcher")

# The clusterings stage 2/2b can warm, and the default list. Both of them:
# each is a separate cache key, so a user who switches to the one that was not
# warmed waits for a full build (about 15 s on an 8,000-part index).
_WARMUP_CLUSTERINGS = ("kmeans", "ward")
_WARMUP_CLUSTERINGS_DEFAULT = ("kmeans", "ward")


def _warmup_choice_setting(var: str, allowed: tuple[str, ...], default: tuple[str, ...]) -> list[str]:
    """Parse a comma-separated list of *allowed* keywords from *var*.

    Unset or empty -> *default*, so adding one of these variables to an existing
    deployment is the only way to change what warmup does. Unknown words are
    dropped rather than failing startup; a list that names nothing valid falls
    back to the default, because "warm nothing" already has a spelling
    (HOOPS_AI_WARMUP_INDEXES unset).
    """
    raw = os.getenv(var, "").strip()
    if not raw:
        return list(default)
    picked = [w.strip().lower() for w in raw.split(",") if w.strip()]
    # The caller's order is kept, and the first entry is meaningful: it is the
    # one /client-config names as the clustering the Shape Map should open with.
    chosen: list[str] = []
    for word in picked:
        if word in allowed and word not in chosen:
            chosen.append(word)
    return chosen or list(default)


def warmup_stages_setting() -> list[str]:
    """Which warmup stages to run, from HOOPS_AI_WARMUP_STAGES (comma list).

    One or more of ``assembly``, ``map``, ``scs``, ``searcher``; unset means all
    of them, which is what warmup has always done. The assembly matcher takes
    about 39 s on an 8,000-part index, so a deployment that does not use
    assembly search can leave it out and start that much sooner.
    """
    return _warmup_choice_setting("HOOPS_AI_WARMUP_STAGES", _WARMUP_STAGES, _WARMUP_STAGES)


def warmup_clustering_setting() -> list[str]:
    """Which clusterings the map stages warm, from HOOPS_AI_WARMUP_CLUSTERING.

    A comma-separated list naming ``kmeans``, ``ward``, or both -- and both is
    the default; an empty or unrecognised value falls back to it. Each
    clustering is a separate cache key and a separate build, so a user who
    switches to the
    one that was not warmed pays for the whole thing: measured at about 15 s on
    an 8,000-part index (11.8 s for the map, 1.6 s for the folded tree, 1.0 s
    for the point cloud). Warming the pair costs about 5 s more at startup now
    that Ward's linkage is built from a blocked matrix product, which is a
    trade worth making once rather than making the first user of each
    clustering make it.

    The ORDER matters. /client-config reports this list, and the web client
    opens its Shape Map on a clustering from it -- its remembered choice when
    that is one of them, otherwise the first entry, which is what
    ``default_clustering`` names. So the first entry is what a fresh browser
    opens on.

    A deployment that wants the shortest possible startup, or that never uses
    Ward, says so explicitly: ``HOOPS_AI_WARMUP_CLUSTERING=kmeans``.
    """
    return _warmup_choice_setting(
        "HOOPS_AI_WARMUP_CLUSTERING", _WARMUP_CLUSTERINGS, _WARMUP_CLUSTERINGS_DEFAULT
    )


# The bounds Ward's cluster ceiling must stay within: /map's Query(ge=2,
# le=1024). There is no default -- unset means the per-index automatic count
# (shape_map.shape_map_auto_cluster_count) is used instead.
_WARMUP_WARD_MAX_MIN = 2
_WARMUP_WARD_MAX_MAX = 1024


def warmup_ward_max_setting() -> Optional[int]:
    """Ward's cluster ceiling, from HOOPS_AI_WARMUP_WARD_MAX, or None if unset.

    None means "no fixed ceiling": the map stages and the web client both fall
    back to the automatic count derived from the index's own part count
    (shape_map.shape_map_auto_cluster_count), so a small index is not cut into
    256 families and a large one is not left coarse. Setting the variable
    overrides that for EVERY index, which is what an operator wants when their
    deployment has one index and one answer.

    The two uses must agree: the map stages warm Ward at this ceiling, and
    /client-config names it as the Max clusters the Shape Map opens with. A map
    cut to a different ceiling is a different cache key and a different set of
    clusters, so opening the UI on a value that was not warmed leaves the first
    user paying the full build. Outside 2..1024 -- the range /map accepts -- the
    value is treated as unset rather than clamped: a ceiling the server would
    reject is not a ceiling, and the automatic count is a better answer than a
    silently different one.

    Leave it unset unless the ceiling has to be the same everywhere: a user who
    wants a different granularity for one session changes Max clusters in the
    Shape Map, and that request is served whether or not it was warmed.
    """
    raw = os.getenv("HOOPS_AI_WARMUP_WARD_MAX", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid HOOPS_AI_WARMUP_WARD_MAX=%r; using the per-index "
            "automatic count instead.", raw,
        )
        return None
    if value < _WARMUP_WARD_MAX_MIN or value > _WARMUP_WARD_MAX_MAX:
        logger.warning(
            "HOOPS_AI_WARMUP_WARD_MAX=%d is out of range [%d, %d]; using the "
            "per-index automatic count instead.",
            value, _WARMUP_WARD_MAX_MIN, _WARMUP_WARD_MAX_MAX,
        )
        return None
    return value
