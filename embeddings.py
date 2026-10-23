"""Shape embedding: the encoders, the two-tier embedding cache, and model
checkpoint resolution.

This module owns the HOOPS embedder singletons, the in-process LRU embedding
cache, the SDK/checkpoint path resolution used to load a model, and the
per-worker intra-op thread cap. It is a leaf: it depends only on the standard
library, ``config``, ``storage`` (to locate a stored CAD file) and
``diagnostics`` (the phase timer). It must never import ``core`` -- the
dependency runs one way, from ``core`` down to ``embeddings``.

Reference convention (see "The re-export convention" in
docs/architecture.md). The names a test swaps must be reached as module
attributes -- ``embeddings.get_embedder``,
``embeddings.compute_embedding``, ``embeddings.compare_embeddings``,
``embeddings._embed_cache_max_bytes``, ``embeddings._embedding_memory_cache``
and ``embeddings._embedding_cache_bytes``. Do NOT write ``from embeddings
import <name>``: a ``from`` import binds the name at import time, so a test
patching ``embeddings.<name>`` would never reach a module that imported the old
object. These six are therefore NOT re-exported by ``core``.

The encoder singleton ``_embedder`` is rebound in place via ``global``. A name
that is rebound must not be re-exported, or ``core.<name>`` would keep pointing
at the stale object; ``core`` reaches it only through the functions moved here,
so no re-export is needed.

Everything else -- the cache accessors and the whole SDK/checkpoint group -- is
stable and never rebound, so ``core`` re-exports those names for its own callers
and for existing ``core.<name>`` references in routers and tests.

The server loads exactly one checkpoint per process. Which file is loaded is
resolved from ``HOOPS_AI_EMBEDDINGS_MODEL_NAME`` (a checkpoint file name) or,
when that is unset, from the ``models/`` directories. A checkpoint is not
labelled ``legacy`` or ``signal``: successive fine-tunes are a continuum, so a
checkpoint is identified by the SHA-256 of its bytes (see ``ckpt_sha256``),
recorded per record when an index is written.

Cache invariant: ``_embedding_memory_cache`` (an ``OrderedDict`` used as an
LRU) and its running size ``_embedding_cache_bytes`` are one unit guarded by
the single ``_embedding_cache_mutex``. Every read, insert and eviction takes
that mutex, and the byte counter is adjusted in the same critical section as
the dict it describes, so the two never drift. Because ``_embedding_cache_bytes``
is an ``int`` it can only be updated by rebinding it under ``global`` -- keep
that ``global`` declaration and the mutex scope exactly as they are.
"""

import hashlib
import logging
import os
import pathlib
import sys
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Optional

import config
import diagnostics
import storage
from config import EnvConfigError, PathConfigError

logger = logging.getLogger(__name__)


_embedder = None
# cache_key -> embedding entry, most-recently-used last. Backed by a disk cache
# (EMBEDDINGS_CACHE_DIR), so an evicted entry is reloaded, not recomputed.
# Bounded by MEMORY via HOOPS_AI_EMBED_CACHE_MAX_MB, tracked in
# _embedding_cache_bytes so eviction does not rescan every entry.
_EMBED_CACHE_MAX_MB_DEFAULT = 256
_embedding_memory_cache: "OrderedDict[str, dict]" = OrderedDict()
_embedding_cache_bytes = 0
# Guards _embedding_memory_cache and _embedding_cache_bytes. Leaf lock.
_embedding_cache_mutex = threading.Lock()

# The intra-op (BLAS/OpenMP) thread cap for embedding workers.
#
# ``embed_shape_batch`` is CPU-bound and spawns one worker process per
# num_workers. Each worker, left to the library default, starts a full
# complement of intra-op threads (one per core) for numpy/torch/FAISS math. Past
# a few workers this makes "num_workers x cores" threads contend for the same
# physical cores; the context-switch overhead drags every worker down, so
# throughput plateaus then collapses (a 16-core host measured ~10.8 files/s at
# 12 workers falling to ~5.6 at 40). Pinning each worker to a single intra-op
# thread removes the collapse and raises the peak (~14 files/s at 16 workers,
# still ~12 at 40): "many workers x one thread each" beats "few workers x many
# threads" at equal total threads. The optimum is num_workers ~ physical cores
# with intra-op = 1.
#
# These are the variable names each math library reads at import time. Setting
# them just before the batch and clearing them just after confines the cap to
# the freshly spawned workers -- the already-imported libraries in this parent
# server process (FAISS search, shape-map projection) keep their own threading.
_INTRAOP_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_INTRAOP_THREADS_DEFAULT = 1


def intraop_threads() -> int:
    """Per-worker BLAS/OpenMP thread cap for embedding, from
    ``HOOPS_AI_INTRAOP_THREADS``.

    Defaults to ``1`` (see :data:`_INTRAOP_ENV_VARS`): the fastest embedding
    configuration is many workers with one intra-op thread each. ``0`` disables
    the cap -- every worker is left at the library default, i.e. the behaviour
    before this setting existed -- for A/B comparison or unusual corpora.
    """
    return config._env_int("HOOPS_AI_INTRAOP_THREADS", _INTRAOP_THREADS_DEFAULT, minimum=0)


@contextmanager
def intraop_thread_limit(threads: int):
    """Temporarily pin the BLAS/OpenMP intra-op thread count via the environment.

    Sets every variable in :data:`_INTRAOP_ENV_VARS` to *threads* on entry and
    restores the prior environment (unsetting what had been unset) on exit, even
    if the body raises. A *threads* of ``0`` or less is a no-op, so callers can
    wrap the embedding call unconditionally.

    Scope is deliberate: this only affects processes that import their math
    libraries *after* entry -- the workers ``embed_shape_batch`` spawns -- not
    the code already running in this process.
    """
    if not threads or threads <= 0:
        yield
        return
    value = str(int(threads))
    saved: dict[str, Optional[str]] = {}
    try:
        for name in _INTRAOP_ENV_VARS:
            saved[name] = os.environ.get(name)
            os.environ[name] = value
        yield
    finally:
        for name, prev in saved.items():
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev


def get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        msg = f"[CONFIG] Required environment variable '{name}' is not set. Set it in {config.ENV_FILE_PATH.name} or the environment."
        logger.error(msg)
        print(msg, flush=True)
        raise EnvConfigError(msg)
    return value


def require_path(path: pathlib.Path, *, env_name: str = "", label: str = "") -> pathlib.Path:
    """Return *path* unchanged, or raise PathConfigError if it does not exist.

    Logs the missing path to the console before raising so that operators can
    immediately see which file or directory is absent.
    """
    if not path.exists():
        origin = f" (from {env_name})" if env_name else (f" ({label})" if label else "")
        msg = f"[CONFIG] Path not found{origin}: {path}"
        logger.error(msg)
        print(msg, flush=True)
        raise PathConfigError(msg)
    return path


def get_sdk_dir() -> pathlib.Path:
    """Return the HOOPS AI SDK install directory (``HOOPS_AI_SDK_DIR``).

    This directory must contain the ``packages/`` subdirectory; that is the only
    part of an SDK install this server reads (see ``get_packages_dir``).

    ``HOOPS_AI_SDK_DIR`` is an optional developer convenience: it lets a machine
    with a full HOOPS AI install reuse that install's bundled checkpoints. A redistributed package has no SDK install, so this is unset
    there. Callers that depend on it must degrade gracefully when it is unset
    rather than treating the absence as a hard error. We deliberately never
    synthesise this path from ``hoops_ai.__file__`` -- nothing lives under that
    path, so a synthesised value would turn a clean "not configured" into a
    confusing ``FileNotFoundError`` at the point of use.
    """
    return require_path(
        pathlib.Path(get_required_env("HOOPS_AI_SDK_DIR")),
        env_name="HOOPS_AI_SDK_DIR",
    )


def sdk_dir_configured() -> bool:
    """Return True when ``HOOPS_AI_SDK_DIR`` is set to an existing directory.

    Lets default-index consumers check availability without raising, so a
    package with no SDK install simply omits the built-in default index.
    """
    value = os.environ.get("HOOPS_AI_SDK_DIR")
    if not value:
        return False
    return pathlib.Path(value).is_dir()


def get_packages_dir() -> pathlib.Path:
    """Return ``<HOOPS_AI_SDK_DIR>/packages``."""
    return require_path(
        get_sdk_dir() / "packages",
        env_name="HOOPS_AI_SDK_DIR",
        label="packages directory",
    )


class ModelResolutionError(RuntimeError):
    """Raised when no usable embedding checkpoint can be resolved, or when a
    models/ directory is ambiguous (two or more ``*.ckpt`` files)."""


def _ckpt_candidates(models_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return the sorted ``*.ckpt`` files directly inside *models_dir* (or [])."""
    if not models_dir.is_dir():
        return []
    return sorted(p for p in models_dir.glob("*.ckpt") if p.is_file())


def resolve_bundled_checkpoint() -> pathlib.Path:
    """Resolve the embedding checkpoint from ``models/`` directories.

    Used when ``HOOPS_AI_EMBEDDINGS_MODEL_NAME`` is unset (the normal case for a
    redistributed package). Search order, per directory:

      1. ``<DATA_ROOT>/models/``  operator-supplied (e.g. a fine-tuned model)
      2. ``<APP_ROOT>/models/``   the default model bundled in the package

    Counting is per directory, never a total: within a directory, exactly one
    ``*.ckpt`` is used; two or more raise an error listing the candidates (the
    operator must keep one, or name it with ``HOOPS_AI_EMBEDDINGS_MODEL_NAME``);
    zero falls through to the next directory. When a data-root model is used
    while a bundled default also exists, a shadowing notice is logged, because
    silently replacing the bundled model is the most dangerous outcome. The
    resolved absolute path and size are always logged at INFO so an operator can
    confirm which model is in use.
    """
    data_models = config.MODELS_DIR
    app_models = config.APP_ROOT.joinpath("models")

    for models_dir, label in ((data_models, "data root"), (app_models, "app")):
        candidates = _ckpt_candidates(models_dir)
        if len(candidates) == 1:
            resolved = candidates[0]
            if label == "data root" and _ckpt_candidates(app_models):
                logger.info(
                    "[MODEL] Using operator model %s from the data root; it "
                    "shadows the bundled default model(s) in %s.",
                    resolved, app_models,
                )
            logger.info(
                "[MODEL] Resolved embedding checkpoint: %s (%d bytes, source: %s).",
                resolved, resolved.stat().st_size, label,
            )
            return resolved
        if len(candidates) >= 2:
            names = ", ".join(p.name for p in candidates)
            msg = (
                f"[MODEL] Ambiguous models directory {models_dir}: found multiple "
                f"checkpoints ({names}). Keep exactly one *.ckpt there, or set "
                f"HOOPS_AI_EMBEDDINGS_MODEL_NAME to the file name to use."
            )
            logger.error(msg)
            print(msg, flush=True)
            raise ModelResolutionError(msg)

    msg = (
        f"[MODEL] No embedding checkpoint found. Place one *.ckpt in "
        f"{data_models} or {app_models}, or set "
        f"HOOPS_AI_EMBEDDINGS_MODEL_NAME to the file name to use."
    )
    logger.error(msg)
    print(msg, flush=True)
    raise ModelResolutionError(msg)


_ckpt_sha256_cache: dict[str, str] = {}


def resolve_checkpoint_path() -> pathlib.Path:
    """Resolve the single embedding checkpoint file without loading it.

    When ``HOOPS_AI_EMBEDDINGS_MODEL_NAME`` is set it names a checkpoint file;
    the file is looked up in three directories, first match wins:

      1. ``<DATA_ROOT>/models/<name>``
      2. ``<APP_ROOT>/models/<name>``
      3. ``<HOOPS_AI_SDK_DIR>/packages/trained_ml_models/<name>``

    Other ``*.ckpt`` files sitting next to it are ignored: naming the file is
    explicit, so a directory holding several tutorial checkpoints is fine. If
    the named file is in none of the three, an error lists all three locations.

    When the variable is unset the checkpoint is resolved from the ``models/``
    directories (redistributed package, see ``resolve_bundled_checkpoint``).
    """
    config.load_env_file()
    ckpt_name = os.environ.get("HOOPS_AI_EMBEDDINGS_MODEL_NAME")
    if not ckpt_name:
        return resolve_bundled_checkpoint()

    search_dirs = [
        config.MODELS_DIR,
        config.APP_ROOT.joinpath("models"),
    ]
    if sdk_dir_configured():
        search_dirs.append(get_packages_dir().joinpath("trained_ml_models"))
    for directory in search_dirs:
        candidate = directory.joinpath(ckpt_name)
        if candidate.is_file():
            logger.info(
                "[MODEL] Resolved embedding checkpoint by name: %s (%d bytes).",
                candidate, candidate.stat().st_size,
            )
            return candidate

    searched = "; ".join(str(d.joinpath(ckpt_name)) for d in search_dirs)
    msg = (
        f"[MODEL] Checkpoint '{ckpt_name}' (HOOPS_AI_EMBEDDINGS_MODEL_NAME) was "
        f"not found. Looked in: {searched}."
    )
    logger.error(msg)
    print(msg, flush=True)
    raise ModelResolutionError(msg)


def _hash_file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ckpt_sha256(path: pathlib.Path) -> str:
    """Return the SHA-256 of the checkpoint at *path*, cached per resolved path.

    Hashing a multi-GB checkpoint takes a few seconds, so the digest is computed
    at most once per process per file. It is only ever computed while an index is
    written -- never on search and never at startup.
    """
    key = str(path)
    cached = _ckpt_sha256_cache.get(key)
    if cached is not None:
        return cached
    digest = _hash_file_sha256(path)
    _ckpt_sha256_cache[key] = digest
    return digest


def get_embedder():
    """Lazy-initialise HOOPSEmbeddings without loading a FAISS index.

    Loads the single checkpoint resolved by ``resolve_checkpoint_path`` and
    registers it as ``"hoops_embeddings_model"``.

    Idempotent: the model registration is guarded so it is never performed twice.
    """
    global _embedder

    if _embedder is not None:
        return _embedder

    from hoops_ai.ml.embeddings import HOOPSEmbeddings

    config.load_env_file()

    trained_model = resolve_checkpoint_path()

    try:
        HOOPSEmbeddings.register_model(
            model_name="hoops_embeddings_model",
            checkpoint_path=str(trained_model),
        )
    except Exception:
        pass  # already registered (e.g. by create_cad_searcher)

    _embedder = HOOPSEmbeddings(model="hoops_embeddings_model")
    return _embedder


def _l2_normalize(v):
    """L2-normalise a 1-D numpy float32 array. Returns a zero vector unchanged."""
    import numpy as np

    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v.astype(np.float32)
    return (v / norm).astype(np.float32)


def _embed_cache_max_bytes() -> int:
    """Memory budget for the in-process embedding cache (HOOPS_AI_EMBED_CACHE_MAX_MB)."""
    return config._env_int(
        "HOOPS_AI_EMBED_CACHE_MAX_MB", _EMBED_CACHE_MAX_MB_DEFAULT, minimum=1
    ) * 1024 * 1024


def _embed_entry_bytes(entry: dict) -> int:
    """Approximate resident size of one cached embedding entry."""
    v = entry.get("vector")
    return int(getattr(v, "nbytes", 0) or 0) + 512  # vector + small metadata


def _embedding_cache_get(cache_key: str) -> Optional[dict]:
    """Return a cached entry (marking it most-recently-used), or None."""
    with _embedding_cache_mutex:
        entry = _embedding_memory_cache.get(cache_key)
        if entry is not None:
            _embedding_memory_cache.move_to_end(cache_key)
        return entry


def _embedding_cache_put(cache_key: str, entry: dict) -> None:
    """Insert *entry* and evict least-recently-used entries past the memory cap."""
    global _embedding_cache_bytes
    with _embedding_cache_mutex:
        old = _embedding_memory_cache.pop(cache_key, None)
        if old is not None:
            _embedding_cache_bytes -= _embed_entry_bytes(old)
        _embedding_memory_cache[cache_key] = entry
        _embedding_cache_bytes += _embed_entry_bytes(entry)
        max_bytes = _embed_cache_max_bytes()
        while len(_embedding_memory_cache) > 1 and _embedding_cache_bytes > max_bytes:
            ev_key, ev_entry = _embedding_memory_cache.popitem(last=False)
            _embedding_cache_bytes -= _embed_entry_bytes(ev_entry)
            logger.info(
                "[CACHE] evicted embedding '%s' (~%d KB); embedding cache now %d entries / ~%d MB",
                ev_key, _embed_entry_bytes(ev_entry) // 1024,
                len(_embedding_memory_cache), _embedding_cache_bytes // (1024 * 1024),
            )


def clear_embedding_memory_cache() -> None:
    """Drop the in-process embedding cache (the disk cache is untouched)."""
    global _embedding_cache_bytes
    with _embedding_cache_mutex:
        _embedding_memory_cache.clear()
        _embedding_cache_bytes = 0


def compute_embedding(file_id: str) -> dict[str, Any]:
    """Compute (or retrieve from cache) the shape embedding for a CAD file.

    The result vector is a single L2-normalised float32 array representing the
    whole part.  For multi-body models the per-body vectors are individually
    L2-normalised, averaged, then re-normalised.

    Returns a dict with keys:
      ``file_id``, ``vector`` (np.ndarray), ``dim``, ``model_name``,
      ``num_bodies``, ``filename``, ``cached`` (bool).
    """
    import numpy as np

    embedder = get_embedder()
    model_name = (
        getattr(embedder, "model_name", None)
        or getattr(embedder, "model", None)
        or "hoops_embeddings_model"
    )

    # Build a filesystem-safe cache key that includes the model name.
    safe_model = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_name)
    cache_key = f"{safe_model}__{file_id}"

    # 1. Memory cache (fastest)
    entry = _embedding_cache_get(cache_key)
    if entry is not None:
        return {**entry, "cached": True}

    # 2. Disk cache (survives server restart)
    config.EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    disk_vec_path = config.EMBEDDINGS_CACHE_DIR / f"{cache_key}.npy"
    disk_meta_path = config.EMBEDDINGS_CACHE_DIR / f"{cache_key}.meta.npy"

    if disk_vec_path.exists() and disk_meta_path.exists():
        try:
            vector = np.load(str(disk_vec_path))
            meta = np.load(str(disk_meta_path), allow_pickle=True).item()
            entry: dict[str, Any] = {
                "file_id": file_id,
                "vector": vector,
                "dim": int(vector.shape[0]),
                "model_name": model_name,
                "num_bodies": int(meta.get("num_bodies", 1)),
                "filename": str(meta.get("filename", "")),
            }
            _embedding_cache_put(cache_key, entry)
            return {**entry, "cached": True}
        except Exception:
            pass  # corrupted cache -- fall through to recompute

    # 3. Compute via HOOPS embedder
    cad_path = storage.find_persistent_CAD_file(file_id)
    # The stored file name pattern is "{file_id}_{original_filename}".
    parts = cad_path.name.split("_", 1)
    filename = parts[1] if len(parts) == 2 and len(parts[0]) == 64 else cad_path.name

    _embed_t0 = time.perf_counter() if config._search_timing_enabled() else 0.0
    raw_embeddings = embedder.embed_shape(str(cad_path))
    if config._search_timing_enabled():
        logger.info(
            "[EMBED_TIMING] file_id=%s embed_shape = %.1f ms (SDK B-rep encode + inference)",
            file_id, (time.perf_counter() - _embed_t0) * 1000.0,
        )

    body_vectors: list = []
    for emb in raw_embeddings:
        v = getattr(emb, "values", None)
        if v is None:
            v = np.array(emb)
        v = np.array(v, dtype=np.float32).flatten()
        body_vectors.append(_l2_normalize(v))

    if not body_vectors:
        raise RuntimeError(f"embed_shape() returned no embeddings for file_id '{file_id}'.")

    if len(body_vectors) == 1:
        vector = body_vectors[0]
    else:
        stacked = np.stack(body_vectors, axis=0)  # (N, D)
        vector = _l2_normalize(stacked.mean(axis=0))

    num_bodies = len(body_vectors)

    entry = {
        "file_id": file_id,
        "vector": vector,
        "dim": int(vector.shape[0]),
        "model_name": model_name,
        "num_bodies": num_bodies,
        "filename": filename,
    }

    # Persist to disk (non-fatal on failure)
    try:
        np.save(str(disk_vec_path), vector)
        np.save(str(disk_meta_path), {"num_bodies": num_bodies, "filename": filename})
    except Exception:
        pass

    _embedding_cache_put(cache_key, entry)
    return {**entry, "cached": False}


def compare_embeddings(file_ids: list[str]) -> dict[str, Any]:
    """Compute an N x N cosine similarity matrix for the given file_ids.

    All embedding vectors are L2-normalised, so cosine similarity equals their
    dot product.  Diagonal entries are forced to exactly ``1.0``.

    Returns a dict with keys:
      ``count``, ``model_name``, ``files``, ``matrix`` (N x N list of lists),
      ``pairs`` (all i<j combos sorted by score descending).
    """
    import numpy as np

    timer = diagnostics._PhaseTimer() if config._search_timing_enabled() else None
    if timer:
        log_compute_environment("COMPARE_ENV")
    embeddings = []
    for fid in file_ids:
        e = compute_embedding(fid)
        if timer:
            timer.lap(f"compute_embedding[{fid[:12]} cached={e['cached']}]")
        embeddings.append(e)
    vectors = np.stack([e["vector"] for e in embeddings], axis=0)  # (N, D)

    # Cosine similarity via dot product (vectors are already L2-normalised)
    raw_matrix = (vectors @ vectors.T).tolist()
    n = len(embeddings)
    matrix = [[round(float(raw_matrix[i][j]), 6) for j in range(n)] for i in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0  # enforce exact diagonal

    files = [
        {
            "index": i,
            "file_id": e["file_id"],
            "filename": e["filename"],
            "num_bodies": e["num_bodies"],
        }
        for i, e in enumerate(embeddings)
    ]

    pairs = [
        {"a": i, "b": j, "score": matrix[i][j]}
        for i in range(n)
        for j in range(i + 1, n)
    ]
    pairs.sort(key=lambda p: p["score"], reverse=True)

    model_name = embeddings[0]["model_name"] if embeddings else "hoops_embeddings_model"
    if timer:
        timer.lap("matrix+pairs")
        timer.emit(
            "COMPARE_TIMING",
            {
                "count": n,
                "file_ids": ",".join(f[:12] for f in file_ids),
            },
        )
    return {
        "count": n,
        "model_name": model_name,
        "files": files,
        "matrix": matrix,
        "pairs": pairs,
    }


def log_compute_environment(tag: str) -> None:
    """Log the interpreter, SDK and math-library configuration.

    A no-op unless ``HOOPS_SEARCH_TIMING=1``. Emitted once at startup and once
    per timed search so two deployments can be compared line for line: a
    CPU-only vs CUDA torch build, or a different thread configuration, explains
    a compute gap on its own. Best-effort: never raises.
    """
    if not config._search_timing_enabled():
        return
    fields: dict[str, Any] = {"python": sys.executable}
    try:
        import hoops_ai  # noqa: F401

        fields["hoops_ai"] = getattr(hoops_ai, "__file__", "?")
        fields["hoops_ai_version"] = getattr(hoops_ai, "__version__", "?")
    except Exception:
        fields["hoops_ai"] = "import-failed"
    try:
        import torch

        fields["torch"] = torch.__version__
        fields["torch_cuda_build"] = getattr(torch.version, "cuda", None)
        try:
            fields["mkl"] = torch.backends.mkl.is_available()
        except Exception:
            fields["mkl"] = "?"
        try:
            fields["cuda_available"] = torch.cuda.is_available()
        except Exception:
            fields["cuda_available"] = "?"
        try:
            fields["torch_threads"] = torch.get_num_threads()
            fields["torch_interop"] = torch.get_num_interop_threads()
        except Exception:
            pass
    except Exception:
        fields["torch"] = "import-failed"
    # Which device the loaded embedder actually sits on -- best-effort, and only
    # if it has already been initialised (do not force a model load just to log).
    try:
        emb = _embedder
        if emb is not None:
            for attr in ("device", "_device"):
                dev = getattr(emb, attr, None)
                if dev is not None:
                    fields["embedder_device"] = str(dev)
                    break
    except Exception:
        pass
    fields["cpu_count"] = os.cpu_count()
    for var in _INTRAOP_ENV_VARS:
        fields[var] = os.environ.get(var, "<unset>")
    ctx = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info("[%s] %s", tag, ctx)
