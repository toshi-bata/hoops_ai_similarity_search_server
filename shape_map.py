"""Index shape map: a 2D cluster map and a point-cloud SCS export.

Two artifacts are built from a stored FAISS index, both without re-embedding.
``compute_index_shape_map`` reads the per-body vectors back, aggregates them to
one L2-normalised point per file id, projects them with a deterministic
economy-SVD PCA into a fixed [-1, 1] square and clusters them with deterministic
FAISS k-means; it returns the JSON 2D map that ``index/{name}/map`` serves.
``compute_index_shape_map_scs`` colours those same points (by cluster or by tag)
and bakes them into a single ``.scs`` point cloud with the license-bearing
scs_points tool so the HOOPS Web Viewer can open the map as a point cloud.

Two caches back these. ``_shape_map_agg`` keeps the few most-recent aggregates
(the expensive SVD result, ~84 MB for 10k parts) keyed by
``(faiss path, mtime_ns, subset_sig)``; a handful are kept so drill-down can walk
back up without re-running the SVD. ``_shape_map_results`` is a small
LRU of finished per-parameter payloads keyed by
``(faiss path, mtime_ns, subset_sig, n_clusters, dims)``. The ``.scs`` files are
cached on disk under ``SCS_MAP_DIR`` and pruned by ``_prune_scs_map_cache``.

Drill-down. A map can be restricted to an explicit set of file ids
(``only_file_ids``), which re-projects and re-clusters within that set, so the
axes are redrawn rather than merely zoomed. The set is too long for a URL, so
``_register_map_subset`` keeps it in a small LRU under a short id that ``/map``
returns and ``/map/scs`` takes back; an evicted id is a 409 telling the client
to re-post ``/map``.

Dependency direction. This module depends only on the standard library,
``config`` (APP_ROOT, SCS_MAP_DIR, _env_int), ``embeddings`` (get_embedder),
``locks`` (the shared index lock plus the shape-map and scs-map build-lock
tables) and ``index_registry`` / ``tags``. The four registry helpers
``_index_faiss_path`` / ``_load_index_schema`` / ``_load_named_index`` /
``_load_shape_index_pickle_compat`` are (A) names reached as
``index_registry.<name>`` (module attribute); ``_validate_index_name`` is a
stable (C) name imported by name. It must never import ``core``: the dependency
runs one way, ``core`` -> ``shape_map`` -> ``tags`` / ``index_registry`` /
``locks`` / ``config`` / ``embeddings``.

Why the tag helpers are imported BY NAME, never ``import tags``. Both
``compute_index_shape_map`` and ``compute_index_shape_map_scs`` have a parameter
called ``tags`` and ``_shape_map_subset_sig`` has a local named ``tags``. Binding
the ``tags`` module at module level would let those shadow the module inside the
function, so any later ``tags.<attr>`` would hit the argument/local instead --
the exact UnboundLocalError-class footgun behind #61. The four helpers
``normalize_tag_filter`` / ``tag_filter_predicate`` / ``tags_by_part`` /
``_shape_map_tags_snapshot`` are stable (C) names, so ``from tags import ...`` is
both safe and sufficient.

Why numpy / faiss / the SDK stay function-local. ``core`` never imports numpy at
module level, so importing this module must not pull in numpy, FAISS or
``hoops_ai``. Every ``import numpy`` / ``import faiss`` / ``from hoops_ai...``
therefore lives inside the function that needs it; only the cheap standard
library and the sibling pure-Python modules are imported at module scope.

Reference convention (see "The re-export convention" in docs/architecture.md).
``_load_index_body_batch``, ``_shape_map_filemeta``, ``compute_index_shape_map``,
``compute_index_shape_map_scs`` and ``compute_index_catalogue`` are the (A)
names: tests swap those
``shape_map.<name>`` attributes to install fakes and assert the orchestrators
drove them, and ``routers/similarity.py`` reaches ``shape_map.compute_index_shape_map``
/ ``shape_map.compute_index_shape_map_scs`` / ``shape_map.compute_index_catalogue``.
They are therefore NOT re-exported
by ``core``; re-exporting would bind a second object under ``core`` while this
module keeps using its own, so a test swap and the live path would diverge -- the
double-truth trap. The core orchestrator ``_warmup_one`` calls them as
``shape_map.compute_index_shape_map`` / ``shape_map.compute_index_shape_map_scs``.
Every other name here is stable and is re-exported from ``core`` by explicit name
for its existing callers (including ``run_index_add_job`` -> ``prewarm_shape_map``,
``run_startup_maintenance`` -> ``clear_scs_map_cache`` and ``_warmup_one`` ->
``ScsToolUnavailable``).
"""


import hashlib
import json
import logging
import math
import os
import pathlib
import statistics
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

import config
import embeddings
import index_registry

# ``_env_int`` is a stdlib-only env helper in config.py; it is never swapped by a
# test, so a plain ``from`` import is safe and lets the bare ``_env_int(...)`` call
# sites keep working. APP_ROOT / SCS_MAP_DIR are reached as ``config.<name>``.
from config import _env_int

# The per-index locks this module needs live in locks.py: the shared index lock
# (also held by part search) and the shape-map / scs-map build-lock tables.
from locks import (
    _get_index_lock,
    _scs_map_build_locks,
    _scs_map_build_locks_mutex,
    _shape_map_build_locks,
    _shape_map_build_locks_mutex,
)

# Stable (C) name in index_registry. The (A) names (_index_faiss_path,
# _load_index_schema, _load_named_index, _load_shape_index_pickle_compat) are
# reached as ``index_registry.<name>`` (module attribute), never imported.
from index_registry import _validate_index_name

# Stable (C) tag helpers imported BY NAME on purpose: compute_index_shape_map and
# compute_index_shape_map_scs have a ``tags`` parameter (and _shape_map_subset_sig
# a ``tags`` local), so ``import tags`` here would let those shadow the module
# inside the function (the #61 footgun). Never ``import tags`` here.
from tags import (
    _shape_map_tags_snapshot,
    normalize_tag_filter,
    tag_filter_predicate,
    tags_by_part,
)


logger = logging.getLogger(__name__)


# --- Build timing ----------------------------------------------------------
# One INFO line per map build, naming every stage and whether it came from
# cache (hit) or was computed (miss). The point of the line is to answer, from
# the log alone, WHICH STAGE WAS NOT A CACHE HIT: "the Ward map is slow" is a
# report that cannot be acted on, while "linkage 38.40s miss" is. The same
# build can cost wildly different amounts depending on what is already warm --
# the aggregate, its projection and Ward's linkage are three separate caches --
# so without the breakdown a slow request and a cold cache look identical.
#
# Measurement only. Nothing here reads or changes a cache key, a return value
# or an error path, and no file id, file name or path is ever logged: an index
# name and numbers, nothing more.

# Thread counts scipy's and numpy's native libraries read at import/call time.
# Logged once per linkage build at DEBUG because they are the usual explanation
# for the same work taking different times in two processes -- a server started
# from a shell that exports them and one started from a service manager are not
# running the same linkage.
_SHAPE_MAP_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

_STAGE_HIT = "hit"
_STAGE_MISS = "miss"

# Rows multiplied against the whole matrix at once when building Ward's
# distances (see _shape_map_condensed_sqdist). At N=7,928 one block's product
# is 512 x 7,928 float64 = about 32 MB: small enough to stay cache-friendly and
# large enough that BLAS can thread it, against the 503 MB the full square
# matrix would need.
_SHAPE_MAP_DIST_BLOCK_ROWS = 512


def _shape_map_rss_mb() -> Optional[float]:
    """This process's resident set size in MB, or None if it cannot be read.

    psutil is not a declared dependency, so this never requires it: the probe
    is the same optional one index_registry uses to size embedding workers, and
    where it is absent the rss field is simply left out of the timing line. No
    other mechanism is attempted -- a measurement is not worth a dependency.
    """
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


class _ShapeMapTiming:
    """Stage timings for one build, rendered as a single log line.

    Stages are recorded in the order they are first seen and re-recording one
    adds to it, so a stage entered twice (the projection of a parent map, say)
    reads as the total it cost. ``note`` carries the plain facts that belong on
    the same line -- the point count, the cluster count, the RSS either side of
    the linkage -- because a time without the size it was spent on cannot be
    compared against another run.
    """

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._stages: "OrderedDict[str, list]" = OrderedDict()
        self._notes: "OrderedDict[str, str]" = OrderedDict()

    def add(self, label: str, seconds: float, state: Optional[str] = None) -> None:
        entry = self._stages.get(label)
        if entry is None:
            self._stages[label] = [float(seconds), state]
            return
        entry[0] += float(seconds)
        if state is not None:
            entry[1] = state

    def stage(self, label: str, state: Optional[str] = None) -> "_ShapeMapStage":
        """Time a block: ``with timing.stage("proj", _STAGE_MISS): ...``."""
        return _ShapeMapStage(self, label, state)

    def note(self, key: str, value: Any) -> None:
        self._notes[key] = str(value)

    def total(self) -> float:
        return time.perf_counter() - self._started

    def render(self) -> str:
        parts = []
        for label, (seconds, state) in self._stages.items():
            parts.append(f"{label} {seconds:.2f}s {state}" if state else f"{label} {seconds:.2f}s")
        parts.extend(f"{key}={value}" for key, value in self._notes.items())
        return ", ".join(parts)


class _ShapeMapStage:
    """Context manager returned by :meth:`_ShapeMapTiming.stage`.

    Records on the way out whether or not the block raised, so a build that
    fails part way still leaves the stage that was running accounted for if the
    caller chooses to log it.
    """

    def __init__(self, timing: "_ShapeMapTiming", label: str, state: Optional[str]) -> None:
        self._timing = timing
        self._label = label
        self._state = state
        self._started = 0.0

    def __enter__(self) -> "_ShapeMapStage":
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._timing.add(self._label, time.perf_counter() - self._started, self._state)
        return False

    def state(self, state: str) -> None:
        """Set hit/miss once the block knows which it turned out to be."""
        self._state = state


def _shape_map_log_threads() -> None:
    """Log the native thread-count variables, once per linkage build."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    shown = ", ".join(
        f"{var}={os.environ.get(var) or 'unset'}" for var in _SHAPE_MAP_THREAD_VARS
    )
    logger.debug("[SHAPEMAP] native thread settings: %s", shown)


# About this many parts to a cluster. The one rule both sides read: warmup
# builds with it, and the client opens on it (through /index/{name}/stats).
_SHAPE_MAP_PARTS_PER_CLUSTER = 32
# The bounds of the counts the UI offers and /map accepts.
_SHAPE_MAP_AUTO_CLUSTERS_MIN = 8
_SHAPE_MAP_AUTO_CLUSTERS_MAX = 1024


def shape_map_auto_cluster_count(n_parts: int) -> int:
    """The cluster count a map of ``n_parts`` parts should open with.

    Aims at about 32 parts to a cluster. That target is what makes tmcad's
    7,928 single parts come out at 256, which is also the number of leaves
    Ward's tree descends to by default -- so the two clusterings describe the
    index at the same granularity and can be read against each other.

    Rounded to a power of two because Ward's tree is built from binary splits:
    its own cut points are powers of two, and a k-means count off that ladder
    would make the two panels hard to compare at a glance.

    Clamped to 8..1024, the range the Clusters control offers and /map accepts,
    and never more than there are parts to cluster. Fewer than two parts still
    yields 2, the smallest count a clustering can mean anything at.
    """
    n = int(n_parts)
    if n < 2:
        return 2
    target = n / _SHAPE_MAP_PARTS_PER_CLUSTER
    if target < 1:
        k = _SHAPE_MAP_AUTO_CLUSTERS_MIN
    else:
        k = 2 ** round(math.log2(target))
    k = max(_SHAPE_MAP_AUTO_CLUSTERS_MIN, min(_SHAPE_MAP_AUTO_CLUSTERS_MAX, k))
    return min(k, n)


# --- Index-based Shape Map (index/{name}/map) ------------------------------
# The economy SVD dominates the cost (~7.6 s on 10k x 2048) and depends only on
# (faiss path, mtime_ns, subset), NOT on n_clusters or dims. So the aggregate +
# projection are cached apart from the finished per-parameter payload.
#
# _shape_map_agg keeps the most-recent aggregates (~84 MB for 10k parts):
# {file_ids, coords, agg} keyed by (faiss path, mtime_ns, subset_sig).
# Drill-down walks between a parent map and a child map repeatedly, so keeping
# only one would re-run the ~7.6 s SVD on every step back up. The root
# aggregate is the big one; a drilled subset is far smaller (~7 MB at 862
# parts), so in practice this holds one large entry plus a few small ones.
_SHAPE_MAP_AGG_CACHE_MAX = 4
_shape_map_agg: "OrderedDict[tuple, Any]" = OrderedDict()

# _shape_map_results is a small LRU of finished response payloads (a few MB
# each) keyed by (faiss path, mtime_ns, subset_sig, n_clusters, dims). Re-running
# k-means for a new cluster count on a cached aggregate is ~0.2 s.
_SHAPE_MAP_RESULT_CACHE_MAX = 8
_shape_map_results: "OrderedDict[tuple, Any]" = OrderedDict()

# _shape_map_linkage keeps the Ward trees. Each one is only (N-1) x 4 float64 --
# 254 KB at 7,928 points -- but costs ~38 s to build, so this is a cache of work,
# not of memory. Keyed by the AGGREGATE's identity and deliberately NOT by the
# projection: the tree is built from agg alone, so switching the map between PCA
# and UMAP must not rebuild it.
_SHAPE_MAP_LINKAGE_CACHE_MAX = 4
_shape_map_linkage: "OrderedDict[tuple, Any]" = OrderedDict()

# Guards the caches above. Innermost lock: never acquire anything else while
# holding it.
_shape_map_cache_mutex = threading.Lock()

# Explicit drill-down subsets, keyed by a short digest of their id list. A
# drill-down map is defined by an arbitrary set of file ids, which is far too
# long to travel in the ``/map/scs`` query string, so /map hands the client a
# ``subset_id`` and /map/scs takes that id back. The registry is an LRU rather
# than a permanent store because a subset is only meaningful while the user is
# looking at it; an expired id yields 409 and the client re-posts /map.
# Guarded by _shape_map_cache_mutex (plain dict work, so the innermost-lock
# rule above still holds).
_SHAPE_MAP_SUBSET_CACHE_MAX = 8
_shape_map_subsets: "OrderedDict[str, list[str]]" = OrderedDict()

# The shape-map build lock (_shape_map_build_locks) now lives in locks.py.

# Index names whose default map is being prewarmed in the background after a
# registration job, so a burst of jobs cannot stack rebuilds of the same map.
_shape_map_prewarm_in_flight: set[str] = set()
_shape_map_prewarm_mutex = threading.Lock()


# ---------------------------------------------------------------------------
# Index-based Shape Map (numeric core)
#
# Ported verbatim from hoops_ai_native_bridge's compute_shape_map: the stored
# per-body vectors of a FAISS index are read back (no re-embedding), aggregated
# to one L2-normalized point per file id, projected with an economy-SVD PCA into
# a fixed [-1, 1] cube, and clustered with deterministic FAISS k-means
# (seed=42, niter=20). Determinism is a contract: the same index and parameters
# always produce the same coordinates and cluster assignments. The guarantee is
# over the *set of files*, not over a stored row order: the aggregate rows are
# sorted canonically by file id (a content hash) before the SVD, the principal
# components carry a fixed sign convention. So registration order, the number
# of registration passes, and which files happened to fall into a second pass
# all leave the coordinates and the cluster membership unchanged.
#
# Cluster NUMBERING is a separate matter. Clusters are renumbered by member
# count descending (ties broken by cohesion, then by mean projected position),
# so the legend lists the biggest families first and can be read from the top.
# That key is deliberately not stable across different file sets or different
# Outliers thresholds: adding parts changes the counts, so the same cluster may
# get a different number and colour. The parts themselves stay where they are.
# ---------------------------------------------------------------------------


def _shape_map_l2norm_rows(mat):
    """L2-normalize the rows of a matrix (zero rows are left unchanged)."""
    import numpy as np

    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


def _load_index_body_batch(name: str) -> tuple:
    """Read the stored per-body ids and vectors of a named index.

    Returns ``(ids, values)`` where *ids* is a list of file_ids (one per stored
    body row) and *values* is the ``(n_bodies, dim)`` matrix of stored body
    embeddings. Loading goes through a fresh ``CADSearch`` exactly like
    hoops_ai_native_bridge, so no CAD file is re-embedded.

    Raises ``KeyError`` when the index does not exist.
    """
    faiss_path = index_registry._index_faiss_path(name)
    if not faiss_path.exists():
        raise KeyError(f"Index '{name}' does not exist.")

    try:
        from hoops_ai.ml.embeddings import CADSearch
    except ImportError:  # older/newer layouts re-export it one level up
        from hoops_ai.ml import CADSearch

    searcher = CADSearch(shape_model=embeddings.get_embedder())
    b = index_registry._load_shape_index_pickle_compat(searcher, faiss_path)
    return (list(b.ids), b.values)


# How the aggregate vectors are flattened to three dimensions FOR DISPLAY.
#
# Only the drawing changes: k-means runs on the full-dimensional aggregate
# either way (see _shape_map_cluster), so switching the projection moves the
# points but never changes which cluster a part belongs to.
_PROJECTION_PCA = "pca"
_PROJECTION_UMAP = "umap"

_UMAP_N_NEIGHBORS_DEFAULT = 15
_UMAP_MIN_DIST_DEFAULT = 0.1
_UMAP_SEED_DEFAULT = 42


class UmapUnavailable(RuntimeError):
    """umap-learn is not installed. Only the ``projection=umap`` path needs it.

    The same shape as :class:`ScsToolUnavailable`: an optional dependency that a
    deployment may simply not have, which must be reported as "this one feature
    is unavailable here" rather than as a failure of the map.
    """


def _normalize_map_projection(projection: Optional[str]) -> str:
    """Normalize a projection name to ``"pca"`` or ``"umap"``.

    Unknown values collapse to ``"pca"`` (the default layout) so a bad query
    never fails a map that would otherwise have been drawn.
    """
    p = str(projection or _PROJECTION_PCA).lower()
    if p == _PROJECTION_UMAP:
        return _PROJECTION_UMAP
    return _PROJECTION_PCA


def _shape_map_umap_available() -> bool:
    """True when umap-learn can be imported, WITHOUT importing it.

    ``find_spec`` only locates the module. Actually importing umap-learn drags
    in numba and compiles on first use, which is seconds of start-up and tens of
    megabytes of memory -- a cost no deployment that never asks for a UMAP map
    should pay. The import happens in :func:`_shape_map_umap_project`, at the
    point where it is actually being used.
    """
    import importlib.util

    return importlib.util.find_spec("umap") is not None


def _shape_map_umap_n_neighbors() -> int:
    """UMAP neighbourhood size, from HOOPS_AI_SHAPE_MAP_UMAP_N_NEIGHBORS.

    Small values follow local detail, large ones the global shape. Read on every
    call, like the other env helpers, so it can be changed without a restart.
    """
    return _env_int(
        "HOOPS_AI_SHAPE_MAP_UMAP_N_NEIGHBORS",
        _UMAP_N_NEIGHBORS_DEFAULT,
        minimum=2,
        maximum=200,
    )


def _shape_map_umap_min_dist() -> float:
    """UMAP minimum point separation, from HOOPS_AI_SHAPE_MAP_UMAP_MIN_DIST.

    How tightly points are allowed to pack together. Read on every call; an
    unparseable value falls back to the default and an out-of-range one is
    clamped, so a typo cannot stop a map being drawn.
    """
    raw = os.environ.get("HOOPS_AI_SHAPE_MAP_UMAP_MIN_DIST", "").strip()
    if not raw:
        return _UMAP_MIN_DIST_DEFAULT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "HOOPS_AI_SHAPE_MAP_UMAP_MIN_DIST=%r is not a number; falling back to %s",
            raw, _UMAP_MIN_DIST_DEFAULT,
        )
        return _UMAP_MIN_DIST_DEFAULT
    return max(0.0, min(0.99, value))


def _shape_map_umap_seed() -> int:
    """UMAP ``random_state``, from HOOPS_AI_SHAPE_MAP_UMAP_SEED.

    Fixing it is what makes the layout reproducible -- and also what makes
    umap-learn run single-threaded, since its parallel code path cannot be made
    deterministic. Read on every call, like the other env helpers.
    """
    return _env_int(
        "HOOPS_AI_SHAPE_MAP_UMAP_SEED",
        _UMAP_SEED_DEFAULT,
        minimum=0,
        maximum=2 ** 31 - 1,
    )


def _shape_map_projection_signature(projection: str) -> tuple:
    """Cache-key fragment for the projection; empty ONLY for PCA.

    Deliberately different from :func:`_shape_map_kmeans_signature`, which is
    empty whenever its settings are at their defaults. Here:

    * ``pca`` returns ``()``. PCA is what every cache entry ever written was
      built with, so an empty fragment leaves every key bit-for-bit unchanged
      and every ``.scs`` already on disk still addressable.
    * ``umap`` returns a NON-EMPTY fragment even when every UMAP setting is at
      its default. Collapsing it to empty at the defaults -- the k-means rule --
      would make a UMAP request hash to the same key as the PCA entry and be
      answered with the PCA coordinates. The defaults are the common case, so
      that would be the normal outcome, not an edge case.
    """
    projection = _normalize_map_projection(projection)
    if projection != _PROJECTION_UMAP:
        return ()
    return ((
        "umap",
        _shape_map_umap_n_neighbors(),
        _shape_map_umap_min_dist(),
        _shape_map_umap_seed(),
    ),)


def _shape_map_umap_project(agg, d3: int):
    """Lay the aggregate vectors out in *d3* dimensions with UMAP, or ``None``.

    ``None`` means "too few points, use PCA instead": UMAP works by building a
    k-nearest-neighbour graph, and a drill-down subset of a handful of parts has
    no neighbourhood structure to build one from. PCA is defined for any number
    of points and is the honest answer there.

    Raises ``ImportError`` when umap-learn is absent; callers check
    :func:`_shape_map_umap_available` first.
    """
    import numpy as np

    n_neighbors = _shape_map_umap_n_neighbors()
    N = int(agg.shape[0])
    if N < max(5, n_neighbors + 1):
        return None

    import umap

    # cosine, because the aggregate rows are L2-normalized (see
    # _shape_map_l2norm_rows) and cosine is the distance the index itself
    # searches with. A Euclidean UMAP would lay the map out by a measure the
    # search does not use.
    reducer = umap.UMAP(
        n_components=d3,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=_shape_map_umap_min_dist(),
        random_state=_shape_map_umap_seed(),
    )
    coords = np.asarray(reducer.fit_transform(agg), dtype=np.float32)

    # Centre it. _shape_map_cluster() fits coordinates into the [-1, 1] cube by
    # dividing by the largest absolute component and nothing else -- it never
    # moves the origin. A PCA projection is mean-centred by construction so that
    # is enough, but UMAP's origin is arbitrary: measured on tmcad (7,928 parts)
    # the UMAP centroid sat at (0.55, 0.59, 0.27), which without this would push
    # the whole cloud into one corner of the cube and draw it at a fraction of
    # the available size.
    coords = coords - coords.mean(axis=0, keepdims=True)
    return np.asarray(coords, dtype=np.float32)


def _shape_map_project(
    b_ids, b_values, only_ids=None, keep_fn=None, projection=_PROJECTION_PCA,
    timings=None,
) -> tuple:
    """Aggregate body rows per file id and project them for display.

    Returns ``(file_ids, coords_full, agg)``:

    * ``file_ids`` – one entry per file id, in canonical (sorted) order,
      independent of how the rows were stored.
    * ``coords_full`` – the ``(N, d3)`` **unscaled** projection, where
      ``d3 = min(3, dim, N)``. *projection* chooses how it is produced: ``"pca"``
      (the default) uses the mean-centred economy SVD and ``"umap"`` uses
      :func:`_shape_map_umap_project`, falling back to PCA when there are too
      few points for a neighbour graph.
      Per-request scaling into the [-1, 1]
      cube is done in :func:`_shape_map_cluster` so a single projection serves
      both 2D and 3D.
    * ``agg`` – the ``(N, dim)`` L2-normalized aggregate, retained for k-means.

    When a subset is requested the projection is computed **on that subset
    only** (matching the reference), so removing parts changes the layout of
    the rest. A subset can be given either as *only_ids* (an explicit id
    allow-set) or as *keep_fn* (a ``part_id -> bool`` predicate, used by the tag
    filter so ``tagged=false`` can keep parts absent from the tag sidecar).
    *keep_fn* takes precedence when both are given.

    *timings*, when given, is a dict this fills with ``{"proj": seconds}``: the
    cost of the projection alone, measured from the first line that touches the
    SVD or UMAP. The caller subtracts it from the whole call to get the cost of
    the aggregation, which is the only way to tell "reading and averaging 12,000
    body rows" apart from "decomposing a 7,928 x 2,048 matrix" in the log.
    """
    import numpy as np
    from collections import OrderedDict

    projection = _normalize_map_projection(projection)
    vals = _shape_map_l2norm_rows(b_values)

    groups: "OrderedDict[str, list]" = OrderedDict()
    for row, fid in enumerate(b_ids):
        groups.setdefault(fid, []).append(row)
    # Canonical order, NOT the order the rows happen to be stored in. A file id
    # is a content hash, so sorting puts the same file in the same place however
    # it was registered -- and an index built in two passes (a short time limit,
    # then the files that timed out) stores its rows in an order that depends on
    # machine load. FAISS seeds its initial centroids from the training rows, so
    # without this the same corpus clusters differently on every rebuild.
    file_ids = sorted(groups.keys(), key=str)

    if keep_fn is not None:
        file_ids = [fid for fid in file_ids if keep_fn(str(fid))]
    elif only_ids:
        allow = set(str(x) for x in only_ids)
        file_ids = [fid for fid in file_ids if str(fid) in allow]

    N = len(file_ids)
    dim = int(vals.shape[1]) if vals.ndim == 2 and vals.shape[0] else 0
    if N == 0:
        return ([], np.zeros((0, 0), dtype=np.float32), np.zeros((0, dim), dtype=np.float32))

    agg = np.zeros((N, dim), dtype=np.float32)
    for i, fid in enumerate(file_ids):
        agg[i] = vals[groups[fid]].mean(axis=0)
    agg = _shape_map_l2norm_rows(agg)

    d3 = max(1, min(3, agg.shape[1], N))
    proj_started = time.perf_counter()
    if projection == _PROJECTION_UMAP:
        # Any failure here propagates. The SVD below has a fallback because a
        # degenerate matrix still has to produce SOME layout, but silently
        # drawing a PCA map for someone who asked for a UMAP one is the worst
        # possible outcome: the picture would be wrong and look right. The only
        # substitution made is the documented too-few-points one, which
        # _shape_map_umap_project signals by returning None.
        umap_coords = _shape_map_umap_project(agg, d3)
        if umap_coords is not None:
            if timings is not None:
                timings["proj"] = time.perf_counter() - proj_started
            return (file_ids, np.asarray(umap_coords, dtype=np.float32), agg)

    mean = agg.mean(axis=0, keepdims=True)
    X = agg - mean
    try:
        _U, _S, Vt = np.linalg.svd(X, full_matrices=False)
        basis = np.array(Vt[:d3], dtype=np.float32)
        # A principal component's sign is mathematically arbitrary: numpy
        # returns whichever one the decomposition happened to produce, and that
        # can flip when the rows change. A flipped axis mirrors the whole map,
        # which reads as "a completely different picture" even when the
        # structure is identical. Fix it: make the largest-magnitude loading of
        # each component positive (the same convention as sklearn's svd_flip).
        for r in range(basis.shape[0]):
            j = int(np.argmax(np.abs(basis[r])))
            if basis[r, j] < 0:
                basis[r] = -basis[r]
        coords = X @ basis.T
    except Exception:
        coords = X[:, :d3]
    coords = np.asarray(coords, dtype=np.float32)
    if timings is not None:
        timings["proj"] = time.perf_counter() - proj_started
    return (file_ids, coords, agg)


# FAISS k-means settings that are otherwise its library defaults. They are
# exposed so their effect can be MEASURED without editing code; the defaults
# below reproduce the current behaviour exactly, so leaving them unset changes
# nothing. Group C knobs (see config.py's catalogue): not in README or
# .env.example, and not exposed through the UI or the API.
#
# * max_points_per_centroid -- FAISS trains on a random subsample when there are
#   more than k * this many vectors. At the default, a 7,928-part index with
#   k=8 learns its centroids from 2,048 points, about a quarter of the corpus.
# * nredo -- restarts from fresh initial centroids, keeping the best. FAISS
#   seeds by shuffling the training vectors and taking the first k (it is not
#   k-means++), so a restart is the only defence against a poor draw.
# * spherical -- re-normalises the centroids after every iteration. The map
#   evaluates its clusters with cosine similarity to a normalised centroid
#   (cohesion, the outlier pass, centre parts), so with this off the clustering
#   and its evaluation do not treat centroids the same way.
_KMEANS_MAX_POINTS_PER_CENTROID_DEFAULT = 256
_KMEANS_NREDO_DEFAULT = 1
_KMEANS_SPHERICAL_DEFAULT = False
_KMEANS_INIT_DEFAULT = "random"

# Spellings accepted for HOOPS_AI_SHAPE_MAP_KMEANS_INIT, mapped onto the three
# canonical names. The aliases are there because every source writes these
# differently and a silent fallback to the default would look like "the setting
# does nothing".
_KMEANS_INIT_ALIASES = {
    "random": "random",
    "kmeans++": "kmeans++",
    "k-means++": "kmeans++",
    "kmeans_plus_plus": "kmeans++",
    "afkmc2": "afkmc2",
    "afk-mc2": "afkmc2",
}


def _shape_map_kmeans_max_points_per_centroid() -> int:
    """Training vectors per centroid, from HOOPS_AI_SHAPE_MAP_KMEANS_MAX_POINTS_PER_CENTROID."""
    return _env_int(
        "HOOPS_AI_SHAPE_MAP_KMEANS_MAX_POINTS_PER_CENTROID",
        _KMEANS_MAX_POINTS_PER_CENTROID_DEFAULT,
        minimum=1,
        maximum=10_000_000,
    )


def _shape_map_kmeans_nredo() -> int:
    """k-means restarts, from HOOPS_AI_SHAPE_MAP_KMEANS_NREDO.

    Capped at 50: each redo is a full k-means run, so a misplaced digit would
    otherwise make every map hundreds of times slower to build.
    """
    return _env_int(
        "HOOPS_AI_SHAPE_MAP_KMEANS_NREDO",
        _KMEANS_NREDO_DEFAULT,
        minimum=1,
        maximum=50,
    )


def _shape_map_kmeans_spherical() -> bool:
    """True when HOOPS_AI_SHAPE_MAP_KMEANS_SPHERICAL=1.

    Read on every call, like the other env helpers, so it can be switched
    without restarting the server.
    """
    return os.environ.get("HOOPS_AI_SHAPE_MAP_KMEANS_SPHERICAL", "").strip() == "1"


def _shape_map_kmeans_init() -> str:
    """k-means initialisation, from HOOPS_AI_SHAPE_MAP_KMEANS_INIT.

    One of ``"random"`` (FAISS's own default: the first k of a random
    permutation of the training vectors), ``"kmeans++"``, or ``"afkmc2"``
    (a sampling approximation of k-means++). Read on every call, like the
    other env helpers, so it can be switched without a restart. An
    unrecognised value falls back to the default.
    """
    raw = os.environ.get("HOOPS_AI_SHAPE_MAP_KMEANS_INIT", "").strip().lower()
    if not raw:
        return _KMEANS_INIT_DEFAULT
    name = _KMEANS_INIT_ALIASES.get(raw)
    if name is None:
        logger.warning(
            "HOOPS_AI_SHAPE_MAP_KMEANS_INIT=%r is not one of random/kmeans++/afkmc2; "
            "falling back to %s",
            raw, _KMEANS_INIT_DEFAULT,
        )
        return _KMEANS_INIT_DEFAULT
    return name


# Which algorithm turns the aggregate vectors into cluster labels. k-means is
# the default and the only one that existed before; "ward" is scipy's
# agglomerative hierarchical clustering, which builds one tree for the whole
# index and cuts it at the requested count. The cut is what makes it worth
# having: every k is a refinement of every smaller k, so raising Clusters (k)
# splits the families already on screen instead of redrawing them.
_CLUSTERING_KMEANS = "kmeans"
_CLUSTERING_WARD = "ward"

# The finest cut Ward offers, and the ceiling the granularity scale is read
# against: granularity 1.0 means this many clusters (or the population, when it
# is smaller). Callers may ask for another ceiling, but this is the one every
# cache key written before it was a parameter assumed, so the key fragments
# stay empty here -- see _shape_map_granularity_signature.
_SHAPE_MAP_WARD_K_MAX = 256

# Point cap for Ward. See _shape_map_linkage_matrix for why there has to be one.
_WARD_MAX_POINTS_DEFAULT = 20000


class WardTooLarge(RuntimeError):
    """The index has more points than Ward clustering can be run on here.

    ``scipy.cluster.hierarchy.linkage`` builds the full condensed distance
    matrix, ``N(N-1)/2`` float64 values, before it starts merging. That is 251
    MB at 7,928 points and grows as N squared, so there is a real ceiling
    rather than merely a slow path. The ceiling is
    HOOPS_AI_SHAPE_MAP_WARD_MAX_POINTS (default 20000) and can be raised on a
    machine with the memory for it.

    A request the data cannot support, not a missing dependency, so unlike
    :class:`UmapUnavailable` this is a client error.
    """


def _normalize_map_clustering(clustering: Optional[str]) -> str:
    """Normalize a clustering name to ``"kmeans"`` / ``"ward"``.

    Unknown values collapse to ``"kmeans"`` (the default algorithm) so a bad
    query never fails a map that would otherwise have been drawn. Same shape as
    :func:`_normalize_map_projection`.
    """
    c = str(clustering or _CLUSTERING_KMEANS).lower()
    return _CLUSTERING_WARD if c == _CLUSTERING_WARD else _CLUSTERING_KMEANS


def _shape_map_ward_max_points() -> int:
    """Ward's point cap, from HOOPS_AI_SHAPE_MAP_WARD_MAX_POINTS.

    Read on every call, like the other env helpers, so it can be raised without
    restarting the server. The default of 20000 is roughly 1.6 GB of condensed
    distance matrix; see :class:`WardTooLarge`.
    """
    return _env_int(
        "HOOPS_AI_SHAPE_MAP_WARD_MAX_POINTS",
        _WARD_MAX_POINTS_DEFAULT,
        minimum=2,
        maximum=200_000,
    )


def _shape_map_clustering_signature(clustering: str) -> tuple:
    """Cache-key fragment for the clustering algorithm; empty ONLY for k-means.

    Follows :func:`_shape_map_projection_signature`, not
    :func:`_shape_map_kmeans_signature`:

    * ``kmeans`` returns ``()``. Every cache entry ever written, and every
      ``.scs`` already on disk, was built with k-means, so an empty fragment
      leaves all of them bit-for-bit addressable.
    * ``ward`` returns a NON-EMPTY fragment even though Ward has no tunable
      parameters at all. The temptation is to return ``()`` because there is
      nothing to put in it -- but then a Ward request would hash to the k-means
      entry for the same index and be answered with the k-means assignment. The
      fragment exists to say "a different algorithm", not to carry settings.
    """
    if _normalize_map_clustering(clustering) != _CLUSTERING_WARD:
        return ()
    return (("ward",),)


def _shape_map_granularity_signature(clustering: str, granularity, ward_max=None) -> tuple:
    """Cache-key fragment for Ward's cut height; empty unless it is in use.

    ``granularity`` is a height at which to cut the Ward tree, so it means
    nothing for k-means and nothing when it was not asked for -- both return
    ``()``, which leaves every entry written before it existed addressable by
    the same key.

    Rounded to 4 decimals, the same rounding
    :func:`_shape_map_ward_cut_height` applies, so the key describes the
    computation that was actually run.

    ``ward_max`` is the ceiling the granularity is read against, so it changes
    which height a given granularity means and has to reach the key as well. It
    is appended only when it is not the default, which keeps every entry and
    every ``.scs`` written before it existed addressable by the same key.
    """
    if _normalize_map_clustering(clustering) != _CLUSTERING_WARD:
        return ()
    if granularity is None:
        return ()
    key = (("g", round(float(granularity), 4)),)
    if ward_max is not None and int(ward_max) != _SHAPE_MAP_WARD_K_MAX:
        key = key + (("wmax", int(ward_max)),)
    return key


def _shape_map_condensed_sqdist(X, block_rows: int = _SHAPE_MAP_DIST_BLOCK_ROWS):
    """The condensed Euclidean distance matrix of *X*, via blocked matrix products.

    scipy's ``linkage`` will compute this itself when handed an observation
    matrix, but it does so pair by pair on a single thread, which is where
    nearly all of a Ward build's time goes. The same distances come out of one
    matrix product per block, and BLAS runs that multi-threaded: measured at
    N=7,928 / 2,048 dimensions, 19.5-20.0 s becomes 3.7-4.0 s.

    The general form is used deliberately::

        d^2 = ||xi||^2 + ||xj||^2 - 2 * (xi . xj)

    Not ``d^2 = 2 - 2s``, which is the same thing but only while every row is a
    unit vector. The rows reaching Ward are L2-normalised today, but that is a
    property of a caller several frames up: if :func:`_shape_map_l2norm_rows`
    ever moves, the short form would keep returning numbers -- wrong ones --
    and the tree would silently be a different tree. The squared norms cost one
    pass over the rows and remove the assumption entirely.

    The N x N square matrix is never formed. Each block of *block_rows* rows is
    multiplied against the whole matrix and only its upper-triangular part is
    written straight into the condensed array, whose entry for ``i < j`` sits
    at ``i*N - i*(i+1)//2 + (j - i - 1)``. Materialising the square first would
    mean a second N^2 float64 allocation -- 503 MB at N=7,928, on top of the
    251 MB the condensed array already needs -- for nothing.

    Rounding can leave ``d^2`` a shade below zero for two numerically identical
    rows, so it is clipped at zero before the square root; without that, NaNs
    appear for exactly the pairs a CAD index has most of.

    Returns a float64 array of length ``N*(N-1)/2``, which is what
    ``linkage(method="ward")`` wants.
    """
    import numpy as np

    # float64 throughout: Ward merges the closest pair at every step, so a
    # last-bit difference in a distance can swap a merge order and hand back a
    # different tree. The aggregate rows arrive as float32.
    Xd = np.ascontiguousarray(X, dtype=np.float64)
    n = int(Xd.shape[0])
    sq_norms = np.einsum("ij,ij->i", Xd, Xd)

    out = np.empty(n * (n - 1) // 2, dtype=np.float64)
    block_rows = max(1, int(block_rows))
    Xt = Xd.T

    for i0 in range(0, n - 1, block_rows):
        i1 = min(i0 + block_rows, n)
        s = Xd[i0:i1] @ Xt
        d2 = sq_norms[i0:i1, None] + sq_norms[None, :] - 2.0 * s
        np.maximum(d2, 0.0, out=d2)
        np.sqrt(d2, out=d2)
        for i in range(i0, i1):
            if i + 1 >= n:
                continue
            start = n * i - i * (i + 1) // 2
            out[start:start + (n - i - 1)] = d2[i - i0, i + 1:]
    return out


def _shape_map_linkage_matrix(Xk):
    """Build the Ward linkage tree over the aggregate rows *Xk*.

    Ward is a Euclidean method, which is what makes it defensible on rows that
    are compared elsewhere by cosine: :func:`_shape_map_l2norm_rows` has already
    L2-normalized them, so ``d^2 = 2 - 2cos`` and Euclidean distance is a
    monotone function of cosine distance. This is Ward on the unit sphere, and
    it orders pairs exactly the way the index's own similarity does.

    The distances are computed here by :func:`_shape_map_condensed_sqdist`
    rather than left to scipy, which computes them single-threaded. Measured at
    N=7,928 / 2,048 dimensions with scipy 1.17.1: 19.5-20.0 s became 3.7-4.0 s,
    and under a busy machine (a browser competing for cores) 64.3 s became the
    low teens. The two paths were checked to agree, not assumed to: cut at
    k=2, k=8 and k=256 they give the same partition, with a largest merge-height
    difference of 2.1e-04.

    The condensed distance matrix is now allocated here -- 251 MB at N=7,928,
    the dominant term in a peak RSS of about 0.6 GB -- and it grows as N
    squared, as does the time. So this refuses outright above
    :func:`_shape_map_ward_max_points` rather than attempting a build that
    would swap the machine. Raises :class:`WardTooLarge`.
    """
    from scipy.cluster.hierarchy import linkage

    n = int(Xk.shape[0])
    cap = _shape_map_ward_max_points()
    if n > cap:
        raise WardTooLarge(
            f"clustering=ward needs an N(N-1)/2 distance matrix and is capped at "
            f"{cap} points; this map has {n}. Raise "
            f"HOOPS_AI_SHAPE_MAP_WARD_MAX_POINTS if this machine has the memory "
            f"for it, or use clustering=kmeans."
        )
    return linkage(_shape_map_condensed_sqdist(Xk), method="ward")


def _shape_map_run_ward(linkage_z, k) -> "Any":
    """Cut the Ward tree *linkage_z* into *k* clusters; returns an ``int64`` array.

    Two things about ``fcluster`` that the caller has to know:

    * It numbers clusters from 1, so one is subtracted here. Everything
      downstream -- the renumbering, the palette, the outlier pass -- assumes
      0-based ids.
    * ``criterion="maxclust"`` returns AT MOST *k* clusters, not exactly *k*.
      The tree may have no cut that separates that many, in which case fewer
      come back and the caller must take the real count from the result.

    Measured at N=7,928: 4.4 ms at k=8 and about 2 ms at k=24/64/128 -- next to
    nothing beside the tree it cuts, which is the whole reason the tree is
    cached and this is not.
    """
    import numpy as np
    from scipy.cluster.hierarchy import fcluster

    labels = fcluster(linkage_z, int(k), criterion="maxclust")
    return np.asarray(labels, dtype=np.int64) - 1


def _shape_map_parent_agg(
    name, path_key, mtime_ns, tag_filter, tag_keep, kind, tags_etag,
):
    """The cached aggregate entry for the UNDRILLED population, as ``(key, entry)``.

    The key is returned with the entry because the linkage cache is keyed by
    it too: the tree is built from these rows and nothing else, so one key
    covers both.

    The entry is the whole thing -- ``file_ids``, ``coords``, ``agg``,
    ``filemeta``, ``tags_map`` -- because a caller that walks the tree needs
    the rows it was cut from (cohesion, centres) and the names to show.

    Lifted verbatim out of :func:`_shape_map_parent_linkage` so the rows can be
    had without building a linkage; see :func:`shape_map_aggregate_vectors`.
    """
    parent_sig = _shape_map_subset_sig(tag_filter, kind, None)
    parent_agg_key = (path_key, mtime_ns, parent_sig, tags_etag if tag_filter else "")

    with _shape_map_cache_mutex:
        # _shape_map_projection_signature(_PROJECTION_PCA) is (), so a PCA
        # svd_key IS the agg_key and this hits directly. A parent that was only
        # ever drawn with UMAP is stored under agg_key + ("umap", ...), so look
        # for any entry whose key starts with it -- the rows are the same
        # either way, and the tree is built from the rows alone.
        parent_agg = _shape_map_agg.get(parent_agg_key)
        if parent_agg is not None:
            _shape_map_agg.move_to_end(parent_agg_key)
        else:
            width = len(parent_agg_key)
            for key in list(_shape_map_agg.keys()):
                if key[:width] == parent_agg_key:
                    parent_agg = _shape_map_agg[key]
                    _shape_map_agg.move_to_end(key)
                    break

    if parent_agg is None:
        # Always PCA: only ``agg`` is wanted here and the projection does not
        # touch it, so running UMAP would be tens of seconds spent on
        # coordinates nobody reads.
        filemeta = _shape_map_filemeta(name)
        keep_fn = _compose_keep(tag_keep, _shape_map_kind_predicate(filemeta, kind))
        ids, values = _load_index_body_batch(name)
        file_ids, coords_full, agg = _shape_map_project(
            ids, values, only_ids=None, keep_fn=keep_fn,
            projection=_PROJECTION_PCA,
        )
        parent_agg = {
            "file_ids": file_ids,
            "coords": coords_full,
            "agg": agg,
            "filemeta": filemeta,
            "tags_map": tags_by_part(name),
        }
        with _shape_map_cache_mutex:
            _shape_map_agg[parent_agg_key] = parent_agg
            _shape_map_agg.move_to_end(parent_agg_key)
            while len(_shape_map_agg) > _SHAPE_MAP_AGG_CACHE_MAX:
                _shape_map_agg.popitem(last=False)

    return (parent_agg_key, parent_agg)


def _shape_map_parent_linkage(
    name, path_key, mtime_ns, tag_filter, tag_keep, kind, tags_etag,
):
    """The Ward tree of the UNDRILLED map, as ``(tree, parent_agg)``.

    ``parent_agg`` is the whole cached aggregate entry -- ``file_ids``,
    ``coords``, ``agg``, ``filemeta``, ``tags_map`` -- and not just the ids,
    because a caller that walks the tree needs the rows it was built from
    (cohesion, centres) and the file names to show for them.

    A drill-down posts its files as an explicit subset, which lands in
    ``subset_sig`` and so in the aggregate's cache key. Left alone, that means a
    brand new tree over the subset -- the user believes they walked down the
    hierarchy, but they are looking at a different computation. This finds the
    tree the parent view was clustered with, so the drill can be served by
    cutting it lower down instead.

    The parent is the same tag filter and the same kind with NO explicit subset,
    which is exactly what ``_shape_map_subset_sig`` produces when it is handed
    no subset id.

    When neither the parent aggregate nor its tree is in cache, both are rebuilt
    here rather than giving up. Falling back on a cache miss would make the
    answer depend on cache state: the same drill-down would descend the parent's
    tree while the parent was warm and silently re-cluster the subset once it
    had been evicted, and those are two different results from one request.
    Returns ``None`` only when the tree genuinely cannot be had -- that is,
    :class:`WardTooLarge` -- which leaves the caller with its ordinary
    build-a-tree-on-the-subset path.
    """
    # The aggregate first, then its tree. The two cache reads used to share one
    # acquisition of the mutex; they no longer do, which changes nothing a
    # caller can see -- each read is still consistent, and whichever of the two
    # is missing is built the same way.
    parent_agg_key, parent_agg = _shape_map_parent_agg(
        name, path_key, mtime_ns, tag_filter, tag_keep, kind, tags_etag,
    )

    with _shape_map_cache_mutex:
        tree = _shape_map_linkage.get(parent_agg_key)
        if tree is not None:
            _shape_map_linkage.move_to_end(parent_agg_key)

    if tree is None:
        try:
            _shape_map_log_threads()
            # Outside the mutex: tens of seconds, and the cache lock must not
            # be held across it.
            tree = _shape_map_linkage_matrix(parent_agg["agg"])
        except WardTooLarge:
            return None
        with _shape_map_cache_mutex:
            _shape_map_linkage[parent_agg_key] = tree
            _shape_map_linkage.move_to_end(parent_agg_key)
            while len(_shape_map_linkage) > _SHAPE_MAP_LINKAGE_CACHE_MAX:
                _shape_map_linkage.popitem(last=False)

    return (tree, parent_agg)


def _shape_map_ward_subset_node(tree, parent_file_ids, file_ids):
    """Rows of *file_ids* in the parent, if they form one node of *tree*.

    Returns an ``int64`` array of parent row indices, or ``None`` when this
    subset is not a node and so cannot be descended into.

    The test is built on the one property cuts of a linkage tree have: raising
    K under ``maxclust`` always REFINES the partition, never redraws it. So
    "the subset is an exact union of whole clusters" is monotone in K -- once it
    holds, splitting those clusters further keeps it holding -- and the smallest
    K where it holds can be found by bisection.

    Being a union is not enough on its own, though: at ``K = n_parent`` every
    point is its own cluster, so EVERY subset is a union there. What separates a
    node from an arbitrary collection is how many clusters it takes. A node
    becomes one whole cluster at some K and is mixed in with others below that,
    so at the smallest K where it is a union it occupies exactly one cluster.
    Anything else needs two or more. Hence the count is checked at K0.

    Costs about log2(n_parent) calls to ``fcluster`` -- 13 or so at 8,000
    points, a couple of milliseconds each.
    """
    import numpy as np
    from scipy.cluster.hierarchy import fcluster

    row_of = {str(fid): i for i, fid in enumerate(parent_file_ids)}
    rows = []
    for fid in file_ids:
        row = row_of.get(str(fid))
        if row is None:
            return None
        rows.append(row)
    rows = np.asarray(rows, dtype=np.int64)

    n_parent = len(parent_file_ids)
    if rows.size == 0 or rows.size == n_parent:
        # Not drilled (or nothing to drill): there is no deeper cut to take.
        return None

    def union_and_count(k):
        labels = np.asarray(fcluster(tree, int(k), criterion="maxclust"))
        used = np.unique(labels[rows])
        covered = int(np.isin(labels, used).sum())
        return covered == int(rows.size), int(used.size)

    lo, hi = 1, n_parent
    while lo < hi:
        mid = (lo + hi) // 2
        if union_and_count(mid)[0]:
            hi = mid
        else:
            lo = mid + 1

    is_union, count = union_and_count(lo)
    if not is_union or count != 1:
        return None
    return rows


def _shape_map_ward_descend(tree, rows, k):
    """Cut *tree* deep enough to split the node *rows* into *k* parts.

    Returns 0-based ``int64`` labels aligned with *rows*. The cluster count
    inside a node is also monotone in K, so the shallowest K that splits it into
    at least *k* is found by the same bisection; the labels are then restricted
    to the node and renumbered 0..m-1 in order of first appearance.

    As with :func:`_shape_map_run_ward`, the count that comes back is not
    guaranteed to be the count that was asked for -- equal merge distances can
    make one step of K split the node into two pieces at once, overshooting
    *k* slightly. The caller must read the real count off the result.
    """
    import numpy as np
    from scipy.cluster.hierarchy import fcluster

    rows = np.asarray(rows, dtype=np.int64)
    want = max(1, int(k))
    n_parent = int(tree.shape[0]) + 1

    def count_at(cut):
        labels = np.asarray(fcluster(tree, int(cut), criterion="maxclust"))
        return labels, int(np.unique(labels[rows]).size)

    lo, hi = 1, n_parent
    while lo < hi:
        mid = (lo + hi) // 2
        if count_at(mid)[1] >= want:
            hi = mid
        else:
            lo = mid + 1

    labels, _ = count_at(lo)
    local = labels[rows]
    out = np.empty(local.shape[0], dtype=np.int64)
    seen: dict = {}
    for i, value in enumerate(local):
        key = int(value)
        if key not in seen:
            seen[key] = len(seen)
        out[i] = seen[key]
    return out


def _shape_map_ward_internal_heights(tree, rows):
    """The merge heights that build *rows* internally, lowest first.

    *rows* is ``None`` for the whole tree, or the parent row numbers of a
    drill-down node (what :func:`_shape_map_ward_subset_node` returns). The
    result is a ``float64`` array in ascending order.

    A merge belongs to a node when BOTH of its children are already inside that
    node -- that is what "this merge is part of the node's own structure" means.
    scipy numbers the leaves ``0..n-1`` and gives row ``m`` the new node number
    ``n + m``, and a row's children always have smaller numbers than the row
    itself, so one pass from the bottom settles every node. The node's own root
    merge qualifies too, so the array is always ``rows.size - 1`` long: a set of
    ``rows.size`` points is assembled by exactly that many merges.

    This replaces the bisections the band ends used to need. The pass is O(n) in
    Python -- a few milliseconds at n = 7,928 -- against a bisection that called
    ``fcluster`` over the whole tree at every step.
    """
    import numpy as np

    heights = np.asarray(tree[:, 2], dtype=np.float64)
    if rows is None:
        # Ward's linkage is monotone, so this is already sorted; sorting is
        # cheap and makes the contract hold whatever produced the tree.
        return np.sort(heights)

    n = int(tree.shape[0]) + 1
    inside = np.zeros(2 * n - 1, dtype=bool)
    inside[np.asarray(rows, dtype=np.int64)] = True
    keep = np.zeros(n - 1, dtype=bool)
    for m in range(n - 1):
        a = int(tree[m, 0])
        b = int(tree[m, 1])
        both = bool(inside[a] and inside[b])
        inside[n + m] = both
        keep[m] = both
    return np.sort(heights[keep])


def _shape_map_ward_cut_height(
    tree, rows, granularity, heights=None, ward_max=_SHAPE_MAP_WARD_K_MAX,
):
    """Turn a 0..1 *granularity* into a height at which to cut *tree*.

    Returns ``(height, height_lo, height_hi, k_max)``. *rows* is ``None`` for
    the whole tree, or the parent row numbers of a drill-down node (what
    :func:`_shape_map_ward_subset_node` returns). *heights* is the output of
    :func:`_shape_map_ward_internal_heights` when the caller already has it.

    Cutting by height is a different statement from cutting by count: a height
    says "anything closer than this is one family", so the SAME height yields a
    different number of clusters on different branches -- a branch full of
    near-identical parts stays one cluster while a varied one falls apart into
    many. ``maxclust`` cannot express that; it forces every branch to the same
    budget.

    ``fcluster(..., criterion="distance")`` performs every merge at or below the
    given height, so cutting at ``heights[j]`` leaves ``m - (j + 1)`` clusters
    whether *rows* is a node or the whole tree -- the node's internal heights
    are exactly the merges that assemble it, so the same arithmetic applies on
    both and no search is needed.

    *granularity* maps to a TARGET COUNT on a log scale, ``1 .. k_max``, and the
    height is the one that produces it. Linear would be useless: 8 to 64
    clusters, the range anyone actually works in, would be squeezed into the
    far end of the slider while most of its travel sat between 64 and k_max.
    The log is how the ticks are spaced, not how the tree is cut -- the cut is
    still a single height, and the height that was really used is reported back
    on the payload so it can be read rather than guessed at.

    *granularity* is rounded to 4 decimals here as well as in the cache key, so
    the value that was hashed is exactly the value that was computed with.

    *ward_max* is the ceiling of that scale, still clamped to the population:
    granularity 1.0 means ``min(m, ward_max)`` clusters. Every caller that also
    builds a tree has to pass the SAME value it gave
    :func:`_shape_map_ward_tree`, or the slider positions would be read against
    one scale and the levels built on another.
    """
    import math

    import numpy as np

    if heights is None:
        heights = _shape_map_ward_internal_heights(tree, rows)
    heights = np.asarray(heights, dtype=np.float64)
    m = int(heights.shape[0]) + 1
    k_max = int(max(1, min(m, int(ward_max))))
    g = min(1.0, max(0.0, round(float(granularity), 4)))
    # One tick below the shallowest merge: the height that leaves every point
    # on its own. np.nextafter rather than 0.0 because duplicated rows merge at
    # a distance of exactly 0.0, and a cut AT 0.0 would already join them.
    below_min = (
        float(np.nextafter(float(heights[0]), -np.inf)) if heights.size else 0.0
    )

    target = int(round(math.exp(g * math.log(float(k_max))))) if k_max > 1 else 1
    target = max(1, min(target, k_max))

    def height_for(count):
        # Cutting at heights[j] performs j + 1 merges, so c clusters remain at
        # j = m - 1 - c. A negative index means c == m, i.e. no merge at all.
        idx = m - 1 - int(count)
        if idx < 0:
            return below_min
        idx = min(idx, heights.shape[0] - 1)
        # Equal merge heights all happen at once, so take the LAST index
        # carrying this value -- cutting there really does leave `count`.
        idx = int(np.searchsorted(heights, heights[idx], side="right")) - 1
        return float(heights[idx])

    height_hi = float(heights[-1]) if heights.size else below_min
    height_lo = height_for(k_max)
    height = height_for(target)

    if height_lo > height_hi:
        height_lo = height_hi
    height = min(max(float(height), height_lo), height_hi)
    return (float(height), float(height_lo), float(height_hi), k_max)


def _shape_map_ward_suggestions(heights, m, k_max, limit=6):
    """Heights where the tree's own merge cost jumps, as slider positions.

    *heights* is :func:`_shape_map_ward_internal_heights` for a population of
    *m* points. Returns up to *limit* entries
    ``{"granularity", "height", "clusters"}``, ordered by cluster count.

    These are observations, not advice. All each one says is that the next
    merge up costs disproportionately more than the ones below it -- the tree
    reporting where joining two families stops being cheap. Which cut to use is
    the reader's decision; nothing here ranks one as the right answer.

    The score is the RATIO ``heights[j + 1] / heights[j]``, not the difference.
    Ward's merge heights grow towards the root, so an absolute difference would
    pick the topmost few merges every time and say nothing about the rest of
    the slider. A ratio is scale-free and compares a jump near the leaves with
    one near the root on equal terms.

    Cuts leaving a single cluster are not offered: one cluster is not a place to
    cut. The band is therefore ``2 .. k_max`` clusters.
    """
    import math

    import numpy as np

    heights = np.asarray(heights, dtype=np.float64)
    if int(k_max) <= 1 or heights.size < 2:
        return []

    k_max = int(k_max)
    m = int(m)
    log_k_max = math.log(float(k_max))
    if log_k_max <= 0.0:
        return []

    # j runs over the cuts leaving 2..k_max clusters: c = m - 1 - j.
    lo = max(0, m - 1 - k_max)
    hi = min(heights.shape[0] - 2, m - 3)
    scored = []
    for j in range(lo, hi + 1):
        base = float(heights[j])
        if base <= 0.0:
            # Identical parts merge at a distance of exactly 0; a ratio against
            # 0 is meaningless. Such rows sit below the band in practice, so
            # this is a guard rather than a case.
            continue
        scored.append((float(heights[j + 1]) / base, j))

    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    # Cluster count is m - 1 - j, so descending j is ascending clusters, which
    # is the order the slider runs in.
    chosen = sorted((j for _, j in scored[: int(limit)]), reverse=True)

    out = []
    for j in chosen:
        clusters = m - 1 - j
        # The exact inverse of the forward map, so feeding this granularity back
        # in lands on this very cut rather than near it.
        g = min(1.0, max(0.0, math.log(float(clusters)) / log_k_max))
        out.append({
            "granularity": round(g, 4),
            "height": float(heights[j]),
            "clusters": int(clusters),
        })
    return out


def _shape_map_run_ward_at_height(tree, height, rows=None):
    """Cut *tree* at *height* and return 0-based ``int64`` labels.

    One ``fcluster`` call, no search: a height IS the cut, unlike a count, which
    has to be hunted for. When *rows* is given the labels are restricted to that
    node and renumbered 0..m-1 in order of first appearance -- and that single
    call is exactly what "cut the parent's own tree lower down" means for a
    drill-down, with no new tree built anywhere.
    """
    import numpy as np
    from scipy.cluster.hierarchy import fcluster

    labels = np.asarray(fcluster(tree, float(height), criterion="distance"))
    if rows is not None:
        labels = labels[np.asarray(rows, dtype=np.int64)]
    out = np.empty(labels.shape[0], dtype=np.int64)
    seen: dict = {}
    for i, value in enumerate(labels):
        key = int(value)
        if key not in seen:
            seen[key] = len(seen)
        out[i] = seen[key]
    return out


def _shape_map_ward_tree(
    tree, agg, file_ids, rows, k_max: int = _SHAPE_MAP_WARD_K_MAX,
) -> tuple:
    """Fold Ward's tree into a shallow tree of levels, for a tree view.

    Returns ``(nodes, leaf_of_member)``. *tree* is the linkage, *agg* the
    L2-normalised aggregate matrix of this population, *file_ids* its file ids
    in the same order, *rows* either ``None`` (the whole tree) or the parent row
    numbers of a drill-down node, and *k_max* the target cluster count of the
    deepest level.

    Each node is ``{"id", "parent", "level", "count", "height", "cohesion",
    "centre"}``; ``parent`` is ``-1`` on the root alone, ``level`` is the target
    cluster count of the level that created the node, and ``centre`` is the file
    id of its centre part. ``leaf_of_member`` is an ``int64`` array of length
    *m* mapping each member onto the deepest node it belongs to.

    The binary dendrogram is not shown as it stands, for two reasons. Reaching a
    family from the root takes dozens of expansions, and a branch holding many
    identical parts merges them one at a time, so it reads as a long chain in
    which every step sheds a single point and says nothing.

    The levels are the target counts 1, 2, 4, 8, ... k_max -- the SAME
    logarithmic scale the Granularity slider uses, so a slider position and a
    level of this tree are the same cut of the same tree, and moving between
    them does not change what is being looked at. A level that did not split a
    node is skipped rather than repeated, which is what collapses those chains.

    That bounds the whole structure: a level at most doubles the number of
    groups and stops at k_max, so there are at most ``2 * k_max - 1`` nodes --
    511 by default, whether the population is 300 parts or 8,000. The tree is
    therefore served whole, with no paging to design around.

    Height is an absolute dissimilarity, so every node on one level was cut at
    the same height: they are families of the same tightness, which is what
    makes two nodes on a level comparable at a glance.
    """
    import math

    import numpy as np

    m = int(tree.shape[0]) + 1 if rows is None else int(np.asarray(rows).size)
    # The granularity scale belongs to the cut helper, which has a ceiling of
    # its own (min(m, ward_max)). Asking it for g = log(c) / log(OUR k_max)
    # would read the position against the wrong scale, so hand it OUR ceiling
    # and invert the one it reports back: g = 1 then means the deepest level,
    # whatever it holds.
    _h0, _lo0, _hi0, band_k_max = _shape_map_ward_cut_height(
        tree, rows, 1.0, ward_max=k_max,
    )
    band_k_max = max(1, int(band_k_max))
    k_max = max(1, min(int(k_max), band_k_max))
    targets = [1]
    while targets[-1] < k_max:
        targets.append(min(k_max, targets[-1] * 2))

    Xk = agg if rows is None else agg[np.asarray(rows, dtype=np.int64)]
    member_file_ids = list(file_ids)
    nodes: list = []
    leaf_of_member = np.zeros(m, dtype=np.int64)
    groups: dict = {}

    for c in targets:
        # The exact inverse of the slider's forward map, read against the cut
        # helper's own ceiling so this level is the cut that slider position
        # gives. Ties can make the realised count differ from the target; the
        # tree is built from the labels that actually came back, not the target.
        g = 0.0 if band_k_max <= 1 else math.log(float(c)) / math.log(float(band_k_max))
        height, _lo, _hi, _k = _shape_map_ward_cut_height(
            tree, rows, g, ward_max=k_max,
        )
        labels = _shape_map_run_ward_at_height(tree, height, rows)

        if not nodes:
            # The coarsest cut is one cluster by construction -- its height is
            # the tree's own last merge -- so the root holds every member.
            labels = np.zeros(m, dtype=np.int64)
            coh = _shape_map_cohesion(Xk, labels)
            ctr = _shape_map_centres(Xk, labels, member_file_ids)
            members = np.arange(m, dtype=np.int64)
            nodes.append({
                "id": 0,
                "parent": -1,
                "level": int(c),
                "count": int(m),
                "height": float(height),
                "cohesion": _shape_map_tree_value(coh, 0),
                "centre": ctr.get(0),
            })
            leaf_of_member[:] = 0
            groups = {0: members}
            continue

        # Once per level, not once per node: both helpers take the whole
        # labelling, and the children of this level are exactly its clusters.
        coh = _shape_map_cohesion(Xk, labels)
        ctr = _shape_map_centres(Xk, labels, member_file_ids)

        next_groups: dict = {}
        for node_id, members in groups.items():
            sub = labels[members]
            uniq = np.unique(sub)
            if uniq.size <= 1:
                # Did not split at this level. Carry the node forward instead of
                # creating a duplicate: this is what collapses the long chains a
                # binary dendrogram would show, where a big block of identical
                # parts sheds one point at a time.
                next_groups[node_id] = members
                continue
            for u in uniq:
                child_members = members[sub == u]
                child_id = len(nodes)
                nodes.append({
                    "id": child_id,
                    "parent": int(node_id),
                    "level": int(c),
                    "count": int(child_members.size),
                    "height": float(height),
                    "cohesion": _shape_map_tree_value(coh, int(u)),
                    "centre": ctr.get(int(u)),
                })
                # Overwritten as the levels descend, so when the last one has
                # run every member names the deepest node holding it -- including
                # on branches that stopped splitting early.
                leaf_of_member[child_members] = child_id
                next_groups[child_id] = child_members
        groups = next_groups

    return nodes, leaf_of_member


def _shape_map_tree_value(mapping, key):
    """A cohesion reading as a plain float, or ``None`` when there is none.

    A cluster of one has no mean similarity to report, and the helpers leave it
    out rather than inventing a 1.0.
    """
    value = mapping.get(key)
    return None if value is None else float(value)


# --- Cohesion frontier -----------------------------------------------------
# Where a catalogue would cut the Ward tree: descend from the root and stop as
# soon as a node is cohesive enough to stand on its own.
#
# These four are pure -- they take the node array ``_shape_map_ward_tree``
# produces and return plain data, with no index, no numpy and no SDK -- so they
# are tested directly and are shared by the catalogue endpoint and by offline
# diagnostic scripts. Such a script imports them FROM HERE; the dependency
# never runs the other way, because production code importing a diagnostic
# script would make tools/ part of the server.
#
# ``shape_map_pair_spread`` at the end of the block is pure in the same sense
# -- data in, data out, tested directly -- but takes vectors rather than nodes
# and so does use numpy. It is the one measurement here that needs the rows the
# tree was cut from.
#
# Reference convention: these are stable (C) names. Nothing swaps them, the
# router never reaches them (it calls compute_index_catalogue), and diagnostic
# scripts import them by name. They are therefore safe to re-export from ``core``,
# but there is nothing to re-export them FOR: ``core`` has no caller for a
# tree-walking helper, and a re-export with no caller is a second name to keep
# true. If one appears, re-export by explicit name as the other stable names
# here are.


def shape_map_tree_children(nodes: list[dict]) -> dict[int, list[int]]:
    """``{node id: [child id, ...]}`` in the order the nodes were given."""
    children: dict[int, list[int]] = {}
    for node in nodes:
        parent = int(node["parent"])
        if parent < 0:
            continue
        children.setdefault(parent, []).append(int(node["id"]))
    return children


def shape_map_tree_roots(nodes: list[dict]) -> list[int]:
    """Ids of the nodes with no parent in *nodes*.

    Normally one, but a node whose parent is missing is treated as a root
    rather than silently dropped: losing a branch would quietly shrink every
    total built from the frontier.
    """
    known = {int(node["id"]) for node in nodes}
    return [
        int(node["id"]) for node in nodes
        if int(node["parent"]) < 0 or int(node["parent"]) not in known
    ]


def shape_map_cohesion_frontier(nodes: list[dict], threshold: float) -> dict[str, list[dict]]:
    """Descend from each root and stop where the branch is good enough.

    Returns ``{"reached": [...], "stalled": [...]}``, both lists of the node
    dicts themselves, largest first.

    A node ends its branch when its cohesion is at least *threshold*; its
    descendants are then not visited at all, so no ancestor of a reached node
    can also be on the frontier. A node with no children that never reached the
    threshold ends its branch too, and is reported separately: it is on the
    frontier because the tree ran out, which is not the same answer, and the
    stalled groups are what a catalogue cannot summarise.

    Cohesion is not assumed to be monotone down the tree, so the test is "the
    first node on this path that reaches it" rather than anything about the
    parent. A node with no cohesion -- :func:`_shape_map_tree_value` reports
    none for a group of one rather than inventing 1.0 -- cannot reach it, by
    that same rule.
    """
    by_id = {int(node["id"]): node for node in nodes}
    children = shape_map_tree_children(nodes)
    reached: list[dict] = []
    stalled: list[dict] = []

    stack = list(reversed(shape_map_tree_roots(nodes)))
    seen: set[int] = set()
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            # A malformed tree (a cycle, or two parents naming one child) would
            # otherwise count members twice or loop for ever.
            raise ValueError(f"node {node_id} reached twice; the tree is not a tree")
        seen.add(node_id)
        node = by_id[node_id]
        cohesion = node.get("cohesion")
        if cohesion is not None and float(cohesion) >= threshold:
            reached.append(node)
            continue
        kids = children.get(node_id) or []
        if not kids:
            stalled.append(node)
            continue
        stack.extend(reversed(kids))

    reached.sort(key=lambda n: (-int(n["count"]), int(n["id"])))
    stalled.sort(key=lambda n: (-int(n["count"]), int(n["id"])))
    return {"reached": reached, "stalled": stalled}


def shape_map_check_frontier(nodes: list[dict], frontier: dict, total: int) -> None:
    """Assert the frontier counts every member exactly once.

    Two things are checked, because they fail differently: the counts must add
    up to *total*, and no frontier node may be an ancestor of another (which
    would add up correctly only by accident). Raises ``AssertionError``, which
    callers let escape: a frontier that is not a partition makes every number
    derived from it meaningless, and a catalogue built from one would tag the
    same part under two entries.
    """
    groups = [*frontier["reached"], *frontier["stalled"]]
    counted = sum(int(node["count"]) for node in groups)
    assert counted == total, (
        f"frontier covers {counted} members, the tree has {total}"
    )
    parent_of = {int(node["id"]): int(node["parent"]) for node in nodes}
    chosen = {int(node["id"]) for node in groups}
    assert len(chosen) == len(groups), "a node appears twice in the frontier"
    for node_id in chosen:
        walker = parent_of.get(node_id, -1)
        while walker >= 0:
            assert walker not in chosen, (
                f"node {node_id} is below node {walker}; both are on the frontier"
            )
            walker = parent_of.get(walker, -1)


def shape_map_frontier_members(
    nodes: list[dict],
    frontier: dict,
    file_ids: list[str],
    leaf: list[int],
) -> dict[int, list[str]]:
    """``{frontier node id: [file_id, ...]}`` from the tree's parallel arrays.

    The tree reports only the DEEPEST node holding each member, so a frontier
    node's membership is found by walking that leaf up its ancestors until a
    frontier node is met -- which must happen, since the frontier cuts every
    root-to-leaf path exactly once. A member whose walk reaches the root
    without meeting one raises ``AssertionError``: silently dropping it would
    produce a catalogue that looks complete and is not.

    Ids keep the order of *file_ids*, so a repeated request lists a group's
    members the same way.
    """
    parent_of = {int(node["id"]): int(node["parent"]) for node in nodes}
    chosen = {int(node["id"]) for node in [*frontier["reached"], *frontier["stalled"]]}
    members: dict[int, list[str]] = {node_id: [] for node_id in chosen}
    for file_id, leaf_id in zip(file_ids, leaf):
        walker = int(leaf_id)
        while walker >= 0 and walker not in chosen:
            walker = parent_of.get(walker, -1)
        assert walker in chosen, (
            f"member under node {leaf_id} is below no frontier node"
        )
        members[walker].append(str(file_id))
    return members


def shape_map_frontier_order(nodes: list[dict], frontier_ids: set[int]) -> list[int]:
    """Frontier ids in the order the tree panel draws them.

    A pre-order walk: a node comes before everything beneath it, and siblings
    come in the order they appear in *nodes*. That sibling order is not a
    choice made here -- web/src/shape-map.js builds its ``childrenOf`` map by
    iterating the node array once and pushing each node onto its parent's list,
    so the array's own order IS the order of the rows on screen. Matching it is
    the whole point: a catalogue that listed the same families in a different
    order from the tree beside it would make the two panels hard to read
    together, which is the only reason to have both.

    The walk stops at a frontier node. Nothing below one can be on the frontier
    (:func:`shape_map_check_frontier` asserts exactly that), so descending
    further could only waste the visit.

    Ids in *frontier_ids* that no root reaches are appended afterwards, in node
    order, rather than dropped: a catalogue missing a group would be a quieter
    fault than one listing it out of place.
    """
    children = shape_map_tree_children(nodes)
    ordered: list[int] = []
    seen: set[int] = set()
    stack = list(reversed(shape_map_tree_roots(nodes)))
    while stack:
        node_id = stack.pop()
        if node_id in frontier_ids:
            ordered.append(node_id)
            seen.add(node_id)
            continue
        stack.extend(reversed(children.get(node_id, [])))
    for node in nodes:
        node_id = int(node["id"])
        if node_id in frontier_ids and node_id not in seen:
            ordered.append(node_id)
            seen.add(node_id)
    return ordered


# How many tag names one catalogue group reports. A group is summarised, not
# listed: past a handful the names stop telling the user anything the count
# does not, and the response already carries every file id.
SHAPE_MAP_GROUP_TAG_LIMIT = 8


def shape_map_group_tags(
    file_ids: list[str],
    tags_map: dict[str, list[str]],
    limit: int = SHAPE_MAP_GROUP_TAG_LIMIT,
) -> tuple[int, list[str]]:
    """``(tagged part count, [tag name, ...])`` for one group's members.

    A part counts once however many tags it carries: the question the count
    answers is "how much of this group has been dealt with", not "how many tags
    were written". The names are the most frequent first, ties broken by name
    so the list is stable between two identical requests, and cut to *limit*.
    """
    tagged = 0
    counts: dict[str, int] = {}
    for file_id in file_ids:
        names = tags_map.get(str(file_id)) or []
        if not names:
            continue
        tagged += 1
        for name in names:
            counts[str(name)] = counts.get(str(name), 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tagged, [name for name, _count in ranked[:limit]]


# Group sizes the summaries count, in ascending order. Below 2 a "family" is a
# single part, and above 20 a catalogue entry stops being something a user can
# scan. Shared so a diagnostic table and the endpoint's stats cannot disagree.
SHAPE_MAP_SIZE_MARKS = (2, 5, 10, 20)


def shape_map_classify_frontier(
    frontier: dict,
    min_reached_parts: int = 5,
) -> dict[str, list[dict]]:
    """Split a frontier into ``confident`` / ``small`` / ``mixed`` groups.

    * ``confident`` -- reached the threshold AND holds at least
      *min_reached_parts* members.
    * ``small`` -- reached the threshold with fewer members than that.
    * ``mixed`` -- never reached it (which includes a group of one, whose
      cohesion is reported as none rather than invented).

    The three lists are the frontier again, so they still partition every
    member exactly once; nothing is dropped and nothing is counted twice.

    *min_reached_parts* is a CLASSIFICATION threshold, never a stopping
    condition for the walk. Stopping on size would change nothing: the walk
    already stops at the first cohesive node, and that node's children are
    smaller still, so there is nowhere further down to go. Keeping it out of
    :func:`shape_map_cohesion_frontier` is what makes the frontier itself
    independent of this knob -- change it and the same groups come back,
    labelled differently.

    Why a size line at all: cohesion is entangled with size. A mean over the
    three pairs of a three-part group clears 0.95 far more easily than a mean
    over the 16,110 pairs of a 180-part group, so "reached the threshold" is a
    much weaker claim for the small one. The line sits where the catalogue
    stops moving with the tree's depth (see the measurement by
    ``_CATALOGUE_DEFAULT_MIN_REACHED``), not at a round number.
    """
    floor = int(min_reached_parts)
    confident: list[dict] = []
    small: list[dict] = []
    for node in frontier["reached"]:
        (confident if int(node["count"]) >= floor else small).append(node)
    return {"confident": confident, "small": small, "mixed": list(frontier["stalled"])}


def shape_map_size_buckets(
    groups: list[dict],
    marks: tuple = SHAPE_MAP_SIZE_MARKS,
) -> list[tuple]:
    """``[(size mark, groups at least that big, members in them), ...]``.

    Cumulative, not bucketed into bands: the question a catalogue asks is "how
    many entries are worth showing", and that is always an "at least this many
    parts" question.
    """
    out: list[tuple] = []
    for mark in marks:
        chosen = [node for node in groups if int(node["count"]) >= mark]
        out.append((mark, len(chosen), sum(int(node["count"]) for node in chosen)))
    return out


def shape_map_size_stats(groups: list[dict]) -> dict[str, float]:
    """Largest, median and singleton count over *groups*."""
    sizes = [int(node["count"]) for node in groups]
    if not sizes:
        return {"groups": 0, "members": 0, "largest": 0, "median": 0.0, "singletons": 0}
    return {
        "groups": len(sizes),
        "members": sum(sizes),
        "largest": max(sizes),
        "median": statistics.median(sizes),
        "singletons": sum(1 for size in sizes if size == 1),
    }


def shape_map_pair_spread(vectors) -> dict:
    """Pairwise cosines INSIDE one group: ``{pairs, min, mean, argmin}``.

    The server never calls this: it is a public helper kept for diagnostics,
    exercised by the unit tests below it, and read by offline measurement
    scripts. Zero production callers is its intended state, not an oversight.

    *vectors* is the aggregate rows of ONE group -- L2-normalised, one row per
    member -- and not the index. A Gram matrix is n x n, so this is meant to be
    called per group, where n is a group size, never over a whole population.

    Why this exists: cohesion is the mean cosine to the group's CENTRE, which
    is a radius, not a diameter. Every member can sit inside the cap of radius
    ``arccos(C)`` and two of them still be ``2 * arccos(C)`` apart. ``min`` is
    that group's actual diameter, expressed as a cosine: the loosest pair it
    contains.

    ``mean`` is the same question without a centre -- the mean cosine over
    every pair -- and is the column that says whether cohesion is flattering
    the group.

    ``min`` is an extreme-value statistic, so it falls as a group grows for
    reasons that have nothing to do with purity: more pairs, more chances at a
    low one. It is only readable next to the member count, which is why a
    diagnostic always prints the two together.

    ``pairs`` is ``n * (n - 1) / 2``. Fewer than two members has no pair at
    all, so ``min`` and ``mean`` are ``None`` and ``argmin`` is ``(-1, -1)``:
    the same convention as :func:`shape_map_cohesion_frontier`, which reports
    no cohesion for a group of one rather than inventing 1.0.

    ``argmin`` is the ``(i, j)``, ``i < j``, of the loosest pair, taken in
    ``np.triu_indices`` order so a tie resolves to the same pair every time.

    ``min`` and ``mean`` are rounded to four decimals, like
    :func:`_shape_map_cohesion`: they are read in a column beside cohesion and
    a different rounding would make that column incomparable.

    Raises ``ValueError`` when a row is not a unit vector (tolerance 1e-3).
    Re-normalising quietly would be worse than failing: the numbers would still
    look like cosines while no longer being the ones cohesion was computed
    from, and the measurement would lie without saying so. The message names
    the row and its norm, never the vector's contents.
    """
    import numpy as np

    mat = np.asarray(vectors, dtype="float32")
    if mat.ndim != 2:
        raise ValueError(f"expected a 2-D array of rows, got ndim {mat.ndim}")
    n = int(mat.shape[0])
    if n:
        norms = np.linalg.norm(mat, axis=1)
        bad = np.nonzero(np.abs(norms - 1.0) > 1e-3)[0]
        if bad.size:
            row = int(bad[0])
            raise ValueError(
                f"row {row} is not a unit vector (norm {float(norms[row]):.6f}); "
                f"{int(bad.size)} of {n} rows are not normalised"
            )
    if n < 2:
        return {"pairs": 0, "min": None, "mean": None, "argmin": (-1, -1)}

    gram = mat @ mat.T
    iu = np.triu_indices(n, k=1)
    sims = gram[iu]
    worst = int(np.argmin(sims))
    return {
        "pairs": int(sims.size),
        "min": round(float(sims[worst]), 4),
        "mean": round(float(sims.mean()), 4),
        "argmin": (int(iu[0][worst]), int(iu[1][worst])),
    }


def _shape_map_kmeans_init_method(faiss, name):
    """The FAISS ``init_method`` constant for *name*, or ``None``.

    Older FAISS builds have no ``ClusteringInitMethod_*`` constants at all;
    there the caller simply omits ``init_method`` and gets FAISS's own random
    initialisation, which is what ``"random"`` means anyway.
    """
    attr = {
        "random": "ClusteringInitMethod_RANDOM",
        "kmeans++": "ClusteringInitMethod_KMEANS_PLUS_PLUS",
        "afkmc2": "ClusteringInitMethod_AFK_MC2",
    }.get(name)
    if attr is None:
        return None
    return getattr(faiss, attr, None)


def _shape_map_kmeans_signature() -> tuple:
    """Cache-key fragment for the k-means settings; empty when all are default.

    These four settings change the ASSIGNMENT, and they are read from the
    environment on every call, so they can change without a restart -- putting
    them in the cache keys is the only thing stopping a changed setting from
    being answered out of the cache built under the old one. Empty at the
    defaults so existing entries, and entries written before these knobs
    existed, still hit.
    """
    mppc = _shape_map_kmeans_max_points_per_centroid()
    nredo = _shape_map_kmeans_nredo()
    spherical = _shape_map_kmeans_spherical()
    init = _shape_map_kmeans_init()
    if (
        mppc == _KMEANS_MAX_POINTS_PER_CENTROID_DEFAULT
        and nredo == _KMEANS_NREDO_DEFAULT
        and spherical == _KMEANS_SPHERICAL_DEFAULT
        and init == _KMEANS_INIT_DEFAULT
    ):
        return ()
    return (("km", mppc, nredo, bool(spherical), init),)


def _shape_map_run_kmeans(Xk, k) -> tuple:
    """Run deterministic FAISS k-means on ``Xk`` and return ``(assign, inertia)``.

    ``Xk`` is the ``(N, d)`` clustering matrix (the full-dimensional aggregate).
    *assign* is an ``int64`` array of length N with each row's cluster id;
    *inertia* is the within-cluster sum of squared distances to the assigned
    centroid (the FAISS L2 search distances are already squared). ``k`` is
    clamped to ``[1, N]``; ``k <= 1`` short-circuits to a single cluster whose
    inertia is the total spread around the global mean.

    ``niter`` and ``seed`` are fixed so a given index and k always produce the
    same map. The remaining FAISS settings come from the environment (see
    :func:`_shape_map_kmeans_signature`) and default to FAISS's own values.
    """
    import numpy as np

    N = int(Xk.shape[0])
    k = max(1, min(int(k), N))
    if k <= 1:
        center = Xk.mean(axis=0, keepdims=True)
        inertia = float(np.sum((Xk - center) ** 2))
        return (np.zeros(N, dtype=np.int64), inertia)
    import faiss

    max_points = _shape_map_kmeans_max_points_per_centroid()
    nredo = _shape_map_kmeans_nredo()
    spherical = _shape_map_kmeans_spherical()
    init = _shape_map_kmeans_init()
    # Only when something has been overridden: a screenshot of a map has to be
    # matchable to the settings it was built with, and a line per default build
    # would drown that out.
    if _shape_map_kmeans_signature():
        logger.info(
            "Shape Map k-means: k=%d max_points_per_centroid=%d nredo=%d spherical=%s init=%s",
            k, max_points, nredo, spherical, init,
        )

    kwargs = {}
    # At the default, init_method is not passed at all, so the call is the one
    # this has always made and every map built so far reproduces byte for byte.
    if init != _KMEANS_INIT_DEFAULT:
        method = _shape_map_kmeans_init_method(faiss, init)
        if method is None:
            logger.warning(
                "HOOPS_AI_SHAPE_MAP_KMEANS_INIT=%s: this FAISS build has no "
                "ClusteringInitMethod_*; using its default initialisation",
                init,
            )
        else:
            kwargs["init_method"] = method

    km = faiss.Kmeans(
        Xk.shape[1], k, niter=20, seed=42, verbose=False,
        max_points_per_centroid=max_points,
        nredo=nredo,
        spherical=spherical,
        **kwargs,
    )
    km.train(Xk)
    dist, a = km.index.search(Xk, 1)
    assign = a.reshape(-1).astype(np.int64)
    inertia = float(dist.reshape(-1).sum())
    return (assign, inertia)


def _shape_map_elbow_k(ks, inertias) -> int:
    """Pick the elbow of an inertia-vs-k curve (Kneedle "distance to line").

    Normalises both axes to ``[0, 1]`` and returns the ``k`` whose point is
    farthest from the straight line joining the first and last candidates. For
    the convex, decreasing inertia curve k-means produces, that farthest point
    is the "elbow" where adding clusters stops paying off. Unlike the
    Calinski-Harabasz variance ratio -- which is biased toward ``k = 2`` on
    large, high-dimensional corpora that split into two dominant super-clusters
    -- the elbow tends to land on a mid-range ``k`` better suited to a browsing
    map. Uses only the scalar inertia per ``k``, so it is negligible on top of
    the k-means runs.
    """
    import numpy as np

    ks = np.asarray(ks, dtype=np.float64)
    ys = np.asarray(inertias, dtype=np.float64)
    n = len(ks)
    if n <= 2:
        return int(ks[0])

    x = (ks - ks[0]) / (ks[-1] - ks[0])
    yr = float(ys.max() - ys.min())
    if yr <= 0.0:
        return int(ks[0])
    y = (ys - ys.min()) / yr

    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    denom = float((dx * dx + dy * dy) ** 0.5)
    if denom <= 0.0:
        return int(ks[0])
    dist = np.abs(dy * (x - x[0]) - dx * (y - y[0])) / denom
    return int(ks[int(np.argmax(dist))])


def _shape_map_choose_optimal_k(Xk, N, kmin: int = 2, kmax: int = 24) -> tuple:
    """Scan ``k`` in ``[kmin, min(kmax, N-1)]`` and keep the elbow clustering.

    Runs k-means for each candidate ``k``, records its inertia, and selects the
    elbow of the inertia curve (see :func:`_shape_map_elbow_k`). Returns
    ``(assign, k)`` for the chosen count. This is the ``n_clusters == -1``
    ("Optimal") mode; it is data-driven, unlike the ``n_clusters == 0``
    point-count heuristic. Deterministic via the fixed k-means seed.
    """
    kmax = min(int(kmax), N - 1)
    kmin = int(kmin)
    if kmax < kmin:
        k = max(1, min(N, kmin))
        assign, _ = _shape_map_run_kmeans(Xk, k)
        return (assign, k)

    ks = list(range(kmin, kmax + 1))
    assigns = {}
    inertias = []
    for k in ks:
        assign, inertia = _shape_map_run_kmeans(Xk, k)
        assigns[k] = assign
        inertias.append(inertia)

    best_k = _shape_map_elbow_k(ks, inertias)
    return (assigns[best_k], best_k)


def _shape_map_centroid_sims(agg, assign, skip_outliers: bool = False):
    """Cosine similarity of every row of *agg* to its own cluster's centre.

    Centroids are recomputed from *assign* rather than taken from
    ``faiss.Kmeans.centroids``: they are exact for the final assignment, whereas
    FAISS's are those of the last training iteration before ``niter`` ran out.
    It also keeps :func:`_shape_map_run_kmeans`'s return value unchanged.

    ``agg`` rows are already L2-normalised by :func:`_shape_map_project`, so
    normalising the centroid makes the dot product a cosine directly -- the same
    similarity the search path reports, so a number here means the same thing as
    a score there.

    Rows whose cluster has a degenerate (zero-norm) centre, and rows skipped by
    *skip_outliers*, keep the neutral value ``1.0``: there is no centre to
    measure them against, so nothing should be concluded from them.
    """
    import numpy as np

    A = np.ascontiguousarray(agg, dtype=np.float32)
    assign = np.asarray(assign)
    sims = np.ones(A.shape[0], dtype=np.float32)
    for c in np.unique(assign):
        if skip_outliers and int(c) < 0:
            continue
        member = assign == c
        centroid = A[member].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm <= 0.0:
            continue
        sims[member] = A[member] @ (centroid / norm)
    return sims


def _shape_map_cohesion(agg, assign) -> dict[int, float]:
    """Mean cosine similarity to the centre, per cluster: ``{cluster_id: float}``.

    A cluster near 1.0 holds parts that all look like its centre; a low one is a
    mixture, and is the cluster worth drilling into. Raising the cluster count is
    a blunter tool -- k-means minimises total inertia, so it will happily split a
    cluster that was already pure -- which is why this number is surfaced in the
    legend instead.

    Cluster ``-1`` (outliers) is excluded from the result AND from the centroids:
    leaving the outliers in would drag every mean back down and hide the effect
    of the very threshold that separated them. Computed regardless of whether the
    outlier pass ran -- one more pass over the same matrix is negligible next to
    the SVD.
    """
    import numpy as np

    assign = np.asarray(assign)
    sims = _shape_map_centroid_sims(agg, assign, skip_outliers=True)
    out: dict[int, float] = {}
    for c in np.unique(assign):
        c = int(c)
        if c < 0:
            continue
        out[c] = round(float(sims[assign == c].mean()), 4)
    return out


def _shape_map_centres(agg, assign, file_ids) -> dict[int, str]:
    """Centre part of each cluster: ``{cluster_id: file_id}``.

    The centre part is the REAL member whose vector is closest to its cluster's
    k-means centroid. The centroid itself is the arithmetic mean of the members
    and is not a part that exists, so this picks the existing part that stands
    in for it -- purely a post-processing step; the clustering is still
    ``faiss.Kmeans`` and is not touched.

    This is NOT a k-medoids medoid. A medoid minimises the sum of distances to
    every other member of its cluster, which is a different definition; the two
    often coincide but they are not the same thing.

    Similarity is measured against the full-dimensional centroid, i.e. in the
    space k-means actually clustered in -- not the 3-D centroid of the projected
    map the client can compute for itself.

    Cluster ``-1`` (outliers) gets no entry: it is a display category, not a
    cluster, and has no centre.
    """
    import numpy as np

    assign = np.asarray(assign)
    sims = _shape_map_centroid_sims(agg, assign, skip_outliers=True)
    out: dict[int, str] = {}
    for c in np.unique(assign):
        c = int(c)
        if c < 0:
            continue
        # Restrict to this cluster's rows BEFORE taking the argmax. The skipped
        # outlier rows (and rows of a degenerate centroid) keep the neutral value
        # 1.0, so an argmax over the whole array would happily return an outlier.
        member = np.flatnonzero(assign == c)
        if member.size == 0:
            continue
        # argmax takes the first maximum, so an all-neutral cluster resolves to
        # its first member -- arbitrary, but deterministic.
        best = int(member[int(np.argmax(sims[member]))])
        out[c] = str(file_ids[best])
    return out


def _shape_map_outlier_ceiling(sims) -> float:
    """Highest outlier threshold that still leaves this clustering readable.

    A point survives the outlier pass when ``sims >= threshold``, so the
    second-largest similarity is exactly the largest threshold that keeps two
    points: at that value the best and second-best points are both kept and
    everything else is separated. One point (or none) is not a map, hence two
    rather than one.

    This costs nothing to find. *sims* is computed from the k-means assignment
    BEFORE the outlier pass is applied, so it does not depend on the threshold
    at all -- the ceiling is one O(N) look at an array that already exists, with
    no re-clustering.

    Returned unrounded, straight out of the array: rounding up would break the
    very guarantee it exists to make. ``0.0`` when there are fewer than two
    points, meaning there is no threshold worth applying.
    """
    import numpy as np

    sims = np.asarray(sims)
    if sims.size < 2:
        return 0.0
    return float(np.partition(sims, -2)[-2])


def _shape_map_outlier_assign(agg, assign, outlier_threshold: float):
    """Move points that are far from their cluster's centroid into cluster ``-1``.

    k-means assigns every point to the NEAREST of k centroids; it never asks
    whether that centroid is actually close. A part with no neighbours is
    therefore absorbed into whichever cluster happens to be least far away, and
    lands in the legend as an ordinary member of it. This pass adds the absolute
    test k-means does not make: recompute each cluster's centroid, take the
    cosine similarity of every point to its own centroid, and tag anything below
    *outlier_threshold* as ``-1``.

    Cosine is the measure used because ``agg`` rows are already L2-normalised by
    :func:`_shape_map_project`, so normalising the centroid makes the dot
    product a cosine directly -- the same similarity the search path reports, so
    a threshold here means the same thing as a score there. The similarities
    themselves come from :func:`_shape_map_centroid_sims`.

    A request above what this clustering can support is CLAMPED, not discarded.
    This pass used to throw itself away and hand back the original clustering
    whenever fewer than two points survived, which from the outside looked like
    the strictest setting quietly producing the same map as Off -- a
    non-monotonic jump with nothing to explain it. Instead the threshold is
    lowered to :func:`_shape_map_outlier_ceiling` and applied there, so raising
    the setting separates steadily more and then simply stops changing.

    Returns ``(assignment, effective_threshold)``. The effective threshold is
    the value actually applied: ``0.0`` when the pass did not run, otherwise
    ``min(requested, ceiling)``, which the caller reports so the UI can say the
    request was capped.
    """
    import numpy as np

    threshold = float(outlier_threshold or 0.0)
    # Off: return before computing sims, so the disabled case costs no extra
    # pass over the vectors.
    if threshold <= 0.0:
        return assign, 0.0

    assign = np.asarray(assign)
    sims = _shape_map_centroid_sims(agg, assign)

    ceiling = _shape_map_outlier_ceiling(sims)
    # Fewer than two points: there is no layout to read whatever we do.
    if ceiling <= 0.0:
        return assign, 0.0

    effective = min(threshold, ceiling)
    out = np.where(sims < effective, np.int64(-1), assign.astype(np.int64))
    return out, effective


def _shape_map_cluster(
    agg, coords_full, file_ids, n_clusters: int, dims: int,
    outlier_threshold: float = 0.0,
    clustering: str = _CLUSTERING_KMEANS,
    linkage_z=None,
    ward_parent=None,
    ward_height=None,
    timings=None,
) -> tuple:
    """Scale the projection into the [-1, 1] cube and cluster the aggregate.

    ``clustering`` selects the algorithm: ``"kmeans"`` (the default, FAISS
    k-means) or ``"ward"`` (scipy's agglomerative hierarchical clustering). For
    Ward, ``n_clusters`` is a height at which to cut one tree rather than a
    number of centroids to fit, which is what makes its cuts nest: the 24
    clusters of a Ward map are an exact refinement of its 8, so raising the
    count splits families instead of redrawing them. *linkage_z* is a tree
    already built for these rows; without one it is built here, which is tens of
    seconds (see :func:`_shape_map_linkage_matrix`).

    *ward_parent* is ``(parent_tree, rows)`` from
    :func:`_shape_map_ward_subset_node`, and changes what Ward means for a
    drill-down: instead of building a new tree over the subset, the PARENT's
    tree is cut lower down (see :func:`_shape_map_ward_descend`). The clusters
    that come back are then an exact refinement of the ones the parent view
    showed, which is what walking down a hierarchy should mean. It is ignored
    for k-means.

    *ward_height* replaces the count with a height: the tree is cut once at that
    distance instead of being searched for a cut that yields *n_clusters*
    pieces, so different branches come back split into different numbers of
    clusters (see :func:`_shape_map_ward_cut_height`). When it is given,
    ``n_clusters`` is ignored. It is ignored for k-means.

    Reuses a projection produced by :func:`_shape_map_project`. Returns
    ``(points, k, cohesion, centres, outlier_threshold_effective)`` where
    *points* is a list of ``(file_id, x, y, z, cluster, sim)`` tuples -- *sim*
    being the point's cosine similarity to its cluster's centroid, 1.0 for the
    Outliers group -- *k* is the
    number of clusters actually used, *cohesion* is ``{cluster_id: mean cosine
    to centre}`` from :func:`_shape_map_cohesion`, *centres* is
    ``{cluster_id: file_id}`` of each cluster's centre part from
    :func:`_shape_map_centres` and *outlier_threshold_effective* is the outlier
    threshold actually applied. Cheap relative
    to the SVD, so it is safe to re-run whenever *n_clusters* or *dims* change
    without recomputing the projection.

    *timings*, when given, is a dict this fills with two entries, in seconds:
    ``"assign"`` -- the clustering itself, which is one ``fcluster`` call for
    Ward and the k-means fit otherwise -- and ``"stats"``, everything after it
    (the outlier pass, cohesion, the canonical renumbering, the centre parts,
    the per-point similarities and the point tuples). The split is there
    because the two answer different questions: the first is the algorithm, the
    second is the fixed per-point work that follows whichever algorithm ran.

    ``n_clusters`` selects the cluster count:

    * ``> 0`` -- use exactly that many clusters.
    * ``0``   -- legacy point-count heuristic ``min(max(2, N // 50), 24)``.
    * ``-1``  -- "Optimal": scan ``k`` in ``[2, min(24, N-1)]`` and pick the
      elbow of the inertia curve (see :func:`_shape_map_choose_optimal_k`).

    ``outlier_threshold`` > 0 additionally splits off points whose cosine
    similarity to their own centroid falls below it, as cluster ``-1`` (see
    :func:`_shape_map_outlier_assign`). A request above what this clustering can
    support is capped, and the value that was really applied comes back as
    *outlier_threshold_effective*. ``0.0`` disables the pass and leaves the
    result bit-for-bit as it was before outliers existed. The returned *k* is
    always the count k-means used: the outlier group is a display category, not
    a cluster, so it is not counted. *cohesion* and *centres* are produced
    either way.
    """
    import numpy as np

    N = len(file_ids)
    if N == 0:
        return ([], 0, {}, {}, 0.0)

    avail = int(coords_full.shape[1]) if coords_full.ndim == 2 else 0
    d = 3 if int(dims) >= 3 else 2
    d = max(1, min(d, avail, N)) if avail else 1

    if avail:
        sub = np.asarray(coords_full[:, :d], dtype=np.float32)
    else:
        sub = np.zeros((N, 1), dtype=np.float32)

    span = float(np.max(np.abs(sub))) if sub.size else 1.0
    if span <= 0.0:
        span = 1.0
    coords = sub / span
    if coords.shape[1] < 3:
        coords = np.concatenate(
            [coords, np.zeros((N, 3 - coords.shape[1]), dtype=np.float32)], axis=1
        )

    Xk = np.ascontiguousarray(agg, dtype=np.float32)
    clustering = _normalize_map_clustering(clustering)
    k_req = int(n_clusters)
    assign_started = time.perf_counter()
    if clustering == _CLUSTERING_WARD:
        if ward_height is not None:
            # A height is the cut itself, so there is nothing to search for:
            # one fcluster call on whichever tree is in play.
            tree = ward_parent[0] if ward_parent is not None else (
                linkage_z if linkage_z is not None else _shape_map_linkage_matrix(Xk)
            )
            rows = ward_parent[1] if ward_parent is not None else None
            assign = _shape_map_run_ward_at_height(tree, ward_height, rows)
        else:
            # The "Optimal" scan picks k from the elbow of the k-means inertia
            # curve, which is not defined for an agglomerative tree. The UI does
            # not offer it (see api.js), so -1 is treated as "no count given" and
            # resolved by the same point-count heuristic as 0.
            k = k_req if k_req > 0 else min(max(2, N // 50), 24)
            k = max(1, min(k, N))
            assign = None
            if ward_parent is not None:
                assign = _shape_map_ward_descend(ward_parent[0], ward_parent[1], k)
            if assign is None:
                tree = linkage_z if linkage_z is not None else _shape_map_linkage_matrix(Xk)
                assign = _shape_map_run_ward(tree, k)
        # maxclust returns AT MOST k clusters, and descending a parent node can
        # overshoot k on tied merge distances, so the count that is reported --
        # and that the renumbering below iterates over -- has to come from the
        # assignment, not from the request.
        k = max(1, int(assign.max()) + 1)
    elif k_req == -1:
        assign, k = _shape_map_choose_optimal_k(Xk, N)
    else:
        if k_req > 0:
            k = k_req
        else:
            k = min(max(2, N // 50), 24)
        k = max(1, min(k, N))
        assign, _ = _shape_map_run_kmeans(Xk, k)

    if timings is not None:
        timings["assign"] = time.perf_counter() - assign_started
    stats_started = time.perf_counter()

    # One extra pass over the same full-dimensional vectors, after every branch
    # above, so the threshold behaves identically whichever way k was chosen.
    assign, outlier_effective = _shape_map_outlier_assign(Xk, assign, outlier_threshold)
    cohesion = _shape_map_cohesion(Xk, assign)

    # FAISS numbers clusters by where its initial centroids happened to land,
    # which carries no meaning and shifts whenever the training rows do.
    # Renumber canonically, so that the legend reads in a useful order and the
    # same partition always gets the same ids, labels and palette slots.
    #
    # The key is the legend's own two columns, so that reading the legend from
    # the top means working through the biggest families first: MEMBER COUNT
    # descending, then COHESION (the mean similarity printed beside the count)
    # descending. The cluster's mean projected position is kept as a canonical
    # last resort, because cohesion is rounded to four decimals and exact ties
    # are common enough to need a tie-break that does not fall back on FAISS's
    # arbitrary numbering.
    #
    # Both numbers are taken AFTER the outlier pass -- which is why this runs
    # here and no longer before it. They are the values the legend displays,
    # and ordering by the pre-outlier ones would show a legend that is not
    # sorted by its own columns. The consequence is worth knowing: moving the
    # Outliers slider changes the member counts, so it can also change the
    # cluster numbering and the palette slots.
    #
    # assign may now contain -1 (the Outliers group). That is a display
    # category, not a cluster, and is never renumbered.
    counts = np.zeros(k, dtype=np.int64)
    centres_xyz = np.zeros((k, 3), dtype=np.float64)
    for c in range(k):
        hit = np.flatnonzero(assign == c)
        counts[c] = hit.size
        if hit.size:
            centres_xyz[c] = coords[hit].mean(axis=0)

    # Clusters FAISS left empty, and clusters the outlier pass emptied, have
    # neither a count nor a position. Park them last, in their original order,
    # where they cannot displace a populated cluster's palette slot.
    def _cluster_sort_key(c):
        if counts[c]:
            return (0, -int(counts[c]), -float(cohesion.get(c, 0.0)),
                    float(centres_xyz[c][0]), float(centres_xyz[c][1]),
                    float(centres_xyz[c][2]))
        return (1, 0, 0.0, 0.0, 0.0, float(c))

    order = sorted(range(k), key=_cluster_sort_key)
    remap = np.empty(k, dtype=np.int64)
    for new_id, old_id in enumerate(order):
        remap[old_id] = new_id
    assign = np.where(assign >= 0, remap[np.clip(assign, 0, k - 1)], np.int64(-1))
    cohesion = {int(remap[c]): v for c, v in cohesion.items()}
    centres = _shape_map_centres(Xk, assign, file_ids)
    # The same similarities cohesion is averaged from, kept per point this time:
    # the map coordinates are a 3-D projection of 2048 dimensions, and carrying
    # the real similarity lets the client show how far the two disagree. Rows in
    # the Outliers group keep the neutral 1.0 skip_outliers leaves behind; they
    # have no centre to be measured against and are not coloured by it.
    sims = _shape_map_centroid_sims(Xk, assign, skip_outliers=True)

    points = [
        (
            str(file_ids[i]),
            float(coords[i, 0]),
            float(coords[i, 1]),
            float(coords[i, 2]),
            int(assign[i]),
            float(sims[i]),
        )
        for i in range(N)
    ]
    if timings is not None:
        timings["stats"] = time.perf_counter() - stats_started
    return (points, int(k), cohesion, centres, float(outlier_effective))


def _compute_shape_map_points(
    name: str, n_clusters: int, dims: int, only_ids=None,
    projection: str = _PROJECTION_PCA,
    clustering: str = _CLUSTERING_KMEANS,
) -> dict[str, Any]:
    """Uncached end-to-end layout for a named index (load, project, cluster).

    Returns ``{"count", "clusters", "points", "cohesion", "centres",
    "outlier_threshold_effective"}`` where
    *points* is a list of ``(file_id, x, y, z, cluster, sim)`` tuples, *cohesion*
    maps a cluster id to its mean cosine similarity to its own centre, and
    *centres* maps a cluster id to the file id of its centre part. Caching and
    the per-subset build lock are layered on top of this in the endpoint path.

    *projection* selects how the points are placed (``"pca"`` or ``"umap"``);
    the clustering is unaffected either way. *clustering* selects the algorithm
    (``"kmeans"`` or ``"ward"``). This path holds no caches at all, so a Ward
    request builds the tree on every call -- which is why the endpoint path
    above, and not this one, is what serves ``/map``.
    """
    ids, values = _load_index_body_batch(name)
    file_ids, coords_full, agg = _shape_map_project(
        ids, values, only_ids, projection=projection
    )
    points, k, cohesion, centres, outlier_effective = _shape_map_cluster(
        agg, coords_full, file_ids, n_clusters, dims, clustering=clustering
    )
    return {
        "count": len(points),
        "clusters": k,
        "points": points,
        "cohesion": cohesion,
        "centres": centres,
        "outlier_threshold_effective": outlier_effective,
    }


def _shape_map_build_lock(key: tuple) -> "threading.Lock":
    """Return the per-subset build lock, creating it on first use."""
    with _shape_map_build_locks_mutex:
        lock = _shape_map_build_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _shape_map_build_locks[key] = lock
        return lock


def _shape_map_filemeta(name: str) -> dict[str, dict[str, Any]]:
    """Return ``{file_id: {"filename", "kind"}}`` for a named index.

    Read once per cold build (kept in the aggregate cache entry) so a warm map
    never rescans metadata. One pass over ``iter_metadata()``; first row per
    file wins.
    """
    lock = _get_index_lock(name)
    with lock:
        vs = index_registry._load_named_index(name)
        out: dict[str, dict[str, Any]] = {}
        for raw in vs.iter_metadata():
            md = raw if isinstance(raw, dict) else (getattr(raw, "metadata", None) or {})
            fid = md.get("file_id")
            if fid is None or fid in out:
                continue
            out[str(fid)] = {
                "filename": md.get("filename"),
                "kind": md.get("kind") or "part",
            }
    return out


def _shape_map_legend(
    points: list, tags_map: dict, color_by: str = "cluster"
) -> list[dict[str, Any]]:
    """Build the server-side legend for the mapped points, following *color_by*.

    Returns a list of ``{"label", "kind", "count", "rgb"}`` entries whose colours
    match the baked point cloud (both go through
    :func:`_shape_map_color_assignment`). ``color_by="cluster"`` keys by k-means
    cluster (``Cluster N``); ``color_by="tag"`` keys by tag (most frequent first)
    plus an ``Untagged`` row. This is why the legend needs the colour mode -- the
    colours are baked into the ``.scs``, so the legend must not invent its own.
    """
    clusters = [int(p[4]) for p in points]
    tags_lists = [list(tags_map.get(p[0]) or []) for p in points]
    _rgbs, legend = _shape_map_color_assignment(clusters, tags_lists, color_by)
    return legend


def _shape_map_build_payload(
    name: str,
    points: list,
    k: int,
    dims: int,
    filemeta: dict,
    tags_map: dict,
    include_tags: bool,
    color_by: str = "cluster",
    cohesion: Optional[dict] = None,
    centres: Optional[dict] = None,
    outlier_threshold_effective: float = 0.0,
) -> dict[str, Any]:
    """Assemble the response payload from numeric points + file metadata.

    Points are reordered so that each display category (cluster / tag / untagged)
    occupies a CONTIGUOUS run, and a ``groups`` list describes those runs
    ``{key, label, rgb, count, offset, node_name}``. This mirrors the baked
    ``.scs``, which authors one node per category in the same order: a viewer maps
    a picked ``(node, index-within-node)`` straight to
    ``groups[node].offset + index`` in this points array. A server-side *legend*
    is always included, coloured for *color_by* so its swatches match the point
    cloud. Per-point ``tags`` are added only when *include_tags* is set.

    *cohesion* (``{cluster_id: mean cosine to centre}``) is attached to the
    legend and group entries of the categories it describes, as ``cohesion``.
    Only ``cluster:N`` categories have one; ``outliers``, ``untagged``,
    ``other`` and every tag category get ``None``, because none of them is a
    k-means cluster and no centre-distance statement applies to them. Colouring
    by tag therefore leaves every entry at ``None``.

    *centres* (``{cluster_id: file_id}``) rides along the same way, as
    ``centre_file_id``: the real part closest to that cluster's k-means centroid
    (see :func:`_shape_map_centres`). Same rule -- only ``cluster:N`` categories
    carry one.

    *outlier_threshold_effective* is a property of the whole map rather than of
    any one category, so it sits at the top level of the payload under that
    name: the outlier threshold actually applied, which is below the requested
    one when the request exceeded what this clustering can support.
    """
    color_by = "tag" if str(color_by).lower() == "tag" else "cluster"
    clusters = [int(p[4]) for p in points]
    tags_lists = [list(tags_map.get(p[0]) or []) for p in points]
    cat_index, categories = _shape_map_categorize(clusters, tags_lists, color_by)

    # Stable reorder: category order first, original order within a category. The
    # SCS tool writes points in this exact order, so pointId within a node maps
    # back to (group offset + pointId) here.
    order = sorted(range(len(points)), key=lambda i: (cat_index[i], i))

    pts: list[dict[str, Any]] = []
    for i in order:
        fid, x, y, z, cluster, sim = points[i]
        meta = filemeta.get(fid) or {}
        entry: dict[str, Any] = {
            "file_id": fid,
            "filename": meta.get("filename"),
            "x": x,
            "y": y,
            "z": z,
            "cluster": cluster,
            "sim": float(sim),
        }
        if include_tags:
            entry["tags"] = list(tags_map.get(fid) or [])
        pts.append(entry)

    # Groups follow the reordered points: one contiguous run per category, in the
    # same order as ``categories`` (which is also the legend order). node_name is
    # keyed by category index, not generation order, so the viewer matches by name.
    from collections import Counter
    cat_counts = Counter(cat_index)

    def _cohesion_of(cat: dict):
        """Cohesion for a category, or None when it is not a k-means cluster."""
        key = str(cat.get("key") or "")
        if not cohesion or not key.startswith("cluster:"):
            return None
        try:
            value = cohesion.get(int(key.split(":", 1)[1]))
        except (TypeError, ValueError):
            return None
        return None if value is None else float(value)

    def _centre_of(cat: dict):
        """Centre part's file id for a category, or None when it is not a cluster."""
        key = str(cat.get("key") or "")
        if not centres or not key.startswith("cluster:"):
            return None
        try:
            value = centres.get(int(key.split(":", 1)[1]))
        except (TypeError, ValueError):
            return None
        return None if value is None else str(value)

    groups: list[dict[str, Any]] = []
    offset = 0
    for ci, cat in enumerate(categories):
        count = int(cat_counts.get(ci, 0))
        if count == 0:
            continue
        groups.append({
            "key": cat["key"],
            "label": cat["label"],
            "rgb": list(cat["rgb"]),
            "count": count,
            "offset": int(offset),
            "node_name": f"group_{ci:04d}",
            "cohesion": _cohesion_of(cat),
            "centre_file_id": _centre_of(cat),
        })
        offset += count

    legend = [
        {"label": c["label"], "kind": c["kind"], "count": int(c["count"]),
         "rgb": list(c["rgb"]), "cohesion": _cohesion_of(c),
         "centre_file_id": _centre_of(c)}
        for c in categories
    ]

    return {
        "index": name,
        "count": len(points),
        "clusters": int(k),
        "dims": 3 if int(dims) >= 3 else 2,
        "points": pts,
        "groups": groups,
        "legend": legend,
        "outlier_threshold_effective": float(outlier_threshold_effective or 0.0),
    }


def _normalize_map_kind(kind: Optional[str]) -> str:
    """Normalize a map kind filter to ``"any"`` / ``"part"`` / ``"assembly"``.

    Unknown values collapse to ``"any"`` (no restriction) so a bad query never
    silently empties the map.
    """
    k = str(kind or "any").lower()
    return k if k in ("part", "assembly") else "any"


def _compose_keep(a, b):
    """AND two ``(file_id) -> bool`` predicates; ``None`` means "keep all".

    Returns ``None`` when both are ``None`` (so callers can skip filtering
    entirely), otherwise a predicate that passes only when every non-None
    predicate passes. Used to combine the tag filter with the kind filter.
    """
    if a is None:
        return b
    if b is None:
        return a
    return lambda fid: a(fid) and b(fid)


def _shape_map_kind_predicate(filemeta: dict, kind: str):
    """Return a ``(file_id) -> bool`` keeping only *kind*, or ``None`` for "any".

    ``kind`` is derived at index time (``>= _MIN_BODIES_FOR_ASSEMBLY`` body rows
    counts as an assembly); ``filemeta`` carries it per file id. A file id absent
    from the metadata is treated as a part, matching ``_shape_map_filemeta``'s
    own default, so a part is never dropped by a missing row.
    """
    kind = _normalize_map_kind(kind)
    if kind == "any":
        return None

    def keep(file_id: str) -> bool:
        meta = filemeta.get(str(file_id)) or {}
        return (meta.get("kind") or "part") == kind

    return keep


def _shape_map_subset_predicate(subset_ids: Optional[list[str]]):
    """Return a ``(file_id) -> bool`` keeping only *subset_ids*, or ``None``.

    ``None`` (no subset) means "keep all", so it composes with the tag and kind
    predicates through :func:`_compose_keep` without a special case.
    """
    if subset_ids is None:
        return None
    allow = set(subset_ids)

    def keep(file_id: str) -> bool:
        return str(file_id) in allow

    return keep


def _normalize_map_subset(only_file_ids) -> Optional[list[str]]:
    """Canonicalise a requested subset to a deduplicated, sorted id list.

    Returns ``None`` for "no subset" (``None`` or an empty list) so callers and
    cache keys treat the whole-index case exactly as they did before subsets
    existed. Sorting makes the identity of a subset independent of the order the
    client happened to send, so the same cluster drilled twice reuses one cache
    entry.
    """
    if not only_file_ids:
        return None
    return sorted({str(x) for x in only_file_ids})


def _register_map_subset(only_file_ids) -> Optional[str]:
    """Store a subset and return the short id that refers to it.

    The id is the first 16 hex characters of the SHA-256 of the canonical id
    list. It travels in a URL, so it is truncated; 64 bits is far beyond what
    an 8-entry LRU could ever collide on, and a collision would only serve the
    wrong (still valid) subset rather than leak anything.
    """
    ids = _normalize_map_subset(only_file_ids)
    if ids is None:
        return None
    digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16]
    with _shape_map_cache_mutex:
        _shape_map_subsets[digest] = ids
        _shape_map_subsets.move_to_end(digest)
        while len(_shape_map_subsets) > _SHAPE_MAP_SUBSET_CACHE_MAX:
            _shape_map_subsets.popitem(last=False)
    return digest


def _resolve_map_subset(subset_id: Optional[str]) -> Optional[list[str]]:
    """Return the id list for *subset_id*, or ``None`` when it is not held.

    ``None`` is also returned for a falsy *subset_id*; callers distinguish the
    two cases themselves (no subset asked for vs. an expired one).
    """
    if not subset_id:
        return None
    with _shape_map_cache_mutex:
        ids = _shape_map_subsets.get(str(subset_id))
        if ids is None:
            return None
        _shape_map_subsets.move_to_end(str(subset_id))
        return list(ids)


def _normalize_outlier_threshold(value) -> float:
    """Canonicalise the outlier cosine threshold for use in a cache key.

    Clamped to ``[0, 1]`` and rounded to three decimals, because it reaches the
    cache key as a float and two requests a floating-point hair apart describe
    the same map. ``0.0`` means the pass is off.
    """
    try:
        v = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not (v > 0.0):
        return 0.0
    return round(min(v, 1.0), 3)


def _shape_map_subset_sig(
    tag_filter: Optional[dict[str, Any]],
    kind: str = "any",
    subset_id: Optional[str] = None,
) -> str:
    """Deterministic cache signature for a tag + kind subset (``"all"`` = no tag).

    ``kind`` restricts the map to parts or assemblies and, like a tag filter,
    changes the projected subset, so it is part of the signature that keys the
    SVD/result and ``.scs`` caches. An explicit drill-down subset does the same
    and appends ``|ids=<subset_id>``; when there is none the signature is left
    byte-for-byte as it was, so every cache entry written before drill-down
    existed is still addressable.
    """
    kind = _normalize_map_kind(kind)
    if not tag_filter:
        base = "all"
    else:
        tags = ",".join(sorted(tag_filter.get("tags") or []))
        base = f"tags={tags}|mode={tag_filter.get('tag_mode')}|tagged={tag_filter.get('tagged')}"
    sig = f"{base}|kind={kind}"
    if subset_id:
        sig = f"{sig}|ids={subset_id}"
    return sig


def compute_index_shape_map(
    name: str,
    n_clusters: int = 0,
    dims: int = 3,
    tags: Optional[list[str]] = None,
    tag_mode: str = "any",
    tagged: Optional[bool] = None,
    include_tags: bool = False,
    color_by: str = "cluster",
    kind: str = "any",
    only_file_ids: Optional[list[str]] = None,
    outlier_threshold: float = 0.0,
    projection: str = _PROJECTION_PCA,
    clustering: str = _CLUSTERING_KMEANS,
    granularity: Optional[float] = None,
    ward_max: int = _SHAPE_MAP_WARD_K_MAX,
) -> dict[str, Any]:
    """Compute (or serve from cache) the index-based Shape Map for *name*.

    The aggregate + projection are cached per ``(faiss path, mtime_ns, subset,
    projection)`` and reused across ``n_clusters`` / ``dims`` changes; the
    finished
    payload is cached per ``(faiss path, mtime_ns, subset, n_clusters, dims,
    include_tags, color_by, tag ETag)``. A registration write bumps the
    ``.faiss`` mtime and so retires both caches automatically; a tag write
    touches only the sidecar, so the tag ETag in the result key (and, when a tag
    filter is active, in the SVD key) is what retires the tag-dependent map after
    tagging. The heavy SVD runs under a per-subset
    build lock and OUTSIDE the per-index lock, so concurrent identical requests
    build it once and part search on the same index is never stalled by it.

    An optional tag filter (*tags* / *tag_mode* / *tagged*, the same vocabulary
    as search) restricts the map to a subset; per the reference, the projection
    and clustering are recomputed **on the subset only**. *kind* (``"any"`` /
    ``"part"`` / ``"assembly"``) restricts it the same way, so a parts-only map
    projects and clusters on parts alone — assemblies (mean-of-body centroids
    that pull toward the middle) no longer skew the axes or the k-means.

    Raises ``KeyError`` when the index does not exist, ``ValueError`` for an
    invalid tag filter.

    *only_file_ids* narrows the map further, to an explicit set of file ids, and
    combines with the tag filter and *kind* rather than replacing them. This is
    what drill-down uses: because the projection is recomputed on the subset,
    the SVD axes are redrawn **within** that subset, which is a different thing
    from zooming into the parent layout. Ids the index does not hold are
    ignored; the response carries a ``subset_id`` that ``/map/scs`` takes back
    so the point cloud matches the same points in the same order. Raises
    ``ValueError`` when fewer than two parts survive, since a map needs at least
    two points to have any layout at all.

    *outlier_threshold* > 0 splits points whose cosine similarity to their own
    k-means centroid falls below it into an ``Outliers`` group (cluster ``-1``)
    instead of leaving them as nominal members of the nearest cluster. ``0.0``
    (the default) disables it and reproduces the previous behaviour exactly.

    *projection* (``"pca"`` -- the default -- or ``"umap"``) chooses how the
    aggregate vectors are flattened to three dimensions for display. It changes
    where the points are drawn and nothing else: k-means still runs on the full
    aggregate, so the cluster each part belongs to is the same either way. It
    keys both caches, since it decides the coordinates. Raises
    :class:`UmapUnavailable` when ``"umap"`` is asked for and umap-learn is not
    installed.

    *clustering* (``"kmeans"`` -- the default -- or ``"ward"``) chooses the
    algorithm. Unlike *projection* it changes the assignment and not the
    coordinates, so it keys the result cache only. Ward's tree is cached
    separately, keyed by the aggregate alone, so switching the projection does
    not rebuild it. Raises :class:`WardTooLarge` when the map has more points
    than Ward can be run on here.

    *granularity* (0.0 coarsest .. 1.0 finest) cuts Ward's tree at a HEIGHT
    instead of to a count, so branches split into as many clusters as their own
    spread warrants rather than all being forced to the same number. It is
    ignored -- and kept out of the cache key -- for k-means and when omitted, so
    the default request is unchanged down to the bytes. When it is in use the
    height that was applied comes back on the payload as ``ward``.

    *ward_max* is the ceiling that granularity scale is read against: 1.0 means
    this many clusters, or the population when it is smaller. It must be the
    same value ``/map/tree`` was given as ``k_max``, or the tree's leaves and
    the map's clusters stop being the same families. It too is kept out of the
    cache key at its default, so every entry written before it was a parameter
    is still addressable.
    """
    _validate_index_name(name)
    # Timed from here so the line covers the whole call, cache probes included.
    timing = _ShapeMapTiming()
    faiss_path = index_registry._index_faiss_path(name)
    if not faiss_path.exists():
        raise KeyError(f"Index '{name}' does not exist.")
    mtime_ns = faiss_path.stat().st_mtime_ns
    path_key = str(faiss_path)

    tag_filter = normalize_tag_filter(tags, tag_mode, tagged)
    kind = _normalize_map_kind(kind)
    subset_ids = _normalize_map_subset(only_file_ids)
    subset_id = _register_map_subset(subset_ids)
    subset_sig = _shape_map_subset_sig(tag_filter, kind, subset_id)
    tag_keep = tag_filter_predicate(name, tag_filter) if tag_filter else None

    n_clusters = int(n_clusters)
    dims = int(dims)
    include_tags = bool(include_tags)
    color_by = "tag" if str(color_by).lower() == "tag" else "cluster"
    projection = _normalize_map_projection(projection)
    clustering = _normalize_map_clustering(clustering)

    # Checked before anything is computed, so an unavailable dependency is
    # reported as such instead of surfacing as an ImportError from inside the
    # projection after the index has already been read back.
    if projection == _PROJECTION_UMAP and not _shape_map_umap_available():
        raise UmapUnavailable(
            "projection=umap requires umap-learn, which is not installed in "
            "this deployment. Run: pip install umap-learn"
        )

    # Tags live in a sidecar; _write_tags_document swaps only that JSON file and
    # never touches the .faiss, so the .faiss mtime does NOT change when a part
    # is tagged. Without the sidecar ETag in the key a "colour by tag" map (or
    # any tag-filtered / tag-labelled map) keeps serving the pre-tag payload --
    # the symptom being freshly-tagged parts still shown as Untagged. Read the
    # tag document once, up front, and derive both the ETag (for the keys) and
    # the tags_map (for the payload) from that single snapshot so they stay
    # consistent.
    tags_map, tags_etag = _shape_map_tags_snapshot(name)

    # color_by only affects the legend colouring (and, via /map/scs, the baked
    # point colours), never the projection or clustering, so it keys the result
    # cache but not the shared SVD cache below. The tag ETag only matters when
    # the output actually depends on tags: a tag filter (subset membership), tag
    # labels in the payload, or tag colouring. A plain cluster-coloured map is
    # tag-independent, so leaving the ETag out of its key avoids needless
    # rebuilds when an unrelated part is tagged.
    tags_affect_result = bool(tag_filter) or include_tags or color_by == "tag"
    result_tags_etag = tags_etag if tags_affect_result else ""
    result_key = (
        path_key, mtime_ns, subset_sig, n_clusters, dims, include_tags, color_by,
        result_tags_etag,
    )
    # The threshold changes which cluster a point is labelled with, but not the
    # projection, so it keys the result cache and not the SVD cache below --
    # exactly like n_clusters. Appended only when it is in use, so every entry
    # written before outliers existed is still addressable by the same key.
    outlier_threshold = _normalize_outlier_threshold(outlier_threshold)
    if outlier_threshold > 0.0:
        result_key = result_key + (outlier_threshold,)
    # Same rule for the k-means settings: they change the assignment, and they
    # come from the environment, so a cached result built under the old ones
    # would otherwise be handed back. Empty at the defaults.
    result_key = result_key + _shape_map_kmeans_signature()
    # The projection changes the coordinates, which the payload carries, so it
    # has to reach the RESULT key as well as the aggregate key below -- the two
    # are separate caches and a hit on either one alone would serve the wrong
    # coordinates. Empty for PCA, so nothing already cached is displaced.
    result_key = result_key + _shape_map_projection_signature(projection)
    # Changes the assignment, not the coordinates -- so it keys the result cache
    # and not svd_key, exactly like n_clusters and the outlier threshold.
    result_key = result_key + _shape_map_clustering_signature(clustering)
    # Ward's cut height. Changes the assignment only, like the two above. Note
    # that n_clusters stays in the key even though granularity overrides it, so
    # two requests that differ only in an ignored n_clusters get two entries
    # holding identical payloads -- wasteful, never wrong.
    result_key = result_key + _shape_map_granularity_signature(
        clustering, granularity, ward_max,
    )

    with _shape_map_cache_mutex:
        cached = _shape_map_results.get(result_key)
        if cached is not None:
            _shape_map_results.move_to_end(result_key)
            return cached

    # Only a tag *filter* changes the projected subset; without a filter the SVD
    # is over every part regardless of tags, so it stays cached across tagging
    # (that is what keeps "colour by tag" fast right after tagging a cluster).
    # The aggregate's identity: what the rows ARE, before any projection of
    # them. The Ward tree is built from these rows alone, so this -- not
    # svd_key -- is what keys it.
    agg_key = (path_key, mtime_ns, subset_sig, tags_etag if tag_filter else "")
    # This cache holds the COORDINATES, so the projection must key it. Leaving
    # it out would hand a UMAP request the PCA coordinates computed earlier for
    # the same subset -- the cached entry looks valid in every other respect.
    svd_key = agg_key + _shape_map_projection_signature(projection)
    build_lock = _shape_map_build_lock(svd_key)
    with build_lock:
        with _shape_map_cache_mutex:
            cached = _shape_map_results.get(result_key)
            if cached is not None:
                _shape_map_results.move_to_end(result_key)
                return cached
            agg_entry = _shape_map_agg.get(svd_key)
            if agg_entry is not None:
                _shape_map_agg.move_to_end(svd_key)

        if agg_entry is None:
            agg_started = time.perf_counter()
            proj_timings: dict = {}
            # filemeta is needed both to restrict by kind before projecting and
            # to label the finished points, so load it once up front and combine
            # the kind predicate with any tag predicate (both must pass).
            filemeta = _shape_map_filemeta(name)
            kind_keep = _shape_map_kind_predicate(filemeta, kind)
            keep_fn = _compose_keep(tag_keep, kind_keep)
            # The drill-down subset is also expressed as a predicate, not left
            # to only_ids alone: _shape_map_project documents (and a test pins)
            # that keep_fn takes precedence, so a subset passed only as only_ids
            # would be silently dropped whenever a tag or kind filter is also
            # active. only_ids is still passed so the plain-subset case reads
            # the way the function's own signature suggests.
            keep_fn = _compose_keep(keep_fn, _shape_map_subset_predicate(subset_ids))
            ids, values = _load_index_body_batch(name)
            file_ids, coords_full, agg = _shape_map_project(
                ids, values, only_ids=subset_ids, keep_fn=keep_fn,
                projection=projection, timings=proj_timings,
            )
            if subset_ids is not None and len(file_ids) < 2:
                raise ValueError(
                    f"Subset of {len(subset_ids)} file id(s) leaves "
                    f"{len(file_ids)} mapped part(s); a map needs at least 2."
                )
            tags_map = tags_by_part(name)
            agg_entry = {
                "file_ids": file_ids,
                "coords": coords_full,
                "agg": agg,
                "filemeta": filemeta,
                "tags_map": tags_map,
            }
            with _shape_map_cache_mutex:
                _shape_map_agg[svd_key] = agg_entry
                _shape_map_agg.move_to_end(svd_key)
                while len(_shape_map_agg) > _SHAPE_MAP_AGG_CACHE_MAX:
                    _shape_map_agg.popitem(last=False)
            # agg is what the whole block cost MINUS the projection inside it:
            # loading the body rows, grouping them per part and averaging. The
            # two are reported apart because they fail for different reasons --
            # one is disk and row count, the other is matrix size.
            proj_seconds = float(proj_timings.get("proj", 0.0))
            timing.add("agg", (time.perf_counter() - agg_started) - proj_seconds, _STAGE_MISS)
            timing.add("proj", proj_seconds, _STAGE_MISS)
        else:
            timing.add("agg", 0.0, _STAGE_HIT)
            timing.add("proj", 0.0, _STAGE_HIT)

        tree = None
        ward_parent = None
        if clustering == _CLUSTERING_WARD:
            # A drill-down should walk DOWN the hierarchy, not start a new one.
            # When the subset is a node of the parent's tree, cut that tree
            # deeper instead of building one over the subset -- which is both
            # the right answer and the faster one, since no linkage is built.
            #
            # ward_parent falls back to None for exactly one reason: the subset
            # is not a node (drilled from a k-means map, assembled from a tag
            # filter, hand-picked). That depends only on the request's own
            # inputs, so the result cache stays unambiguous -- the same key
            # always describes the same computation.
            if subset_ids is not None:
                parent = _shape_map_parent_linkage(
                    name, path_key, mtime_ns, tag_filter, tag_keep, kind, tags_etag,
                )
                if parent is not None:
                    rows = _shape_map_ward_subset_node(
                        parent[0], parent[1]["file_ids"], agg_entry["file_ids"],
                    )
                    if rows is not None:
                        ward_parent = (parent[0], rows)
                        logger.debug(
                            "shape map %s: descending the parent Ward tree for "
                            "a drill-down of %d part(s)", name, len(rows),
                        )
                    else:
                        logger.debug(
                            "shape map %s: drill-down subset is not a node of "
                            "the parent Ward tree; clustering it on its own",
                            name,
                        )
            if ward_parent is None:
                linkage_started = time.perf_counter()
                with _shape_map_cache_mutex:
                    tree = _shape_map_linkage.get(agg_key)
                    if tree is not None:
                        _shape_map_linkage.move_to_end(agg_key)
                if tree is None:
                    _shape_map_log_threads()
                    rss_before = _shape_map_rss_mb()
                    # Outside the mutex: this takes tens of seconds and must not
                    # hold the cache lock. The build lock is keyed by svd_key, so
                    # two requests that differ only in projection can both arrive
                    # here and build the same tree; that is wasteful but correct,
                    # and far cheaper than a second lock.
                    tree = _shape_map_linkage_matrix(agg_entry["agg"])
                    rss_after = _shape_map_rss_mb()
                    if rss_before is not None and rss_after is not None:
                        timing.note("rss", f"{rss_before:.0f}MB->{rss_after:.0f}MB")
                    with _shape_map_cache_mutex:
                        _shape_map_linkage[agg_key] = tree
                        _shape_map_linkage.move_to_end(agg_key)
                        while len(_shape_map_linkage) > _SHAPE_MAP_LINKAGE_CACHE_MAX:
                            _shape_map_linkage.popitem(last=False)
                    timing.add(
                        "linkage", time.perf_counter() - linkage_started, _STAGE_MISS,
                    )
                else:
                    timing.add(
                        "linkage", time.perf_counter() - linkage_started, _STAGE_HIT,
                    )
            else:
                # A drill-down cuts the parent's tree, so no linkage is built
                # here at all; report it as a hit, which is what it amounts to.
                timing.add("linkage", 0.0, _STAGE_HIT)

        # The height to cut at, once the tree to cut is settled. Reported back
        # on the payload with the band it sits in, so the UI can show where the
        # cut landed rather than only what was asked for.
        ward_height = None
        ward_band = None
        if clustering == _CLUSTERING_WARD and granularity is not None:
            cut_started = time.perf_counter()
            cut_tree = ward_parent[0] if ward_parent is not None else tree
            cut_rows = ward_parent[1] if ward_parent is not None else None
            if cut_tree is not None:
                # Walked once here and handed to both, so neither re-walks it.
                cut_heights = _shape_map_ward_internal_heights(cut_tree, cut_rows)
                ward_height, lo, hi, k_max = _shape_map_ward_cut_height(
                    cut_tree, cut_rows, granularity, heights=cut_heights,
                    ward_max=ward_max,
                )
                ward_band = {
                    "cut_height": ward_height,
                    "height_lo": lo,
                    "height_hi": hi,
                    "k_max": int(k_max),
                    "granularity": round(float(granularity), 4),
                    "suggestions": _shape_map_ward_suggestions(
                        cut_heights, int(cut_heights.shape[0]) + 1, int(k_max),
                    ),
                }
            timing.add("cut", time.perf_counter() - cut_started)

        cluster_timings: dict = {}
        points, k, cohesion, centres, outlier_effective = _shape_map_cluster(
            agg_entry["agg"], agg_entry["coords"], agg_entry["file_ids"], n_clusters, dims,
            outlier_threshold=outlier_threshold,
            clustering=clustering,
            linkage_z=tree,
            ward_parent=ward_parent,
            ward_height=ward_height,
            timings=cluster_timings,
        )
        # The assignment belongs to whichever algorithm ran: for Ward it is the
        # fcluster pass, which is part of cutting the tree; for k-means it is
        # the fit itself, and k-means has no linkage or cut to report.
        assign_seconds = float(cluster_timings.get("assign", 0.0))
        if clustering == _CLUSTERING_WARD:
            timing.add("cut", assign_seconds)
        else:
            timing.add("kmeans", assign_seconds, _STAGE_MISS)
        timing.add("stats", float(cluster_timings.get("stats", 0.0)))

        payload_started = time.perf_counter()
        # Use the fresh tags_map snapshot, not agg_entry["tags_map"]: when the SVD
        # is reused across a tag change the cached aggregate carries the pre-tag
        # tags, and the whole point of result_tags_etag is to rebuild the payload
        # with the current tags.
        payload = _shape_map_build_payload(
            name, points, k, dims, agg_entry["filemeta"], tags_map,
            include_tags, color_by, cohesion, centres, outlier_effective,
        )
        # Carried on the payload rather than added by the caller so it survives
        # the result cache: /map/scs needs the same id for the same points, and
        # a cache hit must hand back the id the entry was built for.
        payload["subset_id"] = subset_id
        if ward_band is not None:
            payload["ward"] = ward_band

        with _shape_map_cache_mutex:
            _shape_map_results[result_key] = payload
            _shape_map_results.move_to_end(result_key)
            while len(_shape_map_results) > _SHAPE_MAP_RESULT_CACHE_MAX:
                _shape_map_results.popitem(last=False)
        timing.add("payload", time.perf_counter() - payload_started)

    timing.note("n", len(payload.get("points") or ()))
    timing.note("k", k)
    logger.info(
        "[SHAPEMAP] '%s' %s map built in %.2fs (%s)",
        name, clustering, timing.total(), timing.render(),
    )
    return payload


def compute_index_shape_map_tree(
    name: str,
    tags: Optional[list[str]] = None,
    tag_mode: str = "any",
    tagged: Optional[bool] = None,
    kind: str = "any",
    only_file_ids: Optional[list[str]] = None,
    k_max: int = _SHAPE_MAP_WARD_K_MAX,
) -> dict[str, Any]:
    """Ward's hierarchy for *name*, folded to the levels a tree view can show.

    The population is chosen exactly as ``/map`` chooses it -- the same tag
    filter (*tags* / *tag_mode* / *tagged*), the same *kind*, the same explicit
    *only_file_ids* subset -- so the tree describes the map the caller is
    looking at and not some other set of parts.

    Neither a projection nor a colouring is taken, because a hierarchy has
    neither coordinates nor a legend: it is cut from the aggregate vectors
    alone, and the same tree answers every projection and colour mode.

    *only_file_ids* must be a node of the tree, which is what a drill-down from
    a Ward map always is. Anything else -- a subset carried over from a k-means
    map, or assembled by a filter that cuts across the hierarchy -- has no
    sub-tree to show and raises ``ValueError``.

    Returns ``index``, ``count``, the ``k_max`` actually used, the ``levels``
    present, the member ``file_ids``, a parallel ``leaf`` array naming the
    deepest node holding each member, and the ``nodes`` themselves.

    Raises ``KeyError`` when the index does not exist, ``ValueError`` for an
    invalid tag filter or a subset that is not a node, and
    :class:`WardTooLarge` when the population is larger than Ward can be run on
    here.
    """
    import numpy as np

    _validate_index_name(name)
    # Total only, with the one cache that matters here -- the linkage -- named
    # as hit or miss. A tree request that is slow is a tree request that had to
    # build the linkage; there is no other tens-of-seconds step.
    timing = _ShapeMapTiming()
    faiss_path = index_registry._index_faiss_path(name)
    if not faiss_path.exists():
        raise KeyError(f"Index '{name}' does not exist.")
    mtime_ns = faiss_path.stat().st_mtime_ns
    path_key = str(faiss_path)

    _tags_map, tags_etag = _shape_map_tags_snapshot(name)
    tag_filter = normalize_tag_filter(tags, tag_mode, tagged)
    tag_keep = tag_filter_predicate(name, tag_filter) if tag_filter else None
    kind = _normalize_map_kind(kind)
    subset_ids = _normalize_map_subset(only_file_ids)

    # The key _shape_map_parent_linkage will look under, rebuilt here so the
    # log can say whether the tree was already there without that helper
    # having to report it.
    linkage_key = (
        path_key, mtime_ns, _shape_map_subset_sig(tag_filter, kind, None),
        tags_etag if tag_filter else "",
    )
    with _shape_map_cache_mutex:
        linkage_state = _STAGE_HIT if linkage_key in _shape_map_linkage else _STAGE_MISS

    bundle = _shape_map_parent_linkage(
        name, path_key, mtime_ns, tag_filter, tag_keep, kind, tags_etag,
    )
    if bundle is None:
        raise WardTooLarge(
            "This map has more points than Ward's hierarchy can be built on "
            "here, so there is no tree to show."
        )
    tree, agg_entry = bundle

    rows = None
    if subset_ids is not None:
        rows = _shape_map_ward_subset_node(tree, agg_entry["file_ids"], subset_ids)
        if rows is None:
            raise ValueError(
                "This subset is not a node of the tree, so it has no sub-tree to "
                "show. It came from a different clustering, or from a tag or kind "
                "filter that cuts across the hierarchy."
            )

    if rows is None:
        member_file_ids = list(agg_entry["file_ids"])
    else:
        parent_ids = agg_entry["file_ids"]
        member_file_ids = [parent_ids[int(r)] for r in np.asarray(rows).tolist()]

    # Not cached: the linkage itself already is, and folding it into levels is
    # nine fcluster passes over rows that are in memory -- less work than
    # keeping another cache correct would cost.
    nodes, leaf = _shape_map_ward_tree(
        tree, agg_entry["agg"], member_file_ids, rows, k_max,
    )

    # The deepest level the tree could have, which is the requested k_max capped
    # by the cut helper's own ceiling. Read from the same helper rather than
    # from the levels that produced nodes: a branch that stopped splitting early
    # leaves its level absent, and that says nothing about the cap.
    _h, _lo, _hi, band_k_max = _shape_map_ward_cut_height(
        tree, rows, 1.0, ward_max=k_max,
    )
    effective_k_max = max(1, min(int(k_max), int(band_k_max)))

    filemeta = agg_entry.get("filemeta") or {}
    for node in nodes:
        centre = node.get("centre")
        meta = filemeta.get(centre) if centre is not None else None
        node["filename"] = (meta or {}).get("filename")

    timing.note("n", len(member_file_ids))
    timing.note("k_max", effective_k_max)
    logger.info(
        "[SHAPEMAP] '%s' ward tree built in %.2fs (linkage %s, %s)",
        name, timing.total(), linkage_state, timing.render(),
    )
    return {
        "index": name,
        "count": len(member_file_ids),
        "k_max": effective_k_max,
        "levels": sorted({int(node["level"]) for node in nodes}),
        "file_ids": member_file_ids,
        "leaf": [int(v) for v in np.asarray(leaf).tolist()],
        "nodes": nodes,
    }


def shape_map_aggregate_vectors(
    name: str,
    kind: str = "part",
    tags: Optional[list[str]] = None,
    tag_mode: str = "any",
    tagged: Optional[bool] = None,
) -> tuple[list[str], Any]:
    """``(file_ids, agg)`` for the population a tree request would cut.

    Same arguments, same preparation and same population as
    :func:`compute_index_shape_map_tree` called with no subset, including the
    ORDER of *file_ids* -- so a row number here is the row number the tree's
    nodes were built from.

    No linkage is built, which is the whole reason this is a separate entry
    point: measuring the geometry inside a frontier group needs the rows and
    nothing else, and Ward on the whole population is tens of seconds. Calling
    it after a tree request hits the aggregate cache; calling it alone builds
    the aggregate only.

    The matrix is not part of the tree response and must not become part of
    it -- the router serialises that -- so this is how a diagnostic gets at it:
    an offline script pairs the matrix with :func:`shape_map_pair_spread`.

    Raises the same exceptions as :func:`compute_index_shape_map_tree` for the
    same reasons: ``KeyError`` when the index does not exist, ``ValueError``
    for an invalid tag filter.

    Like the frontier helpers above this is a stable (C) name, and like them it
    is not re-exported from ``core``: no caller there wants a matrix.
    """
    _validate_index_name(name)
    faiss_path = index_registry._index_faiss_path(name)
    if not faiss_path.exists():
        raise KeyError(f"Index '{name}' does not exist.")
    mtime_ns = faiss_path.stat().st_mtime_ns
    path_key = str(faiss_path)

    _tags_map, tags_etag = _shape_map_tags_snapshot(name)
    tag_filter = normalize_tag_filter(tags, tag_mode, tagged)
    tag_keep = tag_filter_predicate(name, tag_filter) if tag_filter else None
    kind = _normalize_map_kind(kind)

    _key, entry = _shape_map_parent_agg(
        name, path_key, mtime_ns, tag_filter, tag_keep, kind, tags_etag,
    )
    return (list(entry["file_ids"]), entry["agg"])


# --- Index-based Shape Map: ordering the legend ----------------------------
# The legend lists clusters by member count descending (see the numbering note
# near _shape_map_l2norm_rows), which is a good order for "what is big" and no
# order at all for "what is like what": a family that k-means split across six
# clusters is six rows scattered the length of the list, and finding them means
# hovering down it.
#
# An alternative order is the leaf order of a Ward dendrogram over the cluster
# CENTROIDS. Clusters whose centres are close end up adjacent, so the rows of
# one family fall together. That it actually does so was measured before any
# of this was built, by a diagnostic script that imports the two functions
# below so that what was measured is what ships.
#
# This changes the ORDER THE LEGEND IS DRAWN IN and nothing else. Cluster
# numbers, colours, the groups array and its offsets, the baked .scs and pick
# resolution are all untouched: a group's position in the payload is how the
# viewer resolves (node, index-within-node) to a point, so the array itself
# must never be reordered. The order below is a separate array of cluster ids
# that the client reads the groups through.

# scipy's optimal_leaf_ordering is Bar-Joseph, O(k^3) in the worst case. At
# k=256 that is ~1.7e7 and runs in milliseconds; at k=1024 it is ~1e9 and
# would stall the request. Above the cap the plain leaf order is used, which
# measured worse but still well above chance (span 141 vs 87, both against a
# shuffled median of 214).
_LEGEND_ORDER_OLO_MAX = 512

# Finished orders are small (one int per cluster), but each one costs a k-means
# map plus a Ward linkage, so a few are worth keeping. Smaller than the map's
# own cache because only one legend is on screen at a time.
_LEGEND_ORDER_CACHE_MAX = 8
_legend_order_results: "OrderedDict[tuple, Any]" = OrderedDict()


def shape_map_cluster_centroids(agg, assign) -> tuple[list[int], Any]:
    """``([cluster id, ...], centroid rows)`` -- the mean of each cluster's members.

    *assign* is one cluster id per row of *agg*, as
    :func:`_shape_map_centroid_sims` and :func:`_shape_map_cohesion` take it.
    The ids come back sorted ascending and row *i* of the matrix is the centre
    of ``ids[i]``.

    Cluster ``-1`` is left out. It is the Outliers row, a display category
    rather than a cluster, and it has no centre worth ordering by. Empty
    clusters are left out for the same reason: the legend has no row for them.

    The centroid is the mean of the members' aggregate rows, re-normalised with
    :func:`_shape_map_l2norm_rows`: this is recomputed from the FINAL
    assignment and so can differ in the last digits from the centroid FAISS
    holds, which is the mean at the iteration it stopped at. For "where is this
    cluster on screen", the recomputed one is the right answer -- it is the
    centre of the points actually drawn in it.
    """
    import numpy as np

    rows = np.asarray(agg, dtype=np.float32)
    assign = np.asarray(assign)
    if rows.ndim != 2:
        raise ValueError("agg must be a 2-D matrix of aggregate vectors")
    if assign.shape[0] != rows.shape[0]:
        raise ValueError(
            f"assign has {assign.shape[0]} entries for {rows.shape[0]} rows"
        )
    ids = [int(c) for c in np.unique(assign) if int(c) >= 0]
    ids = [c for c in ids if bool((assign == c).any())]
    if not ids:
        return ([], _shape_map_l2norm_rows(np.zeros((0, rows.shape[1]), dtype=np.float32)))
    centroids = np.vstack([rows[assign == c].mean(axis=0) for c in ids])
    return (ids, _shape_map_l2norm_rows(centroids))


def shape_map_legend_order(
    centroids, olo_max: int = _LEGEND_ORDER_OLO_MAX,
) -> tuple[list[int], str]:
    """``([row index, ...], method)`` -- Ward leaf order over the centroids.

    The returned list is a permutation of ``range(len(centroids))``, so the
    caller maps it back to its own cluster ids. *method* is ``"ward+olo"`` or
    ``"ward"``, which is what the caller shows when it reports what it did.

    scipy is called directly rather than through
    :func:`_shape_map_linkage_matrix`. That helper exists for a population of
    thousands -- blocked distances, and a WardTooLarge ceiling -- and k
    centroids are a different scale entirely. The rule is the same one: Ward
    over L2-normalised rows, which is Ward on the unit sphere.

    Fewer than two rows have no order to compute: scipy's linkage needs two
    observations to merge, and the one arrangement of nothing or of a single
    row is the arrangement either way.
    """
    import numpy as np

    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering

    rows = np.asarray(centroids, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError("centroids must be a 2-D matrix")
    k = int(rows.shape[0])
    if k < 2:
        return (list(range(k)), "ward")
    link = linkage(rows, method="ward")
    if k <= int(olo_max):
        return ([int(i) for i in leaves_list(optimal_leaf_ordering(link, rows))], "ward+olo")
    return ([int(i) for i in leaves_list(link)], "ward")


def compute_index_legend_order(
    name: str,
    n_clusters: int = 0,
    kind: str = "any",
    tags: Optional[list[str]] = None,
    tag_mode: str = "any",
    tagged: Optional[bool] = None,
    only_file_ids: Optional[list[str]] = None,
    outlier_threshold: float = 0.0,
    clustering: str = _CLUSTERING_KMEANS,
) -> dict[str, Any]:
    """``{"index", "k", "method", "order"}`` -- the legend in similarity order.

    ``order`` is a list of CLUSTER IDS, not row numbers and not groups: the
    groups array is positional (the viewer resolves a pick through
    ``groups[node].offset``), so it is never reordered. The client draws its
    rows through this list and leaves the payload alone.

    The arguments are the map's own, and :func:`compute_index_shape_map` is
    called with them to get the assignment, so this reads the clustering that
    is on screen rather than building a second one -- and normally hits that
    function's cache, which is why there is no separate path to the k-means
    here. ``color_by`` is fixed to ``"cluster"`` because it changes only the
    legend's colouring, never the assignment, and pinning it keeps two requests
    that differ in nothing that matters on one cache entry.

    Only k-means has a question to answer here. Ward's clusters are a cut of a
    tree that already states which families are near which, and the client
    draws them in that tree's order, so asking for this with
    ``clustering="ward"`` is a mistake rather than a preference: it raises
    ``ValueError``.

    Raises ``KeyError`` when the index does not exist and ``ValueError`` for an
    invalid tag filter, like every other entry point here.
    """
    import numpy as np

    _validate_index_name(name)
    clustering = _normalize_map_clustering(clustering)
    if clustering != _CLUSTERING_KMEANS:
        raise ValueError(
            "legend order is for clustering='kmeans' only: Ward's tree already "
            "orders its clusters, and the client draws them in that order."
        )
    faiss_path = index_registry._index_faiss_path(name)
    if not faiss_path.exists():
        raise KeyError(f"Index '{name}' does not exist.")
    mtime_ns = faiss_path.stat().st_mtime_ns

    tag_filter = normalize_tag_filter(tags, tag_mode, tagged)
    kind = _normalize_map_kind(kind)
    subset_ids = _normalize_map_subset(only_file_ids)
    subset_id = _register_map_subset(subset_ids)
    subset_sig = _shape_map_subset_sig(tag_filter, kind, subset_id)
    outlier_threshold = _normalize_outlier_threshold(outlier_threshold)
    _tags_map, tags_etag = _shape_map_tags_snapshot(name)
    # The same elements that key the map's own result, plus the cluster count:
    # anything that moves a part from one cluster to another moves the
    # centroids, and the outlier threshold does exactly that, so it is in the
    # key whether or not it is in use.
    cache_key = (
        str(faiss_path), mtime_ns, subset_sig, int(n_clusters), outlier_threshold,
        clustering, tags_etag if tag_filter else "", _LEGEND_ORDER_OLO_MAX,
    ) + _shape_map_kmeans_signature()
    with _shape_map_cache_mutex:
        cached = _legend_order_results.get(cache_key)
        if cached is not None:
            _legend_order_results.move_to_end(cache_key)
            return cached

    payload = compute_index_shape_map(
        name,
        n_clusters=n_clusters,
        tags=tags,
        tag_mode=tag_mode,
        tagged=tagged,
        color_by="cluster",
        kind=kind,
        only_file_ids=only_file_ids,
        outlier_threshold=outlier_threshold,
        clustering=clustering,
    )
    file_ids, agg = shape_map_aggregate_vectors(
        name, kind=kind, tags=tags, tag_mode=tag_mode, tagged=tagged,
    )
    row_of = {str(fid): i for i, fid in enumerate(file_ids)}
    rows: list[int] = []
    assign: list[int] = []
    for point in payload.get("points") or []:
        file_id = str(point.get("file_id"))
        # A map point outside the aggregate population would mean the two were
        # built from different sets of parts, and the order would then be of
        # clusters that are not the ones on screen. Loud, not skipped.
        assert file_id in row_of, (
            f"part {file_id[:8]} is on the map but not in the aggregate "
            f"population it was built from"
        )
        rows.append(row_of[file_id])
        assign.append(int(point.get("cluster")))

    ids, centroids = shape_map_cluster_centroids(np.asarray(agg)[rows], assign)
    order_rows, method = shape_map_legend_order(centroids)
    result = {
        "index": name,
        "k": len(ids),
        "method": method,
        "order": [ids[i] for i in order_rows],
    }
    with _shape_map_cache_mutex:
        _legend_order_results[cache_key] = result
        _legend_order_results.move_to_end(cache_key)
        while len(_legend_order_results) > _LEGEND_ORDER_CACHE_MAX:
            _legend_order_results.popitem(last=False)
    return result


# --- Index-based Shape Map: the catalogue frontier -------------------------
# One entry per family rather than per cluster: the Ward tree is cut wherever a
# node is cohesive enough to stand on its own.
#
# The defaults below come from a measurement, not a guess. On tmcad (7,928
# single parts), with the frontier cut at the default cohesion:
#
#   k_max=1024   size>=20: 35 groups / 1,843 parts   size>=5: 212 / 3,539
#   k_max=2048   size>=20: 35 groups / 1,843 parts   size>=5: 300 / 4,098
#   k_max=4096   size>=20: 35 groups / 1,843 parts   size>=5: 300 / 4,098
#
# Groups of five or more converge at 2048: going to 4096 adds not one. What it
# does add is singletons (8 -> 360 -> 1,634), and a singleton reaches the
# threshold only because a group of one is trivially cohesive -- "100% of parts
# reached" at 4096 is an arithmetic artefact with nothing behind it. 2048 is
# therefore the default: the deepest cut that still buys families.

# Cohesion is entangled with group size, so reaching the threshold is not on
# its own a claim worth making. This index's root has a cohesion of 0.7049:
# the cosines sit in a narrow band, and the mean of three of them clears 0.96
# easily. A group of three at 0.96 has been observed holding a part that
# plainly does not belong; 180 parts at 0.9576 is a different statement
# altogether, made of 16,110 pairs rather than 3.
#
# Where to draw the line was measured, not chosen. Groups of five or more are
# 300 groups / 4,098 parts at k_max=2048 and the same 300 / 4,098 at 4096,
# while groups of two or more move with the depth (1,016 -> 1,174). Five is
# therefore the size at which the catalogue stops depending on the depth
# parameter -- which is the property that makes it a usable line and not an
# arbitrary one.
_CATALOGUE_DEFAULT_THRESHOLD = 0.95
_CATALOGUE_DEFAULT_K_MAX = 2048
_CATALOGUE_DEFAULT_MIN_PARTS = 1
_CATALOGUE_DEFAULT_MIN_REACHED = 5

# Finished catalogue payloads, keyed as the map's results are. Smaller than
# _SHAPE_MAP_RESULT_CACHE_MAX because one entry carries every file id of the
# index once over, and a catalogue is asked for far less often than a map.
_CATALOGUE_RESULT_CACHE_MAX = 4
_catalogue_results: "OrderedDict[tuple, Any]" = OrderedDict()


def _catalogue_signature(
    threshold: float,
    k_max: int,
    min_parts: int,
    min_reached_parts: int = _CATALOGUE_DEFAULT_MIN_REACHED,
) -> tuple:
    """Cache-key fragment for the catalogue's own knobs; empty at the defaults.

    The house rule for every cache key here: a non-default option appends to
    the key, a default one leaves it byte-for-byte as it was. That is what
    keeps entries written before an option existed addressable by the same key
    -- and it is why the common request, which uses none of these, keys the
    same tuple shape the aggregate caches already use.

    The threshold is rounded to four decimals, the precision the UI shows a
    cohesion at, so two requests that cannot be told apart on screen do not
    build two entries.
    """
    sig: tuple = ()
    if round(float(threshold), 4) != _CATALOGUE_DEFAULT_THRESHOLD:
        sig = sig + (("t", round(float(threshold), 4)),)
    if int(k_max) != _CATALOGUE_DEFAULT_K_MAX:
        sig = sig + (("kmax", int(k_max)),)
    if int(min_parts) != _CATALOGUE_DEFAULT_MIN_PARTS:
        sig = sig + (("min", int(min_parts)),)
    if int(min_reached_parts) != _CATALOGUE_DEFAULT_MIN_REACHED:
        sig = sig + (("conf", int(min_reached_parts)),)
    return sig


def _catalogue_stats(
    classes: dict,
    returned_groups: int,
    min_reached_parts: int,
) -> dict[str, Any]:
    """How good this cut of the tree was, for comparing one cut against another.

    Every figure here EXCEPT ``groups`` describes the whole frontier, before
    ``min_parts`` dropped anything. That is deliberate: the stats answer "was
    this threshold and this depth a good way to cut the tree", and an answer
    that moved when the caller changed a display filter would not be an answer
    at all. ``groups`` is the count actually returned, i.e. after the filter --
    the one number here that is about the response rather than the cut.

    The three classes come from :func:`shape_map_classify_frontier` and still
    partition the population, so ``confident_parts + small_parts +
    mixed_parts`` is the whole of it.

    The arithmetic is :func:`shape_map_size_buckets` and
    :func:`shape_map_size_stats`, the same two functions a diagnostic prints
    its tables from, so a number on screen and a number from an offline
    measurement cannot drift apart.

    Buckets cover the CONFIDENT groups only. "How many families have at least
    five parts" must count neither a branch that never got tight (mixed) nor a
    handful of parts whose tightness says little (small).

    ``confident_singletons`` is zero unless *min_reached_parts* is 1, which is
    what the field is there to make visible rather than to hide.
    """
    conf = shape_map_size_stats(classes["confident"])
    small = shape_map_size_stats(classes["small"])
    mixed = shape_map_size_stats(classes["mixed"])
    return {
        # After min_parts: the number of entries in "groups".
        "groups": int(returned_groups),
        # The size line the confident/small split was made at.
        "min_reached_parts": int(min_reached_parts),
        # Everything below: the whole frontier, before min_parts.
        "confident_groups": int(conf["groups"]),
        "confident_parts": int(conf["members"]),
        "confident_largest": int(conf["largest"]),
        "confident_median": float(conf["median"]),
        "confident_singletons": int(conf["singletons"]),
        "small_groups": int(small["groups"]),
        "small_parts": int(small["members"]),
        "small_largest": int(small["largest"]),
        "small_median": float(small["median"]),
        "small_singletons": int(small["singletons"]),
        "mixed_groups": int(mixed["groups"]),
        "mixed_parts": int(mixed["members"]),
        "mixed_largest": int(mixed["largest"]),
        "mixed_median": float(mixed["median"]),
        "mixed_singletons": int(mixed["singletons"]),
        "buckets": [
            {"min_parts": mark, "groups": count, "parts": parts}
            for mark, count, parts in shape_map_size_buckets(classes["confident"])
        ],
    }


def compute_index_catalogue(
    name: str,
    threshold: float = _CATALOGUE_DEFAULT_THRESHOLD,
    k_max: int = _CATALOGUE_DEFAULT_K_MAX,
    kind: str = "part",
    min_parts: int = _CATALOGUE_DEFAULT_MIN_PARTS,
    min_reached_parts: int = _CATALOGUE_DEFAULT_MIN_REACHED,
    tags: Optional[list[str]] = None,
    tag_mode: str = "any",
    tagged: Optional[bool] = None,
) -> dict[str, Any]:
    """The cohesion frontier of *name*'s Ward tree, with each group's members.

    Walks down from the root and stops each branch at the first node whose
    cohesion reaches *threshold*; branches that never get there end at their
    deepest node. Every part is in exactly one group -- that is asserted, not
    assumed -- so a client can tag or highlight a group without touching a
    part twice.

    Each group carries a ``kind``, which is the authoritative label and the
    one a client should read:

    * ``"confident"`` -- cohesion at or above *threshold*, and at least
      *min_reached_parts* members.
    * ``"small"`` -- cohesion at or above *threshold*, fewer members than
      that. Tight, but over too few pairs for the number to mean much.
    * ``"mixed"`` -- cohesion below *threshold*, or none at all (a group of
      one). The branch ran out before it got tight.

    The older ``reached`` boolean is still present and still true for
    ``confident`` and ``small`` alike, but it over-claims: see
    :func:`shape_map_classify_frontier` for why three parts at 0.96 and 180
    parts at 0.9576 are not the same finding. *min_reached_parts* only labels;
    it never changes which nodes the frontier is made of.

    ``height`` is the node's own Ward merge height -- the absolute dissimilarity
    at which that family was joined. Unlike cohesion it does not vary with how
    many members a group has, and every node created at one level of the tree
    was cut at the same height, so heights are directly comparable between
    groups in a way cohesions are not.

    The tree comes from :func:`compute_index_shape_map_tree`, the same entry
    point ``/map/tree`` serves, so the groups here are nodes the tree panel can
    show and their cohesion is the number it displays.

    *min_parts* drops the groups smaller than it from ``groups`` and counts
    them in ``omitted_by_min_parts``; ``stats`` still describes the whole
    frontier (see :func:`_catalogue_stats`), so what was left out stays
    visible.

    ``file_ids`` is every member of the group, because a catalogue entry is
    only useful if the client can act on its contents. That makes the response
    the size of one map payload, which is a size this API already returns.

    ``tagged_parts`` and ``tags`` summarise what has already been written over
    a group: how many of its members carry at least one tag, and the tag names
    that appear in it (most frequent first, at most
    ``SHAPE_MAP_GROUP_TAG_LIMIT``). They are what lets a client show which
    entries have been dealt with, and they are why a tag write invalidates this
    result -- see the cache key below.

    Groups are returned in the order the tree panel draws them: a pre-order
    walk with siblings in node order (:func:`shape_map_frontier_order`). The
    catalogue is read beside the tree, so the two have to run the same way
    down. A client that wants them biggest-first can sort what it was given;
    it cannot recover the tree's order from a sorted list.

    Raises ``KeyError`` when the index does not exist, ``ValueError`` for an
    invalid tag filter, and :class:`WardTooLarge` when the population is larger
    than Ward can be run on here.
    """
    _validate_index_name(name)
    faiss_path = index_registry._index_faiss_path(name)
    if not faiss_path.exists():
        raise KeyError(f"Index '{name}' does not exist.")
    mtime_ns = faiss_path.stat().st_mtime_ns
    path_key = str(faiss_path)

    tag_filter = normalize_tag_filter(tags, tag_mode, tagged)
    tags_map, tags_etag = _shape_map_tags_snapshot(name)
    kind = _normalize_map_kind(kind)
    subset_sig = _shape_map_subset_sig(tag_filter, kind, None)
    # The ETag is always in the key, unlike the map's, because every group
    # reports which of its parts are tagged. The result therefore changes when
    # ANY part is tagged, not only when a tag filter moves the population --
    # and the catalogue is the tab tags are written from, so a stale entry
    # would be served straight back to the user who had just caused it.
    result_key = (
        path_key, mtime_ns, subset_sig, tags_etag,
    ) + _catalogue_signature(threshold, k_max, min_parts, min_reached_parts)

    with _shape_map_cache_mutex:
        cached = _catalogue_results.get(result_key)
        if cached is not None:
            _catalogue_results.move_to_end(result_key)
            return cached

    started = time.perf_counter()
    tree = compute_index_shape_map_tree(
        name,
        tags=tags,
        tag_mode=tag_mode,
        tagged=tagged,
        kind=kind,
        only_file_ids=None,
        k_max=k_max,
    )
    nodes = list(tree["nodes"])
    total = int(tree["count"])
    frontier = shape_map_cohesion_frontier(nodes, float(threshold))
    shape_map_check_frontier(nodes, frontier, total)
    members = shape_map_frontier_members(
        nodes, frontier, list(tree["file_ids"]), list(tree["leaf"]),
    )

    reached_ids = {int(node["id"]) for node in frontier["reached"]}
    classes = shape_map_classify_frontier(frontier, int(min_reached_parts))
    kind_of = {}
    for label in ("confident", "small", "mixed"):
        for node in classes[label]:
            kind_of[int(node["id"])] = label
    groups = []
    omitted = 0
    # Pre-order, so the list reads down the tree the way the tree panel does.
    frontier_ids = {int(node["id"]) for node in [*frontier["reached"], *frontier["stalled"]]}
    node_of = {int(node["id"]): node for node in [*frontier["reached"], *frontier["stalled"]]}
    for node_id in shape_map_frontier_order(nodes, frontier_ids):
        node = node_of[node_id]
        file_ids = members.get(node_id) or []
        # The two counts come from different places -- the tree's own count and
        # the walk up from the leaves -- so a disagreement means the membership
        # is wrong, not merely surprising.
        assert len(file_ids) == int(node["count"]), (
            f"node {node_id} holds {int(node['count'])} parts but "
            f"{len(file_ids)} were collected"
        )
        if len(file_ids) < int(min_parts):
            omitted += 1
            continue
        cohesion = node.get("cohesion")
        height = node.get("height")
        group_tagged, group_tags = shape_map_group_tags(file_ids, tags_map)
        groups.append({
            "id": node_id,
            "count": len(file_ids),
            "cohesion": None if cohesion is None else float(cohesion),
            "height": None if height is None else float(height),
            "level": int(node.get("level") or 0),
            "kind": kind_of[node_id],
            "reached": node_id in reached_ids,
            "tagged_parts": group_tagged,
            "tags": group_tags,
            "centre_file_id": node.get("centre"),
            "centre_filename": node.get("filename"),
            "file_ids": file_ids,
        })

    payload = {
        "index": name,
        "threshold": round(float(threshold), 4),
        "k_max": int(tree["k_max"]),
        "total_parts": total,
        "omitted_by_min_parts": omitted,
        "stats": _catalogue_stats(
            classes,
            returned_groups=len(groups),
            min_reached_parts=int(min_reached_parts),
        ),
        "groups": groups,
    }
    logger.info(
        "[SHAPEMAP] '%s' catalogue built in %.2fs (t=%.4f k_max=%d, %d groups, "
        "%d confident / %d small / %d mixed parts)",
        name, time.perf_counter() - started, float(threshold), int(tree["k_max"]),
        len(groups), payload["stats"]["confident_parts"],
        payload["stats"]["small_parts"], payload["stats"]["mixed_parts"],
    )

    with _shape_map_cache_mutex:
        _catalogue_results[result_key] = payload
        _catalogue_results.move_to_end(result_key)
        while len(_catalogue_results) > _CATALOGUE_RESULT_CACHE_MAX:
            _catalogue_results.popitem(last=False)
    return payload


# --- Index-based Shape Map: point-cloud SCS export -------------------------
# The HOOPS Web Viewer opens a point cloud from a .scs URL, so the map points are
# baked into a single .scs by the scs_points tool (tools/scs_points). That tool is
# a separate, license-bearing binary that is NOT committed: the operator builds it
# and drops it in <APP_ROOT>/bin (a fixed, non-configurable path). Colours are
# baked into the file, so the colour scheme is part of the cache key.


# The tool is trivial for map-sized inputs (~10k points authored sub-second), so a
# generous timeout only guards against a hung/broken binary, never normal work.
_SCS_POINTS_TIMEOUT_S = 120
_SCS_MAP_CACHE_MAX_DEFAULT = 32
_SCS_UNTAGGED_RGB = (180, 180, 180)

# Categorical palette shared by cluster colouring and tag colouring. A Shape Map
# cluster/tag id is a CATEGORY, not a magnitude, so a continuous colour wheel
# (adjacent categories nearly the same hue) makes neighbouring groups
# unreadable. The palette is discrete on purpose and holds 24 colours so a
# 24-cluster view colours every cluster with a distinct one and a tag-heavy
# index colours every tag. Cluster counts past 24 cycle it (see
# _shape_map_cluster_rgb) rather than extending it.
#
# The first 8 slots are the Okabe-Ito colour-blind-safe qualitative colours
# (their black replaced by a bright neutral), kept first so small category
# counts stay colour-blind friendly. Slots 9-24 extend the palette with Kelly's
# "maximum contrast" colours; the extremely dark and near-grey Kelly entries are
# dropped (they vanish on the dark viewer background or clash with the neutral
# Untagged/Other greys), so every slot stays bright and distinct. They are NOT
# sampled from a continuous colour map -- doing that is what made the previous
# version unreadable. Defined once and used for both colour modes.
_SHAPE_MAP_PALETTE: list[tuple[int, int, int]] = [
    # Okabe-Ito (colour-blind safe), black -> bright neutral.
    (230, 159, 0),    # #E69F00 orange
    (86, 180, 233),   # #56B4E9 sky blue
    (0, 158, 115),    # #009E73 bluish green
    (240, 228, 66),   # #F0E442 yellow
    (0, 114, 178),    # #0072B2 blue
    (213, 94, 0),     # #D55E00 vermillion
    (204, 121, 167),  # #CC79A7 reddish purple
    (222, 222, 222),  # bright neutral (replaces Okabe-Ito black)
    # Kelly maximum-contrast colours (dark / near-grey entries omitted).
    (255, 179, 0),    # vivid yellow
    (255, 104, 0),    # vivid orange
    (193, 0, 32),     # vivid red
    (0, 125, 52),     # vivid green
    (246, 118, 142),  # strong purplish pink
    (0, 83, 138),     # strong blue
    (255, 122, 92),   # strong yellowish pink
    (255, 142, 0),    # vivid orange yellow
    (179, 40, 81),    # strong purplish red
    (244, 200, 0),    # vivid greenish yellow
    (147, 170, 0),    # vivid yellowish green
    (241, 58, 19),    # vivid reddish orange
    (166, 189, 215),  # very light blue
    (206, 162, 98),   # grayish yellow (tan)
    (128, 62, 117),   # strong purple
    (129, 197, 122),  # light green
]
# The neutral the Outliers category is baked in -- the points a threshold put
# outside every cluster. Distinct from the untagged grey so the two never look
# like the same category. No category overflows into it any more: both cluster
# and tag colouring cycle the palette instead (see _shape_map_cluster_rgb).
_SHAPE_MAP_OVERFLOW_RGB = (128, 128, 128)

# The colour every point is baked in when Ward colours by cluster. There the
# client decides the real colour at runtime, from how far the tree is expanded,
# and overwrites what was baked the moment the cloud is on screen. The baked
# colours are therefore visible for an instant only, and a kaleidoscope of 256
# of them flashing past is noise: it shows a grouping nobody asked for and that
# is replaced immediately. A neutral grey reads as what it is -- "no colour
# decided yet" -- and the real colours arrive over it. Every other combination
# keeps its own colours (see compute_index_shape_map_scs): k-means has no tree
# to recolour the cloud from, and Ward + Colour by Tag does not use the tree at
# all, so there the legend colour IS the meaning.
_SHAPE_MAP_WARD_NEUTRAL_RGB = (120, 126, 136)

# Lightness factors applied to the palette past its 24 slots. Index 0 (1.0) is
# the first lap -- the palette itself, untouched -- and is never applied as a
# factor; indices 1 onward are the steps the slots past 24 cycle through, one
# per consecutive slot (see _shape_map_cluster_rgb). Fixed steps rather than a
# continuous darkening function, and bottomed out at 0.40, because past that
# everything sinks into the dark viewer background and the steps stop being
# distinguishable from each other anyway.
_SHAPE_MAP_CYCLE_FACTORS: tuple[float, ...] = (1.0, 0.72, 0.52, 0.40)


def _shape_map_cluster_rgb(i: int) -> tuple[int, int, int]:
    """Colour for the *i*-th category (0-based), cycling the palette past 24.

    Shared by BOTH colourings: cluster *i* and tag *i* get the same colour for
    the same slot. The name is historical -- it was written for clusters, and
    is kept because it is re-exported from ``core`` and named in the module
    layout test; renaming it would churn those for nothing. There is one
    implementation, so the two colourings cannot drift apart.

    Slots 0-23 return ``_SHAPE_MAP_PALETTE[i]`` UNCHANGED, so every map with 24
    or fewer categories is coloured exactly as it always was -- as in every
    earlier version of this function.

    Past that, the LIGHTNESS cycles fastest and the hue steps slowest: slots
    24, 25 and 26 are one hue at the three darker steps of
    ``_SHAPE_MAP_CYCLE_FACTORS``, and slot 27 starts the next hue. Two
    consecutive slots therefore never share a lightness.

    It used to be the other way round -- the lap was the high bits, ``i // 24``
    -- and that was wrong for a reason that is not about aesthetics. Cluster
    ids are a SIZE RANKING (clusters are renumbered by member count
    descending), so the lap tracked size and lightness came to mean "small".
    At k=256 it put 184 of the 256 clusters on the single darkest step, and any
    ordering that groups small clusters together -- the legend's Similarity
    order does exactly that -- drew them as a run of near-black rows. Lightness
    now carries no information at all, which is what it was always meant to
    carry.

    The number of distinct colours is unchanged: 24 + 24 x 3 = 96. Only their
    distribution over the slots changes.

    Repeating at all is deliberate. The obvious alternative -- sampling a
    continuous colour map for as many colours as are needed -- is what an
    earlier version did, and it produced neighbouring clusters nobody could
    tell apart (see the palette comment above). The 24 curated colours stay as
    they are.

    Past 24 categories colour alone stops identifying anything, but colour was
    never the identifier here: the workflow is to find a group in the legend,
    switch the others off, drill in and tag it, and every cluster and every tag
    keeps its own legend row, checkbox and SCS node however many there are.
    Only the colour is shared.
    """
    palette = _SHAPE_MAP_PALETTE
    width = len(palette)
    i = max(0, int(i))
    if i < width:
        return palette[i]
    # Past the first lap the palette is walked SLOWLY and the lightness
    # quickly, so two consecutive slots never share a lightness. See the
    # docstring for why that matters.
    j = i - width
    # max(1, ...) so a one-element _SHAPE_MAP_CYCLE_FACTORS cannot divide by
    # zero here.
    laps = max(1, len(_SHAPE_MAP_CYCLE_FACTORS) - 1)
    factor = _SHAPE_MAP_CYCLE_FACTORS[1 + (j % laps)]
    base = palette[(j // laps) % width]
    return tuple(max(0, min(255, int(round(ch * factor)))) for ch in base)

# Version of the point-cloud COLOURING (palette + colour-assignment logic + the
# baked .scs structure). The map .scs files are cached ON DISK under scs_map/,
# keyed by projection inputs; a colour change alone does not change those inputs,
# so a stale .scs would keep being served with the old colours. This version is
# part of the .scs cache key, so bumping it invalidates every cached file.
#
# ★ BUMP THIS INTEGER whenever you change _SHAPE_MAP_PALETTE, the
# _shape_map_categorize / colour-assignment / grouping logic, or the JSON
# contract the scs_points tool consumes. Forgetting to bump it makes the server
# serve an out-of-date point cloud after a colour change (the bug this guards).
# v2: SCS is now one node per category group (was a single node, per-point RGBA).
# v3: palette extended to 24 colours + category/grouping revisions changed the
# node layout (tag mode now emits one node per tag with Untagged last); the v2
# bump was missed for those changes, so stale v2 .scs files kept their old node
# layout and the client hid the wrong group -- the Untagged cloud stayed visible
# in tag colouring even with its legend box off. Bumping to 3 retires them.
# v4: the "Optimal" cluster count (n_clusters == -1) now selects k by the
# inertia elbow instead of Calinski-Harabasz. The .scs cache is keyed by the
# raw n_clusters (-1), not the resolved k, so an Optimal .scs baked by the old
# logic would otherwise keep being served with its old cluster count while the
# recomputed /map legend shows the new one. Bumping to 4 retires those files.
# v5: cluster id -1 now means "outlier" and is baked as its own trailing node
# instead of being coloured as an ordinary cluster, so the node layout of any
# .scs built with a threshold differs from what v4 would have produced.
# v6: cluster colours now cycle the palette (darkened per lap) instead of
# collapsing everything past slot 24 into one grey "Other" node, so a map with
# more than 24 clusters has both different colours and a different node layout.
# v7: the aggregate rows are now sorted canonically by file id before the SVD,
# the principal components carry a fixed sign convention, and the clusters are
# renumbered largest-first. The same parameters therefore yield different
# coordinates, different cluster assignments and a different node layout than
# before, so every .scs baked at v6 or earlier is stale.
# v8: clusters are now renumbered by their mean projected position (left to
# right along the first principal axis) instead of by size, which proved too
# fragile to reproduce -- at k=32 a handful of parts moving reordered the
# legend. The same partition therefore gets different ids, different colours
# and a different node layout than v7 gave it.
# v9: clusters are now renumbered by member count descending, with cohesion
# descending and then mean projected position as tie-breaks, so the legend is
# sorted by the columns it prints and the largest families come first. Both
# numbers are read after the outlier pass, so the numbering now also depends on
# the Outliers threshold. The same partition therefore gets different ids,
# different colours and a different node layout than v8 gave it.
# v10: Ward colouring by cluster now bakes every point in one neutral grey
# (_SHAPE_MAP_WARD_NEUTRAL_RGB) instead of the per-cluster colour, because the
# client repaints the cloud from the tree anyway. Every Ward .scs baked at v9 or
# earlier still carries the old per-cluster colours, which flash before the
# repaint, so they are stale. k-means and Ward + tag colouring are unaffected by
# the change but share the key, so their caches rebuild once.
# v11: tag colours now cycle the palette (darkened per lap) in exactly the way
# v6 gave clusters, instead of collapsing everything past slot 24 into one grey
# "Other" node, so a map with more than 24 tags has both different colours and a
# different node layout. Maps with 24 tags or fewer are unchanged but share the
# key, so their caches rebuild once.
# v12: past slot 24 the lightness cycles and the hue steps, instead of the
#      other way round. Cluster ids are a size ranking, so the old assignment
#      made lightness track size and sank 184 of 256 clusters onto one step.
_SHAPE_MAP_COLORING_VERSION = 12

# The per-.scs build lock (_scs_map_build_locks) now lives in locks.py.


class ScsToolUnavailable(RuntimeError):
    """The scs_points executable is missing. Only the map/scs route depends on it."""


def _scs_points_exe_path() -> pathlib.Path:
    """Fixed location of the point-cloud authoring tool (not a configurable path)."""
    exe_name = "scs_points.exe" if os.name == "nt" else "scs_points"
    return config.APP_ROOT.joinpath("bin", exe_name)


def _scs_map_cache_max() -> int:
    """Max cached .scs files, from HOOPS_AI_SHAPE_MAP_SCS_CACHE_MAX (default 32)."""
    return _env_int(
        "HOOPS_AI_SHAPE_MAP_SCS_CACHE_MAX", _SCS_MAP_CACHE_MAX_DEFAULT, minimum=1
    )


def _scs_build_lock(key: tuple) -> "threading.Lock":
    """Return the per-key SCS build lock, creating it on first use."""
    with _scs_map_build_locks_mutex:
        lock = _scs_map_build_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _scs_map_build_locks[key] = lock
        return lock


def _shape_map_categorize(
    clusters: list, tags_lists: list, color_by: str
) -> tuple[list[int], list[dict[str, Any]]]:
    """Assign every point to one display CATEGORY and return the ordered category
    list. This is the single source of truth for Shape Map grouping and colour:
    the baked ``.scs`` (one node per category), the ``/map`` groups, and the
    legend all derive from it, so membership and colour always agree.

    Returns ``(cat_index, categories)`` where ``cat_index[i]`` is the index into
    *categories* for point *i* (in the caller's original point order) and each
    category is ``{"key", "label", "kind", "rgb", "count"}``. Categories are
    ordered deterministically: for clusters, by cluster id; for tags, by
    descending count then name, with ``Untagged`` last (kept LAST so a client
    can default it off). Every cluster AND every tag gets its own category
    however many there are -- the palette cycles rather than overflowing, so
    there is no ``Other`` bucket in either colouring -- and cluster ``-1`` is
    the outlier group, likewise emitted last as ``Outliers``. Colours come from
    the shared palette in that order; untagged keeps its own grey. Only
    non-empty categories are returned.
    """
    from collections import Counter

    if color_by == "tag":
        counts: "Counter[str]" = Counter()
        untagged = 0
        for tset in tags_lists:
            if tset:
                for t in tset:
                    counts[t] += 1
            else:
                untagged += 1
        ordered = [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        rank = {t: i for i, t in enumerate(ordered)}

        categories: list[dict[str, Any]] = []
        tag_to_cat: dict[str, int] = {}
        # Every tag gets its own category however many there are: the palette
        # cycles (see _shape_map_cluster_rgb), exactly as it does for clusters.
        # There is no "Other" bucket -- folding the 25th tag onward into one
        # grey row took away the legend row, the checkbox and the SCS node that
        # make a tag usable, and a tag is a name the user chose, so losing it is
        # worse than sharing a colour with the tag 24 slots above it.
        for i, t in enumerate(ordered):
            tag_to_cat[t] = len(categories)
            categories.append(
                {"key": f"tag:{t}", "label": t, "kind": "tag",
                 "rgb": _shape_map_cluster_rgb(i), "count": int(counts[t])}
            )
        untagged_cat: Optional[int] = None
        if untagged:
            untagged_cat = len(categories)
            categories.append(
                {"key": "untagged", "label": "Untagged", "kind": "untagged",
                 "rgb": _SCS_UNTAGGED_RGB, "count": int(untagged)}
            )

        cat_index: list[int] = []
        for tset in tags_lists:
            if tset:
                # A part may carry several tags but a point shows one colour: use
                # its highest-ranked (most common) tag, deterministically.
                primary = min(tset, key=lambda t: rank.get(t, len(ordered)))
                cat_index.append(tag_to_cat.get(primary, 0))
            else:
                cat_index.append(untagged_cat if untagged_cat is not None else 0)
        return cat_index, categories

    # color_by == "cluster"
    cluster_ids = [int(c) for c in clusters]
    counts_c: "Counter[int]" = Counter(cluster_ids)
    # -1 is the outlier group, not a cluster. Left in ordered_c it would sort
    # first, take the first palette colour and be labelled "Cluster 0"; it gets
    # its own category at the end instead, the way Untagged does.
    n_outliers = counts_c.pop(-1, 0)
    ordered_c = sorted(counts_c)
    categories = []
    cl_to_cat: dict[int, int] = {}
    # No "Other" bucket for clusters: every cluster gets its own category, so it
    # keeps its own legend row, checkbox and SCS node at any count. Colour is
    # what repeats past 24 (see _shape_map_cluster_rgb), not identity. Tags are
    # handled the same way above.
    for i, c in enumerate(ordered_c):
        cl_to_cat[c] = len(categories)
        categories.append(
            {"key": f"cluster:{c}", "label": f"Cluster {c + 1}", "kind": "cluster",
             "rgb": _shape_map_cluster_rgb(i), "count": int(counts_c[c])}
        )
    if n_outliers:
        # Last, so a client can default it off exactly as it does Untagged. The
        # remaining cluster ids are still 0..k-1, so "Cluster {c + 1}" above
        # needs no renumbering.
        cl_to_cat[-1] = len(categories)
        categories.append(
            {"key": "outliers", "label": "Outliers", "kind": "outliers",
             "rgb": _SHAPE_MAP_OVERFLOW_RGB, "count": int(n_outliers)}
        )
    cat_index = [cl_to_cat[c] for c in cluster_ids]
    return cat_index, categories


def _shape_map_color_assignment(
    clusters: list, tags_lists: list, color_by: str
) -> tuple[list[tuple[int, int, int]], list[dict[str, Any]]]:
    """Assign a categorical colour to every point and build the matching legend.

    Thin wrapper over :func:`_shape_map_categorize` so callers that only need
    per-point colours and the legend keep working. Returns ``(point_rgbs,
    legend)`` where *point_rgbs* is one ``(r, g, b)`` per point in order and
    *legend* is a list of ``{"label", "kind", "count", "rgb"}`` rows in category
    order (the legend drops the internal ``key``).
    """
    cat_index, categories = _shape_map_categorize(clusters, tags_lists, color_by)
    point_rgbs = [tuple(categories[ci]["rgb"]) for ci in cat_index]
    legend = [
        {"label": c["label"], "kind": c["kind"], "count": int(c["count"]),
         "rgb": list(c["rgb"])}
        for c in categories
    ]
    return point_rgbs, legend


def _touch(path: pathlib.Path) -> None:
    """Bump mtime so the LRU prune treats this file as most-recently-used."""
    try:
        os.utime(path, None)
    except OSError:
        pass


def _prune_scs_map_cache() -> None:
    """Keep only the newest _scs_map_cache_max() .scs files, by mtime."""
    try:
        files = sorted(
            config.SCS_MAP_DIR.glob("*.scs"), key=lambda p: p.stat().st_mtime, reverse=True
        )
    except OSError:
        return
    for stale in files[_scs_map_cache_max():]:
        try:
            stale.unlink()
        except OSError:
            pass


def clear_scs_map_cache() -> int:
    """Delete every cached point-cloud SCS (and temp leftovers) in scs_map/.

    scs_map/ is a regenerable disk cache. A cached .scs is keyed by a digest of
    the Shape Map parameters and _SHAPE_MAP_COLORING_VERSION, but NOT by the
    scs_points binary that authored it. Rebuilding scs_points against a different
    HOOPS Visualize Web SDK therefore produces different bytes under the same
    digest, so a stale .scs keeps being served and, if the SDK version no longer
    matches the web viewer engine, the client fails with "Failed to load the
    map." Clearing the cache at startup forces the first map request to rebuild
    with the current binary. Returns the number of files removed.
    """
    removed = 0
    try:
        stale = list(config.SCS_MAP_DIR.glob("*.scs"))
        stale += list(config.SCS_MAP_DIR.glob("*.scs.tmp"))
        stale += list(config.SCS_MAP_DIR.glob("*.json"))
    except OSError:
        return 0
    for path in stale:
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def compute_index_shape_map_scs(
    name: str,
    n_clusters: int = 0,
    dims: int = 3,
    tags: Optional[list[str]] = None,
    tag_mode: str = "any",
    tagged: Optional[bool] = None,
    color_by: str = "cluster",
    kind: str = "any",
    only_file_ids: Optional[list[str]] = None,
    outlier_threshold: float = 0.0,
    projection: str = _PROJECTION_PCA,
    clustering: str = _CLUSTERING_KMEANS,
    granularity: Optional[float] = None,
    ward_max: int = _SHAPE_MAP_WARD_K_MAX,
) -> pathlib.Path:
    """Build (or serve from cache) the point-cloud ``.scs`` for a named index's map.

    The ``.scs`` is authored from the SAME cached map payload the ``/map`` endpoint
    returns rather than an independent re-projection, so the point order — and thus
    the ``file_id`` correspondence a viewer relies on — is identical by
    construction. Colours are baked in, so the colour scheme (``cluster``/``tag``)
    is part of the cache key alongside ``(faiss path, mtime_ns, subset, dims,
    n_clusters)`` and, when the colours depend on tags, the tag sidecar ETag. A
    registration write bumps the ``.faiss`` mtime and so retires the cache
    automatically; a tag write is caught by the tag ETag, so a re-coloured map is
    rebuilt after tagging rather than served stale.

    Raises :class:`KeyError` for an unknown index and
    :class:`ScsToolUnavailable` when the ``scs_points`` binary is not present.

    *only_file_ids* restricts the cloud to a drill-down subset. It must be the
    same subset ``/map`` was given: the cloud is authored from that payload, so
    passing a different one here would break the point-order correspondence the
    viewer depends on. *outlier_threshold* likewise has to match, since it
    decides which points land in the baked ``Outliers`` node. So does
    *projection*, which decides where every point is in the first place. So does
    *clustering*, which decides which baked node each point ends up in. So does
    *granularity*, which decides how many nodes there are in the first place.
    And so does *ward_max*, the ceiling that granularity is read against.

    Raises :class:`UmapUnavailable` when ``projection="umap"`` is asked for and
    umap-learn is not installed, and :class:`WardTooLarge` when
    ``clustering="ward"`` is asked for on a map with too many points.
    """
    color_by = "tag" if str(color_by).lower() == "tag" else "cluster"
    # Total only: the per-stage breakdown belongs to the map this wraps, which
    # logs its own line. What this one adds is whether the cloud was already on
    # disk, and how the time split between the map and authoring the .scs.
    timing = _ShapeMapTiming()
    faiss_path = index_registry._index_faiss_path(name)
    if not faiss_path.exists():
        raise KeyError(f"Index '{name}' does not exist.")
    mtime_ns = faiss_path.stat().st_mtime_ns

    tag_filter = normalize_tag_filter(tags, tag_mode, tagged)
    kind = _normalize_map_kind(kind)
    subset_ids = _normalize_map_subset(only_file_ids)
    # Registering rather than merely hashing keeps the id alive in the LRU for
    # as long as the point cloud it describes is being fetched.
    subset_id = _register_map_subset(subset_ids)
    subset_sig = _shape_map_subset_sig(tag_filter, kind, subset_id)
    n_clusters = int(n_clusters)
    dims = int(dims)
    projection = _normalize_map_projection(projection)
    clustering = _normalize_map_clustering(clustering)

    # Checked up front, like compute_index_shape_map does, so a deployment
    # without umap-learn says so instead of failing somewhere inside the build.
    if projection == _PROJECTION_UMAP and not _shape_map_umap_available():
        raise UmapUnavailable(
            "projection=umap requires umap-learn, which is not installed in "
            "this deployment. Run: pip install umap-learn"
        )
    # The baked .scs colours depend on tag state whenever the map is coloured by
    # tag or restricted by a tag filter; tagging leaves the .faiss mtime
    # untouched, so the tag ETag must be in the disk-cache key or a stale .scs
    # would be reused after tagging (mirrors compute_index_shape_map's result
    # key). Cluster colouring with no tag filter is tag-independent, so its key
    # omits the ETag and is not churned by unrelated tagging.
    scs_tags_relevant = bool(tag_filter) or color_by == "tag"
    _tags_map, tags_etag = _shape_map_tags_snapshot(name)
    scs_tags_etag = tags_etag if scs_tags_relevant else ""

    key = (str(faiss_path), mtime_ns, subset_sig, dims, n_clusters, color_by,
           scs_tags_etag, _SHAPE_MAP_COLORING_VERSION)
    # Appended only when in use, so every .scs already on disk keeps the same
    # digest and is still served rather than silently rebuilt.
    outlier_threshold = _normalize_outlier_threshold(outlier_threshold)
    if outlier_threshold > 0.0:
        key = key + (outlier_threshold,)
    # The point colours are baked from the cluster assignment, so the k-means
    # settings have to reach the digest and the ETag too. Empty at the defaults,
    # which keeps every .scs already on disk addressable.
    key = key + _shape_map_kmeans_signature()
    # The points are baked at the coordinates the projection produced, so the
    # projection has to reach the digest too, or a UMAP request would be served
    # the PCA cloud already sitting on disk. Empty for PCA, which keeps every
    # existing .scs addressable.
    key = key + _shape_map_projection_signature(projection)
    # The clustering algorithm decides which group each point is baked into, so
    # it has to reach the digest as well. Empty for k-means, which is what every
    # .scs already on disk was built with.
    key = key + _shape_map_clustering_signature(clustering)
    # And so does the height it is cut at, for the same reason. Empty unless a
    # granularity was given, which keeps every existing .scs addressable.
    key = key + _shape_map_granularity_signature(clustering, granularity, ward_max)
    digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()
    out_path = config.SCS_MAP_DIR.joinpath(f"{digest}.scs")

    if out_path.exists():
        _touch(out_path)
        logger.info(
            "[SHAPEMAP] '%s' %s scs cloud served in %.2fs (file hit)",
            name, clustering, timing.total(),
        )
        return out_path

    exe = _scs_points_exe_path()
    if not exe.exists():
        raise ScsToolUnavailable(
            f"point-cloud tool not found at {exe}. Build it from tools/scs_points "
            "(see tools/scs_points/README.md) and place the executable in bin/."
        )

    build_lock = _scs_build_lock(key)
    with build_lock:
        if out_path.exists():
            _touch(out_path)
            logger.info(
                "[SHAPEMAP] '%s' %s scs cloud served in %.2fs (file hit)",
                name, clustering, timing.total(),
            )
            return out_path

        map_started = time.perf_counter()
        # Per-point tags are only needed to colour by tag; skip them otherwise to
        # keep the payload small.
        payload = compute_index_shape_map(
            name,
            n_clusters=n_clusters,
            dims=dims,
            tags=tags,
            tag_mode=tag_mode,
            tagged=tagged,
            include_tags=(color_by == "tag"),
            color_by=color_by,
            kind=kind,
            only_file_ids=subset_ids,
            outlier_threshold=outlier_threshold,
            projection=projection,
            clustering=clustering,
            granularity=granularity,
            ward_max=ward_max,
        )
        # The map itself is timed and logged by compute_index_shape_map; this
        # second number says how much of the .scs request was that map and how
        # much was authoring the cloud.
        map_seconds = time.perf_counter() - map_started
        author_started = time.perf_counter()

        config.SCS_MAP_DIR.mkdir(parents=True, exist_ok=True)
        # The tool authors one node per category group, in the same order and with
        # the same colour the /map payload carries. Identity (file_id / cluster) is
        # deliberately withheld: it is carried only by the point order within each
        # group, which the server and viewer both reconstruct from group.offset.
        json_groups = []
        # One neutral for the whole cloud when the tree owns the colours; see
        # _SHAPE_MAP_WARD_NEUTRAL_RGB. Decided once, outside the loop, so no
        # group can escape it. Ward + Colour by Tag is deliberately NOT included:
        # it draws no tree, so the tag colours baked here are the final ones.
        neutral = (
            _normalize_map_clustering(clustering) == "ward" and color_by == "cluster"
        )
        for g in payload["groups"]:
            off = int(g["offset"])
            cnt = int(g["count"])
            rgb = _SHAPE_MAP_WARD_NEUTRAL_RGB if neutral else g["rgb"]
            segment = payload["points"][off:off + cnt]
            json_groups.append({
                "node_name": g["node_name"],
                "r": int(rgb[0]), "g": int(rgb[1]), "b": int(rgb[2]),
                "points": [{"x": p["x"], "y": p["y"], "z": p["z"]} for p in segment],
            })
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", dir=str(config.SCS_MAP_DIR), delete=False, encoding="utf-8"
        ) as tf:
            json.dump({"version": 2, "groups": json_groups}, tf)
            json_path = pathlib.Path(tf.name)
        tmp_out = config.SCS_MAP_DIR.joinpath(f"{digest}.scs.tmp")
        try:
            proc = subprocess.run(
                [str(exe), "--input", str(json_path), "--output", str(tmp_out)],
                capture_output=True,
                text=True,
                timeout=_SCS_POINTS_TIMEOUT_S,
            )
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(
                    f"scs_points failed (exit {proc.returncode}): {detail}"
                )
            os.replace(str(tmp_out), str(out_path))
        finally:
            for leftover in (json_path, tmp_out):
                try:
                    leftover.unlink()
                except OSError:
                    pass

        _prune_scs_map_cache()
        _touch(out_path)
        logger.info(
            "[SHAPEMAP] '%s' %s scs cloud built in %.2fs (file miss, map %.2fs, "
            "author %.2fs, n=%d)",
            name, clustering, timing.total(), map_seconds,
            time.perf_counter() - author_started, len(payload.get("points") or ()),
        )
        return out_path


def shape_map_prewarm_enabled() -> bool:
    """Whether to prewarm the default map after a registration job.

    From HOOPS_AI_SHAPE_MAP_PREWARM, default on. Only indexes whose map has
    already been requested are warmed (see :func:`_has_cached_shape_map`), so a
    deployment that never calls ``/map`` never pays the SVD.
    """
    raw = os.getenv("HOOPS_AI_SHAPE_MAP_PREWARM", "").strip().lower()
    if not raw:
        return True
    return raw in ("1", "true", "yes", "on")


def _has_cached_shape_map(name: str) -> bool:
    """Whether any map (current or stale) is held for *name*.

    Presence is the signal that ``/map`` has actually been used on this index;
    nothing else populates these caches. A stale entry counts, because it is
    exactly what a registration job has just invalidated.
    """
    path_key = str(index_registry._index_faiss_path(name))
    with _shape_map_cache_mutex:
        if any(k[0] == path_key for k in _shape_map_agg):
            return True
        return any(k[0] == path_key for k in _shape_map_results)


def _rebuild_shape_map(name: str) -> None:
    """Body of the background prewarm. Never raises."""
    try:
        started = time.perf_counter()
        compute_index_shape_map(name)  # default map: full index, auto k, 3D
        logger.info(
            "[SHAPEMAP] prewarmed index '%s' in %.2fs",
            name, time.perf_counter() - started,
        )
    except KeyError:
        logger.info("[SHAPEMAP] prewarm skipped: index '%s' no longer exists", name)
    except Exception as exc:
        logger.warning("[SHAPEMAP] prewarm failed for index '%s': %s", name, exc)
    finally:
        with _shape_map_prewarm_mutex:
            _shape_map_prewarm_in_flight.discard(name)


def prewarm_shape_map(name: str) -> bool:
    """Recompute the default map for *name* in the background after a job.

    Mirrors :func:`prewarm_assembly_matcher`: gated on the feature having been
    used (a cached map exists) and on the enable flag, with only one prewarm per
    index in flight. Warms the same cache key the default ``/map`` request uses,
    so the next interactive call returns instantly. Returns whether a prewarm
    was started.
    """
    if not shape_map_prewarm_enabled():
        return False
    if not _has_cached_shape_map(name):
        logger.debug("[SHAPEMAP] prewarm skipped for index '%s': shape map unused", name)
        return False
    with _shape_map_prewarm_mutex:
        if name in _shape_map_prewarm_in_flight:
            return False
        _shape_map_prewarm_in_flight.add(name)
    thread = threading.Thread(
        target=_rebuild_shape_map,
        args=(name,),
        name=f"shape-map-prewarm-{name}",
        daemon=True,
    )
    thread.start()
    return True
