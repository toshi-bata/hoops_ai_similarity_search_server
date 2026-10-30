"""HTTP-facing facade and the cross-domain orchestrators for the index server.

After the batch 1-8 decomposition this module owns almost no logic of its own.
It is a thin facade: it re-exports the stable names of the sixteen domain
modules (locks, config, storage, embeddings, diagnostics, index_registry, tags,
search, context, assembly, shape_map, licensing, viewer, warmup, jobs, runtime)
-- each pulled in by a ``from <module> import (...)`` block below -- so that
``routers/``, ``tools/`` and ``tests/`` can keep reaching them as
``core.<name>``.

Three functions still live here on purpose: ``remove_from_index``,
``list_index_parts`` and ``delete_index``. Each orchestrates work that spans
sibling domains -- ``remove_from_index`` prunes the tags and context sidecars
alongside the index removal, ``delete_index`` also evicts the assembly-matcher
and tags caches, and ``list_index_parts`` filters a file-level index listing by
the tags sidecar. Those sibling domains (tags, context, assembly) sit *above*
any single index-owning module in the dependency DAG, so pushing one of these
orchestrators down into, say, ``index_registry`` would force that module to
call tags/context/assembly upward and create a cycle. Keeping the three in the
facade is what lets the dependency graph stay one-way (see "Why three
functions stay in the facade" in docs/architecture.md).

Dependency direction. It runs strictly downward: ``core`` imports the domain
modules, never the reverse. No first-class module imports ``core`` -- that
invariant is pinned at runtime by
``tests/test_module_layout.py::ImportHygiene``, which walks every top-level
module and fails if any of them imports ``core``.

Re-export convention (see "The re-export convention" in docs/architecture.md).
Names fall into three classes:

* (A) Swapped by a test, or reached by a router, as ``owner.<name>``. These are
  NOT re-exported here. Re-exporting would create a "double truth": a test
  patching ``owner.name`` would not touch a ``core.name`` bound at import time,
  so the facade and the owner could disagree. Reach an (A) name on its owning
  module -- e.g. ``jobs.get_index_job_store()``,
  ``licensing._find_license_shim_file`` or
  ``index_registry._get_named_searcher``.
* (B) Rebound at runtime with ``global`` (lazy singletons such as
  ``jobs._index_job_store``). Also not re-exported, for the same reason.
* (C) Stable, never swapped. Re-exported here by explicit name (the
  ``from owner import (...)`` blocks below) and reached by callers as
  ``core.<name>``.

So read a name through ``core.<name>`` only when it is a (C) re-export; reach
the swappable (A)/(B) names on their owning module instead. The layout test's
facade classes assert exactly this: ``core.X is owner.X`` for every (C) name
and ``hasattr(core, name) is False`` for every (A)/(B) name.
"""

import logging
import shutil
from typing import Any, Optional

logger = logging.getLogger(__name__)



# The lock hierarchy and its ordering rules live in one module (locks.py) so the
# cross-module ordering stays in one place. The names are imported explicitly --
# never ``import *``, which skips leading-underscore names.
# Re-exported: callers reach these as core.<name> (routers/, tools/, tests/).
from locks import (
    _PROGRESS_STDERR_LOCK,
    _index_locks,
    _index_locks_mutex,
    _get_index_lock,
    _tags_locks,
    _tags_locks_mutex,
    _get_tags_lock,
    _assembly_build_locks,
    _assembly_build_locks_mutex,
    _get_assembly_build_lock,
    _shape_map_build_locks,
    _shape_map_build_locks_mutex,
    _scs_map_build_locks,
    _scs_map_build_locks_mutex,
)

import config
import index_registry

# Configuration, the data root and every derived path live in config.py so the
# path/env surface is owned in one place. Callers reach them as ``config.<name>``
# (a module attribute) -- never ``from config import <name>``, so that a test
# swapping ``config.<name>`` to redirect the data root reaches every reader
# (see "The re-export convention" in docs/architecture.md). Only the exception
# classes are re-exported here: they are never swapped, and routers/ catch them
# as ``core.EnvConfigError``
# / ``core.PathConfigError``.
from config import EnvConfigError, PathConfigError
# ``_env_int`` is a stdlib-only env helper that lives in config.py; it is
# never swapped by a test, so a plain ``from`` import is safe here and lets
# core's many bare ``_env_int(...)`` call sites keep working unchanged.
from config import _env_int

# CAD file storage and ingest live in storage.py. The four swappable names
# (find_persistent_CAD_file, get_CAD_shared_dir, read_too_heavy_files,
# upload_CAD_file_persistent) are reached as ``storage.<name>`` and are NOT
# re-exported. The names below are never swapped, so they are re-exported for
# existing ``core.<name>`` / ``except core.Zip*`` call sites (see section 8.4).
from storage import (
    CAD_ALLOWED_EXTENSIONS,
    ZipSlipError,
    ZipSizeLimitError,
    ZipFileLimitError,
    UploadSizeLimitError,
    _UPLOAD_CHUNK_BYTES,
    _MAX_UPLOAD_BYTES_DEFAULT,
    _FILE_ID_RE,
    _EXISTS_MAX_IDS_DEFAULT,
    _max_upload_bytes,
    exists_max_ids,
    known_persistent_CAD_file_ids,
    check_persistent_CAD_files,
    map_persistent_CAD_files,
    map_duplicate_CAD_filenames,
    delete_persistent_CAD_file,
    list_persistent_CAD_files,
    server_paths_enabled,
    resolve_server_paths,
    ZIP_MAX_TOTAL_BYTES,
    ZIP_MAX_FILES,
    zip_max_total_bytes,
    zip_max_files,
    _BytesUploadFile,
    extract_zip_cad_files,
    get_shared_CAD_file,
)

# Shape embedding, the two-tier cache and model/checkpoint resolution live in
# embeddings.py. The six swappable names (get_embedder, compute_embedding,
# compare_embeddings, _embed_cache_max_bytes, _embedding_memory_cache,
# _embedding_cache_bytes) are reached as ``embeddings.<name>`` and are NOT
# re-exported; neither is the rebound singleton (_embedder). The names below are
# stable and never rebound, so they are re-exported for existing ``core.<name>``
# callers in routers, tools and tests (see "The re-export convention" in
# docs/architecture.md).
from embeddings import (
    _EMBED_CACHE_MAX_MB_DEFAULT,
    _embedding_cache_mutex,
    _embed_entry_bytes,
    _embedding_cache_get,
    _embedding_cache_put,
    _l2_normalize,
    clear_embedding_memory_cache,
    log_compute_environment,
    get_required_env,
    require_path,
    get_sdk_dir,
    sdk_dir_configured,
    get_packages_dir,
    ModelResolutionError,
    resolve_bundled_checkpoint,
    resolve_checkpoint_path,
    ckpt_sha256,
    _ckpt_sha256_cache,
    _INTRAOP_ENV_VARS,
    _INTRAOP_THREADS_DEFAULT,
    intraop_threads,
    intraop_thread_limit,
)

# ``warmup_indexes_setting`` is a stdlib-only env helper owned by config.py
# (like ``_search_timing_enabled``); it is never swapped, so a plain ``from``
# import is safe and re-exports it for the warmup code here and for tests
# that read ``core.warmup_indexes_setting``. ``warmup_clustering_setting`` is
# the same kind of helper, re-exported for ``routers/client_config.py``, which
# answers with the clustering warmup makes hot; ``warmup_ward_max_setting`` is
# re-exported for the same router, which answers with the Ward ceiling warmup
# used; ``warmup_stages_setting`` likewise, because whether 'map' is among the
# stages is what decides whether any clustering is warm at all.
from config import (
    warmup_clustering_setting,
    warmup_indexes_setting,
    warmup_stages_setting,
    warmup_ward_max_setting,
)

# ``demo_enabled`` is the same kind of stdlib-only env helper: it reads
# HOOPS_AI_DEMO on every call, so a test sets the variable rather than swapping
# the function, and a plain ``from`` import is safe. Re-exported so
# ``routers/client_config.py`` reaches it as ``core.demo_enabled`` like every
# other router reaches the server's behaviour through the facade.
from config import demo_enabled

# The named-index registry and the public index API live in index_registry.py.
# The fourteen swappable names (``_load_named_index``, ``_get_named_searcher``,
# ``_load_index_schema``, ``_resolve_index_asset``,
# ``add_to_index``, ``_save_named_index_atomic``, ``_registered_index_ids``,
# ``_named_indexes``, ``_index_faiss_path``, ``_index_cache_max_bytes``,
# ``_index_entry_bytes``, ``list_indexes``, ``_load_shape_index_pickle_compat``,
# ``_embed_bodies_for_index``) are reached as ``index_registry.<name>`` and are
# NOT re-exported. The names below are stable and are re-exported for existing
# ``core.<name>`` callers in routers, tools and tests, and for the orchestration
# operators (delete_index, remove_from_index) that stay in core (see "The
# re-export convention" in docs/architecture.md).
from index_registry import (
    _INDEX_SCHEMA_VERSION,
    _INDEX_NAME_RE,
    _INDEX_CACHE_MAX_MB_DEFAULT,
    _named_indexes_mutex,
    _NAMED_SEARCHER_CACHE_MAX,
    _named_searchers,
    _validate_index_name,
    _index_base_path,
    _get_embedder_dim,
    _evict_named_indexes_to_cap,
    _store_named_index,
    create_index,
    _EMBED_ERROR_DEFAULT,
    _embed_error_detail,
    _MAX_WORKERS_DEFAULT,
    _MODEL_FOOTPRINT_MB_DEFAULT,
    _MIN_FILES_PARALLEL_DEFAULT,
    _HOST_RAM_RESERVE_BYTES,
    _HEAVY_FOOTPRINT_FACTOR,
    _RETRY_TIME_LIMIT_SECONDS,
    _SDK_DEFAULT_TIME_LIMIT_SECONDS,
    _size_workers,
    compute_auto_workers,
    embed_time_limit,
    effective_embed_time_limit,
    plan_embedding_run,
    read_error_summary_reasons,
    _norm_path_key,
    build_embed_specifications,
    _TqdmProgressStderr,
    capture_embed_progress,
    _evict_named_searchers,
    _public_metadata,
    _flatten_body_hit_objects,
    _flatten_body_hits,
    _TAG_FILTER_OVERFETCH,
    _TAG_FILTER_OVERFETCH_MAX,
    _tag_filtered_fetch_k,
    get_index_stats,
    get_index_duplicate_names,
    check_index_model_consistency,
    _require_index_exists,
    _index_thumbnails_dir,
    _index_model_sidecar_path,
    _index_json_path,
    _index_scs_dir,
    _save_index_schema,
    _obb_meta_value,
    _compute_faiss_stats,
    SchemaVersionError,
)

# The index tag sidecar and the tag-filter API live in tags.py. The one
# swappable name (``_read_tags_document``) is reached as ``tags._read_tags_document``
# and is NOT re-exported: a test swaps it to prove the search-cache key reads
# the sidecar, and a re-export would bind the old object and hide the swap
# (see section 8.4). The names below are stable and are re-exported for existing
# ``core.<name>`` callers in routers, tools and tests, and for the orchestrators
# (list_index_parts, delete_index, remove_from_index) that stay in core.
from tags import (
    _TAGS_SCHEMA_VERSION,
    _TAG_MAX_LENGTH,
    _TAGS_PER_PART_DEFAULT,
    _TAGS_CACHE_MAX,
    _tags_cache,
    _tags_cache_mutex,
    TagsFileError,
    TagsPreconditionFailed,
    _index_tags_path,
    max_tags_per_part,
    normalize_tag,
    _normalize_tag_list,
    _warn_on_case_collision,
    _iter_all_tags,
    _empty_tags_document,
    _serialize_tags_document,
    _tags_etag,
    _parse_tags_document,
    _write_tags_document,
    _evict_tags_cache,
    etag_matches,
    _mutate_tags,
    _stamp,
    get_index_tags,
    get_part_tags,
    set_part_tags,
    _bulk_tag_update,
    assign_tag,
    unassign_tag,
    delete_tag,
    prune_tags,
    tags_by_part,
    _shape_map_tags_snapshot,
    normalize_tag_filter,
    tag_filter_predicate,
    tag_filter_diagnostics,
    _TAG_VOCABULARY_MAX,
    _tag_vocabulary_path,
    _read_tag_vocabulary,
    _write_tag_vocabulary,
    get_tag_vocabulary,
    add_tag_vocabulary,
    remove_tag_vocabulary,
    _record_tags_in_vocabulary,
)


# ``search_index`` and ``_search_cache_enabled`` are the two (A) names in
# search.py: routers reach them as ``core.<name>`` and tests swap those same
# attributes, so they are reached as ``search.<name>`` and are NOT re-exported
# here (a re-export would create a second object and hide the swap; section 8.4).
# The stable (C) names below are re-exported for existing ``core.<name>`` callers.
from search import (
    _SEARCH_KINDS,
    _SEARCH_CACHE_MAX_DEFAULT,
    _search_cache,
    _search_cache_mutex,
    _search_cache_capacity,
    clear_search_cache,
    _tag_filter_key,
    _search_cache_key,
    _search_cache_get,
    _search_cache_put,
)

# The context-layer prediction lives in context.py. Every name is stable and is
# re-exported for existing ``core.<name>`` callers (routers, tools, tests) and
# for ``remove_from_index`` in core, which calls ``prune_context`` downward.
from context import (
    _PREDICT_KEYS_DEFAULT,
    max_predict_keys,
    _normalize_predict_keys,
    _context_jsonify,
    predict_context,
    get_index_context_summary,
    get_part_context,
    set_part_context,
    prune_context,
)

# Assembly-to-assembly search lives in assembly.py. ``_get_assembly_matcher``,
# ``prewarm_assembly_matcher`` and ``search_assembly_index`` are the three (A)
# names: routers reach ``assembly.search_assembly_index`` and tests swap all
# three ``assembly.<name>`` attributes to install fakes, so they are reached as
# ``assembly.<name>`` (module attribute) and are NOT re-exported here -- a
# re-export would bind a second object and hide the swap (section 8.4). The core
# orchestrators ``_warmup_one`` and ``run_index_add_job`` call them as
# ``assembly._get_assembly_matcher`` / ``assembly.prewarm_assembly_matcher``. The
# stable (C) names below are re-exported for existing ``core.<name>`` callers and
# for ``delete_index`` in core (``_evict_assembly_matchers``).
from assembly import (
    _ASSEMBLY_COVERAGE_MODES,
    _MIN_BODIES_FOR_ASSEMBLY,
    _ASSEMBLY_MATCHER_CACHE_MAX,
    _assembly_matchers,
    _assembly_matchers_mutex,
    _assembly_prewarm_in_flight,
    _assembly_prewarm_mutex,
    _ASSEMBLY_SEARCH_JOBS_DEFAULT,
    _ASSEMBLY_CANDIDATE_K_MAX,
    _assembly_search_jobs,
    _evict_assembly_matchers_locked,
    _evict_assembly_matchers,
    _lookup_assembly_matcher,
    _has_cached_assembly_matcher,
    assembly_prewarm_enabled,
    _rebuild_assembly_matcher,
    _assembly_metadata_by_id,
    _assembly_hit,
    _assembly_self_hit,
)


# The index shape map (a JSON 2D map and a point-cloud .scs) lives in
# shape_map.py. ``_load_index_body_batch``, ``_shape_map_filemeta``,
# ``compute_index_shape_map`` and ``compute_index_shape_map_scs`` are the four
# (A) names: tests swap those ``shape_map.<name>`` attributes to install fakes,
# and routers reach ``shape_map.compute_index_shape_map`` /
# ``shape_map.compute_index_shape_map_scs``, so they are NOT re-exported here -- a
# re-export would bind a second object and hide the swap (section 8.4). The core
# orchestrator ``_warmup_one`` calls them as ``shape_map.compute_index_shape_map``
# / ``shape_map.compute_index_shape_map_scs``. The stable (C) names below are
# re-exported for existing ``core.<name>`` callers and for the core orchestrators
# (``run_index_add_job`` -> prewarm_shape_map, ``run_startup_maintenance`` ->
# clear_scs_map_cache, ``_warmup_one`` -> ScsToolUnavailable).
from shape_map import (
    _SHAPE_MAP_AGG_CACHE_MAX,
    _shape_map_agg,
    _SHAPE_MAP_RESULT_CACHE_MAX,
    _shape_map_results,
    _shape_map_cache_mutex,
    _SHAPE_MAP_SUBSET_CACHE_MAX,
    _shape_map_subsets,
    _shape_map_subset_predicate,
    _normalize_map_subset,
    _normalize_outlier_threshold,
    _shape_map_outlier_assign,
    _shape_map_outlier_ceiling,
    _shape_map_centroid_sims,
    _shape_map_cohesion,
    _shape_map_centres,
    _SHAPE_MAP_CYCLE_FACTORS,
    _shape_map_cluster_rgb,
    _register_map_subset,
    _resolve_map_subset,
    _shape_map_prewarm_in_flight,
    _shape_map_prewarm_mutex,
    _shape_map_l2norm_rows,
    _shape_map_project,
    _shape_map_run_kmeans,
    _shape_map_kmeans_max_points_per_centroid,
    _shape_map_kmeans_nredo,
    _shape_map_kmeans_spherical,
    _shape_map_kmeans_init,
    _shape_map_kmeans_init_method,
    _shape_map_kmeans_signature,
    _normalize_map_projection,
    _shape_map_umap_available,
    _shape_map_projection_signature,
    UmapUnavailable,
    _normalize_map_clustering,
    _shape_map_clustering_signature,
    WardTooLarge,
    _shape_map_elbow_k,
    _shape_map_choose_optimal_k,
    _shape_map_cluster,
    _compute_shape_map_points,
    _shape_map_build_lock,
    _shape_map_legend,
    _shape_map_build_payload,
    _normalize_map_kind,
    _compose_keep,
    _shape_map_kind_predicate,
    _shape_map_subset_sig,
    _SCS_POINTS_TIMEOUT_S,
    _SCS_MAP_CACHE_MAX_DEFAULT,
    _SCS_UNTAGGED_RGB,
    _SHAPE_MAP_PALETTE,
    _SHAPE_MAP_OVERFLOW_RGB,
    _SHAPE_MAP_COLORING_VERSION,
    ScsToolUnavailable,
    _scs_points_exe_path,
    _scs_map_cache_max,
    _scs_build_lock,
    _shape_map_categorize,
    _shape_map_color_assignment,
    _touch,
    _prune_scs_map_cache,
    clear_scs_map_cache,
    shape_map_prewarm_enabled,
    _has_cached_shape_map,
    _rebuild_shape_map,
    prewarm_shape_map,
)


# The HOOPS AI license bootstrap lives in licensing.py. ``_find_license_shim_file``
# is the one (A) name: tests monkeypatch ``licensing._find_license_shim_file`` to
# simulate a present / absent / mismatched shim, so it is NOT re-exported here -- a
# re-export would leave ``init_hoops_license`` calling a different object than the
# test swapped (section 8.4). The two stable (C) names are re-exported for existing
# ``core.<name>`` callers (``main.py`` calls ``core.init_hoops_license``).
from licensing import _shim_expected_python, init_hoops_license
from viewer import export_scs_for_part
from warmup import (
    _WARMUP_MAX_DEFAULT,
    _warmup_max,
    _resolve_warmup_targets,
    _warmup_one,
    run_startup_warmup,
    start_startup_warmup,
)
from jobs import (
    JOB_KIND_INDEX_ADD,
    _JOB_MAX_CONCURRENCY_DEFAULT,
    _JOB_TTL_DAYS_DEFAULT,
    _JOB_MAX_RECORDS_DEFAULT,
    _index_job_store_lock,
    job_max_concurrency,
    job_ttl_seconds,
    job_max_records,
    _RETRY_REASON_MARKERS,
    annotate_failures,
    _write_add_job_report,
    run_index_add_job,
)
from runtime import (
    _json_safe,
    log_resolved_configuration,
    _OUT_TTL_HOURS_DEFAULT,
    _EMBEDDINGS_CACHE_TTL_DAYS_DEFAULT,
    viewer_output_ttl_seconds,
    embeddings_cache_ttl_seconds,
    sweep_directory,
    ensure_runtime_dirs,
    run_startup_maintenance,
)


# The lock hierarchy and its LOCK ORDERING rules live in locks.py (imported at
# the top of this module); _index_locks and _get_index_lock are defined there.


# ---------------------------------------------------------------------------
# Named index public API
# ---------------------------------------------------------------------------


# _PROGRESS_STDERR_LOCK now lives in locks.py.


def remove_from_index(name: str, part_ids: list[str]) -> dict[str, Any]:
    """Delete *part_ids* from the named index, persist, and remove their assets.

    Raises ``SchemaVersionError`` for legacy (schema v1) indexes.
    """
    _validate_index_name(name)
    schema = index_registry._load_index_schema(name)
    lock = _get_index_lock(name)
    with lock:
        vs = index_registry._load_named_index(name)
        if schema["schema_version"] < _INDEX_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Index '{name}' uses legacy schema v{schema['schema_version']} "
                f"and must be rebuilt as schema v{_INDEX_SCHEMA_VERSION} "
                f"before removing parts."
            )
        vs.delete(part_ids)
        index_registry._save_named_index_atomic(name, vs)

    # Remove thumbnails + stream caches outside the lock (non-fatal)
    thumb_dir = _index_thumbnails_dir(name)
    scs_dir = _index_scs_dir(name)
    for pid in part_ids:
        (thumb_dir / f"{pid}.png").unlink(missing_ok=True)
        (scs_dir / f"{pid}.scs").unlink(missing_ok=True)

    # Tags are keyed by file_id, so entries for removed parts would otherwise
    # linger forever and keep inflating tag counts. Non-fatal: the parts are
    # already gone from the index either way.
    try:
        prune_tags(name, part_ids)
    except Exception as exc:
        logger.warning("[TAGS] could not prune tags for index '%s': %s", name, exc)

    # Context records are keyed by file_id too, so they need the same cleanup.
    try:
        prune_context(name, part_ids)
    except Exception as exc:
        logger.warning("[CONTEXT] could not prune context for index '%s': %s", name, exc)

    return {"name": name, "removed": len(part_ids), "index_count": len(vs.get_ids())}


def list_index_parts(
    name: str,
    offset: int = 0,
    limit: int = 100,
    kind: Optional[str] = None,
    tags: Optional[list[str]] = None,
    tag_mode: str = "any",
    tagged: Optional[bool] = None,
) -> dict[str, Any]:
    """Return a paginated, file-level listing of the named index.

    ``iter_metadata()`` is walked exactly once and collapsed to one item per
    file_id. *limit* is clamped to ``[1, 2000]`` (default 100) and *offset* to
    ``>= 0``. *kind* (``"part"``/``"assembly"``) filters when supplied.

    *tags* / *tag_mode* / *tagged* filter on the tag sidecar. Filtering happens
    before pagination, so ``total`` counts what matched rather than what the
    index holds.

    Each item: ``id, filename, kind, bodies, registered_at, has_thumbnail,
    has_scs, tags`` (the router turns the boolean flags into absolute URLs).

    Raises ``KeyError`` when the index does not exist.
    """
    _validate_index_name(name)
    limit = max(1, min(int(limit), 2000))
    offset = max(0, int(offset))
    tag_filter = normalize_tag_filter(tags, tag_mode, tagged)

    lock = _get_index_lock(name)
    with lock:
        vs = index_registry._load_named_index(name)
        order: list[str] = []
        by_fid: dict[str, dict[str, Any]] = {}
        for raw in vs.iter_metadata():
            md = raw if isinstance(raw, dict) else (getattr(raw, "metadata", None) or {})
            fid = md.get("file_id")
            if fid is None:
                continue
            if fid not in by_fid:
                order.append(fid)
                by_fid[fid] = {
                    "id": fid,
                    "filename": md.get("filename"),
                    "kind": md.get("kind") or "part",
                    "bodies": int(md.get("bodies") or 1),
                    "registered_at": md.get("registered_at"),
                    "thumbnail": md.get("thumbnail"),
                    "scs": md.get("scs"),
                }

    items = [by_fid[f] for f in order]
    if kind:
        items = [it for it in items if it["kind"] == kind]
    # Read the sidecar once, outside the index lock, and reuse it for both the
    # filter and the per-item tag lists.
    by_part = tags_by_part(name)
    if tag_filter:
        matches = tag_filter_predicate(name, tag_filter)
        items = [it for it in items if matches(it["id"])]
    total = len(items)
    page = items[offset : offset + limit]

    # Resolve asset existence (path-traversal safe) and drop the raw rel paths.
    result_items: list[dict[str, Any]] = []
    for it in page:
        has_thumbnail = (
            it.get("thumbnail") is not None
            and index_registry._resolve_index_asset(name, it["id"], "thumbnail") is not None
        )
        has_scs = (
            it.get("scs") is not None
            and index_registry._resolve_index_asset(name, it["id"], "scs") is not None
        )
        result_items.append(
            {
                "id": it["id"],
                "filename": it["filename"],
                "kind": it["kind"],
                "bodies": it["bodies"],
                "registered_at": it["registered_at"],
                "has_thumbnail": has_thumbnail,
                "has_scs": has_scs,
                "tags": by_part.get(it["id"], []),
            }
        )

    return {"total": total, "offset": offset, "limit": limit, "items": result_items}


def delete_index(name: str) -> dict[str, Any]:
    """Delete the named index, its on-disk FAISS files, and all stored thumbnails.

    Raises KeyError if the index does not exist.
    """
    _validate_index_name(name)
    lock = _get_index_lock(name)
    with lock:
        faiss_file = index_registry._index_faiss_path(name)
        if not faiss_file.exists():
            raise KeyError(f"Index '{name}' does not exist.")
        with _named_indexes_mutex:
            index_registry._named_indexes.pop(name, None)
        # Drop cached searchers so a deleted index stops holding memory.
        _evict_named_searchers(name)
        _evict_assembly_matchers(name)
    _evict_tags_cache(name)

    # Remove the entire index directory (faiss + meta + thumbnails)
    index_dir = config.INDEXES_DIR / name
    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)

    return {"name": name, "deleted": True}
