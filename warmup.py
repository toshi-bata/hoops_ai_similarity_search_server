"""Opt-in startup warmup for named indexes.

Owns the background warmup that a fresh process runs so the first real user
does not pay the cold cost of building a searcher, shape map and assembly
matcher. Warmup is opt-in via ``HOOPS_AI_WARMUP_INDEXES`` and capped by
``HOOPS_AI_WARMUP_MAX``; with nothing configured the server does no extra work.

:func:`_warmup_one` is the top-of-stack orchestrator: it drives the assembly,
shape-map and search domains for a single index using the same build-locked
entry points a live request would, so a request arriving mid-warmup shares the
in-flight build instead of starting a second one.

Dependency direction is one-way and this module sits at the very top of the
import graph: ``core`` imports it, and **nothing else does**. It reaches its
dependencies through their owning modules -- ``index_registry`` (searcher,
schema, listing), ``assembly`` (matcher) and ``shape_map`` (map + SCS) -- and
imports :mod:`config` symbols directly. It never imports ``core``.

All six public names are stable (C) names re-exported from ``core`` by
identity. There are no (A) names here: the domain entry points warmup calls
are owned and swapped on *their* modules (``shape_map.compute_index_shape_map``
etc.), which is why the startup-warmup tests patch them there.
"""


import logging
import threading
import time
from typing import Optional

import assembly
import index_registry
import shape_map
from config import (
    _env_int,
    warmup_clustering_setting,
    warmup_indexes_setting,
    warmup_stages_setting,
    warmup_ward_max_setting,
)
from index_registry import _INDEX_SCHEMA_VERSION
from shape_map import ScsToolUnavailable


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Startup warmup (opt-in via HOOPS_AI_WARMUP_INDEXES)
#
# A fresh process pays the cold cost of the first searcher / shape map / matcher
# build. This warms selected indexes in the background at startup so the first
# real user does not. It is opt-in: with nothing configured the server does
# nothing extra, so "started but pegged the CPU for a minute" is never a
# surprise on a redistributed deployment. The administrator names the indexes.
# ---------------------------------------------------------------------------

_WARMUP_MAX_DEFAULT = 1

# ---------------------------------------------------------------------------
# The client's own defaults, which the map stages must use.
#
# Warming a different population than the UI asks for warms a cache key no
# request will ever hit: the map is cached per (kind, n_clusters, dims,
# colouring, projection, clustering), so a single mismatched value means the
# first user still pays the full build. KEEP THESE IN STEP WITH THE CLIENT --
# changing one side alone makes warmup useless rather than wrong, which is the
# hardest kind of mistake to notice:
#
#   _WARMUP_KIND         <- `let kind = 'part'` in web/src/shape-map.js
#                           (the Show: Parts default)
#   _WARMUP_DIMS         <- MAP_DIMS in web/src/shape-map.js
#   _WARMUP_COLOR_BY     <- `let colorBy = 'cluster'` in web/src/shape-map.js
#   _WARMUP_PROJECTION   <- `let projection = 'pca'` in web/src/shape-map.js
#   _WARMUP_INCLUDE_TAGS <- the `includeTags` shapeMap() is called with in
#                           web/src/shape-map.js
#   _WARD_GRANULARITY    <- WARD_GRANULARITY in web/src/shape-map.js
#
# Neither the k-means count nor Ward's ceiling is a constant here. Both come
# from shape_map.shape_map_auto_cluster_count() applied to the index's own part
# count, and the client is handed the same number as default_clusters by
# /index/{name}/stats -- one rule, read by both sides, so they cannot drift.
# HOOPS_AI_WARMUP_WARD_MAX overrides the ceiling for every index when an
# operator sets it; /client-config hands the client that same override (see
# warmup_ward_max_setting in config.py).
# ---------------------------------------------------------------------------

_WARMUP_KIND = "part"
_WARMUP_DIMS = 3
_WARMUP_COLOR_BY = "cluster"
_WARMUP_PROJECTION = "pca"
# compute_index_shape_map defaults this to False, but the client always asks
# for True and include_tags is part of the result cache key -- so leaving it at
# the default warmed an entry no request could ever hit.
_WARMUP_INCLUDE_TAGS = True
_WARD_GRANULARITY = 1.0


def _warmup_max() -> int:
    """Cap on how many indexes to warm, from HOOPS_AI_WARMUP_MAX (default 1).

    Warming every index would make a many-index deployment start very slowly,
    so only the first *N* configured names are warmed.
    """
    return _env_int("HOOPS_AI_WARMUP_MAX", _WARMUP_MAX_DEFAULT, minimum=1)


def _resolve_warmup_targets() -> list[str]:
    """Validate, dedupe and cap the configured warmup targets to existing indexes."""
    requested = warmup_indexes_setting()
    if not requested:
        return []
    existing = {idx["name"] for idx in index_registry.list_indexes()}
    targets: list[str] = []
    for name in requested:
        if name in targets:
            continue
        if name not in existing:
            logger.warning("[WARMUP] skipping unknown index '%s'", name)
            continue
        targets.append(name)
    limit = _warmup_max()
    if len(targets) > limit:
        logger.info(
            "[WARMUP] %d index(es) resolved but HOOPS_AI_WARMUP_MAX=%d; warming the first %d",
            len(targets), limit, limit,
        )
        targets = targets[:limit]
    return targets


def _cluster_count_note(ward: bool, n_clusters: int, ward_max: int) -> str:
    """How many clusters a warmed stage was built with, for the log.

    Which number matters depends on the clustering, and the cache key is built
    from it. Without it in the log there is no way to tell, after the fact,
    whether a cold first view came from warming a count the client never asks
    for.
    """
    return f"ward_max={ward_max}" if ward else f"n_clusters={n_clusters}"


def _warmup_map_stages(name: str, stages: list[str], clusterings: list[str]) -> None:
    """Run the map and map-SCS stages for every configured clustering.

    One pass per clustering, because each is its own cache key: warming k-means
    does nothing for a user who opens the map in Ward, and vice versa.
    """
    # The one rule both sides read. The client gets this same number as
    # default_clusters from /index/{name}/stats, so the map it opens is the map
    # warmed here. Counted over the single parts because the map opens on
    # Show: Parts, which is the population that becomes points.
    try:
        auto = shape_map.shape_map_auto_cluster_count(
            index_registry.get_index_stats(name)["single_part"]
        )
    except Exception as exc:
        logger.warning(
            "[WARMUP] '%s' could not size its clusters (%s); skipping the map "
            "stages rather than warming a count no request will ask for",
            name, exc,
        )
        return
    for clustering in clusterings:
        # True for every clustering that builds a tree, not for Ward alone:
        # average linkage takes the same granularity, ceiling and folded tree.
        # k-means is the only one that does neither.
        ward = clustering != "kmeans"
        # ONE ceiling for all three tree calls below -- the map, its SCS and the
        # tree. Read once here so they cannot drift: the map's clusters are the
        # tree's leaves, and a mismatch leaves the two describing different
        # families (the same reason /map's ward_max and /map/tree's k_max must
        # agree). An operator's fixed ceiling wins over the automatic one.
        ward_max = warmup_ward_max_setting() or auto
        n_clusters = auto
        # Ward always cuts at the deepest level the client asks for, and takes
        # no cluster count; k-means takes the count and no granularity.
        params: dict = {
            "kind": _WARMUP_KIND,
            "dims": _WARMUP_DIMS,
            "color_by": _WARMUP_COLOR_BY,
            "projection": _WARMUP_PROJECTION,
            "include_tags": _WARMUP_INCLUDE_TAGS,
            "clustering": clustering,
        }
        if ward:
            params["granularity"] = _WARD_GRANULARITY
            params["ward_max"] = ward_max
        else:
            params["n_clusters"] = n_clusters

        if "map" in stages:
            # Stage 2: shape map (SVD, ~15 s; Ward's tree ~4 s on 8,000 parts
            # since its distances moved to a blocked matrix product).
            try:
                started = time.perf_counter()
                shape_map.compute_index_shape_map(name, **params)
                logger.info(
                    "[WARMUP] '%s' shape map (%s, %s) ready in %.2fs",
                    name, clustering, _cluster_count_note(ward, n_clusters, ward_max),
                    time.perf_counter() - started,
                )
            except Exception as exc:
                logger.warning(
                    "[WARMUP] '%s' shape map (%s) failed: %s", name, clustering, exc
                )

            # A tree clustering's panel asks for the folded tree as soon as the
            # map lands, so it is part of warming it. Built from the linkage
            # the map above already cached, which is why it is not a stage of
            # its own.
            if ward:
                try:
                    started = time.perf_counter()
                    shape_map.compute_index_shape_map_tree(
                        name, kind=_WARMUP_KIND, k_max=ward_max,
                        clustering=clustering,
                    )
                    logger.info(
                        "[WARMUP] '%s' %s tree (k_max=%d) ready in %.2fs",
                        name, clustering, ward_max, time.perf_counter() - started,
                    )
                except Exception as exc:
                    logger.warning(
                        "[WARMUP] '%s' %s tree failed: %s", name, clustering, exc
                    )

        if "scs" in stages:
            # Stage 2b: the map's point-cloud SCS (~0.4 s), built from the map
            # just warmed above, so the first /map/scs request returns from
            # cache. When the scs_points tool is not installed this is expected
            # to be unavailable; log it at info level and carry on -- a missing
            # tool must not fail warmup.
            try:
                started = time.perf_counter()
                # include_tags is not a parameter of the .scs entry point: it
                # decides for itself (include_tags=(color_by == "tag")) because
                # the tags only matter there when they are what is being baked
                # into the cloud. Passing it would be a TypeError.
                scs_params = {k: v for k, v in params.items() if k != "include_tags"}
                shape_map.compute_index_shape_map_scs(name, **scs_params)
                logger.info(
                    "[WARMUP] '%s' map SCS (%s, %s) ready in %.2fs",
                    name, clustering, _cluster_count_note(ward, n_clusters, ward_max),
                    time.perf_counter() - started,
                )
            except ScsToolUnavailable:
                logger.info(
                    "[WARMUP] '%s' map SCS skipped: point-cloud tool not installed", name
                )
            except Exception as exc:
                logger.warning(
                    "[WARMUP] '%s' map SCS (%s) failed: %s", name, clustering, exc
                )


def _warmup_one(name: str) -> None:
    """Warm the assembly matcher, shape map and searcher for one index.

    Uses the same top-level, build-locked entry points a request would, so a
    request arriving mid-warmup shares the in-flight build instead of starting a
    second one, and the documented lock order is inherited unchanged. Each stage
    is isolated: a failure is logged and the next stage still runs, because a
    broken shape map must not deny the warm searcher. The stages run most
    expensive first (assembly matcher, then shape map, then searcher), and which
    of them run at all is HOOPS_AI_WARMUP_STAGES (default: all).
    """
    try:
        schema = index_registry._load_index_schema(name)
    except Exception as exc:
        logger.warning("[WARMUP] cannot read schema for index '%s': %s", name, exc)
        return

    stages = warmup_stages_setting()
    clusterings = warmup_clustering_setting()

    # The stages are independent, so warm the most expensive first: a request
    # arriving mid-warmup then shares the in-flight build of the thing most
    # likely to make it wait (the assembly matcher, ~20 s), instead of finding
    # only the cheap searcher ready and paying the full build itself. The three
    # stages are ordered assembly matcher -> shape map -> searcher.

    # Stages 1-2 need body-level (schema v2) data; a legacy index skips them,
    # exactly as assembly search and the shape map reject legacy indexes.
    if schema["schema_version"] < _INDEX_SCHEMA_VERSION:
        logger.info(
            "[WARMUP] '%s' is legacy schema; skipping shape map and assembly matcher",
            name,
        )
    else:
        # Stage 1: assembly matcher (k-means, ~20 s).
        if "assembly" in stages:
            try:
                started = time.perf_counter()
                assembly._get_assembly_matcher(name)
                logger.info(
                    "[WARMUP] '%s' assembly matcher ready in %.2fs",
                    name, time.perf_counter() - started,
                )
            except Exception as exc:
                logger.warning("[WARMUP] '%s' assembly matcher failed: %s", name, exc)

        # Stages 2 and 2b: the map and its point cloud, for each clustering.
        _warmup_map_stages(name, stages, clusterings)

    # Stage 3: named searcher (~0.3 s). Available for every schema version, so
    # it runs even for a legacy index that skipped the stages above.
    if "searcher" in stages:
        try:
            started = time.perf_counter()
            index_registry._get_named_searcher(name)
            logger.info(
                "[WARMUP] '%s' searcher ready in %.2fs", name, time.perf_counter() - started
            )
        except Exception as exc:
            logger.warning("[WARMUP] '%s' searcher failed: %s", name, exc)


def run_startup_warmup() -> None:
    """Warm the configured indexes in the current thread. Never raises."""
    targets = _resolve_warmup_targets()
    if not targets:
        return
    logger.info("[WARMUP] warming %d index(es): %s", len(targets), ", ".join(targets))
    overall = time.perf_counter()
    for name in targets:
        try:
            _warmup_one(name)
        except Exception as exc:
            logger.warning("[WARMUP] index '%s' failed: %s", name, exc)
    logger.info("[WARMUP] complete in %.2fs", time.perf_counter() - overall)


def start_startup_warmup() -> Optional[threading.Thread]:
    """Start background warmup if any targets are configured; else do nothing.

    Runs on a daemon thread so the server accepts requests immediately; requests
    arriving before warmup finishes are served cold (as before). Returns the
    thread, or None when warmup is not configured.
    """
    if not warmup_indexes_setting():
        return None
    thread = threading.Thread(
        target=run_startup_warmup, name="startup-warmup", daemon=True
    )
    thread.start()
    return thread