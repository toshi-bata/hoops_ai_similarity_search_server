"""Tests for opt-in startup warmup target resolution (no SDK required).

These verify the pure decision logic: which indexes are warmed, deduping, the
max cap and unknown-name filtering. The heavy builds themselves are mocked, so
no license or FAISS is needed, and we assert warmup drives the same top-level
(build-locked) entry points a request uses.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core
import assembly
import config
import shape_map as shape_map_mod
import index_registry


class WarmupTargetResolution(unittest.TestCase):
    def _with_indexes(self, names):
        return patch.object(index_registry, "list_indexes", return_value=[{"name": n} for n in names]
        )

    def test_empty_setting_means_no_warmup(self):
        with patch.dict(os.environ, {"HOOPS_AI_WARMUP_INDEXES": ""}, clear=False):
            self.assertEqual(core.warmup_indexes_setting(), [])
            self.assertEqual(core._resolve_warmup_targets(), [])

    def test_parses_comma_list_and_trims(self):
        with patch.dict(os.environ, {"HOOPS_AI_WARMUP_INDEXES": " a , b ,c "}, clear=False):
            self.assertEqual(core.warmup_indexes_setting(), ["a", "b", "c"])

    def test_unknown_indexes_filtered(self):
        env = {"HOOPS_AI_WARMUP_INDEXES": "a,ghost", "HOOPS_AI_WARMUP_MAX": "10"}
        with patch.dict(os.environ, env, clear=False), self._with_indexes(["a", "b"]):
            self.assertEqual(core._resolve_warmup_targets(), ["a"])

    def test_dedup_and_max_cap(self):
        env = {"HOOPS_AI_WARMUP_INDEXES": "a,a,b,c", "HOOPS_AI_WARMUP_MAX": "2"}
        with patch.dict(os.environ, env, clear=False), self._with_indexes(["a", "b", "c"]):
            self.assertEqual(core._resolve_warmup_targets(), ["a", "b"])

    def test_start_returns_none_when_not_configured(self):
        with patch.dict(os.environ, {"HOOPS_AI_WARMUP_INDEXES": ""}, clear=False):
            self.assertIsNone(core.start_startup_warmup())


class WarmupDrivesBuildEntryPoints(unittest.TestCase):
    # The stage and clustering selections are read from the environment, and a
    # developer's .env may well set them, so every test here pins them instead
    # of inheriting whatever the host asks for. Clustering is pinned to kmeans
    # rather than left unset: unset now means BOTH clusterings, which calls the
    # map entry points twice and would hide what these tests are about -- the
    # parameters of a single call. The unset default has its own test below.
    _DEFAULT_ENV = {
        "HOOPS_AI_WARMUP_STAGES": "",
        "HOOPS_AI_WARMUP_CLUSTERING": "kmeans",
        "HOOPS_AI_WARMUP_WARD_MAX": "",
    }

    # The map stages must ask for the population the web client asks for first;
    # warming any other parameter set warms a cache key no request will hit.
    # Mirrors the constants at the top of warmup.py.
    #
    # The cluster count is no longer a constant: it is derived per index from
    # single_part, the population Show: Parts puts on the map. 7,928 is tmcad's,
    # and shape_map_auto_cluster_count turns it into 256 -- the same number the
    # client is handed as default_clusters.
    _STATS = {"single_part": 7928}
    _MAP_PARAMS = dict(
        kind="part", dims=3, color_by="cluster", projection="pca",
        include_tags=True, clustering="kmeans", n_clusters=256,
    )
    # The .scs entry point has no include_tags parameter -- it decides for
    # itself from color_by -- so warmup drops that one key before calling it.
    _SCS_PARAMS = {k: v for k, v in _MAP_PARAMS.items() if k != "include_tags"}

    def test_warmup_one_calls_searcher_map_matcher(self):
        schema = {"schema_version": core._INDEX_SCHEMA_VERSION}
        with patch.dict(os.environ, self._DEFAULT_ENV, clear=False), \
             patch.object(index_registry, "_load_index_schema", return_value=schema), \
             patch.object(index_registry, "get_index_stats", return_value=dict(self._STATS)), \
             patch.object(index_registry, "_get_named_searcher") as searcher, \
             patch.object(shape_map_mod, "compute_index_shape_map") as shape_map, \
             patch.object(shape_map_mod, "compute_index_shape_map_scs") as scs, \
             patch.object(assembly, "_get_assembly_matcher") as matcher:
            core._warmup_one("tmcad")
        searcher.assert_called_once_with("tmcad")
        shape_map.assert_called_once_with("tmcad", **self._MAP_PARAMS)
        scs.assert_called_once_with("tmcad", **self._SCS_PARAMS)
        matcher.assert_called_once_with("tmcad")

    def test_the_warmed_map_carries_the_tags_the_client_asks_for(self):
        # include_tags is part of the result cache key, and the client always
        # sends it as true (web/src/shape-map.js calls api.shapeMap with
        # includeTags: true, unconditionally). compute_index_shape_map defaults
        # it to False, so warming without it built an entry no request could
        # ever hit: agg and the projection are shared caches and still hit, but
        # the clustering was redone on the first real request.
        schema = {"schema_version": core._INDEX_SCHEMA_VERSION}
        with patch.dict(os.environ, self._DEFAULT_ENV, clear=False), \
             patch.object(index_registry, "_load_index_schema", return_value=schema), \
             patch.object(index_registry, "get_index_stats", return_value=dict(self._STATS)), \
             patch.object(index_registry, "_get_named_searcher"), \
             patch.object(shape_map_mod, "compute_index_shape_map") as shape_map, \
             patch.object(shape_map_mod, "compute_index_shape_map_scs") as scs, \
             patch.object(assembly, "_get_assembly_matcher"):
            core._warmup_one("tmcad")
        self.assertIs(shape_map.call_args.kwargs["include_tags"], True)
        # And it must NOT reach the .scs entry point, which has no such
        # parameter and would raise TypeError.
        self.assertNotIn("include_tags", scs.call_args.kwargs)

    def test_both_clusterings_are_warmed_when_none_is_named(self):
        # The default is both: each clustering is a separate cache key, so
        # warming only one leaves whoever switches waiting for a full build.
        # Ward's linkage is built from a BLAS distance matrix now, which is
        # what made the second build cheap enough to do at startup.
        schema = {"schema_version": core._INDEX_SCHEMA_VERSION}
        env = {"HOOPS_AI_WARMUP_STAGES": "map", "HOOPS_AI_WARMUP_CLUSTERING": ""}
        with patch.dict(os.environ, env, clear=False), \
             patch.object(config, "read_env_file", return_value={}), \
             patch.object(index_registry, "_load_index_schema", return_value=schema), \
             patch.object(index_registry, "get_index_stats", return_value=dict(self._STATS)), \
             patch.object(shape_map_mod, "compute_index_shape_map") as shape_map, \
             patch.object(shape_map_mod, "compute_index_shape_map_scs"), \
             patch.object(shape_map_mod, "compute_index_shape_map_tree"):
            self.assertEqual(config.warmup_clustering_setting(), ["kmeans", "ward"])
            core._warmup_one("tmcad")
        warmed = [c.kwargs["clustering"] for c in shape_map.call_args_list]
        self.assertEqual(warmed, ["kmeans", "ward"])

    def test_ward_warms_the_deepest_cut_and_the_tree(self):
        # Ward takes a granularity instead of a cluster count, and the Ward
        # panel asks for the folded tree as soon as the map lands.
        schema = {"schema_version": core._INDEX_SCHEMA_VERSION}
        env = {
            "HOOPS_AI_WARMUP_STAGES": "map",
            "HOOPS_AI_WARMUP_CLUSTERING": "ward",
            "HOOPS_AI_WARMUP_WARD_MAX": "",
        }
        with patch.dict(os.environ, env, clear=False), \
             patch.object(index_registry, "_load_index_schema", return_value=schema), \
             patch.object(index_registry, "get_index_stats", return_value=dict(self._STATS)), \
             patch.object(index_registry, "_get_named_searcher") as searcher, \
             patch.object(shape_map_mod, "compute_index_shape_map") as shape_map, \
             patch.object(shape_map_mod, "compute_index_shape_map_tree") as tree, \
             patch.object(shape_map_mod, "compute_index_shape_map_scs") as scs, \
             patch.object(assembly, "_get_assembly_matcher") as matcher:
            core._warmup_one("tmcad")
        shape_map.assert_called_once_with(
            "tmcad", kind="part", dims=3, color_by="cluster", projection="pca",
            include_tags=True, clustering="ward", granularity=1.0, ward_max=256,
        )
        tree.assert_called_once_with("tmcad", kind="part", k_max=256, clustering="ward")
        # Only the map stage was selected.
        scs.assert_not_called()
        searcher.assert_not_called()
        matcher.assert_not_called()

    def test_ward_max_reaches_all_three_ward_builds(self):
        # The map, its point cloud and the tree must be cut to one ceiling:
        # the map's clusters ARE the tree's leaves, so a mismatch leaves the
        # two describing different families.
        schema = {"schema_version": core._INDEX_SCHEMA_VERSION}
        env = {
            "HOOPS_AI_WARMUP_STAGES": "map,scs",
            "HOOPS_AI_WARMUP_CLUSTERING": "ward",
            "HOOPS_AI_WARMUP_WARD_MAX": "512",
        }
        with patch.dict(os.environ, env, clear=False), \
             patch.object(index_registry, "_load_index_schema", return_value=schema), \
             patch.object(index_registry, "get_index_stats", return_value=dict(self._STATS)), \
             patch.object(shape_map_mod, "compute_index_shape_map") as shape_map, \
             patch.object(shape_map_mod, "compute_index_shape_map_tree") as tree, \
             patch.object(shape_map_mod, "compute_index_shape_map_scs") as scs:
            core._warmup_one("tmcad")
        expected = dict(
            kind="part", dims=3, color_by="cluster", projection="pca",
            include_tags=True, clustering="ward", granularity=1.0, ward_max=512,
        )
        shape_map.assert_called_once_with("tmcad", **expected)
        scs.assert_called_once_with(
            "tmcad", **{k: v for k, v in expected.items() if k != "include_tags"}
        )
        tree.assert_called_once_with("tmcad", kind="part", k_max=512, clustering="ward")

    def test_stages_select_which_work_runs(self):
        schema = {"schema_version": core._INDEX_SCHEMA_VERSION}
        env = {"HOOPS_AI_WARMUP_STAGES": "searcher", "HOOPS_AI_WARMUP_CLUSTERING": ""}
        with patch.dict(os.environ, env, clear=False), \
             patch.object(index_registry, "_load_index_schema", return_value=schema), \
             patch.object(index_registry, "get_index_stats", return_value=dict(self._STATS)), \
             patch.object(index_registry, "_get_named_searcher") as searcher, \
             patch.object(shape_map_mod, "compute_index_shape_map") as shape_map, \
             patch.object(shape_map_mod, "compute_index_shape_map_scs") as scs, \
             patch.object(assembly, "_get_assembly_matcher") as matcher:
            core._warmup_one("tmcad")
        searcher.assert_called_once_with("tmcad")
        shape_map.assert_not_called()
        scs.assert_not_called()
        matcher.assert_not_called()

    def test_legacy_schema_skips_heavy_stages(self):
        schema = {"schema_version": 1}
        with patch.dict(os.environ, self._DEFAULT_ENV, clear=False), \
             patch.object(index_registry, "_load_index_schema", return_value=schema), \
             patch.object(index_registry, "get_index_stats", return_value=dict(self._STATS)), \
             patch.object(index_registry, "_get_named_searcher") as searcher, \
             patch.object(shape_map_mod, "compute_index_shape_map") as shape_map, \
             patch.object(shape_map_mod, "compute_index_shape_map_scs"), \
             patch.object(assembly, "_get_assembly_matcher") as matcher:
            core._warmup_one("legacy_idx")
        searcher.assert_called_once()
        shape_map.assert_not_called()
        matcher.assert_not_called()

    def test_the_warmed_count_follows_the_index_size(self):
        # Not a constant: a small index must be warmed at the count the client
        # will ask for, or the first map it opens is built from scratch.
        schema = {"schema_version": core._INDEX_SCHEMA_VERSION}
        env = {
            "HOOPS_AI_WARMUP_STAGES": "map",
            "HOOPS_AI_WARMUP_CLUSTERING": "kmeans",
            "HOOPS_AI_WARMUP_WARD_MAX": "",
        }
        with patch.dict(os.environ, env, clear=False), \
             patch.object(index_registry, "_load_index_schema", return_value=schema), \
             patch.object(index_registry, "get_index_stats", return_value={"single_part": 512}), \
             patch.object(shape_map_mod, "compute_index_shape_map") as shape_map:
            core._warmup_one("small")
        self.assertEqual(
            shape_map.call_args.kwargs["n_clusters"],
            shape_map_mod.shape_map_auto_cluster_count(512),
        )

    def test_stage_failure_does_not_stop_later_stages(self):
        schema = {"schema_version": core._INDEX_SCHEMA_VERSION}
        with patch.dict(os.environ, self._DEFAULT_ENV, clear=False), \
             patch.object(index_registry, "_load_index_schema", return_value=schema), \
             patch.object(index_registry, "get_index_stats", return_value=dict(self._STATS)), \
             patch.object(index_registry, "_get_named_searcher", side_effect=RuntimeError("boom")), \
             patch.object(shape_map_mod, "compute_index_shape_map") as shape_map, \
             patch.object(shape_map_mod, "compute_index_shape_map_scs"), \
             patch.object(assembly, "_get_assembly_matcher") as matcher:
            core._warmup_one("tmcad")  # must not raise
        shape_map.assert_called_once()
        matcher.assert_called_once()


if __name__ == "__main__":
    unittest.main()
