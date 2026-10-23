"""Unit tests for the index-based Shape Map numeric core (SDK-free).

These exercise ``core._shape_map_project`` / ``core._shape_map_cluster`` and the
``core._compute_shape_map_points`` orchestrator with a mocked index loader, so
they run without the HOOPS AI SDK or a license. They pin the behaviour that must
match hoops_ai_native_bridge: one point per file id, subset-only projection, and
deterministic (seed=42) cluster assignments.
"""

import json
import os
import unittest
from unittest.mock import patch

import numpy as np
import pytest

import core
import shape_map
import index_registry
import config


# The expected value of core._SHAPE_MAP_COLORING_VERSION.
#
# That number keys the .scs disk cache: every file baked under an older value
# is stale and must not be served. Pinning it here is deliberate -- it is a
# tripwire, so that bumping the version in shape_map.py is always paired with a
# conscious edit of this line, and so that a change which DOES alter the baked
# colours, the coordinates or the node layout can never ship without one.
#
# History of the bumps this pin has been moved for:
#   v7  rows sorted canonically by file id before the SVD, a fixed sign
#       convention on the principal components, clusters renumbered
#   v8  renumbering keyed on mean projected position instead of size
#   v9  renumbering keyed on member count (then cohesion, then position),
#       read after the outlier pass
#   v10 Ward colouring by cluster bakes one neutral grey instead of the
#       per-cluster colours, because the client repaints the cloud from the tree
#   v11 tag colours cycle the palette as cluster colours have since v6, so a map
#       with more than 24 tags has one node per tag instead of a grey "Other"
#   v12 past slot 24 the lightness cycles and the hue steps, instead of the
#       other way round: cluster ids are a size ranking, so the old assignment
#       made lightness mean "small" and sank 184 of 256 clusters onto one step
EXPECTED_COLORING_VERSION = 12


def _fake_batch(n_files=6, bodies_per_file=2, dim=8, seed=0):
    """Build a synthetic (ids, values) pair: contiguous body rows per file id."""
    rng = np.random.default_rng(seed)
    ids = []
    rows = []
    for f in range(n_files):
        # Give each file a distinct centre so clusters are well separated.
        centre = rng.normal(size=dim).astype(np.float32)
        for _ in range(bodies_per_file):
            rows.append(centre + 0.01 * rng.normal(size=dim).astype(np.float32))
            ids.append(f"file_{f}")
    return ids, np.asarray(rows, dtype=np.float32)


class ShapeMapProjectionTests(unittest.TestCase):
    def test_one_point_per_file_id(self):
        ids, values = _fake_batch(n_files=5, bodies_per_file=3, dim=8)
        file_ids, coords, agg = core._shape_map_project(ids, values)
        self.assertEqual(file_ids, [f"file_{i}" for i in range(5)])
        self.assertEqual(agg.shape[0], 5)
        self.assertEqual(coords.shape[0], 5)

    def test_aggregate_is_mean_then_normalized(self):
        ids, values = _fake_batch(n_files=2, bodies_per_file=2, dim=4)
        _file_ids, _coords, agg = core._shape_map_project(ids, values)
        # Every aggregated row is unit length (rows were non-zero).
        norms = np.linalg.norm(agg, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), rtol=1e-5, atol=1e-5)

    def test_only_ids_projects_subset_only(self):
        ids, values = _fake_batch(n_files=6, bodies_per_file=2, dim=8)
        subset = ["file_1", "file_3", "file_5"]
        file_ids, coords, agg = core._shape_map_project(ids, values, only_ids=subset)
        self.assertEqual(file_ids, subset)
        self.assertEqual(agg.shape[0], 3)
        self.assertEqual(coords.shape[0], 3)

    def test_empty_subset_returns_empty(self):
        ids, values = _fake_batch(n_files=3, bodies_per_file=1, dim=4)
        file_ids, coords, agg = core._shape_map_project(ids, values, only_ids=["nope"])
        self.assertEqual(file_ids, [])
        self.assertEqual(agg.shape[0], 0)

    def test_keep_fn_projects_subset_only(self):
        ids, values = _fake_batch(n_files=6, bodies_per_file=2, dim=8)
        keep = {"file_0", "file_2", "file_4"}
        file_ids, coords, agg = core._shape_map_project(
            ids, values, keep_fn=lambda fid: fid in keep
        )
        self.assertEqual(file_ids, ["file_0", "file_2", "file_4"])
        self.assertEqual(agg.shape[0], 3)

    def test_keep_fn_takes_precedence_over_only_ids(self):
        ids, values = _fake_batch(n_files=4, bodies_per_file=1, dim=4)
        file_ids, _c, _a = core._shape_map_project(
            ids, values, only_ids=["file_0"], keep_fn=lambda fid: fid == "file_3"
        )
        self.assertEqual(file_ids, ["file_3"])


class ShapeMapLegendTests(unittest.TestCase):
    def _points(self, spec):
        # spec: list of (file_id, cluster)
        return [(fid, 0.0, 0.0, 0.0, cl) for fid, cl in spec]

    def test_tag_legend_counts_and_orders(self):
        points = self._points([("a", 0), ("b", 0), ("c", 1), ("d", 1)])
        tags_map = {"a": ["bracket"], "b": ["bracket", "steel"], "c": ["steel"]}
        legend = core._shape_map_legend(points, tags_map, "tag")
        # "bracket": 2, "steel": 2 (ties sorted by name), then Untagged: 1 (d).
        # Colours come from the shared palette (bracket=slot 0, steel=slot 1) and
        # untagged keeps the grey.
        self.assertEqual(
            legend[0],
            {"label": "bracket", "kind": "tag", "count": 2, "rgb": [230, 159, 0]},
        )
        self.assertEqual(
            legend[1],
            {"label": "steel", "kind": "tag", "count": 2, "rgb": [86, 180, 233]},
        )
        self.assertEqual(
            legend[-1],
            {"label": "Untagged", "kind": "untagged", "count": 1, "rgb": [180, 180, 180]},
        )

    def test_cluster_legend_labels_and_colours(self):
        points = self._points([("a", 0), ("b", 0), ("c", 1)])
        legend = core._shape_map_legend(points, {}, "cluster")
        # Clusters are 1-based in the label and coloured from the shared palette.
        self.assertEqual(
            legend,
            [
                {"label": "Cluster 1", "kind": "cluster", "count": 2, "rgb": [230, 159, 0]},
                {"label": "Cluster 2", "kind": "cluster", "count": 1, "rgb": [86, 180, 233]},
            ],
        )

    def test_clusters_past_the_palette_keep_their_own_rows(self):
        # More clusters than palette slots: each one still gets its own legend
        # row (and its own SCS node); only the colour repeats, darkened.
        n_palette = len(core._SHAPE_MAP_PALETTE)
        spec = [(f"f{c}", c) for c in range(n_palette + 3)]
        legend = core._shape_map_legend(self._points(spec), {}, "cluster")
        self.assertEqual(len(legend), n_palette + 3)
        self.assertTrue(all(row["kind"] == "cluster" for row in legend))
        self.assertEqual(legend[-1]["label"], f"Cluster {n_palette + 3}")


class ShapeMapPayloadTests(unittest.TestCase):
    def _points(self):
        return [("a", 0.1, 0.2, 0.0, 0, 0.95), ("b", -0.1, 0.0, 0.0, 1, 0.80)]

    def test_payload_omits_tags_by_default(self):
        payload = core._shape_map_build_payload(
            "tmcad", self._points(), 2, 3,
            {"a": {"filename": "a.stp"}, "b": {"filename": "b.stp"}},
            {"a": ["bracket"]}, include_tags=False,
        )
        self.assertEqual(payload["index"], "tmcad")
        self.assertEqual(payload["count"], 2)
        self.assertNotIn("tags", payload["points"][0])
        self.assertIn("legend", payload)

    def test_payload_includes_tags_when_requested(self):
        payload = core._shape_map_build_payload(
            "tmcad", self._points(), 2, 3, {},
            {"a": ["bracket"]}, include_tags=True,
        )
        self.assertEqual(payload["points"][0]["tags"], ["bracket"])
        self.assertEqual(payload["points"][1]["tags"], [])

    def test_every_payload_key_is_a_response_field(self):
        # The endpoint hands the payload to ShapeMapResponse whole, and pydantic
        # ignores keys the model does not declare -- so a field added to the
        # payload but not to the model would vanish from the response with no
        # error at all. Nested fields are safe (the sub-dicts go over in one
        # piece); it is top-level additions that slip through, which is how
        # outlier_threshold_effective came to be null for a while.
        from routers.similarity import ShapeMapResponse

        payload = core._shape_map_build_payload(
            "tmcad", self._points(), 2, 3, {},
            {}, include_tags=True, color_by="cluster",
            cohesion={0: 0.9}, centres={0: "a"}, outlier_threshold_effective=0.9,
        )
        payload["subset_id"] = "deadbeef"
        self.assertEqual(set(payload) - set(ShapeMapResponse.model_fields), set())

    def test_subset_signature_is_stable(self):
        f = core.normalize_tag_filter(["b", "a"], "all", True)
        self.assertEqual(core._shape_map_subset_sig(None), "all|kind=any")
        # Tag order does not change the signature.
        g = core.normalize_tag_filter(["a", "b"], "all", True)
        self.assertEqual(core._shape_map_subset_sig(f), core._shape_map_subset_sig(g))

    def test_kind_changes_signature(self):
        # part / assembly / any are DIFFERENT subsets and must not share a cache
        # entry; an unknown kind collapses to "any".
        any_sig = core._shape_map_subset_sig(None, "any")
        part_sig = core._shape_map_subset_sig(None, "part")
        asm_sig = core._shape_map_subset_sig(None, "assembly")
        self.assertNotEqual(any_sig, part_sig)
        self.assertNotEqual(any_sig, asm_sig)
        self.assertNotEqual(part_sig, asm_sig)
        self.assertEqual(core._shape_map_subset_sig(None, "bogus"), any_sig)

    def test_kind_predicate_and_compose(self):
        filemeta = {
            "a": {"filename": "a", "kind": "part"},
            "b": {"filename": "b", "kind": "assembly"},
            "c": {"filename": "c"},  # missing kind -> treated as part
        }
        self.assertIsNone(core._shape_map_kind_predicate(filemeta, "any"))
        part = core._shape_map_kind_predicate(filemeta, "part")
        asm = core._shape_map_kind_predicate(filemeta, "assembly")
        self.assertEqual([fid for fid in "abc" if part(fid)], ["a", "c"])
        self.assertEqual([fid for fid in "abc" if asm(fid)], ["b"])
        # A file id with no metadata row defaults to part, so it is never
        # dropped from a parts-only map.
        self.assertTrue(part("zzz"))
        self.assertFalse(asm("zzz"))
        # _compose_keep ANDs predicates; None means "keep all".
        self.assertIsNone(core._compose_keep(None, None))
        self.assertIs(core._compose_keep(part, None), part)
        self.assertIs(core._compose_keep(None, asm), asm)
        only_b = core._compose_keep(lambda fid: fid in ("b", "c"), asm)
        self.assertEqual([fid for fid in "abc" if only_b(fid)], ["b"])

    def test_tag_mode_changes_signature(self):
        # tags=a,b&mode=any and tags=a,b&mode=all are DIFFERENT subsets; they
        # must not share a projection cache entry.
        any_sig = core._shape_map_subset_sig(core.normalize_tag_filter(["a", "b"], "any", None))
        all_sig = core._shape_map_subset_sig(core.normalize_tag_filter(["a", "b"], "all", None))
        self.assertNotEqual(any_sig, all_sig)

    def test_tagged_flag_changes_signature(self):
        true_sig = core._shape_map_subset_sig(core.normalize_tag_filter(None, "any", True))
        false_sig = core._shape_map_subset_sig(core.normalize_tag_filter(None, "any", False))
        self.assertNotEqual(true_sig, false_sig)
        # tagged=true with a tag list also differs from the same tags untagged.
        t_sig = core._shape_map_subset_sig(core.normalize_tag_filter(["a"], "any", True))
        f_sig = core._shape_map_subset_sig(core.normalize_tag_filter(["a"], "any", False))
        self.assertNotEqual(t_sig, f_sig)


class ShapeMapClusterTests(unittest.TestCase):
    def test_single_cluster_scales_into_cube_no_faiss(self):
        # k=1 takes the no-faiss branch, so this runs SDK-free and still covers
        # the [-1, 1] cube scaling and the 3-axis padding.
        ids, values = _fake_batch(n_files=6, bodies_per_file=2, dim=8)
        file_ids, coords, agg = core._shape_map_project(ids, values)
        points, k, cohesion, _centres, _eff = core._shape_map_cluster(agg, coords, file_ids, n_clusters=1, dims=3)
        self.assertEqual(k, 1)
        self.assertEqual(len(points), 6)
        max_abs = max(abs(c) for _f, x, y, z, _cl, _s in points for c in (x, y, z))
        self.assertLessEqual(max_abs, 1.0 + 1e-4)
        self.assertTrue(all(cl == 0 for _f, _x, _y, _z, cl, _s in points))

    def test_dims_two_zeroes_third_axis_no_faiss(self):
        ids, values = _fake_batch(n_files=6, bodies_per_file=2, dim=8)
        file_ids, coords, agg = core._shape_map_project(ids, values)
        points, _k, _coh, _centres, _eff = core._shape_map_cluster(agg, coords, file_ids, n_clusters=1, dims=2)
        for _fid, _x, _y, z, _cluster, _sim in points:
            self.assertEqual(z, 0.0)

    def test_points_shape_and_range(self):
        pytest.importorskip("faiss")
        ids, values = _fake_batch(n_files=8, bodies_per_file=2, dim=8)
        file_ids, coords, agg = core._shape_map_project(ids, values)
        points, k, _coh, _centres, _eff = core._shape_map_cluster(agg, coords, file_ids, n_clusters=3, dims=3)
        self.assertEqual(len(points), 8)
        self.assertEqual(k, 3)
        for fid, x, y, z, cluster, sim in points:
            self.assertIn(fid, file_ids)
            for c in (x, y, z):
                self.assertLessEqual(abs(c), 1.0 + 1e-4)
            self.assertTrue(0 <= cluster < k)
            # Cosine similarity, so it cannot leave [-1, 1].
            self.assertLessEqual(abs(sim), 1.0 + 1e-4)

    def test_deterministic_across_runs(self):
        pytest.importorskip("faiss")
        ids, values = _fake_batch(n_files=10, bodies_per_file=2, dim=8)
        file_ids, coords, agg = core._shape_map_project(ids, values)
        p1, k1, _c1, _centres, _eff = core._shape_map_cluster(agg, coords, file_ids, n_clusters=4, dims=3)
        p2, k2, _c2, _centres, _eff = core._shape_map_cluster(agg, coords, file_ids, n_clusters=4, dims=3)
        self.assertEqual(k1, k2)
        self.assertEqual(p1, p2)

    def test_auto_cluster_count(self):
        pytest.importorskip("faiss")
        ids, values = _fake_batch(n_files=200, bodies_per_file=1, dim=8)
        file_ids, coords, agg = core._shape_map_project(ids, values)
        _points, k, _coh, _centres, _eff = core._shape_map_cluster(agg, coords, file_ids, n_clusters=0, dims=3)
        # min(max(2, N // 50), 24) with N=200 -> 4
        self.assertEqual(k, 4)


class ShapeMapAutoClusterCountTests(unittest.TestCase):
    """The one rule warmup builds with and the client opens on.

    Runs SDK-free: it is arithmetic over a part count, with no index behind it.
    """

    def test_tmcad_sized_index_lands_on_the_ward_leaf_count(self):
        # 7,928 / 32 = 247.75, and the nearest power of two is 256 -- the same
        # number of leaves Ward's tree descends to by default.
        self.assertEqual(shape_map.shape_map_auto_cluster_count(7928), 256)

    def test_exact_power_of_two_is_kept(self):
        self.assertEqual(shape_map.shape_map_auto_cluster_count(512), 16)

    def test_small_index_stops_at_the_floor(self):
        self.assertEqual(shape_map.shape_map_auto_cluster_count(100), 8)

    def test_never_more_clusters_than_parts(self):
        self.assertLessEqual(shape_map.shape_map_auto_cluster_count(10), 10)

    def test_a_single_part_still_yields_a_usable_count(self):
        self.assertEqual(shape_map.shape_map_auto_cluster_count(1), 2)

    def test_huge_index_stops_at_the_ceiling(self):
        # 1024 is what /map accepts and what the Clusters control offers.
        self.assertEqual(shape_map.shape_map_auto_cluster_count(1_000_000), 1024)


class ComputeShapeMapPointsTests(unittest.TestCase):
    def test_orchestrator_uses_loader(self):
        pytest.importorskip("faiss")
        ids, values = _fake_batch(n_files=7, bodies_per_file=2, dim=8)
        with patch.object(
            shape_map, "_load_index_body_batch", return_value=(ids, values)
        ):
            result = core._compute_shape_map_points("tmcad", n_clusters=3, dims=3)
        self.assertEqual(result["count"], 7)
        self.assertEqual(result["clusters"], 3)
        self.assertEqual(len(result["points"]), 7)

    def test_orchestrator_is_deterministic(self):
        pytest.importorskip("faiss")
        ids, values = _fake_batch(n_files=12, bodies_per_file=2, dim=8)
        with patch.object(
            shape_map, "_load_index_body_batch", return_value=(ids, values)
        ):
            r1 = core._compute_shape_map_points("tmcad", 4, 3)
            r2 = core._compute_shape_map_points("tmcad", 4, 3)
        self.assertEqual(r1["points"], r2["points"])


class ShapeMapGroupingTests(unittest.TestCase):
    def _points(self, spec):
        # spec: list of (file_id, cluster)
        return [(fid, 0.0, 0.0, 0.0, cl, 1.0) for fid, cl in spec]

    def test_categorize_cluster_index_and_categories(self):
        cat_index, cats = core._shape_map_categorize([2, 0, 2, 0], [[], [], [], []], "cluster")
        # Categories are ordered by cluster id (0 then 2); cat_index maps points.
        labels = [c["label"] for c in cats]
        self.assertEqual(labels, ["Cluster 1", "Cluster 3"])
        self.assertEqual(cat_index, [1, 0, 1, 0])

    def test_categorize_tag_untagged_is_last(self):
        tags = [["b"], [], ["a"], ["a"]]
        cat_index, cats = core._shape_map_categorize([0, 0, 0, 0], tags, "tag")
        # a:2, b:1 -> a first, then b, Untagged last. Point 1 (no tags) -> last.
        self.assertEqual([c["kind"] for c in cats], ["tag", "tag", "untagged"])
        self.assertEqual(cats[0]["label"], "a")
        self.assertEqual(cat_index[1], 2)

    def test_payload_groups_are_contiguous_and_ordered(self):
        # Interleaved clusters must be reordered so each cluster is one run.
        points = self._points([("a", 1), ("b", 0), ("c", 1), ("d", 0)])
        payload = core._shape_map_build_payload(
            "tmcad", points, 2, 3,
            {fid: {"filename": f"{fid}.stp"} for fid in "abcd"}, {},
            include_tags=False, color_by="cluster",
        )
        groups = payload["groups"]
        self.assertEqual([g["label"] for g in groups], ["Cluster 1", "Cluster 2"])
        # Every group's slice is homogeneous in cluster and matches offset/count.
        for g in groups:
            seg = payload["points"][g["offset"]:g["offset"] + g["count"]]
            self.assertEqual(len(seg), g["count"])
        # Points are grouped: cluster 0 (b, d) then cluster 1 (a, c).
        self.assertEqual([p["file_id"] for p in payload["points"]], ["b", "d", "a", "c"])
        self.assertEqual(groups[0]["offset"], 0)
        self.assertEqual(groups[1]["offset"], 2)

    def test_group_node_names_unique_and_match_legend_order(self):
        points = self._points([("a", 0), ("b", 1), ("c", 2)])
        payload = core._shape_map_build_payload(
            "tmcad", points, 3, 3, {}, {}, include_tags=False, color_by="cluster",
        )
        names = [g["node_name"] for g in payload["groups"]]
        self.assertEqual(len(names), len(set(names)))
        # Groups and legend share the same category order.
        self.assertEqual(
            [g["label"] for g in payload["groups"]],
            [e["label"] for e in payload["legend"]],
        )

    def test_group_rgb_matches_legend_rgb(self):
        points = self._points([("a", 0), ("b", 1)])
        payload = core._shape_map_build_payload(
            "tmcad", points, 2, 3, {}, {}, include_tags=False, color_by="cluster",
        )
        for g, e in zip(payload["groups"], payload["legend"]):
            self.assertEqual(g["rgb"], e["rgb"])


class ClearScsMapCacheTests(unittest.TestCase):
    def test_removes_scs_and_temp_files_keeps_others(self):
        import pathlib
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            (d / "a.scs").write_bytes(b"1")
            (d / "b.scs").write_bytes(b"2")
            (d / "c.scs.tmp").write_bytes(b"3")
            (d / "d.json").write_text("{}")
            keep = d / "README.md"
            keep.write_text("keep")
            with patch.object(config, "SCS_MAP_DIR", d):
                removed = core.clear_scs_map_cache()
            self.assertEqual(removed, 4)
            self.assertFalse((d / "a.scs").exists())
            self.assertFalse((d / "b.scs").exists())
            self.assertFalse((d / "c.scs.tmp").exists())
            self.assertFalse((d / "d.json").exists())
            self.assertTrue(keep.exists())

    def test_returns_zero_when_empty(self):
        import pathlib
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(config, "SCS_MAP_DIR", pathlib.Path(tmp)):
                self.assertEqual(core.clear_scs_map_cache(), 0)


class ShapeMapTagCacheInvalidationTests(unittest.TestCase):
    """A tag write must retire the tag-dependent map caches.

    Regression for the reported bug: after tagging a part the "Colored by Tag"
    map kept showing every part as Untagged. Tags live in a sidecar and a tag
    write leaves the ``.faiss`` mtime untouched, so the shape-map caches (keyed
    on that mtime) served the pre-tag payload. These tests drive
    ``compute_index_shape_map`` end to end with a fake body batch / filemeta and
    a REAL tag sidecar, so they fail if the tag ETag is dropped from the key.
    """

    NAME = "tagmap"

    def setUp(self):
        import pathlib
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._prev_indexes_dir = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmp.name)
        self.addCleanup(setattr, config, "INDEXES_DIR", self._prev_indexes_dir)

        # A real .faiss file so existence + mtime_ns behave normally; a tag write
        # must NOT touch it, which is the whole point of the test.
        faiss_path = index_registry._index_faiss_path(self.NAME)
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss_path.write_bytes(b"fake-index")

        self._file_ids = ["file_0", "file_1", "file_2"]
        ids, values = _fake_batch(n_files=3, bodies_per_file=2, dim=8, seed=1)
        self._ids, self._values = ids, values
        self._filemeta = {
            fid: {"filename": f"{fid}.stp", "kind": "part"} for fid in self._file_ids
        }

        patches = [
            patch.object(shape_map, "_load_index_body_batch",
                         return_value=(self._ids, self._values)),
            patch.object(shape_map, "_shape_map_filemeta", return_value=self._filemeta),
            patch.object(index_registry, "_registered_index_ids",
                         return_value=set(self._file_ids)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        core._tags_cache.clear()
        core._tags_locks.clear()
        core._shape_map_results.clear()
        core._shape_map_agg.clear()
        self.addCleanup(core._shape_map_results.clear)
        self.addCleanup(core._shape_map_agg.clear)
        self.addCleanup(core._tags_cache.clear)
        self.addCleanup(core._tags_locks.clear)

    @staticmethod
    def _legend_labels(payload):
        return [e["label"] for e in payload["legend"]]

    def test_color_by_tag_reflects_new_tag_after_write(self):
        # Cold "colour by tag" map: nothing is tagged yet -> only Untagged.
        first = shape_map.compute_index_shape_map(
            self.NAME, n_clusters=1, dims=3, color_by="tag",
        )
        self.assertIn("Untagged", self._legend_labels(first))
        self.assertNotIn("bracket", self._legend_labels(first))

        # Tag one part. This writes only the sidecar; the .faiss mtime is unchanged.
        core.set_part_tags(self.NAME, "file_0", ["bracket"])

        # Same request again: without the tag ETag in the key this returns the
        # stale (all-Untagged) payload; with it, "bracket" now appears.
        second = shape_map.compute_index_shape_map(
            self.NAME, n_clusters=1, dims=3, color_by="tag",
        )
        self.assertIn("bracket", self._legend_labels(second),
                      "colour-by-tag map served a stale pre-tag payload")
        self.assertIsNot(second, first)

    def test_include_tags_payload_updates_after_write(self):
        shape_map.compute_index_shape_map(
            self.NAME, n_clusters=1, dims=3, include_tags=True,
        )
        core.set_part_tags(self.NAME, "file_1", ["v2"])
        after = shape_map.compute_index_shape_map(
            self.NAME, n_clusters=1, dims=3, include_tags=True,
        )
        tags_for_file_1 = next(
            p["tags"] for p in after["points"] if p["file_id"] == "file_1"
        )
        self.assertEqual(tags_for_file_1, ["v2"])

    def test_cluster_map_is_reused_across_unrelated_tagging(self):
        # A plain cluster map does not depend on tags, so tagging must NOT churn
        # its cache: the second call returns the very same cached object.
        first = shape_map.compute_index_shape_map(
            self.NAME, n_clusters=1, dims=3, color_by="cluster",
        )
        core.set_part_tags(self.NAME, "file_0", ["bracket"])
        second = shape_map.compute_index_shape_map(
            self.NAME, n_clusters=1, dims=3, color_by="cluster",
        )
        self.assertIs(second, first)


class ShapeMapCohesionEndToEndTests(ShapeMapTagCacheInvalidationTests):
    """Cohesion survives the full compute path, including the .scs branch."""

    NAME = "cohesionmap"

    def test_cluster_map_legend_carries_cohesion(self):
        payload = shape_map.compute_index_shape_map(
            self.NAME, n_clusters=1, dims=3, color_by="cluster",
        )
        clusters = [row for row in payload["legend"] if row["kind"] == "cluster"]
        self.assertTrue(clusters)
        for row in clusters:
            self.assertIsInstance(row["cohesion"], float)
            self.assertLessEqual(row["cohesion"], 1.0 + 1e-6)

    def test_scs_path_still_builds_after_the_extra_return_value(self):
        # compute_index_shape_map_scs goes through the same _shape_map_cluster
        # call, so a missed unpack there would surface as a TypeError here. The
        # license-bearing tool is stubbed out; only the plumbing is under test.
        import pathlib
        import subprocess as sp
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        scs_dir = pathlib.Path(tmp.name)
        exe = scs_dir.joinpath("scs_points.exe")
        exe.write_bytes(b"stub")
        written = {}

        def fake_run(argv, **kwargs):
            out = pathlib.Path(argv[argv.index("--output") + 1])
            written["groups"] = json.loads(
                pathlib.Path(argv[argv.index("--input") + 1]).read_text("utf-8")
            )["groups"]
            out.write_bytes(b"scs")
            return sp.CompletedProcess(argv, 0, "", "")

        prev_dir = config.SCS_MAP_DIR
        config.SCS_MAP_DIR = scs_dir
        self.addCleanup(setattr, config, "SCS_MAP_DIR", prev_dir)
        with patch.object(shape_map, "_scs_points_exe_path", return_value=exe), \
                patch.object(shape_map.subprocess, "run", side_effect=fake_run):
            path = shape_map.compute_index_shape_map_scs(
                self.NAME, n_clusters=1, dims=3,
            )
        self.assertTrue(pathlib.Path(path).exists())
        total = sum(len(g["points"]) for g in written["groups"])
        self.assertEqual(total, len(self._file_ids))

    def test_coloring_version_is_unchanged_by_cohesion(self):
        # Cohesion is a number in the JSON payload: it changes neither the baked
        # colours nor the point order, so adding it did not invalidate any .scs
        # file. The version has moved on since, for changes that DID affect the
        # baked output (see EXPECTED_COLORING_VERSION) -- this test guards the
        # rule, not a particular number, so it reads the same pin.
        self.assertEqual(core._SHAPE_MAP_COLORING_VERSION, EXPECTED_COLORING_VERSION)


class ShapeMapWardNeutralScsTests(ShapeMapTagCacheInvalidationTests):
    """Ward bakes one neutral grey; nothing else loses its colours.

    Under Ward the client repaints the point cloud from the tree, so the baked
    colours show for an instant only. They are therefore baked neutral (see
    ``_SHAPE_MAP_WARD_NEUTRAL_RGB``). k-means has no tree to repaint from, and
    Ward + Colour by Tag does not use the tree at all, so both keep their own
    colours -- which is what these tests pin.
    """

    NAME = "wardneutral"

    def _bake(self, **kwargs):
        """Run the .scs build with the licensed tool stubbed; return the JSON."""
        import pathlib
        import subprocess as sp
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        scs_dir = pathlib.Path(tmp.name)
        exe = scs_dir.joinpath("scs_points.exe")
        exe.write_bytes(b"stub")
        written = {}

        def fake_run(argv, **_kwargs):
            out = pathlib.Path(argv[argv.index("--output") + 1])
            written["groups"] = json.loads(
                pathlib.Path(argv[argv.index("--input") + 1]).read_text("utf-8")
            )["groups"]
            out.write_bytes(b"scs")
            return sp.CompletedProcess(argv, 0, "", "")

        prev_dir = config.SCS_MAP_DIR
        config.SCS_MAP_DIR = scs_dir
        self.addCleanup(setattr, config, "SCS_MAP_DIR", prev_dir)
        with patch.object(shape_map, "_scs_points_exe_path", return_value=exe), \
                patch.object(shape_map.subprocess, "run", side_effect=fake_run):
            shape_map.compute_index_shape_map_scs(self.NAME, dims=3, **kwargs)
        self.assertTrue(written.get("groups"))
        return written["groups"]

    @staticmethod
    def _rgbs(groups):
        return {(g["r"], g["g"], g["b"]) for g in groups}

    def test_ward_cluster_bakes_every_group_neutral(self):
        groups = self._bake(clustering="ward", granularity=1.0, color_by="cluster")
        self.assertEqual(self._rgbs(groups), {shape_map._SHAPE_MAP_WARD_NEUTRAL_RGB})

    def test_kmeans_keeps_the_per_category_colours(self):
        # One cluster is enough to pin the rule and keeps the test independent
        # of faiss, which the k-means path needs only past k=1.
        groups = self._bake(n_clusters=1, clustering="kmeans", color_by="cluster")
        self.assertNotIn(shape_map._SHAPE_MAP_WARD_NEUTRAL_RGB, self._rgbs(groups))
        # The baked colour is the one the /map payload advertises for the same
        # category, which is what the legend beside the cloud is drawn from.
        payload = shape_map.compute_index_shape_map(
            self.NAME, n_clusters=1, dims=3, clustering="kmeans", color_by="cluster",
        )
        advertised = {g["node_name"]: tuple(g["rgb"]) for g in payload["groups"]}
        for g in groups:
            self.assertEqual((g["r"], g["g"], g["b"]), advertised[g["node_name"]])

    def test_ward_with_tag_colouring_keeps_the_tag_colours(self):
        core.set_part_tags(self.NAME, "file_0", ["bracket"])
        groups = self._bake(clustering="ward", granularity=1.0, color_by="tag")
        self.assertNotIn(shape_map._SHAPE_MAP_WARD_NEUTRAL_RGB, self._rgbs(groups))
        payload = shape_map.compute_index_shape_map(
            self.NAME, dims=3, clustering="ward", granularity=1.0, color_by="tag",
        )
        advertised = {g["node_name"]: tuple(g["rgb"]) for g in payload["groups"]}
        for g in groups:
            self.assertEqual((g["r"], g["g"], g["b"]), advertised[g["node_name"]])


class ShapeMapSubsetRegistryTests(unittest.TestCase):
    """The drill-down subset registry and its effect on the cache signature."""

    def setUp(self):
        shape_map._shape_map_subsets.clear()
        self.addCleanup(shape_map._shape_map_subsets.clear)

    def test_no_subset_leaves_the_signature_byte_identical(self):
        # Pinned deliberately: any change here would invalidate every cached
        # map and .scs file already on disk for non-drill-down requests.
        sig = shape_map._shape_map_subset_sig(None, "part")
        self.assertEqual(sig, shape_map._shape_map_subset_sig(None, "part", None))
        self.assertNotIn("|ids=", sig)

    def test_subset_is_appended_to_the_signature(self):
        subset_id = shape_map._register_map_subset(["b", "a"])
        sig = shape_map._shape_map_subset_sig(None, "part", subset_id)
        self.assertTrue(sig.endswith(f"|ids={subset_id}"))

    def test_id_is_order_independent(self):
        first = shape_map._register_map_subset(["c", "a", "b"])
        second = shape_map._register_map_subset(["b", "c", "a", "a"])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)

    def test_round_trip_returns_sorted_ids(self):
        subset_id = shape_map._register_map_subset(["c", "a", "b"])
        self.assertEqual(shape_map._resolve_map_subset(subset_id), ["a", "b", "c"])

    def test_unknown_id_resolves_to_none(self):
        self.assertIsNone(shape_map._resolve_map_subset("deadbeefdeadbeef"))
        self.assertIsNone(shape_map._resolve_map_subset(None))

    def test_empty_list_means_no_subset(self):
        self.assertIsNone(shape_map._normalize_map_subset([]))
        self.assertIsNone(shape_map._register_map_subset([]))

    def test_registry_is_bounded(self):
        for i in range(shape_map._SHAPE_MAP_SUBSET_CACHE_MAX + 4):
            shape_map._register_map_subset([f"id_{i}"])
        self.assertEqual(
            len(shape_map._shape_map_subsets), shape_map._SHAPE_MAP_SUBSET_CACHE_MAX
        )


class ShapeMapDrillDownTests(ShapeMapTagCacheInvalidationTests):
    """``only_file_ids`` must reach the projection and shrink the map."""

    NAME = "drillmap"

    def test_only_ids_is_passed_to_the_projection(self):
        real = shape_map._shape_map_project
        seen = {}

        def spy(
            ids, values, only_ids=None, keep_fn=None, projection="pca",
            timings=None,
        ):
            seen["only_ids"] = only_ids
            return real(
                ids, values, only_ids=only_ids, keep_fn=keep_fn,
                projection=projection, timings=timings,
            )

        with patch.object(shape_map, "_shape_map_project", side_effect=spy):
            shape_map.compute_index_shape_map(
                self.NAME, n_clusters=1, dims=3, only_file_ids=["file_2", "file_0"],
            )
        self.assertEqual(seen["only_ids"], ["file_0", "file_2"])

    def test_subset_restricts_the_payload(self):
        payload = shape_map.compute_index_shape_map(
            self.NAME, n_clusters=1, dims=3, only_file_ids=["file_0", "file_1"],
        )
        self.assertEqual(
            sorted(p["file_id"] for p in payload["points"]), ["file_0", "file_1"]
        )
        self.assertIsNotNone(payload["subset_id"])

    def test_subset_survives_a_kind_filter(self):
        # keep_fn takes precedence over only_ids inside _shape_map_project, so a
        # subset combined with another filter is the case that would silently
        # fall back to the whole index if the subset were not also a predicate.
        payload = shape_map.compute_index_shape_map(
            self.NAME, n_clusters=1, dims=3, kind="part",
            only_file_ids=["file_0", "file_1"],
        )
        self.assertEqual(
            sorted(p["file_id"] for p in payload["points"]), ["file_0", "file_1"]
        )

    def test_no_subset_reports_subset_id_none(self):
        payload = shape_map.compute_index_shape_map(self.NAME, n_clusters=1, dims=3)
        self.assertIsNone(payload["subset_id"])

    def test_subset_of_one_is_rejected(self):
        with self.assertRaises(ValueError):
            shape_map.compute_index_shape_map(
                self.NAME, n_clusters=1, dims=3, only_file_ids=["file_0"],
            )

    def test_unknown_ids_are_ignored(self):
        payload = shape_map.compute_index_shape_map(
            self.NAME, n_clusters=1, dims=3,
            only_file_ids=["file_0", "file_1", "not_in_index"],
        )
        self.assertEqual(payload["count"], 2)


class ShapeMapOutlierTests(unittest.TestCase):
    """Splitting points that k-means only absorbed for lack of anywhere else."""

    def _agg(self):
        # Two tight clusters plus one row pointing somewhere else entirely.
        # Rows are L2-normalised, as _shape_map_project leaves them.
        rows = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.99, 0.14, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.14, 0.99, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    def test_disabled_threshold_returns_the_same_object(self):
        agg = self._agg()
        assign = np.array([0, 0, 1, 1, 0], dtype=np.int64)
        out, effective = core._shape_map_outlier_assign(agg, assign, 0.0)
        self.assertIs(out, assign)
        self.assertEqual(effective, 0.0)

    def test_far_point_becomes_minus_one(self):
        agg = self._agg()
        assign = np.array([0, 0, 1, 1, 0], dtype=np.int64)
        out, effective = core._shape_map_outlier_assign(agg, assign, 0.8)
        self.assertEqual(out[4], -1)
        # The four points that really do sit on their centroid are untouched.
        self.assertEqual(list(out[:4]), [0, 0, 1, 1])
        # Nothing to cap here: the request is below the ceiling.
        self.assertAlmostEqual(effective, 0.8)

    def test_an_impossible_threshold_is_capped_not_abandoned(self):
        # The regression this guards: the pass used to hand back the original
        # clustering whole when fewer than two points survived, so asking for
        # the strictest setting produced exactly the same map as Off -- no
        # outliers at all. It must now clamp down to the ceiling and still
        # separate something.
        agg = self._agg()
        assign = np.array([0, 0, 1, 1, 0], dtype=np.int64)
        out, effective = core._shape_map_outlier_assign(agg, assign, 0.999)
        self.assertIsNot(out, assign)
        self.assertIn(-1, list(out))
        self.assertGreater(effective, 0.0)
        self.assertLess(effective, 0.999)

    def test_capping_matches_asking_for_the_ceiling(self):
        agg = self._agg()
        assign = np.array([0, 0, 1, 1, 0], dtype=np.int64)
        sims = core._shape_map_centroid_sims(agg, assign)
        ceiling = core._shape_map_outlier_ceiling(sims)
        capped, capped_eff = core._shape_map_outlier_assign(agg, assign, 0.999)
        exact, exact_eff = core._shape_map_outlier_assign(agg, assign, ceiling)
        self.assertEqual(list(capped), list(exact))
        self.assertAlmostEqual(capped_eff, ceiling)
        self.assertAlmostEqual(exact_eff, ceiling)
        # Exactly two points survive at the ceiling, which is what makes it the
        # ceiling.
        self.assertEqual(int(np.count_nonzero(np.asarray(capped) >= 0)), 2)

    def test_ceiling_is_zero_without_two_points(self):
        self.assertEqual(core._shape_map_outlier_ceiling(np.array([], dtype=np.float32)), 0.0)
        self.assertEqual(core._shape_map_outlier_ceiling(np.array([0.9], dtype=np.float32)), 0.0)

    def test_cluster_passes_the_threshold_through(self):
        pytest.importorskip("faiss")
        agg = self._agg()
        coords = np.zeros((5, 3), dtype=np.float32)
        file_ids = [f"f{i}" for i in range(5)]
        points, k, _coh, _centres, effective = core._shape_map_cluster(
            agg, coords, file_ids, 2, 3, outlier_threshold=0.8
        )
        self.assertIn(-1, [p[4] for p in points])
        # k counts the k-means clusters; the outlier group is not one of them.
        self.assertEqual(k, 2)
        self.assertGreater(effective, 0.0)

    def test_cluster_is_unchanged_when_disabled(self):
        pytest.importorskip("faiss")
        agg = self._agg()
        coords = np.zeros((5, 3), dtype=np.float32)
        file_ids = [f"f{i}" for i in range(5)]
        before, k0, _c0, _centres, _e0 = core._shape_map_cluster(agg, coords, file_ids, 2, 3)
        after, k1, _c1, _centres, _e1 = core._shape_map_cluster(
            agg, coords, file_ids, 2, 3, outlier_threshold=0.0
        )
        self.assertEqual(before, after)
        self.assertEqual(k0, k1)

    def test_normalized_threshold_rounds_and_clamps(self):
        self.assertEqual(core._normalize_outlier_threshold(0.0), 0.0)
        self.assertEqual(core._normalize_outlier_threshold(None), 0.0)
        self.assertEqual(core._normalize_outlier_threshold(-1), 0.0)
        self.assertEqual(core._normalize_outlier_threshold("bad"), 0.0)
        self.assertEqual(core._normalize_outlier_threshold(0.70004), 0.7)
        self.assertEqual(core._normalize_outlier_threshold(5), 1.0)


class ShapeMapOutlierCategoryTests(unittest.TestCase):
    """Cluster -1 must read as "Outliers", not as an extra cluster."""

    def test_outliers_are_last_and_not_labelled_cluster_zero(self):
        clusters = [0, 0, 1, 1, -1]
        cat_index, categories = core._shape_map_categorize(clusters, [[]] * 5, "cluster")
        labels = [c["label"] for c in categories]
        self.assertEqual(labels, ["Cluster 1", "Cluster 2", "Outliers"])
        self.assertNotIn("Cluster 0", labels)
        self.assertEqual(categories[-1]["key"], "outliers")
        self.assertEqual(categories[-1]["kind"], "outliers")
        self.assertEqual(categories[-1]["count"], 1)
        self.assertEqual(cat_index[4], 2)

    def test_no_outlier_category_without_outliers(self):
        _cat_index, categories = core._shape_map_categorize(
            [0, 0, 1, 1], [[]] * 4, "cluster"
        )
        self.assertEqual([c["key"] for c in categories], ["cluster:0", "cluster:1"])

    def test_coloring_version_matches_the_pin(self):
        # The authoritative pin. See EXPECTED_COLORING_VERSION at the top of
        # this file for what the number means and why it is written down twice.
        self.assertEqual(core._SHAPE_MAP_COLORING_VERSION, EXPECTED_COLORING_VERSION)


class ShapeMapCohesionTests(unittest.TestCase):
    """Per-cluster mean cosine to the centre, surfaced in the legend."""

    def _agg(self, rows):
        import numpy as np
        A = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return A / norms

    def test_identical_vectors_are_perfectly_cohesive(self):
        import numpy as np
        agg = self._agg([[1, 0, 0]] * 4)
        assign = np.array([0, 0, 1, 1], dtype=np.int64)
        cohesion = core._shape_map_cohesion(agg, assign)
        self.assertEqual(sorted(cohesion), [0, 1])
        for value in cohesion.values():
            self.assertAlmostEqual(value, 1.0, places=3)

    def test_mixed_cluster_scores_lower_than_a_tight_one(self):
        import numpy as np
        agg = self._agg([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]])
        assign = np.array([0, 0, 1, 1], dtype=np.int64)
        cohesion = core._shape_map_cohesion(agg, assign)
        self.assertAlmostEqual(cohesion[0], 1.0, places=3)
        self.assertLess(cohesion[1], cohesion[0])

    def test_outliers_are_excluded_from_the_mean(self):
        import numpy as np
        agg = self._agg([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]])
        # Same population, but the odd one out has been split off as -1: the
        # remaining cluster must score higher than it did with it included.
        with_outlier = core._shape_map_cohesion(
            agg, np.array([0, 0, 0, 0], dtype=np.int64)
        )
        without = core._shape_map_cohesion(
            agg, np.array([0, 0, 0, -1], dtype=np.int64)
        )
        self.assertNotIn(-1, without)
        self.assertGreater(without[0], with_outlier[0])
        self.assertAlmostEqual(without[0], 1.0, places=3)

    def test_cluster_returns_cohesion_with_outliers_off(self):
        # Cohesion is not a by-product of the outlier pass: it must be there
        # even when the threshold is disabled.
        ids, values = _fake_batch(n_files=6, bodies_per_file=2, dim=8)
        file_ids, coords, agg = core._shape_map_project(ids, values)
        points, k, cohesion, _centres, _eff = core._shape_map_cluster(
            agg, coords, file_ids, n_clusters=1, dims=3, outlier_threshold=0.0
        )
        self.assertEqual(k, 1)
        self.assertEqual(sorted(cohesion), [0])
        self.assertLessEqual(cohesion[0], 1.0 + 1e-6)
        self.assertEqual(len(points), 6)

    def test_empty_index_returns_an_empty_cohesion_map(self):
        import numpy as np
        points, k, cohesion, _centres, _eff = core._shape_map_cluster(
            np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32),
            [], n_clusters=1, dims=3,
        )
        self.assertEqual((points, k, cohesion), ([], 0, {}))


class ShapeMapCohesionPayloadTests(unittest.TestCase):
    """Cohesion reaches the legend/groups, and only for k-means clusters."""

    def _points(self):
        return [
            ("a", 0.0, 0.0, 0.0, 0, 0.9),
            ("b", 0.0, 0.0, 0.0, 1, 0.8),
            ("c", 0.0, 0.0, 0.0, -1, 1.0),
        ]

    def _payload(self, color_by="cluster", tags_map=None):
        return core._shape_map_build_payload(
            "tmcad", self._points(), 2, 3, {}, tags_map or {},
            include_tags=False, color_by=color_by,
            cohesion={0: 0.9412, 1: 0.5},
        )

    def test_cluster_rows_carry_their_cohesion(self):
        payload = self._payload()
        by_label = {row["label"]: row for row in payload["legend"]}
        self.assertAlmostEqual(by_label["Cluster 1"]["cohesion"], 0.9412)
        self.assertAlmostEqual(by_label["Cluster 2"]["cohesion"], 0.5)
        groups = {g["key"]: g for g in payload["groups"]}
        self.assertAlmostEqual(groups["cluster:0"]["cohesion"], 0.9412)

    def test_non_cluster_categories_have_none(self):
        payload = self._payload()
        outliers = [row for row in payload["legend"] if row["kind"] == "outliers"]
        self.assertEqual(len(outliers), 1)
        self.assertIsNone(outliers[0]["cohesion"])
        self.assertIsNone(
            {g["key"]: g for g in payload["groups"]}["outliers"]["cohesion"]
        )

    def test_untagged_and_tag_rows_have_none(self):
        payload = self._payload(color_by="tag", tags_map={"a": ["bracket"]})
        kinds = {row["kind"] for row in payload["legend"]}
        self.assertIn("untagged", kinds)
        self.assertIn("tag", kinds)
        for row in payload["legend"]:
            self.assertIsNone(row["cohesion"])
        for group in payload["groups"]:
            self.assertIsNone(group["cohesion"])

    def test_tag_rows_past_the_palette_have_no_cohesion(self):
        # Tag colouring cycles the palette instead of overflowing, so there is
        # no "other" row any more; the rows past slot 24 are ordinary tag rows
        # and, like every tag row, carry no cohesion.
        n = len(core._SHAPE_MAP_PALETTE) + 2
        tags_map = {f"f{i}": [f"t{i:03d}"] for i in range(n)}
        points = [(f"f{i}", 0.0, 0.0, 0.0, 0, 1.0) for i in range(n)]
        payload = core._shape_map_build_payload(
            "tmcad", points, 1, 3, {}, tags_map, include_tags=False,
            color_by="tag", cohesion={0: 0.9},
        )
        self.assertNotIn("other", [row["kind"] for row in payload["legend"]])
        tag_rows = [row for row in payload["legend"] if row["kind"] == "tag"]
        self.assertEqual(len(tag_rows), n)
        for row in tag_rows:
            self.assertIsNone(row["cohesion"])

    def test_missing_cohesion_leaves_rows_at_none(self):
        # The scs path and any older caller may not pass one.
        payload = core._shape_map_build_payload(
            "tmcad", self._points(), 2, 3, {}, {}, include_tags=False,
        )
        for row in payload["legend"]:
            self.assertIsNone(row["cohesion"])


class ShapeMapCentreTests(unittest.TestCase):
    """The real part standing in for each cluster's k-means centroid."""

    def _agg(self, rows):
        import numpy as np
        A = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return A / norms

    def test_centre_is_the_member_nearest_the_centroid(self):
        import numpy as np
        # Rows 0 and 1 straddle the centroid; row 1 sits exactly on it.
        agg = self._agg([[1.0, 0.3, 0.0], [1.0, 0.0, 0.0], [1.0, -0.3, 0.0]])
        assign = np.array([0, 0, 0], dtype=np.int64)
        centres = core._shape_map_centres(agg, assign, ["a", "b", "c"])
        self.assertEqual(centres, {0: "b"})

    def test_an_outlier_is_never_chosen_as_a_centre(self):
        import numpy as np
        # _shape_map_centroid_sims(skip_outliers=True) leaves the neutral value
        # 1.0 on the rows it skipped, so an argmax over the whole array would
        # return the outlier -- which is higher than anything a real member of a
        # spread-out cluster can score. The mask is what prevents it.
        agg = self._agg([
            [1.0, 0.4, 0.0],   # cluster 0, off to one side
            [1.0, 0.0, 0.0],   # cluster 0, the true centre
            [1.0, -0.4, 0.0],  # cluster 0, off to the other side
            [0.0, 0.0, 1.0],   # outlier, nothing like the rest
        ])
        assign = np.array([0, 0, 0, -1], dtype=np.int64)
        centres = core._shape_map_centres(agg, assign, ["a", "b", "c", "stray"])
        self.assertEqual(sorted(centres), [0])
        self.assertNotIn(-1, centres)
        self.assertNotEqual(centres[0], "stray")
        self.assertEqual(centres[0], "b")

    def test_every_cluster_gets_its_own_centre(self):
        import numpy as np
        agg = self._agg([[1, 0, 0], [1, 0.2, 0], [0, 1, 0], [0.2, 1, 0]])
        assign = np.array([0, 0, 1, 1], dtype=np.int64)
        centres = core._shape_map_centres(agg, assign, ["a", "b", "c", "d"])
        self.assertEqual(sorted(centres), [0, 1])
        self.assertIn(centres[0], ("a", "b"))
        self.assertIn(centres[1], ("c", "d"))

    def test_cluster_returns_centres_with_outliers_off(self):
        ids, values = _fake_batch(n_files=6, bodies_per_file=2, dim=8)
        file_ids, coords, agg = core._shape_map_project(ids, values)
        _points, k, _cohesion, centres, _eff = core._shape_map_cluster(
            agg, coords, file_ids, n_clusters=1, dims=3, outlier_threshold=0.0
        )
        self.assertEqual(k, 1)
        self.assertEqual(sorted(centres), [0])
        self.assertIn(centres[0], file_ids)

    def test_centres_reach_the_legend_and_groups(self):
        points = [
            ("a", 0.0, 0.0, 0.0, 0, 0.9),
            ("b", 0.0, 0.0, 0.0, 1, 0.8),
            ("c", 0.0, 0.0, 0.0, -1, 1.0),
        ]
        payload = core._shape_map_build_payload(
            "tmcad", points, 2, 3, {}, {}, include_tags=False,
            color_by="cluster", centres={0: "a", 1: "b"},
        )
        by_label = {row["label"]: row for row in payload["legend"]}
        self.assertEqual(by_label["Cluster 1"]["centre_file_id"], "a")
        self.assertEqual(by_label["Cluster 2"]["centre_file_id"], "b")
        groups = {g["key"]: g for g in payload["groups"]}
        self.assertEqual(groups["cluster:0"]["centre_file_id"], "a")
        # Outliers are not a cluster, so they have no centre.
        self.assertIsNone(groups["outliers"]["centre_file_id"])
        outliers = [r for r in payload["legend"] if r["kind"] == "outliers"]
        self.assertIsNone(outliers[0]["centre_file_id"])

    def test_missing_centres_leave_rows_at_none(self):
        payload = core._shape_map_build_payload(
            "tmcad", [("a", 0.0, 0.0, 0.0, 0, 1.0)], 1, 3, {}, {}, include_tags=False,
        )
        for row in payload["legend"]:
            self.assertIsNone(row["centre_file_id"])
        for group in payload["groups"]:
            self.assertIsNone(group["centre_file_id"])


class ShapeMapKmeansEnvTests(unittest.TestCase):
    """The FAISS k-means knobs: default to today's behaviour, key the caches."""

    def test_unset_environment_reproduces_the_current_behaviour(self):
        pytest.importorskip("faiss")
        ids, values = _fake_batch(n_files=10, bodies_per_file=2, dim=8)
        file_ids, coords, agg = core._shape_map_project(ids, values)
        with patch.dict(os.environ, {}, clear=False):
            for name in (
                "HOOPS_AI_SHAPE_MAP_KMEANS_MAX_POINTS_PER_CENTROID",
                "HOOPS_AI_SHAPE_MAP_KMEANS_NREDO",
                "HOOPS_AI_SHAPE_MAP_KMEANS_SPHERICAL",
                "HOOPS_AI_SHAPE_MAP_KMEANS_INIT",
            ):
                os.environ.pop(name, None)
            p1, k1, _c1, _ctr1, _e1 = core._shape_map_cluster(
                agg, coords, file_ids, n_clusters=4, dims=3
            )
            p2, k2, _c2, _ctr2, _e2 = core._shape_map_cluster(
                agg, coords, file_ids, n_clusters=4, dims=3
            )
        self.assertEqual(k1, k2)
        self.assertEqual(p1, p2)

    def test_signature_is_empty_at_the_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            for name in (
                "HOOPS_AI_SHAPE_MAP_KMEANS_MAX_POINTS_PER_CENTROID",
                "HOOPS_AI_SHAPE_MAP_KMEANS_NREDO",
                "HOOPS_AI_SHAPE_MAP_KMEANS_SPHERICAL",
                "HOOPS_AI_SHAPE_MAP_KMEANS_INIT",
            ):
                os.environ.pop(name, None)
            # Empty, so appending it leaves every existing cache key untouched.
            self.assertEqual(core._shape_map_kmeans_signature(), ())
            base = ("tmcad", 3, 8)
            self.assertEqual(base + core._shape_map_kmeans_signature(), base)

    def test_any_override_changes_the_signature(self):
        base = ("tmcad", 3, 8)
        for name, value in (
            ("HOOPS_AI_SHAPE_MAP_KMEANS_MAX_POINTS_PER_CENTROID", "1024"),
            ("HOOPS_AI_SHAPE_MAP_KMEANS_NREDO", "5"),
            ("HOOPS_AI_SHAPE_MAP_KMEANS_SPHERICAL", "1"),
            ("HOOPS_AI_SHAPE_MAP_KMEANS_INIT", "kmeans++"),
            ("HOOPS_AI_SHAPE_MAP_KMEANS_INIT", "afkmc2"),
        ):
            with self.subTest(name=name):
                with patch.dict(os.environ, {name: value}):
                    sig = core._shape_map_kmeans_signature()
                    self.assertNotEqual(sig, ())
                    self.assertNotEqual(base + sig, base)

    def test_bad_values_fall_back_to_the_defaults(self):
        cases = (
            ("HOOPS_AI_SHAPE_MAP_KMEANS_NREDO", "abc"),
            ("HOOPS_AI_SHAPE_MAP_KMEANS_NREDO", "0"),
            ("HOOPS_AI_SHAPE_MAP_KMEANS_NREDO", "500"),
            ("HOOPS_AI_SHAPE_MAP_KMEANS_MAX_POINTS_PER_CENTROID", "not-a-number"),
            ("HOOPS_AI_SHAPE_MAP_KMEANS_MAX_POINTS_PER_CENTROID", "0"),
            ("HOOPS_AI_SHAPE_MAP_KMEANS_MAX_POINTS_PER_CENTROID", "99999999"),
            # Anything but "1" leaves spherical off; there is no failure mode to
            # warn about, so it is simply not enabled.
            ("HOOPS_AI_SHAPE_MAP_KMEANS_SPHERICAL", "true"),
            # An unknown initialisation is reported and ignored rather than
            # being handed to FAISS.
            ("HOOPS_AI_SHAPE_MAP_KMEANS_INIT", "elkan"),
            ("HOOPS_AI_SHAPE_MAP_KMEANS_INIT", ""),
            # The default spelled out explicitly is still the default, so it
            # must not start keying the caches differently.
            ("HOOPS_AI_SHAPE_MAP_KMEANS_INIT", "RANDOM"),
        )
        for name, value in cases:
            with self.subTest(name=name, value=value):
                with patch.dict(os.environ, {name: value}):
                    self.assertEqual(
                        core._shape_map_kmeans_nredo(),
                        shape_map._KMEANS_NREDO_DEFAULT,
                    )
                    self.assertEqual(
                        core._shape_map_kmeans_max_points_per_centroid(),
                        shape_map._KMEANS_MAX_POINTS_PER_CENTROID_DEFAULT,
                    )
                    self.assertEqual(
                        core._shape_map_kmeans_spherical(),
                        shape_map._KMEANS_SPHERICAL_DEFAULT,
                    )
                    self.assertEqual(
                        core._shape_map_kmeans_init(),
                        shape_map._KMEANS_INIT_DEFAULT,
                    )
                    # A rejected value must not reach the cache keys either.
                    self.assertEqual(core._shape_map_kmeans_signature(), ())

    def test_initialisation_aliases_resolve_to_one_name(self):
        cases = {
            "kmeans++": "kmeans++",
            "k-means++": "kmeans++",
            "kmeans_plus_plus": "kmeans++",
            "  KMeans++  ": "kmeans++",
            "afkmc2": "afkmc2",
            "afk-mc2": "afkmc2",
            "random": "random",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                with patch.dict(
                    os.environ, {"HOOPS_AI_SHAPE_MAP_KMEANS_INIT": value}
                ):
                    self.assertEqual(core._shape_map_kmeans_init(), expected)

    def test_a_faiss_without_the_constants_is_not_an_error(self):
        """A build with no ClusteringInitMethod_* must degrade, not raise."""

        class _Bare:
            pass

        bare = _Bare()
        for name in ("random", "kmeans++", "afkmc2"):
            with self.subTest(name=name):
                self.assertIsNone(core._shape_map_kmeans_init_method(bare, name))
        # An unknown name is None too, so the caller never passes a stray value.
        self.assertIsNone(core._shape_map_kmeans_init_method(bare, "elkan"))

        class _Full:
            ClusteringInitMethod_RANDOM = 0
            ClusteringInitMethod_KMEANS_PLUS_PLUS = 1
            ClusteringInitMethod_AFK_MC2 = 2

        full = _Full()
        self.assertEqual(core._shape_map_kmeans_init_method(full, "kmeans++"), 1)
        self.assertEqual(core._shape_map_kmeans_init_method(full, "afkmc2"), 2)


class ShapeMapPaletteCycleTests(unittest.TestCase):
    """Past 24 clusters the palette repeats, darkened, instead of overflowing."""

    def test_first_lap_is_the_palette_verbatim(self):
        for i, rgb in enumerate(core._SHAPE_MAP_PALETTE):
            self.assertEqual(core._shape_map_cluster_rgb(i), rgb)

    def test_second_lap_is_a_darker_shade_of_the_first(self):
        n = len(core._SHAPE_MAP_PALETTE)
        base = core._shape_map_cluster_rgb(0)
        wrapped = core._shape_map_cluster_rgb(n)
        self.assertNotEqual(wrapped, base)
        # Same hue family: each channel scaled by the same lap factor.
        factor = core._SHAPE_MAP_CYCLE_FACTORS[1]
        self.assertEqual(wrapped, tuple(round(ch * factor) for ch in base))

    def test_every_generated_colour_is_three_bytes(self):
        for i in range(0, len(core._SHAPE_MAP_PALETTE) * 5):
            rgb = core._shape_map_cluster_rgb(i)
            self.assertEqual(len(rgb), 3)
            for ch in rgb:
                self.assertIsInstance(ch, int)
                self.assertGreaterEqual(ch, 0)
                self.assertLessEqual(ch, 255)

    def test_thirty_clusters_get_thirty_categories_and_no_other(self):
        clusters = list(range(30))
        _cat_index, categories = core._shape_map_categorize(
            clusters, [[]] * len(clusters), "cluster"
        )
        self.assertEqual(len(categories), 30)
        self.assertTrue(all(c["kind"] == "cluster" for c in categories))
        self.assertNotIn("other", [c["key"] for c in categories])
        self.assertEqual(categories[-1]["label"], "Cluster 30")

    def test_outliers_still_trail_a_wrapped_cluster_set(self):
        clusters = list(range(30)) + [-1]
        _cat_index, categories = core._shape_map_categorize(
            clusters, [[]] * len(clusters), "cluster"
        )
        self.assertEqual(categories[-1]["key"], "outliers")
        self.assertEqual(categories[-1]["rgb"], core._SHAPE_MAP_OVERFLOW_RGB)

    def test_tag_colouring_cycles_the_palette_too(self):
        # Tags are named and unbounded, and folding them cost each one its
        # legend row, checkbox and SCS node. They now cycle exactly as clusters
        # do, so there is no "Other" bucket in either colouring.
        n = len(core._SHAPE_MAP_PALETTE) + 3
        tags = [[f"t{i:03d}"] for i in range(n)]
        _cat_index, categories = core._shape_map_categorize([0] * n, tags, "tag")
        self.assertEqual(len(categories), n)
        self.assertTrue(all(c["kind"] == "tag" for c in categories))
        self.assertNotIn("other", [c["key"] for c in categories])
        self.assertEqual(
            [c["key"] for c in categories], [f"tag:t{i:03d}" for i in range(n)]
        )

    def test_tag_colours_past_the_palette_are_darker_than_the_first_lap(self):
        size = len(core._SHAPE_MAP_PALETTE)
        n = size + 3
        tags = [[f"t{i:03d}"] for i in range(n)]
        _cat_index, categories = core._shape_map_categorize([0] * n, tags, "tag")
        for i in range(size, n):
            # Since v12 the lightness cycles and the hue steps, so slots 24, 25
            # and 26 are all darkenings of the FIRST palette colour rather than
            # of the slot one lap back.
            steps = len(core._SHAPE_MAP_CYCLE_FACTORS) - 1
            first_lap = tuple(categories[(i - size) // steps]["rgb"])
            wrapped = tuple(categories[i]["rgb"])
            self.assertNotEqual(wrapped, first_lap)
            for later, base in zip(wrapped, first_lap):
                self.assertLessEqual(later, base)
            self.assertLess(sum(wrapped), sum(first_lap))

    def test_tag_colouring_within_the_palette_is_unchanged(self):
        # Regression guard: maps with 24 tags or fewer must look exactly as
        # they did before the palette started cycling.
        size = len(core._SHAPE_MAP_PALETTE)
        tags = [[f"t{i:03d}"] for i in range(size)]
        _cat_index, categories = core._shape_map_categorize([0] * size, tags, "tag")
        self.assertEqual(
            [tuple(c["rgb"]) for c in categories], list(core._SHAPE_MAP_PALETTE)
        )

    def test_untagged_stays_last_past_the_palette(self):
        n = len(core._SHAPE_MAP_PALETTE) + 3
        tags = [[f"t{i:03d}"] for i in range(n)] + [[], []]
        _cat_index, categories = core._shape_map_categorize([0] * len(tags), tags, "tag")
        self.assertEqual(categories[-1]["key"], "untagged")
        self.assertEqual(categories[-1]["rgb"], core._SCS_UNTAGGED_RGB)
        self.assertEqual(categories[-1]["count"], 2)

    def test_coloring_version_is_past_the_tag_cycle_bump(self):
        # The .scs files bake one node per category, so the tag node layout
        # changed with the cycle; a stale v10 file would hide the wrong group.
        self.assertGreaterEqual(core._SHAPE_MAP_COLORING_VERSION, 11)
        self.assertEqual(core._SHAPE_MAP_COLORING_VERSION, EXPECTED_COLORING_VERSION)

    def test_the_first_twenty_four_slots_are_the_palette_untouched(self):
        # Stated separately from test_first_lap_is_the_palette_verbatim because
        # this is the core guarantee of v12: every map with 24 categories or
        # fewer is coloured exactly as it was under every earlier version.
        self.assertEqual(
            [core._shape_map_cluster_rgb(i) for i in range(24)],
            list(core._SHAPE_MAP_PALETTE[:24]),
        )

    def test_no_two_consecutive_slots_share_a_lightness(self):
        # The point of v12: cluster ids are a size ranking, so a lightness that
        # moved only every 24 slots made lightness mean "small".
        factors = core._SHAPE_MAP_CYCLE_FACTORS[1:]
        width = len(core._SHAPE_MAP_PALETTE)
        for i in range(width, 512):
            with self.subTest(i=i):
                self.assertNotEqual(
                    factors[(i - width) % len(factors)],
                    factors[(i + 1 - width) % len(factors)],
                )
                self.assertNotEqual(
                    core._shape_map_cluster_rgb(i), core._shape_map_cluster_rgb(i + 1)
                )

    def test_three_consecutive_slots_are_one_hue_then_the_next_hue_starts(self):
        width = len(core._SHAPE_MAP_PALETTE)
        factors = core._SHAPE_MAP_CYCLE_FACTORS[1:]
        base = core._SHAPE_MAP_PALETTE[0]
        for step, factor in enumerate(factors):
            self.assertEqual(
                core._shape_map_cluster_rgb(width + step),
                tuple(max(0, min(255, round(ch * factor))) for ch in base),
            )
        # The slot after the last step moves on to the next palette colour.
        nxt = core._shape_map_cluster_rgb(width + len(factors))
        self.assertEqual(
            nxt,
            tuple(
                max(0, min(255, round(ch * factors[0])))
                for ch in core._SHAPE_MAP_PALETTE[1]
            ),
        )

    def test_any_three_consecutive_slots_past_the_palette_mix_lightnesses(self):
        width = len(core._SHAPE_MAP_PALETTE)
        factors = core._SHAPE_MAP_CYCLE_FACTORS[1:]
        for i in range(width, 512):
            window = {factors[(i + d - width) % len(factors)] for d in range(3)}
            with self.subTest(i=i):
                self.assertGreaterEqual(len(window), 2)

    def test_two_hundred_and_fifty_six_slots_use_all_ninety_six_colours(self):
        seen = {core._shape_map_cluster_rgb(i) for i in range(256)}
        self.assertEqual(len(seen), 96)

    def test_the_colour_of_a_slot_is_deterministic(self):
        for i in (0, 23, 24, 25, 97, 255, 1023):
            with self.subTest(i=i):
                self.assertEqual(
                    core._shape_map_cluster_rgb(i), core._shape_map_cluster_rgb(i)
                )


class ShapeMapOutlierCacheKeyTests(ShapeMapTagCacheInvalidationTests):
    """The threshold keys the result cache but must not disturb the old keys."""

    NAME = "outliermap"

    def test_disabled_threshold_hits_the_pre_outlier_cache_entry(self):
        first = shape_map.compute_index_shape_map(self.NAME, n_clusters=1, dims=3)
        second = shape_map.compute_index_shape_map(
            self.NAME, n_clusters=1, dims=3, outlier_threshold=0.0
        )
        self.assertIs(second, first, "outlier_threshold=0 changed the cache key")

    def test_enabled_threshold_uses_a_different_entry(self):
        first = shape_map.compute_index_shape_map(self.NAME, n_clusters=1, dims=3)
        second = shape_map.compute_index_shape_map(
            self.NAME, n_clusters=1, dims=3, outlier_threshold=0.8
        )
        self.assertIsNot(second, first)

    def test_equivalent_thresholds_share_an_entry(self):
        first = shape_map.compute_index_shape_map(
            self.NAME, n_clusters=1, dims=3, outlier_threshold=0.8
        )
        second = shape_map.compute_index_shape_map(
            self.NAME, n_clusters=1, dims=3, outlier_threshold=0.80004
        )
        self.assertIs(second, first)


class ShapeMapProjectionMethodTests(unittest.TestCase):
    """The selectable display projection (PCA, the default, or UMAP)."""

    def test_normalizer_folds_to_pca(self):
        for raw in ("PCA", "pca", None, "", "nonsense", "Umapp"):
            with self.subTest(raw=raw):
                self.assertEqual(
                    shape_map._normalize_map_projection(raw), "pca"
                )

    def test_normalizer_recognises_umap(self):
        for raw in ("umap", "UMAP", "Umap"):
            with self.subTest(raw=raw):
                self.assertEqual(
                    shape_map._normalize_map_projection(raw), "umap"
                )

    def test_pca_signature_is_empty(self):
        # This emptiness is load-bearing, not an implementation detail: it is
        # what keeps every cache key -- and so every .scs already baked on disk
        # -- bit-for-bit unchanged now that a projection exists. Anything that
        # makes the PCA fragment non-empty silently invalidates the lot.
        self.assertEqual(shape_map._shape_map_projection_signature("pca"), ())
        self.assertEqual(shape_map._shape_map_projection_signature(None), ())
        self.assertEqual(
            shape_map._shape_map_projection_signature("nonsense"), ()
        )

    def test_umap_signature_is_never_empty(self):
        # Deliberately unlike the k-means signature, which collapses to empty
        # at its defaults: an empty UMAP fragment would hash to the PCA key and
        # be answered with the PCA coordinates. The defaults are the common
        # case, so assert it explicitly with the environment untouched.
        sig = shape_map._shape_map_projection_signature("umap")
        self.assertTrue(sig)
        self.assertEqual(len(sig), 1)
        entry = sig[0]
        self.assertEqual(entry[0], "umap")
        self.assertIn(shape_map._shape_map_umap_n_neighbors(), entry)
        self.assertIn(shape_map._shape_map_umap_min_dist(), entry)
        self.assertIn(shape_map._shape_map_umap_seed(), entry)

    def test_default_projection_is_pca(self):
        ids, values = _fake_batch(n_files=7, bodies_per_file=2, dim=8)
        a_ids, a_coords, a_agg = shape_map._shape_map_project(ids, values)
        b_ids, b_coords, b_agg = shape_map._shape_map_project(
            ids, values, projection="pca"
        )
        self.assertEqual(a_ids, b_ids)
        np.testing.assert_array_equal(a_coords, b_coords)
        np.testing.assert_array_equal(a_agg, b_agg)

    @unittest.skipUnless(
        shape_map._shape_map_umap_available(), "umap-learn not installed"
    )
    def test_umap_projection_is_centred(self):
        ids, values = _fake_batch(n_files=40, bodies_per_file=2, dim=8, seed=3)
        file_ids, coords, agg = shape_map._shape_map_project(
            ids, values, projection="umap"
        )
        self.assertEqual(len(file_ids), 40)
        self.assertEqual(coords.shape, (40, 3))
        # _shape_map_cluster only scales, never re-centres, so the projection
        # itself has to hand back a mean-centred cloud.
        np.testing.assert_allclose(
            coords.mean(axis=0), np.zeros(3), atol=1e-4
        )


class ShapeMapClusteringMethodTests(unittest.TestCase):
    """The selectable clustering algorithm (k-means, the default, or Ward)."""

    @staticmethod
    def _partition(points):
        """The clustering as a family of SETS of file ids.

        Compared this way on purpose: the ids themselves are reassigned by the
        canonical renumbering (member count, then cohesion), so two runs that
        found the same families can label them differently. What has to hold
        between k=2 and k=4 is a property of the partition, not of the labels.
        """
        groups = {}
        for fid, _x, _y, _z, cluster, _sim in points:
            groups.setdefault(cluster, set()).add(fid)
        return {frozenset(members) for members in groups.values()}

    def test_normalizer_folds_to_kmeans(self):
        for raw in ("KMEANS", "kmeans", None, "", "nonsense", "wards"):
            with self.subTest(raw=raw):
                self.assertEqual(
                    shape_map._normalize_map_clustering(raw), "kmeans"
                )

    def test_normalizer_recognises_ward(self):
        for raw in ("ward", "WARD", "Ward"):
            with self.subTest(raw=raw):
                self.assertEqual(
                    shape_map._normalize_map_clustering(raw), "ward"
                )

    def test_kmeans_signature_is_empty(self):
        # This emptiness is what keeps every existing cache entry and every
        # .scs already baked on disk valid: they were all built with k-means,
        # and an empty fragment leaves their keys bit-for-bit unchanged.
        # Anything that makes this non-empty silently invalidates the lot.
        self.assertEqual(shape_map._shape_map_clustering_signature("kmeans"), ())
        self.assertEqual(shape_map._shape_map_clustering_signature(None), ())
        self.assertEqual(
            shape_map._shape_map_clustering_signature("nonsense"), ()
        )

    def test_ward_signature_is_never_empty(self):
        # Ward has no tunable parameters at all, so there is nothing to put in
        # the fragment -- which is exactly why this needs asserting. An empty
        # one would hash to the k-means entry for the same index and be
        # answered with the k-means assignment.
        sig = shape_map._shape_map_clustering_signature("ward")
        self.assertTrue(sig)
        self.assertEqual(len(sig), 1)
        self.assertEqual(sig[0][0], "ward")

    def test_default_clustering_is_kmeans(self):
        pytest.importorskip("faiss")
        ids, values = _fake_batch(n_files=10, bodies_per_file=2, dim=8)
        file_ids, coords, agg = shape_map._shape_map_project(ids, values)
        a = shape_map._shape_map_cluster(agg, coords, file_ids, 4, 3)
        b = shape_map._shape_map_cluster(
            agg, coords, file_ids, 4, 3, clustering="kmeans"
        )
        self.assertEqual(a, b)

    def test_ward_cuts_nest(self):
        # The reason Ward is offered at all: every cut of one tree refines
        # every coarser cut, so raising Clusters (k) splits the families
        # already on screen instead of redrawing them. k-means gives no such
        # guarantee -- it refits from scratch at each k.
        pytest.importorskip("scipy")
        ids, values = _fake_batch(n_files=24, bodies_per_file=2, dim=8, seed=5)
        file_ids, coords, agg = shape_map._shape_map_project(ids, values)
        coarse, k2, _c, _ce, _e = shape_map._shape_map_cluster(
            agg, coords, file_ids, 2, 3, outlier_threshold=0.0, clustering="ward"
        )
        fine, k4, _c, _ce, _e = shape_map._shape_map_cluster(
            agg, coords, file_ids, 4, 3, outlier_threshold=0.0, clustering="ward"
        )
        self.assertEqual(k2, 2)
        self.assertEqual(k4, 4)
        coarse_sets = self._partition(coarse)
        fine_sets = self._partition(fine)
        self.assertEqual(len(coarse_sets), 2)
        self.assertEqual(len(fine_sets), 4)
        for small in fine_sets:
            # Each fine cluster lies wholly inside exactly one coarse cluster.
            containers = [big for big in coarse_sets if small <= big]
            self.assertEqual(
                len(containers), 1,
                f"{sorted(small)} is not contained in exactly one coarse cluster",
            )
        # And nothing was lost or gained on the way down.
        self.assertEqual(
            set().union(*coarse_sets), set().union(*fine_sets)
        )

    def test_ward_refuses_above_the_point_cap(self):
        pytest.importorskip("scipy")
        ids, values = _fake_batch(n_files=10, bodies_per_file=2, dim=8)
        file_ids, coords, agg = shape_map._shape_map_project(ids, values)
        with patch.dict(
            os.environ, {"HOOPS_AI_SHAPE_MAP_WARD_MAX_POINTS": "4"}, clear=False
        ):
            self.assertEqual(shape_map._shape_map_ward_max_points(), 4)
            with self.assertRaises(shape_map.WardTooLarge) as ctx:
                shape_map._shape_map_cluster(
                    agg, coords, file_ids, 2, 3, clustering="ward"
                )
        # The message has to name the numbers and the knob, because the only
        # way out of this is to change one of them.
        message = str(ctx.exception)
        self.assertIn("10", message)
        self.assertIn("4", message)
        self.assertIn("HOOPS_AI_SHAPE_MAP_WARD_MAX_POINTS", message)


class ShapeMapCondensedDistanceTests(unittest.TestCase):
    """Ward's distances are built here now, so they must still be scipy's."""

    # float64 carries about 16 significant digits, and the error of
    # ||xi||^2 + ||xj||^2 - 2*(xi . xj) grows with the dimension and the row
    # magnitude: at D=16 and |x| of order 1 that is a few times 1e-14 in d^2,
    # and the square root halves it. 1e-10 is therefore several orders of
    # magnitude looser than the arithmetic needs while still being a thousand
    # times tighter than float32 could manage -- so it cannot pass by accident
    # on a precision mistake.
    TOLERANCE = 1e-10

    @staticmethod
    def _rows(n=23, dim=16, seed=11, normalize=False):
        import numpy as np

        rng = np.random.default_rng(seed)
        rows = rng.standard_normal((n, dim)).astype(np.float32)
        if normalize:
            rows = shape_map._shape_map_l2norm_rows(rows)
        return rows

    def _assert_matches_pdist(self, rows, block_rows):
        import numpy as np
        from scipy.spatial.distance import pdist

        got = shape_map._shape_map_condensed_sqdist(rows, block_rows=block_rows)
        want = pdist(np.asarray(rows, dtype=np.float64))
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got.dtype, np.dtype(np.float64))
        self.assertLess(float(np.max(np.abs(got - want))), self.TOLERANCE)

    def test_matches_pdist_on_normalised_rows(self):
        pytest.importorskip("scipy")
        self._assert_matches_pdist(self._rows(normalize=True), 7)

    def test_matches_pdist_on_rows_that_are_not_normalised(self):
        # The short form d^2 = 2 - 2s is only correct for unit vectors. These
        # rows are not, so a regression to that form fails here rather than
        # silently building a different tree once _shape_map_l2norm_rows moves.
        pytest.importorskip("scipy")
        self._assert_matches_pdist(self._rows(normalize=False), 7)

    def test_the_block_width_does_not_change_the_answer(self):
        # Blocking is an implementation detail of how the products are issued;
        # one row at a time, an awkward width and a single block covering
        # everything must all give the same distances.
        pytest.importorskip("scipy")
        rows = self._rows(n=23, normalize=True)
        for block_rows in (1, 7, 100):
            with self.subTest(block_rows=block_rows):
                self._assert_matches_pdist(rows, block_rows)

    def test_a_repeated_row_gives_exactly_zero(self):
        # Two identical rows make d^2 come out a shade negative, and without
        # the clip that is a NaN -- for exactly the near-duplicate pairs a CAD
        # index is full of.
        import numpy as np
        pytest.importorskip("scipy")

        rows = self._rows(n=6, normalize=True)
        rows[3] = rows[1]
        got = shape_map._shape_map_condensed_sqdist(rows)
        self.assertTrue(bool(np.all(np.isfinite(got))))
        n = rows.shape[0]
        i, j = 1, 3
        self.assertEqual(float(got[n * i - i * (i + 1) // 2 + (j - i - 1)]), 0.0)


class ShapeMapLinkageEquivalenceTests(unittest.TestCase):
    """The tree built from our distances is the tree scipy would have built."""

    @staticmethod
    def _rows(n=60, dim=16, seed=3):
        import numpy as np

        rng = np.random.default_rng(seed)
        rows = rng.standard_normal((n, dim)).astype(np.float32)
        return shape_map._shape_map_l2norm_rows(rows)

    @staticmethod
    def _same_partition(a, b) -> bool:
        """Whether two labellings are the same partition, renumbering aside.

        Cluster ids are arbitrary, so the map from one labelling to the other
        has to be checked in BOTH directions: a labelling that merged two
        clusters would still be a well-defined function one way round.
        """
        forward: dict = {}
        backward: dict = {}
        for left, right in zip(list(a), list(b)):
            if forward.setdefault(left, right) != right:
                return False
            if backward.setdefault(right, left) != left:
                return False
        return True

    def test_the_cut_matches_scipy_on_observations(self):
        import numpy as np
        pytest.importorskip("scipy")
        from scipy.cluster.hierarchy import fcluster, linkage

        rows = self._rows()
        ours = shape_map._shape_map_linkage_matrix(rows)
        theirs = linkage(np.asarray(rows, dtype=np.float64), method="ward")

        for k in (2, 4, 8, 16):
            with self.subTest(k=k):
                a = fcluster(ours, k, criterion="maxclust").tolist()
                b = fcluster(theirs, k, criterion="maxclust").tolist()
                self.assertEqual(len(set(a)), len(set(b)))
                self.assertTrue(
                    self._same_partition(a, b),
                    f"the two trees partition differently at k={k}",
                )

    def test_the_merge_heights_match_scipy(self):
        # Same shape, same order, same distances: column 3 of the linkage
        # matrix is where each merge happened, so a large difference here would
        # mean the merge ORDER diverged even where the cut happens to agree.
        import numpy as np
        pytest.importorskip("scipy")
        from scipy.cluster.hierarchy import linkage

        rows = self._rows()
        ours = np.asarray(shape_map._shape_map_linkage_matrix(rows), dtype=np.float64)
        theirs = linkage(np.asarray(rows, dtype=np.float64), method="ward")
        self.assertEqual(ours.shape, theirs.shape)
        self.assertLess(float(np.max(np.abs(ours[:, 2] - theirs[:, 2]))), 1e-9)


class ShapeMapWardDrillTests(unittest.TestCase):
    """Drilling into a Ward map descends the parent tree instead of rebuilding."""

    @staticmethod
    def _parent(n_files=24, seed=5):
        """A tree over a synthetic parent map, with its file ids."""
        ids, values = _fake_batch(n_files=n_files, bodies_per_file=2, dim=8, seed=seed)
        file_ids, coords, agg = shape_map._shape_map_project(ids, values)
        tree = shape_map._shape_map_linkage_matrix(agg)
        return tree, file_ids, coords, agg

    @staticmethod
    def _partition(points):
        groups = {}
        for fid, _x, _y, _z, cluster, _sim in points:
            groups.setdefault(cluster, set()).add(fid)
        return {frozenset(members) for members in groups.values()}

    def test_a_whole_cluster_is_recognised_as_a_node(self):
        # What Drill in produces when the user drills a cluster of the map
        # they are looking at: the subset IS a node of the tree, so it has a
        # position in the hierarchy to descend from.
        pytest.importorskip("scipy")
        tree, file_ids, _coords, _agg = self._parent()
        labels = shape_map._shape_map_run_ward(tree, 2)
        chosen = [fid for fid, lab in zip(file_ids, labels) if int(lab) == 0]
        rows = shape_map._shape_map_ward_subset_node(tree, file_ids, chosen)
        self.assertIsNotNone(rows)
        self.assertEqual(sorted(int(r) for r in rows),
                         sorted(file_ids.index(fid) for fid in chosen))

    def test_a_mixed_subset_is_not_a_node(self):
        # Half of one cluster and half of another -- what a tag filter or a
        # hand-picked selection looks like. There is no single node that holds
        # it, so the caller has to cluster it on its own.
        pytest.importorskip("scipy")
        tree, file_ids, _coords, _agg = self._parent()
        labels = shape_map._shape_map_run_ward(tree, 2)
        left = [fid for fid, lab in zip(file_ids, labels) if int(lab) == 0]
        right = [fid for fid, lab in zip(file_ids, labels) if int(lab) == 1]
        self.assertGreaterEqual(len(left), 2)
        self.assertGreaterEqual(len(right), 2)
        mixed = left[: len(left) // 2 or 1] + right[: len(right) // 2 or 1]
        self.assertIsNone(
            shape_map._shape_map_ward_subset_node(tree, file_ids, mixed)
        )

    def test_descending_covers_the_subset_exactly(self):
        # The descent must answer for the drilled files and for nothing else:
        # the labels come from the parent's tree, so an off-by-one in the row
        # mapping would quietly describe other parts.
        pytest.importorskip("scipy")
        tree, file_ids, coords, agg = self._parent()
        labels = shape_map._shape_map_run_ward(tree, 2)
        chosen = [fid for fid, lab in zip(file_ids, labels) if int(lab) == 0]
        rows = shape_map._shape_map_ward_subset_node(tree, file_ids, chosen)

        keep = set(chosen)
        sub_ids = [fid for fid in file_ids if fid in keep]
        sub_rows = [file_ids.index(fid) for fid in sub_ids]
        points, _k, _c, _ce, _e = shape_map._shape_map_cluster(
            agg[sub_rows], coords[sub_rows], sub_ids, 2, 3,
            outlier_threshold=0.0, clustering="ward",
            ward_parent=(tree, rows),
        )
        self.assertEqual({p[0] for p in points}, keep)

    def test_descended_cuts_nest(self):
        # The point of the whole change: inside a drill-down, raising the
        # cluster count still splits what is on screen rather than redrawing
        # it, because both cuts come from the one parent tree.
        pytest.importorskip("scipy")
        tree, file_ids, coords, agg = self._parent()
        labels = shape_map._shape_map_run_ward(tree, 2)
        chosen = [fid for fid, lab in zip(file_ids, labels) if int(lab) == 0]
        rows = shape_map._shape_map_ward_subset_node(tree, file_ids, chosen)

        keep = set(chosen)
        sub_ids = [fid for fid in file_ids if fid in keep]
        sub_rows = [file_ids.index(fid) for fid in sub_ids]

        def cut(k):
            points, _k, _c, _ce, _e = shape_map._shape_map_cluster(
                agg[sub_rows], coords[sub_rows], sub_ids, k, 3,
                outlier_threshold=0.0, clustering="ward",
                ward_parent=(tree, rows),
            )
            return self._partition(points)

        coarse_sets = cut(2)
        fine_sets = cut(4)
        self.assertGreaterEqual(len(fine_sets), len(coarse_sets))
        for small in fine_sets:
            containers = [big for big in coarse_sets if small <= big]
            self.assertEqual(
                len(containers), 1,
                f"{sorted(small)} is not contained in exactly one coarse cluster",
            )
        self.assertEqual(set().union(*coarse_sets), set().union(*fine_sets))


class ShapeMapWardGranularityTests(unittest.TestCase):
    """Cutting the Ward tree at a height instead of to a fixed cluster count."""

    @staticmethod
    def _tree(n_files=24, seed=7):
        ids, values = _fake_batch(n_files=n_files, bodies_per_file=2, dim=8, seed=seed)
        file_ids, coords, agg = shape_map._shape_map_project(ids, values)
        tree = shape_map._shape_map_linkage_matrix(agg)
        return tree, file_ids, coords, agg

    @staticmethod
    def _partition(points):
        groups = {}
        for fid, _x, _y, _z, cluster, _sim in points:
            groups.setdefault(cluster, set()).add(fid)
        return {frozenset(members) for members in groups.values()}

    def _cut(self, tree, agg, coords, file_ids, granularity):
        height, _lo, _hi, k_max = shape_map._shape_map_ward_cut_height(
            tree, None, granularity,
        )
        points, k, _c, _ce, _e = shape_map._shape_map_cluster(
            agg, coords, file_ids, 0, 3, outlier_threshold=0.0,
            clustering="ward", linkage_z=tree, ward_height=height,
        )
        return points, k, height, k_max

    def test_zero_granularity_is_one_cluster(self):
        # The coarse end of the slider is the root of the tree: everything is
        # within the cut height of everything else, so one family comes back.
        pytest.importorskip("scipy")
        tree, file_ids, coords, agg = self._tree()
        _points, k, _h, _k_max = self._cut(tree, agg, coords, file_ids, 0.0)
        self.assertEqual(k, 1)

    def test_full_granularity_reaches_k_max(self):
        # The fine end is the cap the helper reports, min(N, 256). Anything
        # less would mean the slider cannot reach the detail it advertises.
        pytest.importorskip("scipy")
        tree, file_ids, coords, agg = self._tree()
        _points, k, _h, k_max = self._cut(tree, agg, coords, file_ids, 1.0)
        self.assertEqual(k_max, min(len(file_ids), 256))
        self.assertEqual(k, k_max)

    def test_ward_max_moves_the_ceiling(self):
        # The ceiling is a parameter, still clamped to the population: a
        # granularity means a count read against min(m, ward_max), so asking
        # for another ceiling has to move it and nothing else.
        pytest.importorskip("scipy")
        tree, file_ids, _coords, _agg = self._tree()
        m = len(file_ids)
        _h, _lo, _hi, k_max = shape_map._shape_map_ward_cut_height(
            tree, None, 1.0, ward_max=512,
        )
        self.assertEqual(k_max, min(m, 512))
        # Below the population it is the ceiling that binds, which is what
        # makes it worth passing.
        _h2, _lo2, _hi2, k_max_small = shape_map._shape_map_ward_cut_height(
            tree, None, 1.0, ward_max=8,
        )
        self.assertEqual(k_max_small, 8)

    def test_cluster_count_never_falls_as_granularity_rises(self):
        # Raising the slider lowers the cut, and lowering a cut can only split
        # families, never merge them. If this ever reverses, the control is
        # lying about what it does.
        pytest.importorskip("scipy")
        tree, file_ids, coords, agg = self._tree()
        counts = []
        for g in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(granularity=g):
                _points, k, _h, _k_max = self._cut(tree, agg, coords, file_ids, g)
                counts.append(k)
        self.assertEqual(counts, sorted(counts))

    def test_the_reported_height_reproduces_the_partition(self):
        # The payload reports cut_height so the cut can be read rather than
        # guessed at. That is only worth anything if cutting at the reported
        # height really does give back what was served.
        pytest.importorskip("scipy")
        from scipy.cluster.hierarchy import fcluster

        tree, file_ids, coords, agg = self._tree()
        points, _k, height, _k_max = self._cut(tree, agg, coords, file_ids, 0.5)
        labels = fcluster(tree, float(height), criterion="distance")
        expected = {}
        for fid, lab in zip(file_ids, labels):
            expected.setdefault(int(lab), set()).add(fid)
        self.assertEqual(
            self._partition(points),
            {frozenset(members) for members in expected.values()},
        )

    def test_no_granularity_leaves_the_count_path_untouched(self):
        # Backward compatibility: a request without a granularity has to be
        # the same computation, point for point, that it was before heights
        # existed -- otherwise every cached map and every baked .scs is wrong.
        pytest.importorskip("scipy")
        tree, file_ids, coords, agg = self._tree()
        before = shape_map._shape_map_cluster(
            agg, coords, file_ids, 4, 3, outlier_threshold=0.0,
            clustering="ward", linkage_z=tree,
        )
        after = shape_map._shape_map_cluster(
            agg, coords, file_ids, 4, 3, outlier_threshold=0.0,
            clustering="ward", linkage_z=tree, ward_height=None,
        )
        self.assertEqual(before, after)

    def test_granularity_signature_is_empty_unless_in_use(self):
        # Same rule as the clustering fragment: empty keeps every entry and
        # every .scs already on disk addressable by the key it was written on.
        self.assertEqual(
            shape_map._shape_map_granularity_signature("ward", None), ()
        )
        self.assertEqual(
            shape_map._shape_map_granularity_signature("kmeans", 0.5), ()
        )
        sig = shape_map._shape_map_granularity_signature("ward", 0.5)
        self.assertEqual(sig, (("g", 0.5),))

    def test_ward_max_reaches_the_key_only_when_it_is_not_the_default(self):
        # The default is what every entry and every .scs written before this
        # was a parameter assumed, so the key has to stay byte for byte the
        # same there; another ceiling means another cut and must not collide.
        self.assertEqual(
            shape_map._shape_map_granularity_signature("ward", 0.5, 256),
            (("g", 0.5),),
        )
        self.assertEqual(
            shape_map._shape_map_granularity_signature("ward", 0.5, None),
            (("g", 0.5),),
        )
        self.assertEqual(
            shape_map._shape_map_granularity_signature("ward", 0.5, 512),
            (("g", 0.5), ("wmax", 512)),
        )
        # Still nothing to key when the tree is not being cut at a height.
        self.assertEqual(
            shape_map._shape_map_granularity_signature("ward", None, 512), ()
        )
        self.assertEqual(
            shape_map._shape_map_granularity_signature("kmeans", 0.5, 512), ()
        )


class ShapeMapWardSuggestionTests(unittest.TestCase):
    """The tree reporting where its own merge cost jumps."""

    @staticmethod
    def _tree(n_files=24, seed=7):
        ids, values = _fake_batch(n_files=n_files, bodies_per_file=2, dim=8, seed=seed)
        file_ids, coords, agg = shape_map._shape_map_project(ids, values)
        tree = shape_map._shape_map_linkage_matrix(agg)
        return tree, file_ids, coords, agg

    def test_whole_tree_heights_are_the_linkage_column(self):
        # With no node to restrict to, every merge builds the population, so
        # the array is the linkage's own height column and nothing else.
        pytest.importorskip("scipy")
        import numpy as np

        tree, _file_ids, _coords, _agg = self._tree()
        heights = shape_map._shape_map_ward_internal_heights(tree, None)
        np.testing.assert_allclose(heights, np.sort(tree[:, 2]))

    def test_node_heights_are_the_merges_that_build_it(self):
        # A node of m points is assembled by exactly m - 1 merges, and every
        # one of them is a row of the parent tree -- this is the claim that
        # replaced the bisection, so it is the one worth pinning.
        pytest.importorskip("scipy")
        import numpy as np

        tree, file_ids, _coords, _agg = self._tree()
        labels = shape_map._shape_map_run_ward(tree, 2)
        chosen = [fid for fid, lab in zip(file_ids, labels) if int(lab) == 0]
        rows = shape_map._shape_map_ward_subset_node(tree, file_ids, chosen)
        self.assertIsNotNone(rows)

        heights = shape_map._shape_map_ward_internal_heights(tree, rows)
        self.assertEqual(int(heights.shape[0]), int(np.asarray(rows).size) - 1)
        self.assertTrue(np.all(np.diff(heights) >= 0))
        column = set(np.asarray(tree[:, 2], dtype=float).tolist())
        for value in heights.tolist():
            self.assertIn(value, column)

    def test_each_suggestion_round_trips_through_the_slider(self):
        # A suggestion is only usable if putting its granularity on the slider
        # lands on that cut. If the forward and inverse maps ever drift apart,
        # clicking a suggestion would quietly give a different map.
        pytest.importorskip("scipy")
        import numpy as np
        from scipy.cluster.hierarchy import fcluster

        tree, file_ids, _coords, _agg = self._tree()
        heights = shape_map._shape_map_ward_internal_heights(tree, None)
        m = int(heights.shape[0]) + 1
        k_max = min(m, 256)
        cuts = shape_map._shape_map_ward_suggestions(heights, m, k_max)
        self.assertTrue(cuts)
        for cut in cuts:
            with self.subTest(clusters=cut["clusters"]):
                height, _lo, _hi, _k = shape_map._shape_map_ward_cut_height(
                    tree, None, cut["granularity"],
                )
                self.assertAlmostEqual(height, cut["height"])
                labels = fcluster(tree, float(height), criterion="distance")
                self.assertEqual(
                    int(np.unique(labels).size), int(cut["clusters"])
                )

    def test_suggestions_are_ordered_and_capped(self):
        # They are drawn along a slider, so they come back in slider order,
        # and never more than the caller asked for.
        pytest.importorskip("scipy")
        tree, _file_ids, _coords, _agg = self._tree()
        heights = shape_map._shape_map_ward_internal_heights(tree, None)
        m = int(heights.shape[0]) + 1
        cuts = shape_map._shape_map_ward_suggestions(heights, m, min(m, 256), limit=3)
        self.assertLessEqual(len(cuts), 3)
        counts = [c["clusters"] for c in cuts]
        self.assertEqual(counts, sorted(counts))
        self.assertGreaterEqual(min(counts), 2)

    def test_nothing_to_suggest_when_the_band_is_a_single_cluster(self):
        # k_max <= 1 leaves no cut that splits anything, and an empty list is
        # the honest answer rather than a fabricated one.
        pytest.importorskip("scipy")
        tree, _file_ids, _coords, _agg = self._tree()
        heights = shape_map._shape_map_ward_internal_heights(tree, None)
        self.assertEqual(
            shape_map._shape_map_ward_suggestions(heights, heights.shape[0] + 1, 1),
            [],
        )


class ShapeMapWardTreeTests(unittest.TestCase):
    """Folding Ward's binary tree into the levels a tree view can show."""

    @staticmethod
    def _tree(n_files=24, seed=7):
        ids, values = _fake_batch(n_files=n_files, bodies_per_file=2, dim=8, seed=seed)
        file_ids, coords, agg = shape_map._shape_map_project(ids, values)
        tree = shape_map._shape_map_linkage_matrix(agg)
        return tree, file_ids, coords, agg

    @staticmethod
    def _subtrees(nodes, leaf_of_member):
        """Member set of every node, rebuilt from leaf_of_member alone."""
        import numpy as np

        own = {node["id"]: set() for node in nodes}
        for member, node_id in enumerate(np.asarray(leaf_of_member).tolist()):
            own[int(node_id)].add(member)
        children = {node["id"]: [] for node in nodes}
        for node in nodes:
            if node["parent"] != -1:
                children[node["parent"]].append(node["id"])

        subtrees = {}

        def walk(node_id):
            members = set(own[node_id])
            for child in children[node_id]:
                members |= walk(child)
            subtrees[node_id] = members
            return members

        root = next(node["id"] for node in nodes if node["parent"] == -1)
        walk(root)
        return subtrees, children

    def _node_rows(self, tree, file_ids):
        labels = shape_map._shape_map_run_ward(tree, 2)
        chosen = [fid for fid, lab in zip(file_ids, labels) if int(lab) == 0]
        rows = shape_map._shape_map_ward_subset_node(tree, file_ids, chosen)
        self.assertIsNotNone(rows)
        return rows, chosen

    def test_the_whole_tree_fits_in_one_response(self):
        # A level at most doubles the groups and stops at k_max, so the node
        # count is bounded whatever the population is. That bound is the reason
        # the tree can be served whole, with no paging.
        pytest.importorskip("scipy")
        tree, file_ids, _coords, agg = self._tree()
        nodes, _leaf = shape_map._shape_map_ward_tree(tree, agg, file_ids, None, k_max=8)
        self.assertLessEqual(len(nodes), 2 * 8 - 1)

    def test_there_is_one_root_and_it_holds_everyone(self):
        # The root is the population itself: anything else would mean the tree
        # describes only part of what was asked about.
        pytest.importorskip("scipy")
        tree, file_ids, _coords, agg = self._tree()
        nodes, _leaf = shape_map._shape_map_ward_tree(tree, agg, file_ids, None, k_max=8)
        roots = [node for node in nodes if node["parent"] == -1]
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0]["count"], len(file_ids))

    def test_a_parents_count_is_its_childrens(self):
        # The counts are what the view shows, so they have to add up: a split
        # distributes a node's members, it never drops or duplicates one.
        pytest.importorskip("scipy")
        tree, file_ids, _coords, agg = self._tree()
        nodes, _leaf = shape_map._shape_map_ward_tree(tree, agg, file_ids, None, k_max=8)
        by_id = {node["id"]: node for node in nodes}
        totals: dict = {}
        for node in nodes:
            if node["parent"] != -1:
                totals[node["parent"]] = totals.get(node["parent"], 0) + node["count"]
        for parent_id, total in totals.items():
            with self.subTest(node=parent_id):
                self.assertEqual(total, by_id[parent_id]["count"])

    def test_members_land_on_nodes_that_have_no_children(self):
        # leaf_of_member names the DEEPEST node holding a member. If it ever
        # named a node that went on to split, the view would show that member
        # twice: once there and once under a child.
        pytest.importorskip("scipy")
        import numpy as np

        tree, file_ids, _coords, agg = self._tree()
        nodes, leaf = shape_map._shape_map_ward_tree(tree, agg, file_ids, None, k_max=8)
        parents = {node["parent"] for node in nodes if node["parent"] != -1}
        for node_id in np.unique(np.asarray(leaf)).tolist():
            with self.subTest(node=int(node_id)):
                self.assertNotIn(int(node_id), parents)

    def test_every_level_is_an_exact_refinement(self):
        # Rebuilt from leaf_of_member alone, a parent's members are the disjoint
        # union of its children's. This is the whole contract of the structure:
        # descending the view narrows the population and never changes it.
        pytest.importorskip("scipy")
        tree, file_ids, _coords, agg = self._tree()
        nodes, leaf = shape_map._shape_map_ward_tree(tree, agg, file_ids, None, k_max=8)
        subtrees, children = self._subtrees(nodes, leaf)
        for node in nodes:
            kids = children[node["id"]]
            if not kids:
                continue
            with self.subTest(node=node["id"]):
                union = set()
                total = 0
                for child in kids:
                    union |= subtrees[child]
                    total += len(subtrees[child])
                self.assertEqual(total, len(union))  # disjoint
                self.assertEqual(union, subtrees[node["id"]])
                self.assertEqual(len(union), node["count"])

    def test_two_clusters_give_two_levels(self):
        # The levels are the targets 1, 2, 4, ... k_max, so k_max = 2 is the
        # root and one split beneath it -- nothing deeper gets invented.
        pytest.importorskip("scipy")
        tree, file_ids, _coords, agg = self._tree()
        nodes, _leaf = shape_map._shape_map_ward_tree(tree, agg, file_ids, None, k_max=2)
        self.assertEqual(sorted({node["level"] for node in nodes}), [1, 2])
        self.assertLessEqual(len(nodes), 3)

    def test_a_drill_down_node_folds_the_same_way(self):
        # Inside a node the levels are cut from the parent's own tree, so every
        # property above has to survive the restriction -- and the root has to
        # be the node, not the index it came from.
        pytest.importorskip("scipy")
        tree, file_ids, _coords, agg = self._tree()
        rows, chosen = self._node_rows(tree, file_ids)
        sub_ids = [file_ids[int(r)] for r in rows]
        nodes, leaf = shape_map._shape_map_ward_tree(tree, agg, sub_ids, rows, k_max=8)

        self.assertLessEqual(len(nodes), 2 * 8 - 1)
        roots = [node for node in nodes if node["parent"] == -1]
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0]["count"], len(chosen))

        subtrees, children = self._subtrees(nodes, leaf)
        parents = {node["parent"] for node in nodes if node["parent"] != -1}
        for node_id in set(int(v) for v in leaf.tolist()):
            self.assertNotIn(node_id, parents)
        for node in nodes:
            kids = children[node["id"]]
            if not kids:
                continue
            with self.subTest(node=node["id"]):
                union = set()
                total = 0
                for child in kids:
                    union |= subtrees[child]
                    total += len(subtrees[child])
                self.assertEqual(total, len(union))
                self.assertEqual(union, subtrees[node["id"]])
                self.assertEqual(len(union), node["count"])


class ShapeMapPairSpreadTests(unittest.TestCase):
    """The pairwise cosines inside one group.

    Cohesion is the mean cosine to the centre -- a radius. These are the
    distances between the members themselves, which is the only way to see
    whether a group that clears the threshold actually holds together.
    """

    @staticmethod
    def _unit(*angles):
        """Unit rows in the plane at the given angles, in radians."""
        return np.array(
            [[np.cos(a), np.sin(a)] for a in angles], dtype="float32",
        ).reshape(-1, 2)

    def test_three_known_angles_give_the_hand_computed_values(self):
        # 0, 30 and 90 degrees: the pairs are cos30, cos90 and cos60.
        spread = shape_map.shape_map_pair_spread(
            self._unit(0.0, np.pi / 6, np.pi / 2)
        )
        expect = [
            round(float(np.cos(np.pi / 6)), 4),
            0.0,
            round(float(np.cos(np.pi / 3)), 4),
        ]
        self.assertEqual(spread["pairs"], 3)
        self.assertEqual(spread["min"], min(expect))
        self.assertAlmostEqual(spread["mean"], round(sum(expect) / 3, 4), places=3)
        # The widest pair is the first row against the third.
        self.assertEqual(spread["argmin"], (0, 2))

    def test_fewer_than_two_members_have_no_pair_at_all(self):
        for rows in (self._unit(), self._unit(0.3)):
            with self.subTest(n=len(rows)):
                spread = shape_map.shape_map_pair_spread(rows)
                self.assertEqual(spread["pairs"], 0)
                self.assertIsNone(spread["min"])
                self.assertIsNone(spread["mean"])
                self.assertEqual(spread["argmin"], (-1, -1))

    def test_two_members_have_one_pair_and_min_equals_mean(self):
        spread = shape_map.shape_map_pair_spread(self._unit(0.0, 0.4))
        self.assertEqual(spread["pairs"], 1)
        self.assertEqual(spread["min"], spread["mean"])
        self.assertEqual(spread["min"], round(float(np.cos(0.4)), 4))
        self.assertEqual(spread["argmin"], (0, 1))

    def test_identical_members_are_perfectly_tight(self):
        spread = shape_map.shape_map_pair_spread(self._unit(0.7, 0.7, 0.7))
        self.assertEqual(spread["pairs"], 3)
        self.assertEqual(spread["min"], 1.0)
        self.assertEqual(spread["mean"], 1.0)

    def test_a_row_that_is_not_a_unit_vector_is_refused(self):
        rows = self._unit(0.0, 0.5, 1.0)
        rows[1] *= 2.0
        with self.assertRaises(ValueError) as caught:
            shape_map.shape_map_pair_spread(rows)
        message = str(caught.exception)
        # The row is named, so the caller can find it...
        self.assertIn("row 1", message)
        # ...and the vector itself never is.
        self.assertNotIn("0.5", message)
        self.assertNotIn(str(float(rows[1][0])), message)

    def test_renormalising_quietly_is_not_an_option(self):
        """A scaled row must fail, not be fixed: the numbers would still look
        like cosines while no longer being comparable with cohesion."""
        rows = self._unit(0.0, 0.5)
        rows[0] *= 0.5
        with self.assertRaises(ValueError):
            shape_map.shape_map_pair_spread(rows)

    def test_a_tie_resolves_to_the_same_pair_every_time(self):
        # Four rows at the compass points: 0-2 and 1-3 are both opposite, so
        # the minimum (-1.0) is a tie between two pairs.
        rows = self._unit(0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
        first = shape_map.shape_map_pair_spread(rows)
        second = shape_map.shape_map_pair_spread(rows)
        self.assertEqual(first["min"], -1.0)
        self.assertEqual(first["argmin"], second["argmin"])
        # np.triu_indices order, which is lexicographic in (i, j).
        self.assertEqual(first["argmin"], (0, 2))


class ShapeMapClusterCentroidTests(unittest.TestCase):
    """shape_map_cluster_centroids(agg, assign) -> (ids, centroid rows)."""

    @staticmethod
    def _unit(*angles):
        return np.array(
            [[np.cos(a), np.sin(a), 0.0] for a in angles], dtype="float32"
        )

    def test_the_centroid_is_the_normalised_mean_of_the_members(self):
        rows = self._unit(0.0, np.pi / 2, np.pi, np.pi)
        ids, centroids = shape_map.shape_map_cluster_centroids(rows, [0, 0, 1, 1])
        self.assertEqual(ids, [0, 1])
        # Cluster 0 is the bisector of 0 and 90 degrees; cluster 1 is 180.
        np.testing.assert_allclose(
            centroids[0], [np.sqrt(0.5), np.sqrt(0.5), 0.0], atol=1e-6,
        )
        np.testing.assert_allclose(centroids[1], [-1.0, 0.0, 0.0], atol=1e-6)

    def test_rows_are_unit_vectors(self):
        rows = self._unit(0.0, 0.3, 2.0, 2.4) * 3.0
        _ids, centroids = shape_map.shape_map_cluster_centroids(rows, [0, 0, 1, 1])
        np.testing.assert_allclose(
            np.linalg.norm(centroids, axis=1), [1.0, 1.0], atol=1e-6,
        )

    def test_outliers_are_not_a_cluster(self):
        """Cluster -1 is a display category: it has no centre to order by."""
        rows = self._unit(0.0, 0.3, 2.0)
        ids, centroids = shape_map.shape_map_cluster_centroids(rows, [-1, 0, 0])
        self.assertEqual(ids, [0])
        self.assertEqual(centroids.shape[0], 1)

    def test_ids_come_back_sorted_and_match_their_rows(self):
        rows = self._unit(0.0, np.pi / 2, np.pi)
        ids, centroids = shape_map.shape_map_cluster_centroids(rows, [7, 2, 5])
        self.assertEqual(ids, [2, 5, 7])
        np.testing.assert_allclose(centroids[0], rows[1], atol=1e-6)
        np.testing.assert_allclose(centroids[2], rows[0], atol=1e-6)

    def test_nothing_but_outliers_yields_no_clusters(self):
        rows = self._unit(0.0, 0.3)
        ids, centroids = shape_map.shape_map_cluster_centroids(rows, [-1, -1])
        self.assertEqual(ids, [])
        self.assertEqual(centroids.shape[0], 0)

    def test_an_assignment_of_the_wrong_length_is_refused(self):
        with self.assertRaises(ValueError):
            shape_map.shape_map_cluster_centroids(self._unit(0.0, 0.3), [0])


class ShapeMapLegendOrderTests(unittest.TestCase):
    """shape_map_legend_order(centroids) -> ([row index, ...], method)."""

    @staticmethod
    def _three_pairs():
        """Six centroids in three tight pairs, interleaved so no two neighbours
        in the input are a pair -- which is the case the order has to fix."""
        angles = [0.0, 2.0, 4.0, 0.05, 2.05, 4.05]
        return np.array(
            [[np.cos(a), np.sin(a), 0.0] for a in angles], dtype="float32"
        )

    def test_the_pairs_end_up_adjacent(self):
        order, method = shape_map.shape_map_legend_order(self._three_pairs())
        self.assertEqual(method, "ward+olo")
        self.assertEqual(sorted(order), [0, 1, 2, 3, 4, 5])
        at = {row: i for i, row in enumerate(order)}
        for a, b in ((0, 3), (1, 4), (2, 5)):
            self.assertEqual(abs(at[a] - at[b]), 1, f"{a} and {b} are not adjacent")

    def test_past_the_cap_the_plain_leaf_order_is_used(self):
        rows = self._three_pairs()
        order, method = shape_map.shape_map_legend_order(rows, olo_max=5)
        self.assertEqual(method, "ward")
        self.assertEqual(sorted(order), [0, 1, 2, 3, 4, 5])
        # The cap is a cost ceiling, not a different grouping: the plain leaf
        # order still keeps each pair together on this input.
        at = {row: i for i, row in enumerate(order)}
        for a, b in ((0, 3), (1, 4), (2, 5)):
            self.assertEqual(abs(at[a] - at[b]), 1)

    def test_exactly_at_the_cap_optimal_leaf_ordering_still_runs(self):
        _order, method = shape_map.shape_map_legend_order(
            self._three_pairs(), olo_max=6,
        )
        self.assertEqual(method, "ward+olo")

    def test_no_clusters(self):
        rows = np.zeros((0, 3), dtype="float32")
        self.assertEqual(shape_map.shape_map_legend_order(rows), ([], "ward"))

    def test_one_cluster(self):
        rows = np.array([[1.0, 0.0, 0.0]], dtype="float32")
        self.assertEqual(shape_map.shape_map_legend_order(rows), ([0], "ward"))

    def test_two_clusters(self):
        """scipy's leaf order over two observations is [0, 1], which is what
        the legend ordering relied on before the code moved here."""
        rows = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32")
        order, method = shape_map.shape_map_legend_order(rows)
        self.assertEqual(order, [0, 1])
        self.assertEqual(method, "ward+olo")

    def test_the_default_cap_is_the_measured_one(self):
        self.assertEqual(shape_map._LEGEND_ORDER_OLO_MAX, 512)


class ComputeIndexLegendOrderTests(unittest.TestCase):
    """compute_index_legend_order: the refusal that needs no index."""

    def test_ward_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            shape_map.compute_index_legend_order("tmcad", clustering="ward")
        self.assertIn("kmeans", str(caught.exception))


if __name__ == "__main__":
    unittest.main()

