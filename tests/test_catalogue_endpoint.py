"""Tests for the catalogue frontier and POST /similarity/index/{name}/map/catalogue.

SDK-free: the Ward tree is faked, so these pin the frontier rule, the
partition guarantee, the ordering, the min_parts accounting, the cache key and
the endpoint's validation without an index, a license or faiss.
"""

import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import index_registry  # noqa: E402
import shape_map  # noqa: E402

# The endpoint reads nothing from the environment itself, but config helpers on
# the import path do, and a developer's .env must not decide whether a test
# passes. Fixed here once, for every test in the file.
_ENV = {
    "HOOPS_AI_DATA_DIR": "C:\\tmp\\data",
    "HOOPS_LICENSE": "x",
}


def _make_app():
    from routers.similarity import router

    app = FastAPI()
    app.include_router(router)
    return app


def _tree():
    """A five-part tree: one tight pair, one loose pair, one lone part.

              0 (5, coh 0.10)
             /        |       \\
        1 (2, 0.98)  2 (2, 0.40)  3 (1, None)
                      /     \\
                 4 (1, None) 5 (1, None)

    At threshold 0.95 node 1 is reached; node 2 is not, so the descent carries
    on to its two childless children, which stall; node 3 stalls at once
    because a group of one has no cohesion to reach anything with.
    """
    nodes = [
        {"id": 0, "parent": -1, "level": 0, "count": 5, "height": 1.0,
         "cohesion": 0.10, "centre": "f0", "filename": "a.stp"},
        {"id": 1, "parent": 0, "level": 1, "count": 2, "height": 0.2,
         "cohesion": 0.98, "centre": "f1", "filename": "b.stp"},
        {"id": 2, "parent": 0, "level": 1, "count": 2, "height": 0.8,
         "cohesion": 0.40, "centre": "f3", "filename": "c.stp"},
        {"id": 3, "parent": 0, "level": 1, "count": 1, "height": 0.0,
         "cohesion": None, "centre": "f5", "filename": "d.stp"},
        {"id": 4, "parent": 2, "level": 2, "count": 1, "height": 0.0,
         "cohesion": None, "centre": "f3", "filename": "c.stp"},
        {"id": 5, "parent": 2, "level": 2, "count": 1, "height": 0.0,
         "cohesion": None, "centre": "f4", "filename": "e.stp"},
    ]
    return {
        "index": "tmcad",
        "count": 5,
        "k_max": 2048,
        "levels": [0, 1, 2],
        "file_ids": ["f1", "f2", "f3", "f4", "f5"],
        # Deepest node holding each file id, in the same order.
        "leaf": [1, 1, 4, 5, 3],
        "nodes": nodes,
    }


class FrontierPureTests(unittest.TestCase):
    """The descent and the partition proof, with no index anywhere near."""

    def test_frontier_splits_into_reached_and_stalled(self):
        frontier = shape_map.shape_map_cohesion_frontier(_tree()["nodes"], 0.95)
        self.assertEqual([n["id"] for n in frontier["reached"]], [1])
        self.assertEqual(sorted(n["id"] for n in frontier["stalled"]), [3, 4, 5])

    def test_frontier_is_a_partition(self):
        tree = _tree()
        frontier = shape_map.shape_map_cohesion_frontier(tree["nodes"], 0.95)
        shape_map.shape_map_check_frontier(tree["nodes"], frontier, tree["count"])

    def test_members_follow_the_leaves_up_to_their_group(self):
        tree = _tree()
        frontier = shape_map.shape_map_cohesion_frontier(tree["nodes"], 0.95)
        members = shape_map.shape_map_frontier_members(
            tree["nodes"], frontier, tree["file_ids"], tree["leaf"],
        )
        self.assertEqual(members[1], ["f1", "f2"])
        self.assertEqual(members[4], ["f3"])
        self.assertEqual(members[5], ["f4"])
        self.assertEqual(members[3], ["f5"])

    def test_order_is_pre_order_with_siblings_in_node_order(self):
        nodes = _tree()["nodes"]
        order = shape_map.shape_map_frontier_order(nodes, {1, 3, 4, 5})
        self.assertEqual(order, [1, 4, 5, 3])

    def test_order_stops_at_a_frontier_node(self):
        # With the root itself on the frontier nothing below it is listed.
        order = shape_map.shape_map_frontier_order(_tree()["nodes"], {0})
        self.assertEqual(order, [0])

    def test_order_keeps_an_unreachable_id_rather_than_dropping_it(self):
        nodes = [
            {"id": 0, "parent": -1, "count": 2},
            {"id": 1, "parent": 0, "count": 1},
            # Parented to a node that is not here: not reachable from a root.
            {"id": 9, "parent": 7, "count": 1},
        ]
        self.assertEqual(
            shape_map.shape_map_frontier_order(nodes, {1, 9}), [1, 9]
        )

    def test_group_tags_count_each_part_once_and_rank_by_frequency(self):
        tags_map = {
            "f1": ["bracket", "steel"],
            "f2": ["bracket"],
            "f3": [],
        }
        tagged, names = shape_map.shape_map_group_tags(["f1", "f2", "f3"], tags_map)
        self.assertEqual(tagged, 2)
        self.assertEqual(names, ["bracket", "steel"])

    def test_group_tags_cut_the_list_at_the_limit(self):
        tags_map = {f"f{i}": [f"t{i}"] for i in range(12)}
        tagged, names = shape_map.shape_map_group_tags(
            [f"f{i}" for i in range(12)], tags_map,
        )
        self.assertEqual(tagged, 12)
        self.assertEqual(len(names), shape_map.SHAPE_MAP_GROUP_TAG_LIMIT)

    def test_root_reached_stops_everything_below_it(self):
        frontier = shape_map.shape_map_cohesion_frontier(_tree()["nodes"], 0.05)
        self.assertEqual([n["id"] for n in frontier["reached"]], [0])
        self.assertEqual(frontier["stalled"], [])

    def test_check_rejects_a_frontier_that_misses_members(self):
        tree = _tree()
        frontier = {"reached": [tree["nodes"][1]], "stalled": []}
        with self.assertRaises(AssertionError):
            shape_map.shape_map_check_frontier(tree["nodes"], frontier, tree["count"])


class CatalogueSignatureTests(unittest.TestCase):
    """The house rule: a default option must leave the key untouched."""

    def test_defaults_add_no_fragment(self):
        self.assertEqual(shape_map._catalogue_signature(0.95, 2048, 1), ())

    def test_non_default_threshold_adds_a_fragment(self):
        self.assertEqual(
            shape_map._catalogue_signature(0.9, 2048, 1), (("t", 0.9),)
        )

    def test_non_default_k_max_adds_a_fragment(self):
        self.assertEqual(
            shape_map._catalogue_signature(0.95, 1024, 1), (("kmax", 1024),)
        )

    def test_non_default_min_parts_adds_a_fragment(self):
        self.assertEqual(
            shape_map._catalogue_signature(0.95, 2048, 5), (("min", 5),)
        )

    def test_default_min_reached_parts_adds_no_fragment(self):
        self.assertEqual(shape_map._catalogue_signature(0.95, 2048, 1, 5), ())

    def test_non_default_min_reached_parts_adds_a_fragment(self):
        self.assertEqual(
            shape_map._catalogue_signature(0.95, 2048, 1, 3), (("conf", 3),)
        )

    def test_fragments_accumulate_in_a_fixed_order(self):
        self.assertEqual(
            shape_map._catalogue_signature(0.5, 256, 3, 2),
            (("t", 0.5), ("kmax", 256), ("min", 3), ("conf", 2)),
        )


class ComputeCatalogueTests(unittest.TestCase):
    """compute_index_catalogue over a faked tree."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        faiss_path = pathlib.Path(self._tmp.name) / "tmcad.faiss"
        faiss_path.write_bytes(b"not a real index")
        self._faiss_path = faiss_path
        # A shared cache would let one test answer another's question.
        shape_map._catalogue_results.clear()
        self.addCleanup(shape_map._catalogue_results.clear)
        patcher = patch.dict(os.environ, _ENV, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _compute(self, tags_map=None, tags_etag="etag", **kwargs):
        with patch.object(
            index_registry, "_index_faiss_path", return_value=self._faiss_path
        ), patch.object(
            shape_map, "_shape_map_tags_snapshot",
            return_value=(tags_map or {}, tags_etag),
        ), patch.object(
            shape_map, "compute_index_shape_map_tree", return_value=_tree()
        ):
            return shape_map.compute_index_catalogue("tmcad", **kwargs)

    def test_groups_partition_the_population(self):
        result = self._compute()
        seen = [fid for group in result["groups"] for fid in group["file_ids"]]
        self.assertEqual(len(seen), result["total_parts"])
        self.assertEqual(len(set(seen)), len(seen))
        self.assertEqual(set(seen), set(_tree()["file_ids"]))

    def test_part_totals_split_by_how_the_branch_ended(self):
        # min_reached_parts=2 makes the tight pair confident; at the default 5
        # the same group is "small" (see the test below).
        result = self._compute(min_reached_parts=2)
        self.assertEqual(result["total_parts"], 5)
        self.assertEqual(result["stats"]["confident_parts"], 2)
        self.assertEqual(result["stats"]["small_parts"], 0)
        self.assertEqual(result["stats"]["mixed_parts"], 3)
        self.assertEqual(result["omitted_by_min_parts"], 0)

    def test_the_three_classes_partition_the_population(self):
        for floor in (1, 2, 3, 5, 100):
            with self.subTest(min_reached_parts=floor):
                stats = self._compute(min_reached_parts=floor)["stats"]
                self.assertEqual(
                    stats["confident_parts"] + stats["small_parts"] + stats["mixed_parts"],
                    5,
                )
                self.assertEqual(
                    stats["confident_groups"] + stats["small_groups"]
                    + stats["mixed_groups"],
                    4,
                )

    def test_a_tight_group_below_the_line_is_small_not_confident(self):
        # The pair holds 2 parts. At 2 it is confident, at 3 it is small, and
        # the boundary is inclusive: count == min_reached_parts is confident.
        by_floor = {
            floor: {g["id"]: g["kind"] for g in self._compute(min_reached_parts=floor)["groups"]}
            for floor in (2, 3)
        }
        self.assertEqual(by_floor[2][1], "confident")
        self.assertEqual(by_floor[3][1], "small")
        # Everything the descent never got tight on is mixed either way.
        for floor in (2, 3):
            for node_id in (3, 4, 5):
                self.assertEqual(by_floor[floor][node_id], "mixed")

    def test_min_reached_parts_only_labels_and_never_moves_the_frontier(self):
        wide = self._compute(min_reached_parts=1)
        narrow = self._compute(min_reached_parts=100)
        self.assertEqual(
            [(g["id"], g["count"], tuple(g["file_ids"])) for g in wide["groups"]],
            [(g["id"], g["count"], tuple(g["file_ids"])) for g in narrow["groups"]],
        )
        # Only the labels, and the counts derived from them, differ.
        self.assertEqual(wide["groups"][0]["kind"], "confident")
        self.assertEqual(narrow["groups"][0]["kind"], "small")

    def test_a_group_reports_the_tree_nodes_merge_height(self):
        height_of = {int(n["id"]): n["height"] for n in _tree()["nodes"]}
        for group in self._compute()["groups"]:
            self.assertIn("height", group)
            self.assertEqual(group["height"], height_of[group["id"]])

    def test_stats_match_the_known_tree(self):
        stats = self._compute(min_reached_parts=2)["stats"]
        self.assertEqual(stats["groups"], 4)
        self.assertEqual(stats["min_reached_parts"], 2)
        self.assertEqual(stats["confident_groups"], 1)
        self.assertEqual(stats["confident_parts"], 2)
        self.assertEqual(stats["confident_largest"], 2)
        self.assertEqual(stats["confident_median"], 2.0)
        self.assertEqual(stats["confident_singletons"], 0)
        self.assertEqual(stats["small_groups"], 0)
        self.assertEqual(stats["small_parts"], 0)
        self.assertEqual(stats["mixed_groups"], 3)
        self.assertEqual(stats["mixed_parts"], 3)
        self.assertEqual(stats["mixed_largest"], 1)
        self.assertEqual(stats["mixed_median"], 1.0)
        self.assertEqual(stats["mixed_singletons"], 3)
        self.assertEqual(stats["buckets"], [
            {"min_parts": 2, "groups": 1, "parts": 2},
            {"min_parts": 5, "groups": 0, "parts": 0},
            {"min_parts": 10, "groups": 0, "parts": 0},
            {"min_parts": 20, "groups": 0, "parts": 0},
        ])

    def test_buckets_count_confident_groups_only(self):
        # The same pair, now below the line: it leaves the buckets with it.
        stats = self._compute(min_reached_parts=3)["stats"]
        self.assertEqual(stats["buckets"][0], {"min_parts": 2, "groups": 0, "parts": 0})

    def test_stats_are_measured_before_min_parts_filters(self):
        wide = self._compute()["stats"]
        narrow = self._compute(min_parts=2)["stats"]
        # The cut of the tree did not change, so neither may its measurement.
        self.assertEqual(wide["buckets"], narrow["buckets"])
        self.assertEqual(wide["confident_parts"], narrow["confident_parts"])
        self.assertEqual(wide["small_parts"], narrow["small_parts"])
        self.assertEqual(wide["mixed_parts"], narrow["mixed_parts"])
        self.assertEqual(wide["confident_groups"], narrow["confident_groups"])
        self.assertEqual(wide["mixed_groups"], narrow["mixed_groups"])
        # Only the number of entries actually returned moves.
        self.assertEqual(wide["groups"], 4)
        self.assertEqual(narrow["groups"], 1)

    def test_groups_follow_the_tree_in_pre_order(self):
        # Root 0's children are 1, 2, 3 in node order, and the tree panel draws
        # them in exactly that order. Node 1 is on the frontier so nothing
        # below it is visited; node 2 is not, so its children 4 and 5 come
        # before its sibling 3.
        result = self._compute()
        self.assertEqual([g["id"] for g in result["groups"]], [1, 4, 5, 3])

    def test_min_parts_omits_groups_without_hiding_them(self):
        result = self._compute(min_parts=2)
        self.assertEqual([g["id"] for g in result["groups"]], [1])
        self.assertEqual(result["stats"]["groups"], 1)
        self.assertEqual(result["omitted_by_min_parts"], 3)
        # The totals still describe the whole index, not the filtered view.
        self.assertEqual(result["total_parts"], 5)
        self.assertEqual(result["stats"]["mixed_parts"], 3)

    def test_a_group_carries_its_centre_for_the_thumbnail_route(self):
        group = self._compute()["groups"][0]
        self.assertEqual(group["centre_file_id"], "f1")
        self.assertEqual(group["centre_filename"], "b.stp")
        self.assertEqual(group["level"], 1)

    def test_a_group_reports_how_much_of_it_is_tagged(self):
        # f1 and f2 are node 1's two members; only f1 carries anything, and a
        # part with two tags still counts once.
        result = self._compute(tags_map={"f1": ["bracket", "steel"]})
        groups = {g["id"]: g for g in result["groups"]}
        self.assertEqual(groups[1]["tagged_parts"], 1)
        self.assertEqual(groups[1]["count"], 2)
        self.assertEqual(sorted(groups[1]["tags"]), ["bracket", "steel"])
        # A group nobody has touched says so rather than leaving the fields out.
        self.assertEqual(groups[3]["tagged_parts"], 0)
        self.assertEqual(groups[3]["tags"], [])

    def test_tagging_a_part_invalidates_the_catalogue(self):
        first = self._compute()
        # Same request, new tags ETag: the result carries tag counts now, so it
        # must be rebuilt rather than served from the old entry.
        second = self._compute(tags_map={"f1": ["bracket"]}, tags_etag="etag-2")
        self.assertIsNot(first, second)
        self.assertEqual(
            {g["id"]: g["tagged_parts"] for g in second["groups"]}[1], 1
        )

    def test_the_same_tags_hit_the_same_entry(self):
        # The cached entry is the same object, tags and all.
        first = self._compute(tags_map={"f1": ["bracket"]})
        second = self._compute(tags_map={"f1": ["bracket"]})
        self.assertIs(first, second)

    def test_result_is_cached_under_the_same_key(self):
        first = self._compute()
        with patch.object(
            index_registry, "_index_faiss_path", return_value=self._faiss_path
        ), patch.object(
            shape_map, "_shape_map_tags_snapshot", return_value=({}, "etag")
        ), patch.object(
            shape_map, "compute_index_shape_map_tree",
            side_effect=AssertionError("the tree was rebuilt"),
        ):
            second = shape_map.compute_index_catalogue("tmcad")
        self.assertIs(first, second)

    def test_a_non_default_option_is_a_different_entry(self):
        self._compute()
        with patch.object(
            index_registry, "_index_faiss_path", return_value=self._faiss_path
        ), patch.object(
            shape_map, "_shape_map_tags_snapshot", return_value=({}, "etag")
        ), patch.object(
            shape_map, "compute_index_shape_map_tree", return_value=_tree()
        ) as tree:
            shape_map.compute_index_catalogue("tmcad", min_parts=2)
        self.assertEqual(tree.call_count, 1)

    def test_missing_index_raises_key_error(self):
        missing = pathlib.Path(self._tmp.name) / "nope.faiss"
        with patch.object(index_registry, "_index_faiss_path", return_value=missing):
            with self.assertRaises(KeyError):
                shape_map.compute_index_catalogue("tmcad")


class CatalogueEndpointTests(unittest.TestCase):
    """The route: wiring, response shape and the rejections."""

    def setUp(self):
        self.client = TestClient(_make_app())
        patcher = patch.dict(os.environ, _ENV, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _payload(self, groups=None):
        return {
            "index": "tmcad",
            "threshold": 0.95,
            "k_max": 2048,
            "total_parts": 5,
            "omitted_by_min_parts": 3,
            "stats": {
                "groups": 1,
                "min_reached_parts": 2,
                "confident_groups": 1,
                "confident_parts": 2,
                "confident_largest": 2,
                "confident_median": 2.0,
                "confident_singletons": 0,
                "small_groups": 0,
                "small_parts": 0,
                "small_largest": 0,
                "small_median": 0.0,
                "small_singletons": 0,
                "mixed_groups": 3,
                "mixed_parts": 3,
                "mixed_largest": 1,
                "mixed_median": 1.0,
                "mixed_singletons": 3,
                "buckets": [{"min_parts": 2, "groups": 1, "parts": 2}],
            },
            "groups": groups if groups is not None else [{
                "id": 1,
                "count": 2,
                "cohesion": 0.98,
                "height": 0.2,
                "level": 1,
                "kind": "confident",
                "reached": True,
                "centre_file_id": "f1",
                "centre_filename": "b.stp",
                "file_ids": ["f1", "f2"],
            }],
        }

    def test_defaults_reach_the_computation_unchanged(self):
        with patch.object(
            shape_map, "compute_index_catalogue", return_value=self._payload()
        ) as compute:
            resp = self.client.post("/similarity/index/tmcad/map/catalogue")
        self.assertEqual(resp.status_code, 200)
        kwargs = compute.call_args.kwargs
        self.assertEqual(kwargs["threshold"], 0.95)
        self.assertEqual(kwargs["k_max"], 2048)
        self.assertEqual(kwargs["kind"], "part")
        self.assertEqual(kwargs["min_parts"], 1)
        self.assertEqual(kwargs["min_reached_parts"], 5)
        self.assertIsNone(kwargs["tags"])

    def test_response_carries_the_groups_whole(self):
        with patch.object(
            shape_map, "compute_index_catalogue", return_value=self._payload()
        ):
            body = self.client.post("/similarity/index/tmcad/map/catalogue").json()
        self.assertEqual(body["omitted_by_min_parts"], 3)
        self.assertEqual(body["stats"]["buckets"][0]["parts"], 2)
        self.assertEqual(body["groups"][0]["file_ids"], ["f1", "f2"])
        self.assertEqual(body["groups"][0]["kind"], "confident")
        self.assertEqual(body["groups"][0]["height"], 0.2)
        self.assertTrue(body["groups"][0]["reached"])

    def test_a_stalled_group_may_have_no_cohesion(self):
        groups = [{
            "id": 3, "count": 1, "cohesion": None, "height": 0.0, "level": 1,
            "kind": "mixed", "reached": False,
            "centre_file_id": "f5", "centre_filename": "d.stp",
            "file_ids": ["f5"],
        }]
        with patch.object(
            shape_map, "compute_index_catalogue",
            return_value=self._payload(groups=groups),
        ):
            body = self.client.post("/similarity/index/tmcad/map/catalogue").json()
        self.assertIsNone(body["groups"][0]["cohesion"])
        self.assertEqual(body["groups"][0]["kind"], "mixed")

    def test_min_reached_parts_reaches_the_computation(self):
        with patch.object(
            shape_map, "compute_index_catalogue", return_value=self._payload()
        ) as compute:
            resp = self.client.post(
                "/similarity/index/tmcad/map/catalogue?min_reached_parts=3"
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(compute.call_args.kwargs["min_reached_parts"], 3)

    def test_min_reached_parts_below_one_is_rejected(self):
        resp = self.client.post(
            "/similarity/index/tmcad/map/catalogue?min_reached_parts=0"
        )
        self.assertEqual(resp.status_code, 422)

    def test_threshold_at_or_below_zero_is_rejected(self):
        resp = self.client.post("/similarity/index/tmcad/map/catalogue?threshold=0")
        self.assertEqual(resp.status_code, 422)

    def test_threshold_above_one_is_rejected(self):
        resp = self.client.post("/similarity/index/tmcad/map/catalogue?threshold=1.5")
        self.assertEqual(resp.status_code, 422)

    def test_k_max_below_two_is_rejected(self):
        resp = self.client.post("/similarity/index/tmcad/map/catalogue?k_max=1")
        self.assertEqual(resp.status_code, 422)

    def test_min_parts_below_one_is_rejected(self):
        resp = self.client.post("/similarity/index/tmcad/map/catalogue?min_parts=0")
        self.assertEqual(resp.status_code, 422)

    def test_missing_index_is_404(self):
        with patch.object(
            shape_map, "compute_index_catalogue", side_effect=KeyError("no such index")
        ):
            resp = self.client.post("/similarity/index/tmcad/map/catalogue")
        self.assertEqual(resp.status_code, 404)

    def test_tags_are_split_before_the_computation_sees_them(self):
        with patch.object(
            shape_map, "compute_index_catalogue", return_value=self._payload()
        ) as compute:
            self.client.post(
                "/similarity/index/tmcad/map/catalogue?tags=sheet,bracket&tag_mode=all"
            )
        kwargs = compute.call_args.kwargs
        self.assertEqual(kwargs["tags"], ["sheet", "bracket"])
        self.assertEqual(kwargs["tag_mode"], "all")


if __name__ == "__main__":
    unittest.main()
