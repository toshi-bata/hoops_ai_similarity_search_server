"""Router tests for POST /similarity/index/{name}/map (SDK-free).

These mock ``shape_map.compute_index_shape_map`` so they exercise the endpoint
wiring, validation and response shape without the HOOPS AI SDK or a license.
"""

import os
import sys
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core  # noqa: E402
import shape_map  # noqa: E402


def _make_app():
    from routers.similarity import router

    app = FastAPI()
    app.include_router(router)
    return app


def _fake_result(n=3):
    return {
        "index": "tmcad",
        "count": n,
        "clusters": 2,
        "dims": 3,
        "points": [
            {
                "file_id": f"id_{i}",
                "filename": f"part_{i}.stp",
                "x": 0.1 * i,
                "y": -0.1 * i,
                "z": 0.0,
                "cluster": i % 2,
            }
            for i in range(n)
        ],
        "legend": [
            {"label": "Cluster 1", "kind": "cluster", "count": n, "rgb": [230, 159, 0],
             "cohesion": 0.94},
            {"label": "Outliers", "kind": "outliers", "count": 0, "rgb": [128, 128, 128],
             "cohesion": None},
        ],
    }


class ShapeMapEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(_make_app())

    def test_endpoint_is_registered(self):
        # A registered route validates input rather than 404-ing on an unknown
        # path. With a mocked core the happy path returns 200.
        with patch.object(shape_map, "compute_index_shape_map", return_value=_fake_result()):
            resp = self.client.post("/similarity/index/tmcad/map")
        self.assertEqual(resp.status_code, 200)

    def test_response_shape(self):
        with patch.object(
            shape_map, "compute_index_shape_map", return_value=_fake_result(4)
        ) as mock_fn:
            resp = self.client.post("/similarity/index/tmcad/map?n_clusters=2&dims=3")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["index"], "tmcad")
        self.assertEqual(body["count"], 4)
        self.assertEqual(body["clusters"], 2)
        self.assertEqual(body["dims"], 3)
        self.assertEqual(len(body["points"]), 4)
        self.assertEqual(
            set(body["points"][0].keys()),
            {"file_id", "filename", "x", "y", "z", "cluster", "sim", "tags"},
        )
        mock_fn.assert_called_once_with(
            "tmcad", n_clusters=2, dims=3, tags=None, tag_mode="any", tagged=None,
            include_tags=False, color_by="cluster", kind="any", only_file_ids=None,
            outlier_threshold=0.0, projection="pca", clustering="kmeans",
            granularity=None, ward_max=256,
        )
        # Cohesion is carried through the response model for clusters only.
        by_kind = {row["kind"]: row for row in body["legend"]}
        self.assertAlmostEqual(by_kind["cluster"]["cohesion"], 0.94)
        self.assertIsNone(by_kind["outliers"]["cohesion"])

    def test_missing_index_returns_404(self):
        with patch.object(
            shape_map, "compute_index_shape_map", side_effect=KeyError("Index 'nope' does not exist.")
        ):
            resp = self.client.post("/similarity/index/nope/map")
        self.assertEqual(resp.status_code, 404)

    def test_dims_out_of_range_is_422(self):
        resp = self.client.post("/similarity/index/tmcad/map?dims=4")
        self.assertEqual(resp.status_code, 422)

    def test_below_min_clusters_is_422(self):
        # -1 is the valid "Optimal" sentinel; anything below it is out of range.
        resp = self.client.post("/similarity/index/tmcad/map?n_clusters=-2")
        self.assertEqual(resp.status_code, 422)


class ShapeMapDrillDownEndpointTests(unittest.TestCase):
    """The drill-down subset: request body, subset_id round trip and expiry."""

    def setUp(self):
        self.client = TestClient(_make_app())

    def test_body_only_file_ids_is_forwarded(self):
        with patch.object(
            shape_map, "compute_index_shape_map", return_value=_fake_result()
        ) as mock_fn:
            resp = self.client.post(
                "/similarity/index/tmcad/map",
                json={"only_file_ids": ["id_0", "id_1"]},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_fn.call_args.kwargs["only_file_ids"], ["id_0", "id_1"])

    def test_subset_id_is_returned(self):
        result = _fake_result()
        result["subset_id"] = "0123456789abcdef"
        with patch.object(shape_map, "compute_index_shape_map", return_value=result):
            resp = self.client.post("/similarity/index/tmcad/map")
        self.assertEqual(resp.json()["subset_id"], "0123456789abcdef")

    def test_subset_id_is_null_without_a_subset(self):
        with patch.object(shape_map, "compute_index_shape_map", return_value=_fake_result()):
            resp = self.client.post("/similarity/index/tmcad/map")
        self.assertIsNone(resp.json()["subset_id"])

    def test_too_small_subset_is_422(self):
        with patch.object(
            shape_map,
            "compute_index_shape_map",
            side_effect=ValueError("only_file_ids left 1 part; need at least 2."),
        ):
            resp = self.client.post(
                "/similarity/index/tmcad/map", json={"only_file_ids": ["id_0"]}
            )
        self.assertEqual(resp.status_code, 422)

    def test_unknown_scs_subset_is_409(self):
        # 409, not 404: 404 means "no such index". The client is meant to read
        # this as "re-post /map" rather than "this index is gone".
        with patch.object(shape_map, "_resolve_map_subset", return_value=None):
            resp = self.client.get("/similarity/index/tmcad/map/scs?subset=deadbeefdeadbeef")
        self.assertEqual(resp.status_code, 409)
        self.assertIn("subset", resp.json()["detail"].lower())

    def test_known_scs_subset_is_forwarded(self):
        with patch.object(shape_map, "_resolve_map_subset", return_value=["id_0", "id_1"]):
            with patch.object(
                shape_map, "compute_index_shape_map_scs", side_effect=KeyError("stop here")
            ) as mock_fn:
                self.client.get("/similarity/index/tmcad/map/scs?subset=0123456789abcdef")
        self.assertEqual(mock_fn.call_args.kwargs["only_file_ids"], ["id_0", "id_1"])


class ShapeMapOutlierEndpointTests(unittest.TestCase):
    """The outlier_threshold query parameter on both map routes."""

    def setUp(self):
        self.client = TestClient(_make_app())

    def test_map_forwards_the_threshold(self):
        with patch.object(
            shape_map, "compute_index_shape_map", return_value=_fake_result()
        ) as mock_fn:
            resp = self.client.post("/similarity/index/tmcad/map?outlier_threshold=0.8")
        self.assertEqual(resp.status_code, 200)
        self.assertAlmostEqual(mock_fn.call_args.kwargs["outlier_threshold"], 0.8)

    def test_map_defaults_to_disabled(self):
        with patch.object(
            shape_map, "compute_index_shape_map", return_value=_fake_result()
        ) as mock_fn:
            self.client.post("/similarity/index/tmcad/map")
        self.assertEqual(mock_fn.call_args.kwargs["outlier_threshold"], 0.0)

    def test_scs_forwards_the_threshold(self):
        with patch.object(
            shape_map, "compute_index_shape_map_scs", side_effect=KeyError("stop here")
        ) as mock_fn:
            self.client.get("/similarity/index/tmcad/map/scs?outlier_threshold=0.75")
        self.assertAlmostEqual(mock_fn.call_args.kwargs["outlier_threshold"], 0.75)

    def test_out_of_range_threshold_is_422(self):
        self.assertEqual(
            self.client.post("/similarity/index/tmcad/map?outlier_threshold=1.5").status_code,
            422,
        )
        self.assertEqual(
            self.client.post("/similarity/index/tmcad/map?outlier_threshold=-0.1").status_code,
            422,
        )


class ShapeMapWardMaxEndpointTests(unittest.TestCase):
    """The ward_max query parameter on both map routes.

    It has to reach the computation on every route that cuts the tree, because
    /map/tree's k_max is read against the same ceiling: the two disagreeing is
    what makes the tree's leaves stop being the map's clusters.
    """

    def setUp(self):
        self.client = TestClient(_make_app())

    def test_map_forwards_ward_max(self):
        with patch.object(
            shape_map, "compute_index_shape_map", return_value=_fake_result()
        ) as mock_fn:
            resp = self.client.post("/similarity/index/tmcad/map?ward_max=512")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_fn.call_args.kwargs["ward_max"], 512)

    def test_map_defaults_to_256(self):
        with patch.object(
            shape_map, "compute_index_shape_map", return_value=_fake_result()
        ) as mock_fn:
            self.client.post("/similarity/index/tmcad/map")
        self.assertEqual(mock_fn.call_args.kwargs["ward_max"], 256)

    def test_scs_forwards_ward_max(self):
        with patch.object(
            shape_map, "compute_index_shape_map_scs", side_effect=KeyError("stop here")
        ) as mock_fn:
            self.client.get("/similarity/index/tmcad/map/scs?ward_max=512")
        self.assertEqual(mock_fn.call_args.kwargs["ward_max"], 512)

    def test_scs_defaults_to_256(self):
        with patch.object(
            shape_map, "compute_index_shape_map_scs", side_effect=KeyError("stop here")
        ) as mock_fn:
            self.client.get("/similarity/index/tmcad/map/scs")
        self.assertEqual(mock_fn.call_args.kwargs["ward_max"], 256)

    def test_out_of_range_ward_max_is_422(self):
        self.assertEqual(
            self.client.post("/similarity/index/tmcad/map?ward_max=1").status_code,
            422,
        )
        self.assertEqual(
            self.client.post("/similarity/index/tmcad/map?ward_max=4097").status_code,
            422,
        )
        self.assertEqual(
            self.client.get(
                "/similarity/index/tmcad/map/scs?ward_max=4097"
            ).status_code,
            422,
        )

    def test_both_routes_accept_the_depths_the_tree_does(self):
        """One Max clusters value goes to all three routes, so a depth any one
        of them refuses is a depth the control cannot offer."""
        for ward_max in (2048, 4096):
            with patch.object(
                shape_map, "compute_index_shape_map", return_value=_fake_result()
            ) as mock_fn:
                resp = self.client.post(
                    f"/similarity/index/tmcad/map?ward_max={ward_max}"
                )
            self.assertEqual(resp.status_code, 200, ward_max)
            self.assertEqual(mock_fn.call_args.kwargs["ward_max"], ward_max)

            with patch.object(
                shape_map, "compute_index_shape_map_scs",
                side_effect=KeyError("stop here"),
            ) as mock_fn:
                self.client.get(
                    f"/similarity/index/tmcad/map/scs?ward_max={ward_max}"
                )
            self.assertEqual(mock_fn.call_args.kwargs["ward_max"], ward_max)


class ShapeMapTreeEndpointTests(unittest.TestCase):
    """POST /map/tree: wiring, validation and that no field is dropped."""

    def setUp(self):
        self.client = TestClient(_make_app())

    @staticmethod
    def _fake_tree():
        return {
            "index": "tmcad",
            "count": 3,
            "k_max": 8,
            "levels": [1, 2],
            "file_ids": ["id_0", "id_1", "id_2"],
            "leaf": [1, 1, 2],
            "nodes": [
                {"id": 0, "parent": -1, "level": 1, "count": 3, "height": 9.5,
                 "cohesion": 0.42, "centre": "id_0", "filename": "part_0.stp"},
                {"id": 1, "parent": 0, "level": 2, "count": 2, "height": 4.25,
                 "cohesion": 0.88, "centre": "id_1", "filename": "part_1.stp"},
                # A cluster of one has no mean similarity to report.
                {"id": 2, "parent": 0, "level": 2, "count": 1, "height": 4.25,
                 "cohesion": None, "centre": "id_2", "filename": "part_2.stp"},
            ],
        }

    def test_endpoint_is_registered(self):
        with patch.object(
            shape_map, "compute_index_shape_map_tree", return_value=self._fake_tree()
        ):
            resp = self.client.post("/similarity/index/tmcad/map/tree")
        self.assertEqual(resp.status_code, 200)

    def test_response_carries_the_whole_tree(self):
        with patch.object(
            shape_map, "compute_index_shape_map_tree", return_value=self._fake_tree()
        ) as mock_fn:
            resp = self.client.post("/similarity/index/tmcad/map/tree?k_max=8")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["index"], "tmcad")
        self.assertEqual(body["count"], 3)
        self.assertEqual(body["k_max"], 8)
        self.assertEqual(body["levels"], [1, 2])
        self.assertEqual(body["file_ids"], ["id_0", "id_1", "id_2"])
        self.assertEqual(body["leaf"], [1, 1, 2])
        self.assertEqual(len(body["nodes"]), 3)
        self.assertEqual(
            set(body["nodes"][0].keys()),
            {"id", "parent", "level", "count", "height", "cohesion", "centre",
             "filename"},
        )
        self.assertAlmostEqual(body["nodes"][1]["cohesion"], 0.88)
        self.assertIsNone(body["nodes"][2]["cohesion"])
        mock_fn.assert_called_once_with(
            "tmcad", tags=None, tag_mode="any", tagged=None, kind="any",
            only_file_ids=None, k_max=8,
        )

    def test_k_max_out_of_range_is_422(self):
        self.assertEqual(
            self.client.post("/similarity/index/tmcad/map/tree?k_max=1").status_code,
            422,
        )
        self.assertEqual(
            self.client.post("/similarity/index/tmcad/map/tree?k_max=4097").status_code,
            422,
        )

    def test_k_max_accepts_the_depths_the_catalogue_offers(self):
        """The two must take the same range, or a catalogue's ids mean nothing
        to the tree it is shown against."""
        for k_max in (2048, 4096):
            with patch.object(
                shape_map, "compute_index_shape_map_tree",
                return_value=self._fake_tree(),
            ):
                resp = self.client.post(
                    f"/similarity/index/tmcad/map/tree?k_max={k_max}"
                )
            self.assertEqual(resp.status_code, 200, k_max)

    def test_missing_index_returns_404(self):
        with patch.object(
            shape_map,
            "compute_index_shape_map_tree",
            side_effect=KeyError("Index 'nope' does not exist."),
        ):
            resp = self.client.post("/similarity/index/nope/map/tree")
        self.assertEqual(resp.status_code, 404)

    def test_body_only_file_ids_is_forwarded(self):
        with patch.object(
            shape_map, "compute_index_shape_map_tree", return_value=self._fake_tree()
        ) as mock_fn:
            resp = self.client.post(
                "/similarity/index/tmcad/map/tree",
                json={"only_file_ids": ["id_0", "id_1"]},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_fn.call_args.kwargs["only_file_ids"], ["id_0", "id_1"])

    def test_subset_that_is_not_a_node_is_400(self):
        with patch.object(
            shape_map,
            "compute_index_shape_map_tree",
            side_effect=ValueError("This subset is not a node of the tree."),
        ):
            resp = self.client.post(
                "/similarity/index/tmcad/map/tree",
                json={"only_file_ids": ["id_0", "id_9"]},
            )
        self.assertEqual(resp.status_code, 400)


class ShapeMapLegendOrderEndpointTests(unittest.TestCase):
    """POST /map/legend-order: wiring, and the two refusals."""

    def setUp(self):
        self.client = TestClient(_make_app())

    @staticmethod
    def _fake_order():
        return {"index": "tmcad", "k": 3, "method": "ward+olo", "order": [2, 0, 1]}

    def test_endpoint_is_registered_and_returns_the_order(self):
        with patch.object(
            shape_map, "compute_index_legend_order", return_value=self._fake_order()
        ) as mock_fn:
            resp = self.client.post("/similarity/index/tmcad/map/legend-order")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["order"], [2, 0, 1])
        self.assertEqual(body["method"], "ward+olo")
        self.assertEqual(body["k"], 3)
        mock_fn.assert_called_once_with(
            "tmcad", n_clusters=0, kind="any", tags=None, tag_mode="any",
            tagged=None, only_file_ids=None, outlier_threshold=0.0,
            clustering="kmeans",
        )

    def test_the_map_parameters_are_forwarded(self):
        with patch.object(
            shape_map, "compute_index_legend_order", return_value=self._fake_order()
        ) as mock_fn:
            resp = self.client.post(
                "/similarity/index/tmcad/map/legend-order"
                "?n_clusters=64&kind=part&outlier_threshold=0.3",
                json={"only_file_ids": ["id_0", "id_1"]},
            )
        self.assertEqual(resp.status_code, 200)
        kwargs = mock_fn.call_args.kwargs
        self.assertEqual(kwargs["n_clusters"], 64)
        self.assertEqual(kwargs["kind"], "part")
        self.assertAlmostEqual(kwargs["outlier_threshold"], 0.3)
        self.assertEqual(kwargs["only_file_ids"], ["id_0", "id_1"])

    def test_ward_is_400(self):
        """Ward's clusters are a cut of a tree that already orders them."""
        with patch.object(
            shape_map,
            "compute_index_legend_order",
            side_effect=ValueError("legend order is for clustering='kmeans' only"),
        ):
            resp = self.client.post(
                "/similarity/index/tmcad/map/legend-order?clustering=ward"
            )
        self.assertEqual(resp.status_code, 400)

    def test_missing_index_returns_404(self):
        with patch.object(
            shape_map,
            "compute_index_legend_order",
            side_effect=KeyError("Index 'nope' does not exist."),
        ):
            resp = self.client.post("/similarity/index/nope/map/legend-order")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
