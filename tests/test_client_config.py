"""Tests for /client-config, the one server setting the web client reads.

The endpoint exists so the client can hide the webinar-demo-only Estimate panel
unless HOOPS_AI_DEMO says otherwise. Both states are pinned here: a fresh clone
must get ``demo: false`` without setting anything, and a deployment that asks
for the demo must get ``demo: true``.

``config.read_env_file`` is stubbed out in every case so the developer's own
.env cannot decide the outcome.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.testclient import TestClient

import config


def _make_app():
    from routers import client_config

    app = FastAPI()
    app.include_router(client_config.router)
    return app


class ClientConfigTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(_make_app())

    def _get(self, env_value):
        environ = {k: v for k, v in os.environ.items() if k != "HOOPS_AI_DEMO"}
        if env_value is not None:
            environ["HOOPS_AI_DEMO"] = env_value
        with patch.dict(os.environ, environ, clear=True), patch.object(
            config, "read_env_file", return_value={}
        ):
            return self.client.get("/client-config")

    def test_demo_is_false_by_default(self):
        response = self._get(None)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["demo"], False)

    def test_demo_is_true_when_enabled(self):
        response = self._get("true")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["demo"], True)

    def test_umap_availability_is_reported(self):
        # A yes/no about an optional Python package, so the client can leave
        # the UMAP projection out of its menu instead of offering a choice
        # that would answer 503.
        import core

        body = self._get(None).json()
        self.assertEqual(
            body["umap_available"], core._shape_map_umap_available()
        )

    def test_response_carries_nothing_but_client_settings(self):
        # Guard the module's rule: no paths, no key, no server internals.
        self.assertEqual(
            set(self._get(None).json()),
            {
                "demo",
                "umap_available",
                "warm_clusterings",
                "default_clustering",
                "default_ward_max",
            },
        )


class DefaultClusteringTests(unittest.TestCase):
    """Which clusterings the client's Shape Map may open with.

    Opening the map on a clustering startup warmup did not build means the
    first user pays the full build, so the server names the warm ones. The
    environment is pinned in every case (the developer's own .env sets these
    variables on this machine).
    """

    def setUp(self):
        self.client = TestClient(_make_app())

    def _body(self, env):
        base = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("HOOPS_AI_WARMUP_")
        }
        base.update(env)
        with patch.dict(os.environ, base, clear=True), patch.object(
            config, "read_env_file", return_value={}
        ):
            return self.client.get("/client-config").json()

    def _get(self, env):
        return self._body(env)["default_clustering"]

    def _warm(self, env):
        return self._body(env)["warm_clusterings"]

    def test_nothing_is_warm_when_warmup_is_not_configured(self):
        self.assertEqual(self._warm({}), [])
        self.assertIsNone(self._get({}))

    def test_both_clusterings_are_warm_when_none_is_named(self):
        # Warmup runs on its own default, which is now BOTH clusterings: each
        # is a separate cache key, so warming only one leaves whoever switches
        # waiting for a full build. k-means leads, so a fresh browser with
        # nothing remembered opens on it.
        env = {"HOOPS_AI_WARMUP_INDEXES": "tmcad"}
        self.assertEqual(self._warm(env), ["kmeans", "ward"])
        self.assertEqual(self._get(env), "kmeans")

    def test_ward_is_reported(self):
        env = {
            "HOOPS_AI_WARMUP_INDEXES": "tmcad",
            "HOOPS_AI_WARMUP_CLUSTERING": "ward",
        }
        self.assertEqual(self._warm(env), ["ward"])
        self.assertEqual(self._get(env), "ward")

    def test_every_warmed_clustering_is_listed_in_order(self):
        env = {
            "HOOPS_AI_WARMUP_INDEXES": "tmcad",
            "HOOPS_AI_WARMUP_CLUSTERING": "ward,kmeans",
        }
        self.assertEqual(self._warm(env), ["ward", "kmeans"])
        # The compatibility field leads with the first entry.
        self.assertEqual(self._get(env), "ward")

    def test_first_entry_wins_when_several_are_warmed(self):
        env = {
            "HOOPS_AI_WARMUP_INDEXES": "tmcad",
            "HOOPS_AI_WARMUP_CLUSTERING": "kmeans,ward",
        }
        self.assertEqual(self._get(env), "kmeans")

    def test_nothing_is_warm_when_warmup_is_off(self):
        # Nothing is built at startup, so there is no warm clustering to lead
        # the user to.
        env = {"HOOPS_AI_WARMUP_CLUSTERING": "ward"}
        self.assertEqual(self._warm(env), [])
        self.assertIsNone(self._get(env))

    def test_nothing_is_warm_when_the_map_stage_is_not_warmed(self):
        # The searcher is warmed but no map is built, so every clustering is
        # cold. 'searcher' rather than a made-up stage name: an unrecognised
        # list falls back to all stages, which would warm the map after all.
        env = {
            "HOOPS_AI_WARMUP_INDEXES": "tmcad",
            "HOOPS_AI_WARMUP_CLUSTERING": "ward",
            "HOOPS_AI_WARMUP_STAGES": "searcher",
        }
        self.assertEqual(self._warm(env), [])
        self.assertIsNone(self._get(env))


class DefaultWardMaxTests(unittest.TestCase):
    """The Max clusters the client's Shape Map opens with.

    Null unless an operator fixed one for every index: without it the client
    uses the per-index default_clusters from /index/{name}/stats, which is the
    rule warmup builds with. The environment is pinned in every case, like
    DefaultClusteringTests.
    """

    def setUp(self):
        self.client = TestClient(_make_app())

    def _get(self, env):
        base = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("HOOPS_AI_WARMUP_")
        }
        base.update(env)
        with patch.dict(os.environ, base, clear=True), patch.object(
            config, "read_env_file", return_value={}
        ):
            return self.client.get("/client-config").json()["default_ward_max"]

    def test_unset_means_the_per_index_default(self):
        self.assertIsNone(self._get({}))

    def test_configured_value_is_reported(self):
        self.assertEqual(self._get({"HOOPS_AI_WARMUP_WARD_MAX": "512"}), 512)

    def test_reported_even_when_warmup_is_off(self):
        # Unlike the clustering, this is the UI's own ceiling as well as the
        # warmed one, so it is answered whether or not warmup runs.
        env = {"HOOPS_AI_WARMUP_WARD_MAX": "128", "HOOPS_AI_WARMUP_INDEXES": ""}
        self.assertEqual(self._get(env), 128)

    def test_out_of_range_means_the_per_index_default(self):
        # 2..1024 is what POST /map accepts, so a larger value would be
        # rejected by the very request this ceiling is meant to warm. A ceiling
        # the server would refuse is no ceiling at all.
        self.assertIsNone(self._get({"HOOPS_AI_WARMUP_WARD_MAX": "4096"}))

    def test_the_setting_itself_is_what_is_reported(self):
        # The field is warmup_ward_max_setting() verbatim: one rule, read by
        # warmup and handed to the client, so the two cannot drift.
        for env, expected in (
            ({}, None),
            ({"HOOPS_AI_WARMUP_WARD_MAX": "256"}, 256),
            ({"HOOPS_AI_WARMUP_WARD_MAX": "1"}, None),
            ({"HOOPS_AI_WARMUP_WARD_MAX": "nonsense"}, None),
        ):
            with self.subTest(env=env):
                base = {
                    k: v
                    for k, v in os.environ.items()
                    if not k.startswith("HOOPS_AI_WARMUP_")
                }
                base.update(env)
                with patch.dict(os.environ, base, clear=True), patch.object(
                    config, "read_env_file", return_value={}
                ):
                    self.assertEqual(config.warmup_ward_max_setting(), expected)
                self.assertEqual(self._get(env), expected)


class ExistsChunkLimitTests(unittest.TestCase):
    """The client's exists batch size must stay within the server's limit.

    POST /files/exists rejects more ids than ``storage.exists_max_ids()`` with
    422, and web/src/api.js splits its request into EXISTS_CHUNK-sized batches
    to stay under it. The two numbers live in different languages in different
    files, so raising one without the other breaks nothing visible: the client
    keeps working on small folders and returns 422 only once someone registers
    a folder larger than one batch. This test reads the JS source -- it does not
    run it -- and pins the relationship.
    """

    def test_exists_chunk_is_within_the_server_limit(self):
        import pathlib
        import re

        import storage

        source = (
            pathlib.Path(__file__).resolve().parent.parent
            / "web"
            / "src"
            / "api.js"
        ).read_text(encoding="utf-8")
        match = re.search(r"^const EXISTS_CHUNK\s*=\s*(\d+)\s*;", source, re.M)
        self.assertIsNotNone(match, "EXISTS_CHUNK not found in web/src/api.js")
        chunk = int(match.group(1))
        self.assertGreater(chunk, 0)
        self.assertLessEqual(chunk, storage._EXISTS_MAX_IDS_DEFAULT)


if __name__ == "__main__":
    unittest.main()
