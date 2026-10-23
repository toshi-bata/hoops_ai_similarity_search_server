"""Regression guard: named-index management and the model-consistency
endpoint are part of the redistributable product and must be
served without any feature flag.

They used to sit behind a demo-only gate that returned 404 when unset; that gate
has been removed. These tests fail if the gate is ever reintroduced.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.testclient import TestClient

import core
import index_registry


def _make_app():
    from routers import similarity

    app = FastAPI()
    app.include_router(similarity.router)
    return app


class PublicEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(_make_app())

    def test_index_list_is_public(self):
        with patch.object(index_registry, "list_indexes", return_value=[]):
            response = self.client.get("/similarity/index/list")
        self.assertEqual(response.status_code, 200)

    def test_model_consistency_endpoint_is_registered(self):
        # The digest-consistency route must be wired up. Assert on the OpenAPI
        # schema so the check does not depend on an index existing on disk.
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/similarity/index/{name}/model-consistency", schema["paths"])


if __name__ == "__main__":
    unittest.main()
