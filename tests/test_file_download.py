"""SDK-free unit tests for GET /files/{file_id}/download.

The server does not render CAD; the gallery hands a file_id to this endpoint
and the client renders the returned bytes locally. These tests cover the
happy path (original filename + content) and the not-found case. No HOOPS AI
license is required.
"""

import hashlib
import io
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core
import storage
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _Upload:
    def __init__(self, data: bytes, filename: str):
        self.filename = filename
        self.file = io.BytesIO(data)


def _make_app():
    from routers import files

    app = FastAPI()
    app.include_router(files.router)
    return app


class FileDownloadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = pathlib.Path(self._tmp.name)
        self._orig = storage.get_CAD_shared_dir
        storage.get_CAD_shared_dir = lambda: self.store
        self.client = TestClient(_make_app())

    def tearDown(self):
        storage.get_CAD_shared_dir = self._orig
        self._tmp.cleanup()

    def test_download_returns_original_bytes_and_filename(self):
        data = b"solid body" * 100
        fid, _, _ = storage.upload_CAD_file_persistent(_Upload(data, "widget.stp"))

        response = self.client.get(f"/files/{fid}/download")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, data)
        self.assertIn("widget.stp", response.headers.get("content-disposition", ""))

    def test_download_unknown_file_id_returns_404(self):
        missing = hashlib.sha256(b"never uploaded").hexdigest()
        response = self.client.get(f"/files/{missing}/download")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
