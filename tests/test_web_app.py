"""Serving guarantees for the built web client at /app.

Two things are easy to get wrong and only show up in the redistributable
package (not in a dev environment), so they are pinned here:

* /app/ serves the client's index.html.
* .wasm files are served with the application/wasm content type; browsers
  refuse to instantiate WebAssembly delivered with any other type.

The tests create their own probe files under static/app and remove them again,
so they pass whether or not a real build output is present.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

import main

_APP_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "app")


class WebAppServingTests(unittest.TestCase):
    def setUp(self):
        os.makedirs(_APP_DIR, exist_ok=True)
        self.client = TestClient(main.app)
        self._created = []

        # Provide an index.html only if a real build has not produced one, so
        # we never clobber committed build output.
        self._index_path = os.path.join(_APP_DIR, "index.html")
        if not os.path.exists(self._index_path):
            with open(self._index_path, "w", encoding="utf-8") as fh:
                fh.write("<!doctype html><title>probe</title>")
            self._created.append(self._index_path)

        # A probe .wasm file to check the served content type.
        self._wasm_path = os.path.join(_APP_DIR, "__mime_probe__.wasm")
        with open(self._wasm_path, "wb") as fh:
            fh.write(b"\x00asm\x01\x00\x00\x00")
        self._created.append(self._wasm_path)

    def tearDown(self):
        for path in self._created:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_app_root_serves_index_html(self):
        response = self.client.get("/app/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_wasm_is_served_as_application_wasm(self):
        response = self.client.get("/app/__mime_probe__.wasm")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["content-type"].split(";")[0].strip(),
            "application/wasm",
        )


if __name__ == "__main__":
    unittest.main()
