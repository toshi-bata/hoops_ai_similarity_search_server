"""Tests for reporting one geometry registered under several filenames.

file_ids are content hashes, so byte-identical CAD files become a single index
entry however many times they are uploaded. What survives the collapse is the
original filename each copy arrived with, which the store keeps as a separate
``<sha256>_<originalname>`` file and which nothing used to report. These cover
the three places that now do: the storage scan, GET .../duplicates, and the
``[DUPLICATES]`` section of the registration report.

SDK-free: the CAD store is a tmp directory of empty files, so no license, no
faiss and no real index is involved. The server's own data/ directory is never
read.
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

import core  # noqa: E402
import index_registry  # noqa: E402
import jobs  # noqa: E402
import storage  # noqa: E402

# Config helpers on the import path read the environment, and a developer's
# .env must not decide whether a test passes.
_ENV = {
    "HOOPS_AI_DATA_DIR": "C:\\tmp\\data",
    "HOOPS_LICENSE": "x",
}

# Three distinct content hashes. Real ids are 64 lower-case hex characters and
# the scan slices on that width, so the fakes have to be the real shape.
_ID_A = "a" * 64
_ID_B = "b" * 64
_ID_C = "c" * 64


def _make_app():
    from routers.similarity import router

    app = FastAPI()
    app.include_router(router)
    return app


class _StoreTestCase(unittest.TestCase):
    """A tmp CAD store holding two names for _ID_A and one each for B and C."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = pathlib.Path(self._tmp.name)
        for name in (
            f"{_ID_A}_bracket.stp",
            f"{_ID_A}_bracket_rev_b.stp",
            f"{_ID_B}_housing.stp",
            f"{_ID_C}_shaft.stp",
        ):
            (self.store / name).write_bytes(b"")
        # A directory and a stray file must both be ignored by the scan.
        (self.store / f"{_ID_B}_adirectory").mkdir()
        (self.store / "not-a-stored-file.txt").write_bytes(b"")
        patcher = patch.object(storage, "get_CAD_shared_dir", return_value=self.store)
        patcher.start()
        self.addCleanup(patcher.stop)


class MapDuplicateFilenamesTests(_StoreTestCase):
    """storage.map_duplicate_CAD_filenames over a faked store."""

    def test_only_ids_with_several_names_come_back(self):
        result = storage.map_duplicate_CAD_filenames([_ID_A, _ID_B, _ID_C])
        self.assertEqual(
            result, {_ID_A: ["bracket.stp", "bracket_rev_b.stp"]}
        )

    def test_the_sha256_prefix_is_stripped_and_no_path_is_returned(self):
        names = storage.map_duplicate_CAD_filenames([_ID_A])[_ID_A]
        for n in names:
            self.assertNotIn(_ID_A, n)
            self.assertNotIn(os.sep, n)

    def test_ids_outside_the_request_are_not_scanned_in(self):
        self.assertEqual(storage.map_duplicate_CAD_filenames([_ID_B, _ID_C]), {})

    def test_an_empty_request_reads_nothing(self):
        self.assertEqual(storage.map_duplicate_CAD_filenames([]), {})

    def test_the_single_path_resolver_still_answers_once_per_id(self):
        # The companion scan must keep its one-path-per-id contract: other
        # callers (the embedding pass) depend on it.
        paths = storage.map_persistent_CAD_files([_ID_A, _ID_B])
        self.assertEqual(sorted(paths), sorted([_ID_A, _ID_B]))
        self.assertIsInstance(paths[_ID_A], pathlib.Path)


class IndexDuplicateNamesTests(_StoreTestCase):
    """index_registry.get_index_duplicate_names over a faked index."""

    def _duplicates(self, ids):
        with patch.object(
            index_registry, "_require_index_exists", return_value=None
        ), patch.object(
            index_registry, "_registered_index_ids", return_value=set(ids)
        ):
            return index_registry.get_index_duplicate_names("tmcad")

    def test_counts_shapes_and_redundant_copies(self):
        result = self._duplicates([_ID_A, _ID_B, _ID_C])
        self.assertEqual(result["name"], "tmcad")
        self.assertEqual(result["total"], 1)
        # Two names for one geometry is one redundant copy.
        self.assertEqual(result["extra_copies"], 1)
        self.assertEqual(
            result["items"],
            [{"file_id": _ID_A, "names": ["bracket.stp", "bracket_rev_b.stp"]}],
        )

    def test_an_index_without_duplicates_reports_zero(self):
        result = self._duplicates([_ID_B, _ID_C])
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["extra_copies"], 0)
        self.assertEqual(result["items"], [])

    def test_a_missing_index_is_a_key_error(self):
        with patch.object(
            index_registry, "_index_faiss_path",
            return_value=self.store / "nope.faiss",
        ):
            with self.assertRaises(KeyError):
                index_registry.get_index_duplicate_names("tmcad")

    def test_core_reexports_it(self):
        self.assertIs(core.get_index_duplicate_names,
                      index_registry.get_index_duplicate_names)


class DuplicatesEndpointTests(unittest.TestCase):
    """GET /similarity/index/{name}/duplicates."""

    def setUp(self):
        patcher = patch.dict(os.environ, _ENV, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(_make_app())
        self.payload = {
            "name": "tmcad",
            "total": 2,
            "extra_copies": 3,
            "items": [
                {"file_id": _ID_A, "names": ["bracket.stp", "bracket_rev_b.stp"]},
                {"file_id": _ID_B, "names": ["h.stp", "h2.stp", "h3.stp"]},
            ],
        }

    def _get(self, query="", payload=None):
        with patch.object(
            core, "get_index_duplicate_names",
            return_value=payload if payload is not None else self.payload,
        ):
            return self.client.get(f"/similarity/index/tmcad/duplicates{query}")

    def test_returns_the_counts_and_the_items(self):
        resp = self._get()
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["total"], 2)
        self.assertEqual(body["extra_copies"], 3)
        self.assertEqual(len(body["items"]), 2)
        self.assertEqual(body["items"][0]["names"],
                         ["bracket.stp", "bracket_rev_b.stp"])

    def test_the_counts_cover_the_index_not_the_page(self):
        # A page of one must still report both geometries and all three
        # redundant copies, or the headline figure would shrink with the page.
        body = self._get("?limit=1").json()
        self.assertEqual(len(body["items"]), 1)
        self.assertEqual(body["total"], 2)
        self.assertEqual(body["extra_copies"], 3)
        self.assertEqual(body["limit"], 1)

    def test_offset_skips_items(self):
        body = self._get("?offset=1").json()
        self.assertEqual([it["file_id"] for it in body["items"]], [_ID_B])
        self.assertEqual(body["offset"], 1)

    def test_no_server_path_is_exposed(self):
        # The names must be bare filenames: the layout of the CAD store is not
        # the client's business.
        body = self._get().json()
        for item in body["items"]:
            for n in item["names"]:
                self.assertNotIn("/", n)
                self.assertNotIn("\\", n)
        self.assertNotIn("uploads", self._get().text)

    def test_a_missing_index_is_404(self):
        with patch.object(
            core, "get_index_duplicate_names", side_effect=KeyError("Index 'x'")
        ):
            resp = self.client.get("/similarity/index/tmcad/duplicates")
        self.assertEqual(resp.status_code, 404)

    def test_an_invalid_name_is_rejected_before_any_scan(self):
        with patch.object(core, "get_index_duplicate_names") as fn:
            resp = self.client.get("/similarity/index/Bad%20Name/duplicates")
        self.assertEqual(resp.status_code, 422)
        fn.assert_not_called()


class ReportDuplicatesSectionTests(unittest.TestCase):
    """The [DUPLICATES] section of the registration report."""

    def _report(self, duplicate_names):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "report.txt"
            jobs._write_add_job_report(
                path, "tmcad", ["a.stp"], [], [], {},
                duplicate_names,
            )
            return path.read_text(encoding="utf-8")

    def test_the_section_lists_every_name_of_each_geometry(self):
        text = self._report({_ID_A: ["bracket.stp", "bracket_rev_b.stp"]})
        self.assertIn("[DUPLICATES]", text)
        self.assertIn(f"{_ID_A}\tbracket.stp, bracket_rev_b.stp", text)

    def test_the_header_counts_the_geometries(self):
        text = self._report({
            _ID_A: ["a.stp", "a2.stp"],
            _ID_B: ["b.stp", "b2.stp", "b3.stp"],
        })
        self.assertIn("duplicate_names=2", text)
        self.assertIn("2 geometries, 3 redundant copy/copies", text)

    def test_the_section_is_present_even_when_empty(self):
        # Absence of the section would be indistinguishable from an older
        # report, so it is always written.
        text = self._report({})
        self.assertIn("[DUPLICATES]", text)
        self.assertIn("duplicate_names=0", text)

    def test_the_section_says_the_index_holds_one_entry_each(self):
        # The wording is the point: a reader must not conclude the index has
        # duplicate entries.
        text = self._report({_ID_A: ["a.stp", "a2.stp"]})
        self.assertIn("ONE entry", text)

    def test_the_argument_is_optional(self):
        # Callers that have nothing to say must still be able to write one.
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "report.txt"
            jobs._write_add_job_report(path, "tmcad", [], [], [], {})
            self.assertIn("[DUPLICATES]", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
