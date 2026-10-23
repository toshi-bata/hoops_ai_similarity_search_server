"""SDK-free unit tests for the bulk upload path (PR A of Phase 3).

Covers:
  * streaming upload: chunked write, single-pass hashing, temp-file rename,
    per-file size cap
  * POST /files/upload-batch (multi-file) and the unchanged POST /files/upload
  * POST /files/exists (known / unknown / invalid, dedup, cap)
  * ZIP limits driven by environment variables
  * the extended CAD extension list

No HOOPS AI license is required: nothing here touches the SDK.
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
from fastapi.testclient import TestClient


class _Upload:
    """Minimal duck-typed UploadFile whose stream is read in chunks."""

    def __init__(self, data: bytes, filename: str):
        self.filename = filename
        self.file = io.BytesIO(data)


class _StoreBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = pathlib.Path(self._tmp.name)
        self._orig = storage.get_CAD_shared_dir
        storage.get_CAD_shared_dir = lambda: self.store
        self._env_keys = (
            "HOOPS_AI_MAX_UPLOAD_BYTES",
            "HOOPS_AI_EXISTS_MAX_IDS",
            "HOOPS_AI_ZIP_MAX_TOTAL_BYTES",
            "HOOPS_AI_ZIP_MAX_FILES",
        )
        self._saved_env = {k: os.environ.get(k) for k in self._env_keys}

    def tearDown(self):
        storage.get_CAD_shared_dir = self._orig
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()


class TestStreamingUpload(_StoreBase):
    def test_file_id_is_the_sha256_of_the_content(self):
        data = b"solid body" * 1000
        fid, path, existed = storage.upload_CAD_file_persistent(_Upload(data, "a.stp"))
        self.assertEqual(fid, hashlib.sha256(data).hexdigest())
        self.assertFalse(existed)
        self.assertEqual(path.read_bytes(), data)

    def test_content_larger_than_one_chunk_hashes_correctly(self):
        data = os.urandom(core._UPLOAD_CHUNK_BYTES * 2 + 12345)
        fid, path, _ = storage.upload_CAD_file_persistent(_Upload(data, "big.stp"))
        self.assertEqual(fid, hashlib.sha256(data).hexdigest())
        self.assertEqual(path.stat().st_size, len(data))

    def test_second_upload_is_idempotent(self):
        data = b"same"
        first, path_a, existed_a = storage.upload_CAD_file_persistent(_Upload(data, "a.stp"))
        second, path_b, existed_b = storage.upload_CAD_file_persistent(_Upload(data, "a.stp"))
        self.assertEqual(first, second)
        self.assertEqual(path_a, path_b)
        self.assertFalse(existed_a)
        self.assertTrue(existed_b)

    def test_no_temporary_files_are_left_behind(self):
        storage.upload_CAD_file_persistent(_Upload(b"x", "a.stp"))
        storage.upload_CAD_file_persistent(_Upload(b"x", "a.stp"))  # duplicate path
        self.assertEqual([p.name for p in self.store.glob("*.part")], [])

    def test_oversized_upload_raises_and_leaves_nothing(self):
        os.environ["HOOPS_AI_MAX_UPLOAD_BYTES"] = "16"
        with self.assertRaises(core.UploadSizeLimitError):
            storage.upload_CAD_file_persistent(_Upload(b"y" * 64, "big.stp"))
        self.assertEqual(list(self.store.iterdir()), [])

    def test_size_cap_is_read_per_call(self):
        os.environ["HOOPS_AI_MAX_UPLOAD_BYTES"] = "16"
        with self.assertRaises(core.UploadSizeLimitError):
            storage.upload_CAD_file_persistent(_Upload(b"y" * 64, "big.stp"))
        os.environ["HOOPS_AI_MAX_UPLOAD_BYTES"] = "1024"
        fid, _, _ = storage.upload_CAD_file_persistent(_Upload(b"y" * 64, "big.stp"))
        self.assertTrue(fid)

    def test_missing_filename_raises(self):
        with self.assertRaises(RuntimeError):
            storage.upload_CAD_file_persistent(_Upload(b"x", ""))

    def test_directory_components_are_stripped_from_the_name(self):
        _, path, _ = storage.upload_CAD_file_persistent(_Upload(b"x", "../../evil.stp"))
        self.assertEqual(path.parent, self.store)
        self.assertTrue(path.name.endswith("_evil.stp"))


class TestExistsCheck(_StoreBase):
    def setUp(self):
        super().setUp()
        self.known_id, _, _ = storage.upload_CAD_file_persistent(_Upload(b"known", "k.stp"))
        self.missing_id = "b" * 64

    def test_known_and_unknown_are_split(self):
        out = core.check_persistent_CAD_files([self.known_id, self.missing_id])
        self.assertEqual(out["known"], [self.known_id])
        self.assertEqual(out["unknown"], [self.missing_id])
        self.assertEqual(out["invalid"], [])

    def test_malformed_ids_are_reported_as_invalid(self):
        out = core.check_persistent_CAD_files(["short", "z" * 64, "A" * 64, ""])
        self.assertEqual(out["unknown"], [])
        self.assertEqual(len(out["invalid"]), 4)

    def test_duplicates_are_collapsed(self):
        out = core.check_persistent_CAD_files([self.known_id, self.known_id])
        self.assertEqual(out["known"], [self.known_id])

    def test_input_order_is_preserved(self):
        other = "c" * 64
        out = core.check_persistent_CAD_files([other, self.missing_id])
        self.assertEqual(out["unknown"], [other, self.missing_id])

    def test_empty_store_reports_everything_unknown(self):
        core.delete_persistent_CAD_file(self.known_id)
        out = core.check_persistent_CAD_files([self.known_id])
        self.assertEqual(out["unknown"], [self.known_id])

    def test_partial_files_are_not_treated_as_known(self):
        (self.store / ".upload_deadbeef.part").write_bytes(b"x")
        self.assertEqual(core.known_persistent_CAD_file_ids(), {self.known_id})


class TestEnvLimits(_StoreBase):
    def test_zip_defaults(self):
        self.assertEqual(core.zip_max_total_bytes(), core.ZIP_MAX_TOTAL_BYTES)
        self.assertEqual(core.zip_max_files(), core.ZIP_MAX_FILES)

    def test_zip_limits_are_overridable(self):
        os.environ["HOOPS_AI_ZIP_MAX_FILES"] = "500"
        os.environ["HOOPS_AI_ZIP_MAX_TOTAL_BYTES"] = str(4 * 1024 * 1024 * 1024)
        self.assertEqual(core.zip_max_files(), 500)
        self.assertEqual(core.zip_max_total_bytes(), 4 * 1024 * 1024 * 1024)

    def test_invalid_values_fall_back_to_defaults(self):
        os.environ["HOOPS_AI_ZIP_MAX_FILES"] = "not-a-number"
        self.assertEqual(core.zip_max_files(), core.ZIP_MAX_FILES)

    def test_out_of_range_values_fall_back_to_defaults(self):
        os.environ["HOOPS_AI_ZIP_MAX_FILES"] = "0"
        self.assertEqual(core.zip_max_files(), core.ZIP_MAX_FILES)

    def test_exists_cap_default_and_override(self):
        self.assertEqual(core.exists_max_ids(), 1000)
        os.environ["HOOPS_AI_EXISTS_MAX_IDS"] = "5"
        self.assertEqual(core.exists_max_ids(), 5)


class TestExtensionList(unittest.TestCase):
    """The list must cover what hoops_ai_qt_sandbox's cadNameFilters() collects."""

    def test_previously_supported_extensions_are_retained(self):
        for ext in (".step", ".stp", ".iges", ".igs", ".x_t", ".x_b",
                    ".sat", ".ipt", ".prt", ".sldprt", ".catpart"):
            self.assertIn(ext, core.CAD_ALLOWED_EXTENSIONS)

    def test_qt_sandbox_formats_are_covered(self):
        for ext in (".catproduct", ".sldasm", ".iam", ".asm", ".jt", ".prc",
                    ".3dm", ".ifc", ".stl", ".obj", ".hsf", ".xmt_txt"):
            self.assertIn(ext, core.CAD_ALLOWED_EXTENSIONS)

    def test_all_entries_are_lower_case_and_dotted(self):
        for ext in core.CAD_ALLOWED_EXTENSIONS:
            self.assertTrue(ext.startswith("."))
            self.assertEqual(ext, ext.lower())

    def test_archive_extension_is_not_treated_as_cad(self):
        self.assertNotIn(".zip", core.CAD_ALLOWED_EXTENSIONS)


class TestUploadEndpoints(_StoreBase):
    def setUp(self):
        super().setUp()
        import main
        self.client = TestClient(main.app)

    def test_single_upload_response_shape_is_unchanged(self):
        resp = self.client.post("/files/upload", files={"file": ("a.stp", b"data")})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(set(resp.json()), {"file_id", "filename", "already_existed"})

    def test_batch_upload_returns_one_entry_per_file(self):
        resp = self.client.post(
            "/files/upload-batch",
            files=[("files", ("a.stp", b"aaa")), ("files", ("b.stp", b"bbb"))],
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["count"], 2)
        self.assertEqual(len(body["files"]), 2)
        self.assertEqual(body["errors"], [])

    def test_batch_upload_accounts_for_every_input(self):
        os.environ["HOOPS_AI_MAX_UPLOAD_BYTES"] = "4"
        resp = self.client.post(
            "/files/upload-batch",
            files=[("files", ("ok.stp", b"ab")), ("files", ("big.stp", b"abcdefgh"))],
        )
        body = resp.json()
        self.assertEqual(len(body["files"]) + len(body["errors"]), 2)
        self.assertEqual(body["errors"][0]["filename"], "big.stp")

    def test_oversized_single_upload_is_413(self):
        os.environ["HOOPS_AI_MAX_UPLOAD_BYTES"] = "4"
        resp = self.client.post("/files/upload", files={"file": ("a.stp", b"abcdefgh")})
        self.assertEqual(resp.status_code, 413)

    def test_exists_endpoint_splits_ids(self):
        fid = self.client.post(
            "/files/upload", files={"file": ("a.stp", b"data")}
        ).json()["file_id"]
        resp = self.client.post(
            "/files/exists", json={"file_ids": [fid, "d" * 64, "nope"]}
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["known"], [fid])
        self.assertEqual(body["unknown"], ["d" * 64])
        self.assertEqual(body["invalid"], ["nope"])

    def test_exists_endpoint_rejects_oversized_batches(self):
        os.environ["HOOPS_AI_EXISTS_MAX_IDS"] = "2"
        resp = self.client.post("/files/exists", json={"file_ids": ["a" * 64] * 3})
        self.assertEqual(resp.status_code, 422)

    def test_exists_endpoint_accepts_an_empty_list(self):
        resp = self.client.post("/files/exists", json={"file_ids": []})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"known": [], "unknown": [], "invalid": []})


if __name__ == "__main__":
    unittest.main()
