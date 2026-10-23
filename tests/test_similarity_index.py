"""Tests for the named index management API (POST/GET/DELETE /similarity/index/*).

Tests that require a HOOPS AI license are decorated with ``@unittest.skip`` to match the
project convention.  All validation-layer and mock-based tests run without a license.
"""

import io
import os
import pathlib
import sys
import tempfile
import threading
import unittest
import zipfile
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
import index_registry
import embeddings


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_app():
    """Return a minimal FastAPI app with only the similarity router mounted."""
    from fastapi import FastAPI
    from routers.similarity import router

    app = FastAPI()
    app.include_router(router)
    return app


def _make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _fake_vs(ids: list[str] | None = None):
    """Build a minimal FaissVectorStore mock."""
    ids = ids or []
    vs = MagicMock()
    vs.get_ids.return_value = list(ids)
    vs.count.return_value = len(ids)
    vs.query.return_value = []
    return vs


def _fake_embedding(file_id: str, dim: int = 4):
    import numpy as np

    v = np.array([1.0, 0.0, 0.0, 0.0][:dim], dtype=np.float32)
    return {
        "file_id": file_id,
        "vector": v,
        "dim": dim,
        "model_name": "hoops_embeddings_model",
        "num_bodies": 1,
        "filename": f"{file_id[:8]}.step",
        "cached": False,
    }


# ---------------------------------------------------------------------------
# Lifecycle test: create → list → add → search → remove → delete
# ---------------------------------------------------------------------------


class TestIndexLifecycle(unittest.TestCase):
    """Full lifecycle with mocked FaissVectorStore and compute_embedding."""

    def setUp(self):
        import core
        import config
        from fastapi.testclient import TestClient

        self.client = TestClient(_make_app())
        self.core = core

        # Each test gets a fresh temp dir for INDEXES_DIR
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_indexes_dir = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmpdir.name)
        # Clear the in-memory cache between tests
        index_registry._named_indexes.clear()
        core._index_locks.clear()

    def tearDown(self):
        import core

        config.INDEXES_DIR = self._orig_indexes_dir
        index_registry._named_indexes.clear()
        core._index_locks.clear()
        self._tmpdir.cleanup()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _mock_faiss(self, dim: int = 4):
        """Return a context-manager that patches FaissVectorStore and related imports."""
        import numpy as np
        from unittest.mock import MagicMock, patch

        # We need to patch inside hoops_ai.ml.embeddings AND inside core's local import
        vs_instance = MagicMock()
        id_store: list[str] = []

        def upsert(records):
            for r in records:
                if r.id not in id_store:
                    id_store.append(r.id)

        def delete(ids):
            for i in ids:
                if i in id_store:
                    id_store.remove(i)

        vs_instance.get_ids.side_effect = lambda: list(id_store)
        vs_instance.count.side_effect = lambda: len(id_store)
        vs_instance.upsert.side_effect = upsert
        vs_instance.delete.side_effect = delete
        vs_instance.save = MagicMock()

        FaissVectorStoreMock = MagicMock()
        FaissVectorStoreMock.return_value = vs_instance
        FaissVectorStoreMock.load.return_value = vs_instance

        EmbeddingMock = MagicMock()
        VectorRecordMock = MagicMock(side_effect=lambda id, embedding, metadata: MagicMock(id=id))

        embedder_mock = MagicMock()
        embedder_mock.embedding_dim = dim

        patches = [
            patch("core.FaissVectorStore", FaissVectorStoreMock, create=True),
        ]

        # We also need to mock the import inside the function bodies
        import_patch = patch.dict(
            "sys.modules",
            {
                "hoops_ai": MagicMock(),
                "hoops_ai.ml": MagicMock(),
                "hoops_ai.ml.embeddings": MagicMock(
                    FaissVectorStore=FaissVectorStoreMock,
                    Embedding=EmbeddingMock,
                    VectorRecord=VectorRecordMock,
                ),
            },
        )
        return import_patch, FaissVectorStoreMock, vs_instance, embedder_mock, EmbeddingMock, VectorRecordMock

    def test_create_then_list(self):
        """create_index creates a .faiss file; list_indexes reports it."""
        import core

        fid = "a" * 64
        mock_embedder = MagicMock()
        mock_embedder.embedding_dim = 4

        FaissVS = MagicMock()
        vs_inst = MagicMock()
        vs_inst.get_ids.return_value = []
        vs_inst.count.return_value = 0
        vs_inst.save.side_effect = lambda path: (
            pathlib.Path(path + ".faiss").write_bytes(b"fake"),
            pathlib.Path(path + ".meta").write_bytes(b"fake"),
        )
        FaissVS.return_value = vs_inst
        FaissVS.load.return_value = vs_inst

        with (
            patch("embeddings.get_embedder", return_value=mock_embedder),
            patch.dict(
                "sys.modules",
                {"hoops_ai.ml.embeddings": MagicMock(
                    FaissVectorStore=FaissVS,
                    Embedding=MagicMock(),
                    VectorRecord=MagicMock(),
                )},
            ),
        ):
            result = core.create_index("demo")
            self.assertEqual(result["name"], "demo")
            self.assertEqual(result["count"], 0)
            self.assertEqual(result["dim"], 4)

            indexes = index_registry.list_indexes()
            names = [i["name"] for i in indexes]
            self.assertIn("demo", names)

    def test_add_increments_count(self):
        """add_to_index upserts body-level records and increments the count."""
        import core

        vs_inst = _fake_vs([])
        id_store: list[str] = []

        def upsert_impl(records):
            for r in records:
                if hasattr(r, "id") and isinstance(r.id, str) and r.id not in id_store:
                    id_store.append(r.id)

        vs_inst.upsert.side_effect = upsert_impl
        vs_inst.get_ids.side_effect = lambda: list(id_store)

        fid = "a" * 64

        # schema v2 index on disk so the write guard passes
        (config.INDEXES_DIR / "demo").mkdir(parents=True, exist_ok=True)
        (config.INDEXES_DIR / "demo" / "index.faiss").write_bytes(b"fake")
        (config.INDEXES_DIR / "demo" / "index.meta").write_bytes(b"fake")
        core._save_index_schema("demo", 2)
        index_registry._named_indexes["demo"] = vs_inst

        fake_record = MagicMock(id=fid)

        with (
            patch.object(
                index_registry, "_embed_bodies_for_index", return_value=([fake_record], [])
            ),
            patch.object(index_registry, "_save_named_index_atomic"),
        ):
            result = index_registry.add_to_index("demo", [fid])

        self.assertEqual(result["added"], 1)
        self.assertEqual(result["updated"], 0)
        self.assertEqual(len(result["errors"]), 0)

    def test_readd_counts_as_updated(self):
        """Re-adding an existing file_id is counted as 'updated', not 'added'."""
        import core

        fid = "b" * 64
        vs_inst = _fake_vs([fid])  # already contains fid

        (config.INDEXES_DIR / "myidx").mkdir(parents=True, exist_ok=True)
        (config.INDEXES_DIR / "myidx" / "index.faiss").write_bytes(b"fake")
        (config.INDEXES_DIR / "myidx" / "index.meta").write_bytes(b"fake")
        core._save_index_schema("myidx", 2)
        index_registry._named_indexes["myidx"] = vs_inst

        fake_record = MagicMock(id=fid)

        with (
            patch.object(
                index_registry, "_embed_bodies_for_index", return_value=([fake_record], [])
            ),
            patch.object(index_registry, "_save_named_index_atomic"),
        ):
            result = index_registry.add_to_index("myidx", [fid])

        self.assertEqual(result["added"], 0)
        self.assertEqual(result["updated"], 1)
        # delete was called once for the existing id
        vs_inst.delete.assert_called_once_with([fid])

    def test_search_returns_empty_for_empty_index(self):
        """search_index on a 0-entry index returns empty hits without error."""
        import core
        import search

        vs_inst = _fake_vs([])
        (config.INDEXES_DIR / "empty").mkdir(parents=True, exist_ok=True)
        (config.INDEXES_DIR / "empty" / "index.faiss").write_bytes(b"fake")
        (config.INDEXES_DIR / "empty" / "index.meta").write_bytes(b"fake")
        index_registry._named_indexes["empty"] = vs_inst

        with patch.object(embeddings, "compute_embedding", side_effect=_fake_embedding):
            result = search.search_index("empty", "a" * 64, top_k=5)

        self.assertEqual(result["hits"], [])
        self.assertEqual(result["count"], 0)
        vs_inst.query.assert_not_called()

    def test_remove_decrements_count(self):
        """remove_from_index calls delete and saves."""
        import core

        fid = "c" * 64
        vs_inst = _fake_vs([fid])
        deleted: list[str] = []

        def fake_delete(ids):
            for i in ids:
                if i in vs_inst.get_ids.return_value:
                    vs_inst.get_ids.return_value = [x for x in vs_inst.get_ids.return_value if x != i]
                deleted.extend(ids)

        vs_inst.delete.side_effect = fake_delete

        (config.INDEXES_DIR / "idx").mkdir(parents=True, exist_ok=True)
        (config.INDEXES_DIR / "idx" / "index.faiss").write_bytes(b"fake")
        (config.INDEXES_DIR / "idx" / "index.meta").write_bytes(b"fake")
        core._save_index_schema("idx", 2)
        index_registry._named_indexes["idx"] = vs_inst

        FaissVS = MagicMock()
        with (
            patch.dict(
                "sys.modules",
                {"hoops_ai.ml.embeddings": MagicMock(FaissVectorStore=FaissVS)},
            ),
            patch.object(index_registry, "_save_named_index_atomic"),
        ):
            result = core.remove_from_index("idx", [fid])

        self.assertEqual(result["removed"], 1)
        self.assertIn(fid, deleted)

    def test_delete_index_removes_files(self):
        """delete_index removes the entire index directory from disk."""
        import core

        (config.INDEXES_DIR / "todel").mkdir(parents=True, exist_ok=True)
        (config.INDEXES_DIR / "todel" / "index.faiss").write_bytes(b"data")
        (config.INDEXES_DIR / "todel" / "index.meta").write_bytes(b"data")

        result = core.delete_index("todel")
        self.assertTrue(result["deleted"])
        self.assertFalse((config.INDEXES_DIR / "todel").exists())
        self.assertNotIn("todel", index_registry._named_indexes)


# ---------------------------------------------------------------------------
# The name "default" is an ordinary index name
# ---------------------------------------------------------------------------


class TestDefaultNameAllowed(unittest.TestCase):
    """``default`` used to be a reserved read-only name; it is now ordinary."""

    def setUp(self):
        import core
        from fastapi.testclient import TestClient

        self.client = TestClient(_make_app())
        self.core = core
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_indexes_dir = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmpdir.name)
        index_registry._named_indexes.clear()
        core._index_locks.clear()

    def tearDown(self):
        import core

        config.INDEXES_DIR = self._orig_indexes_dir
        index_registry._named_indexes.clear()
        core._index_locks.clear()
        self._tmpdir.cleanup()

    def test_default_can_be_created_and_deleted_like_any_name(self):
        import core

        mock_embedder = MagicMock()
        mock_embedder.embedding_dim = 4

        FaissVS = MagicMock()
        vs_inst = MagicMock()
        vs_inst.get_ids.return_value = []
        vs_inst.count.return_value = 0
        vs_inst.save.side_effect = lambda path: (
            pathlib.Path(path + ".faiss").write_bytes(b"fake"),
            pathlib.Path(path + ".meta").write_bytes(b"fake"),
        )
        FaissVS.return_value = vs_inst
        FaissVS.load.return_value = vs_inst

        with (
            patch("embeddings.get_embedder", return_value=mock_embedder),
            patch.dict(
                "sys.modules",
                {"hoops_ai.ml.embeddings": MagicMock(
                    FaissVectorStore=FaissVS,
                    Embedding=MagicMock(),
                    VectorRecord=MagicMock(),
                )},
            ),
        ):
            result = core.create_index("default")
            self.assertEqual(result["name"], "default")
            self.assertTrue(
                (config.INDEXES_DIR / "default" / "index.faiss").exists()
            )

            deleted = core.delete_index("default")
            self.assertEqual(deleted["name"], "default")
            self.assertFalse((config.INDEXES_DIR / "default").exists())


# ---------------------------------------------------------------------------
# Invalid index name validation
# ---------------------------------------------------------------------------


class TestInvalidIndexNames(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(_make_app())

    def _assert_422_on_create(self, name: str):
        resp = self.client.post(f"/similarity/index/create?name={name}")
        self.assertIn(
            resp.status_code,
            (422,),
            msg=f"Expected 422 for name={name!r}, got {resp.status_code}",
        )

    def test_uppercase_name_rejected(self):
        self._assert_422_on_create("MyIndex")

    def test_space_in_name_rejected(self):
        # Spaces cause FastAPI to reject or core validates
        resp = self.client.post("/similarity/index/create", params={"name": "my index"})
        self.assertIn(resp.status_code, (422,))

    def test_path_traversal_rejected(self):
        resp = self.client.post("/similarity/index/create", params={"name": "../evil"})
        self.assertIn(resp.status_code, (422,))

    def test_too_long_name_rejected(self):
        long_name = "a" * 65
        self._assert_422_on_create(long_name)

    def test_empty_name_rejected(self):
        # FastAPI requires non-empty query param
        resp = self.client.post("/similarity/index/create?name=")
        self.assertIn(resp.status_code, (422,))


# ---------------------------------------------------------------------------
# Duplicate create returns 409
# ---------------------------------------------------------------------------


class TestDuplicateCreate(unittest.TestCase):
    def setUp(self):
        import core
        from fastapi.testclient import TestClient

        self.client = TestClient(_make_app())
        self.core = core
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_indexes_dir = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmpdir.name)
        index_registry._named_indexes.clear()
        core._index_locks.clear()

    def tearDown(self):
        import core

        config.INDEXES_DIR = self._orig_indexes_dir
        index_registry._named_indexes.clear()
        core._index_locks.clear()
        self._tmpdir.cleanup()

    def test_duplicate_create_returns_409(self):
        """Creating an index that already exists must return 409."""
        import core

        # Pre-create the index directory with the new structure
        (config.INDEXES_DIR / "exists").mkdir(parents=True, exist_ok=True)
        (config.INDEXES_DIR / "exists" / "index.faiss").write_bytes(b"data")
        (config.INDEXES_DIR / "exists" / "index.meta").write_bytes(b"data")

        with patch.object(embeddings, "get_embedder", return_value=MagicMock(embedding_dim=4)):
            try:
                core.create_index("exists")
                self.fail("Expected FileExistsError")
            except FileExistsError:
                pass  # expected


# ---------------------------------------------------------------------------
# Delete without confirm returns 409
# ---------------------------------------------------------------------------


class TestDeleteConfirmRequired(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(_make_app())

    def test_delete_without_confirm_returns_409(self):
        """DELETE /similarity/index/{name} without ?confirm=true must return 409."""
        response = self.client.delete("/similarity/index/demo")
        self.assertEqual(response.status_code, 409)
        self.assertIn("confirm=true", response.json()["detail"])

    def test_delete_with_confirm_false_returns_409(self):
        response = self.client.delete("/similarity/index/demo?confirm=false")
        self.assertEqual(response.status_code, 409)


# ---------------------------------------------------------------------------
# Atomic save unit test
# ---------------------------------------------------------------------------


class TestAtomicSave(unittest.TestCase):
    """Verify that _save_named_index_atomic writes temp files then renames them."""

    def setUp(self):
        import core
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_indexes_dir = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmpdir.name)

    def tearDown(self):
        import core
        config.INDEXES_DIR = self._orig_indexes_dir
        self._tmpdir.cleanup()

    def test_atomic_save_creates_target_files_via_rename(self):
        """save() must write temp files and replace() them into the final paths."""
        import core

        written_paths: list[str] = []

        def fake_save(path: str):
            pathlib.Path(path + ".faiss").write_bytes(b"faiss_data")
            pathlib.Path(path + ".meta").write_bytes(b"meta_data")
            written_paths.append(path)

        vs = MagicMock()
        vs.save.side_effect = fake_save

        index_registry._save_named_index_atomic("mytest", vs)

        dest_faiss = config.INDEXES_DIR / "mytest" / "index.faiss"
        dest_meta = config.INDEXES_DIR / "mytest" / "index.meta"
        self.assertTrue(dest_faiss.exists(), "Final index.faiss must exist after atomic save")
        self.assertTrue(dest_meta.exists(), "Final index.meta must exist after atomic save")
        self.assertEqual(dest_faiss.read_bytes(), b"faiss_data")
        self.assertEqual(dest_meta.read_bytes(), b"meta_data")

        self.assertEqual(len(written_paths), 1)
        tmp_base = written_paths[0]
        self.assertFalse(
            pathlib.Path(tmp_base + ".faiss").exists(),
            "Temp .faiss must not remain after successful replace()",
        )

    def test_atomic_save_cleans_up_temp_on_failure(self):
        """If replace() fails, temp files must be cleaned up."""
        import core

        def bad_save(path: str):
            pathlib.Path(path + ".faiss").write_bytes(b"x")
            pathlib.Path(path + ".meta").write_bytes(b"y")
            raise RuntimeError("disk full")

        vs = MagicMock()
        vs.save.side_effect = bad_save

        with self.assertRaises(RuntimeError):
            index_registry._save_named_index_atomic("fail_idx", vs)

        # No temp files should remain
        remaining = list(config.INDEXES_DIR.glob("_tmp_*.faiss"))
        self.assertEqual(remaining, [], "Temp .faiss must be cleaned up after failure")


# ---------------------------------------------------------------------------
# Router-level validation tests (no HOOPS AI license needed)
# ---------------------------------------------------------------------------


class TestRouterValidation(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(_make_app())

    def test_add_with_no_input_returns_422(self):
        """POST /similarity/index/add without any input must return 422."""
        response = self.client.post("/similarity/index/add?name=demo")
        self.assertEqual(response.status_code, 422)

    def test_search_with_no_file_or_id_returns_422(self):
        """POST /similarity/index/{name}/search without file or file_id must return 422."""
        response = self.client.post("/similarity/index/demo/search")
        self.assertEqual(response.status_code, 422)

    def test_remove_parts_with_no_ids_returns_422(self):
        """DELETE /similarity/index/{name}/parts without part_ids must return 422 (missing required query)."""
        response = self.client.delete("/similarity/index/demo/parts")
        # FastAPI returns 422 for missing required query params
        self.assertIn(response.status_code, (422, 400))

    def test_list_indexes_returns_200(self):
        """GET /similarity/index/list must return 200 even when no indexes exist."""
        import core
        with patch.object(index_registry, "list_indexes", return_value=[]):
            response = self.client.get("/similarity/index/list")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_stats_carry_the_default_cluster_count(self):
        """GET /similarity/index/{name}/stats answers default_clusters.

        The Shape Map's controls open on it, and startup warmup builds with the
        same rule, so a client that reads it opens a map that was warmed.
        Derived from single_part because the map opens on Show: Parts.
        """
        import core
        import shape_map
        stats = {
            "name": "demo",
            "files": 9000,
            "bodies": 12000,
            "assemblies": 1072,
            "single_part": 7928,
            "dim": 256,
            "schema_version": 2,
            "last_modified": None,
        }
        with patch.object(core, "get_index_stats", return_value=dict(stats)):
            response = self.client.get("/similarity/index/demo/stats")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["default_clusters"],
            shape_map.shape_map_auto_cluster_count(7928),
        )


# ---------------------------------------------------------------------------
# HOOPS AI dependent integration tests – skipped without runtime
# ---------------------------------------------------------------------------


@unittest.skip(
    "Requires HOOPS AI license and trained model checkpoint. "
    "Run manually in a configured HOOPS AI environment."
)
class TestIndexManagementIntegration(unittest.TestCase):
    """End-to-end tests — require HOOPS AI runtime."""

    def test_full_lifecycle(self):
        """create → add → search → remove → delete lifecycle with real CAD files."""
        raise NotImplementedError("Provide real .step file paths to run this test.")


if __name__ == "__main__":
    unittest.main()
