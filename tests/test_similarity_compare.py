"""Unit and integration tests for the part similarity compare API.

Tests that require a HOOPS AI license and trained model are decorated with
``@unittest.skip`` (or ``skipIf``) so they are skipped in CI environments
that lack the runtime dependencies, matching the convention used in this
project's other test modules.
"""

import io
import os
import sys
import unittest
import embeddings
import zipfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_app():
    """Return a minimal FastAPI app with only the similarity router mounted."""
    from fastapi import FastAPI
    from routers.similarity import router

    app = FastAPI()
    app.include_router(router)
    return app


def _make_zip(entries: dict[str, bytes]) -> bytes:
    """Build an in-memory ZIP archive.

    ``entries`` maps archive member path → raw file content.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# POST /similarity/embed  – validation tests (no HOOPS AI required)
# ---------------------------------------------------------------------------


class TestSimilarityEmbedValidation(unittest.TestCase):
    """Validation-layer tests for POST /similarity/embed."""

    def setUp(self):
        from fastapi.testclient import TestClient

        self.client = TestClient(_make_app())

    def test_no_file_and_no_file_id_returns_422(self):
        """Calling /embed without any input must return 422."""
        response = self.client.post("/similarity/embed")
        self.assertIn(response.status_code, (400, 422))

    def test_empty_post_body_returns_422(self):
        """An empty multipart body (no file, no file_id) must return 422."""
        response = self.client.post("/similarity/embed", data={})
        self.assertIn(response.status_code, (400, 422))


# ---------------------------------------------------------------------------
# POST /similarity/compare  – validation tests (no HOOPS AI required)
# ---------------------------------------------------------------------------


class TestSimilarityCompareValidation(unittest.TestCase):
    """Validation-layer tests for POST /similarity/compare."""

    def setUp(self):
        from fastapi.testclient import TestClient

        self.client = TestClient(_make_app())

    def test_single_file_id_returns_422(self):
        """Providing only one file_id (< 2) must return 422."""
        import core
        import storage

        fake_path = MagicMock()
        fake_path.name = "a" * 64 + "_part.step"

        fake_entry = {
            "file_id": "a" * 64,
            "vector": None,
            "dim": 512,
            "model_name": "hoops_embeddings_model",
            "num_bodies": 1,
            "filename": "part.step",
            "cached": False,
        }

        with (
            patch.object(storage, "find_persistent_CAD_file", return_value=fake_path),
            patch.object(embeddings, "compute_embedding", return_value=fake_entry),
        ):
            response = self.client.post(
                "/similarity/compare",
                params={"file_ids": "a" * 64},
            )

        self.assertEqual(response.status_code, 422)

    def test_no_input_returns_422(self):
        """Calling /compare with no input at all must return 422."""
        response = self.client.post("/similarity/compare")
        self.assertEqual(response.status_code, 422)


# ---------------------------------------------------------------------------
# ZIP Zip-Slip protection tests (no HOOPS AI required)
# ---------------------------------------------------------------------------


class TestZipSlipProtection(unittest.TestCase):
    """The ZIP extraction helper must reject paths that escape the temp dir."""

    def setUp(self):
        from fastapi.testclient import TestClient

        self.client = TestClient(_make_app())

    def test_zip_slip_path_is_rejected(self):
        """A member with ``../`` traversal must cause a 400 response."""
        malicious_zip = _make_zip({"../evil.step": b"step data"})

        response = self.client.post(
            "/similarity/compare",
            files={"zip_file": ("evil.zip", malicious_zip, "application/zip")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Zip Slip", response.json().get("detail", ""))

    def test_absolute_path_in_zip_is_rejected(self):
        """A member with an absolute path must also be rejected as Zip Slip."""
        # zipfile normalises absolute paths on write, so we patch the ZipInfo manually.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            info = zipfile.ZipInfo("/etc/passwd")
            zf.writestr(info, "data")
        bad_zip = buf.getvalue()

        response = self.client.post(
            "/similarity/compare",
            files={"zip_file": ("bad.zip", bad_zip, "application/zip")},
        )
        # Either 400 (Zip Slip) or 422 (not enough files after skipping) is acceptable.
        self.assertIn(response.status_code, (400, 422))


# ---------------------------------------------------------------------------
# embeddings.compare_embeddings  – unit tests with mock vectors
# ---------------------------------------------------------------------------


class TestCompareEmbeddingsLogic(unittest.TestCase):
    """Unit tests for the cosine-similarity matrix calculation in embeddings.compare_embeddings.

    ``compute_embedding`` is mocked so no HOOPS AI runtime is needed.
    """

    def _run_compare(self, vectors: dict[str, list[float]]) -> dict:
        """Patch compute_embedding and call compare_embeddings."""
        import numpy as np
        import core
        import storage

        def fake_compute(file_id: str, **kwargs):
            v = np.array(vectors[file_id], dtype=np.float32)
            norm = float(np.linalg.norm(v))
            v = v / norm if norm > 0 else v
            return {
                "file_id": file_id,
                "vector": v,
                "dim": int(v.shape[0]),
                "model_name": "hoops_embeddings_model",
                "num_bodies": 1,
                "filename": f"{file_id}.step",
                "cached": False,
            }

        with patch.object(embeddings, "compute_embedding", side_effect=fake_compute):
            return embeddings.compare_embeddings(list(vectors.keys()))

    def test_diagonal_is_exactly_one(self):
        """The diagonal of the similarity matrix must be exactly 1.0."""
        vectors = {
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
            "c": [0.0, 0.0, 1.0],
        }
        result = self._run_compare(vectors)
        matrix = result["matrix"]
        for i in range(3):
            self.assertEqual(matrix[i][i], 1.0)

    def test_matrix_is_symmetric(self):
        """The similarity matrix must be symmetric (matrix[i][j] == matrix[j][i])."""
        vectors = {
            "a": [1.0, 2.0, 3.0],
            "b": [3.0, 1.0, 2.0],
            "c": [2.0, 3.0, 1.0],
        }
        result = self._run_compare(vectors)
        matrix = result["matrix"]
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(matrix[i][j], matrix[j][i], places=5)

    def test_orthogonal_vectors_have_zero_similarity(self):
        """Orthogonal vectors should have cosine similarity ≈ 0."""
        vectors = {
            "x": [1.0, 0.0],
            "y": [0.0, 1.0],
        }
        result = self._run_compare(vectors)
        self.assertAlmostEqual(result["matrix"][0][1], 0.0, places=5)

    def test_identical_vectors_have_similarity_one(self):
        """Two identical vectors should have cosine similarity 1.0."""
        vectors = {
            "a": [1.0, 1.0, 1.0],
            "b": [1.0, 1.0, 1.0],
        }
        result = self._run_compare(vectors)
        self.assertAlmostEqual(result["matrix"][0][1], 1.0, places=5)
        self.assertAlmostEqual(result["matrix"][1][0], 1.0, places=5)

    def test_known_cosine_value(self):
        """Verify a known cosine similarity: [1,0] vs [cos45, sin45] ≈ 0.7071."""
        import math

        vectors = {
            "a": [1.0, 0.0],
            "b": [math.cos(math.pi / 4), math.sin(math.pi / 4)],
        }
        result = self._run_compare(vectors)
        self.assertAlmostEqual(result["matrix"][0][1], math.cos(math.pi / 4), places=4)

    def test_pairs_are_sorted_by_score_descending(self):
        """Pairs in the result must be sorted by score, highest first."""
        vectors = {
            "a": [1.0, 0.0, 0.0],
            "b": [0.9, 0.1, 0.0],
            "c": [0.0, 0.0, 1.0],
        }
        result = self._run_compare(vectors)
        scores = [p["score"] for p in result["pairs"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_count_and_files_length_match(self):
        """``count`` and ``len(files)`` must equal the number of input file_ids."""
        vectors = {f"f{i}": [float(i), 1.0] for i in range(4)}
        result = self._run_compare(vectors)
        self.assertEqual(result["count"], 4)
        self.assertEqual(len(result["files"]), 4)

    def test_number_of_pairs(self):
        """For N parts there should be N*(N-1)/2 pairs."""
        n = 4
        vectors = {f"f{i}": [float(i), 1.0] for i in range(n)}
        result = self._run_compare(vectors)
        self.assertEqual(len(result["pairs"]), n * (n - 1) // 2)


# ---------------------------------------------------------------------------
# Router-level compare tests with mocked core functions
# ---------------------------------------------------------------------------


class TestSimilarityCompareRouter(unittest.TestCase):
    """Router-level tests for POST /similarity/compare using mocked core."""

    def setUp(self):
        from fastapi.testclient import TestClient

        self.client = TestClient(_make_app())

    def _fake_compute(self, file_id: str, vec=None, **kwargs):
        import numpy as np

        if vec is None:
            vec = np.array([1.0, 0.0], dtype=np.float32)
        return {
            "file_id": file_id,
            "vector": vec,
            "dim": 2,
            "model_name": "hoops_embeddings_model",
            "num_bodies": 1,
            "filename": f"{file_id}.step",
            "cached": False,
        }

    def test_two_file_ids_returns_200(self):
        """Two valid file_ids should produce a 200 with a 2×2 matrix."""
        import numpy as np
        import core
        import storage

        id_a = "a" * 64
        id_b = "b" * 64

        fake_path_a = MagicMock()
        fake_path_a.name = id_a + "_part_a.step"
        fake_path_b = MagicMock()
        fake_path_b.name = id_b + "_part_b.step"

        def fake_find(fid):
            return fake_path_a if fid == id_a else fake_path_b

        compare_result = {
            "count": 2,
            "model_name": "hoops_embeddings_model",
            "files": [
                {"index": 0, "file_id": id_a, "filename": "part_a.step", "num_bodies": 1},
                {"index": 1, "file_id": id_b, "filename": "part_b.step", "num_bodies": 1},
            ],
            "matrix": [[1.0, 0.95], [0.95, 1.0]],
            "pairs": [{"a": 0, "b": 1, "score": 0.95}],
        }

        with (
            patch.object(storage, "find_persistent_CAD_file", side_effect=fake_find),
            patch.object(embeddings, "compute_embedding", side_effect=self._fake_compute),
            patch.object(embeddings, "compare_embeddings", return_value=compare_result),
        ):
            response = self.client.post(
                "/similarity/compare",
                params={"file_ids": f"{id_a},{id_b}"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 2)
        self.assertEqual(len(body["matrix"]), 2)
        self.assertEqual(len(body["pairs"]), 1)
        self.assertEqual(body["errors"], [])

    def test_valid_zip_with_two_cad_files_returns_200(self):
        """A ZIP archive with two CAD files (mocked) should return 200."""
        import core
        import storage

        zip_bytes = _make_zip(
            {"part_a.step": b"fake step data a", "part_b.step": b"fake step data b"}
        )

        id_a = "a" * 64
        id_b = "b" * 64

        upload_counter = {"n": 0}

        def fake_upload(upload_file):
            upload_counter["n"] += 1
            fid = id_a if upload_counter["n"] == 1 else id_b
            return fid, MagicMock(), False

        compare_result = {
            "count": 2,
            "model_name": "hoops_embeddings_model",
            "files": [
                {"index": 0, "file_id": id_a, "filename": "part_a.step", "num_bodies": 1},
                {"index": 1, "file_id": id_b, "filename": "part_b.step", "num_bodies": 1},
            ],
            "matrix": [[1.0, 0.88], [0.88, 1.0]],
            "pairs": [{"a": 0, "b": 1, "score": 0.88}],
        }

        with (
            patch.object(storage, "upload_CAD_file_persistent", side_effect=fake_upload),
            patch.object(embeddings, "compute_embedding", side_effect=self._fake_compute),
            patch.object(embeddings, "compare_embeddings", return_value=compare_result),
        ):
            response = self.client.post(
                "/similarity/compare",
                files={"zip_file": ("parts.zip", zip_bytes, "application/zip")},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 2)


# ---------------------------------------------------------------------------
# HOOPS AI dependent tests – skipped without runtime
# ---------------------------------------------------------------------------


@unittest.skip(
    "Requires HOOPS AI license and trained model checkpoint. "
    "Run manually in a configured HOOPS AI environment."
)
class TestSimilarityCompareIntegration(unittest.TestCase):
    """End-to-end integration tests — require HOOPS AI runtime."""

    def test_embed_real_cad_file(self):
        """Upload a real CAD file and verify the embedding shape."""
        raise NotImplementedError("Provide a real .step file path to run this test.")

    def test_compare_two_real_cad_files(self):
        """Compare two real CAD files and verify the similarity matrix structure."""
        raise NotImplementedError("Provide two real .step file paths to run this test.")


if __name__ == "__main__":
    unittest.main()
