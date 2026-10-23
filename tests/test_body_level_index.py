"""SDK-free unit tests for the body-level named-index migration.

Covers the pure/testable pieces of the schema-v2 change:
  * per-file_id search dedup (keep best score, rank, truncate)
  * ``list_index_parts`` paging bounds (offset past end, limit clamp)
  * schema fallback (index.json / model.json / neither)
  * OBB serialisation guard (drop when > 512 bytes or not JSON-serialisable)
  * per-part asset resolution path-traversal protection

FaissVectorStore is faked; no HOOPS AI license is required.
"""

import os
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core
import index_registry
import embeddings
import storage
import config


def _hit(id_, score):
    return SimpleNamespace(id=id_, score=score, metadata={"file_id": id_})


class TestObbSerializationGuard(unittest.TestCase):
    def test_small_value_kept(self):
        val = {"min": [0.0, 0.0, 0.0], "max": [1.0, 2.0, 3.0]}
        out = core._obb_meta_value(val)
        self.assertIsInstance(out, str)
        self.assertIn("min", out)

    def test_oversize_value_dropped(self):
        val = {"data": list(range(500))}  # serialises to well over 512 bytes
        self.assertIsNone(core._obb_meta_value(val))

    def test_non_serializable_dropped(self):
        self.assertIsNone(core._obb_meta_value({1, 2, 3}))  # set is not JSON

    def test_boundary_512(self):
        # Construct a value whose JSON is exactly <= 512 bytes and just over.
        ok = {"s": "x" * 400}
        self.assertIsNotNone(core._obb_meta_value(ok))
        too_big = {"s": "x" * 520}
        self.assertIsNone(core._obb_meta_value(too_big))


class TestSchemaFallback(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmp.name)

    def tearDown(self):
        config.INDEXES_DIR = self._orig
        self._tmp.cleanup()

    def _mk(self, name):
        d = config.INDEXES_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_index_json_takes_precedence(self):
        d = self._mk("a")
        (d / "index.json").write_text('{"model": "signal", "schema_version": 2}')
        (d / "model.json").write_text('{"model": "legacy"}')
        schema = index_registry._load_index_schema("a")
        self.assertEqual(schema, {"schema_version": 2})

    def test_model_json_only_is_schema_1(self):
        d = self._mk("b")
        (d / "model.json").write_text('{"model": "signal"}')
        schema = index_registry._load_index_schema("b")
        self.assertEqual(schema, {"schema_version": 1})

    def test_neither_is_legacy_schema_1(self):
        self._mk("c")
        schema = index_registry._load_index_schema("c")
        self.assertEqual(schema, {"schema_version": 1})

    def test_legacy_model_field_is_ignored(self):
        d = self._mk("d")
        (d / "index.json").write_text('{"model": "signal", "schema_version": 2}')
        schema = index_registry._load_index_schema("d")
        self.assertEqual(schema, {"schema_version": 2})
        self.assertNotIn("model", schema)


class TestResolveIndexAssetTraversal(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmp.name)
        # Create a valid thumbnail and a secret file outside the index dir.
        self.idx = config.INDEXES_DIR / "idx"
        (self.idx / "thumbnails").mkdir(parents=True, exist_ok=True)
        self.part_id = "a" * 64
        (self.idx / "thumbnails" / f"{self.part_id}.png").write_bytes(b"png")
        (config.INDEXES_DIR / "secret.png").write_bytes(b"secret")

    def tearDown(self):
        config.INDEXES_DIR = self._orig
        self._tmp.cleanup()

    def test_valid_part_resolves(self):
        path = index_registry._resolve_index_asset("idx", self.part_id, "thumbnail")
        self.assertIsNotNone(path)
        self.assertTrue(path.is_file())

    def test_missing_part_returns_none(self):
        self.assertIsNone(index_registry._resolve_index_asset("idx", "b" * 64, "thumbnail"))

    def test_traversal_part_id_returns_none(self):
        # Attempt to escape the index directory via a crafted part_id.
        malicious = os.path.join("..", "..", "secret")
        self.assertIsNone(index_registry._resolve_index_asset("idx", malicious, "thumbnail"))

    def test_unknown_asset_returns_none(self):
        self.assertIsNone(index_registry._resolve_index_asset("idx", self.part_id, "bogus"))


class TestListIndexPartsPaging(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmp.name)
        (config.INDEXES_DIR / "idx").mkdir(parents=True, exist_ok=True)
        core._save_index_schema("idx", 2)

        # Fake vs: 5 files, each with a couple of body rows (metadata repeats).
        self._metas = []
        for i in range(5):
            fid = f"{i:064d}"
            kind = "assembly" if i % 2 == 0 else "part"
            bodies = 3 if kind == "assembly" else 1
            for _ in range(bodies):
                self._metas.append(
                    {
                        "file_id": fid,
                        "filename": f"file{i}.step",
                        "kind": kind,
                        "bodies": bodies,
                        "registered_at": "2026-01-01T00:00:00Z",
                    }
                )
        vs = MagicMock()
        vs.iter_metadata.side_effect = lambda: iter(self._metas)
        index_registry._named_indexes["idx"] = vs

    def tearDown(self):
        index_registry._named_indexes.pop("idx", None)
        config.INDEXES_DIR = self._orig
        self._tmp.cleanup()

    def test_total_is_distinct_files(self):
        result = core.list_index_parts("idx", offset=0, limit=100)
        self.assertEqual(result["total"], 5)
        self.assertEqual(len(result["items"]), 5)

    def test_offset_past_end_returns_empty(self):
        result = core.list_index_parts("idx", offset=99, limit=10)
        self.assertEqual(result["total"], 5)
        self.assertEqual(result["items"], [])
        self.assertEqual(result["offset"], 99)

    def test_limit_clamped_low(self):
        result = core.list_index_parts("idx", offset=0, limit=0)
        self.assertEqual(result["limit"], 1)
        self.assertEqual(len(result["items"]), 1)

    def test_limit_clamped_high(self):
        result = core.list_index_parts("idx", offset=0, limit=999999)
        self.assertEqual(result["limit"], 2000)

    def test_negative_offset_clamped(self):
        result = core.list_index_parts("idx", offset=-5, limit=10)
        self.assertEqual(result["offset"], 0)

    def test_kind_filter(self):
        parts = core.list_index_parts("idx", offset=0, limit=100, kind="part")
        self.assertTrue(all(it["kind"] == "part" for it in parts["items"]))
        self.assertEqual(parts["total"], 2)  # i=1,3 are parts

        asms = core.list_index_parts("idx", offset=0, limit=100, kind="assembly")
        self.assertEqual(asms["total"], 3)  # i=0,2,4

    def test_item_shape(self):
        result = core.list_index_parts("idx", offset=0, limit=1)
        item = result["items"][0]
        self.assertIn("has_thumbnail", item)
        self.assertIn("has_scs", item)
        self.assertFalse(item["has_thumbnail"])  # no asset files created
        self.assertFalse(item["has_scs"])


class TestEmbedBodiesMissingFiles(unittest.TestCase):
    """Files that embed_shape_batch silently skips must surface in errors."""

    def setUp(self):
        import types

        # Fake the hoops_ai.ml.embeddings symbols that _embed_bodies_for_index
        # imports, so no SDK/license is needed.
        self._saved_mods = {}
        for name in ("hoops_ai", "hoops_ai.ml", "hoops_ai.ml.embeddings"):
            self._saved_mods[name] = sys.modules.get(name)
        pkg = types.ModuleType("hoops_ai")
        ml = types.ModuleType("hoops_ai.ml")
        emb = types.ModuleType("hoops_ai.ml.embeddings")

        class _Embedding:
            def __init__(self, values, model, dim):
                self.values, self.model, self.dim = values, model, dim

        class _VectorRecord:
            def __init__(self, id, embedding, metadata):
                self.id, self.embedding, self.metadata = id, embedding, metadata

        emb.Embedding = _Embedding
        emb.VectorRecord = _VectorRecord
        pkg.ml = ml
        ml.embeddings = emb
        sys.modules["hoops_ai"] = pkg
        sys.modules["hoops_ai.ml"] = ml
        sys.modules["hoops_ai.ml.embeddings"] = emb

        self._tmp = tempfile.TemporaryDirectory()
        self._uploads = tempfile.TemporaryDirectory()
        self._orig_idx = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmp.name)

        # Three real CAD files so path .resolve() round-trips inside _fid_for_row.
        self.fids = []
        self.paths = {}
        self.filenames = {}
        for i in range(3):
            fid = f"{i:064d}"
            fname = f"file{i}.step"
            p = pathlib.Path(self._uploads.name) / f"{fid}_{fname}"
            p.write_bytes(b"x")
            self.fids.append(fid)
            self.paths[fid] = p
            self.filenames[fid] = fname

        self._orig_get_embedder = embeddings.get_embedder
        self._orig_find = storage.find_persistent_CAD_file
        storage.find_persistent_CAD_file = lambda fid: self.paths[fid]

    def tearDown(self):
        embeddings.get_embedder = self._orig_get_embedder
        storage.find_persistent_CAD_file = self._orig_find
        config.INDEXES_DIR = self._orig_idx
        self._tmp.cleanup()
        self._uploads.cleanup()
        for name, mod in self._saved_mods.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    def _install_embedder(self, errors_meta, include_errors=True):
        """Fake embedder whose batch returns rows for the first two files only
        (file index 0 gets 2 body rows, file index 1 gets 1), skipping file 2.
        """
        import numpy as np

        def _embed_shape_batch(paths, show_progress=False, specifications=None, **kwargs):
            # paths[0] -> 2 rows, paths[1] -> 1 row; paths[2] is skipped.
            ids = [paths[0], paths[0], paths[1]]
            values = np.ones((3, 4), dtype=np.float32)
            meta = {
                "kind": ["assembly", "assembly", "part"],
                "failed_count": 1,
                "total_processed": 3,
            }
            if include_errors:
                meta["errors"] = errors_meta
            return SimpleNamespace(ids=ids, values=values, metadata=meta)

        embedder = SimpleNamespace(
            model_name="fake-model", embed_shape_batch=_embed_shape_batch
        )
        embeddings.get_embedder = lambda: embedder

    def test_missing_file_reported_with_list_errors(self):
        skipped = self.filenames[self.fids[2]]
        self._install_embedder([{"file": skipped, "error": "CAD load failed"}])
        records, errors = index_registry._embed_bodies_for_index(self.fids, "idx")

        # Two successful files became records; the skipped one is in errors.
        self.assertEqual({r.id for r in records}, {self.fids[0], self.fids[1]})
        self.assertEqual(len(records), 3)  # 2 rows + 1 row
        err_ids = {e["file_id"] for e in errors}
        self.assertIn(self.fids[2], err_ids)
        detail = next(e["detail"] for e in errors if e["file_id"] == self.fids[2])
        self.assertEqual(detail, "CAD load failed")
        self.assertTrue(detail)

    def test_detail_never_empty_or_none(self):
        self._install_embedder([{"file": self.filenames[self.fids[2]], "error": ""}])
        _records, errors = index_registry._embed_bodies_for_index(self.fids, "idx")
        detail = next(e["detail"] for e in errors if e["file_id"] == self.fids[2])
        self.assertIsNotNone(detail)
        self.assertNotEqual(detail, "")

    def test_dict_errors_structure(self):
        skipped_fname = self.filenames[self.fids[2]]
        self._install_embedder({skipped_fname: "timeout while loading"})
        _records, errors = index_registry._embed_bodies_for_index(self.fids, "idx")
        detail = next(e["detail"] for e in errors if e["file_id"] == self.fids[2])
        self.assertEqual(detail, "timeout while loading")

    def test_missing_errors_key_falls_back(self):
        self._install_embedder(None, include_errors=False)
        _records, errors = index_registry._embed_bodies_for_index(self.fids, "idx")
        detail = next(e["detail"] for e in errors if e["file_id"] == self.fids[2])
        self.assertTrue(detail)
        self.assertIn("embedding produced no rows", detail)

    def test_unrelated_list_errors_falls_back(self):
        # errors present but with no entry matching the skipped file.
        self._install_embedder([{"file": "someone_else.step", "error": "boom"}])
        _records, errors = index_registry._embed_bodies_for_index(self.fids, "idx")
        detail = next(e["detail"] for e in errors if e["file_id"] == self.fids[2])
        self.assertIn("embedding produced no rows", detail)

    def test_success_not_dragged_down_by_failure(self):
        self._install_embedder([{"file": self.filenames[self.fids[2]], "error": "x"}])
        records, errors = index_registry._embed_bodies_for_index(self.fids, "idx")
        added_files = {r.id for r in records}
        self.assertEqual(len(added_files), 2)
        self.assertNotIn(self.fids[2], added_files)

    def test_warns_when_errors_metadata_has_unexpected_type(self):
        # A dict still works via the fallback, but must not degrade silently.
        self._install_embedder({self.filenames[self.fids[2]]: "boom"})
        with self.assertLogs(index_registry.logger, level="WARNING") as cm:
            index_registry._embed_bodies_for_index(self.fids, "idx")
        self.assertTrue(
            any("unexpected type" in m for m in cm.output),
            f"expected an unexpected-type warning, got: {cm.output}",
        )

    def test_warns_when_no_error_entry_matches_the_failed_file(self):
        # Non-empty list whose entries reference some other file, and no
        # error_summary.json to fall back on: the gap must be announced.
        self._install_embedder(["Failed to compute embedding for /other/file.step: boom"])
        with self.assertLogs(index_registry.logger, level="WARNING") as cm:
            _records, errors = index_registry._embed_bodies_for_index(self.fids, "idx")
        self.assertTrue(
            any("reasons are unreliable" in m for m in cm.output),
            f"expected an unattributable-reason warning, got: {cm.output}",
        )
        detail = next(e["detail"] for e in errors if e["file_id"] == self.fids[2])
        self.assertEqual(detail, core._EMBED_ERROR_DEFAULT)

    def test_no_degradation_warning_on_the_expected_format(self):
        path = str(self.paths[self.fids[2]].resolve())
        self._install_embedder(
            ["Failed to compute embedding for {}: bad format".format(path)]
        )
        with self.assertLogs(index_registry.logger, level="WARNING") as cm:
            # Emit a sentinel so assertLogs never fails on "no logs at all".
            index_registry.logger.warning("sentinel")
            index_registry._embed_bodies_for_index(self.fids, "idx")
        joined = " ".join(cm.output)
        self.assertNotIn("unexpected type", joined)
        self.assertNotIn("format may have changed", joined)


class TestEmbedErrorDetail(unittest.TestCase):
    """Reason extraction from batch.metadata['errors'].

    The real shape observed on hoops_ai 1.1.0 is a list of plain strings:
      "Failed to compute embedding for <abs path>: <reason>"
    """

    REAL_REASON = "Error in generating scs image for file {p}"
    REAL_MSG = "Failed to compute embedding for {p}: " + REAL_REASON

    def test_real_observed_string_format(self):
        p = os.path.join("root", "temp", "broken.stp")
        errs = [self.REAL_MSG.format(p=p)]
        out = core._embed_error_detail(errs, "a" * 64, "broken.stp", p)
        # Kept verbatim: the message is short and already names the file.
        self.assertEqual(out, self.REAL_MSG.format(p=p))

    def test_similar_filename_does_not_false_match(self):
        # Regression: substring matching made "0.stp" match "100.stp".
        other = os.path.join("root", "mix", "100.stp")
        mine = os.path.join("root", "mix", "0.stp")
        errs = [self.REAL_MSG.format(p=other)]
        out = core._embed_error_detail(errs, "b" * 64, "0.stp", mine)
        self.assertIn("embedding produced no rows", out)

    def test_correct_entry_picked_among_several(self):
        a = os.path.join("root", "mix", "1.stp")
        b = os.path.join("root", "mix", "10.stp")
        errs = [
            "Failed to compute embedding for {}: reason A".format(a),
            "Failed to compute embedding for {}: reason B".format(b),
        ]
        self.assertEqual(core._embed_error_detail(errs, "c" * 64, "10.stp", b), errs[1])
        self.assertEqual(core._embed_error_detail(errs, "d" * 64, "1.stp", a), errs[0])

    def test_unparseable_string_kept_verbatim(self):
        p = os.path.join("root", "temp", "x.stp")
        errs = ["something odd happened with " + p]
        out = core._embed_error_detail(errs, "e" * 64, "x.stp", p)
        self.assertEqual(out, "something odd happened with " + p)

    def test_dict_keyed_by_path(self):
        p = os.path.join("root", "temp", "x.stp")
        out = core._embed_error_detail({p: "timeout"}, "f" * 64, "x.stp", p)
        self.assertEqual(out, "timeout")

    def test_list_of_dicts(self):
        p = os.path.join("root", "temp", "x.stp")
        errs = [{"file": p, "detail": "load failed"}]
        out = core._embed_error_detail(errs, "0" * 64, "x.stp", p)
        self.assertEqual(out, "load failed")

    def test_empty_and_none_fall_back(self):
        p = os.path.join("root", "temp", "x.stp")
        for errs in (None, [], {}, "", 123):
            out = core._embed_error_detail(errs, "1" * 64, "x.stp", p)
            self.assertIn("embedding produced no rows", out)
            self.assertTrue(out)

    def test_matched_but_blank_reason_falls_back(self):
        p = os.path.join("root", "temp", "x.stp")
        errs = [{"file": p, "detail": "", "message": None}]
        out = core._embed_error_detail(errs, "2" * 64, "x.stp", p)
        self.assertTrue(out.strip())

    def test_matches_by_file_id(self):
        fid = "3" * 64
        errs = ["embedding failed for " + fid]
        out = core._embed_error_detail(errs, fid, "x.stp", os.path.join("root", "x.stp"))
        self.assertEqual(out, "embedding failed for " + fid)


class TestEmbedBodiesOrdering(unittest.TestCase):
    """Records must be written in the caller's file_ids order, not the order
    embed_shape_batch happens to finish its per-file workers in."""

    def setUp(self):
        import types

        self._saved_mods = {}
        for name in ("hoops_ai", "hoops_ai.ml", "hoops_ai.ml.embeddings"):
            self._saved_mods[name] = sys.modules.get(name)
        pkg = types.ModuleType("hoops_ai")
        ml = types.ModuleType("hoops_ai.ml")
        emb = types.ModuleType("hoops_ai.ml.embeddings")

        class _Embedding:
            def __init__(self, values, model, dim):
                self.values, self.model, self.dim = values, model, dim

        class _VectorRecord:
            def __init__(self, id, embedding, metadata):
                self.id, self.embedding, self.metadata = id, embedding, metadata

        emb.Embedding = _Embedding
        emb.VectorRecord = _VectorRecord
        pkg.ml = ml
        ml.embeddings = emb
        sys.modules["hoops_ai"] = pkg
        sys.modules["hoops_ai.ml"] = ml
        sys.modules["hoops_ai.ml.embeddings"] = emb

        self._tmp = tempfile.TemporaryDirectory()
        self._uploads = tempfile.TemporaryDirectory()
        self._orig_idx = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmp.name)

        self.fids = []
        self.paths = {}
        for i in range(4):
            fid = f"{i:064d}"
            p = pathlib.Path(self._uploads.name) / f"{fid}_file{i}.step"
            p.write_bytes(b"x")
            self.fids.append(fid)
            self.paths[fid] = p

        self._orig_get_embedder = embeddings.get_embedder
        self._orig_find = storage.find_persistent_CAD_file
        storage.find_persistent_CAD_file = lambda fid: self.paths[fid]

    def tearDown(self):
        embeddings.get_embedder = self._orig_get_embedder
        storage.find_persistent_CAD_file = self._orig_find
        config.INDEXES_DIR = self._orig_idx
        self._tmp.cleanup()
        self._uploads.cleanup()
        for name, mod in self._saved_mods.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    def _install_embedder(self, completion_order_bodies):
        """completion_order_bodies: list of (fid, n_bodies) in the (scrambled)
        order the batch reports them, i.e. worker completion order."""
        import numpy as np

        by_path = {str(p): fid for fid, p in self.paths.items()}

        def _embed_shape_batch(paths, show_progress=False, specifications=None, **kwargs):
            path_by_fid = {by_path[p]: p for p in paths}
            ids = []
            for fid, n in completion_order_bodies:
                ids.extend([path_by_fid[fid]] * n)
            values = np.arange(len(ids) * 4, dtype=np.float32).reshape(len(ids), 4) + 1.0
            meta = {"kind": ["part"] * len(ids)}
            return SimpleNamespace(ids=ids, values=values, metadata=meta)

        embedder = SimpleNamespace(
            model_name="fake-model", embed_shape_batch=_embed_shape_batch
        )
        embeddings.get_embedder = lambda: embedder

    def _record_file_order(self, records):
        order = []
        for r in records:
            if not order or order[-1] != r.id:
                order.append(r.id)
        return order

    def test_records_follow_file_ids_not_completion_order(self):
        # Workers finish in a scrambled order (2, 0, 3, 1) with varying body counts.
        self._install_embedder(
            [(self.fids[2], 1), (self.fids[0], 2), (self.fids[3], 1), (self.fids[1], 3)]
        )
        records, errors = index_registry._embed_bodies_for_index(self.fids, "idx")
        self.assertEqual(errors, [])
        # File order in the records is the input order, regardless of completion.
        self.assertEqual(self._record_file_order(records), self.fids)
        # Nothing dropped: 2 + 3 + 1 + 1 = 7 body rows total.
        self.assertEqual(len(records), 7)

    def test_a_different_completion_order_yields_the_same_record_order(self):
        self._install_embedder(
            [(self.fids[1], 3), (self.fids[3], 1), (self.fids[0], 2), (self.fids[2], 1)]
        )
        records, _ = index_registry._embed_bodies_for_index(self.fids, "idx")
        self.assertEqual(self._record_file_order(records), self.fids)

    def test_per_file_body_rows_keep_ascending_source_order(self):
        # File 0 has 3 bodies whose source rows must stay in ascending order so
        # the vectors are not permuted within a file.
        import numpy as np

        self._install_embedder([(self.fids[0], 3), (self.fids[1], 1)])
        records, _ = index_registry._embed_bodies_for_index(
            [self.fids[0], self.fids[1]], "idx"
        )
        file0 = [r for r in records if r.id == self.fids[0]]
        firsts = [float(np.asarray(r.embedding.values)[0]) for r in file0]
        self.assertEqual(firsts, sorted(firsts))


if __name__ == "__main__":
    unittest.main()

