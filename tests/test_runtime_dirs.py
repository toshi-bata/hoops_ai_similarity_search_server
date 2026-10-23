"""Runtime directory lifecycle: no startup wipe, TTL sweep for transient dirs.

The old lifespan rmtree'd ``uploads/``, ``out/`` and ``embeddings_cache/`` on
every start. ``uploads/`` is the CAD store holding the only copy of the payload
behind every registered index, so that deleted registered data; and a second
instance started on another port deleted the running instance's files. These
tests pin the replacement behaviour.
"""
import inspect
import os
import pathlib
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
import storage
import config
import main


class _EnvGuard(unittest.TestCase):
    _VARS = (
        "HOOPS_AI_OUT_TTL_HOURS",
        "HOOPS_AI_EMBEDDINGS_CACHE_TTL_DAYS",
        "HOOPS_AI_CAD_SHARED_DIR",
    )

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self._VARS}
        for k in self._VARS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestNoStartupWipe(unittest.TestCase):
    def test_lifespan_does_not_remove_trees(self):
        source = inspect.getsource(main.lifespan)
        self.assertNotIn("rmtree", source)

    def test_lifespan_runs_startup_maintenance(self):
        source = inspect.getsource(main.lifespan)
        self.assertIn("run_startup_maintenance", source)

    def test_startup_maintenance_never_touches_the_cad_store(self):
        source = inspect.getsource(core.run_startup_maintenance)
        self.assertNotIn("CAD_shared", source)
        self.assertNotIn("CAD_UPLOAD_DIR", source)


class TestSweepDirectory(_EnvGuard):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def _make(self, name, age_seconds):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        stamp = time.time() - age_seconds
        os.utime(path, (stamp, stamp))
        return path

    def test_old_files_are_removed(self):
        old = self._make("old.png", 7200)
        stats = core.sweep_directory(self.root, 3600)
        self.assertFalse(old.exists())
        self.assertEqual(stats["removed"], 1)

    def test_fresh_files_are_kept(self):
        fresh = self._make("fresh.png", 10)
        stats = core.sweep_directory(self.root, 3600)
        self.assertTrue(fresh.exists())
        self.assertEqual(stats["kept"], 1)
        self.assertEqual(stats["removed"], 0)

    def test_zero_ttl_disables_the_sweep(self):
        old = self._make("old.png", 999999)
        stats = core.sweep_directory(self.root, 0)
        self.assertTrue(old.exists())
        self.assertEqual(stats, {"removed": 0, "kept": 0, "failed": 0})

    def test_negative_ttl_disables_the_sweep(self):
        old = self._make("old.png", 999999)
        core.sweep_directory(self.root, -1)
        self.assertTrue(old.exists())

    def test_missing_directory_is_not_an_error(self):
        stats = core.sweep_directory(self.root / "nope", 3600)
        self.assertEqual(stats["removed"], 0)

    def test_nested_files_are_swept(self):
        old = self._make("sub/deep/old.scs", 7200)
        core.sweep_directory(self.root, 3600)
        self.assertFalse(old.exists())

    def test_emptied_directories_are_removed(self):
        self._make("sub/old.scs", 7200)
        core.sweep_directory(self.root, 3600)
        self.assertFalse((self.root / "sub").exists())

    def test_directories_with_survivors_are_kept(self):
        self._make("sub/old.scs", 7200)
        self._make("sub/fresh.scs", 10)
        core.sweep_directory(self.root, 3600)
        self.assertTrue((self.root / "sub").exists())

    def test_explicit_now_is_honoured(self):
        path = self._make("file.png", 0)
        core.sweep_directory(self.root, 3600, now=time.time() + 7200)
        self.assertFalse(path.exists())


class TestTtlSettings(_EnvGuard):
    def test_out_ttl_defaults_to_24h(self):
        self.assertEqual(core.viewer_output_ttl_seconds(), 24 * 3600)

    def test_out_ttl_is_configurable(self):
        os.environ["HOOPS_AI_OUT_TTL_HOURS"] = "2"
        self.assertEqual(core.viewer_output_ttl_seconds(), 7200)

    def test_out_ttl_zero_disables(self):
        os.environ["HOOPS_AI_OUT_TTL_HOURS"] = "0"
        self.assertEqual(core.viewer_output_ttl_seconds(), 0)

    def test_out_ttl_invalid_falls_back_to_default(self):
        os.environ["HOOPS_AI_OUT_TTL_HOURS"] = "not-a-number"
        self.assertEqual(core.viewer_output_ttl_seconds(), 24 * 3600)

    def test_embeddings_cache_is_kept_forever_by_default(self):
        self.assertEqual(core.embeddings_cache_ttl_seconds(), 0)

    def test_embeddings_cache_ttl_is_configurable(self):
        os.environ["HOOPS_AI_EMBEDDINGS_CACHE_TTL_DAYS"] = "3"
        self.assertEqual(core.embeddings_cache_ttl_seconds(), 3 * 86400)


class TestCadStorePathIsSingleSourced(_EnvGuard):
    def test_shared_dir_env_overrides_the_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["HOOPS_AI_CAD_SHARED_DIR"] = tmp
            self.assertEqual(
                storage.get_CAD_shared_dir(), pathlib.Path(tmp).resolve()
            )

    def test_default_is_the_uploads_directory(self):
        self.assertEqual(storage.get_CAD_shared_dir(), config.CAD_UPLOAD_DIR.resolve())

    def test_ensure_runtime_dirs_creates_the_configured_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pathlib.Path(tmp) / "cad-store"
            os.environ["HOOPS_AI_CAD_SHARED_DIR"] = str(store)
            core.ensure_runtime_dirs()
            self.assertTrue(store.is_dir())

    def test_ensure_runtime_dirs_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pathlib.Path(tmp) / "cad-store"
            os.environ["HOOPS_AI_CAD_SHARED_DIR"] = str(store)
            core.ensure_runtime_dirs()
            marker = store / "keep.txt"
            marker.write_text("keep")
            core.ensure_runtime_dirs()
            self.assertTrue(marker.exists())


class TestStartupMaintenance(_EnvGuard):
    def test_reports_both_transient_directories(self):
        os.environ["HOOPS_AI_OUT_TTL_HOURS"] = "0"
        os.environ["HOOPS_AI_EMBEDDINGS_CACHE_TTL_DAYS"] = "0"
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["HOOPS_AI_CAD_SHARED_DIR"] = str(pathlib.Path(tmp) / "store")
            result = core.run_startup_maintenance()
        self.assertEqual(set(result), {"out", "embeddings_cache"})

    def test_does_not_delete_files_in_the_cad_store(self):
        os.environ["HOOPS_AI_OUT_TTL_HOURS"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            store = pathlib.Path(tmp) / "store"
            store.mkdir()
            payload = store / ("a" * 64 + "_part.stp")
            payload.write_bytes(b"cad")
            stamp = time.time() - 999999
            os.utime(payload, (stamp, stamp))
            os.environ["HOOPS_AI_CAD_SHARED_DIR"] = str(store)
            core.run_startup_maintenance()
            self.assertTrue(payload.exists())


if __name__ == "__main__":
    unittest.main()
