"""Tests for configuration resolution in a redistributed (no-SDK) deployment.

All tests run without a HOOPS AI license or SDK install: they exercise the pure
resolution logic that decides where settings, the data root and model
checkpoints come from.
"""

import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core
import index_registry
import config


class ListIndexesWithoutSdk(unittest.TestCase):
    """list_indexes() must not raise when HOOPS_AI_SDK_DIR / FAISS vars are unset."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_idx = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmp.name)

    def tearDown(self):
        config.INDEXES_DIR = self._orig_idx
        self._tmp.cleanup()

    def test_no_default_entry_and_no_error(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("HOOPS_AI_SDK_DIR",)
        }
        with patch.dict(os.environ, env, clear=True), \
                patch.object(config, "read_env_file", return_value={}):
            result = index_registry.list_indexes()
        names = [entry["name"] for entry in result]
        self.assertNotIn("default", names)

    def test_sdk_dir_configured_false_when_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "HOOPS_AI_SDK_DIR"}
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(core.sdk_dir_configured())


class DataRootResolution(unittest.TestCase):
    """_resolve_data_root() must honour the documented precedence order."""

    def _env_without_data_dir(self):
        return {k: v for k, v in os.environ.items() if k != "HOOPS_AI_DATA_DIR"}

    def test_os_environ_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HOOPS_AI_DATA_DIR": tmp}, clear=False):
                root, source = config._resolve_data_root()
            self.assertEqual(root, pathlib.Path(tmp))
            self.assertIn("os.environ", source)

    def test_dotenv_used_when_env_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = pathlib.Path(tmp) / "chosen"
            env_file = pathlib.Path(tmp) / ".env"
            env_file.write_text(f"HOOPS_AI_DATA_DIR={data_dir}\n", encoding="utf-8")
            with patch.dict(os.environ, self._env_without_data_dir(), clear=True), \
                    patch.object(config, "ENV_FILE_PATH", env_file):
                root, source = config._resolve_data_root()
            self.assertEqual(root, data_dir)
            self.assertIn(".env", source)

    def test_default_is_app_root_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = pathlib.Path(tmp)
            # Even when APP_ROOT already holds an indexes/ directory, the default
            # is APP_ROOT/data -- there is no auto-detection that reuses it.
            (app / "indexes").mkdir()
            missing_env = app / ".env"
            with patch.dict(os.environ, self._env_without_data_dir(), clear=True), \
                    patch.object(config, "APP_ROOT", app), \
                    patch.object(config, "ENV_FILE_PATH", missing_env):
                root, source = config._resolve_data_root()
            self.assertEqual(root, app / "data")
            self.assertEqual(source, "APP_ROOT/data (default)")

    def test_default_is_app_root_data_without_existing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = pathlib.Path(tmp)
            missing_env = app / ".env"
            with patch.dict(os.environ, self._env_without_data_dir(), clear=True), \
                    patch.object(config, "APP_ROOT", app), \
                    patch.object(config, "ENV_FILE_PATH", missing_env):
                root, source = config._resolve_data_root()
            self.assertEqual(root, app / "data")
            self.assertEqual(source, "APP_ROOT/data (default)")

    def test_ensure_writable_passes_for_tmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(config, "DATA_ROOT", pathlib.Path(tmp)):
                config.ensure_data_root_writable()  # must not raise

    def test_ensure_writable_raises_when_not_writable(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = pathlib.Path(tmp) / "afile"
            blocker.write_bytes(b"")
            # A path whose parent is a regular file can never be created.
            bad = blocker / "sub"
            with patch.object(config, "DATA_ROOT", bad):
                with self.assertRaises(core.EnvConfigError):
                    config.ensure_data_root_writable()


class BundledCheckpointResolution(unittest.TestCase):
    """resolve_bundled_checkpoint() counts *.ckpt per directory, data root first."""

    def _dirs(self, tmp):
        data_models = pathlib.Path(tmp) / "data" / "models"
        app_root = pathlib.Path(tmp) / "app"
        app_models = app_root / "models"
        data_models.mkdir(parents=True)
        app_models.mkdir(parents=True)
        return data_models, app_root, app_models

    def _patch(self, data_models, app_root):
        return patch.multiple(config, MODELS_DIR=data_models, APP_ROOT=app_root)

    def test_app_default_used_when_data_root_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_models, app_root, app_models = self._dirs(tmp)
            (app_models / "best.ckpt").write_bytes(b"x")
            with self._patch(data_models, app_root):
                resolved = core.resolve_bundled_checkpoint()
            self.assertEqual(resolved, app_models / "best.ckpt")

    def test_data_root_shadows_app_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_models, app_root, app_models = self._dirs(tmp)
            (data_models / "best.ckpt").write_bytes(b"x")
            (app_models / "best.ckpt").write_bytes(b"y")
            with self._patch(data_models, app_root):
                resolved = core.resolve_bundled_checkpoint()
            self.assertEqual(resolved, data_models / "best.ckpt")

    def test_two_ckpts_in_one_dir_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_models, app_root, _ = self._dirs(tmp)
            (data_models / "a.ckpt").write_bytes(b"x")
            (data_models / "b.ckpt").write_bytes(b"y")
            with self._patch(data_models, app_root):
                with self.assertRaises(core.ModelResolutionError):
                    core.resolve_bundled_checkpoint()

    def test_one_in_each_dir_is_not_an_error(self):
        # A fine-tuned model in the data root plus the bundled default in the app
        # is the normal state, not an ambiguity: per-directory counting picks the
        # data-root model.
        with tempfile.TemporaryDirectory() as tmp:
            data_models, app_root, app_models = self._dirs(tmp)
            (data_models / "mine.ckpt").write_bytes(b"x")
            (app_models / "best.ckpt").write_bytes(b"y")
            with self._patch(data_models, app_root):
                resolved = core.resolve_bundled_checkpoint()
            self.assertEqual(resolved, data_models / "mine.ckpt")

    def test_no_ckpt_anywhere_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_models, app_root, _ = self._dirs(tmp)
            with self._patch(data_models, app_root):
                with self.assertRaises(core.ModelResolutionError):
                    core.resolve_bundled_checkpoint()


class NamedCheckpointResolution(unittest.TestCase):
    """resolve_checkpoint_path() honours HOOPS_AI_EMBEDDINGS_MODEL_NAME."""

    def setUp(self):
        # Never read the on-disk .env: the checkpoint name is set per test.
        self._patch_env = patch.object(config, "load_env_file", lambda *a, **k: None)
        self._patch_env.start()

    def tearDown(self):
        self._patch_env.stop()

    def _dirs(self, tmp):
        data_models = pathlib.Path(tmp) / "data" / "models"
        app_root = pathlib.Path(tmp) / "app"
        app_models = app_root / "models"
        sdk = pathlib.Path(tmp) / "sdk"
        sdk_models = sdk / "packages" / "trained_ml_models"
        for d in (data_models, app_models, sdk_models):
            d.mkdir(parents=True)
        return data_models, app_root, app_models, sdk, sdk_models

    def test_named_file_resolves_from_data_root_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_models, app_root, app_models, _sdk, _ = self._dirs(tmp)
            (data_models / "m.ckpt").write_bytes(b"x")
            (app_models / "m.ckpt").write_bytes(b"y")
            with patch.multiple(config, MODELS_DIR=data_models, APP_ROOT=app_root), \
                 patch.dict(os.environ, {"HOOPS_AI_EMBEDDINGS_MODEL_NAME": "m.ckpt"}, clear=False):
                os.environ.pop("HOOPS_AI_SDK_DIR", None)
                resolved = core.resolve_checkpoint_path()
            self.assertEqual(resolved, data_models / "m.ckpt")

    def test_named_file_falls_through_to_app_then_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            _data, app_root, _app_models, sdk, sdk_models = self._dirs(tmp)
            (sdk_models / "m.ckpt").write_bytes(b"z")
            with patch.multiple(config, MODELS_DIR=pathlib.Path(tmp) / "data" / "models",
                                APP_ROOT=app_root), \
                 patch.dict(os.environ, {
                     "HOOPS_AI_EMBEDDINGS_MODEL_NAME": "m.ckpt",
                     "HOOPS_AI_SDK_DIR": str(sdk),
                 }, clear=False):
                resolved = core.resolve_checkpoint_path()
            self.assertEqual(resolved, sdk_models / "m.ckpt")

    def test_named_file_ignores_other_ckpts_in_the_same_dir(self):
        # A tutorial folder holding several checkpoints must not be an error when
        # the file is named explicitly.
        with tempfile.TemporaryDirectory() as tmp:
            data_models, app_root, _app_models, _sdk, _ = self._dirs(tmp)
            (data_models / "wanted.ckpt").write_bytes(b"x")
            (data_models / "other1.ckpt").write_bytes(b"y")
            (data_models / "other2.ckpt").write_bytes(b"z")
            with patch.multiple(config, MODELS_DIR=data_models, APP_ROOT=app_root), \
                 patch.dict(os.environ, {"HOOPS_AI_EMBEDDINGS_MODEL_NAME": "wanted.ckpt"}, clear=False):
                os.environ.pop("HOOPS_AI_SDK_DIR", None)
                resolved = core.resolve_checkpoint_path()
            self.assertEqual(resolved, data_models / "wanted.ckpt")

    def test_named_file_missing_lists_all_locations(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_models, app_root, _app_models, sdk, _ = self._dirs(tmp)
            with patch.multiple(config, MODELS_DIR=data_models, APP_ROOT=app_root), \
                 patch.dict(os.environ, {
                     "HOOPS_AI_EMBEDDINGS_MODEL_NAME": "nope.ckpt",
                     "HOOPS_AI_SDK_DIR": str(sdk),
                 }, clear=False):
                with self.assertRaises(core.ModelResolutionError) as ctx:
                    core.resolve_checkpoint_path()
            msg = str(ctx.exception)
            self.assertIn(str(data_models / "nope.ckpt"), msg)
            self.assertIn("trained_ml_models", msg)

    def test_unset_name_delegates_to_bundled_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_models, app_root, app_models, _sdk, _ = self._dirs(tmp)
            (app_models / "best.ckpt").write_bytes(b"x")
            with patch.multiple(config, MODELS_DIR=data_models, APP_ROOT=app_root), \
                 patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HOOPS_AI_EMBEDDINGS_MODEL_NAME", None)
                resolved = core.resolve_checkpoint_path()
            self.assertEqual(resolved, app_models / "best.ckpt")


class IndexModelConsistency(unittest.TestCase):
    """check_index_model_consistency() classifies match / partial / mismatch."""

    def _run(self, stamped_digests, current="CURRENT"):
        from types import SimpleNamespace

        metas = []
        for i, dg in enumerate(stamped_digests):
            m = {"file_id": f"{i:064d}", "kind": "part"}
            if dg is not None:
                m["_ckpt_sha256"] = dg
            metas.append(m)
        vs = SimpleNamespace(iter_metadata=lambda: iter(metas))
        with patch.object(index_registry, "resolve_checkpoint_path",
                          return_value=pathlib.Path("x.ckpt")), \
             patch.object(index_registry, "ckpt_sha256", return_value=current), \
             patch.object(index_registry, "_load_named_index", return_value=vs):
            return core.check_index_model_consistency("demo")

    def test_all_match(self):
        out = self._run(["CURRENT", "CURRENT"])
        self.assertEqual(out["status"], "match")
        self.assertEqual(out["matching_files"], 2)
        self.assertEqual(out["mismatching_files"], 0)
        self.assertEqual(out["current_ckpt_sha256"], "CURRENT")

    def test_partial_mismatch(self):
        out = self._run(["CURRENT", "OLD"])
        self.assertEqual(out["status"], "partial_mismatch")
        self.assertEqual(out["matching_files"], 1)
        self.assertEqual(out["mismatching_files"], 1)

    def test_full_mismatch(self):
        out = self._run(["OLD", "OLD"])
        self.assertEqual(out["status"], "mismatch")
        self.assertEqual(out["matching_files"], 0)
        self.assertEqual(out["mismatching_files"], 2)

    def test_unknown_digests_are_a_mismatch(self):
        out = self._run([None, None])
        self.assertEqual(out["status"], "mismatch")
        self.assertEqual(out["unknown_files"], 2)
        self.assertEqual(out["matching_files"], 0)


class CheckpointIdentityRecording(unittest.TestCase):
    """Model identity is recorded (record-only) without changing responses."""

    def test_ckpt_sha256_is_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = pathlib.Path(tmp) / "best.ckpt"
            ckpt.write_bytes(b"hello world")
            core._ckpt_sha256_cache.pop(str(ckpt), None)
            first = core.ckpt_sha256(ckpt)
            self.assertIn(str(ckpt), core._ckpt_sha256_cache)
            # Mutate the file: the cached digest must be returned unchanged.
            ckpt.write_bytes(b"different bytes")
            self.assertEqual(core.ckpt_sha256(ckpt), first)
            core._ckpt_sha256_cache.pop(str(ckpt), None)

    def test_save_index_schema_records_only_version(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(config, "INDEXES_DIR", pathlib.Path(tmp)):
                core._save_index_schema("idx", 2)
                data = json.loads(
                    (pathlib.Path(tmp) / "idx" / "index.json").read_text(encoding="utf-8")
                )
        self.assertEqual(data["schema_version"], 2)
        self.assertEqual(set(data.keys()), {"schema_version"})

    def test_ckpt_sha_key_is_excluded_from_public_metadata(self):
        public = core._public_metadata(
            {"file_id": "x", "kind": "part", "_ckpt_sha256": "deadbeef"}
        )
        self.assertNotIn("_ckpt_sha256", public)
        self.assertIn("file_id", public)


class RedistEnvTemplate(unittest.TestCase):
    """The redistributable .env template must never carry a license key."""

    def test_no_license_line(self):
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        template = repo_root / "tools" / "redist.env.example"
        self.assertTrue(template.exists())
        import re

        for line in template.read_text(encoding="utf-8").splitlines():
            self.assertIsNone(
                re.match(r"\s*HOOPS_AI_LICENSE\s*=", line),
                f"license key must not appear in {template.name}: {line!r}",
            )


class LicenseShimVersionParsing(unittest.TestCase):
    """Parsing helpers used to tell a version mismatch from a dev checkout."""

    def test_expected_python_parsed_from_abi_tag(self):
        p = pathlib.Path("hoops_license_shim.cp312-win_amd64.pyd")
        self.assertEqual(core._shim_expected_python(p), "3.12")

    def test_expected_python_none_for_pure_python(self):
        p = pathlib.Path("hoops_license_shim.py")
        self.assertIsNone(core._shim_expected_python(p))


if __name__ == "__main__":
    unittest.main()