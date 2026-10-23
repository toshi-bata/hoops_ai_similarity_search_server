"""Pin the core facade re-exports to the modules that now own the objects.

core.py used to define the whole lock hierarchy itself. The refactoring moves it
into locks.py, but external callers (routers/, tools/, tests/) still reach the
locks as ``core.<name>``. That only stays correct while ``core.<name>`` is the
*same object* as ``locks.<name>`` -- not merely an equal copy. A re-export made
by value (for example rebinding ``core.<name>`` to a fresh object) would pass an
equality check yet silently split the two modules' views of the lock, so these
tests assert identity (``is``), not equality.

They also guard the per-key lock factories: asking for the same key twice must
return the very same lock, or two callers would serialise on different objects
and the "one build per subject" guarantee would be lost.

The same file now also pins the config.py split (ConfigFacade) and the storage.py
split (StorageFacade): the swappable path/env and CAD-store names must be absent
from core, while the stable objects (exception classes, constants, ingest
helpers) stay re-exported by identity.

Finally it pins the embeddings.py split (EmbeddingsFacade) and the diagnostics.py
split (DiagnosticsFacade). The six encoder/cache names tests swap and the three
singletons rebound with ``global`` must be absent from core; the stable model
constants, SDK/checkpoint resolvers and intra-op helpers stay re-exported by
identity. ``_search_timing_enabled`` (now config) and ``_PhaseTimer`` (now
diagnostics) must likewise be gone from core.

It finally pins the index_registry.py split (IndexRegistryFacade): the fifteen
named-index/registry names tests swap must be absent from core, while the stable
constants, path helpers, public-index API and schema helpers stay re-exported by
identity. ``delete_index`` and ``remove_from_index`` are not in either list here
because they still *live* in core (they orchestrate sibling-domain cleanup).

It finally pins the tags.py split (TagsFacade): ``_read_tags_document`` is the one
swappable name tests replace to prove ``core._search_cache_key`` reads the tag
sidecar, so it must be absent from core; every other tag name stays re-exported by
identity. ``list_index_parts`` is not listed here because it still lives in core.

It finally pins the search.py split (SearchFacade) and the context.py split
(ContextFacade). ``search_index`` and ``_search_cache_enabled`` are the two
swappable names -- routers reach them as ``search.<name>`` and tests swap those
same attributes -- so both must be absent from core; the ten stable search-cache
names stay re-exported by identity. Every context-prediction name is stable and
stays re-exported by identity. ``context`` reads ``_SEARCH_KINDS`` from ``search``
(a single owner for the accepted ``kind`` set), which this file also pins by
identity.

It finally pins the assembly.py split (AssemblyFacade). ``_get_assembly_matcher``,
``prewarm_assembly_matcher`` and ``search_assembly_index`` are the three swappable
names -- routers and tests reach them as ``assembly.<name>`` and swap those same
attributes -- so all three must be absent from core; the nineteen stable
assembly-cache and helper names stay re-exported by identity. This file also
asserts ``assembly`` never binds the ``tags`` module (``import tags``), because
``search_assembly_index`` has a ``tags`` parameter that such an import would let
shadow the module -- the #61 footgun.

It finally pins the shape_map.py split (ShapeMapFacade). ``_load_index_body_batch``,
``_shape_map_filemeta``, ``compute_index_shape_map`` and
``compute_index_shape_map_scs`` are the four swappable names -- routers and tests
reach them as ``shape_map.<name>`` and swap those same attributes -- so all four
must be absent from core; the forty stable state, numeric-map and scs-colouring
names stay re-exported by identity. This file also asserts ``shape_map`` never
binds the ``tags`` module (``import tags``), because ``compute_index_shape_map``
and ``compute_index_shape_map_scs`` have a ``tags`` parameter that such an import
would let shadow the module -- the #61 footgun.

It finally pins the last five splits: licensing.py (LicensingFacade),
viewer.py (ViewerFacade), warmup.py (WarmupFacade), jobs.py (JobsFacade) and
runtime.py (RuntimeFacade). ``_find_license_shim_file`` (swapped by the
license tests) and the job-store singleton ``get_index_job_store`` /
``_index_job_store`` (swapped and, for the latter, rebound with ``global``)
must be absent from core; every other name in those five modules stays
re-exported by identity. The per-module ``does_not_import_core`` checks are
replaced by a single ``ImportHygiene`` scan that walks every first-class
module, so modules added by later refactors are covered automatically.
"""

import pathlib
import sys
import unittest
import ast
import os

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import core
import config
import storage
import locks
import embeddings
import diagnostics
import index_registry
import tags
import search
import context
import assembly
import shape_map
import licensing
import viewer
import warmup
import jobs
import runtime

# Names that moved from core.py into locks.py and are re-exported through core.
_MOVED_NAMES = (
    "_PROGRESS_STDERR_LOCK",
    "_index_locks",
    "_index_locks_mutex",
    "_get_index_lock",
    "_tags_locks",
    "_tags_locks_mutex",
    "_get_tags_lock",
    "_assembly_build_locks",
    "_assembly_build_locks_mutex",
    "_get_assembly_build_lock",
    "_shape_map_build_locks",
    "_shape_map_build_locks_mutex",
    "_scs_map_build_locks",
    "_scs_map_build_locks_mutex",
)


class FacadeReexportIdentity(unittest.TestCase):
    """core.<name> must be the same object as locks.<name>, not a copy."""

    def test_moved_names_are_the_same_object(self):
        for name in _MOVED_NAMES:
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(locks, name))


class LockFactoryIdentity(unittest.TestCase):
    """The per-key lock factories must return the same lock for the same key."""

    def test_get_index_lock_is_idempotent(self):
        self.assertIs(core._get_index_lock("layout-test"),
                      core._get_index_lock("layout-test"))

    def test_get_tags_lock_is_idempotent(self):
        self.assertIs(core._get_tags_lock("layout-test"),
                      core._get_tags_lock("layout-test"))

    def test_get_assembly_build_lock_is_idempotent(self):
        self.assertIs(core._get_assembly_build_lock("layout-test"),
                      core._get_assembly_build_lock("layout-test"))


# Configuration names that moved from core.py into config.py.
#
# The path constants and .env helpers are NOT re-exported through core: tests and
# tools swap them to redirect the data root, and that only works if every reader
# reaches them as ``config.<name>``. If core re-exported (say) INDEXES_DIR, a
# stale ``core.INDEXES_DIR`` would keep pointing at the original object while
# ``config.INDEXES_DIR`` was swapped, so the two modules would disagree about the
# data root. Asserting these names are *absent* from core proves nobody can read
# the stale copy. See "The re-export convention" in docs/architecture.md.
_CONFIG_NAMES_NOT_ON_CORE = (
    "APP_ROOT",
    "ENV_FILE_PATH",
    "DATA_ROOT",
    "_DATA_ROOT_SOURCE",
    "CAD_UPLOAD_DIR",
    "CAD_VIEWER_OUTPUT_DIR",
    "EMBEDDINGS_CACHE_DIR",
    "INDEXES_DIR",
    "JOBS_DIR",
    "MODELS_DIR",
    "SCS_MAP_DIR",
    "TAG_VOCABULARY_FILE",
    "read_env_file",
    "load_env_file",
    "get_required_file_env",
    "ensure_data_root_writable",
    "_resolve_data_root",
)

# The exception classes are the one exception: they are never swapped, and
# routers/ catch them as ``core.EnvConfigError`` / ``core.PathConfigError``, so
# core re-exports the very same class objects config defines.
_CONFIG_REEXPORTS = (
    "EnvConfigError",
    "PathConfigError",
)


class ConfigFacade(unittest.TestCase):
    """Pin the config.py split: swappable names off core, exceptions re-exported.

    This guards the failure mode from "The re-export convention" in
    docs/architecture.md: if a moved path/env name were still reachable as
    ``core.<name>``, a test swapping
    ``config.<name>`` would silently leave the stale core copy in place.
    """

    def test_swappable_names_are_not_reexported_by_core(self):
        for name in _CONFIG_NAMES_NOT_ON_CORE:
            with self.subTest(name=name):
                self.assertFalse(
                    hasattr(core, name),
                    f"core.{name} must not exist; read it as config.{name}",
                )

    def test_exception_classes_are_reexported_by_identity(self):
        for name in _CONFIG_REEXPORTS:
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(config, name))

    def test_env_int_is_reexported_from_config(self):
        # ``_env_int`` moved to config.py alongside the other .env helpers so
        # that storage.py could reuse it without importing core. It is never
        # swapped, so core imports it by value and both modules must see the
        # very same function object.
        self.assertIs(core._env_int, config._env_int)

    def test_demo_enabled_is_reexported_from_config(self):
        # ``demo_enabled`` reads HOOPS_AI_DEMO on every call, so a test sets the
        # variable instead of swapping the function; that makes it a stable (C)
        # name, re-exported by identity for ``routers/client_config.py``.
        self.assertIs(core.demo_enabled, config.demo_enabled)


# CAD file storage and ingest names that moved from core.py into storage.py.
#
# These four are swapped by tests to stub the CAD store, so they are NOT
# re-exported through core: a reader must reach them as ``storage.<name>``. If core
# re-exported (say) find_persistent_CAD_file, a test swapping
# ``storage.find_persistent_CAD_file`` would leave a stale ``core`` copy behind and
# the two modules would disagree about how a file_id resolves. Asserting these
# names are *absent* from core proves nobody can read the stale copy. See
# "The re-export convention" in docs/architecture.md.
_STORAGE_NAMES_NOT_ON_CORE = (
    "find_persistent_CAD_file",
    "get_CAD_shared_dir",
    "read_too_heavy_files",
    "upload_CAD_file_persistent",
)

# The rest of the module is stable -- never swapped -- so core re-exports the very
# same objects storage.py defines: existing ``core.<name>`` and ``except core.Zip*``
# call sites keep working unchanged.
_STORAGE_REEXPORTS = (
    "CAD_ALLOWED_EXTENSIONS",
    "ZipSlipError",
    "ZipSizeLimitError",
    "ZipFileLimitError",
    "UploadSizeLimitError",
    "_UPLOAD_CHUNK_BYTES",
    "_MAX_UPLOAD_BYTES_DEFAULT",
    "_FILE_ID_RE",
    "_EXISTS_MAX_IDS_DEFAULT",
    "_max_upload_bytes",
    "exists_max_ids",
    "known_persistent_CAD_file_ids",
    "check_persistent_CAD_files",
    "map_persistent_CAD_files",
    "delete_persistent_CAD_file",
    "list_persistent_CAD_files",
    "server_paths_enabled",
    "resolve_server_paths",
    "ZIP_MAX_TOTAL_BYTES",
    "ZIP_MAX_FILES",
    "zip_max_total_bytes",
    "zip_max_files",
    "_BytesUploadFile",
    "extract_zip_cad_files",
    "get_shared_CAD_file",
)


class StorageFacade(unittest.TestCase):
    """Pin the storage.py split: swappable names off core, stable names re-exported.

    Two assertions guard the same failure mode from "The re-export convention"
    in docs/architecture.md from opposite sides. The four store-access functions
    must be *absent*
    from core so no reader can hold a stale copy while a test swaps
    ``storage.<name>``. Everything else must be the *same object* on both modules
    so ``core.<name>`` and ``storage.<name>`` can never drift apart.
    """

    def test_swappable_names_are_not_reexported_by_core(self):
        for name in _STORAGE_NAMES_NOT_ON_CORE:
            with self.subTest(name=name):
                self.assertFalse(
                    hasattr(core, name),
                    f"core.{name} must not exist; read it as storage.{name}",
                )

    def test_stable_names_are_reexported_by_identity(self):
        for name in _STORAGE_REEXPORTS:
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(storage, name))


# Encoder and embedding-cache names that moved from core.py into embeddings.py.
#
# Tests swap these six to stub the encoder or reset the two-tier cache, so they
# are NOT re-exported through core: a reader must reach them as
# ``embeddings.<name>``. If core re-exported (say) get_embedder, a test swapping
# ``embeddings.get_embedder`` would leave a stale ``core`` copy behind. The
# singleton below is rebound in place with ``global`` (``_embedder`` is built
# lazily); re-exporting a rebound name would freeze ``core.<name>`` at its
# import-time value while embeddings moved on, so it too must be absent. See
# "The re-export convention" in docs/architecture.md.
_EMBEDDINGS_NAMES_NOT_ON_CORE = (
    # (A) swapped by tests
    "get_embedder",
    "compute_embedding",
    "compare_embeddings",
    "_embed_cache_max_bytes",
    "_embedding_memory_cache",
    "_embedding_cache_bytes",
    # (B) rebound with ``global``
    "_embedder",
)

# The rest of the embedding surface is stable -- never swapped, never rebound --
# so core re-exports the very same objects embeddings.py defines: existing
# ``core.<name>`` call sites in routers/, tools/ and tests keep working.
_EMBEDDINGS_REEXPORTS = (
    "_EMBED_CACHE_MAX_MB_DEFAULT",
    "_embedding_cache_mutex",
    "_embed_entry_bytes",
    "_embedding_cache_get",
    "_embedding_cache_put",
    "_l2_normalize",
    "clear_embedding_memory_cache",
    "log_compute_environment",
    "get_required_env",
    "require_path",
    "get_sdk_dir",
    "sdk_dir_configured",
    "get_packages_dir",
    "ModelResolutionError",
    "resolve_bundled_checkpoint",
    "resolve_checkpoint_path",
    "ckpt_sha256",
    "_ckpt_sha256_cache",
    "_INTRAOP_ENV_VARS",
    "_INTRAOP_THREADS_DEFAULT",
    "intraop_threads",
    "intraop_thread_limit",
)


class EmbeddingsFacade(unittest.TestCase):
    """Pin the embeddings.py split: swappable/rebound names off core, stable re-exported.

    The eight names in ``_EMBEDDINGS_NAMES_NOT_ON_CORE`` must be *absent* from core
    so no reader can hold a stale copy while a test swaps ``embeddings.<name>`` or
    while embeddings rebinds a ``global``. Everything else must be the *same
    object* on both modules so ``core.<name>`` and ``embeddings.<name>`` can never
    drift apart. See "The re-export convention" in docs/architecture.md.
    """

    def test_swapped_and_rebound_names_are_not_reexported_by_core(self):
            for name in _EMBEDDINGS_NAMES_NOT_ON_CORE:
                with self.subTest(name=name):
                    self.assertFalse(
                        hasattr(core, name),
                        f"core.{name} must not exist; read it as embeddings.{name}",
                    )

    def test_stable_names_are_reexported_by_identity(self):
            for name in _EMBEDDINGS_REEXPORTS:
                with self.subTest(name=name):
                    self.assertIs(getattr(core, name), getattr(embeddings, name))


class DiagnosticsFacade(unittest.TestCase):
    """Pin the timing-helper split: ``_search_timing_enabled``/``_PhaseTimer`` off core.

    The two request-timing helpers left core in different directions:
    ``_search_timing_enabled`` is an env flag that now lives beside the other
    .env helpers in config.py, and ``_PhaseTimer`` is a dependency-free stopwatch
    that now lives in the leaf module diagnostics.py. Neither is swapped by tests,
    so core does not re-export them; callers reach them as
    ``config._search_timing_enabled`` / ``diagnostics._PhaseTimer``.
    """

    def test_timing_helpers_are_not_on_core(self):
            for name in ("_search_timing_enabled", "_PhaseTimer"):
                with self.subTest(name=name):
                    self.assertFalse(
                        hasattr(core, name),
                        f"core.{name} must not exist after the timing-helper split",
                    )

    def test_timing_helpers_live_on_their_owners(self):
            self.assertTrue(hasattr(config, "_search_timing_enabled"))
            self.assertTrue(hasattr(diagnostics, "_PhaseTimer"))

    def test_diagnostics_is_a_leaf_module(self):
            source = pathlib.Path(diagnostics.__file__).read_text(encoding="utf-8")
            for line in source.splitlines():
                stripped = line.strip()
                for forbidden in ("core", "config", "embeddings"):
                    self.assertNotEqual(stripped, f"import {forbidden}")
                    self.assertFalse(stripped.startswith(f"from {forbidden} "))
                    self.assertFalse(stripped.startswith(f"import {forbidden} "))


# Named-index registry names that moved from core.py into index_registry.py.
#
# Tests swap these fifteen to stub the index store, the searcher cache, the
# schema/model sidecars or the body-level embedding pass, so they are NOT
# re-exported through core: a reader must reach them as
# ``index_registry.<name>``. If core re-exported (say) _load_named_index, a test
# swapping ``index_registry._load_named_index`` would leave a stale ``core`` copy
# behind and the two modules would disagree about what the index holds.
# ``add_to_index`` calls ``_embed_bodies_for_index`` by bare name, so patching
# the core copy would miss the internal call -- that is why
# ``_embed_bodies_for_index`` belongs here and not in the re-export list. No name
# in this cluster is rebound with ``global`` (the (B) set is empty). Asserting
# these names are *absent* from core proves nobody can read the stale copy. See
# "The re-export convention" in docs/architecture.md.
_INDEX_REGISTRY_NAMES_NOT_ON_CORE = (
    "_load_named_index",
    "_get_named_searcher",
    "_load_index_schema",
    "_resolve_index_asset",
    "add_to_index",
    "_save_named_index_atomic",
    "_registered_index_ids",
    "_named_indexes",
    "_index_faiss_path",
    "_index_cache_max_bytes",
    "_index_entry_bytes",
    "list_indexes",
    "_load_shape_index_pickle_compat",
    "_embed_bodies_for_index",
)

# The rest of the registry surface is stable -- never swapped, never rebound --
# so core re-exports the very same objects index_registry.py defines: existing
# ``core.<name>`` call sites in routers/, tools/ and tests keep working, and the
# orchestration operators ``delete_index`` / ``remove_from_index`` that stay in
# core reach the registry through these names.
_INDEX_REGISTRY_REEXPORTS = (
    "_INDEX_SCHEMA_VERSION",
    "_INDEX_NAME_RE",
    "_INDEX_CACHE_MAX_MB_DEFAULT",
    "_named_indexes_mutex",
    "_NAMED_SEARCHER_CACHE_MAX",
    "_named_searchers",
    "_validate_index_name",
    "_index_base_path",
    "_get_embedder_dim",
    "_evict_named_indexes_to_cap",
    "_store_named_index",
    "create_index",
    "_EMBED_ERROR_DEFAULT",
    "_embed_error_detail",
    "_MAX_WORKERS_DEFAULT",
    "_MODEL_FOOTPRINT_MB_DEFAULT",
    "_MIN_FILES_PARALLEL_DEFAULT",
    "_HOST_RAM_RESERVE_BYTES",
    "compute_auto_workers",
    "embed_time_limit",
    "read_error_summary_reasons",
    "_norm_path_key",
    "build_embed_specifications",
    "_TqdmProgressStderr",
    "capture_embed_progress",
    "_evict_named_searchers",
    "_public_metadata",
    "_flatten_body_hit_objects",
    "_flatten_body_hits",
    "_TAG_FILTER_OVERFETCH",
    "_TAG_FILTER_OVERFETCH_MAX",
    "_tag_filtered_fetch_k",
    "get_index_stats",
    "check_index_model_consistency",
    "_require_index_exists",
    "_index_thumbnails_dir",
    "_index_model_sidecar_path",
    "_index_json_path",
    "_index_scs_dir",
    "_save_index_schema",
    "_obb_meta_value",
    "_compute_faiss_stats",
    "SchemaVersionError",
)


class IndexRegistryFacade(unittest.TestCase):
    """Pin the index_registry.py split: swappable names off core, stable re-exported.

    The fourteen names in ``_INDEX_REGISTRY_NAMES_NOT_ON_CORE`` must be *absent*
    from core so no reader can hold a stale copy while a test swaps
    ``index_registry.<name>``. Everything else must be the *same object* on both
    modules so ``core.<name>`` and ``index_registry.<name>`` can never drift
    apart. See "The re-export convention" in docs/architecture.md.
    """

    def test_swappable_names_are_not_reexported_by_core(self):
        for name in _INDEX_REGISTRY_NAMES_NOT_ON_CORE:
            with self.subTest(name=name):
                self.assertFalse(
                    hasattr(core, name),
                    f"core.{name} must not exist; read it as index_registry.{name}",
                )

    def test_stable_names_are_reexported_by_identity(self):
        for name in _INDEX_REGISTRY_REEXPORTS:
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(index_registry, name))


# Tag sidecar names that moved from core.py into tags.py.
#
# ``_read_tags_document`` is the one swapped name: tests replace
# ``tags._read_tags_document`` to prove ``core._search_cache_key`` reads the tag
# sidecar exactly once per predicate. If core re-exported it, the swap would not
# reach the internal call and a stale search result could be served after a tag
# edit -- so it must be *absent* from core. No name in this cluster is rebound
# with ``global`` (the (B) set is empty). See "The re-export convention" in
# docs/architecture.md.
_TAGS_NAMES_NOT_ON_CORE = (
    "_read_tags_document",
)

# The rest of the tag surface is stable -- never swapped, never rebound -- so
# core re-exports the very same objects tags.py defines: existing ``core.<name>``
# call sites in routers/, tools/ and tests keep working, and the orchestrators
# ``list_index_parts`` / ``delete_index`` / ``remove_from_index`` that stay in
# core reach the tag cluster through these names.
_TAGS_REEXPORTS = (
    "_TAGS_SCHEMA_VERSION",
    "_TAG_MAX_LENGTH",
    "_TAGS_PER_PART_DEFAULT",
    "_TAGS_CACHE_MAX",
    "_tags_cache",
    "_tags_cache_mutex",
    "TagsFileError",
    "TagsPreconditionFailed",
    "_index_tags_path",
    "max_tags_per_part",
    "normalize_tag",
    "_normalize_tag_list",
    "_warn_on_case_collision",
    "_iter_all_tags",
    "_empty_tags_document",
    "_serialize_tags_document",
    "_tags_etag",
    "_parse_tags_document",
    "_write_tags_document",
    "_evict_tags_cache",
    "etag_matches",
    "_mutate_tags",
    "_stamp",
    "get_index_tags",
    "get_part_tags",
    "set_part_tags",
    "_bulk_tag_update",
    "assign_tag",
    "unassign_tag",
    "delete_tag",
    "prune_tags",
    "tags_by_part",
    "_shape_map_tags_snapshot",
    "normalize_tag_filter",
    "tag_filter_predicate",
    "tag_filter_diagnostics",
    "_TAG_VOCABULARY_MAX",
    "_tag_vocabulary_path",
    "_read_tag_vocabulary",
    "_write_tag_vocabulary",
    "get_tag_vocabulary",
    "add_tag_vocabulary",
    "remove_tag_vocabulary",
    "_record_tags_in_vocabulary",
)


class TagsFacade(unittest.TestCase):
    """Pin the tags.py split: the swapped name off core, stable names re-exported.

    ``_read_tags_document`` must be *absent* from core so no reader can hold a
    stale copy while a test swaps ``tags._read_tags_document`` (the search-cache
    key reads it). Everything else must be the *same object* on both modules so
    ``core.<name>`` and ``tags.<name>`` can never drift apart. See "The
    re-export convention" in docs/architecture.md.
    """

    def test_swappable_names_are_not_reexported_by_core(self):
        for name in _TAGS_NAMES_NOT_ON_CORE:
            with self.subTest(name=name):
                self.assertFalse(
                    hasattr(core, name),
                    f"core.{name} must not exist; read it as tags.{name}",
                )

    def test_stable_names_are_reexported_by_identity(self):
        for name in _TAGS_REEXPORTS:
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(tags, name))


# Search names that moved from core.py into search.py.
#
# ``search_index`` and ``_search_cache_enabled`` are the two swapped names:
# routers reach them as ``search.<name>`` and tests replace those same attributes
# to install fake searchers and to force the response cache on/off. If core
# re-exported them, the swap and the module's own internal use would diverge --
# the double-truth trap -- so they must be *absent* from core. No name in this
# cluster is rebound with ``global`` (the (B) set is empty). See "The re-export
# convention" in docs/architecture.md.
_SEARCH_NAMES_NOT_ON_CORE = (
    "search_index",
    "_search_cache_enabled",
)

# The rest of the search surface is stable -- never swapped, never rebound -- so
# core re-exports the very same objects search.py defines for existing
# ``core.<name>`` call sites in routers/, tools/ and tests.
_SEARCH_REEXPORTS = (
    "_SEARCH_KINDS",
    "_SEARCH_CACHE_MAX_DEFAULT",
    "_search_cache",
    "_search_cache_mutex",
    "_search_cache_capacity",
    "clear_search_cache",
    "_tag_filter_key",
    "_search_cache_key",
    "_search_cache_get",
    "_search_cache_put",
)

# Context-prediction names that moved from core.py into context.py. Every name is
# stable (never swapped, never rebound), so core re-exports the same objects for
# existing ``core.<name>`` call sites and for ``remove_from_index`` in core,
# which calls ``prune_context`` downward.
_CONTEXT_REEXPORTS = (
    "_PREDICT_KEYS_DEFAULT",
    "max_predict_keys",
    "_normalize_predict_keys",
    "_context_jsonify",
    "predict_context",
    "get_index_context_summary",
    "get_part_context",
    "set_part_context",
    "prune_context",
)


class SearchFacade(unittest.TestCase):
    """Pin the search.py split: the swapped names off core, stable names re-exported.

    ``search_index`` and ``_search_cache_enabled`` must be *absent* from core so a
    test swapping ``core.<name>`` cannot leave the module using a second object.
    Everything else must be the *same object* on both modules so ``core.<name>``
    and ``search.<name>`` can never drift apart. See "The re-export convention"
    in docs/architecture.md.
    """

    def test_swappable_names_are_not_reexported_by_core(self):
        for name in _SEARCH_NAMES_NOT_ON_CORE:
            with self.subTest(name=name):
                self.assertFalse(
                    hasattr(core, name),
                    f"core.{name} must not exist; read it as search.{name}",
                )

    def test_stable_names_are_reexported_by_identity(self):
        for name in _SEARCH_REEXPORTS:
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(search, name))


class ContextFacade(unittest.TestCase):
    """Pin the context.py split: every context name re-exported from core by identity.

    All nine context-prediction names are stable, so ``core.<name>`` must be the
    *same object* as ``context.<name>``. ``context`` also reads ``_SEARCH_KINDS``
    from ``search`` so both validate the same set of accepted kinds; that single
    owner is pinned by identity here. See "The re-export convention" in
    docs/architecture.md.
    """

    def test_stable_names_are_reexported_by_identity(self):
        for name in _CONTEXT_REEXPORTS:
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(context, name))

    def test_context_reads_search_kinds_from_search(self):
        self.assertIs(context._SEARCH_KINDS, search._SEARCH_KINDS)


# Assembly names that tests and routers swap on ``assembly`` to install fakes, so
# they must NOT exist on core -- a core re-export would be a second object and a
# swap would leave the live path using the old one (section 8.4).
_ASSEMBLY_NAMES_NOT_ON_CORE = (
    "_get_assembly_matcher",
    "prewarm_assembly_matcher",
    "search_assembly_index",
)

# The rest of the assembly surface is stable -- never swapped, never rebound -- so
# core re-exports the very same objects assembly.py defines, including for
# ``delete_index`` in core (which calls ``_evict_assembly_matchers``).
_ASSEMBLY_REEXPORTS = (
    "_ASSEMBLY_COVERAGE_MODES",
    "_MIN_BODIES_FOR_ASSEMBLY",
    "_ASSEMBLY_MATCHER_CACHE_MAX",
    "_assembly_matchers",
    "_assembly_matchers_mutex",
    "_assembly_prewarm_in_flight",
    "_assembly_prewarm_mutex",
    "_ASSEMBLY_SEARCH_JOBS_DEFAULT",
    "_ASSEMBLY_CANDIDATE_K_MAX",
    "_assembly_search_jobs",
    "_evict_assembly_matchers_locked",
    "_evict_assembly_matchers",
    "_lookup_assembly_matcher",
    "_has_cached_assembly_matcher",
    "assembly_prewarm_enabled",
    "_rebuild_assembly_matcher",
    "_assembly_metadata_by_id",
    "_assembly_hit",
    "_assembly_self_hit",
)


class AssemblyFacade(unittest.TestCase):
    """Pin the assembly.py split: swapped names off core, stable names re-exported.

    ``_get_assembly_matcher``, ``prewarm_assembly_matcher`` and
    ``search_assembly_index`` must be *absent* from core so a test swapping
    ``assembly.<name>`` cannot leave a second object live on core. Everything else
    must be the *same object* on both modules. See "The re-export
    convention" in docs/architecture.md.
    """

    def test_swappable_names_are_not_reexported_by_core(self):
        for name in _ASSEMBLY_NAMES_NOT_ON_CORE:
            with self.subTest(name=name):
                self.assertFalse(
                    hasattr(core, name),
                    f"core.{name} must not exist; read it as assembly.{name}",
                )

    def test_stable_names_are_reexported_by_identity(self):
        for name in _ASSEMBLY_REEXPORTS:
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(assembly, name))

    def test_assembly_does_not_bind_the_tags_module(self):
        """``import tags`` here would let the ``tags`` parameter of
        ``search_assembly_index`` shadow the module -- the #61 footgun. The tag
        helpers must be reached via ``from tags import ...`` instead.
        """
        source = pathlib.Path(assembly.__file__).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            self.assertNotEqual(
                stripped, "import tags",
                "assembly.py must not bind the tags module; import the helpers by name",
            )
            self.assertFalse(
                stripped.startswith("import tags "),
                "assembly.py must not bind the tags module; import the helpers by name",
            )


# Shape-map names that tests and routers swap on ``shape_map`` to install fakes,
# so they must NOT exist on core -- a core re-export would be a second object and
# a swap would leave the live path using the old one (section 8.4).
_SHAPE_MAP_NAMES_NOT_ON_CORE = (
    "_load_index_body_batch",
    "_shape_map_filemeta",
    "compute_index_shape_map",
    "compute_index_shape_map_scs",
)

# The rest of the shape-map surface is stable -- never swapped, never rebound --
# so core re-exports the very same objects shape_map.py defines, including for the
# core orchestrators (``run_index_add_job`` -> prewarm_shape_map,
# ``run_startup_maintenance`` -> clear_scs_map_cache, ``_warmup_one`` ->
# ScsToolUnavailable).
_SHAPE_MAP_REEXPORTS = (
    "_SHAPE_MAP_AGG_CACHE_MAX",
    "_shape_map_agg",
    "_SHAPE_MAP_RESULT_CACHE_MAX",
    "_shape_map_results",
    "_shape_map_cache_mutex",
    "_SHAPE_MAP_SUBSET_CACHE_MAX",
    "_shape_map_subsets",
    "_shape_map_subset_predicate",
    "_normalize_map_subset",
    "_normalize_outlier_threshold",
    "_shape_map_outlier_assign",
    "_shape_map_outlier_ceiling",
    "_shape_map_centroid_sims",
    "_shape_map_cohesion",
    "_shape_map_centres",
    "_SHAPE_MAP_CYCLE_FACTORS",
    "_shape_map_cluster_rgb",
    "_register_map_subset",
    "_resolve_map_subset",
    "_shape_map_prewarm_in_flight",
    "_shape_map_prewarm_mutex",
    "_shape_map_l2norm_rows",
    "_shape_map_project",
    "_shape_map_run_kmeans",
    "_shape_map_kmeans_max_points_per_centroid",
    "_shape_map_kmeans_nredo",
    "_shape_map_kmeans_spherical",
    "_shape_map_kmeans_init",
    "_shape_map_kmeans_init_method",
    "_shape_map_kmeans_signature",
    "_shape_map_elbow_k",
    "_shape_map_choose_optimal_k",
    "_shape_map_cluster",
    "_compute_shape_map_points",
    "_shape_map_build_lock",
    "_shape_map_legend",
    "_shape_map_build_payload",
    "_normalize_map_kind",
    "_compose_keep",
    "_shape_map_kind_predicate",
    "_shape_map_subset_sig",
    "_SCS_POINTS_TIMEOUT_S",
    "_SCS_MAP_CACHE_MAX_DEFAULT",
    "_SCS_UNTAGGED_RGB",
    "_SHAPE_MAP_PALETTE",
    "_SHAPE_MAP_OVERFLOW_RGB",
    "_SHAPE_MAP_COLORING_VERSION",
    "ScsToolUnavailable",
    "_scs_points_exe_path",
    "_scs_map_cache_max",
    "_scs_build_lock",
    "_shape_map_categorize",
    "_shape_map_color_assignment",
    "_touch",
    "_prune_scs_map_cache",
    "clear_scs_map_cache",
    "shape_map_prewarm_enabled",
    "_has_cached_shape_map",
    "_rebuild_shape_map",
    "prewarm_shape_map",
)


class ShapeMapFacade(unittest.TestCase):
    """Pin the shape_map.py split: swapped names off core, stable names re-exported.

    ``_load_index_body_batch``, ``_shape_map_filemeta``, ``compute_index_shape_map``
    and ``compute_index_shape_map_scs`` must be *absent* from core so a test
    swapping ``shape_map.<name>`` cannot leave a second object live on core.
    Everything else must be the *same object* on both modules. See "The
    re-export convention" in docs/architecture.md.
    """

    def test_swappable_names_are_not_reexported_by_core(self):
        for name in _SHAPE_MAP_NAMES_NOT_ON_CORE:
            with self.subTest(name=name):
                self.assertFalse(
                    hasattr(core, name),
                    f"core.{name} must not exist; read it as shape_map.{name}",
                )

    def test_stable_names_are_reexported_by_identity(self):
        for name in _SHAPE_MAP_REEXPORTS:
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(shape_map, name))

    def test_shape_map_does_not_bind_the_tags_module(self):
        """``import tags`` here would let the ``tags`` parameter of
        ``compute_index_shape_map`` / ``compute_index_shape_map_scs`` (and the
        ``tags`` local of ``_shape_map_subset_sig``) shadow the module -- the #61
        footgun. The tag helpers must be reached via ``from tags import ...``.
        """
        source = pathlib.Path(shape_map.__file__).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            self.assertNotEqual(
                stripped, "import tags",
                "shape_map.py must not bind the tags module; import the helpers by name",
            )
            self.assertFalse(
                stripped.startswith("import tags "),
                "shape_map.py must not bind the tags module; import the helpers by name",
            )


_LICENSING_NAMES_NOT_ON_CORE = (
    "_find_license_shim_file",
)
_LICENSING_REEXPORTS = (
    "_shim_expected_python",
    "init_hoops_license",
)


class LicensingFacade(unittest.TestCase):
    """Pin the licensing.py split: the swappable shim finder off core, stable re-exported.

    ``_find_license_shim_file`` is swapped by the license tests
    (``licensing._find_license_shim_file``), so it must be *absent* from core;
    the two stable names stay re-exported by identity. See "The re-export
    convention" in docs/architecture.md.
    """

    def test_swappable_names_are_not_reexported_by_core(self):
        for name in _LICENSING_NAMES_NOT_ON_CORE:
            with self.subTest(name=name):
                self.assertFalse(
                    hasattr(core, name),
                    f"core.{name} must not exist; read it as licensing.{name}",
                )

    def test_stable_names_are_reexported_by_identity(self):
        for name in _LICENSING_REEXPORTS:
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(licensing, name))


_VIEWER_REEXPORTS = (
    "export_scs_for_part",
)


class ViewerFacade(unittest.TestCase):
    """Pin the viewer.py split: the stream-cache export re-exported by identity.

    ``export_scs_for_part`` is a stable (C) name -- routers reach it as
    ``core.export_scs_for_part`` and it is the same object viewer.py defines.
    It owns no swappable name.
    """

    def test_stable_names_are_reexported_by_identity(self):
        for name in _VIEWER_REEXPORTS:
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(viewer, name))


_WARMUP_REEXPORTS = (
    "_WARMUP_MAX_DEFAULT",
    "_warmup_max",
    "_resolve_warmup_targets",
    "_warmup_one",
    "run_startup_warmup",
    "start_startup_warmup",
)


class WarmupFacade(unittest.TestCase):
    """Pin the warmup.py split: all six startup-warmup names re-exported by identity.

    Warmup owns no swappable (A) name -- the domain entry points it drives are
    swapped on their own modules (``shape_map.compute_index_shape_map`` etc.) --
    so every warmup name stays re-exported from core by identity.
    """

    def test_stable_names_are_reexported_by_identity(self):
        for name in _WARMUP_REEXPORTS:
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(warmup, name))


_JOBS_NAMES_NOT_ON_CORE = (
    "get_index_job_store",
    "_index_job_store",
)
_JOBS_REEXPORTS = (
    "JOB_KIND_INDEX_ADD",
    "_JOB_MAX_CONCURRENCY_DEFAULT",
    "_JOB_TTL_DAYS_DEFAULT",
    "_JOB_MAX_RECORDS_DEFAULT",
    "_index_job_store_lock",
    "job_max_concurrency",
    "job_ttl_seconds",
    "job_max_records",
    "_RETRY_REASON_MARKERS",
    "annotate_failures",
    "_write_add_job_report",
    "run_index_add_job",
)


class JobsFacade(unittest.TestCase):
    """Pin the jobs.py split: the job-store singleton off core, stable names re-exported.

    ``get_index_job_store`` is swapped by the job tests and ``_index_job_store``
    is both swapped and rebound with ``global`` as a lazy singleton, so both must
    be *absent* from core; reach them as ``jobs.<name>``. The remaining twelve
    names stay re-exported by identity. See "The re-export convention" in
    docs/architecture.md.
    """

    def test_swappable_names_are_not_reexported_by_core(self):
        for name in _JOBS_NAMES_NOT_ON_CORE:
            with self.subTest(name=name):
                self.assertFalse(
                    hasattr(core, name),
                    f"core.{name} must not exist; read it as jobs.{name}",
                )

    def test_stable_names_are_reexported_by_identity(self):
        for name in _JOBS_REEXPORTS:
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(jobs, name))


_RUNTIME_REEXPORTS = (
    "_json_safe",
    "log_resolved_configuration",
    "_OUT_TTL_HOURS_DEFAULT",
    "_EMBEDDINGS_CACHE_TTL_DAYS_DEFAULT",
    "viewer_output_ttl_seconds",
    "embeddings_cache_ttl_seconds",
    "sweep_directory",
    "ensure_runtime_dirs",
    "run_startup_maintenance",
)


class RuntimeFacade(unittest.TestCase):
    """Pin the runtime.py split: all nine runtime-lifecycle names re-exported by identity.

    Runtime owns no swappable (A) name. Its one cross-domain call reaches
    ``jobs.get_index_job_store`` on the owning module, so every runtime name
    stays re-exported from core by identity.
    """

    def test_stable_names_are_reexported_by_identity(self):
        for name in _RUNTIME_REEXPORTS:
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(runtime, name))


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Directories that never hold first-party source: version control, caches,
# bundled dependencies/virtualenvs, build output and the large data roots.
_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "site-packages",
    "build",
    "dist",
    # vendored upstream code we must not modify: a future vendor refresh must
    # not turn ModuleShadowGuard (or any _repo_python_files scan) red over code
    # that is not ours to fix.
    "vendor",
    "bin",
    "uploads",
    "indexes",
    "thumbnails",
    "scs",
    "scs_map",
}


def _repo_python_files():
    """Every first-party .py file under the repo root.

    Walks the tree rather than shelling out to ``git`` so the scan cannot be
    perturbed by another test's environment or working-directory changes. Caches,
    bundled dependencies, virtualenvs, build output and the data roots are pruned.
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT):
        dirnames[:] = [
            d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        for name in filenames:
            if name.endswith(".py"):
                files.append(pathlib.Path(dirpath) / name)
    return files


def _module_import_names(tree):
    """Names bound by a module-level ``import x`` / ``import x as y`` statement.

    ``from m import n`` is deliberately excluded: the shadow that breaks code is a
    module object reached as ``X.<attr>``, and only ``import`` binds such a name.
    """
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _own_scope_nodes(func):
    """Nodes in *func*'s own scope, not descending into nested def/lambda/class."""
    result = []
    stack = list(func.body)
    while stack:
        node = stack.pop()
        result.append(node)
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
            ):
                continue
            stack.append(child)
    return result


def _locally_bound_names(func, own):
    """Names bound as parameters or local assignments in *func*'s own scope."""
    bound = set()
    args = func.args
    for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
        bound.add(arg.arg)
    if args.vararg:
        bound.add(args.vararg.arg)
    if args.kwarg:
        bound.add(args.kwarg.arg)
    for node in own:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                        bound.add(name.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            for name in ast.walk(node.target):
                if isinstance(name, ast.Name):
                    bound.add(name.id)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            for name in ast.walk(node.optional_vars):
                if isinstance(name, ast.Name):
                    bound.add(name.id)
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


def _find_module_shadow_violations(path):
    """Return (line, func, module, attr) for locals that shadow an imported module.

    A violation is a function that (1) binds a name as a parameter or local
    assignment which is also a module-level ``import`` name, without a ``global``
    for it, and (2) uses that same name as ``X.<attr>`` in the same scope -- which
    at runtime raises ``UnboundLocalError`` (see #61). A bare shadow with no
    attribute access is not a violation.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_names = _module_import_names(tree)
    violations = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        own = _own_scope_nodes(func)
        declared_global = set()
        for node in own:
            if isinstance(node, ast.Global):
                declared_global.update(node.names)
        shadowed = (_locally_bound_names(func, own) & module_names) - declared_global
        if not shadowed:
            continue
        for node in own:
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and isinstance(node.value.ctx, ast.Load)
                and node.value.id in shadowed
            ):
                violations.append((node.lineno, func.name, node.value.id, node.attr))
    return violations


class ModuleShadowGuard(unittest.TestCase):
    """Forbid a local/argument that shadows an imported module and uses its attrs.

    Batch 2c rewrote ``_PhaseTimer()`` to ``diagnostics._PhaseTimer()`` inside
    ``search_index``, whose tail already bound a local named ``diagnostics``.
    Python makes ``diagnostics`` local for the whole function, so the module
    attribute access raised ``UnboundLocalError`` whenever timing was on (#61).
    A one-line rename fixed it; this guard stops the pattern coming back as the
    remaining batches keep moving code. Only the *coexistence* of the shadow and
    a ``X.<attr>`` use is a violation -- a bare shadow (for example the ``tags``
    argument of ``search_index``, which never does ``tags.<attr>``) is allowed.
    """

    def test_no_local_shadows_an_imported_module_it_uses(self):
        offenders = []
        for path in _repo_python_files():
            try:
                found = _find_module_shadow_violations(path)
            except (SyntaxError, UnicodeDecodeError):
                continue
            rel = path.relative_to(_REPO_ROOT).as_posix()
            for line, func, module, attr in found:
                offenders.append(
                    f"{rel}:{line}: function '{func}()' binds a local named "
                    f"'{module}' that shadows the imported module '{module}', "
                    f"then uses '{module}.{attr}' at line {line}"
                )
        self.assertEqual(
            offenders,
            [],
            "Local/argument shadows an imported module it calls "
            "(UnboundLocalError risk, see #61):\n" + "\n".join(offenders),
        )


_IMPORT_CORE_ALLOWED = {"core.py", "main.py"}
_FIRST_CLASS_SKIP_DIRS = _SKIP_DIRS | {"tests", "routers", "tools", "vendor"}


def _first_class_modules():
    """First-party modules that must not import core.

    Walks like :func:`_repo_python_files` but also prunes the packages that
    legitimately reach into core (``routers``, ``tools``, ``tests``) and the
    bundled ``vendor`` tree, and drops ``core.py`` itself and ``main.py`` (the
    ASGI composition root that wires core up). Scanning rather than listing names
    means a module added by a later refactor is covered without editing this test.
    """
    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT):
        dirnames[:] = [
            d for d in dirnames
            if d not in _FIRST_CLASS_SKIP_DIRS and not d.startswith(".")
        ]
        for name in filenames:
            if name.endswith(".py") and name not in _IMPORT_CORE_ALLOWED:
                yield pathlib.Path(dirpath) / name


class ImportHygiene(unittest.TestCase):
    """No first-class domain module may import core (the dependency is one-way).

    Replaces the per-module ``test_<mod>_does_not_import_core`` methods with a
    single scan so every current and future first-class module is checked without
    editing this file. ``core.py`` (defines the surface) and ``main.py`` (the
    composition root) are exempt; ``routers``/``tools``/``tests`` reach core by
    design and are pruned.
    """

    def test_first_class_modules_do_not_import_core(self):
        for path in _first_class_modules():
            rel = path.relative_to(_REPO_ROOT).as_posix()
            with self.subTest(module=rel):
                source = path.read_text(encoding="utf-8")
                for line in source.splitlines():
                    stripped = line.strip()
                    self.assertNotEqual(
                        stripped, "import core", f"{rel} must not import core"
                    )
                    self.assertFalse(
                        stripped.startswith("from core "),
                        f"{rel} must not import from core",
                    )
                    self.assertFalse(
                        stripped.startswith("import core "),
                        f"{rel} must not import core",
                    )


if __name__ == "__main__":
    unittest.main()
