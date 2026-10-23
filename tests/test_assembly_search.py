"""SDK-free unit tests for assembly-to-assembly search.

Covers ``assembly.search_assembly_index()`` and the
``POST /similarity/index/{name}/search-assembly`` endpoint:

  * the fixed AssemblyMatcher arguments that keep us aligned with
    hoops_ai_native_bridge (hungarian / search / assemblies_only / reuse)
  * self-hit removal and include_self promotion (same trap as PR #5)
  * the matcher's key names mapped onto the response fields
  * matcher caching (mtime invalidation, single-entry cap)
  * parameter validation

AssemblyMatcher, CADSearch and the vector store are faked; no HOOPS AI license
is required.
"""

import os
import pathlib
import sys
import tempfile
import types
import unittest
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core
import assembly
import index_registry
import embeddings
import storage
import config


def _result(assembly, score, geom=0.0, coverage=0.0, matched=0, n_parts=0, m=0):
    """One AssemblyMatcher.search() result row (keys observed in the tutorial)."""
    return {
        "assembly": assembly,
        "score": score,
        "geom_score": geom,
        "bop_sim": 0.0,
        "coverage": coverage,
        "matched": matched,
        "M": m,
        "N": n_parts,
        "n_parts": n_parts,
        "display_metadata": None,
    }


class _FakeMatcher:
    """Records the kwargs search() was called with and returns canned rows."""

    def __init__(self, results):
        self._results = results
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self._results


class _AssemblySearchBase(unittest.TestCase):
    QUERY_ID = "q" * 64
    A = "a" * 64
    B = "b" * 64
    C = "c" * 64

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_idx_dir = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmp.name)

        self.results = [
            _result(self.A, 0.91, geom=0.88, coverage=0.75, matched=7, n_parts=9, m=8),
            _result(self.B, 0.62, geom=0.60, coverage=0.50, matched=4, n_parts=6, m=8),
            _result(self.C, 0.31, geom=0.30, coverage=0.20, matched=2, n_parts=3, m=8),
        ]
        self.matcher = _FakeMatcher(self.results)

        self.registered = {self.A, self.B, self.C}
        self.metas = [
            {"file_id": self.A, "filename": "a.stp", "kind": "assembly", "bodies": 9,
             "_query_body_rep_count": 1},
            {"file_id": self.B, "filename": "b.stp", "kind": "assembly", "bodies": 6},
            {"file_id": self.C, "filename": "c.stp", "kind": "assembly", "bodies": 3},
        ]

        vs = MagicMock()
        vs.get_ids.side_effect = lambda: list(self.registered)
        vs.iter_metadata.side_effect = lambda: iter(self.metas)

        self._saved = {
            "_load_named_index": index_registry._load_named_index,
            "_get_assembly_matcher": assembly._get_assembly_matcher,
            "_load_index_schema": index_registry._load_index_schema,
            "find_persistent_CAD_file": storage.find_persistent_CAD_file,
            "_resolve_index_asset": index_registry._resolve_index_asset,
        }
        index_registry._load_named_index = lambda name: vs
        assembly._get_assembly_matcher = lambda name: self.matcher
        self.schema = {"schema_version": core._INDEX_SCHEMA_VERSION}
        index_registry._load_index_schema = lambda name: self.schema
        self.resolve_calls = []

        def _resolve_cad(fid):
            self.resolve_calls.append(fid)
            return pathlib.Path(self._tmp.name) / f"{fid}.stp"

        storage.find_persistent_CAD_file = _resolve_cad
        index_registry._resolve_index_asset = lambda name, pid, asset: None

    def tearDown(self):
        for attr, val in self._saved.items():
            if attr == "find_persistent_CAD_file":
                mod = storage
            elif attr == "_get_assembly_matcher":
                mod = assembly
            else:
                mod = index_registry
            setattr(mod, attr, val)
        config.INDEXES_DIR = self._orig_idx_dir
        self._tmp.cleanup()


class TestBridgeArguments(_AssemblySearchBase):
    """The fixed arguments must stay identical to hoops_ai_native_bridge."""

    def test_fixed_arguments_match_bridge(self):
        assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        kwargs = self.matcher.calls[0]
        self.assertEqual(kwargs["method"], "hungarian")
        self.assertEqual(kwargs["candidate_mode"], "search")
        self.assertIs(kwargs["assemblies_only"], True)
        self.assertIs(kwargs["reuse_index_vectors"], True)

    def test_requests_one_extra_result_for_the_self_hit(self):
        assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        self.assertEqual(self.matcher.calls[0]["top_k"], 6)

    def test_defaults_match_bridge(self):
        assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        kwargs = self.matcher.calls[0]
        self.assertEqual(kwargs["candidate_k"], 30)
        self.assertAlmostEqual(kwargs["sim_thresh"], 0.80)
        self.assertAlmostEqual(kwargs["bop_weight"], 0.30)
        self.assertEqual(kwargs["coverage_mode"], "symmetric")
        self.assertIs(kwargs["use_idf"], True)

    def test_tunables_are_forwarded(self):
        assembly.search_assembly_index(
            "idx", self.QUERY_ID, 5,
            candidate_k=50, sim_thresh=0.5, bop_weight=0.0,
            coverage_mode="containment", use_idf=False,
        )
        kwargs = self.matcher.calls[0]
        self.assertEqual(kwargs["candidate_k"], 50)
        self.assertAlmostEqual(kwargs["sim_thresh"], 0.5)
        self.assertAlmostEqual(kwargs["bop_weight"], 0.0)
        self.assertEqual(kwargs["coverage_mode"], "containment")
        self.assertIs(kwargs["use_idf"], False)

    def test_query_is_passed_as_a_resolved_path(self):
        assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        self.assertTrue(self.matcher.calls[0]["query_path"].endswith(f"{self.QUERY_ID}.stp"))


    def test_n_jobs_comes_from_the_environment(self):
        os.environ["HOOPS_AI_ASSEMBLY_SEARCH_JOBS"] = "3"
        try:
            assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        finally:
            os.environ.pop("HOOPS_AI_ASSEMBLY_SEARCH_JOBS", None)
        self.assertEqual(self.matcher.calls[0]["n_jobs"], 3)


class TestSearchLocking(_AssemblySearchBase):
    """Neither resolving the matcher nor matching may hold the index lock.

    Building a matcher takes ~20 s, so a cold assembly search holding the index
    lock across it would stall part search on the same index. The lock is for
    the metadata read only.
    """

    def _index_lock_is_free(self):
        lock = core._get_index_lock("idx")
        free = lock.acquire(blocking=False)
        if free:
            lock.release()
        return free

    def test_matcher_is_resolved_outside_the_index_lock(self):
        seen = []
        assembly._get_assembly_matcher = lambda name: (
            seen.append(self._index_lock_is_free()) or self.matcher
        )
        assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        self.assertEqual(seen, [True])

    def test_matching_runs_outside_the_index_lock(self):
        seen = []
        real_search = self.matcher.search

        def search(**kwargs):
            seen.append(self._index_lock_is_free())
            return real_search(**kwargs)

        self.matcher.search = search
        assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        self.assertEqual(seen, [True])

    def test_the_metadata_read_is_still_locked(self):
        seen = []
        vs = MagicMock()
        vs.get_ids.side_effect = lambda: list(self.registered)

        def iter_metadata():
            seen.append(self._index_lock_is_free())
            return iter(self.metas)

        vs.iter_metadata.side_effect = iter_metadata
        index_registry._load_named_index = lambda name: vs
        assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        self.assertEqual(seen, [False])


class TestQueryIdentity(_AssemblySearchBase):
    """A registered query must reach the matcher as its record id so that
    reuse_index_vectors=True can hit the stored per-body vectors."""

    def _register_query(self):
        self.registered.add(self.QUERY_ID)
        self.metas.append(
            {"file_id": self.QUERY_ID, "filename": "q.stp", "kind": "assembly", "bodies": 8}
        )

    def test_registered_query_is_passed_as_its_file_id(self):
        self._register_query()
        assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        self.assertEqual(self.matcher.calls[0]["query_path"], self.QUERY_ID)

    def test_registered_query_does_not_touch_the_filesystem(self):
        self._register_query()
        assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        self.assertEqual(self.resolve_calls, [])

    def test_unregistered_query_is_passed_as_a_path(self):
        assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        query = self.matcher.calls[0]["query_path"]
        self.assertNotEqual(query, self.QUERY_ID)
        self.assertTrue(query.endswith(f"{self.QUERY_ID}.stp"))

    def test_unregistered_query_resolves_the_upload(self):
        assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        self.assertEqual(self.resolve_calls, [self.QUERY_ID])

    def test_query_path_is_always_a_string(self):
        assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        self._register_query()
        assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        self.assertEqual(len(self.matcher.calls), 2)
        for call in self.matcher.calls:
            self.assertIsInstance(call["query_path"], str)

    def test_missing_upload_still_raises_for_unregistered_query(self):
        def _boom(fid):
            raise FileNotFoundError(fid)

        storage.find_persistent_CAD_file = _boom
        with self.assertRaises(FileNotFoundError):
            assembly.search_assembly_index("idx", self.QUERY_ID, 5)

    def test_self_hit_handling_survives_the_id_path(self):
        # candidates.discard(query_path) now works, but the explicit filter must
        # still hold when the matcher returns the query anyway.
        self._register_query()
        self.results.insert(
            0, _result(self.QUERY_ID, 1.0, geom=1.0, coverage=1.0, matched=8, n_parts=8, m=8)
        )
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 3)
        self.assertNotIn(self.QUERY_ID, [h["id"] for h in out["hits"]])
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 3, include_self=True)
        ids = [h["id"] for h in out["hits"]]
        self.assertEqual(ids[0], self.QUERY_ID)
        self.assertEqual(ids.count(self.QUERY_ID), 1)
        self.assertEqual(len(ids), 3)

    def test_include_self_works_when_matcher_drops_the_query(self):
        # With the id form the matcher discards the query itself, so the self
        # hit has to be synthesised from the stored metadata.
        self._register_query()
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 3, include_self=True)
        ids = [h["id"] for h in out["hits"]]
        self.assertEqual(ids, [self.QUERY_ID, self.A, self.B])
        self.assertEqual(out["hits"][0]["score"], 1.0)


class TestResultMapping(_AssemblySearchBase):
    def test_matcher_keys_map_to_response_fields(self):
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        hit = out["hits"][0]
        self.assertEqual(hit["id"], self.A)
        self.assertAlmostEqual(hit["score"], 0.91)
        self.assertAlmostEqual(hit["geom_score"], 0.88)
        self.assertAlmostEqual(hit["coverage"], 0.75)
        self.assertEqual(hit["matched_parts"], 7)      # <- matched
        self.assertEqual(hit["candidate_parts"], 9)    # <- n_parts
        self.assertEqual(hit["query_parts"], 8)        # <- M

    def test_metadata_comes_from_the_index(self):
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        self.assertEqual(out["hits"][0]["metadata"]["filename"], "a.stp")
        self.assertEqual(out["hits"][0]["metadata"]["bodies"], 9)

    def test_metadata_is_read_in_a_single_pass(self):
        vs = index_registry._load_named_index("idx")
        assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        self.assertEqual(vs.iter_metadata.call_count, 1)

    def test_no_underscore_keys_in_any_hit(self):
        for include_self in (False, True):
            self.registered.add(self.QUERY_ID)
            out = assembly.search_assembly_index(
                "idx", self.QUERY_ID, 5, include_self=include_self
            )
            for hit in out["hits"]:
                leaked = [k for k in hit["metadata"] if k.startswith("_")]
                self.assertEqual(leaked, [], f"include_self={include_self}: {leaked}")

    def test_unknown_id_still_yields_metadata(self):
        self.results.append(_result("z" * 64, 0.1))
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 10)
        self.assertEqual(out["hits"][-1]["metadata"], {"file_id": "z" * 64})

    def test_malformed_rows_are_skipped(self):
        self.results.insert(0, "not-a-dict")
        self.results.insert(1, {"score": 0.5})  # no assembly id
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 10)
        self.assertEqual([h["id"] for h in out["hits"]], [self.A, self.B, self.C])

    def test_empty_index_returns_no_hits(self):
        self.registered.clear()
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        self.assertEqual(out, {"hits": [], "count": 0, "image_url": None})
        self.assertEqual(self.matcher.calls, [])


class TestSelfHit(_AssemblySearchBase):
    """AssemblyMatcher cannot drop the query itself: ids are SHA-256, the
    query is an uploads/<sha>_<name> path (see PR #5)."""

    def setUp(self):
        super().setUp()
        self.registered.add(self.QUERY_ID)
        self.metas.append(
            {"file_id": self.QUERY_ID, "filename": "q.stp", "kind": "assembly", "bodies": 8}
        )
        # The matcher returns the query as a perfect self match.
        self.results.insert(
            0, _result(self.QUERY_ID, 1.0, geom=1.0, coverage=1.0, matched=8, n_parts=8, m=8)
        )

    def test_self_hit_excluded_by_default(self):
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 3)
        self.assertNotIn(self.QUERY_ID, [h["id"] for h in out["hits"]])

    def test_excluding_self_still_fills_top_k(self):
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 3)
        self.assertEqual(len(out["hits"]), 3)
        self.assertEqual([h["id"] for h in out["hits"]], [self.A, self.B, self.C])

    def test_include_self_promotes_without_duplicating(self):
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 3, include_self=True)
        ids = [h["id"] for h in out["hits"]]
        self.assertEqual(ids[0], self.QUERY_ID)
        self.assertEqual(ids.count(self.QUERY_ID), 1)
        self.assertEqual(out["hits"][0]["score"], 1.0)

    def test_include_self_does_not_exceed_top_k(self):
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 3, include_self=True)
        self.assertEqual(len(out["hits"]), 3)
        self.assertEqual(out["count"], 3)

    def test_self_hit_carries_stored_metadata_and_body_count(self):
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 3, include_self=True)
        self.assertEqual(out["hits"][0]["metadata"]["filename"], "q.stp")
        self.assertEqual(out["hits"][0]["query_parts"], 8)
        self.assertEqual(out["hits"][0]["candidate_parts"], 8)

    def test_include_self_noop_when_query_not_registered(self):
        self.registered.discard(self.QUERY_ID)
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 5, include_self=True)
        self.assertNotIn(self.QUERY_ID, [h["id"] for h in out["hits"]])


class TestValidation(_AssemblySearchBase):
    def test_invalid_coverage_mode_raises(self):
        with self.assertRaises(ValueError):
            assembly.search_assembly_index("idx", self.QUERY_ID, 5, coverage_mode="bogus")
        self.assertEqual(self.matcher.calls, [])

    def test_valid_coverage_modes_accepted(self):
        for mode in ("symmetric", "containment", "jaccard"):
            assembly.search_assembly_index("idx", self.QUERY_ID, 5, coverage_mode=mode)
        self.assertEqual(len(self.matcher.calls), 3)

    def test_candidate_k_out_of_range_raises(self):
        for bad in (0, -1, core._ASSEMBLY_CANDIDATE_K_MAX + 1):
            with self.assertRaises(ValueError):
                assembly.search_assembly_index("idx", self.QUERY_ID, 5, candidate_k=bad)

    def test_sim_thresh_out_of_range_raises(self):
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                assembly.search_assembly_index("idx", self.QUERY_ID, 5, sim_thresh=bad)

    def test_bop_weight_out_of_range_raises(self):
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                assembly.search_assembly_index("idx", self.QUERY_ID, 5, bop_weight=bad)

    def test_non_positive_top_k_returns_one_hit(self):
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 0)
        self.assertEqual(len(out["hits"]), 1)

    def test_legacy_schema_v1_is_rejected(self):
        # A v1 index holds one averaged vector per file, so every entry looks
        # like a single-body part and assemblies_only would discard everything.
        self.schema = {"schema_version": 1}
        with self.assertRaises(core.SchemaVersionError):
            assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        self.assertEqual(self.matcher.calls, [])


class TestImageUrl(_AssemblySearchBase):
    def test_image_url_is_always_none(self):
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 3)
        self.assertIsNone(out["image_url"])

    def test_hits_are_returned_without_image(self):
        out = assembly.search_assembly_index("idx", self.QUERY_ID, 3)
        self.assertEqual(out["count"], len(out["hits"]))


class TestAssemblySearchJobs(unittest.TestCase):
    def setUp(self):
        self._orig = os.environ.get("HOOPS_AI_ASSEMBLY_SEARCH_JOBS")

    def tearDown(self):
        if self._orig is None:
            os.environ.pop("HOOPS_AI_ASSEMBLY_SEARCH_JOBS", None)
        else:
            os.environ["HOOPS_AI_ASSEMBLY_SEARCH_JOBS"] = self._orig

    def test_default_matches_bridge(self):
        os.environ.pop("HOOPS_AI_ASSEMBLY_SEARCH_JOBS", None)
        self.assertEqual(core._assembly_search_jobs(), 8)

    def test_custom_value(self):
        os.environ["HOOPS_AI_ASSEMBLY_SEARCH_JOBS"] = "2"
        self.assertEqual(core._assembly_search_jobs(), 2)

    def test_invalid_values_fall_back(self):
        for bad in ("", "abc", "0", "-4"):
            os.environ["HOOPS_AI_ASSEMBLY_SEARCH_JOBS"] = bad
            self.assertEqual(core._assembly_search_jobs(), 8, bad)


class _MatcherCacheBase(unittest.TestCase):
    """Fakes for CADSearch / AssemblyMatcher plus a scratch INDEXES_DIR."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_idx_dir = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmp.name)
        self._orig_get_embedder = embeddings.get_embedder
        embeddings.get_embedder = lambda: SimpleNamespace(model_name="fake")

        self._saved_mods = {}
        for name in ("hoops_ai", "hoops_ai.ml", "hoops_ai.ml.embeddings",
                     "vendor", "vendor.assembly_matcher"):
            self._saved_mods[name] = sys.modules.get(name)

        pkg = types.ModuleType("hoops_ai")
        ml = types.ModuleType("hoops_ai.ml")
        emb = types.ModuleType("hoops_ai.ml.embeddings")

        class _CADSearch:
            def __init__(self, shape_model=None):
                self.loaded = None

            def load_shape_index(self, path=None):
                self.loaded = path
                return SimpleNamespace(ids=[], values=[], model="fake")

        emb.CADSearch = _CADSearch
        pkg.ml = ml
        ml.embeddings = emb
        sys.modules["hoops_ai"] = pkg
        sys.modules["hoops_ai.ml"] = ml
        sys.modules["hoops_ai.ml.embeddings"] = emb

        class _AssemblyMatcher:
            instances = 0

            def __init__(self, searcher=None, embedding_batch=None, min_bodies_for_assembly=2):
                type(self).instances += 1
                self.serial = type(self).instances
                self.searcher = searcher
                self.batch = embedding_batch
                self.min_bodies = min_bodies_for_assembly
                self.weights_built = 0

            def build_part_rarity_weights(self):
                self.weights_built += 1

        vendor_pkg = types.ModuleType("vendor")
        matcher_mod = types.ModuleType("vendor.assembly_matcher")
        matcher_mod.AssemblyMatcher = _AssemblyMatcher
        vendor_pkg.assembly_matcher = matcher_mod
        sys.modules["vendor"] = vendor_pkg
        sys.modules["vendor.assembly_matcher"] = matcher_mod

        self.AssemblyMatcher = _AssemblyMatcher
        _AssemblyMatcher.instances = 0
        core._assembly_matchers.clear()

    def tearDown(self):
        core._assembly_matchers.clear()
        embeddings.get_embedder = self._orig_get_embedder
        config.INDEXES_DIR = self._orig_idx_dir
        for name, mod in self._saved_mods.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        self._tmp.cleanup()

    def _mk_index(self, name):
        d = config.INDEXES_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.faiss").write_bytes(b"faiss")
        return d / "index.faiss"


class TestAssemblyMatcherCache(_MatcherCacheBase):
    """The matcher is expensive (FAISS k-means over every body), so it is
    cached by (faiss path, mtime) and only one instance is kept."""

    def test_missing_index_raises_key_error(self):
        with self.assertRaises(KeyError):
            assembly._get_assembly_matcher("nope")

    def test_same_mtime_reuses_instance(self):
        self._mk_index("a")
        m1 = assembly._get_assembly_matcher("a")
        m2 = assembly._get_assembly_matcher("a")
        self.assertIs(m1, m2)
        self.assertEqual(self.AssemblyMatcher.instances, 1)
        self.assertEqual(m1.weights_built, 1)

    def test_built_from_the_loaded_index(self):
        p = self._mk_index("a")
        m = assembly._get_assembly_matcher("a")
        self.assertEqual(m.searcher.loaded, str(p))
        self.assertIsNotNone(m.batch)
        self.assertEqual(m.min_bodies, core._MIN_BODIES_FOR_ASSEMBLY)

    def test_mtime_change_rebuilds_instance(self):
        p = self._mk_index("a")
        m1 = assembly._get_assembly_matcher("a")
        st = p.stat()
        os.utime(p, ns=(st.st_atime_ns + 10**9, st.st_mtime_ns + 10**9))
        m2 = assembly._get_assembly_matcher("a")
        self.assertIsNot(m1, m2)
        self.assertEqual(len(core._assembly_matchers), 1)

    def test_only_one_instance_is_kept(self):
        for n in ("a", "b"):
            self._mk_index(n)
        assembly._get_assembly_matcher("a")
        assembly._get_assembly_matcher("b")
        self.assertEqual(len(core._assembly_matchers), core._ASSEMBLY_MATCHER_CACHE_MAX)
        self.assertEqual(core._ASSEMBLY_MATCHER_CACHE_MAX, 1)
        paths = [k[0] for k in core._assembly_matchers]
        self.assertEqual(paths, [str(config.INDEXES_DIR / "b" / "index.faiss")])

    def test_evict_drops_entry(self):
        self._mk_index("a")
        assembly._get_assembly_matcher("a")
        core._evict_assembly_matchers("a")
        self.assertEqual(len(core._assembly_matchers), 0)


class TestAssemblyPrewarm(_MatcherCacheBase):
    """Rebuilding the matcher in the background after a registration job.

    Inherits the SDK fakes from _MatcherCacheBase. The rebuild is exercised
    synchronously via _rebuild_assembly_matcher(); the threading wrapper is
    covered separately by joining the spawned thread.
    """

    def setUp(self):
        super().setUp()
        core._assembly_prewarm_in_flight.clear()
        self._orig_prewarm_env = os.environ.get("HOOPS_AI_ASSEMBLY_PREWARM")
        os.environ.pop("HOOPS_AI_ASSEMBLY_PREWARM", None)

    def tearDown(self):
        core._assembly_prewarm_in_flight.clear()
        if self._orig_prewarm_env is None:
            os.environ.pop("HOOPS_AI_ASSEMBLY_PREWARM", None)
        else:
            os.environ["HOOPS_AI_ASSEMBLY_PREWARM"] = self._orig_prewarm_env
        super().tearDown()

    def _mk_index(self, name, schema_version=None):
        path = super()._mk_index(name)
        import json

        version = (
            core._INDEX_SCHEMA_VERSION if schema_version is None else schema_version
        )
        with open(core._index_json_path(name), "w", encoding="utf-8") as fh:
            json.dump({"schema_version": version}, fh)
        return path

    def _touch(self, path):
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns + 10**9, st.st_mtime_ns + 10**9))

    # --- when a rebuild is worth starting ---------------------------------

    def test_skipped_when_assembly_search_never_used(self):
        self._mk_index("a")
        self.assertFalse(assembly.prewarm_assembly_matcher("a"))
        self.assertEqual(self.AssemblyMatcher.instances, 0)

    def test_started_when_a_matcher_is_cached(self):
        path = self._mk_index("a")
        assembly._get_assembly_matcher("a")
        self._touch(path)
        self.assertTrue(assembly.prewarm_assembly_matcher("a"))
        self._join_prewarm()
        self.assertEqual(self.AssemblyMatcher.instances, 2)

    def test_stale_entry_still_counts_as_used(self):
        """The job has just invalidated the entry; that is exactly the case
        worth rebuilding, so a non-matching mtime must not disqualify it."""
        path = self._mk_index("a")
        assembly._get_assembly_matcher("a")
        self._touch(path)
        self.assertTrue(core._has_cached_assembly_matcher("a"))

    def test_other_index_does_not_count(self):
        self._mk_index("a")
        self._mk_index("b")
        assembly._get_assembly_matcher("b")
        self.assertFalse(core._has_cached_assembly_matcher("a"))
        self.assertFalse(assembly.prewarm_assembly_matcher("a"))

    def test_env_switch_disables_it(self):
        path = self._mk_index("a")
        assembly._get_assembly_matcher("a")
        self._touch(path)
        os.environ["HOOPS_AI_ASSEMBLY_PREWARM"] = "false"
        self.assertFalse(assembly.prewarm_assembly_matcher("a"))
        self.assertEqual(self.AssemblyMatcher.instances, 1)

    def test_enabled_by_default(self):
        self.assertTrue(core.assembly_prewarm_enabled())

    def test_only_one_rebuild_per_index_in_flight(self):
        self._mk_index("a")
        assembly._get_assembly_matcher("a")
        core._assembly_prewarm_in_flight.add("a")
        self.assertFalse(assembly.prewarm_assembly_matcher("a"))

    # --- the rebuild itself ------------------------------------------------

    def test_rebuild_replaces_the_stale_entry(self):
        path = self._mk_index("a")
        first = assembly._get_assembly_matcher("a")
        self._touch(path)
        core._rebuild_assembly_matcher("a")
        self.assertEqual(len(core._assembly_matchers), 1)
        cached = next(iter(core._assembly_matchers.values()))
        self.assertIsNot(cached, first)
        self.assertEqual(cached.weights_built, 1)

    def test_rebuild_makes_the_next_search_warm(self):
        path = self._mk_index("a")
        assembly._get_assembly_matcher("a")
        self._touch(path)
        core._rebuild_assembly_matcher("a")
        built = self.AssemblyMatcher.instances
        assembly._get_assembly_matcher("a")
        self.assertEqual(self.AssemblyMatcher.instances, built)

    def test_rebuild_skips_legacy_schema(self):
        """Assembly search rejects schema v1, so a matcher would never be used."""
        self._mk_index("a", schema_version=1)
        core._rebuild_assembly_matcher("a")
        self.assertEqual(self.AssemblyMatcher.instances, 0)

    def test_rebuild_survives_a_deleted_index(self):
        self._mk_index("a")
        assembly._get_assembly_matcher("a")
        (config.INDEXES_DIR / "a" / "index.faiss").unlink()
        core._rebuild_assembly_matcher("a")  # must not raise

    def test_rebuild_clears_the_in_flight_flag_on_failure(self):
        self._mk_index("a")
        core._assembly_prewarm_in_flight.add("a")
        orig = assembly._get_assembly_matcher
        assembly._get_assembly_matcher = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        try:
            core._rebuild_assembly_matcher("a")
        finally:
            assembly._get_assembly_matcher = orig
        self.assertNotIn("a", core._assembly_prewarm_in_flight)

    def test_in_flight_flag_is_cleared_after_success(self):
        path = self._mk_index("a")
        assembly._get_assembly_matcher("a")
        self._touch(path)
        assembly.prewarm_assembly_matcher("a")
        self._join_prewarm()
        self.assertNotIn("a", core._assembly_prewarm_in_flight)

    def _join_prewarm(self):
        import threading

        for thread in threading.enumerate():
            if thread.name.startswith("assembly-prewarm-"):
                thread.join(timeout=10)
                self.assertFalse(thread.is_alive())


class TestMatcherBuildLocking(_MatcherCacheBase):
    """The build must not hold the index lock across the k-means.

    Holding it would stall part search on the same index for ~20 s, which is
    exactly what PR #5 moved out of the locked section. Only the disk read is
    locked; the k-means afterwards touches nothing but the copy it loaded.
    """

    def setUp(self):
        super().setUp()
        self._events = []
        matcher_cls = self.AssemblyMatcher
        test = self

        def build_part_rarity_weights(inner_self):
            # The expensive phase: record whether the index lock is free here.
            lock = core._get_index_lock("a")
            acquired = lock.acquire(blocking=False)
            test._events.append(("kmeans_index_lock_free", acquired))
            if acquired:
                lock.release()
            inner_self.weights_built += 1

        matcher_cls.build_part_rarity_weights = build_part_rarity_weights

    def _mk_index(self, name):
        path = super()._mk_index(name)
        import json

        with open(core._index_json_path(name), "w", encoding="utf-8") as fh:
            json.dump(
                {"schema_version": core._INDEX_SCHEMA_VERSION}, fh
            )
        return path

    def test_index_lock_is_free_during_the_kmeans(self):
        self._mk_index("a")
        assembly._get_assembly_matcher("a")
        self.assertEqual(self._events, [("kmeans_index_lock_free", True)])

    def test_the_disk_read_happens_under_the_index_lock(self):
        # .faiss and .meta are replaced in two steps, so the load must be
        # paired with a save rather than racing it.
        seen = {}
        self._mk_index("a")
        orig = index_registry._load_shape_index_pickle_compat

        def load(searcher, path):
            seen["locked"] = not core._get_index_lock("a").acquire(blocking=False)
            if not seen["locked"]:
                core._get_index_lock("a").release()
            return orig(searcher, path)

        index_registry._load_shape_index_pickle_compat = load
        try:
            assembly._get_assembly_matcher("a")
        finally:
            index_registry._load_shape_index_pickle_compat = orig
        self.assertTrue(seen["locked"])

    def test_concurrent_callers_build_once(self):
        import threading

        self._mk_index("a")
        results = []
        barrier = threading.Barrier(3)

        def run():
            barrier.wait(timeout=10)
            results.append(assembly._get_assembly_matcher("a"))

        threads = [threading.Thread(target=run) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            self.assertFalse(t.is_alive())
        self.assertEqual(self.AssemblyMatcher.instances, 1)
        self.assertEqual(len(results), 3)
        self.assertIs(results[0], results[1])
        self.assertIs(results[1], results[2])

    def test_an_index_written_during_the_build_is_not_served(self):
        """A matcher is keyed by the mtime it was loaded at, so one built from
        vectors that have since been replaced is born stale, never handed out."""
        path = self._mk_index("a")
        matcher_cls = self.AssemblyMatcher
        orig_build = matcher_cls.build_part_rarity_weights

        def build(inner_self):
            st = path.stat()
            os.utime(path, ns=(st.st_atime_ns + 10**9, st.st_mtime_ns + 10**9))
            orig_build(inner_self)

        matcher_cls.build_part_rarity_weights = build
        first = assembly._get_assembly_matcher("a")
        matcher_cls.build_part_rarity_weights = orig_build
        second = assembly._get_assembly_matcher("a")
        self.assertIsNot(first, second)


class TestAssemblyEndpoint(unittest.TestCase):
    """Router-level validation and forwarding."""

    QUERY_ID = "a" * 64

    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers.similarity import router

        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

        self._orig = assembly.search_assembly_index
        self.calls = []

        def _fake(name, file_id, top_k, candidate_k=30, sim_thresh=0.80,
                  bop_weight=0.30, coverage_mode="symmetric", use_idf=True,
                  include_self=False, tags=None,
                  tag_mode="any", tagged=None):
            self.calls.append(
                {"name": name, "file_id": file_id, "top_k": top_k,
                 "candidate_k": candidate_k, "sim_thresh": sim_thresh,
                 "bop_weight": bop_weight, "coverage_mode": coverage_mode,
                 "use_idf": use_idf, "include_self": include_self,
                 "tags": tags, "tag_mode": tag_mode, "tagged": tagged}
            )
            return {
                "hits": [{
                    "id": "b" * 64, "score": 0.9, "geom_score": 0.8,
                    "coverage": 0.7, "matched_parts": 3, "candidate_parts": 5,
                    "query_parts": 4, "metadata": {"filename": "b.stp"},
                }],
                "count": 1,
                "image_url": None,
            }

        assembly.search_assembly_index = _fake

    def tearDown(self):
        assembly.search_assembly_index = self._orig

    def _post(self, query=""):
        url = f"/similarity/index/idx/search-assembly?file_id={self.QUERY_ID}"
        if query:
            url += "&" + query
        return self.client.post(url)

    def test_defaults_match_bridge(self):
        resp = self._post()
        self.assertEqual(resp.status_code, 200)
        call = self.calls[0]
        self.assertEqual(call["top_k"], 10)
        self.assertEqual(call["candidate_k"], 30)
        self.assertAlmostEqual(call["sim_thresh"], 0.80)
        self.assertAlmostEqual(call["bop_weight"], 0.30)
        self.assertEqual(call["coverage_mode"], "symmetric")
        self.assertTrue(call["use_idf"])
        self.assertFalse(call["include_self"])

    def test_response_carries_the_diagnostic_fields(self):
        body = self._post().json()
        hit = body["hits"][0]
        self.assertEqual(hit["matched_parts"], 3)
        self.assertEqual(hit["candidate_parts"], 5)
        self.assertEqual(hit["query_parts"], 4)
        self.assertAlmostEqual(hit["geom_score"], 0.8)
        self.assertAlmostEqual(hit["coverage"], 0.7)
        self.assertEqual(body["count"], 1)
        self.assertIsNone(body["image_url"])

    def test_missing_query_is_422(self):
        resp = self.client.post("/similarity/index/idx/search-assembly")
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(self.calls, [])

    def test_invalid_coverage_mode_is_422(self):
        self.assertEqual(self._post("coverage_mode=bogus").status_code, 422)
        self.assertEqual(self.calls, [])

    def test_valid_coverage_modes_are_forwarded(self):
        for mode in ("symmetric", "containment", "jaccard"):
            self.calls.clear()
            self.assertEqual(self._post("coverage_mode=" + mode).status_code, 200)
            self.assertEqual(self.calls[0]["coverage_mode"], mode)

    def test_candidate_k_out_of_range_is_422(self):
        for bad in ("candidate_k=0", "candidate_k=1001"):
            self.assertEqual(self._post(bad).status_code, 422, bad)
        self.assertEqual(self.calls, [])

    def test_sim_thresh_out_of_range_is_422(self):
        for bad in ("sim_thresh=-0.1", "sim_thresh=1.1"):
            self.assertEqual(self._post(bad).status_code, 422, bad)
        self.assertEqual(self.calls, [])

    def test_bop_weight_out_of_range_is_422(self):
        for bad in ("bop_weight=-0.1", "bop_weight=1.1"):
            self.assertEqual(self._post(bad).status_code, 422, bad)
        self.assertEqual(self.calls, [])

    def test_core_value_error_becomes_422(self):
        def _raise(*a, **kw):
            raise ValueError("Invalid coverage_mode 'x'.")

        assembly.search_assembly_index = _raise
        resp = self._post()
        self.assertEqual(resp.status_code, 422)

    def test_unknown_index_is_404(self):
        def _raise(*a, **kw):
            raise KeyError("Index 'idx' does not exist.")

        assembly.search_assembly_index = _raise
        self.assertEqual(self._post().status_code, 404)

    def test_legacy_schema_is_409(self):
        def _raise(*a, **kw):
            raise core.SchemaVersionError("Index 'idx' uses legacy schema v1")

        assembly.search_assembly_index = _raise
        resp = self._post()
        self.assertEqual(resp.status_code, 409)
        self.assertIn("legacy schema", resp.json()["detail"])

    def test_flags_are_forwarded(self):
        self.assertEqual(self._post("include_self=true&use_idf=false").status_code, 200)
        self.assertTrue(self.calls[0]["include_self"])
        self.assertFalse(self.calls[0]["use_idf"])

    def test_endpoint_is_public(self):
        # Assembly search is a production feature served without any flag.
        self.assertEqual(self._post().status_code, 200)


class TestAssemblyTiming(_AssemblySearchBase):
    """HOOPS_SEARCH_TIMING=1 instruments the search without changing it.

    Assembly search used to carry no instrumentation at all, so a 65 s run and
    an instant one left identical, empty logs. The phases are laid out like the
    ones in ``search_index`` and share their labels where they mean the same
    thing, so the two can be read side by side.
    """

    def test_timed_search_does_not_crash(self):
        """Regression for #61, now for this module.

        ``search_assembly_index`` reaches ``diagnostics._PhaseTimer`` at module
        level while it also has a ``tags`` parameter and a ``tag_diag`` local
        that was once called ``diagnostics``. A local of that name would shadow
        the module for the whole function and every timed search would raise
        ``UnboundLocalError``. Enable the flag and assert a normal result.
        """
        with patch.dict(os.environ, {"HOOPS_SEARCH_TIMING": "1"}, clear=False):
            out = assembly.search_assembly_index("idx", self.QUERY_ID, 10)
        self.assertEqual([h["id"] for h in out["hits"]], [self.A, self.B, self.C])

    def test_timing_is_logged_under_its_own_tag(self):
        with patch.dict(os.environ, {"HOOPS_SEARCH_TIMING": "1"}, clear=False):
            with self.assertLogs("diagnostics", level="INFO") as captured:
                assembly.search_assembly_index("idx", self.QUERY_ID, 10)
        lines = [m for m in captured.output if "ASSEMBLY_TIMING" in m]
        self.assertTrue(lines, "no ASSEMBLY_TIMING line was logged")
        # Part search logs under SEARCH_TIMING; the tags are kept apart so the
        # two searches never interleave in one grep.
        self.assertFalse([m for m in captured.output if "[SEARCH_TIMING]" in m])
        joined = "\n".join(lines)
        # matcher_cached is the decisive field: the k-means behind a cold
        # matcher is the suspected cost (~19 s at 42k bodies).
        self.assertIn("matcher_cached=", joined)
        self.assertIn("query_in_index=", joined)
        self.assertIn("get_matcher", joined)
        self.assertIn("matcher_search", joined)
        self.assertIn("TOTAL", joined)

    def test_empty_index_is_timed_too(self):
        self.registered.clear()
        with patch.dict(os.environ, {"HOOPS_SEARCH_TIMING": "1"}, clear=False):
            with self.assertLogs("diagnostics", level="INFO") as captured:
                out = assembly.search_assembly_index("idx", self.QUERY_ID, 5)
        self.assertEqual(out, {"hits": [], "count": 0, "image_url": None})
        self.assertTrue([m for m in captured.output if "empty_index=True" in m])

    def test_silent_without_the_flag(self):
        env = {k: v for k, v in os.environ.items() if k != "HOOPS_SEARCH_TIMING"}
        with patch.dict(os.environ, env, clear=True):
            logger = logging.getLogger("diagnostics")
            with self.assertLogs(logger, level="INFO") as captured:
                logger.info("probe")  # assertLogs needs at least one record
                assembly.search_assembly_index("idx", self.QUERY_ID, 10)
        self.assertEqual(captured.output, ["INFO:diagnostics:probe"])


if __name__ == "__main__":
    unittest.main()
