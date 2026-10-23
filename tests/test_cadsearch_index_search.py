"""SDK-free unit tests for CADSearch-backed named-index search.

Covers the pieces introduced when ``search_index()`` moved from
``FaissVectorStore.query()`` to ``CADSearch.search_by_shape()``:

  * bridge-identical flattening of ``List[List[VectorHit]]`` (no re-sorting)
  * de-duplication by hit id and early stop at ``top_k``
  * the ``kind`` filter kwarg and the ``include_self`` promotion
  * CADSearch instance caching (mtime invalidation, LRU cap)

CADSearch and the vector store are faked; no HOOPS AI license is required.
"""

import os
import pathlib
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core
import search
import index_registry
import embeddings
import storage
import config


def _vh(id_, score, **meta):
    """Fake VectorHit (public attrs observed on hoops_ai 1.1.0)."""
    md = {"file_id": id_}
    md.update(meta)
    return SimpleNamespace(id=id_, score=score, match_count=1, metadata=md)


class TestFlattenBodyHits(unittest.TestCase):
    """Must match hoops_ai_native_bridge's run_part_search() exactly."""

    def test_preserves_body_order_and_does_not_resort(self):
        # The 2nd body list starts with a *higher* score than anything in the
        # 1st. Re-sorting would hoist it to the front; the bridge does not.
        results = [
            [_vh("a", 0.50), _vh("b", 0.40)],
            [_vh("c", 0.99), _vh("d", 0.10)],
        ]
        out = core._flatten_body_hits(results, top_k=10)
        self.assertEqual([h["id"] for h in out], ["a", "b", "c", "d"])
        self.assertEqual(out[0]["score"], 0.5)
        # Proof that ordering is not by score.
        self.assertLess(out[0]["score"], out[2]["score"])

    def test_dedups_ids_across_body_lists(self):
        results = [
            [_vh("a", 0.9), _vh("b", 0.8)],
            [_vh("b", 0.95), _vh("a", 0.7), _vh("c", 0.6)],
        ]
        out = core._flatten_body_hits(results, top_k=10)
        self.assertEqual([h["id"] for h in out], ["a", "b", "c"])
        # First occurrence wins, so b keeps 0.8 (not the later 0.95).
        self.assertEqual(out[1]["score"], 0.8)

    def test_stops_as_soon_as_top_k_reached(self):
        results = [
            [_vh("a", 0.9), _vh("b", 0.8), _vh("c", 0.7)],
            [_vh("d", 0.6)],
        ]
        out = core._flatten_body_hits(results, top_k=2)
        self.assertEqual([h["id"] for h in out], ["a", "b"])

    def test_keeps_metadata(self):
        results = [[_vh("a", 0.9, kind="part", filename="a.stp")]]
        out = core._flatten_body_hits(results, top_k=5)
        self.assertEqual(out[0]["metadata"]["kind"], "part")
        self.assertEqual(out[0]["metadata"]["filename"], "a.stp")

    def test_skips_entries_without_id(self):
        results = [[SimpleNamespace(id="", score=0.9), _vh("a", 0.5)]]
        out = core._flatten_body_hits(results, top_k=5)
        self.assertEqual([h["id"] for h in out], ["a"])

    def test_empty_and_none_inputs(self):
        self.assertEqual(core._flatten_body_hits([], top_k=5), [])
        self.assertEqual(core._flatten_body_hits(None, top_k=5), [])
        self.assertEqual(core._flatten_body_hits([[], None], top_k=5), [])

    def test_non_positive_top_k_returns_one(self):
        results = [[_vh("a", 0.9), _vh("b", 0.8)]]
        self.assertEqual(len(core._flatten_body_hits(results, top_k=0)), 1)


class _FakeSearcher:
    """Records the kwargs search_by_shape() was called with."""

    def __init__(self, results):
        self._results = results
        self.calls = []

    def search_by_shape(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return self._results


class TestSearchIndex(unittest.TestCase):
    QUERY_ID = "q" * 64

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_idx_dir = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmp.name)

        self.results = [
            [_vh("a" * 64, 0.9), _vh("b" * 64, 0.8)],
            [_vh("c" * 64, 0.95)],
        ]
        self.searcher = _FakeSearcher(self.results)

        self.registered = {"a" * 64, "b" * 64, "c" * 64}
        self.metas = [{"file_id": i, "kind": "part", "filename": "x.stp"} for i in sorted(self.registered)]

        vs = MagicMock()
        vs.get_ids.side_effect = lambda: list(self.registered)
        vs.iter_metadata.side_effect = lambda: iter(self.metas)

        self._saved = {
            "_load_named_index": index_registry._load_named_index,
            "_get_named_searcher": index_registry._get_named_searcher,
            "find_persistent_CAD_file": storage.find_persistent_CAD_file,
            "_resolve_index_asset": index_registry._resolve_index_asset,
        }
        index_registry._load_named_index = lambda name: vs
        index_registry._get_named_searcher = lambda name: self.searcher
        storage.find_persistent_CAD_file = lambda fid: pathlib.Path(self._tmp.name) / f"{fid}.stp"
        index_registry._resolve_index_asset = lambda name, pid, asset: None

    def tearDown(self):
        for attr, val in self._saved.items():
            setattr(storage if attr == "find_persistent_CAD_file" else index_registry, attr, val)
        config.INDEXES_DIR = self._orig_idx_dir
        self._tmp.cleanup()

    def test_kind_any_passes_no_filters_kwarg(self):
        search.search_index("idx", self.QUERY_ID, 10, kind="any")
        _path, kwargs = self.searcher.calls[0]
        self.assertNotIn("filters", kwargs)
        # One extra candidate is requested so that dropping the self hit does
        # not shrink the result below top_k.
        self.assertEqual(kwargs["top_k"], 11)

    def test_kind_part_passes_filters(self):
        search.search_index("idx", self.QUERY_ID, 5, kind="part")
        _path, kwargs = self.searcher.calls[0]
        self.assertEqual(kwargs["filters"], {"kind": "part"})

    def test_kind_assembly_passes_filters(self):
        search.search_index("idx", self.QUERY_ID, 5, kind="assembly")
        _path, kwargs = self.searcher.calls[0]
        self.assertEqual(kwargs["filters"], {"kind": "assembly"})

    def test_invalid_kind_raises_value_error(self):
        with self.assertRaises(ValueError):
            search.search_index("idx", self.QUERY_ID, 5, kind="bogus")

    def test_search_receives_resolved_path_not_file_id(self):
        search.search_index("idx", self.QUERY_ID, 5)
        path, _kwargs = self.searcher.calls[0]
        self.assertTrue(path.endswith(f"{self.QUERY_ID}.stp"))

    def test_results_keep_body_order(self):
        out = search.search_index("idx", self.QUERY_ID, 10)
        self.assertEqual([h["id"] for h in out["hits"]], ["a" * 64, "b" * 64, "c" * 64])
        self.assertEqual(out["count"], 3)

    def test_empty_index_returns_no_hits(self):
        self.registered.clear()
        out = search.search_index("idx", self.QUERY_ID, 10)
        self.assertEqual(out, {"hits": [], "count": 0, "image_url": None})
        self.assertEqual(self.searcher.calls, [])

    def test_include_self_inserts_registered_query_first(self):
        self.registered.add(self.QUERY_ID)
        self.metas.append({"file_id": self.QUERY_ID, "kind": "assembly", "filename": "q.stp"})
        out = search.search_index("idx", self.QUERY_ID, 10, include_self=True)
        self.assertEqual(out["hits"][0]["id"], self.QUERY_ID)
        self.assertEqual(out["hits"][0]["score"], 1.0)
        self.assertEqual(out["hits"][0]["metadata"]["filename"], "q.stp")

    def test_include_self_respects_top_k(self):
        self.registered.add(self.QUERY_ID)
        out = search.search_index("idx", self.QUERY_ID, 2, include_self=True)
        self.assertEqual(len(out["hits"]), 2)
        self.assertEqual(out["hits"][0]["id"], self.QUERY_ID)

    def test_include_self_noop_when_query_not_registered(self):
        out = search.search_index("idx", self.QUERY_ID, 10, include_self=True)
        self.assertNotIn(self.QUERY_ID, [h["id"] for h in out["hits"]])
        self.assertEqual(len(out["hits"]), 3)

    def test_include_self_does_not_duplicate_existing_hit(self):
        # Query already comes back as a normal hit; it must be promoted, not doubled.
        self.registered.add(self.QUERY_ID)
        self.results[1].append(_vh(self.QUERY_ID, 0.42))
        out = search.search_index("idx", self.QUERY_ID, 10, include_self=True)
        ids = [h["id"] for h in out["hits"]]
        self.assertEqual(ids.count(self.QUERY_ID), 1)
        self.assertEqual(ids[0], self.QUERY_ID)

    def test_include_self_false_by_default(self):
        self.registered.add(self.QUERY_ID)
        out = search.search_index("idx", self.QUERY_ID, 10)
        self.assertNotEqual(out["hits"][0]["id"], self.QUERY_ID)


class TestSelfHitAndMetadata(unittest.TestCase):
    """include_self semantics and stripping of SDK-internal metadata keys."""

    QUERY_ID = "q" * 64

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_idx_dir = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmp.name)

        # The query itself is returned by CADSearch as a perfect self match:
        # record ids are SHA-256 while the query is an uploads/<sha>_<name>
        # path, so the SDK cannot recognise them as the same object.
        self.self_hit_meta = {
            "file_id": self.QUERY_ID,
            "filename": "1.stp",
            "kind": "assembly",
            "_query_body_rep_count": 1,
            "_query_body_rep_indices": [0],
        }
        self.results = [
            [
                _vh(self.QUERY_ID, 1.0, **self.self_hit_meta),
                _vh("a" * 64, 0.64, file_id="a" * 64, filename="100.stp",
                    kind="assembly", _query_body_rep_count=1),
            ],
            [
                _vh("b" * 64, 0.41, file_id="b" * 64, filename="10.stp",
                    kind="assembly", _query_body_rep_indices=[0]),
                _vh("c" * 64, 0.30, file_id="c" * 64, filename="2.stp",
                    kind="part"),
            ],
        ]
        self.searcher = _FakeSearcher(self.results)
        self.registered = {self.QUERY_ID, "a" * 64, "b" * 64, "c" * 64}
        self.metas = [
            {"file_id": self.QUERY_ID, "filename": "1.stp", "kind": "assembly"},
        ]

        vs = MagicMock()
        vs.get_ids.side_effect = lambda: list(self.registered)
        vs.iter_metadata.side_effect = lambda: iter(self.metas)

        self._saved = {
            "_load_named_index": index_registry._load_named_index,
            "_get_named_searcher": index_registry._get_named_searcher,
            "find_persistent_CAD_file": storage.find_persistent_CAD_file,
            "_resolve_index_asset": index_registry._resolve_index_asset,
        }
        index_registry._load_named_index = lambda name: vs
        index_registry._get_named_searcher = lambda name: self.searcher
        storage.find_persistent_CAD_file = lambda fid: pathlib.Path(self._tmp.name) / f"{fid}.stp"
        index_registry._resolve_index_asset = lambda name, pid, asset: pathlib.Path("exists.png")

    def tearDown(self):
        for attr, val in self._saved.items():
            setattr(storage if attr == "find_persistent_CAD_file" else index_registry, attr, val)
        config.INDEXES_DIR = self._orig_idx_dir
        self._tmp.cleanup()

    def test_self_hit_excluded_by_default(self):
        out = search.search_index("idx", self.QUERY_ID, 3)
        self.assertNotIn(self.QUERY_ID, [h["id"] for h in out["hits"]])

    def test_excluding_self_still_fills_top_k(self):
        out = search.search_index("idx", self.QUERY_ID, 3)
        self.assertEqual(len(out["hits"]), 3)
        self.assertEqual(
            [h["metadata"]["filename"] for h in out["hits"]],
            ["100.stp", "10.stp", "2.stp"],
        )
        # An extra candidate must be requested to compensate for the self hit.
        self.assertEqual(self.searcher.calls[0][1]["top_k"], 4)

    def test_include_self_promotes_without_duplicating(self):
        out = search.search_index("idx", self.QUERY_ID, 3, include_self=True)
        ids = [h["id"] for h in out["hits"]]
        self.assertEqual(ids[0], self.QUERY_ID)
        self.assertEqual(ids.count(self.QUERY_ID), 1)
        self.assertEqual(out["hits"][0]["score"], 1.0)

    def test_include_self_does_not_exceed_top_k(self):
        out = search.search_index("idx", self.QUERY_ID, 3, include_self=True)
        self.assertEqual(len(out["hits"]), 3)
        self.assertEqual(out["count"], 3)

    def test_no_underscore_keys_in_any_hit(self):
        for include_self in (False, True):
            out = search.search_index("idx", self.QUERY_ID, 4, include_self=include_self)
            for hit in out["hits"]:
                leaked = [k for k in hit["metadata"] if k.startswith("_")]
                self.assertEqual(leaked, [], f"include_self={include_self}: {leaked}")

    def test_self_hit_metadata_matches_normal_hit_shape(self):
        out = search.search_index("idx", self.QUERY_ID, 3, include_self=True)
        self.assertEqual(
            set(out["hits"][0]["metadata"]),
            {"file_id", "filename", "kind"},
        )
        self.assertEqual(
            set(out["hits"][0]["metadata"]), set(out["hits"][1]["metadata"])
        )

    def test_source_metadata_is_not_mutated(self):
        search.search_index("idx", self.QUERY_ID, 4)
        self.assertIn("_query_body_rep_count", self.results[0][0].metadata)
        self.assertIn("_query_body_rep_count", self.results[0][1].metadata)


class TestPublicMetadata(unittest.TestCase):
    def test_strips_underscore_keys_and_copies(self):
        src = {"file_id": "x", "_internal": 1}
        out = core._public_metadata(src)
        self.assertEqual(out, {"file_id": "x"})
        self.assertIsNot(out, src)
        self.assertIn("_internal", src)

    def test_non_dict_returns_empty(self):
        self.assertEqual(core._public_metadata(None), {})
        self.assertEqual(core._public_metadata(["a"]), {})


class TestNamedSearcherCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_idx_dir = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmp.name)
        self._orig_get_embedder = embeddings.get_embedder
        embeddings.get_embedder = lambda: SimpleNamespace(model_name="fake")

        # Fake hoops_ai.ml.embeddings.CADSearch
        self._saved_mods = {}
        for n in ("hoops_ai", "hoops_ai.ml", "hoops_ai.ml.embeddings"):
            self._saved_mods[n] = sys.modules.get(n)
        pkg = types.ModuleType("hoops_ai")
        ml = types.ModuleType("hoops_ai.ml")
        emb = types.ModuleType("hoops_ai.ml.embeddings")

        class _CADSearch:
            instances = 0

            def __init__(self, shape_model=None):
                type(self).instances += 1
                self.serial = type(self).instances
                self.loaded = None

            def load_shape_index(self, path=None):
                self.loaded = path

        emb.CADSearch = _CADSearch
        pkg.ml = ml
        ml.embeddings = emb
        sys.modules["hoops_ai"] = pkg
        sys.modules["hoops_ai.ml"] = ml
        sys.modules["hoops_ai.ml.embeddings"] = emb
        self.CADSearch = _CADSearch

        core._named_searchers.clear()

    def tearDown(self):
        core._named_searchers.clear()
        embeddings.get_embedder = self._orig_get_embedder
        config.INDEXES_DIR = self._orig_idx_dir
        for n, m in self._saved_mods.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
        self._tmp.cleanup()

    def _mk_index(self, name):
        d = config.INDEXES_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.faiss").write_bytes(b"faiss")
        return d / "index.faiss"

    def test_missing_index_raises_key_error(self):
        with self.assertRaises(KeyError):
            index_registry._get_named_searcher("nope")

    def test_same_mtime_reuses_instance(self):
        self._mk_index("a")
        s1 = index_registry._get_named_searcher("a")
        s2 = index_registry._get_named_searcher("a")
        self.assertIs(s1, s2)
        self.assertEqual(self.CADSearch.instances, 1)

    def test_index_is_loaded_from_faiss_path(self):
        p = self._mk_index("a")
        s = index_registry._get_named_searcher("a")
        self.assertEqual(s.loaded, str(p))

    def test_mtime_change_rebuilds_instance(self):
        p = self._mk_index("a")
        s1 = index_registry._get_named_searcher("a")
        st = p.stat()
        os.utime(p, ns=(st.st_atime_ns + 10**9, st.st_mtime_ns + 10**9))
        s2 = index_registry._get_named_searcher("a")
        self.assertIsNot(s1, s2)
        # The stale generation must not linger in the cache.
        self.assertEqual(len(core._named_searchers), 1)

    def test_lru_cap_evicts_oldest(self):
        for n in ("a", "b", "c"):
            self._mk_index(n)
        index_registry._get_named_searcher("a")
        index_registry._get_named_searcher("b")
        index_registry._get_named_searcher("c")
        self.assertEqual(len(core._named_searchers), core._NAMED_SEARCHER_CACHE_MAX)
        paths = [k[0] for k in core._named_searchers]
        self.assertNotIn(str(config.INDEXES_DIR / "a" / "index.faiss"), paths)

    def test_recent_use_survives_eviction(self):
        for n in ("a", "b", "c"):
            self._mk_index(n)
        sa = index_registry._get_named_searcher("a")
        index_registry._get_named_searcher("b")
        index_registry._get_named_searcher("a")  # refresh a
        index_registry._get_named_searcher("c")  # evicts b, not a
        paths = [k[0] for k in core._named_searchers]
        self.assertIn(str(config.INDEXES_DIR / "a" / "index.faiss"), paths)
        self.assertNotIn(str(config.INDEXES_DIR / "b" / "index.faiss"), paths)
        self.assertIs(index_registry._get_named_searcher("a"), sa)

    def test_evict_named_searchers_drops_entry(self):
        self._mk_index("a")
        index_registry._get_named_searcher("a")
        core._evict_named_searchers("a")
        self.assertEqual(len(core._named_searchers), 0)


class TestSearchEndpointParams(unittest.TestCase):
    """Router-level validation of the new `kind` / `include_self` parameters."""

    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers.similarity import router

        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

        self._orig = search.search_index
        self.calls = []

        def _fake(name, file_id, top_k, kind="any", include_self=False,
                  tags=None, tag_mode="any", tagged=None):
            self.calls.append(
                {"name": name, "file_id": file_id, "top_k": top_k,
                 "kind": kind, "include_self": include_self,
                 "tags": tags, "tag_mode": tag_mode, "tagged": tagged}
            )
            return {"hits": [], "count": 0, "image_url": None}

        search.search_index = _fake

    def tearDown(self):
        search.search_index = self._orig

    def _post(self, query):
        return self.client.post("/similarity/index/idx/search?file_id={}&{}".format("a" * 64, query))

    def test_invalid_kind_is_422(self):
        resp = self._post("kind=bogus")
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(self.calls, [])

    def test_kind_defaults_to_any(self):
        self.assertEqual(self._post("top_k=3").status_code, 200)
        self.assertEqual(self.calls[0]["kind"], "any")
        self.assertFalse(self.calls[0]["include_self"])

    def test_valid_kinds_are_forwarded(self):
        for k in ("any", "part", "assembly"):
            self.calls.clear()
            self.assertEqual(self._post("kind=" + k).status_code, 200)
            self.assertEqual(self.calls[0]["kind"], k)

    def test_include_self_is_forwarded(self):
        self.assertEqual(self._post("include_self=true").status_code, 200)
        self.assertTrue(self.calls[0]["include_self"])


class TestWarmEndpoint(unittest.TestCase):
    """POST /index/{name}/warm reuses the exact search path and reports status."""

    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers.similarity import router

        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

        self._orig = search.search_index
        self._orig_enabled = search._search_cache_enabled
        self.calls = []

        def _fake(name, file_id, top_k, kind="any", include_self=False,
                  tags=None, tag_mode="any", tagged=None):
            self.calls.append(
                {"name": name, "file_id": file_id, "top_k": top_k, "kind": kind,
                 "include_self": include_self, "tags": tags,
                 "tag_mode": tag_mode, "tagged": tagged}
            )
            return {"hits": [{"id": "x", "score": 0.5, "metadata": {}}],
                    "count": 1, "image_url": None}

        search.search_index = _fake
        search._search_cache_enabled = lambda: True

    def tearDown(self):
        search.search_index = self._orig
        search._search_cache_enabled = self._orig_enabled

    def test_warm_forwards_the_same_params(self):
        resp = self.client.post(
            "/similarity/index/idx/warm?file_id={}&top_k=7&kind=part&tags=a,b&tag_mode=all".format("a" * 64)
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "warmed")
        self.assertEqual(body["count"], 1)
        self.assertTrue(body["cache_enabled"])
        self.assertEqual(self.calls[0]["top_k"], 7)
        self.assertEqual(self.calls[0]["kind"], "part")
        self.assertEqual(self.calls[0]["tags"], ["a", "b"])
        self.assertEqual(self.calls[0]["tag_mode"], "all")

    def test_warm_requires_file_id(self):
        resp = self.client.post("/similarity/index/idx/warm")
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(self.calls, [])

    def test_warm_reports_cache_disabled(self):
        search._search_cache_enabled = lambda: False
        resp = self.client.post("/similarity/index/idx/warm?file_id={}".format("a" * 64))
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["cache_enabled"])


class TestSearchResultCache(unittest.TestCase):
    """SDK-free tests for the part-search response cache (#42)."""

    QUERY_ID = "q" * 64

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_idx_dir = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmp.name)
        # A real .faiss file so the cache key's mtime component resolves.
        d = config.INDEXES_DIR / "idx"
        d.mkdir(parents=True, exist_ok=True)
        self.faiss = d / "index.faiss"
        self.faiss.write_bytes(b"faiss")

        self.results = [[_vh("a" * 64, 0.9), _vh("b" * 64, 0.8)], [_vh("c" * 64, 0.95)]]
        self.searcher = _FakeSearcher(self.results)
        self.registered = {"a" * 64, "b" * 64, "c" * 64}

        vs = MagicMock()
        vs.get_ids.side_effect = lambda: list(self.registered)
        vs.iter_metadata.side_effect = lambda: iter(
            [{"file_id": i, "kind": "part"} for i in sorted(self.registered)]
        )

        self._saved = {
            "_load_named_index": index_registry._load_named_index,
            "_get_named_searcher": index_registry._get_named_searcher,
            "find_persistent_CAD_file": storage.find_persistent_CAD_file,
        }
        index_registry._load_named_index = lambda name: vs
        index_registry._get_named_searcher = lambda name: self.searcher
        storage.find_persistent_CAD_file = lambda fid: pathlib.Path(self._tmp.name) / f"{fid}.stp"

        core.clear_search_cache()
        self._env = patch.dict(os.environ, {"HOOPS_AI_SEARCH_CACHE": "1"}, clear=False)
        self._env.start()

    def tearDown(self):
        self._env.stop()
        core.clear_search_cache()
        for attr, val in self._saved.items():
            setattr(storage if attr == "find_persistent_CAD_file" else index_registry, attr, val)
        config.INDEXES_DIR = self._orig_idx_dir
        self._tmp.cleanup()

    def test_repeat_search_is_served_from_cache(self):
        a = search.search_index("idx", self.QUERY_ID, 10)
        b = search.search_index("idx", self.QUERY_ID, 10)
        self.assertEqual(len(self.searcher.calls), 1)  # second call did not search
        self.assertEqual(a, b)

    def test_faiss_mtime_change_invalidates(self):
        search.search_index("idx", self.QUERY_ID, 10)
        st = self.faiss.stat()
        os.utime(self.faiss, ns=(st.st_atime_ns + 10**9, st.st_mtime_ns + 10**9))
        search.search_index("idx", self.QUERY_ID, 10)
        self.assertEqual(len(self.searcher.calls), 2)

    def test_tagging_invalidates_via_etag(self):
        search.search_index("idx", self.QUERY_ID, 10)
        # Write a tag sidecar: the .faiss is untouched, only the ETag changes.
        core.set_part_tags("idx", "a" * 64, ["bracket"])
        search.search_index("idx", self.QUERY_ID, 10)
        self.assertEqual(len(self.searcher.calls), 2)

    def test_disabled_env_bypasses_cache(self):
        with patch.dict(os.environ, {"HOOPS_AI_SEARCH_CACHE": "0"}, clear=False):
            search.search_index("idx", self.QUERY_ID, 10)
            search.search_index("idx", self.QUERY_ID, 10)
        self.assertEqual(len(self.searcher.calls), 2)

    def test_different_params_are_separate_entries(self):
        search.search_index("idx", self.QUERY_ID, 5)
        search.search_index("idx", self.QUERY_ID, 10)
        search.search_index("idx", self.QUERY_ID, 5, kind="part")
        self.assertEqual(len(self.searcher.calls), 3)

    def test_tag_order_shares_one_entry(self):
        self.registered.add(self.QUERY_ID)
        core.set_part_tags("idx", "a" * 64, ["x", "y"])
        search.search_index("idx", self.QUERY_ID, 10, tags=["x", "y"], tag_mode="all")
        n = len(self.searcher.calls)
        search.search_index("idx", self.QUERY_ID, 10, tags=["y", "x"], tag_mode="all")
        self.assertEqual(len(self.searcher.calls), n)  # order-insensitive key

    def test_returned_dict_is_isolated_copy(self):
        a = search.search_index("idx", self.QUERY_ID, 10)
        a["hits"].append({"id": "poison", "score": 0.0, "metadata": {}})
        a["count"] = 999
        b = search.search_index("idx", self.QUERY_ID, 10)
        self.assertEqual(len(self.searcher.calls), 1)  # still a cache hit
        self.assertNotIn("poison", [h["id"] for h in b["hits"]])
        self.assertEqual(b["count"], 3)

    def test_timed_search_does_not_crash(self):
        """Regression for #61: HOOPS_SEARCH_TIMING=1 must not raise.

        ``search_index`` builds a ``_PhaseTimer`` via ``diagnostics._PhaseTimer()``
        only when timing is enabled. A local later named ``diagnostics`` used to
        shadow the module for the whole function, so with timing on the module
        attribute access raised ``UnboundLocalError`` and every timed search
        crashed. Enable the flag and assert a normal result comes back.
        """
        with patch.dict(os.environ, {"HOOPS_SEARCH_TIMING": "1"}, clear=False):
            result = search.search_index("idx", self.QUERY_ID, 10)
        self.assertEqual(result["count"], 3)


if __name__ == "__main__":
    unittest.main()
