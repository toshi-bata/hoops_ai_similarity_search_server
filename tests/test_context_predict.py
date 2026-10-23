"""Tests for core.predict_context using fakes (no SDK).

predict_context wires together the search path and the Context Layer. Here the
searcher, the index and the predictor are all fakes, so the test checks the
*plumbing* -- self-hit exclusion, the considered/with-context bookkeeping,
coverage, and the shape of the response -- without needing hoops_ai.
"""

import pathlib
import sys
import tempfile
import unittest
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import context_store  # noqa: E402
import core  # noqa: E402
import index_registry
import storage  # noqa: E402
import config

QID = "q" * 64
N1 = "1" * 64
N2 = "2" * 64
N3 = "3" * 64


class _Hit:
    def __init__(self, id, score):
        self.id = id
        self.score = score
        self.metadata = {"file_id": id}


class _FakeVS:
    def __init__(self, ids):
        self._ids = ids

    def get_ids(self):
        return list(self._ids)

    def iter_metadata(self):
        return [{"file_id": i} for i in self._ids]


class _FakeSearcher:
    def __init__(self, results):
        self._results = results

    def search_by_shape(self, path, top_k=None, **kwargs):
        return self._results


class _FakePrediction:
    def __init__(self, value, confidence, evidence_count, status, coverage):
        self.value = value
        self.confidence = confidence
        self.evidence_count = evidence_count
        self.status = status
        self.coverage = coverage
        self.alternatives = [(value, confidence)]
        self.reasons = ["fake"]


class _FakePredictor:
    def __init__(self, mapping):
        self._mapping = mapping

    def infer(self, hits, keys, **kwargs):
        return {k: self._mapping[k] for k in keys if k in self._mapping}


class TestPredictContext(unittest.TestCase):
    NAME = "idx"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = {}
        for attr in (
            "_load_named_index",
            "_get_named_searcher", "find_persistent_CAD_file",
        ):
            self._saved[attr] = getattr(storage if attr == "find_persistent_CAD_file" else index_registry, attr)
        # INDEXES_DIR lives on config now, so it is saved/restored separately.
        self._saved_idx = config.INDEXES_DIR
        self._saved_make = context_store.make_memory_context_provider
        self._saved_build = context_store.build_context_predictor

        config.INDEXES_DIR = pathlib.Path(self._tmp.name)
        self.index_dir = config.INDEXES_DIR / self.NAME
        self.index_dir.mkdir(parents=True)

        self.registered = [QID, N1, N2, N3]
        self.results = [[
            _Hit(QID, 1.0), _Hit(N1, 0.9), _Hit(N2, 0.8), _Hit(N3, 0.7),
        ]]

        index_registry._load_named_index = lambda name: _FakeVS(self.registered)
        index_registry._get_named_searcher = lambda name: _FakeSearcher(self.results)
        storage.find_persistent_CAD_file = lambda file_id: pathlib.Path("query.stp")
        context_store.make_memory_context_provider = lambda contexts, numeric_override=None: object()
        context_store.build_context_predictor = lambda index_dir, provider: _FakePredictor(
            {
                "Material": _FakePrediction("Steel", 0.8, 2, "ready", 1.0),
                "UnitCost": _FakePrediction(15.0, 0.6, 2, "ready", 1.0),
            }
        )

        # n1 and n2 carry records; n3 does not.
        context_store._context_cache.clear()
        context_store._context_locks.clear()
        context_store.set_part_context(
            self.index_dir, N1, {"Material": "Steel", "UnitCost": 10.0}
        )
        context_store.set_part_context(
            self.index_dir, N2, {"Material": "Steel", "UnitCost": 20.0}
        )

    def tearDown(self):
        for attr, value in self._saved.items():
            setattr(storage if attr == "find_persistent_CAD_file" else index_registry, attr, value)
        config.INDEXES_DIR = self._saved_idx
        context_store.make_memory_context_provider = self._saved_make
        context_store.build_context_predictor = self._saved_build
        context_store._context_cache.clear()
        context_store._context_locks.clear()
        self._tmp.cleanup()

    def test_self_excluded_and_bookkeeping(self):
        result = core.predict_context(self.NAME, QID, ["Material", "UnitCost"], top_k=10)
        # q is the query and must not appear as its own neighbour.
        self.assertNotIn(QID, [h["id"] for h in result["hits"]])
        self.assertEqual(result["candidates_examined"], 3)  # n1, n2, n3
        self.assertEqual(result["hits_considered"], 3)
        self.assertEqual(result["hits_with_context"], 2)   # n1, n2
        self.assertAlmostEqual(result["coverage"], 2 / 3, places=4)

    def test_predictions_and_hit_values(self):
        result = core.predict_context(self.NAME, QID, ["Material", "UnitCost"], top_k=10)
        self.assertEqual(result["predictions"]["Material"]["value"], "Steel")
        self.assertEqual(result["predictions"]["UnitCost"]["evidence_count"], 2)
        # alternatives (a tuple in the fake) is coerced to JSON-safe form.
        self.assertIsInstance(result["predictions"]["Material"]["alternatives"], list)
        hit_ids = {h["id"] for h in result["hits"]}
        self.assertEqual(hit_ids, {N1, N2})
        for h in result["hits"]:
            self.assertEqual(set(h["values"]), {"Material", "UnitCost"})

    def test_empty_index_raises(self):
        index_registry._load_named_index = lambda name: _FakeVS([])
        with self.assertRaises(ValueError):
            core.predict_context(self.NAME, QID, ["Material"], top_k=10)

    def test_no_context_still_returns_structure(self):
        # Remove all records: prediction should be empty but well-formed.
        context_store.set_part_context(self.index_dir, N1, {})
        context_store.set_part_context(self.index_dir, N2, {})
        result = core.predict_context(self.NAME, QID, ["Material"], top_k=10)
        self.assertEqual(result["hits_with_context"], 0)
        self.assertEqual(result["predictions"], {})
        self.assertEqual(result["hits"], [])

    def test_bad_keys_rejected(self):
        with self.assertRaises(ValueError):
            core.predict_context(self.NAME, QID, [], top_k=10)

    def test_bad_kind_rejected(self):
        with self.assertRaises(ValueError):
            core.predict_context(self.NAME, QID, ["Material"], kind="widget")

    def test_kind_and_status_policy_flow_through(self):
        seen = {}

        class _CapturingSearcher:
            def __init__(self, results):
                self._results = results

            def search_by_shape(self, path, top_k=None, **kwargs):
                seen["search_kwargs"] = kwargs
                return self._results

        class _CapturingPredictor:
            def infer(self, hits, keys, **kwargs):
                seen["infer_kwargs"] = kwargs
                return {}

        index_registry._get_named_searcher = lambda name: _CapturingSearcher(self.results)
        context_store.build_context_predictor = (
            lambda index_dir, provider: _CapturingPredictor()
        )
        core.predict_context(
            self.NAME, QID, ["Material"], top_k=10, kind="part",
            status_policy={"min_top_share": 0.5, "not_allowed": 1},
        )
        # kind is forwarded as a search filter.
        self.assertEqual(seen["search_kwargs"].get("filters"), {"kind": "part"})
        # status_policy reaches infer, but only whitelisted keys survive.
        self.assertEqual(
            seen["infer_kwargs"].get("status_policy"), {"min_top_share": 0.5}
        )

    def test_any_kind_sends_no_filter(self):
        seen = {}

        class _CapturingSearcher:
            def __init__(self, results):
                self._results = results

            def search_by_shape(self, path, top_k=None, **kwargs):
                seen["search_kwargs"] = kwargs
                return self._results

        index_registry._get_named_searcher = lambda name: _CapturingSearcher(self.results)
        core.predict_context(self.NAME, QID, ["Material"], top_k=10, kind="any")
        self.assertNotIn("filters", seen["search_kwargs"])

    def test_neighbor_ids_pin_the_considered_set(self):
        # Pinning to n1 and n3 keeps exactly those (search order), ignoring n2,
        # and does not apply a top_k cap. n3 carries no record, so only n1 feeds
        # the prediction.
        result = core.predict_context(
            self.NAME, QID, ["Material", "UnitCost"], top_k=10,
            neighbor_ids=[N1, N3],
        )
        self.assertEqual(result["hits_considered"], 2)
        self.assertEqual(result["hits_with_context"], 1)
        self.assertEqual({h["id"] for h in result["hits"]}, {N1})
        # candidates_examined still counts the whole search pool minus self.
        self.assertEqual(result["candidates_examined"], 3)

    def test_neighbor_ids_drop_self_and_unknown(self):
        # The query id and any id not in the search pool are silently skipped, so
        # only n2 remains.
        result = core.predict_context(
            self.NAME, QID, ["Material"], top_k=10,
            neighbor_ids=[QID, N2, "z" * 64],
        )
        self.assertEqual(result["hits_considered"], 1)
        self.assertEqual({h["id"] for h in result["hits"]}, {N2})

    def test_neighbor_ids_skip_tag_filter(self):
        # With pinned neighbours the tag filter must not further narrow the set:
        # a tag filter that would match nothing still yields the pinned hits.
        result = core.predict_context(
            self.NAME, QID, ["Material"], top_k=10,
            neighbor_ids=[N1, N2], tags=["nonexistent-tag"],
        )
        self.assertEqual({h["id"] for h in result["hits"]}, {N1, N2})

    def test_hits_carry_filename_and_predictions_injected_context(self):
        result = core.predict_context(self.NAME, QID, ["Material"], top_k=10)
        for h in result["hits"]:
            self.assertIn("filename", h)
            self.assertIn("has_thumbnail", h)
        self.assertIn("injected_context", result["predictions"]["Material"])

    def test_numpy_prediction_values_are_json_serializable(self):
        # The SDK returns numpy scalars for numeric predictions; numpy.float64
        # even passes an isinstance(float) check, so it must be unwrapped or the
        # response fails to serialize (a plain 500 after the endpoint's try).
        class _NumpyLikeScalar:
            def __init__(self, value):
                self._value = value

            def item(self):
                return self._value

        _NumpyLikeScalar.__module__ = "numpy"

        context_store.build_context_predictor = lambda index_dir, provider: _FakePredictor(
            {
                "UnitCost": _FakePrediction(
                    _NumpyLikeScalar(15.5), _NumpyLikeScalar(0.6), 2, "ready", 1.0
                ),
            }
        )
        result = core.predict_context(self.NAME, QID, ["UnitCost"], top_k=10)
        pred = result["predictions"]["UnitCost"]
        self.assertEqual(pred["value"], 15.5)
        self.assertIsInstance(pred["value"], float)
        self.assertNotIsInstance(pred["value"], _NumpyLikeScalar)
        # The whole result must round-trip through JSON, as the response does.
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
