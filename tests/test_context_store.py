"""Tests for the context sidecar (indexes/<name>/context.json).

Context records are held in a sidecar for the same reason tags are (see
test_index_tags.py): writing them into the FaissVectorStore metadata would call
save(), rewrite the .faiss file and force a 19.3 s assembly-matcher rebuild.
These tests pin the behaviour the sidecar has to provide. They import only
``context_store``, which is SDK-free.
"""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import context_store  # noqa: E402


ID_A = "a" * 64
ID_B = "b" * 64
ID_C = "c" * 64


class _StoreBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.index_dir = pathlib.Path(self._tmp.name)
        context_store._context_cache.clear()
        context_store._context_locks.clear()

    def tearDown(self):
        context_store._context_cache.clear()
        context_store._context_locks.clear()
        self._tmp.cleanup()

    def raw(self):
        return json.loads(context_store.context_path(self.index_dir).read_text("utf-8"))


class TestStatusPolicy(unittest.TestCase):
    def test_clean_keeps_only_whitelisted_numbers(self):
        cleaned = context_store.clean_status_policy(
            {"min_top_share": 0.6, "min_margin": 0.1, "bogus": 1, "min_observed_hits": True}
        )
        self.assertEqual(cleaned, {"min_top_share": 0.6, "min_margin": 0.1})

    def test_clean_non_mapping_is_empty(self):
        self.assertEqual(context_store.clean_status_policy("nope"), {})
        self.assertEqual(context_store.clean_status_policy(None), {})

    def test_resolve_merges_override_over_rules(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        index_dir = pathlib.Path(tmp.name)
        context_store.rules_path(index_dir).write_text(
            json.dumps({"status_policy": {"min_top_share": 0.4, "min_margin": 0.2}}),
            encoding="utf-8",
        )
        resolved = context_store.resolve_status_policy(
            index_dir, {"min_top_share": 0.7}
        )
        # override wins for min_top_share, rules value kept for min_margin
        self.assertEqual(resolved, {"min_top_share": 0.7, "min_margin": 0.2})


class TestUnits(unittest.TestCase):
    def test_units_preserved_and_summarised(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        index_dir = pathlib.Path(tmp.name)
        context_store._context_cache.clear()
        doc = {
            "version": 1,
            "units": {"UnitCost": "USD", "LeadTimeDays": "days", "bad": 5},
            "parts": {ID_A: {"context": {"UnitCost": 10.0}}},
        }
        context_store.context_path(index_dir).write_text(
            json.dumps(doc), encoding="utf-8"
        )
        summary, _etag = context_store.context_summary(index_dir)
        # non-string unit label is dropped; valid ones are surfaced.
        self.assertEqual(
            summary["units"], {"UnitCost": "USD", "LeadTimeDays": "days"}
        )

    def test_units_survive_a_part_write(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        index_dir = pathlib.Path(tmp.name)
        context_store._context_cache.clear()
        doc = {"version": 1, "units": {"UnitCost": "EUR"}, "parts": {}}
        context_store.context_path(index_dir).write_text(
            json.dumps(doc), encoding="utf-8"
        )
        context_store.set_part_context(index_dir, ID_A, {"UnitCost": 5.0})
        summary, _etag = context_store.context_summary(index_dir)
        self.assertEqual(summary["units"], {"UnitCost": "EUR"})


class TestValueValidation(unittest.TestCase):
    def test_key_rules(self):
        self.assertEqual(context_store.normalize_key("  Material "), "Material")
        for bad in ("", "   ", "a" * 65, "bad\nkey"):
            with self.assertRaises(ValueError):
                context_store.normalize_key(bad)
        with self.assertRaises(ValueError):
            context_store.normalize_key(123)

    def test_scalar_values_accepted(self):
        ctx = context_store.normalize_context(
            {"Material": " Steel ", "UnitCost": 12.5, "LotSize": 10, "Buy": True, "X": None}
        )
        self.assertEqual(ctx["Material"], "Steel")
        self.assertEqual(ctx["UnitCost"], 12.5)
        self.assertIs(ctx["Buy"], True)
        self.assertIsNone(ctx["X"])

    def test_non_scalar_and_infinite_rejected(self):
        with self.assertRaises(ValueError):
            context_store.normalize_context({"k": {"nested": 1}})
        with self.assertRaises(ValueError):
            context_store.normalize_context({"k": [1, 2]})
        with self.assertRaises(ValueError):
            context_store.normalize_context({"k": float("inf")})

    def test_too_many_keys_rejected(self):
        many = {f"k{i}": i for i in range(context_store.max_keys_per_part() + 1)}
        with self.assertRaises(ValueError):
            context_store.normalize_context(many)


class TestReadWrite(_StoreBase):
    def test_missing_file_is_empty_document(self):
        doc, etag = context_store.read_context_document(self.index_dir)
        self.assertEqual(doc["parts"], {})
        self.assertTrue(etag.startswith('"'))

    def test_set_and_get_roundtrip(self):
        result, etag = context_store.set_part_context(
            self.index_dir, ID_A, {"Material": "Steel", "UnitCost": 10.0}
        )
        self.assertEqual(result["context"]["Material"], "Steel")
        got, _ = context_store.get_part_context(self.index_dir, ID_A)
        self.assertEqual(got["context"]["UnitCost"], 10.0)
        # The stored entry carries the synthetic/updated markers.
        self.assertIn("updated_at", self.raw()["parts"][ID_A])

    def test_empty_context_clears_entry(self):
        context_store.set_part_context(self.index_dir, ID_A, {"Material": "Steel"})
        context_store.set_part_context(self.index_dir, ID_A, {})
        got, _ = context_store.get_part_context(self.index_dir, ID_A)
        self.assertEqual(got["context"], {})
        self.assertNotIn(ID_A, self.raw()["parts"])

    def test_get_contexts_omits_missing(self):
        context_store.set_part_context(self.index_dir, ID_A, {"Material": "Steel"})
        out = context_store.get_contexts(self.index_dir, [ID_A, ID_B])
        self.assertIn(ID_A, out)
        self.assertNotIn(ID_B, out)

    def test_registered_ids_gate(self):
        with self.assertRaises(ValueError):
            context_store.set_part_context(
                self.index_dir, ID_A, {"Material": "Steel"}, registered_ids={ID_B}
            )

    def test_source_marker_persisted(self):
        context_store.set_part_context(
            self.index_dir, ID_A, {"Material": "Steel"}, source="synthetic sample data"
        )
        self.assertEqual(self.raw()["parts"][ID_A]["source"], "synthetic sample data")


class TestConcurrencyControl(_StoreBase):
    def test_if_match_mismatch_raises(self):
        _, etag = context_store.set_part_context(
            self.index_dir, ID_A, {"Material": "Steel"}
        )
        # A stale ETag must be rejected.
        with self.assertRaises(context_store.ContextPreconditionFailed):
            context_store.set_part_context(
                self.index_dir, ID_B, {"Material": "Aluminum"}, if_match='"stale"'
            )
        # The current ETag is accepted.
        context_store.set_part_context(
            self.index_dir, ID_B, {"Material": "Aluminum"}, if_match=etag
        )

    def test_etag_changes_with_content(self):
        _, e1 = context_store.set_part_context(self.index_dir, ID_A, {"Material": "Steel"})
        _, e2 = context_store.set_part_context(self.index_dir, ID_A, {"Material": "Brass"})
        self.assertNotEqual(e1, e2)


class TestSummaryAndNumeric(_StoreBase):
    def setUp(self):
        super().setUp()
        context_store.set_part_context(
            self.index_dir, ID_A, {"Material": "Steel", "UnitCost": 10.0, "LotSize": 5}
        )
        context_store.set_part_context(
            self.index_dir, ID_B, {"Material": "Brass", "UnitCost": 20.0}
        )
        context_store.set_part_context(
            self.index_dir, ID_C, {"Material": "Steel", "LotSize": "many"}
        )

    def test_summary_counts_and_types(self):
        result, _ = context_store.context_summary(self.index_dir)
        by_key = {k["key"]: k for k in result["keys"]}
        self.assertEqual(by_key["Material"]["count"], 3)
        self.assertEqual(by_key["Material"]["type"], "string")
        self.assertEqual(by_key["UnitCost"]["type"], "numeric")
        # LotSize is 5 (int) in one part and "many" (str) in another -> mixed.
        self.assertEqual(by_key["LotSize"]["type"], "mixed")
        self.assertEqual(result["parts_with_context"], 3)

    def test_list_numeric_keys(self):
        # UnitCost is always numeric; LotSize is disqualified by the string value.
        self.assertEqual(context_store.list_numeric_keys(self.index_dir), ["UnitCost"])


class TestPrune(_StoreBase):
    def test_prune_removes_only_targets(self):
        context_store.set_part_context(self.index_dir, ID_A, {"Material": "Steel"})
        context_store.set_part_context(self.index_dir, ID_B, {"Material": "Brass"})
        removed = context_store.prune_context(self.index_dir, [ID_A, "unknown"])
        self.assertEqual(removed, 1)
        self.assertNotIn(ID_A, self.raw()["parts"])
        self.assertIn(ID_B, self.raw()["parts"])


class TestParsing(_StoreBase):
    def _write(self, obj):
        context_store.context_path(self.index_dir).write_text(
            json.dumps(obj), encoding="utf-8"
        )
        context_store._context_cache.clear()

    def test_version_mismatch_raises(self):
        self._write({"version": 999, "parts": {}})
        with self.assertRaises(context_store.ContextFileError):
            context_store.read_context_document(self.index_dir)

    def test_non_scalar_values_dropped_not_fatal(self):
        self._write(
            {
                "version": 1,
                "parts": {ID_A: {"context": {"good": "x", "bad": {"n": 1}}}},
            }
        )
        doc, _ = context_store.read_context_document(self.index_dir)
        self.assertEqual(doc["parts"][ID_A]["context"], {"good": "x"})

    def test_entry_with_only_bad_values_is_dropped(self):
        self._write(
            {"version": 1, "parts": {ID_A: {"context": {"bad": [1, 2]}}}}
        )
        doc, _ = context_store.read_context_document(self.index_dir)
        self.assertNotIn(ID_A, doc["parts"])


class TestRulesConfig(_StoreBase):
    def _write_rules(self, obj):
        context_store.rules_path(self.index_dir).write_text(
            json.dumps(obj), encoding="utf-8"
        )

    def test_missing_file_is_empty(self):
        self.assertEqual(context_store.load_rules_config(self.index_dir), {})

    def test_unknown_options_dropped_and_known_kept(self):
        self._write_rules(
            {
                "categorical": {"temperature": 8.0, "bogus": 1, "normalize": "lower"},
                "numeric": {"log_scale": False, "not_a_param": True},
                "per_key": {
                    "UnitCost": {"type": "numeric", "min_evidence": 5},
                    "Material": {"type": "categorical", "normalize": "nope"},
                    "Bad": {"type": "unknown"},
                },
                "numeric_keys": ["UnitCost", 3],
            }
        )
        cfg = context_store.load_rules_config(self.index_dir)
        self.assertEqual(cfg["categorical"]["temperature"], 8.0)
        self.assertNotIn("bogus", cfg["categorical"])
        self.assertEqual(cfg["categorical"]["normalize"], "lower")
        self.assertEqual(cfg["numeric"], {"log_scale": False})
        self.assertEqual(cfg["per_key"]["UnitCost"], {"type": "numeric", "min_evidence": 5})
        # The unknown normalizer name is stripped, the rule kept.
        self.assertEqual(cfg["per_key"]["Material"], {"type": "categorical"})
        self.assertNotIn("Bad", cfg["per_key"])
        self.assertEqual(cfg["numeric_keys"], ["UnitCost"])

    def test_malformed_rules_ignored(self):
        context_store.rules_path(self.index_dir).write_text("not json", encoding="utf-8")
        self.assertEqual(context_store.load_rules_config(self.index_dir), {})


if __name__ == "__main__":
    unittest.main()
