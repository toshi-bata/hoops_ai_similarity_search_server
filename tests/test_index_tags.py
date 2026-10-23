"""Tests for index tags (indexes/<name>/tags.json).

Tags are held in a sidecar rather than in the FaissVectorStore record metadata.
That was a measured decision, not a preference: persisting metadata means
calling save(), which always rewrites the .faiss file and moves its mtime, and
both the searcher cache and the assembly matcher cache are keyed by that mtime.
A tag edit would have cost a 19.3 s matcher rebuild. These tests pin the
behaviour the sidecar has to provide instead.
"""

import json
import pathlib
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import core  # noqa: E402
import search  # noqa: E402
import index_registry
import tags
import storage  # noqa: E402
import config


ID_A = "a" * 64
ID_B = "b" * 64
ID_C = "c" * 64


class _TagsBase(unittest.TestCase):
    """A disposable index directory with three registered parts."""

    NAME = "idx"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_dir = config.INDEXES_DIR
        config.INDEXES_DIR = pathlib.Path(self._tmp.name)
        (config.INDEXES_DIR / self.NAME).mkdir(parents=True)
        # _require_index_exists() only checks that the FAISS file is there.
        (config.INDEXES_DIR / self.NAME / "index.faiss").write_bytes(b"fake")

        self.registered = {ID_A, ID_B, ID_C}
        self._orig_ids = index_registry._registered_index_ids
        index_registry._registered_index_ids = lambda name: set(self.registered)

        core._tags_cache.clear()
        core._tags_locks.clear()

    def tearDown(self):
        index_registry._registered_index_ids = self._orig_ids
        config.INDEXES_DIR = self._orig_dir
        core._tags_cache.clear()
        core._tags_locks.clear()
        self._tmp.cleanup()

    def tags_path(self):
        return core._index_tags_path(self.NAME)

    def read_raw(self):
        return json.loads(self.tags_path().read_text(encoding="utf-8"))


class TestTagValidation(unittest.TestCase):
    """Tag text rules, applied at every entry point."""

    def test_surrounding_whitespace_is_trimmed(self):
        self.assertEqual(core.normalize_tag("  bracket \t"), "bracket")

    def test_empty_and_whitespace_only_are_rejected(self):
        for value in ("", "   ", "\t"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    core.normalize_tag(value)

    def test_control_characters_are_rejected(self):
        for value in ("a\nb", "a\tb", "a\x00b", "a\x7fb"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    core.normalize_tag(value)

    def test_length_limit_is_64(self):
        self.assertEqual(len(core.normalize_tag("x" * 64)), 64)
        with self.assertRaises(ValueError):
            core.normalize_tag("x" * 65)

    def test_path_separators_are_rejected(self):
        # The tag is a path segment in /index/{name}/tags/{tag}/parts; a
        # separator would route the request elsewhere and 404 confusingly.
        for value in ("a/b", "a\\b"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    core.normalize_tag(value)

    def test_case_is_significant(self):
        self.assertEqual(core.normalize_tag("Bracket"), "Bracket")
        self.assertNotEqual(core.normalize_tag("Bracket"), core.normalize_tag("bracket"))

    def test_non_strings_are_rejected(self):
        for value in (None, 1, ["a"]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    core.normalize_tag(value)

    def test_duplicates_collapse_preserving_order(self):
        self.assertEqual(
            core._normalize_tag_list([" b ", "a", "b"]), ["b", "a"]
        )

    def test_per_part_limit_is_enforced(self):
        limit = core.max_tags_per_part()
        core._normalize_tag_list([f"t{i}" for i in range(limit)])
        with self.assertRaises(ValueError):
            core._normalize_tag_list([f"t{i}" for i in range(limit + 1)])


class TestPartTags(_TagsBase):
    """Reading and replacing the tag set of one part."""

    def test_unknown_part_reads_as_empty(self):
        result, _etag = core.get_part_tags(self.NAME, ID_A)
        self.assertEqual(result, {"part_id": ID_A, "tags": []})

    def test_missing_index_is_a_key_error(self):
        with self.assertRaises(KeyError):
            core.get_part_tags("nope", ID_A)

    def test_set_then_get_round_trips(self):
        core.set_part_tags(self.NAME, ID_A, ["bracket", "v2"])
        result, _ = core.get_part_tags(self.NAME, ID_A)
        self.assertEqual(result["tags"], ["bracket", "v2"])

    def test_set_replaces_rather_than_merges(self):
        core.set_part_tags(self.NAME, ID_A, ["bracket", "v2"])
        core.set_part_tags(self.NAME, ID_A, ["v3"])
        result, _ = core.get_part_tags(self.NAME, ID_A)
        self.assertEqual(result["tags"], ["v3"])

    def test_empty_list_clears_the_entry(self):
        core.set_part_tags(self.NAME, ID_A, ["bracket"])
        core.set_part_tags(self.NAME, ID_A, [])
        self.assertEqual(self.read_raw()["parts"], {})

    def test_unregistered_part_is_rejected(self):
        with self.assertRaises(ValueError):
            core.set_part_tags(self.NAME, "f" * 64, ["bracket"])

    def test_invalid_tag_is_rejected_before_anything_is_written(self):
        with self.assertRaises(ValueError):
            core.set_part_tags(self.NAME, ID_A, ["ok", ""])
        self.assertFalse(self.tags_path().exists())

    def test_client_id_is_recorded(self):
        core.set_part_tags(self.NAME, ID_A, ["bracket"], client_id="toshi")
        entry = self.read_raw()["parts"][ID_A]
        self.assertEqual(entry["updated_by"], "toshi")
        self.assertTrue(entry["updated_at"].endswith("Z"))

    def test_absent_client_id_records_no_updated_by(self):
        core.set_part_tags(self.NAME, ID_A, ["bracket"])
        self.assertNotIn("updated_by", self.read_raw()["parts"][ID_A])

    def test_case_only_collision_is_logged_not_merged(self):
        core.set_part_tags(self.NAME, ID_A, ["bracket"])
        with self.assertLogs(tags.logger, level="WARNING") as captured:
            core.set_part_tags(self.NAME, ID_B, ["Bracket"])
        self.assertTrue(any("only by case" in line for line in captured.output))
        result, _ = core.get_index_tags(self.NAME)
        self.assertEqual(
            sorted(t["tag"] for t in result["tags"]), ["Bracket", "bracket"]
        )


class TestTagListing(_TagsBase):
    """The tag list and its ordering."""

    def test_empty_index_lists_nothing(self):
        result, _ = core.get_index_tags(self.NAME)
        self.assertEqual(result, {"tags": [], "total": 0})

    def test_counts_parts_not_occurrences(self):
        core.set_part_tags(self.NAME, ID_A, ["bracket", "v2"])
        core.set_part_tags(self.NAME, ID_B, ["bracket"])
        result, _ = core.get_index_tags(self.NAME)
        self.assertEqual(
            result["tags"],
            [{"tag": "bracket", "count": 2}, {"tag": "v2", "count": 1}],
        )

    def test_ties_break_on_name_ascending(self):
        core.set_part_tags(self.NAME, ID_A, ["zeta", "alpha"])
        result, _ = core.get_index_tags(self.NAME)
        self.assertEqual([t["tag"] for t in result["tags"]], ["alpha", "zeta"])


class TestBulkTagging(_TagsBase):
    """Assign / unassign across many parts, and whole-tag deletion."""

    def test_assign_adds_to_every_registered_id(self):
        result, _ = core.assign_tag(self.NAME, "bracket", [ID_A, ID_B])
        self.assertEqual(result["updated"], 2)
        self.assertEqual(result["skipped"], [])
        listing, _ = core.get_index_tags(self.NAME)
        self.assertEqual(listing["tags"][0]["count"], 2)

    def test_unregistered_ids_are_skipped_not_fatal(self):
        result, _ = core.assign_tag(self.NAME, "bracket", [ID_A, "f" * 64])
        self.assertEqual(result["updated"], 1)
        self.assertEqual(
            result["skipped"], [{"part_id": "f" * 64, "reason": "not_registered"}]
        )

    def test_assigning_twice_is_idempotent(self):
        core.assign_tag(self.NAME, "bracket", [ID_A])
        result, _ = core.assign_tag(self.NAME, "bracket", [ID_A])
        self.assertEqual(result["updated"], 0)
        entry, _ = core.get_part_tags(self.NAME, ID_A)
        self.assertEqual(entry["tags"], ["bracket"])

    def test_duplicate_ids_in_one_request_count_once(self):
        result, _ = core.assign_tag(self.NAME, "bracket", [ID_A, ID_A])
        self.assertEqual(result["updated"], 1)

    def test_tag_limit_is_reported_per_part(self):
        limit = core.max_tags_per_part()
        core.set_part_tags(self.NAME, ID_A, [f"t{i}" for i in range(limit)])
        result, _ = core.assign_tag(self.NAME, "extra", [ID_A, ID_B])
        self.assertEqual(result["updated"], 1)
        self.assertEqual(
            result["skipped"], [{"part_id": ID_A, "reason": "tag_limit_reached"}]
        )

    def test_unassign_removes_only_that_tag(self):
        core.set_part_tags(self.NAME, ID_A, ["bracket", "v2"])
        result, _ = core.unassign_tag(self.NAME, "bracket", [ID_A])
        self.assertEqual(result["updated"], 1)
        entry, _ = core.get_part_tags(self.NAME, ID_A)
        self.assertEqual(entry["tags"], ["v2"])

    def test_unassigning_the_last_tag_drops_the_entry(self):
        core.set_part_tags(self.NAME, ID_A, ["bracket"])
        core.unassign_tag(self.NAME, "bracket", [ID_A])
        self.assertEqual(self.read_raw()["parts"], {})

    def test_unassigning_an_absent_tag_changes_nothing(self):
        core.set_part_tags(self.NAME, ID_A, ["v2"])
        result, _ = core.unassign_tag(self.NAME, "bracket", [ID_A])
        self.assertEqual(result["updated"], 0)

    def test_delete_tag_clears_it_everywhere(self):
        core.assign_tag(self.NAME, "bracket", [ID_A, ID_B])
        core.assign_tag(self.NAME, "v2", [ID_A])
        result, _ = core.delete_tag(self.NAME, "bracket")
        self.assertEqual(result, {"tag": "bracket", "removed_from": 2})
        listing, _ = core.get_index_tags(self.NAME)
        self.assertEqual(listing["tags"], [{"tag": "v2", "count": 1}])

    def test_delete_of_an_unused_tag_is_not_an_error(self):
        result, _ = core.delete_tag(self.NAME, "nothing")
        self.assertEqual(result["removed_from"], 0)


class TestPersistence(_TagsBase):
    """The file on disk, its cache, and how corruption is handled."""

    def test_write_is_atomic_and_leaves_no_temp_file(self):
        core.set_part_tags(self.NAME, ID_A, ["bracket"])
        leftovers = list((config.INDEXES_DIR / self.NAME).glob("_tmp_tags_*"))
        self.assertEqual(leftovers, [])

    def test_document_carries_the_schema_version(self):
        core.set_part_tags(self.NAME, ID_A, ["bracket"])
        raw = self.read_raw()
        self.assertEqual(raw["version"], core._TAGS_SCHEMA_VERSION)
        self.assertTrue(raw["updated_at"].endswith("Z"))

    def test_missing_file_reads_as_an_empty_document(self):
        doc, etag = tags._read_tags_document(self.NAME)
        self.assertEqual(doc["parts"], {})
        self.assertTrue(etag.startswith('"'))

    def test_an_external_edit_is_picked_up(self):
        core.set_part_tags(self.NAME, ID_A, ["bracket"])
        raw = self.read_raw()
        raw["parts"][ID_B] = {"tags": ["added-by-hand"]}
        # Enlarging the file changes (mtime_ns, size), which is the cache key.
        self.tags_path().write_text(json.dumps(raw), encoding="utf-8")
        entry, _ = core.get_part_tags(self.NAME, ID_B)
        self.assertEqual(entry["tags"], ["added-by-hand"])

    def test_the_cache_returns_a_copy(self):
        core.set_part_tags(self.NAME, ID_A, ["bracket"])
        first, _ = tags._read_tags_document(self.NAME)
        first["parts"].clear()
        second, _ = tags._read_tags_document(self.NAME)
        self.assertIn(ID_A, second["parts"])

    def test_the_cache_is_bounded(self):
        for i in range(core._TAGS_CACHE_MAX + 3):
            name = f"idx{i}"
            (config.INDEXES_DIR / name).mkdir()
            (config.INDEXES_DIR / name / "index.faiss").write_bytes(b"fake")
            core.set_part_tags(name, ID_A, ["bracket"])
            tags._read_tags_document(name)
        self.assertLessEqual(len(core._tags_cache), core._TAGS_CACHE_MAX)

    def test_corrupt_json_raises_rather_than_erasing(self):
        self.tags_path().write_text("{not json", encoding="utf-8")
        with self.assertRaises(core.TagsFileError):
            tags._read_tags_document(self.NAME)

    def test_an_unknown_version_raises(self):
        self.tags_path().write_text(
            json.dumps({"version": 99, "parts": {}}), encoding="utf-8"
        )
        with self.assertRaises(core.TagsFileError):
            tags._read_tags_document(self.NAME)

    def test_a_mutation_on_a_corrupt_file_does_not_overwrite_it(self):
        # Silently starting from an empty document would erase every tag.
        payload = '{"version": 1, "parts": {"x": "not-an-object"}}'
        self.tags_path().write_text(payload, encoding="utf-8")
        with self.assertRaises(core.TagsFileError):
            core.set_part_tags(self.NAME, ID_A, ["bracket"])
        self.assertEqual(self.tags_path().read_text(encoding="utf-8"), payload)

    def test_filtering_degrades_loudly_when_the_file_is_unreadable(self):
        self.tags_path().write_text("{not json", encoding="utf-8")
        with self.assertLogs(tags.logger, level="WARNING"):
            self.assertEqual(core.tags_by_part(self.NAME), {})


class TestOptimisticLocking(_TagsBase):
    """If-Match, so two editors cannot silently overwrite one another."""

    def test_absent_if_match_is_allowed(self):
        core.set_part_tags(self.NAME, ID_A, ["bracket"], if_match=None)
        entry, _ = core.get_part_tags(self.NAME, ID_A)
        self.assertEqual(entry["tags"], ["bracket"])

    def test_matching_etag_is_allowed(self):
        _, etag = core.get_index_tags(self.NAME)
        core.set_part_tags(self.NAME, ID_A, ["bracket"], if_match=etag)

    def test_star_matches_anything(self):
        core.set_part_tags(self.NAME, ID_A, ["bracket"], if_match="*")

    def test_stale_etag_is_rejected(self):
        _, stale = core.get_index_tags(self.NAME)
        core.set_part_tags(self.NAME, ID_A, ["bracket"])
        with self.assertRaises(core.TagsPreconditionFailed):
            core.set_part_tags(self.NAME, ID_B, ["v2"], if_match=stale)

    def test_a_rejected_write_changes_nothing(self):
        _, stale = core.get_index_tags(self.NAME)
        core.set_part_tags(self.NAME, ID_A, ["bracket"])
        with self.assertRaises(core.TagsPreconditionFailed):
            core.set_part_tags(self.NAME, ID_A, ["clobbered"], if_match=stale)
        entry, _ = core.get_part_tags(self.NAME, ID_A)
        self.assertEqual(entry["tags"], ["bracket"])

    def test_the_etag_changes_only_when_the_content_does(self):
        _, before = core.get_index_tags(self.NAME)
        core.assign_tag(self.NAME, "bracket", ["f" * 64])  # all skipped
        _, unchanged = core.get_index_tags(self.NAME)
        self.assertEqual(before, unchanged)
        core.assign_tag(self.NAME, "bracket", [ID_A])
        _, changed = core.get_index_tags(self.NAME)
        self.assertNotEqual(before, changed)

    def test_weak_validator_prefix_is_accepted(self):
        _, etag = core.get_index_tags(self.NAME)
        self.assertTrue(core.etag_matches(f"W/{etag}", etag))

    def test_a_list_of_etags_matches_on_any_member(self):
        _, etag = core.get_index_tags(self.NAME)
        self.assertTrue(core.etag_matches(f'"other", {etag}', etag))
        self.assertFalse(core.etag_matches('"other", "another"', etag))

    def test_the_check_happens_inside_the_lock(self):
        """Two writers holding the same ETag: exactly one must win.

        Checking If-Match outside the lock would let both read the same value,
        both pass, and the second overwrite the first without noticing.
        """
        _, etag = core.get_index_tags(self.NAME)
        barrier = threading.Barrier(2)
        outcomes = []

        def writer(part_id, tag):
            barrier.wait(timeout=10)
            try:
                core.set_part_tags(self.NAME, part_id, [tag], if_match=etag)
                outcomes.append("ok")
            except core.TagsPreconditionFailed:
                outcomes.append("412")

        threads = [
            threading.Thread(target=writer, args=(ID_A, "first")),
            threading.Thread(target=writer, args=(ID_B, "second")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            self.assertFalse(t.is_alive())
        self.assertEqual(sorted(outcomes), ["412", "ok"])
        self.assertEqual(len(self.read_raw()["parts"]), 1)


class TestPruneOnRemoval(_TagsBase):
    """Tags must not outlive the parts they describe."""

    def test_prune_drops_only_the_named_ids(self):
        core.assign_tag(self.NAME, "bracket", [ID_A, ID_B])
        removed = core.prune_tags(self.NAME, [ID_A])
        self.assertEqual(removed, 1)
        self.assertEqual(list(self.read_raw()["parts"]), [ID_B])

    def test_prune_of_untagged_ids_is_a_no_op(self):
        core.assign_tag(self.NAME, "bracket", [ID_A])
        self.assertEqual(core.prune_tags(self.NAME, [ID_C]), 0)

    def test_prune_on_a_missing_index_is_silent(self):
        self.assertEqual(core.prune_tags("nope", [ID_A]), 0)

    def test_remove_from_index_prunes(self):
        core.assign_tag(self.NAME, "bracket", [ID_A, ID_B])
        vs = MagicMock()
        vs.get_ids.side_effect = lambda: [ID_B, ID_C]
        saved = {
            "_load_named_index": index_registry._load_named_index,
            "_load_index_schema": index_registry._load_index_schema,
            "_save_named_index_atomic": index_registry._save_named_index_atomic,
        }
        index_registry._load_named_index = lambda name: vs
        index_registry._load_index_schema = lambda name: {
            "schema_version": core._INDEX_SCHEMA_VERSION
        }
        index_registry._save_named_index_atomic = lambda name, store: None
        try:
            core.remove_from_index(self.NAME, [ID_A])
        finally:
            for attr, val in saved.items():
                setattr(storage if attr == "find_persistent_CAD_file" else index_registry, attr, val)
        self.assertEqual(list(self.read_raw()["parts"]), [ID_B])


class TestTagFilterPredicate(_TagsBase):
    """The filter shared by the listing and both search paths."""

    def setUp(self):
        super().setUp()
        core.set_part_tags(self.NAME, ID_A, ["bracket", "v2"])
        core.set_part_tags(self.NAME, ID_B, ["bracket"])

    def test_no_filter_requested_returns_none(self):
        self.assertIsNone(core.normalize_tag_filter(None, "any", None))
        self.assertIsNone(core.normalize_tag_filter([], "any", None))

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            core.normalize_tag_filter(["bracket"], "either", None)

    def test_invalid_tag_text_is_rejected(self):
        with self.assertRaises(ValueError):
            core.normalize_tag_filter([""], "any", None)

    def test_any_matches_one_of_the_tags(self):
        flt = core.normalize_tag_filter(["v2", "other"], "any", None)
        pred = core.tag_filter_predicate(self.NAME, flt)
        self.assertTrue(pred(ID_A))
        self.assertFalse(pred(ID_B))

    def test_all_requires_every_tag(self):
        flt = core.normalize_tag_filter(["bracket", "v2"], "all", None)
        pred = core.tag_filter_predicate(self.NAME, flt)
        self.assertTrue(pred(ID_A))
        self.assertFalse(pred(ID_B))

    def test_tagged_true_and_false(self):
        pred_true = core.tag_filter_predicate(
            self.NAME, core.normalize_tag_filter(None, "any", True)
        )
        pred_false = core.tag_filter_predicate(
            self.NAME, core.normalize_tag_filter(None, "any", False)
        )
        self.assertTrue(pred_true(ID_A))
        self.assertFalse(pred_true(ID_C))
        self.assertFalse(pred_false(ID_A))
        self.assertTrue(pred_false(ID_C))

    def test_the_sidecar_is_read_once_per_predicate(self):
        reads = []
        orig = tags._read_tags_document
        tags._read_tags_document = lambda name: reads.append(name) or orig(name)
        try:
            pred = core.tag_filter_predicate(
                self.NAME, core.normalize_tag_filter(["bracket"], "any", None)
            )
            for _ in range(50):
                pred(ID_A)
        finally:
            tags._read_tags_document = orig
        self.assertEqual(len(reads), 1)


class _FakeSearcher:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def search_by_shape(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return self._results


def _vh(hit_id, score):
    hit = MagicMock()
    hit.id = hit_id
    hit.score = score
    hit.metadata = {"file_id": hit_id}
    return hit


class TestSearchTagFilter(_TagsBase):
    """Tag filtering on the part-search path.

    The filter runs after CADSearch has ranked, so the guarantee under test is
    that scores and relative order are untouched and only membership changes.
    """

    QUERY_ID = "q" * 64

    def setUp(self):
        super().setUp()
        core.set_part_tags(self.NAME, ID_A, ["bracket"])

        self.searcher = _FakeSearcher([[_vh(ID_A, 0.9), _vh(ID_B, 0.8), _vh(ID_C, 0.7)]])
        vs = MagicMock()
        vs.get_ids.side_effect = lambda: sorted(self.registered)
        vs.iter_metadata.side_effect = lambda: iter(
            [{"file_id": i, "kind": "part"} for i in sorted(self.registered)]
        )
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

        # Each method asserts the searcher was called (self.searcher.calls[0]).
        # The part-search response cache is on by default and keyed partly on the
        # index.faiss mtime; _TagsBase writes that file fresh in every setUp, but
        # two fast consecutive tests can land on the same coarse mtime, so an
        # identical (query, top_k, tags) request would hit the previous test's
        # cached response and never touch the searcher. Start every test with an
        # empty cache (see TestSearchResultCache in test_cadsearch_index_search).
        core.clear_search_cache()

    def tearDown(self):
        core.clear_search_cache()
        for attr, val in self._saved.items():
            setattr(storage if attr == "find_persistent_CAD_file" else index_registry, attr, val)
        super().tearDown()

    def test_no_filter_leaves_the_request_unchanged(self):
        # Regression guard: the tag work must not alter an untagged search.
        search.search_index(self.NAME, self.QUERY_ID, 10)
        _path, kwargs = self.searcher.calls[0]
        self.assertEqual(kwargs["top_k"], 11)

    def test_a_filter_over_fetches_candidates(self):
        search.search_index(self.NAME, self.QUERY_ID, 10, tags=["bracket"])
        _path, kwargs = self.searcher.calls[0]
        self.assertEqual(kwargs["top_k"], 10 * core._TAG_FILTER_OVERFETCH + 1)

    def test_the_over_fetch_is_capped(self):
        search.search_index(self.NAME, self.QUERY_ID, 100, tags=["bracket"])
        _path, kwargs = self.searcher.calls[0]
        self.assertEqual(kwargs["top_k"], core._TAG_FILTER_OVERFETCH_MAX + 1)

    def test_only_matching_hits_are_returned(self):
        out = search.search_index(self.NAME, self.QUERY_ID, 10, tags=["bracket"])
        self.assertEqual([h["id"] for h in out["hits"]], [ID_A])

    def test_scores_are_not_rescored_by_filtering(self):
        unfiltered = search.search_index(self.NAME, self.QUERY_ID, 10)
        filtered = search.search_index(self.NAME, self.QUERY_ID, 10, tags=["bracket"])
        original = {h["id"]: h["score"] for h in unfiltered["hits"]}
        for hit in filtered["hits"]:
            self.assertEqual(hit["score"], original[hit["id"]])

    def test_a_filter_can_return_fewer_than_top_k(self):
        out = search.search_index(self.NAME, self.QUERY_ID, 10, tags=["bracket"])
        self.assertLess(out["count"], 10)

    def test_tagged_false_selects_the_untagged(self):
        out = search.search_index(self.NAME, self.QUERY_ID, 10, tagged=False)
        self.assertEqual([h["id"] for h in out["hits"]], [ID_B, ID_C])

    def test_the_self_hit_obeys_the_filter(self):
        # The query is registered but untagged, so a "bracket" search must not
        # return it just because include_self was asked for.
        self.registered.add(self.QUERY_ID)
        out = search.search_index(
            self.NAME, self.QUERY_ID, 10, include_self=True, tags=["bracket"]
        )
        self.assertNotIn(self.QUERY_ID, [h["id"] for h in out["hits"]])

    def test_the_self_hit_is_kept_when_it_matches(self):
        self.registered.add(self.QUERY_ID)
        core.set_part_tags(self.NAME, self.QUERY_ID, ["bracket"])
        out = search.search_index(
            self.NAME, self.QUERY_ID, 10, include_self=True, tags=["bracket"]
        )
        self.assertEqual(out["hits"][0]["id"], self.QUERY_ID)

    def test_an_invalid_filter_raises_before_searching(self):
        with self.assertRaises(ValueError):
            search.search_index(self.NAME, self.QUERY_ID, 10, tag_mode="either")
        self.assertEqual(self.searcher.calls, [])


    def test_an_unfiltered_search_omits_the_diagnostics(self):
        out = search.search_index(self.NAME, self.QUERY_ID, 10)
        self.assertNotIn("tag_filter", out)

    def test_a_filtered_search_reports_how_many_carry_the_tag(self):
        # The point of the field: a rare tag can rank outside the candidate
        # pool, and an empty hit list would otherwise look like a bug.
        core.set_part_tags(self.NAME, ID_B, ["rare"])
        core.set_part_tags(self.NAME, ID_C, ["rare"])
        out = search.search_index(self.NAME, self.QUERY_ID, 10, tags=["rare"])
        diag = out["tag_filter"]
        self.assertEqual(diag["matching_parts_in_index"], 2)
        self.assertEqual(diag["candidates_examined"], 3)
        self.assertEqual(diag["tags"], ["rare"])
        self.assertEqual(diag["tag_mode"], "any")
        self.assertIsNone(diag["tagged"])

    def test_the_diagnostics_distinguish_absent_from_unranked(self):
        out = search.search_index(self.NAME, self.QUERY_ID, 10, tags=["never-used"])
        self.assertEqual(out["count"], 0)
        self.assertEqual(out["tag_filter"]["matching_parts_in_index"], 0)
        self.assertGreater(out["tag_filter"]["candidates_examined"], 0)


class TestListPartsTagFilter(_TagsBase):
    """Tag filtering and tag output on the parts listing."""

    def setUp(self):
        super().setUp()
        core.set_part_tags(self.NAME, ID_A, ["bracket", "v2"])
        core.set_part_tags(self.NAME, ID_B, ["bracket"])

        vs = MagicMock()
        vs.iter_metadata.side_effect = lambda: iter(
            [
                {"file_id": ID_A, "filename": "a.stp", "kind": "part", "bodies": 1},
                {"file_id": ID_B, "filename": "b.stp", "kind": "part", "bodies": 1},
                {"file_id": ID_C, "filename": "c.stp", "kind": "assembly", "bodies": 4},
            ]
        )
        self._saved = {
            "_load_named_index": index_registry._load_named_index,
            "_resolve_index_asset": index_registry._resolve_index_asset,
        }
        index_registry._load_named_index = lambda name: vs
        index_registry._resolve_index_asset = lambda name, pid, asset: None

    def tearDown(self):
        for attr, val in self._saved.items():
            setattr(storage if attr == "find_persistent_CAD_file" else index_registry, attr, val)
        super().tearDown()

    def test_items_carry_their_tags(self):
        out = core.list_index_parts(self.NAME)
        by_id = {it["id"]: it["tags"] for it in out["items"]}
        self.assertEqual(by_id[ID_A], ["bracket", "v2"])
        self.assertEqual(by_id[ID_C], [])

    def test_filtering_happens_before_pagination(self):
        out = core.list_index_parts(self.NAME, tags=["v2"])
        self.assertEqual(out["total"], 1)
        self.assertEqual([it["id"] for it in out["items"]], [ID_A])

    def test_tag_and_kind_filters_combine(self):
        out = core.list_index_parts(self.NAME, kind="assembly", tagged=True)
        self.assertEqual(out["total"], 0)

    def test_tagged_false_lists_the_untagged(self):
        out = core.list_index_parts(self.NAME, tagged=False)
        self.assertEqual([it["id"] for it in out["items"]], [ID_C])


class TestTagEndpoints(_TagsBase):
    """Router behaviour: status codes, ETag headers and the confirm gate."""

    def setUp(self):
        super().setUp()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers.similarity import router

        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def url(self, suffix=""):
        return f"/similarity/index/{self.NAME}{suffix}"

    def test_tag_list_is_not_demo_gated(self):
        # Tags are a production feature of the shared server.
        resp = self.client.get(self.url("/tags"))
        self.assertEqual(resp.status_code, 200)

    def test_tag_list_carries_an_etag(self):
        resp = self.client.get(self.url("/tags"))
        self.assertTrue(resp.headers["ETag"].startswith('"'))

    def test_unknown_index_is_404(self):
        resp = self.client.get("/similarity/index/missing/tags")
        self.assertEqual(resp.status_code, 404)

    def test_invalid_index_name_is_422(self):
        resp = self.client.get("/similarity/index/NOT VALID/tags")
        self.assertEqual(resp.status_code, 422)

    def test_put_replaces_the_tag_set(self):
        resp = self.client.put(
            self.url(f"/parts/{ID_A}/tags"), json={"tags": ["bracket", "v2"]}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"part_id": ID_A, "tags": ["bracket", "v2"]})

    def test_put_records_the_client_id_header(self):
        self.client.put(
            self.url(f"/parts/{ID_A}/tags"),
            json={"tags": ["bracket"]},
            headers={"X-Client-Id": "workstation-3"},
        )
        self.assertEqual(self.read_raw()["parts"][ID_A]["updated_by"], "workstation-3")

    def test_put_with_an_invalid_tag_is_422(self):
        resp = self.client.put(self.url(f"/parts/{ID_A}/tags"), json={"tags": [" "]})
        self.assertEqual(resp.status_code, 422)

    def test_put_for_an_unregistered_part_is_422(self):
        resp = self.client.put(
            self.url(f"/parts/{'f' * 64}/tags"), json={"tags": ["bracket"]}
        )
        self.assertEqual(resp.status_code, 422)

    def test_stale_if_match_is_412(self):
        stale = self.client.get(self.url("/tags")).headers["ETag"]
        self.client.put(self.url(f"/parts/{ID_A}/tags"), json={"tags": ["bracket"]})
        resp = self.client.put(
            self.url(f"/parts/{ID_B}/tags"),
            json={"tags": ["v2"]},
            headers={"If-Match": stale},
        )
        self.assertEqual(resp.status_code, 412)

    def test_current_if_match_succeeds(self):
        etag = self.client.get(self.url("/tags")).headers["ETag"]
        resp = self.client.put(
            self.url(f"/parts/{ID_A}/tags"),
            json={"tags": ["bracket"]},
            headers={"If-Match": etag},
        )
        self.assertEqual(resp.status_code, 200)

    def test_bulk_assign_reports_skipped_ids(self):
        resp = self.client.post(
            self.url("/tags/bracket/parts"), json={"part_ids": [ID_A, "f" * 64]}
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["updated"], 1)
        self.assertEqual(body["skipped"], [{"part_id": "f" * 64, "reason": "not_registered"}])

    def test_bulk_assign_without_ids_is_422(self):
        resp = self.client.post(self.url("/tags/bracket/parts"), json={"part_ids": []})
        self.assertEqual(resp.status_code, 422)

    def test_bulk_unassign_removes_the_tag(self):
        self.client.post(self.url("/tags/bracket/parts"), json={"part_ids": [ID_A]})
        resp = self.client.request(
            "DELETE", self.url("/tags/bracket/parts"), json={"part_ids": [ID_A]}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["updated"], 1)

    def test_deleting_a_tag_requires_confirmation(self):
        self.client.post(self.url("/tags/bracket/parts"), json={"part_ids": [ID_A]})
        resp = self.client.delete(self.url("/tags/bracket"))
        self.assertEqual(resp.status_code, 409)
        entry, _ = core.get_part_tags(self.NAME, ID_A)
        self.assertEqual(entry["tags"], ["bracket"])

    def test_confirmed_delete_removes_the_tag(self):
        self.client.post(self.url("/tags/bracket/parts"), json={"part_ids": [ID_A, ID_B]})
        resp = self.client.delete(self.url("/tags/bracket?confirm=true"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"tag": "bracket", "removed_from": 2})

    def test_get_part_tags_returns_an_empty_list(self):
        resp = self.client.get(self.url(f"/parts/{ID_A}/tags"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["tags"], [])


if __name__ == "__main__":
    unittest.main()
