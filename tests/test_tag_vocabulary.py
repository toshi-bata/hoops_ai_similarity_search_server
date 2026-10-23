"""Tests for the tag vocabulary (DATA_ROOT/tag_vocabulary.json).

The vocabulary holds the NAMES in use, which is a different thing from which
parts carry them. Its reason to exist is the one case tags.json cannot cover:
untag every part and the wording is gone, so the next person retypes "Socket
head cap screw" from memory and gets it slightly wrong. These tests pin that
separation -- names survive untagging, and removing a name from the vocabulary
leaves the tagged parts alone.

They also pin the second decision: there is ONE vocabulary for the whole
application, not one per index, and it is guarded by its own process-wide lock
rather than by the per-index tags lock.
"""

import json
import pathlib
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import core  # noqa: E402
import config  # noqa: E402
import index_registry  # noqa: E402
import locks  # noqa: E402
import tags  # noqa: E402


ID_A = "a" * 64
ID_B = "b" * 64


class _VocabularyBase(unittest.TestCase):
    """A disposable data root with two indexes, each holding two parts."""

    NAME = "idx"
    OTHER = "other"

    def setUp(self):
        # The tag limits are read from the environment, so they are pinned
        # here: a developer's .env must not be able to change what these
        # tests assert.
        self._env = mock.patch.dict(
            "os.environ", {"HOOPS_AI_MAX_TAGS_PER_PART": "32"}, clear=False
        )
        self._env.start()
        self.addCleanup(self._env.stop)

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = pathlib.Path(self._tmp.name)
        self._orig_dir = config.INDEXES_DIR
        self._orig_vocabulary = config.TAG_VOCABULARY_FILE
        config.INDEXES_DIR = root / "indexes"
        config.TAG_VOCABULARY_FILE = root / "tag_vocabulary.json"
        for name in (self.NAME, self.OTHER):
            (config.INDEXES_DIR / name).mkdir(parents=True)
            (config.INDEXES_DIR / name / "index.faiss").write_bytes(b"fake")

        self.registered = {ID_A, ID_B}
        self._orig_ids = index_registry._registered_index_ids
        index_registry._registered_index_ids = lambda name: set(self.registered)

        core._tags_cache.clear()
        core._tags_locks.clear()
        self._reset_vocabulary_lock()

    def tearDown(self):
        index_registry._registered_index_ids = self._orig_ids
        config.INDEXES_DIR = self._orig_dir
        config.TAG_VOCABULARY_FILE = self._orig_vocabulary
        core._tags_cache.clear()
        core._tags_locks.clear()
        self._reset_vocabulary_lock()

    def _reset_vocabulary_lock(self):
        # The vocabulary lock is process-wide, so a test that left it held
        # would hang every test after it. Replacing it is cheaper than relying
        # on every test to unwind.
        locks._TAG_VOCABULARY_LOCK = threading.Lock()

    def vocabulary_path(self):
        return core._tag_vocabulary_path()

    def write_vocabulary_file(self, text: str):
        path = self.vocabulary_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def read_vocabulary_file(self):
        return json.loads(self.vocabulary_path().read_text(encoding="utf-8"))


class TestVocabularyFile(_VocabularyBase):
    """Where it lives, what shape it has, and what a damaged one costs."""

    def test_it_sits_at_the_data_root_not_under_an_index(self):
        self.assertEqual(self.vocabulary_path(), config.TAG_VOCABULARY_FILE)
        self.assertEqual(self.vocabulary_path().name, "tag_vocabulary.json")
        self.assertNotIn(config.INDEXES_DIR, self.vocabulary_path().parents)

    def test_the_path_takes_no_index_name(self):
        # The centre of this change: one file for the application. A path
        # helper that accepted a name would be the first step back to one file
        # per index.
        self.assertEqual(core._tag_vocabulary_path(), core._tag_vocabulary_path())
        with self.assertRaises(TypeError):
            core._tag_vocabulary_path(self.NAME)

    def test_missing_file_is_an_empty_vocabulary(self):
        self.assertFalse(self.vocabulary_path().exists())
        self.assertEqual(core.get_tag_vocabulary(), {"tags": [], "total": 0})

    def test_the_written_shape_is_a_plain_list_of_names(self):
        core.add_tag_vocabulary(["Nut", "Panel nut"])
        self.assertEqual(self.read_vocabulary_file(), {"tags": ["Nut", "Panel nut"]})

    def test_hand_written_order_is_preserved(self):
        self.write_vocabulary_file('{"tags": ["Zeta", "Alpha", "Mu"]}')
        self.assertEqual(core.get_tag_vocabulary()["tags"], ["Zeta", "Alpha", "Mu"])

    def test_broken_json_reads_as_empty_with_a_warning(self):
        self.write_vocabulary_file("{not json at all")
        with self.assertLogs("tags", level="WARNING"):
            self.assertEqual(core.get_tag_vocabulary()["tags"], [])

    def test_empty_file_reads_as_empty_with_a_warning(self):
        self.write_vocabulary_file("")
        with self.assertLogs("tags", level="WARNING"):
            self.assertEqual(core.get_tag_vocabulary()["tags"], [])

    def test_wrong_toplevel_type_reads_as_empty(self):
        self.write_vocabulary_file('["Nut"]')
        with self.assertLogs("tags", level="WARNING"):
            self.assertEqual(core.get_tag_vocabulary()["tags"], [])

    def test_tags_member_that_is_not_a_list_reads_as_empty(self):
        self.write_vocabulary_file('{"tags": "Nut"}')
        with self.assertLogs("tags", level="WARNING"):
            self.assertEqual(core.get_tag_vocabulary()["tags"], [])

    def test_absent_tags_member_is_an_empty_vocabulary(self):
        self.write_vocabulary_file("{}")
        self.assertEqual(core.get_tag_vocabulary()["tags"], [])

    def test_unusable_entries_are_dropped_and_the_rest_survive(self):
        self.write_vocabulary_file('{"tags": ["Nut", "", 7, "bad/name", "Washer"]}')
        with self.assertLogs("tags", level="WARNING"):
            names = core.get_tag_vocabulary()["tags"]
        self.assertEqual(names, ["Nut", "Washer"])

    def test_a_broken_file_does_not_stop_tagging(self):
        self.write_vocabulary_file("{broken")
        core.assign_tag(self.NAME, "Nut", [ID_A])
        self.assertEqual(core.tags_by_part(self.NAME)[ID_A], ["Nut"])


class TestVocabularyIsSharedByEveryIndex(_VocabularyBase):
    """One list for the application, whichever index was tagged."""

    def test_two_indexes_write_into_the_same_vocabulary(self):
        core.assign_tag(self.NAME, "Nut", [ID_A])
        core.assign_tag(self.OTHER, "Washer", [ID_B])
        self.assertEqual(core.get_tag_vocabulary()["tags"], ["Nut", "Washer"])
        self.assertEqual(self.read_vocabulary_file(), {"tags": ["Nut", "Washer"]})

    def test_no_index_directory_holds_a_copy(self):
        core.assign_tag(self.NAME, "Nut", [ID_A])
        core.assign_tag(self.OTHER, "Washer", [ID_B])
        for name in (self.NAME, self.OTHER):
            with self.subTest(name=name):
                self.assertFalse(
                    (config.INDEXES_DIR / name / "tag_vocabulary.json").exists()
                )

    def test_a_name_from_another_index_is_offered_here(self):
        # The cost of sharing one file, pinned on purpose: clients are expected
        # to list the tags in use first and these after.
        core.assign_tag(self.OTHER, "Panel nut", [ID_A])
        self.assertEqual(core.get_tag_vocabulary()["tags"], ["Panel nut"])
        self.assertEqual(core.get_index_tags(self.NAME)[0]["tags"], [])

    def test_removing_a_name_removes_it_for_every_index(self):
        core.assign_tag(self.NAME, "Nut", [ID_A])
        core.assign_tag(self.OTHER, "Nut", [ID_B])
        core.remove_tag_vocabulary("Nut")
        self.assertEqual(core.get_tag_vocabulary()["tags"], [])
        self.assertEqual(core.tags_by_part(self.NAME)[ID_A], ["Nut"])
        self.assertEqual(core.tags_by_part(self.OTHER)[ID_B], ["Nut"])


class TestVocabularyLock(_VocabularyBase):
    """Its own lock, because the file it guards is not per index."""

    def test_the_vocabulary_lock_is_not_a_tags_lock(self):
        vocabulary_lock = locks._get_tag_vocabulary_lock()
        self.assertIs(vocabulary_lock, locks._get_tag_vocabulary_lock())
        for name in (self.NAME, self.OTHER):
            with self.subTest(name=name):
                self.assertIsNot(vocabulary_lock, locks._get_tags_lock(name))

    def test_a_write_takes_the_vocabulary_lock(self):
        real = locks._get_tag_vocabulary_lock()
        taken = []

        def spy():
            taken.append(True)
            return real

        with mock.patch.object(tags, "_get_tag_vocabulary_lock", spy):
            core.add_tag_vocabulary(["Nut"])
            core.remove_tag_vocabulary("Nut")
        self.assertEqual(len(taken), 2)

    def test_a_held_tags_lock_does_not_block_the_vocabulary(self):
        # If the vocabulary still used the per-index tags lock, this would
        # deadlock -- which is exactly the failure a shared file plus a
        # per-index lock produces under two concurrent taggers.
        done = threading.Event()

        def write():
            core.add_tag_vocabulary(["Nut"])
            done.set()

        with locks._get_tags_lock(self.NAME), locks._get_tags_lock(self.OTHER):
            worker = threading.Thread(target=write, daemon=True)
            worker.start()
            worker.join(timeout=10)
        self.assertTrue(done.is_set())
        self.assertEqual(core.get_tag_vocabulary()["tags"], ["Nut"])


class TestVocabularyFollowsAssignment(_VocabularyBase):
    """Every route that puts a tag on a part records the name."""

    def test_bulk_assignment_records_the_name(self):
        core.assign_tag(self.NAME, "Socket head cap screw", [ID_A])
        self.assertEqual(core.get_tag_vocabulary()["tags"], ["Socket head cap screw"])

    def test_setting_a_parts_tags_records_every_name(self):
        core.set_part_tags(self.NAME, ID_A, ["Nut", "Panel nut"])
        self.assertEqual(core.get_tag_vocabulary()["tags"], ["Nut", "Panel nut"])

    def test_a_name_is_recorded_once(self):
        core.assign_tag(self.NAME, "Nut", [ID_A])
        core.assign_tag(self.NAME, "Nut", [ID_B])
        self.assertEqual(core.get_tag_vocabulary()["tags"], ["Nut"])

    def test_assignment_records_the_name_even_when_every_id_is_skipped(self):
        result, _etag = core.assign_tag(self.NAME, "Nut", ["z" * 64])
        self.assertEqual(result["updated"], 0)
        self.assertEqual(core.get_tag_vocabulary()["tags"], ["Nut"])

    def test_the_name_survives_unassigning_it_from_every_part(self):
        # The point of the whole sidecar.
        core.assign_tag(self.NAME, "Panel nut", [ID_A, ID_B])
        core.unassign_tag(self.NAME, "Panel nut", [ID_A, ID_B])
        self.assertEqual(core.get_index_tags(self.NAME)[0]["tags"], [])
        self.assertEqual(core.get_tag_vocabulary()["tags"], ["Panel nut"])

    def test_the_name_survives_deleting_the_tag_from_the_index(self):
        core.assign_tag(self.NAME, "Panel nut", [ID_A, ID_B])
        core.delete_tag(self.NAME, "Panel nut")
        self.assertEqual(core.get_index_tags(self.NAME)[0]["tags"], [])
        self.assertEqual(core.get_tag_vocabulary()["tags"], ["Panel nut"])

    def test_the_name_survives_clearing_a_parts_tags(self):
        core.set_part_tags(self.NAME, ID_A, ["Nut"])
        core.set_part_tags(self.NAME, ID_A, [])
        self.assertEqual(core.get_tag_vocabulary()["tags"], ["Nut"])

    def test_unassignment_does_not_invent_a_name(self):
        core.unassign_tag(self.NAME, "Never used", [ID_A])
        self.assertEqual(core.get_tag_vocabulary()["tags"], [])


class TestVocabularyEditing(_VocabularyBase):
    """Adding ahead of use, and tidying a name away."""

    def test_adding_keeps_what_is_already_there(self):
        core.add_tag_vocabulary(["Nut"])
        result = core.add_tag_vocabulary(["Washer"])
        self.assertEqual(result["tags"], ["Nut", "Washer"])
        self.assertEqual(result["added"], ["Washer"])

    def test_adding_a_name_twice_is_ignored(self):
        core.add_tag_vocabulary(["Nut"])
        result = core.add_tag_vocabulary(["Nut"])
        self.assertEqual(result["added"], [])
        self.assertEqual(result["rejected"], [])
        self.assertEqual(result["tags"], ["Nut"])

    def test_duplicates_within_one_request_collapse(self):
        result = core.add_tag_vocabulary(["Nut", "Nut"])
        self.assertEqual(result["tags"], ["Nut"])

    def test_adding_nothing_leaves_no_file_behind(self):
        core.add_tag_vocabulary([])
        self.assertFalse(self.vocabulary_path().exists())

    def test_removing_a_name_leaves_the_tagged_parts_alone(self):
        core.assign_tag(self.NAME, "Nut", [ID_A, ID_B])
        result = core.remove_tag_vocabulary("Nut")
        self.assertTrue(result["removed"])
        self.assertEqual(result["tags"], [])
        self.assertEqual(core.tags_by_part(self.NAME)[ID_A], ["Nut"])
        self.assertEqual(
            core.get_index_tags(self.NAME)[0]["tags"], [{"tag": "Nut", "count": 2}]
        )

    def test_removing_a_name_that_is_not_there_is_not_an_error(self):
        result = core.remove_tag_vocabulary("Nut")
        self.assertFalse(result["removed"])
        self.assertEqual(result["tags"], [])

    def test_removing_one_name_keeps_the_others_in_order(self):
        core.add_tag_vocabulary(["Nut", "Washer", "Screw"])
        result = core.remove_tag_vocabulary("Washer")
        self.assertEqual(result["tags"], ["Nut", "Screw"])


class TestVocabularyLimit(_VocabularyBase):
    """The cap refuses names and says so, rather than dropping them."""

    def test_names_past_the_limit_are_rejected_and_reported(self):
        with mock.patch.object(tags, "_TAG_VOCABULARY_MAX", 3):
            result = core.add_tag_vocabulary(["a", "b", "c", "d", "e"])
        self.assertEqual(result["added"], ["a", "b", "c"])
        self.assertEqual(result["rejected"], ["d", "e"])
        self.assertEqual(result["total"], 3)
        self.assertEqual(self.read_vocabulary_file(), {"tags": ["a", "b", "c"]})

    def test_a_full_vocabulary_still_lets_a_part_be_tagged(self):
        with mock.patch.object(tags, "_TAG_VOCABULARY_MAX", 1):
            core.add_tag_vocabulary(["Nut"])
            with self.assertLogs("tags", level="WARNING"):
                core.assign_tag(self.NAME, "Washer", [ID_A])
        self.assertEqual(core.tags_by_part(self.NAME)[ID_A], ["Washer"])
        self.assertEqual(core.get_tag_vocabulary()["tags"], ["Nut"])

    def test_a_file_longer_than_the_limit_is_truncated_on_read(self):
        with mock.patch.object(tags, "_TAG_VOCABULARY_MAX", 2):
            self.write_vocabulary_file('{"tags": ["a", "b", "c"]}')
            self.assertEqual(core.get_tag_vocabulary()["tags"], ["a", "b"])


class TestVocabularyNormalization(_VocabularyBase):
    """The vocabulary uses normalize_tag, never a rule of its own."""

    def test_surrounding_whitespace_is_trimmed(self):
        result = core.add_tag_vocabulary(["  Nut \t"])
        self.assertEqual(result["tags"], ["Nut"])

    def test_empty_and_whitespace_only_are_rejected(self):
        for value in ("", "   ", "\t"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    core.add_tag_vocabulary([value])

    def test_control_characters_are_rejected(self):
        for value in ("a\nb", "a\x00b", "a\x7fb"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    core.add_tag_vocabulary([value])

    def test_the_length_limit_is_the_tag_length_limit(self):
        core.add_tag_vocabulary(["x" * core._TAG_MAX_LENGTH])
        with self.assertRaises(ValueError):
            core.add_tag_vocabulary(["x" * (core._TAG_MAX_LENGTH + 1)])

    def test_path_separators_are_rejected(self):
        # Same reason as the tags themselves: the name is a path segment in
        # DELETE /tag-vocabulary/{tag}.
        for value in ("a/b", "a\\b"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    core.add_tag_vocabulary([value])
                with self.assertRaises(ValueError):
                    core.remove_tag_vocabulary(value)

    def test_a_non_list_body_is_rejected(self):
        for value in ("Nut", 7, {"tags": ["Nut"]}):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    core.add_tag_vocabulary(value)

    def test_case_is_significant_here_too(self):
        result = core.add_tag_vocabulary(["Nut", "nut"])
        self.assertEqual(result["tags"], ["Nut", "nut"])


class TestVocabularyEndpoints(_VocabularyBase):
    """Router behaviour for the three vocabulary routes."""

    URL = "/similarity/tag-vocabulary"

    def setUp(self):
        super().setUp()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers.similarity import router

        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def test_the_routes_take_no_index_name(self):
        from routers.similarity import router

        paths = {
            route.path for route in router.routes if "tag-vocabulary" in route.path
        }
        self.assertEqual(
            paths,
            {"/similarity/tag-vocabulary", "/similarity/tag-vocabulary/{tag}"},
        )

    def test_get_returns_the_names(self):
        core.add_tag_vocabulary(["Nut", "Washer"])
        resp = self.client.get(self.URL)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"tags": ["Nut", "Washer"], "total": 2})

    def test_post_adds_and_reports_what_it_added(self):
        self.client.post(self.URL, json={"tags": ["Nut"]})
        resp = self.client.post(self.URL, json={"tags": ["Nut", "Washer"]})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["tags"], ["Nut", "Washer"])
        self.assertEqual(body["added"], ["Washer"])
        self.assertEqual(body["rejected"], [])

    def test_delete_removes_the_name_but_not_the_tag(self):
        core.assign_tag(self.NAME, "Nut", [ID_A])
        resp = self.client.delete(f"{self.URL}/Nut")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["removed"])
        self.assertEqual(resp.json()["tags"], [])
        self.assertEqual(core.tags_by_part(self.NAME)[ID_A], ["Nut"])

    def test_the_route_covers_every_index_while_tags_stay_per_index(self):
        core.assign_tag(self.NAME, "Nut", [ID_A])
        core.assign_tag(self.OTHER, "Washer", [ID_B])
        vocabulary = self.client.get(self.URL).json()
        in_use = self.client.get(f"/similarity/index/{self.NAME}/tags").json()
        self.assertEqual(vocabulary["tags"], ["Nut", "Washer"])
        self.assertEqual([t["tag"] for t in in_use["tags"]], ["Nut"])

    def test_invalid_name_is_422(self):
        resp = self.client.post(self.URL, json={"tags": [" "]})
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
