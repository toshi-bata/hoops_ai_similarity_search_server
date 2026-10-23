"""Tests for the client's tag-candidate list (web/src/tag-candidates.js).

The candidate list is the one piece of the tagging UI that can be wrong
without looking wrong: a datalist with no options is indistinguishable from
an index that has no tags yet. It WAS wrong -- the vocabulary response
(``{"tags": [...], "total": N}``) was read as the ``{data, etag}`` wrapper
that only the ETag routes return, so every name fetched was dropped on the
floor and no error was raised anywhere.

So the parsing and the ordering were split out of tag-picker.js into pure
functions with no imports, and this module runs them. There is no JavaScript
test runner in this repository and adding one for four assertions is not a
trade worth making, so the functions are executed with Node -- which the
project already requires for ``node --check`` -- through a small generated
driver. The tests skip when Node is absent, exactly as the tests that need an
optional Python package do.

What is pinned here is the contract the server side has to keep:

  * the payload shape ``{"tags": [...], "total": N}`` of both tag endpoints,
  * that an index with no tags in use still offers the vocabulary,
  * that a name in both lists appears once,
  * and the order -- in use by descending count, then vocabulary-only by name
    -- which matters because the vocabulary is application-wide and carries
    names from other indexes.
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

_MODULE = pathlib.Path(__file__).resolve().parent.parent / "web" / "src" / "tag-candidates.js"


def _node() -> str | None:
    return shutil.which("node")


@unittest.skipIf(_node() is None, "Node is not installed")
class TagCandidateMergeTests(unittest.TestCase):
    """mergeTagCandidates(in_use, vocabulary) -> [name, ...]."""

    def merge(self, in_use, vocabulary):
        """Run the real JS function and return its result."""
        url = _MODULE.resolve().as_uri()
        driver = (
            f"import {{ mergeTagCandidates }} from {json.dumps(url)};\n"
            f"const inUse = {json.dumps(in_use)};\n"
            f"const vocabulary = {json.dumps(vocabulary)};\n"
            "process.stdout.write("
            "JSON.stringify(mergeTagCandidates(inUse, vocabulary)));\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "driver.mjs"
            path.write_text(driver, encoding="utf-8")
            result = subprocess.run(
                [_node(), str(path)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_names_are_read_from_the_wrapped_payload(self):
        """Both endpoints answer {"tags": [...], "total": N}, not a bare list.

        This is the shape the bug got wrong, so it is the first thing pinned.
        """
        names = self.merge(
            {"tags": [{"tag": "Nut", "count": 3}], "total": 1},
            {"tags": ["Washer"], "total": 1},
        )
        self.assertEqual(names, ["Nut", "Washer"])

    def test_vocabulary_survives_an_index_with_no_tags_in_use(self):
        """An empty index is when a remembered name is worth the most."""
        for in_use in ({"tags": [], "total": 0}, None):
            with self.subTest(in_use=in_use):
                names = self.merge(in_use, {"tags": ["Panel nut", "Nut"], "total": 2})
                self.assertEqual(names, ["Nut", "Panel nut"])

    def test_a_name_in_both_lists_appears_once(self):
        """In use AND remembered is one name, offered in the in-use position."""
        names = self.merge(
            {"tags": [{"tag": "Nut", "count": 9}], "total": 1},
            {"tags": ["Nut", "Washer"], "total": 2},
        )
        self.assertEqual(names, ["Nut", "Washer"])

    def test_in_use_comes_first_by_count_then_vocabulary_by_name(self):
        """The order the application-wide vocabulary makes necessary."""
        names = self.merge(
            {
                "tags": [
                    {"tag": "Bolt", "count": 2},
                    {"tag": "Nut", "count": 40},
                    {"tag": "Angle", "count": 2},
                ],
                "total": 3,
            },
            {"tags": ["Zip tie", "Bolt", "Bracket"], "total": 3},
        )
        self.assertEqual(names, ["Nut", "Angle", "Bolt", "Bracket", "Zip tie"])

    def test_a_malformed_payload_yields_no_names_rather_than_failing(self):
        """A candidate list is an aid; it must never take a panel down."""
        self.assertEqual(self.merge(None, None), [])
        self.assertEqual(self.merge({"tags": "Nut"}, {"tags": [{"nope": 1}, ""]}), [])

    def test_a_bare_array_is_accepted_too(self):
        """Tolerated so a future unwrapped response does not empty the list."""
        names = self.merge([{"tag": "Nut", "count": 1}], ["Washer"])
        self.assertEqual(names, ["Nut", "Washer"])


@unittest.skipIf(_node() is None, "Node is not installed")
class TagCandidateSourceTests(unittest.TestCase):
    """The picker must read the vocabulary response as the body itself.

    GET /tag-vocabulary carries no ETag, so api.getTagVocabulary() resolves to
    ``{tags, total}``; GET /index/{name}/tags does carry one, so it resolves to
    ``{data, etag}``. Reading the first as the second is a silent failure, so
    the call is pinned here in the only way a Python test can: by reading the
    source.
    """

    def test_the_vocabulary_response_is_not_unwrapped(self):
        source = (
            pathlib.Path(__file__).resolve().parent.parent
            / "web"
            / "src"
            / "tag-picker.js"
        ).read_text(encoding="utf-8")
        self.assertIn("vocabularyPayload = await api.getTagVocabulary();", source)
        self.assertNotIn("{ data } = await api.getTagVocabulary()", source)


if __name__ == "__main__":
    unittest.main()
