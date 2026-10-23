"""Parity tests for the two hit-flattening paths in core.

``_flatten_body_hits`` (the public {id, score, metadata} view used by search)
and ``_flatten_body_hit_objects`` (the VectorHit objects handed to the Context
Layer predictor) must apply exactly the same concatenate / de-duplicate / cap
rules, or a prediction would be built from a different neighbour set than the
search that produced it. These tests pin that parity. They exercise only pure
functions in core, so no SDK is needed.
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import core  # noqa: E402


class _Hit:
    """Minimal stand-in for VectorHit (id / score / metadata)."""

    def __init__(self, id, score, metadata=None):
        self.id = id
        self.score = score
        self.metadata = metadata


def _results():
    # Two per-body ranked lists. "a" appears in both bodies; the second
    # occurrence must be dropped, keeping the first body's ranking.
    return [
        [_Hit("a", 0.9, {"file_id": "a", "_internal": 1}), _Hit("b", 0.8, {})],
        [_Hit("a", 0.5, {}), _Hit("c", 0.7, {"file_id": "c"})],
    ]


class TestFlattenParity(unittest.TestCase):
    def test_same_ids_same_order(self):
        objs = core._flatten_body_hit_objects(_results(), 10)
        dicts = core._flatten_body_hits(_results(), 10)
        self.assertEqual([o.id for o in objs], ["a", "b", "c"])
        self.assertEqual([d["id"] for d in dicts], [o.id for o in objs])

    def test_cap_matches(self):
        for k in (1, 2, 3, 5):
            objs = core._flatten_body_hit_objects(_results(), k)
            dicts = core._flatten_body_hits(_results(), k)
            self.assertEqual(len(objs), min(k, 3))
            self.assertEqual([d["id"] for d in dicts], [o.id for o in objs])

    def test_dedup_keeps_first_occurrence(self):
        objs = core._flatten_body_hit_objects(_results(), 10)
        # "a" keeps the first body's score, not the second body's 0.5.
        self.assertEqual(objs[0].id, "a")
        self.assertEqual(objs[0].score, 0.9)

    def test_dict_view_strips_internal_metadata(self):
        dicts = core._flatten_body_hits(_results(), 10)
        self.assertEqual(dicts[0]["metadata"], {"file_id": "a"})
        self.assertEqual(dicts[0]["score"], 0.9)

    def test_empty_and_none_safe(self):
        self.assertEqual(core._flatten_body_hit_objects([], 5), [])
        self.assertEqual(core._flatten_body_hits(None, 5), [])
        self.assertEqual(core._flatten_body_hit_objects([[], None], 5), [])


if __name__ == "__main__":
    unittest.main()
