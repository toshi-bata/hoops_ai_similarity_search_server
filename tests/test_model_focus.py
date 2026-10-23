"""Tests for pinned-category model selection (web/src/model-focus.js).

splitFocusedCandidates() decides what the 3D-model view loads when one or more
categories are pinned: the pinned ones IN FULL, then the ordinary
representative spread over whatever budget is left. Two properties are worth
pinning down. The candidate list must come back untouched -- it is the same
array the ordinary selection runs on, and a mutation here would quietly change
what the unpinned map shows. And a pinned category is taken whole or not at
all: half a cluster on screen cannot be told apart from the whole of it.

Run with Node, like tests/test_legend_order.py and for the same reason: the
module has no imports precisely so that it can be, and there is no JavaScript
test runner in this repository.
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

_MODULE = pathlib.Path(__file__).resolve().parent.parent / "web" / "src" / "model-focus.js"


def _node() -> str | None:
    return shutil.which("node")


def _candidates(*names):
    """One entry per name; `cat` is the category, null for "no category"."""
    return [{"cat": name, "i": i} for i, name in enumerate(names)]


@unittest.skipIf(_node() is None, "Node is not installed")
class SplitFocusedCandidatesTests(unittest.TestCase):
    """splitFocusedCandidates(candidates, keyOf, focusNames, budget)."""

    def split(self, candidates, focus, budget):
        """Run the real JS function; return the result and the input after it."""
        url = _MODULE.resolve().as_uri()
        driver = (
            f"import {{ splitFocusedCandidates }} from {json.dumps(url)};\n"
            f"const candidates = {json.dumps(candidates)};\n"
            f"const focus = new Set({json.dumps(focus)});\n"
            "const r = splitFocusedCandidates(\n"
            f"  candidates, (e) => e.cat, focus, {int(budget)},\n"
            ");\n"
            "process.stdout.write(JSON.stringify({\n"
            "  head: r.head, rest: r.rest, room: r.room,\n"
            "  used: [...r.used], dropped: r.dropped, candidates,\n"
            "}));\n"
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

    def test_two_pinned_categories_come_through_whole_and_in_order(self):
        cands = _candidates("a", "b", "a", "c", "b", "a")
        out = self.split(cands, ["a", "b"], 10)
        self.assertEqual([e["i"] for e in out["head"]], [0, 2, 5, 1, 4])
        self.assertEqual([e["i"] for e in out["rest"]], [3])
        self.assertEqual(out["room"], 5)
        self.assertEqual(sorted(out["used"]), ["a", "b"])
        self.assertEqual(out["dropped"], [])

    def test_nothing_pinned_leaves_the_whole_list_to_the_spread(self):
        cands = _candidates("a", "b", "c")
        out = self.split(cands, [], 7)
        self.assertEqual(out["head"], [])
        self.assertEqual(out["rest"], cands)
        self.assertEqual(out["room"], 7)
        self.assertEqual(out["used"], [])
        self.assertEqual(out["dropped"], [])

    def test_the_candidate_list_is_not_touched(self):
        """The one that matters: it is the ordinary selection's input too."""
        cands = _candidates("a", "b", "a", None)
        out = self.split(cands, ["a"], 1)
        self.assertEqual(out["candidates"], cands)

    def test_an_entry_with_no_category_is_always_left_to_the_spread(self):
        cands = _candidates("a", None, "a")
        out = self.split(cands, ["a"], 10)
        self.assertEqual([e["i"] for e in out["head"]], [0, 2])
        self.assertEqual([e["i"] for e in out["rest"]], [1])

    def test_a_pinned_name_the_map_does_not_have_is_harmless(self):
        cands = _candidates("a", "b")
        out = self.split(cands, ["ghost", "a"], 10)
        self.assertEqual([e["i"] for e in out["head"]], [0])
        self.assertEqual(out["used"], ["a"])
        # Not dropped either: it was not refused for want of budget.
        self.assertEqual(out["dropped"], [])

    def test_a_category_that_does_not_fit_is_left_out_whole(self):
        cands = _candidates("a", "a", "b", "b", "b")
        out = self.split(cands, ["a", "b"], 4)
        self.assertEqual([e["i"] for e in out["head"]], [0, 1])
        self.assertEqual([e["i"] for e in out["rest"]], [2, 3, 4])
        self.assertEqual(out["room"], 2)
        self.assertEqual(out["used"], ["a"])
        self.assertEqual(out["dropped"], ["b"])

    def test_the_first_category_alone_can_exhaust_the_budget(self):
        cands = _candidates("a", "a", "a", "b")
        out = self.split(cands, ["a", "b"], 2)
        self.assertEqual(out["head"], [])
        self.assertEqual([e["i"] for e in out["rest"]], [0, 1, 2, 3])
        self.assertEqual(out["room"], 2)
        self.assertEqual(out["used"], [])
        self.assertEqual(out["dropped"], ["a", "b"])

    def test_exactly_the_budget_fits(self):
        """The boundary is inclusive -- a cluster of 600 fills a budget of 600."""
        cands = _candidates("a", "a", "a")
        out = self.split(cands, ["a"], 3)
        self.assertEqual(len(out["head"]), 3)
        self.assertEqual(out["room"], 0)
        self.assertEqual(out["dropped"], [])

    def test_a_pinned_category_over_budget_still_leaves_room_for_the_spread(self):
        cands = _candidates("a", "a", "b")
        out = self.split(cands, ["a"], 1)
        self.assertEqual(out["head"], [])
        self.assertEqual(out["room"], 1)
        self.assertEqual(out["dropped"], ["a"])

    def test_the_pinning_order_decides_which_categories_fit(self):
        """Pinned order, not size: which clusters appear must be predictable."""
        cands = _candidates("big", "big", "big", "small")
        out = self.split(cands, ["big", "small"], 3)
        self.assertEqual(out["used"], ["big"])
        self.assertEqual(out["dropped"], ["small"])
        out = self.split(cands, ["small", "big"], 3)
        self.assertEqual(out["used"], ["small"])
        self.assertEqual(out["dropped"], ["big"])

    def test_an_empty_candidate_list(self):
        out = self.split([], ["a"], 5)
        self.assertEqual(out["head"], [])
        self.assertEqual(out["rest"], [])
        self.assertEqual(out["room"], 5)


if __name__ == "__main__":
    unittest.main()
