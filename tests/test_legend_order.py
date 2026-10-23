"""Tests for the legend's draw order (web/src/legend-order.js).

The groups array is POSITIONAL: the viewer resolves a pick to
``groups[node].offset + index``, and the baked ``.scs`` holds one node per
category in that order. So the legend's similarity ordering must never touch
it -- it reorders the indices the rows are drawn from and leaves the array
alone. A bug here would not look like a bug: the rows would be in some order,
and the picks would land on the wrong parts.

Run with Node, like tests/test_tag_candidates.py and for the same reason: the
module has no imports precisely so that it can be, and there is no JavaScript
test runner in this repository.
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

_MODULE = pathlib.Path(__file__).resolve().parent.parent / "web" / "src" / "legend-order.js"


def _node() -> str | None:
    return shutil.which("node")


def _groups(*keys):
    """Minimal legend groups: the key is all the function reads."""
    return [{"key": key, "node_name": f"N{i}"} for i, key in enumerate(keys)]


@unittest.skipIf(_node() is None, "Node is not installed")
class LegendRenderOrderTests(unittest.TestCase):
    """legendRenderOrder(groups, order) -> [index into groups, ...]."""

    def order(self, groups, order):
        """Run the real JS function; return (indices, groups afterwards)."""
        url = _MODULE.resolve().as_uri()
        driver = (
            f"import {{ legendRenderOrder }} from {json.dumps(url)};\n"
            f"const groups = {json.dumps(groups)};\n"
            f"const order = {json.dumps(order)};\n"
            "const out = legendRenderOrder(groups, order);\n"
            "process.stdout.write(JSON.stringify({ out, groups }));\n"
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
        payload = json.loads(result.stdout)
        return (payload["out"], payload["groups"])

    def test_clusters_are_drawn_in_the_order_given(self):
        groups = _groups("cluster:0", "cluster:1", "cluster:2")
        out, _after = self.order(groups, [2, 0, 1])
        self.assertEqual(out, [2, 0, 1])

    def test_the_groups_array_is_not_touched(self):
        """The one that matters: a reordered array misdirects every pick."""
        groups = _groups("cluster:0", "cluster:1", "cluster:2")
        _out, after = self.order(groups, [2, 1, 0])
        self.assertEqual(after, groups)

    def test_no_order_keeps_the_array_order(self):
        groups = _groups("cluster:0", "cluster:1")
        self.assertEqual(self.order(groups, None)[0], [0, 1])
        self.assertEqual(self.order(groups, [])[0], [0, 1])

    def test_outliers_are_not_dropped(self):
        """A missing row is a category nobody can switch back on."""
        groups = _groups("cluster:0", "outliers", "cluster:1")
        out, _after = self.order(groups, [1, 0])
        self.assertEqual(out, [2, 0, 1])

    def test_categories_not_in_the_order_keep_their_relative_order(self):
        groups = _groups("untagged", "cluster:5", "other", "cluster:4")
        out, _after = self.order(groups, [4, 5])
        self.assertEqual(out, [3, 1, 0, 2])

    def test_a_cluster_the_order_does_not_name_goes_last(self):
        groups = _groups("cluster:0", "cluster:1", "cluster:2")
        out, _after = self.order(groups, [2, 0])
        self.assertEqual(out, [2, 0, 1])

    def test_an_unknown_cluster_id_in_the_order_is_ignored(self):
        groups = _groups("cluster:0", "cluster:1")
        out, _after = self.order(groups, [9, 1, 0])
        self.assertEqual(out, [1, 0])

    def test_a_repeated_id_does_not_repeat_a_row(self):
        groups = _groups("cluster:0", "cluster:1")
        out, _after = self.order(groups, [1, 1, 0])
        self.assertEqual(out, [1, 0])

    def test_the_result_is_always_a_permutation(self):
        groups = _groups("cluster:3", "outliers", "cluster:1", "untagged", "cluster:2")
        for order in ([2, 1, 3], [], [3], [99], [1, 2, 3]):
            with self.subTest(order=order):
                out, _after = self.order(groups, order)
                self.assertEqual(sorted(out), list(range(len(groups))))

    def test_empty_legend(self):
        self.assertEqual(self.order([], [1, 2])[0], [])


if __name__ == "__main__":
    unittest.main()
