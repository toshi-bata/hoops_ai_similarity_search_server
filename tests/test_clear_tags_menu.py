"""The map's "remove every tag from these parts" item.

The Ward tree's node menu and the legend's cluster rows both stand for a set of
parts, so both offer the item -- and both must offer the same one. These are
source inspections, because the client cannot be executed here.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_SRC = REPO_ROOT / "web" / "src"
SHAPE_MAP = WEB_SRC / "shape-map.js"
API = WEB_SRC / "api.js"


class BothPanelsShareOneBuilder(unittest.TestCase):
    def setUp(self):
        self.source = SHAPE_MAP.read_text(encoding="utf-8")

    def test_the_builder_exists(self):
        self.assertIn("function clearTagsMenuItems(", self.source)

    def test_the_tree_and_the_legend_both_call_it(self):
        # Two call sites besides the definition: the tree node's contextmenu
        # and groupMenuItems(). A panel that built its own item would show up
        # here as a missing call.
        calls = re.findall(r"for \(const item of clearTagsMenuItems\(", self.source)
        self.assertEqual(len(calls), 2)
        tree = self.source.index("Tag these ${node.count} parts")
        legend = self.source.index("function groupMenuItems(")
        after_tree = self.source.index("clearTagsMenuItems(", tree)
        after_legend = self.source.index("clearTagsMenuItems(", legend)
        self.assertNotEqual(after_tree, after_legend)

    def test_the_write_is_confirmed_first(self):
        body = self.source[self.source.index("async function clearTagsFromParts("):]
        body = body[: body.index("\n  }\n")]
        confirm = body.index("window.confirm(")
        call = body.index("api.clearPartTags(")
        self.assertLess(confirm, call)
        self.assertIn("This cannot be undone.", body)

    def test_the_result_is_not_an_alert(self):
        # The outcome goes to the shared banner, not to a modal the user has
        # to dismiss before they can read the map again.
        self.assertNotIn("alert(", self.source)


class TheClientCallsTheClearRoute(unittest.TestCase):
    def test_the_api_helper_confirms_to_the_server(self):
        source = API.read_text(encoding="utf-8")
        body = source[source.index("export function clearPartTags("):]
        body = body[: body.index("\n}\n")]
        self.assertIn("/parts/tags/clear", body)
        self.assertIn("confirm: true", body)


if __name__ == "__main__":
    unittest.main()
