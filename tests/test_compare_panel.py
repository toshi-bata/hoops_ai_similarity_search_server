"""The comparison panel's "Align centres" switch and properties table.

Source inspections, because the client cannot be executed here.
"""
import pathlib
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_WEB = _ROOT / "web"


class ComparePanelMarkupTests(unittest.TestCase):
    def setUp(self):
        self.html = (_WEB / "index.html").read_text(encoding="utf-8")

    def test_the_align_switch_and_properties_table_exist(self):
        self.assertIn('id="compare-align"', self.html)
        self.assertIn('id="compare-props"', self.html)

    def test_the_explanatory_note_is_gone(self):
        self.assertNotIn("compare-note", self.html)
        css = (_WEB / "src" / "styles.css").read_text(encoding="utf-8")
        self.assertNotIn(".compare-note", css)


class CompareControllerTests(unittest.TestCase):
    def test_the_controller_exposes_the_new_calls(self):
        viewer = (_WEB / "src" / "viewer.js").read_text(encoding="utf-8")
        self.assertIn("setAligned(on)", viewer)
        self.assertIn("readProperties(which)", viewer)

    def test_the_panel_drives_them(self):
        compare = (_WEB / "src" / "compare.js").read_text(encoding="utf-8")
        self.assertIn("controller.setAligned(", compare)
        self.assertIn("ctrl.readProperties(", compare)


if __name__ == "__main__":
    unittest.main()
