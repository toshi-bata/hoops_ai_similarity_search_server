"""The viewport's light/dark switch.

The 3D viewer and the Shape Map share one pane, and its background is CSS: the
WebGL canvas is transparent and .viewer-pane's background shows through it. The
switch therefore redefines CSS custom properties and never touches the viewer's
own API -- the same way the HOOPS reference demo-app does it. These tests are
source inspections, because the client cannot be executed here.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_SRC = REPO_ROOT / "web" / "src"
STYLES = WEB_SRC / "styles.css"
INDEX_HTML = REPO_ROOT / "web" / "index.html"

# The dark literals the viewport chrome used to carry directly. Every one of
# them now belongs to a token definition and nowhere else; a rule that spells a
# colour out again is a rule the switch cannot reach.
VIEWPORT_LITERALS = (
    "#2b2f36",
    "rgba(20, 24, 30,",
    "rgba(38, 43, 51,",
    "#262b33",
    "#f4f5f7",
)

# Blocks that are allowed to spell those literals out, because defining them is
# their whole job.
TOKEN_SELECTORS = (":root", "[data-viewport-theme='light']")


def _blocks(css):
    """Yield (selector, body) for every top-level rule in the stylesheet."""
    without_comments = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", without_comments):
        yield match.group(1).strip(), match.group(2)


class TheHeaderCarriesTheSwitch(unittest.TestCase):
    def setUp(self):
        self.html = INDEX_HTML.read_text(encoding="utf-8")

    def test_the_toggle_button_exists(self):
        self.assertIn('id="viewport-theme-toggle"', self.html)
        self.assertIn('id="viewport-theme-icon"', self.html)

    def test_the_toggle_sits_in_the_app_header(self):
        header = self.html.split('<header class="app-header">', 1)[1]
        header = header.split("</header>", 1)[0]
        self.assertIn('id="viewport-theme-toggle"', header)

    def test_the_toggle_is_labelled_for_assistive_technology(self):
        button = self.html.split('id="viewport-theme-toggle"', 1)[1]
        button = button.split(">", 1)[0]
        self.assertIn("aria-label=", button)
        self.assertIn("title=", button)


class TheViewportHasALightLivery(unittest.TestCase):
    def setUp(self):
        self.css = STYLES.read_text(encoding="utf-8")

    def test_a_light_block_redefines_the_tokens(self):
        selectors = [selector for selector, _ in _blocks(self.css)]
        self.assertIn("[data-viewport-theme='light']", selectors)

    def test_the_light_block_only_redefines_tokens_the_dark_one_defines(self):
        dark = []
        light = []
        for selector, body in _blocks(self.css):
            names = re.findall(r"(--viewport-[\w-]+)\s*:", body)
            if selector == ":root":
                dark.extend(names)
            elif selector == "[data-viewport-theme='light']":
                light.extend(names)
        self.assertTrue(dark, "no dark viewport tokens found")
        self.assertTrue(light, "no light viewport tokens found")
        # Same names on both sides: a token defined on one side only would be
        # stuck at the other theme's value.
        self.assertEqual(sorted(dark), sorted(light))

    def test_the_pane_background_comes_from_a_token(self):
        for selector, body in _blocks(self.css):
            if selector == ".viewer-pane":
                self.assertIn("var(--viewport-bg)", body)
                return
        self.fail(".viewer-pane rule not found")

    def test_no_viewport_rule_spells_a_dark_colour_out(self):
        for selector, body in _blocks(self.css):
            if selector in TOKEN_SELECTORS:
                continue
            for literal in VIEWPORT_LITERALS:
                self.assertNotIn(
                    literal,
                    body,
                    f"{selector} hard-codes {literal}; the switch cannot reach it",
                )


class TheViewerApiIsNotUsedForTheBackground(unittest.TestCase):
    def test_no_client_module_calls_set_background_color(self):
        # Confirmed against the reference demo-app: its own switch records zero
        # calls to view.setBackgroundColor() and getBackgroundColor() reports
        # {top: null, bottom: null} in both states. A painted background would
        # sit on top of the CSS one and the two would disagree. Comments are
        # stripped first, because explaining the rule is not breaking it.
        offenders = []
        for path in sorted(WEB_SRC.glob("*.js")):
            source = path.read_text(encoding="utf-8")
            source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
            source = re.sub(r"(?m)^\s*//.*$", "", source)
            if "setBackgroundColor" in source:
                offenders.append(path.name)
        self.assertEqual(offenders, [])


class ThePortalledMenuFollowsTheTheme(unittest.TestCase):
    def test_the_row_menu_carries_the_panes_theme(self):
        # .map-row-menu is appended to <body>, so it inherits nothing from
        # .viewer-pane and has to be told which theme is current.
        source = (WEB_SRC / "shape-map.js").read_text(encoding="utf-8")
        self.assertIn("viewportTheme()", source)
        self.assertIn("menu.dataset.viewportTheme", source)


class TheChoiceIsRemembered(unittest.TestCase):
    def setUp(self):
        self.source = (WEB_SRC / "viewport-theme.js").read_text(encoding="utf-8")

    def test_the_storage_key_is_unique(self):
        key = re.search(r"const STORAGE_KEY = '([^']+)'", self.source).group(1)
        others = []
        for path in sorted(WEB_SRC.glob("*.js")):
            if path.name == "viewport-theme.js":
                continue
            others.extend(re.findall(r"'([\w.-]+)'", path.read_text(encoding="utf-8")))
        self.assertNotIn(key, others)

    def test_dark_stays_the_default(self):
        self.assertIn("const DEFAULT_THEME = 'dark'", self.source)


if __name__ == "__main__":
    unittest.main()
