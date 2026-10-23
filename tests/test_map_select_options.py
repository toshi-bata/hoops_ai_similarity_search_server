"""Guard: the map's <select> options, the client's allow-sets and the server agree.

The Layout select once offered a third layout, the server understood it, and
the client quietly normalised it to PCA -- so choosing it sent no ``projection``
at all and drew a PCA map with no sign that anything had been ignored. Nothing
failed, which is why a test rather than a code review has to hold this. (That
layout has since been withdrawn; the trap it fell into has not.)

The same shape as tests/test_env_doc_consistency.py: a cross-file agreement that
no single file can check for itself. Three sides are compared, not two -- the
bug was precisely that two of them (HTML and server) were right while the third
was not, so any pairwise check would have passed.
"""

import os
import re
import unittest

import shape_map

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# One <select ...> ... </select> block, captured by its id.
_SELECT = r'<select\s+id="{id}"[^>]*>(.*?)</select>'
_OPTION_VALUE = re.compile(r'<option\s+value="([^"]*)"')
# `const NAME = new Set(['a', 'b']);` -- the client's allow-set.
_JS_SET = r"const\s+{name}\s*=\s*new\s+Set\(\[([^\]]*)\]\)"
_JS_STRING = re.compile(r"""['"]([^'"]*)['"]""")


def _read(relative_path: str) -> str:
    with open(os.path.join(_ROOT, *relative_path.split("/")), encoding="utf-8") as handle:
        return handle.read()


def _select_values(html: str, select_id: str) -> set:
    """The values a <select> offers, as written in web/index.html."""
    match = re.search(_SELECT.format(id=re.escape(select_id)), html, re.DOTALL)
    assert match, f'web/index.html has no <select id="{select_id}">'
    values = set(_OPTION_VALUE.findall(match.group(1)))
    assert values, f'<select id="{select_id}"> offers no options'
    return values


def _js_set_values(source: str, name: str) -> set:
    """The members of a `new Set([...])` of string literals in a JS module."""
    match = re.search(_JS_SET.format(name=re.escape(name)), source, re.DOTALL)
    assert match, f"{name} is not defined as a `new Set([...])`"
    values = set(_JS_STRING.findall(match.group(1)))
    assert values, f"{name} is empty"
    return values


def _server_constant_values(prefix: str) -> set:
    """Every ``shape_map`` constant named ``prefix`` + something, as a set."""
    values = {
        value
        for name, value in vars(shape_map).items()
        if name.startswith(prefix) and isinstance(value, str)
    }
    assert values, f"shape_map defines no {prefix}* constants"
    return values


class MapProjectionOptionsTests(unittest.TestCase):
    """The Layout select, shape-map.js and shape_map.py must offer one set."""

    def test_html_options_match_the_client_allow_set(self):
        html = _select_values(_read("web/index.html"), "map-projection")
        js = _js_set_values(_read("web/src/shape-map.js"), "MAP_PROJECTIONS")
        self.assertEqual(
            html,
            js,
            "The Layout <select> and MAP_PROJECTIONS in web/src/shape-map.js differ.\n"
            f"  only in index.html:  {sorted(html - js)}\n"
            f"  only in shape-map.js: {sorted(js - html)}",
        )

    def test_html_options_match_the_server(self):
        html = _select_values(_read("web/index.html"), "map-projection")
        server = _server_constant_values("_PROJECTION_")
        self.assertEqual(
            html,
            server,
            "The Layout <select> and shape_map.py's _PROJECTION_* constants differ.\n"
            f"  only in index.html: {sorted(html - server)}\n"
            f"  only in shape_map:  {sorted(server - html)}",
        )

    def test_the_server_accepts_every_offered_layout_unchanged(self):
        """Behavioural check: the normalizer must not collapse an offered value.

        The constants existing is not enough -- _normalize_map_projection is a
        chain of comparisons, and a value it forgets silently becomes PCA, which
        is the client-side bug this file exists for, one layer down.
        """
        for value in sorted(_select_values(_read("web/index.html"), "map-projection")):
            with self.subTest(projection=value):
                self.assertEqual(shape_map._normalize_map_projection(value), value)


class MapClusteringOptionsTests(unittest.TestCase):
    """The Clustering select and the server's _CLUSTERING_* constants agree.

    Only two sides here. The client's setClustering() tests one value against a
    literal and falls back to the other, so it has no declared set to compare
    against; it cannot gain or lose a value without that line being rewritten,
    which is a different kind of change from adding an <option>. The same is
    true of Colour by (setColorBy) and Show (setKind), and neither of those has
    server-side constants to compare against either, so they are not pinned
    here. Giving each of them an allow-set would make them checkable -- worth
    doing if a third value is ever added to any of them.
    """

    def test_html_options_match_the_server(self):
        html = _select_values(_read("web/index.html"), "map-clustering")
        server = _server_constant_values("_CLUSTERING_")
        self.assertEqual(
            html,
            server,
            "The Clustering <select> and shape_map.py's _CLUSTERING_* constants differ.\n"
            f"  only in index.html: {sorted(html - server)}\n"
            f"  only in shape_map:  {sorted(server - html)}",
        )


if __name__ == "__main__":
    unittest.main()
