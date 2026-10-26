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
# `const NAME = ['a', 'b'];` -- the same list where main.js keeps it as an array.
_JS_ARRAY = r"const\s+{name}\s*=\s*\[([^\]]*)\]"
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


def _js_array_values(source: str, name: str) -> set:
    """The members of a `const NAME = [...]` of string literals in a JS module."""
    match = re.search(_JS_ARRAY.format(name=re.escape(name)), source, re.DOTALL)
    assert match, f"{name} is not defined as a `[...]` array literal"
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
    """The Clustering select, shape-map.js and shape_map.py must offer one set.

    Three sides here too, for the reason the file's docstring gives. Until a
    third clustering existed the client tested one value against a literal and
    fell back to the other, so there was no declared set to compare against;
    adding average linkage turned that line into the same trap the Layout
    select fell into, and it was rewritten as MAP_CLUSTERINGS.

    Colour by (setColorBy) and Show (setKind) are still two-valued literals and
    have no server-side constants to compare against, so they are not pinned
    here. Worth giving each of them an allow-set if a third value is ever added.
    """

    def test_html_options_match_the_client_allow_set(self):
        html = _select_values(_read("web/index.html"), "map-clustering")
        js = _js_set_values(_read("web/src/shape-map.js"), "MAP_CLUSTERINGS")
        self.assertEqual(
            html,
            js,
            "The Clustering <select> and MAP_CLUSTERINGS in web/src/shape-map.js differ.\n"
            f"  only in index.html:  {sorted(html - js)}\n"
            f"  only in shape-map.js: {sorted(js - html)}",
        )

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

    def test_the_server_accepts_every_offered_clustering_unchanged(self):
        """Behavioural check, as for the Layout select one layer down.

        _normalize_map_clustering collapses what it does not recognise to
        k-means, so a clustering the constants know about but the normalizer
        forgets would silently draw a k-means map.
        """
        for value in sorted(_select_values(_read("web/index.html"), "map-clustering")):
            with self.subTest(clustering=value):
                self.assertEqual(shape_map._normalize_map_clustering(value), value)


class MapTreeClusteringTests(unittest.TestCase):
    """Which clusterings build a tree must be one answer, not three guesses.

    Adding average linkage left one line in main.js comparing the select's
    value with 'ward' directly. Everything else had moved to a predicate, so
    the map itself clustered correctly -- only the toolbar did not, and it
    offered Clusters (k) (the k-means control, ceiling 1024) where Max clusters
    (ceiling 4096) belonged. The deeper cuts were simply unreachable, which is
    the whole point of comparing the two linkage rules.

    The DOM layout itself is awkward to test without a browser, so this pins
    the three declared sets the layout is driven from, and then pins that
    main.js holds no literal comparison that could drift away from them again.
    """

    def test_tree_clusterings_agree_across_the_two_client_modules(self):
        shape = _js_set_values(_read("web/src/shape-map.js"), "MAP_TREE_CLUSTERINGS")
        main = _js_array_values(_read("web/src/main.js"), "MAP_TREE_CLUSTERINGS")
        self.assertEqual(
            shape,
            main,
            "MAP_TREE_CLUSTERINGS differs between web/src/shape-map.js and web/src/main.js.\n"
            f"  only in shape-map.js: {sorted(shape - main)}\n"
            f"  only in main.js:      {sorted(main - shape)}",
        )

    def test_tree_clusterings_agree_with_the_server(self):
        main = _js_array_values(_read("web/src/main.js"), "MAP_TREE_CLUSTERINGS")
        server = set(shape_map._CLUSTERING_TREE)
        self.assertEqual(
            main,
            server,
            "MAP_TREE_CLUSTERINGS and shape_map._CLUSTERING_TREE differ.\n"
            f"  only in main.js:   {sorted(main - server)}\n"
            f"  only in shape_map: {sorted(server - main)}",
        )

    def test_every_tree_clustering_is_an_offered_clustering(self):
        html = _select_values(_read("web/index.html"), "map-clustering")
        for value in sorted(shape_map._CLUSTERING_TREE):
            with self.subTest(clustering=value):
                self.assertIn(value, html)

    def test_main_js_never_compares_a_clustering_with_a_literal(self):
        """The one line that was missed, pinned so the next one cannot be.

        web/src/api.js is deliberately exempt: its two comparisons implement a
        different rule -- omit the parameter at the server's default so the
        request stays byte for byte what it was -- and that rule really does
        name one value.
        """
        source = _read("web/src/main.js")
        offenders = []
        for line_no, line in enumerate(source.splitlines(), start=1):
            code = line.split("//", 1)[0]
            for value in sorted(shape_map._CLUSTERING_TREE):
                if f"=== '{value}'" in code or f"!== '{value}'" in code:
                    offenders.append(f"  web/src/main.js:{line_no}: {line.strip()}")
        self.assertEqual(
            offenders,
            [],
            "web/src/main.js compares a clustering with a literal:\n"
            + "\n".join(offenders)
            + "\nUse treeClusteringSelected() or MAP_TREE_CLUSTERINGS instead: a "
            "literal here is what left Max clusters hidden under average linkage.",
        )


class MapClusteringStateTests(unittest.TestCase):
    """Client state that belongs to one map must be keyed by that map.

    Here rather than in a file of its own because it is the same failure the
    rest of this module guards: a value the client knows about that one place
    does not account for. The legend's checkboxes and the Similarity order were
    keyed by the node names the server returned, and node_name is the
    category's index (``group_0000``, ``group_0001``, ...) with nothing of its
    content in it -- so the list says only how many categories there are, and
    three different pairs of settings with equal counts shared one slot. They
    are keyed by the REQUEST now, and the tests below hold that line.

    Source inspection, like test_main_js_never_compares_a_clustering_with_a
    _literal above: the behaviour needs a browser, the invariant does not.
    """

    # Keys of the `params` object in load() that do NOT change which categories
    # the server returns, and so are deliberately absent from mapSignature.
    # Anything added to `params` fails the test below until it is either put in
    # the signature or listed here with its reason.
    _NOT_IN_SIGNATURE = {
        # A constant (MAP_DIMS). It fixes how many coordinates each point has,
        # not how the points are grouped.
        "dims",
        # The clustering runs on the full-dimensional vectors, so the layout
        # moves the points and leaves the categories exactly as they were.
        "projection",
    }

    def _function_body(self, source: str, opening: str) -> str:
        """The text from `opening` to the end of its brace-balanced block."""
        start = source.index(opening)
        depth = 0
        for offset in range(start, len(source)):
            char = source[offset]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return source[start:offset + 1]
        self.fail(f"{opening!r} is not brace-balanced in web/src/shape-map.js")

    def _signature_source(self) -> str:
        """mapSignature's array literal, comments stripped."""
        source = _read("web/src/shape-map.js")
        start = source.index("mapSignature = [")
        end = source.index("].join('|')", start)
        body = source[start:end]
        return "".join(line.split("//", 1)[0] for line in body.splitlines())

    def _params_keys(self) -> set:
        """The keys of the `params` object literal load() sends to /map."""
        source = _read("web/src/shape-map.js")
        body = self._function_body(source, "const params = {")
        keys = set()
        for line in body.splitlines():
            code = line.split("//", 1)[0]
            # Top-level keys only: nested values are indented further, and the
            # object is written one key per line at a fixed depth. The comma
            # alternative catches shorthand properties (`nClusters,`).
            match = re.match(r"^ {8}([A-Za-z_]\w*)\s*[:,]", code)
            if match:
                keys.add(match.group(1))
        assert keys, "load()'s `params` object has no keys"
        return keys

    def test_every_map_parameter_is_in_the_signature_or_excused(self):
        """A new request argument must be ruled in or out, not forgotten.

        Each of the three bugs was one argument that changed the categories and
        was not named. The point of this test is that the fourth cannot be
        silent: adding a key to `params` fails here until someone decides.
        """
        signature = self._signature_source()
        for key in sorted(self._params_keys() - self._NOT_IN_SIGNATURE):
            with self.subTest(param=key):
                self.assertIn(
                    f"params.{key}",
                    signature,
                    f"load() sends `{key}` to /map but mapSignature does not "
                    "name it. Either add it to the signature, or add it to "
                    "_NOT_IN_SIGNATURE with the reason it cannot change which "
                    "categories the server returns.",
                )

    def test_the_legend_signature_carries_the_clustering(self):
        """Named on its own, so the clustering going missing says so by name."""
        self.assertIn(
            "params.clustering",
            self._signature_source(),
            "mapSignature does not include the clustering, so Ward and Average "
            "share one legendStateFor() slot: they cut trees of the same shape "
            "at the same ceiling and nothing else in the signature differs.",
        )

    def test_the_legend_signature_is_not_built_from_the_response(self):
        """node_name is the category's index; it identifies nothing."""
        source = _read("web/src/shape-map.js")
        body = self._function_body(source, "function renderLegend(groups) {")
        code = "".join(line.split("//", 1)[0] for line in body.splitlines())
        self.assertNotIn(
            "node_name",
            code[:code.index("const remembered")],
            "renderLegend() is deriving its signature from the groups again. "
            "node_name is f'group_{ci:04d}' on the server, so the list names "
            "only how many categories there are: every two settings with the "
            "same count would share one slot.",
        )

    def test_the_untagged_pin_is_named_by_a_sequence_number(self):
        """Reload replaces the pin with nothing else in the signature moving."""
        self.assertIn(
            "untaggedPinSeq",
            self._signature_source(),
            "mapSignature does not name untaggedPinSeq. The Untagged-only pin "
            "is a bare list of file ids with no server-minted id, so Reload -- "
            "which drops it and takes another over a smaller population -- "
            "would otherwise produce the same signature as before.",
        )

    def test_set_clustering_discards_the_tree(self):
        source = _read("web/src/shape-map.js")
        body = self._function_body(source, "async setClustering(value) {")
        for name in ("treeData", "treeOpen", "treeSeeded", "treeColorOf"):
            with self.subTest(state=name):
                self.assertIn(
                    f"{name} =",
                    body,
                    f"setClustering() does not discard {name}. Ward and average "
                    "build different trees, so every node id names a family of "
                    "the clustering being left -- the same reason setWardMax() "
                    "discards it.",
                )


if __name__ == "__main__":
    unittest.main()
