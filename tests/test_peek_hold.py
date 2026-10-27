"""Source tripwires for the hover preview's hold (web/src/shape-map.js).

Hovering a legend or tree row previews that cluster on its own. The preview
used to end the moment the pointer left the row, which meant that opening the
row's right-click menu and choosing Tag... took the picture away at exactly
the moment the user was asked to name what they were looking at. A hold now
keeps it up while the menu, and the tag popover it raises, are open.

The design has two parts that are easy to undo by accident, and neither shows
up as a failure anywhere else -- the symptom is a preview that blinks or one
that never comes back, both of which only a human at the screen would notice.
So they are pinned here, by reading the source, the way
tests/test_map_select_options.py pins the clustering options.

What is NOT pinned here is the behaviour itself. These modules need a DOM and
a viewer, so running them is not on offer; what can be checked is that the
shape of the solution is still the one that was reasoned about.
"""

import pathlib
import re
import unittest

_WEB = pathlib.Path(__file__).resolve().parent.parent / "web" / "src"
_SHAPE_MAP = _WEB / "shape-map.js"
_TAG_PICKER = _WEB / "tag-picker.js"

# The two wrappers that consult the hold. Everything the pointer drives must
# go through one of them; everything else -- a rebuild, a checkbox, switching
# index -- calls endPeek() directly and is meant to.
_HOVER_WRAPPERS = ("hoverPeek", "hoverEndPeek")


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_body(source: str, opening: str) -> str:
    """The text of a function, from its opening line to its matching brace.

    Counting starts at the brace that opens the BODY, not at the start of the
    signature: a destructured parameter with a default balances its own
    braces, and counting from the signature would stop there and hand back
    the first line alone.
    """
    start = source.index(opening)
    depth = 0
    for i in range(start + len(opening) - 1, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise AssertionError(f"unbalanced braces after {opening!r}")


class PeekHoldTests(unittest.TestCase):
    """The hold, as written in shape-map.js."""

    def test_tag_parts_does_not_end_the_hover_preview(self):
        """tagParts() used to call endPeek() before opening the popover.

        The stated reason was that a running preview would override the
        visibility being tagged against. It does not: the file ids are fixed
        when the menu is opened, so what is on screen cannot change what is
        written. Putting that call back would take the preview away again.
        """
        body = _function_body(_read(_SHAPE_MAP), "function tagParts(label, fileIds) {")
        self.assertNotIn("endPeek(", body)
        # It holds instead, and gives the popover a way to say it has gone.
        self.assertIn("holdPeek('tag')", body)
        self.assertIn("releasePeek('tag')", body)

    def test_every_hover_handler_goes_through_the_hold(self):
        """A pointer handler that calls endPeek()/schedulePeek() itself is
        invisible to the hold, so the preview would still vanish from that one
        row. There are six of them (two panels, plus each panel's container),
        and the next row type added would make a seventh.
        """
        lines = _read(_SHAPE_MAP).splitlines()
        for i, line in enumerate(lines):
            if not re.search(r"\b(on)?mouse(enter|leave)\b", line):
                continue
            # The call may sit a line or two below the handler it belongs to.
            window = "\n".join(lines[i : i + 3])
            with self.subTest(line=i + 1):
                for raw in ("endPeek(", "schedulePeek("):
                    self.assertNotIn(
                        raw,
                        window,
                        msg=f"line {i + 1} peeks without the hold; "
                        f"use one of {_HOVER_WRAPPERS}",
                    )

    def test_the_hold_is_a_token_not_a_counter(self):
        """A counter turns a missed release into a hold that never reaches
        zero, and the symptom -- hover previews silently dead for the rest of
        the session -- is not one anybody would trace back to a menu. The
        token is overwritten on acquire and checked on release, so a leak is
        bounded by the next menu.
        """
        source = _read(_SHAPE_MAP)
        for counting in ("peekHold++", "++peekHold", "peekHold +=", "peekHold -="):
            self.assertNotIn(counting, source)
        release = _function_body(source, "function releasePeek(token) {")
        self.assertIn("if (peekHold !== token) return;", release)

    def test_the_row_menu_holds_and_releases(self):
        """The menu is the other half: it takes the hold when it opens and
        gives it up when it closes, except on the handover to an item that
        opens a panel of its own.

        The release is in a `finally` because an item that throws would
        otherwise leave the hold standing, and a standing hold is not an error
        anyone sees -- the hover preview just stops working for the rest of
        the session.
        """
        source = _read(_SHAPE_MAP)
        self.assertIn("holdPeek('menu')", _function_body(source, "function openRowMenu(event, items, peek = null) {"))
        self.assertIn("releasePeek('menu')", _function_body(source, "function closeRowMenu(options) {"))
        self.assertRegex(
            source,
            r"\} finally \{\n(?:\s*//[^\n]*\n)*\s*releasePeek\('menu'\);\n\s*\}",
        )

    def test_the_preview_ends_before_an_ordinary_item_runs(self):
        """An item that touches the model selection must not run with the
        preview alive: refreshModelSelection() diffs against `modelFileIds`,
        which does not include the preview's models, so it claims them and the
        endPeek() that follows unloads what the item has just shown. That is
        how Pin parts broke. Only the handover to the tag popover is allowed
        to keep the hold, and the release has to come BEFORE item.run().
        """
        source = _read(_SHAPE_MAP)
        release = source.index("if (!keep) releasePeek('menu');")
        run = source.index("          item.run();")
        self.assertLess(release, run)

    def test_only_the_tag_items_keep_the_preview(self):
        """`keepPeek` is for an item that opens a panel which takes the hold
        for itself. Pin parts, Isolate, Hide, Show all, Heat map and Drill in
        are not such items, and putting it on one of them is the Pin parts bug
        again.
        """
        lines = _read(_SHAPE_MAP).splitlines()
        labels = []
        for i, line in enumerate(lines):
            if "keepPeek: true" not in line:
                continue
            # Prose about the option is not a use of it.
            if line.strip().startswith("//"):
                continue
            # The one that is not an item: the click handler always lets
            # closeRowMenu() carry the hold over, and decides separately.
            if "closeRowMenu(" in line:
                continue
            label = next(
                (lines[j] for j in range(i, max(i - 8, -1), -1) if "label:" in lines[j]),
                None,
            )
            self.assertIsNotNone(label, msg=f"line {i + 1}: keepPeek on no item")
            labels.append(label.strip())
        self.assertEqual(len(labels), 2, msg=f"expected the two tag items, got {labels}")
        for label in labels:
            with self.subTest(label=label):
                self.assertTrue(
                    label.startswith("label: 'Tag") or label.startswith("label: `Tag"),
                    msg=f"{label} is not a tag item and must not keep the preview",
                )


class TagPopoverCloseTests(unittest.TestCase):
    """openTagPopover()'s optional onClose (web/src/tag-picker.js)."""

    def test_the_popover_reports_that_it_closed(self):
        source = _read(_TAG_PICKER)
        self.assertIn("onClose = null", source)
        self.assertIn("if (panel.onClose) panel.onClose();", _function_body(source, "function closeOpenPopover() {"))

    def test_on_close_never_reaches_the_picker(self):
        """It is destructured out of the options, so the picker -- shared with
        the gallery, the results and the catalogue -- is handed exactly what
        it was handed before this option existed.
        """
        opening = _read(_TAG_PICKER).index("export function openTagPopover(")
        signature = _read(_TAG_PICKER)[opening : _read(_TAG_PICKER).index("\n", opening)]
        self.assertIn("onClose", signature)
        self.assertIn("...options", signature)
        self.assertLess(signature.index("onClose"), signature.index("...options"))


if __name__ == "__main__":
    unittest.main()
