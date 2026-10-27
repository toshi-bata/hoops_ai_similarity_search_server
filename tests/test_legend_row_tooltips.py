"""The list rows carry no native tooltips (web/src/shape-map.js).

appendGroupIdentity() builds a row for both panels -- the legend's categories
and the tree's cut -- so a tooltip added there appears on every row of a list
that can run to hundreds. The tree rows already stated the rule, in the
comment that explains why the merge height is NOT in a tooltip:

    a tooltip over every row of a long list follows the pointer around and
    covers the rows being read

and the shared builder was simply not following it. It is followed now: what
the cohesion column means is on the Legend heading, once, and the row itself
says nothing on hover.

This is a source tripwire rather than a behavioural test -- the module needs a
DOM and a viewer to run -- in the same spirit as tests/test_peek_hold.py. It
exists because a tooltip is a one-line addition that looks helpful in
isolation, which is exactly how the two removed here were written.

Toolbar controls in web/index.html keep their title attributes on purpose:
those sit over a single control the pointer has stopped on, not over a list.
"""

import pathlib
import unittest

_SHAPE_MAP = pathlib.Path(__file__).resolve().parent.parent / "web" / "src" / "shape-map.js"


def _function_body(source: str, opening: str) -> str:
    """The text of a function, from its opening line to its matching brace.

    Counting starts at the brace that opens the BODY, not at the start of the
    signature: a destructured parameter with a default -- which is how
    appendGroupIdentity() is written -- balances its own braces, and counting
    from the signature would stop there and hand back the first line alone.
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


class LegendRowTooltipTests(unittest.TestCase):
    def test_the_shared_row_builder_sets_no_tooltip(self):
        source = _SHAPE_MAP.read_text(encoding="utf-8")
        body = _function_body(
            source,
            "function appendGroupIdentity(row, group, { stretchLabel = true } = {}) {",
        )
        offenders = [
            line.strip()
            for line in body.splitlines()
            if ".title =" in line and not line.strip().startswith("//")
        ]
        self.assertEqual(
            offenders,
            [],
            msg="a row tooltip is back; put what it says on the Legend heading instead",
        )


if __name__ == "__main__":
    unittest.main()
