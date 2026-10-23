"""Guard: README internal links must resolve, and both languages must match.

Stripping the English echo from every Japanese heading broke every in-page link
that pointed at the old anchor, and a silently dead anchor is invisible until a
reader clicks it. This test recomputes the anchors from the headings and checks
every `](#...)` link against them.

It also checks that README.md and README_ja.md have the same heading structure.
The two files are meant to be the same document in two languages; a section
added to one and forgotten in the other is the failure mode this catches.
"""

import io
import pathlib
import re
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_EN = _REPO_ROOT / "README.md"
_JA = _REPO_ROOT / "README_ja.md"

_HEADING = re.compile(r"^(#{1,6}) +(.*?)\s*$")
_FENCE = re.compile(r"^\s*```")
_LINK = re.compile(r"\]\(#([^)]+)\)")


def _read(path):
    with io.open(path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def _headings(text):
    """Return [(level, title)] for every heading outside a fenced code block.

    Code blocks matter: the shell comments in this README start with '# ', which
    is indistinguishable from a level-1 heading without fence tracking.
    """
    out = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING.match(line)
        if m:
            out.append((len(m.group(1)), m.group(2)))
    return out


def _anchor(title):
    """GitHub's heading-anchor rule, as far as these READMEs exercise it.

    Lowercase, drop everything that is not a letter, digit, underscore, hyphen
    or space, then turn spaces into hyphens. Japanese characters are letters to
    ``\\w``, so they survive, which is what makes the Japanese anchors work.
    """
    text = title.lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def _links(text):
    """Return [(anchor, line_number)] for in-page links outside code blocks."""
    out = []
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in _LINK.finditer(line):
            out.append((m.group(1), number))
    return out


class ReadmeLinks(unittest.TestCase):
    def test_internal_links_resolve(self):
        for path in (_EN, _JA):
            with self.subTest(readme=path.name):
                text = _read(path)
                anchors = {_anchor(title) for _level, title in _headings(text)}
                self.assertTrue(anchors, "no headings were found")
                found = _links(text)
                self.assertTrue(found, "no internal links were found")
                for anchor, line in found:
                    self.assertIn(
                        anchor,
                        anchors,
                        f"{path.name}:{line} links to #{anchor}, which no heading "
                        "produces. Update the link when you rename a heading.",
                    )


class ReadmeParity(unittest.TestCase):
    def test_both_languages_have_the_same_structure(self):
        en = _headings(_read(_EN))
        ja = _headings(_read(_JA))
        self.assertEqual(
            [level for level, _title in en],
            [level for level, _title in ja],
            "README.md and README_ja.md have different heading structures. "
            "Every section must exist in both languages, at the same depth and "
            "in the same order.",
        )

    def test_japanese_headings_do_not_echo_the_english_title(self):
        # The Japanese headings used to read "## データディレクトリ (Data
        # directories)". The echo is gone; this keeps it from coming back, since
        # re-adding it would silently break every anchor again.
        for _level, title in _headings(_read(_JA)):
            with self.subTest(heading=title):
                self.assertFalse(
                    re.search(r"\([^()]*[A-Za-z][^()]*\)\s*$", title)
                    and not title.endswith("`)"),
                    f"Japanese heading '{title}' looks like it ends with an "
                    "English translation. Write the heading in Japanese only.",
                )


if __name__ == "__main__":
    unittest.main()
