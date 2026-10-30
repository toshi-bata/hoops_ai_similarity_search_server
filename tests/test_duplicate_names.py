"""Explaining the gap between the folder a user picked and the index.

A file_id is the SHA-256 of a CAD file's bytes, so byte-identical files
collapse into a single index entry. A client that registers a folder therefore
sees "10897 CAD file(s) selected" and a report that says ``input=10250``; the
647 missing from the index are the copies that shared a file_id with another
file in the same selection.

The figures come from the client that queued the job and describe **that
submission only**. An earlier version scanned the whole CAD store for names,
which mixed in filenames from registrations the reader had never made; the
first test class below is the guard against that returning.

SDK-free: no license, no faiss and no real index. The server's own data/
directory is never read.
"""

import os
import pathlib
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jobs  # noqa: E402
import upload_paths  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Real ids are 64 lower-case hex characters and the sanitiser checks that
# width, so the fakes have to be the real shape.
_ID_A = "a" * 64
_ID_B = "b" * 64


def _report(collapsed=None, selected=None, added=("a.stp",), failed=()):
    """Render a registration report and return its text."""
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "report.txt"
        jobs._write_add_job_report(
            path, "tmcad", list(added), list(failed), [], {},
            collapsed, selected,
        )
        return path.read_text(encoding="utf-8")


class RunLocalOnlyTests(unittest.TestCase):
    """The defect this feature was rebuilt for: names from another run.

    Registering the same geometry twice, from two different folders, must give
    two reports that each name only their own folder. Nothing is shared between
    them, so the second report cannot know the first one's path.
    """

    def test_a_later_run_does_not_inherit_an_earlier_runs_paths(self):
        first = {_ID_A: ["old_project/bracket.stp", "old_project/copy.stp"]}
        second = {_ID_A: ["new_project/part.stp", "new_project/part_rev_b.stp"]}
        text = _report(second, selected=2)
        self.assertIn("new_project/part.stp", text)
        for path in first[_ID_A]:
            self.assertNotIn(path, text)

    def test_a_run_that_sends_nothing_reports_nothing(self):
        # No selection information means no claim about one. A store scan would
        # have invented names here.
        text = _report(None, selected=None)
        self.assertIn("[DUPLICATES]", text)
        self.assertIn("0 shapes", text)
        self.assertNotIn("# Selection:", text)

    def test_only_ids_this_job_submitted_are_reported(self):
        """jobs.py filters the caller's map down to the job's own inputs."""
        source = (REPO_ROOT / "jobs.py").read_text(encoding="utf-8")
        body = source[source.index("def run_index_add_job("):]
        self.assertIn("if fid in set(file_ids)", body)
        # The store-wide scan must not come back.
        self.assertNotIn("map_duplicate_CAD_filenames", source)


class CollapsedCopiesSectionTests(unittest.TestCase):
    """The [DUPLICATES] section and the selection accounting above it."""

    def test_the_section_lists_every_path_of_each_shape(self):
        text = _report({_ID_A: ["asm/bracket.stp", "asm/bracket_rev_b.stp"]})
        self.assertIn("[DUPLICATES]", text)
        self.assertIn(f"{_ID_A}\tasm/bracket.stp, asm/bracket_rev_b.stp", text)

    def test_the_header_counts_the_copies_that_did_not_go_in(self):
        text = _report({
            _ID_A: ["a.stp", "a2.stp"],
            _ID_B: ["b.stp", "b2.stp", "b3.stp"],
        })
        self.assertIn("collapsed_copies=3", text)
        self.assertIn("2 shapes", text)

    def test_the_section_says_it_covers_this_selection_only(self):
        # The wording is the point: a reader must not take it for a statement
        # about the index, nor conclude the index holds duplicate entries.
        text = _report({_ID_A: ["a.stp", "a2.stp"]})
        self.assertIn("THIS selection", text)
        self.assertIn("this registration only", text)

    def test_the_section_is_present_even_when_empty(self):
        # Absence would be indistinguishable from an older report.
        self.assertIn("[DUPLICATES]", _report({}))

    def test_the_gap_between_selection_and_submission_is_stated(self):
        text = _report({_ID_A: ["a.stp", "a2.stp"]}, selected=3, added=["a.stp", "b.stp"])
        self.assertIn("# Selection: selected=3 submitted=2 collapsed=1", text)

    def test_the_arguments_are_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "report.txt"
            jobs._write_add_job_report(path, "tmcad", [], [], [], {})
            self.assertIn("[DUPLICATES]", path.read_text(encoding="utf-8"))


class SanitizeDuplicatePathsTests(unittest.TestCase):
    """The one gate the client's map passes through."""

    def test_paths_are_cleaned_and_sorted(self):
        got = upload_paths.sanitize_duplicate_paths(
            {_ID_A: ["asm\\b.stp", "./asm/a.stp"]}
        )
        self.assertEqual(got, {_ID_A: ["asm/a.stp", "asm/b.stp"]})

    def test_a_traversal_path_is_dropped(self):
        got = upload_paths.sanitize_duplicate_paths(
            {_ID_A: ["..\\..\\x.stp", "C:\\Windows\\y.stp", "ok/a.stp", "ok/b.stp"]}
        )
        self.assertEqual(got, {_ID_A: ["ok/a.stp", "ok/b.stp"]})

    def test_an_id_left_with_one_path_is_dropped(self):
        # One path is not a collapsed copy; keeping it would overstate the gap.
        self.assertEqual(
            upload_paths.sanitize_duplicate_paths({_ID_A: ["../evil", "ok.stp"]}),
            {},
        )

    def test_a_forged_file_id_is_refused(self):
        for key in ("", "not-a-hash", "A" * 64, "a" * 63):
            with self.subTest(key=key):
                self.assertEqual(
                    upload_paths.sanitize_duplicate_paths({key: ["a.stp", "b.stp"]}),
                    {},
                )

    def test_a_single_id_cannot_grow_without_limit(self):
        many = [f"d{i}/a.stp" for i in range(50)]
        got = upload_paths.sanitize_duplicate_paths({_ID_A: many})
        self.assertEqual(len(got[_ID_A]), upload_paths._MAX_PATHS_PER_ID)

    def test_junk_is_survived(self):
        for raw in (None, [], "paths", {_ID_A: "a.stp"}, {_ID_A: None}):
            with self.subTest(raw=raw):
                self.assertEqual(upload_paths.sanitize_duplicate_paths(raw), {})


class ClientShowsTheGapOnlyInTheResultTests(unittest.TestCase):
    """Nothing about duplicates may appear before a folder has been picked."""

    def _register_js(self):
        return (REPO_ROOT / "web" / "src" / "register.js").read_text(encoding="utf-8")

    def test_the_gap_is_written_when_a_job_settles(self):
        body = self._register_js()
        start = body.index("async function onJobSettled(")
        end = body.index("async function resubmit(")
        self.assertIn("gapLine", body[start:end])

    def test_nothing_duplicate_related_runs_on_open(self):
        body = self._register_js()
        start = body.index("function open() {")
        end = body.index("function close() {")
        opening = body[start:end]
        for forbidden in ("gapLine", "Duplicates", "duplicate"):
            with self.subTest(token=forbidden):
                self.assertNotIn(forbidden, opening)

    def test_the_removed_panel_is_gone(self):
        html = (REPO_ROOT / "web" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("register-duplicates", html)
        css = (REPO_ROOT / "web" / "src" / "styles.css").read_text(encoding="utf-8")
        self.assertNotIn("register-duplicates", css)

    def test_the_removed_endpoint_is_gone(self):
        api = (REPO_ROOT / "web" / "src" / "api.js").read_text(encoding="utf-8")
        self.assertNotIn("getIndexDuplicates", api)
        router = (REPO_ROOT / "routers" / "similarity.py").read_text(encoding="utf-8")
        self.assertNotIn('"/index/{name}/duplicates"', router)
        self.assertNotIn("get_index_duplicate_names", router)

    def test_the_selection_is_sent_with_the_job(self):
        body = self._register_js()
        self.assertIn("selectedCount: selectedFiles.length", body)
        self.assertIn("duplicatePaths: collapsed", body)
        api = (REPO_ROOT / "web" / "src" / "api.js").read_text(encoding="utf-8")
        self.assertIn("duplicate_paths: duplicatePaths", api)


class SummaryLineUsesTheServersFiguresTests(unittest.TestCase):
    """The result summary must print the job's figures, not invent its own.

    The client once rebuilt the gap as ``selected - (added + failed)``. That
    subtracts how embedding went from how many files were picked, so a run of
    42 files with one embedding failure announced 41 byte-identical copies
    while its report said ``collapsed=0``.
    """

    def setUp(self):
        self.js = (REPO_ROOT / "web" / "src" / "register.js").read_text(encoding="utf-8")
        start = self.js.index("const { selected, submitted, collapsed } = s;")
        self.block = self.js[start: self.js.index("retryableIds =", start)]

    def test_the_figures_come_straight_from_the_summary(self):
        self.assertIn("const { selected, submitted, collapsed } = s;", self.block)
        for derived in ("s.added", "s.failed", "added", "failed"):
            with self.subTest(token=derived):
                self.assertNotIn(derived, self.block)

    def test_no_count_is_rebuilt_from_added_plus_failed(self):
        # Tripwire: catch the arithmetic itself anywhere in the client, not
        # just in the block above, so moving it does not hide it.
        offenders = []
        for path in sorted((REPO_ROOT / "web" / "src").rglob("*.js")):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(
                r"(added|failed)\s*(\?\?\s*\d+\s*\)?)?\s*[+-]\s*"
                r"[^;\n]*\b(added|failed)\b",
                text,
            ):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.name}:{line} {match.group(0)[:48]}")
        self.assertEqual(
            offenders, [],
            "the gap is selected - submitted; the server publishes it",
        )

    def test_nothing_is_claimed_when_nothing_collapsed(self):
        self.assertIn("collapsed > 0", self.block)
        # One branch writes text, one hides the line. A second "all N went in"
        # branch would print a standing duplicate notice on every run.
        self.assertEqual(self.block.count("gapLine.hidden = false;"), 1)

    def test_the_wording_does_not_claim_the_files_were_embedded(self):
        self.assertIn("submitted", self.block)
        self.assertNotIn("went to the index", self.block)

    def test_a_job_without_a_selection_shows_nothing(self):
        self.assertIn("selected != null && submitted != null", self.block)
        # Resubmit is such a job: its input is the previous job's retryable
        # ids, which bear no relation to the original selection.
        resubmit = self.js[
            self.js.index("async function resubmit("): self.js.index("// -- Jobs list")
        ]
        for token in ("selectedCount", "duplicatePaths"):
            with self.subTest(token=token):
                self.assertNotIn(token, resubmit)


class SummaryAndReportAgreeTests(unittest.TestCase):
    """The summary figures and the report's "# Selection:" line are one sum."""

    def setUp(self):
        self.py = (REPO_ROOT / "jobs.py").read_text(encoding="utf-8")

    def test_the_summary_counts_the_lists_the_report_prints(self):
        body = self.py[self.py.index("def run_index_add_job("):]
        self.assertIn("submitted = len(report_added) + len(errors)", body)
        self.assertIn("collapsed=max(0, int(selected_count) - submitted),", body)
        # The same two lists are what the report is handed.
        self.assertIn("            report_added,", body)

    def test_the_report_computes_the_gap_the_same_way(self):
        body = self.py[
            self.py.index("def _write_add_job_report("):
            self.py.index("def run_index_add_job(")
        ]
        self.assertIn("submitted = len(added) + len(failed)", body)
        self.assertIn("collapsed={max(0, selected_count - submitted)}", body)

    def test_the_two_agree_on_a_worked_example(self):
        # 3 picked, 2 submitted (one of which fails to embed): the gap is 1,
        # and the embedding failure must not touch it.
        text = _report(
            {_ID_A: ["a.stp", "copy/a.stp"]},
            selected=3,
            added=["a.stp"],
            failed=[{"file_id": _ID_B, "filename": "b.stp", "error": "boom"}],
        )
        self.assertIn("# Selection: selected=3 submitted=2 collapsed=1", text)


class HtmlPatternTests(unittest.TestCase):
    """The result pane carries the gap line, and only it."""

    def test_the_gap_line_lives_inside_the_result_box(self):
        html = (REPO_ROOT / "web" / "index.html").read_text(encoding="utf-8")
        match = re.search(
            r'<div id="register-result".*?</div>\s*</div>', html, re.S
        )
        self.assertIsNotNone(match, "web/index.html has no register-result box")
        self.assertIn('id="register-gap"', match.group(0))


if __name__ == "__main__":
    unittest.main()
