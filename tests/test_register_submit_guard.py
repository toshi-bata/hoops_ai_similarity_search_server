"""Upload & register does not come back after a run that reached the server.

runBulk() used to end with an unconditional

    } finally {
      submitButton.disabled = selectedFiles.length === 0;
    }

and the selection survives a successful run, so the button was live again the
moment the job was *queued* -- followJob() is not awaited. Pressing it a second
time re-ran the whole upload + register pass over the same files, which on a
ten-thousand-file folder is an expensive accident rather than a cosmetic one.

The button is now given back only when nothing was queued, and the run that
queued the job decides afterwards: a 'done' job clears the selection (which
disables the button through the existing empty-selection rule and clears the
"N CAD file(s) selected." line), while 'failed' or 'canceled' puts it back.

This is a source tripwire rather than a behavioural test -- the module needs a
DOM and a live server to run -- in the same spirit as
tests/test_legend_row_tooltips.py. It exists because the one-line finally reads
perfectly reasonable on its own; nothing in it says "this runs while the job is
still going".
"""

import pathlib
import re
import unittest

_REGISTER = pathlib.Path(__file__).resolve().parent.parent / "web" / "src" / "register.js"


def _function_body(source: str, opening: str) -> str:
    """The text of a function, from its opening line to its matching brace.

    Counting starts at the brace that opens the BODY, not at the start of the
    signature, so a destructured parameter with a default cannot close the
    count inside the parameter list.
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


class RegisterSubmitGuardTests(unittest.TestCase):
    def setUp(self):
        self.source = _REGISTER.read_text(encoding="utf-8")
        self.body = _function_body(self.source, "async function runBulk() {")

    def test_the_button_is_not_handed_back_unconditionally(self):
        # The guard has to involve whether a job was queued. Matching the bare
        # assignment catches a revert to the original line as well as anyone
        # re-adding it alongside the new one.
        bare = [
            line.strip()
            for line in self.body.splitlines()
            if re.match(r"^submitButton\.disabled\s*=\s*selectedFiles\.length === 0;$", line.strip())
        ]
        self.assertEqual(
            bare,
            [],
            msg=(
                "runBulk() hands the button back from the selection alone; it runs "
                "while the queued job is still going, so the run becomes repeatable"
            ),
        )
        self.assertRegex(
            self.body,
            r"submitButton\.disabled = queued \|\| selectedFiles\.length === 0;",
            msg="runBulk() no longer keeps the button disabled once a job was queued",
        )

    def test_only_a_finished_job_clears_the_selection(self):
        # 'done' is the only terminal status that means the files were actually
        # registered: a 'failed' or 'canceled' job must leave the selection in
        # place, because Resubmit only covers server-marked retryable ids.
        self.assertRegex(
            self.body,
            r"onSettled: \(job\) => setSelection\(job\.status === 'done' \? \[\] : selectedFiles\)",
            msg="runBulk() no longer clears the selection on a finished job only",
        )

    def test_the_settled_callback_is_optional_for_the_other_paths(self):
        # Resubmit and "register the open file" own no selection and must keep
        # working untouched, so the callback stays opt-in all the way down.
        for opening in (
            "async function queueAndFollow(name, fileIds, options, { embedFrom = 0, embedTo = 100, onSettled = null } = {}) {",
            "async function followJob(name, jobId, onSettled = null) {",
            "async function onJobSettled(name, job, onSettled = null) {",
        ):
            with self.subTest(opening=opening.split("(")[0]):
                self.assertIn(opening, self.source)


if __name__ == "__main__":
    unittest.main()
