"""Client-supplied upload paths: what is accepted, and what never reaches disk.

The first class is the important one. ``webkitRelativePath`` is attacker
controlled, so the rule under test is that nothing a client sends can influence
where a file is written, and that a value which could not have come from a
folder picker is refused rather than salvaged.
"""
import json
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import storage  # noqa: E402
import upload_paths  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SHA_A = "a" * 64
SHA_B = "b" * 64


class TestPathTraversalIsRefused(unittest.TestCase):
    """Refused outright, not trimmed to a basename.

    A folder picker never produces any of these, so the value is not a path the
    server asked for; showing a salvaged piece of it would put a forged string
    in front of an operator as if it were their own folder layout.
    """

    REFUSED = [
        "../../etc/passwd",
        "..\\..\\x.stp",
        "a/../../b.stp",
        "C:\\Windows\\x.stp",
        "c:/windows/x.stp",
        "/etc/passwd",
        "\\\\server\\share\\x.stp",
        "/",
        "..",
        "",
        "   ",
        "a/b/\n[ADDED] forged line",
        "a\tb.stp",
        "x" * 401,
        "/".join(["d"] * 40) + "/x.stp",
    ]

    def test_dangerous_values_produce_nothing(self):
        for value in self.REFUSED:
            with self.subTest(value=value[:40]):
                self.assertEqual(upload_paths.sanitize_relative_path(value), "")

    def test_non_strings_produce_nothing(self):
        for value in (None, 7, b"a/b.stp", ["a"], {"a": 1}):
            with self.subTest(value=repr(value)):
                self.assertEqual(upload_paths.sanitize_relative_path(value), "")

    def test_a_refused_value_is_never_recorded(self):
        with _store(self) as path:
            stored = upload_paths.record([(SHA_A, v) for v in self.REFUSED])
            self.assertEqual(stored, 0)
            self.assertFalse(path.exists(), "nothing to record must write nothing")
            self.assertEqual(upload_paths.read_all(), {})

    def test_a_hand_edited_file_is_re_checked_on_read(self):
        """A value that never passed the check must not reach a report."""
        with _store(self) as path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "schema_version": 1,
                "paths": {SHA_A: ["../../etc/passwd", "ok/part.stp"]},
            }), encoding="utf-8")
            self.assertEqual(upload_paths.read_all(), {SHA_A: ["ok/part.stp"]})


class TestAcceptedPaths(unittest.TestCase):
    def test_a_folder_path_survives(self):
        self.assertEqual(
            upload_paths.sanitize_relative_path("gearbox/housing/cover.stp"),
            "gearbox/housing/cover.stp",
        )

    def test_backslashes_become_forward_slashes(self):
        self.assertEqual(
            upload_paths.sanitize_relative_path("gearbox\\housing\\cover.stp"),
            "gearbox/housing/cover.stp",
        )

    def test_a_bare_filename_survives(self):
        self.assertEqual(upload_paths.sanitize_relative_path("cover.stp"), "cover.stp")

    def test_redundant_segments_are_dropped(self):
        self.assertEqual(
            upload_paths.sanitize_relative_path("./a//b/./c.stp"), "a/b/c.stp"
        )


class TestSeveralPathsPerFileId(unittest.TestCase):
    """The same bytes in two folders is the case this exists to show."""

    def test_one_id_keeps_every_folder_it_arrived_in(self):
        with _store(self):
            upload_paths.record([
                (SHA_A, "asm1/cover.stp"),
                (SHA_A, "asm2/cover.stp"),
                (SHA_B, "asm1/shaft.stp"),
            ])
            self.assertEqual(
                upload_paths.read_all(),
                {SHA_A: ["asm1/cover.stp", "asm2/cover.stp"], SHA_B: ["asm1/shaft.stp"]},
            )

    def test_recording_the_same_path_twice_adds_nothing(self):
        with _store(self):
            self.assertEqual(upload_paths.record([(SHA_A, "a/x.stp")]), 1)
            self.assertEqual(upload_paths.record([(SHA_A, "a/x.stp")]), 0)
            self.assertEqual(upload_paths.read_all()[SHA_A], ["a/x.stp"])

    def test_paths_per_id_are_bounded(self):
        with _store(self):
            upload_paths.record([(SHA_A, f"d{i}/x.stp") for i in range(40)])
            self.assertEqual(len(upload_paths.read_all()[SHA_A]), 8)

    def test_only_the_asked_for_ids_come_back(self):
        with _store(self):
            upload_paths.record([(SHA_A, "a/x.stp"), (SHA_B, "b/y.stp")])
            self.assertEqual(upload_paths.paths_for([SHA_B]), {SHA_B: ["b/y.stp"]})

    def test_a_bad_file_id_is_ignored(self):
        with _store(self):
            self.assertEqual(upload_paths.record([("../x", "a/x.stp")]), 0)


class TestFallbackForDataWithoutPaths(unittest.TestCase):
    """Everything uploaded before this existed has no recorded path."""

    def test_names_without_a_recorded_path_are_returned_unchanged(self):
        self.assertEqual(
            upload_paths.decorate_names(None, ["cover.stp", "shaft.stp"]),
            ["cover.stp", "shaft.stp"],
        )

    def test_a_recorded_path_decorates_the_name_it_matches(self):
        self.assertEqual(
            upload_paths.decorate_names(
                ["asm1/cover.stp"], ["cover.stp", "shaft.stp"]
            ),
            ["asm1/cover.stp", "shaft.stp"],
        )

    def test_the_original_filename_is_recovered_from_a_stored_name(self):
        self.assertEqual(storage.original_filename(f"{SHA_A}_cover.stp"), "cover.stp")

    def test_the_store_directory_is_never_disclosed(self):
        self.assertEqual(
            storage.original_filename(f"C:\\data\\uploads\\{SHA_A}_cover.stp"),
            "cover.stp",
        )
        self.assertEqual(
            storage.original_filename(f"/srv/data/uploads/{SHA_A}_cover.stp"),
            "cover.stp",
        )

    def test_a_name_without_the_prefix_keeps_its_basename(self):
        self.assertEqual(storage.original_filename("/srv/data/cover.stp"), "cover.stp")
        self.assertEqual(storage.original_filename(""), "")


class TestReportHidesServerPaths(unittest.TestCase):
    """A tripwire: putting a stored path back into the report must fail here."""

    SOURCES = ("jobs.py",)

    def test_the_report_builder_never_resolves_a_stored_path(self):
        text = (REPO_ROOT / "jobs.py").read_text(encoding="utf-8")
        start = text.index("def run_index_add_job(")
        body = text[start:]
        for forbidden in ("path.resolve()", "str(path)"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, body)

    def test_every_name_in_the_report_goes_through_the_display_map(self):
        text = (REPO_ROOT / "jobs.py").read_text(encoding="utf-8")
        self.assertIn("display_by_fid", text)
        self.assertIn("storage.original_filename", text)

    def test_a_written_report_carries_no_server_path(self):
        import jobs
        import tempfile
        timings = {"num_workers": 2, "time_limit": 300}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.txt"
            jobs._write_add_job_report(
                out,
                "idx",
                ["asm1/cover.stp", "shaft.stp"],
                [{"filename": "asm2/bad.stp", "detail": "timeout", "retryable": True}],
                ["asm1/heavy.stp"],
                timings,
                {SHA_A: ["asm1/cover.stp", "asm2/cover.stp"]},
            )
            text = out.read_text(encoding="utf-8")
        for forbidden in ("data/uploads", "data\\uploads", "C:\\", "/srv/"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)
        # The 64-char prefix is this server's bookkeeping; a name carrying it
        # means a stored filename reached the report unconverted.
        self.assertNotIn(f"{SHA_A}_", text)
        self.assertIn("asm1/cover.stp", text)


class TestUploadBatchRecordsThePaths(unittest.TestCase):
    """The positional pairing between `files` and `relative_paths`."""

    def _client(self, cad_dir):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers.files import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_each_file_keeps_its_own_folder(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, _store(self):
            cad = Path(tmp) / "uploads"
            cad.mkdir()
            with patch.object(storage, "get_CAD_shared_dir", lambda: cad):
                res = self._client(cad).post(
                    "/files/upload-batch",
                    files=[
                        ("files", ("a.stp", b"alpha", "application/octet-stream")),
                        ("files", ("b.stp", b"beta", "application/octet-stream")),
                    ],
                    data={"relative_paths": ["asm1/a.stp", "asm2/b.stp"]},
                )
            self.assertEqual(res.status_code, 200)
            ids = [f["file_id"] for f in res.json()["files"]]
            recorded = upload_paths.read_all()
            self.assertEqual(recorded[ids[0]], ["asm1/a.stp"])
            self.assertEqual(recorded[ids[1]], ["asm2/b.stp"])

    def test_a_traversal_path_does_not_decide_where_the_file_lands(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, _store(self):
            cad = Path(tmp) / "uploads"
            cad.mkdir()
            with patch.object(storage, "get_CAD_shared_dir", lambda: cad):
                res = self._client(cad).post(
                    "/files/upload-batch",
                    files=[("files", ("a.stp", b"alpha", "application/octet-stream"))],
                    data={"relative_paths": ["..\\..\\..\\evil.stp"]},
                )
            self.assertEqual(res.status_code, 200)
            fid = res.json()["files"][0]["file_id"]
            # Stored under the usual name, inside the store, and nothing else
            # was created anywhere.
            self.assertEqual([p.name for p in cad.iterdir()], [f"{fid}_a.stp"])
            self.assertEqual(list(Path(tmp).iterdir()), [cad])
            self.assertEqual(upload_paths.read_all(), {})

    def test_omitting_the_field_still_uploads(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, _store(self):
            cad = Path(tmp) / "uploads"
            cad.mkdir()
            with patch.object(storage, "get_CAD_shared_dir", lambda: cad):
                res = self._client(cad).post(
                    "/files/upload-batch",
                    files=[("files", ("a.stp", b"alpha", "application/octet-stream"))],
                )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["count"], 1)
            self.assertEqual(upload_paths.read_all(), {})


class TestUploadResponseUsesTheClientName(unittest.TestCase):
    """The upload endpoints must answer with the name that was sent.

    ``<sha256>_<name>`` is this server's bookkeeping. Handing it to a caller
    forces the caller to know the rename rule, which is exactly the coupling
    this check exists to keep broken.
    """

    STORED_NAME_RE = re.compile(r"^[0-9a-f]{64}_", re.I)

    def _upload(self, endpoint, cad, **kwargs):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers.files import router
        app = FastAPI()
        app.include_router(router)
        with patch.object(storage, "get_CAD_shared_dir", lambda: cad):
            return TestClient(app).post(endpoint, **kwargs)

    def test_single_upload_answers_with_the_sent_name(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cad = Path(tmp) / "uploads"
            cad.mkdir()
            res = self._upload(
                "/files/upload", cad,
                files={"file": ("cover.stp", b"alpha", "application/octet-stream")},
            )
            self.assertEqual(res.status_code, 200)
            body = res.json()
            self.assertEqual(body["filename"], "cover.stp")
            self.assertIsNone(self.STORED_NAME_RE.match(body["filename"]))
            # The bytes still landed under the stored name; only the answer changed.
            self.assertEqual(
                [p.name for p in cad.iterdir()], [f"{body['file_id']}_cover.stp"]
            )

    def test_batch_upload_answers_with_the_sent_names(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, _store(self):
            cad = Path(tmp) / "uploads"
            cad.mkdir()
            res = self._upload(
                "/files/upload-batch", cad,
                files=[
                    ("files", ("cover.stp", b"alpha", "application/octet-stream")),
                    ("files", ("shaft.stp", b"beta", "application/octet-stream")),
                ],
            )
            self.assertEqual(res.status_code, 200)
            names = [f["filename"] for f in res.json()["files"]]
            self.assertEqual(names, ["cover.stp", "shaft.stp"])
            for name in names:
                with self.subTest(name=name):
                    self.assertIsNone(self.STORED_NAME_RE.match(name))


class TestClientDoesNotReimplementTheRenameRule(unittest.TestCase):
    """A tripwire: the client must not learn how files are named on disk.

    ``main.js`` used to carry a ``/^[0-9a-f]{64}_(.+)$/`` that undid the stored
    name the server handed back. The server now hands back the client's own
    name, so any 64-wide match in the client means the coupling has returned.
    """

    def test_no_sha_prefix_pattern_appears_in_the_client(self):
        offenders = []
        for path in sorted((REPO_ROOT / "web" / "src").rglob("*.js")):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r"\{\s*64\s*\}|\b64\b\s*\)", text):
                window = text[match.start(): match.end() + 24]
                if "_" in window or "0-9a-f" in window.lower():
                    line = text.count("\n", 0, match.start()) + 1
                    offenders.append(f"{path.name}:{line} {window.strip()[:40]}")
        self.assertEqual(
            offenders, [], "the stored-name rule must stay on the server"
        )


class TestCoreReExportsAreTheSameObjects(unittest.TestCase):
    """core.<name> must BE upload_paths.<name>, not an equal copy."""

    PAIRS = (
        ("record_upload_paths", "record"),
        ("upload_paths_for", "paths_for"),
        ("decorate_upload_names", "decorate_names"),
        ("sanitize_relative_path", "sanitize_relative_path"),
    )

    def test_each_facade_name_is_bound_to_the_module_object(self):
        import core
        for public, private in self.PAIRS:
            with self.subTest(name=public):
                self.assertIs(getattr(core, public), getattr(upload_paths, private))


class _store:
    """Point the store at a temp directory for the duration of a block."""

    def __init__(self, case):
        self._case = case
        self._tmp = None
        self._patch = None

    def __enter__(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._patch = patch.object(upload_paths, "_store_path",
                                   lambda: root / "upload_paths.json")
        self._patch.start()
        return root / "upload_paths.json"

    def __exit__(self, *exc):
        self._patch.stop()
        self._tmp.cleanup()
        return False


if __name__ == "__main__":
    unittest.main()
