"""The resolved run options a registration job would use.

Covers GET /similarity/embedding-plan and the invariants the two passes have to
keep, plus a tripwire for the rule this whole feature exists to protect: the
sizing arithmetic lives on the server and is never copied into the client.
"""
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import index_registry  # noqa: E402
from routers.index_jobs import router  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# A fixed environment, so a developer's own .env cannot decide a result.
ENV = {"HOOPS_AI_DATA_DIR": "C:\\tmp\\data", "HOOPS_LICENSE": "x"}


def _client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestEmbeddingPlanEndpoint(unittest.TestCase):
    def test_plan_reports_both_passes(self):
        with patch.dict(os.environ, ENV, clear=True):
            res = _client().get("/similarity/embedding-plan", params={"file_count": 500})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["file_count"], 500)
        for key in ("first_pass", "retry_pass"):
            with self.subTest(pass_name=key):
                self.assertGreaterEqual(body[key]["workers"], 1)
                self.assertGreaterEqual(body[key]["time_limit"], 1)
                self.assertIn("cap", body[key]["basis"])
                self.assertIn("cores", body[key]["basis"])
                self.assertIn("files", body[key]["basis"])

    def test_retry_pass_trades_workers_for_time(self):
        """Fewer workers, a longer budget -- the documented second pass.

        The relation is asserted, not the numbers: both depend on the host's
        cores and free RAM. ``<=`` rather than ``<`` because the file count
        bounds both passes, so a small selection can pin them together.
        """
        with patch.dict(os.environ, ENV, clear=True):
            body = _client().get(
                "/similarity/embedding-plan", params={"file_count": 500}
            ).json()
        self.assertLessEqual(body["retry_pass"]["workers"], body["first_pass"]["workers"])
        self.assertGreater(
            body["retry_pass"]["time_limit"], body["first_pass"]["time_limit"]
        )

    def test_retry_pass_is_strictly_smaller_on_a_known_host(self):
        """With the host pinned, the heavy-file factor has to actually bite."""
        env = dict(ENV)
        env["HOOPS_AI_MAX_WORKERS"] = "12"
        saved = sys.modules.get("psutil", "absent")
        sys.modules["psutil"] = None  # makes `import psutil` raise ImportError
        try:
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(
                        index_registry, "_physical_core_cap", return_value=(12, True)
                    ):
                body = _client().get(
                    "/similarity/embedding-plan", params={"file_count": 5000}
                ).json()
        finally:
            if saved == "absent":
                sys.modules.pop("psutil", None)
            else:
                sys.modules["psutil"] = saved
        self.assertEqual(body["first_pass"]["workers"], 12)
        self.assertEqual(body["retry_pass"]["workers"], 6)

    def test_an_unset_budget_is_reported_as_the_sdk_default(self):
        """0 means "pass nothing"; the UI must never be shown that number."""
        env = dict(ENV)
        env.pop("HOOPS_AI_EMBED_TIME_LIMIT", None)
        with patch.dict(os.environ, env, clear=True):
            body = _client().get(
                "/similarity/embedding-plan", params={"file_count": 10}
            ).json()
        self.assertEqual(body["first_pass"]["time_limit"], 120)
        self.assertNotEqual(body["retry_pass"]["time_limit"], 0)

    def test_a_configured_budget_wins(self):
        env = dict(ENV)
        env["HOOPS_AI_EMBED_TIME_LIMIT"] = "300"
        with patch.dict(os.environ, env, clear=True):
            body = _client().get(
                "/similarity/embedding-plan", params={"file_count": 10}
            ).json()
        self.assertEqual(body["first_pass"]["time_limit"], 300)

    def test_no_files_still_answers(self):
        with patch.dict(os.environ, ENV, clear=True):
            res = _client().get("/similarity/embedding-plan", params={"file_count": 0})
        self.assertEqual(res.status_code, 200)
        self.assertGreaterEqual(res.json()["first_pass"]["workers"], 1)

    def test_a_negative_file_count_is_rejected(self):
        with patch.dict(os.environ, ENV, clear=True):
            res = _client().get("/similarity/embedding-plan", params={"file_count": -1})
        self.assertEqual(res.status_code, 422)


class TestEffectiveTimeLimit(unittest.TestCase):
    def test_zero_becomes_the_sdk_default(self):
        with patch.dict(os.environ, ENV, clear=True):
            self.assertEqual(index_registry.embed_time_limit(), 0)
            self.assertEqual(index_registry.effective_embed_time_limit(), 120)

    def test_the_report_names_the_budget_that_was_in_force(self):
        """jobs.py must not print `time_limit=0s`, which reads as "no budget"."""
        text = _report_header({"num_workers": 8, "time_limit": 0})
        self.assertNotIn("time_limit=0s", text)
        self.assertIn("120s", text)
        self.assertIn("SDK default", text)
        self.assertIn("time_limit=300s", _report_header(
            {"num_workers": 8, "time_limit": 300}
        ))


def _report_header(timings):
    """Render a report for *timings* and return its `# Embedding:` line."""
    import tempfile

    import jobs
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.txt"
        jobs._write_add_job_report(path, "idx", [], [], [], timings)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# Embedding:"):
                return line
    return ""


class TestSizingIsNotDuplicatedInTheClient(unittest.TestCase):
    """The rule may exist in exactly one place.

    ``compute_auto_workers`` has been re-tuned repeatedly (the cap moved 8 -> 12,
    the core count went from ``logical // 2`` to a physical-core reading). Every
    one of those changes would have silently diverged from a second copy in the
    browser, where nothing would have failed. So: the client may read the
    server's answer, but it may not compute one.
    """

    # A min() over two or more of the quantities the server weighs is the shape
    # a re-implementation takes; the field assignments the client legitimately
    # does are not.
    SIZING_TERMS = ("cores", "cpu", "ram", "footprint", "cap", "workers")

    def test_web_src_contains_no_worker_sizing_arithmetic(self):
        offenders = []
        for path in sorted((REPO_ROOT / "web" / "src").rglob("*.js")):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r"Math\.min\(([^;{}]*?)\)", text, re.S):
                arg = match.group(1).lower()
                if "," not in arg:
                    continue
                hits = {term for term in self.SIZING_TERMS if term in arg}
                if len(hits) >= 2:
                    line = text.count("\n", 0, match.start()) + 1
                    offenders.append(f"{path.name}:{line} {sorted(hits)}")
        self.assertEqual(offenders, [], "worker sizing must stay on the server")

    def test_the_client_asks_the_server_for_the_plan(self):
        api = (REPO_ROOT / "web" / "src" / "api.js").read_text(encoding="utf-8")
        # The call itself, not the comment above it.
        self.assertIn("request('/similarity/embedding-plan'", api)
        register = (REPO_ROOT / "web" / "src" / "register.js").read_text(encoding="utf-8")
        self.assertIn("getEmbeddingPlan", register)


class TestOmittedOptionsStillReachTheServerSizing(unittest.TestCase):
    """An empty field must keep meaning "you decide"."""

    def test_read_options_omits_an_empty_field(self):
        text = (REPO_ROOT / "web" / "src" / "register.js").read_text(encoding="utf-8")
        start = text.index("function readOptions(")
        body = text[start:text.index("\n  }", start)]
        # Both assignments sit behind a truthiness check on the raw string, so a
        # cleared box contributes no key to the request body at all.
        self.assertIn("if (workersRaw) {", body)
        self.assertIn("if (timeRaw) {", body)
        self.assertIn("const options = {};", body)


if __name__ == "__main__":
    unittest.main()
