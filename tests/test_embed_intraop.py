"""Tests for the embedding intra-op thread cap (HOOPS_AI_INTRAOP_THREADS).

All tests run without a HOOPS AI license or SDK install: they exercise the pure
environment-scoping logic that pins each embedding worker's BLAS/OpenMP thread
count, not the embedding itself.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core


class IntraopThreadsSetting(unittest.TestCase):
    """intraop_threads() reads HOOPS_AI_INTRAOP_THREADS, defaulting to 1."""

    def test_default_is_one(self):
        env = {k: v for k, v in os.environ.items() if k != "HOOPS_AI_INTRAOP_THREADS"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(core.intraop_threads(), 1)

    def test_explicit_value_honoured(self):
        with patch.dict(os.environ, {"HOOPS_AI_INTRAOP_THREADS": "4"}, clear=False):
            self.assertEqual(core.intraop_threads(), 4)

    def test_zero_disables_cap(self):
        with patch.dict(os.environ, {"HOOPS_AI_INTRAOP_THREADS": "0"}, clear=False):
            self.assertEqual(core.intraop_threads(), 0)

    def test_invalid_falls_back_to_default(self):
        with patch.dict(os.environ, {"HOOPS_AI_INTRAOP_THREADS": "not-a-number"}, clear=False):
            self.assertEqual(core.intraop_threads(), 1)


class IntraopThreadLimitContext(unittest.TestCase):
    """intraop_thread_limit() sets the BLAS/OpenMP vars, then restores them."""

    def test_sets_all_vars_then_restores_unset(self):
        env = {k: v for k, v in os.environ.items() if k not in core._INTRAOP_ENV_VARS}
        with patch.dict(os.environ, env, clear=True):
            with core.intraop_thread_limit(1):
                for name in core._INTRAOP_ENV_VARS:
                    self.assertEqual(os.environ.get(name), "1")
            # Every variable was unset before, so it must be unset again.
            for name in core._INTRAOP_ENV_VARS:
                self.assertNotIn(name, os.environ)

    def test_restores_prior_value(self):
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "8"}, clear=False):
            with core.intraop_thread_limit(2):
                self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "2")
            self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "8")

    def test_restores_on_exception(self):
        env = {k: v for k, v in os.environ.items() if k not in core._INTRAOP_ENV_VARS}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError):
                with core.intraop_thread_limit(1):
                    raise RuntimeError("boom")
            for name in core._INTRAOP_ENV_VARS:
                self.assertNotIn(name, os.environ)

    def test_zero_is_a_noop(self):
        env = {k: v for k, v in os.environ.items() if k not in core._INTRAOP_ENV_VARS}
        with patch.dict(os.environ, env, clear=True):
            with core.intraop_thread_limit(0):
                for name in core._INTRAOP_ENV_VARS:
                    self.assertNotIn(name, os.environ)


class IntraopThreadLimitLeakCheck(unittest.TestCase):
    """``intraop_thread_limit`` must leave the whole environment byte-identical
    before and after, so registration cannot leave the search process pinned to
    one BLAS thread. Registration runs in this same process before any search,
    so a leak here would make every later search slow.
    """

    def _assert_roundtrip_identical(self, threads):
        before = dict(os.environ)
        with core.intraop_thread_limit(threads):
            pass
        after = dict(os.environ)
        self.assertEqual(before, after)

    def test_identical_when_vars_were_unset(self):
        env = {k: v for k, v in os.environ.items() if k not in core._INTRAOP_ENV_VARS}
        with patch.dict(os.environ, env, clear=True):
            self._assert_roundtrip_identical(1)

    def test_identical_when_vars_were_set(self):
        preset = {name: "8" for name in core._INTRAOP_ENV_VARS}
        with patch.dict(os.environ, preset, clear=False):
            self._assert_roundtrip_identical(1)

    def test_identical_for_the_disabled_cap(self):
        # threads=0 is the "cap disabled" path; it must also leave no trace.
        env = {k: v for k, v in os.environ.items() if k not in core._INTRAOP_ENV_VARS}
        with patch.dict(os.environ, env, clear=True):
            self._assert_roundtrip_identical(0)


if __name__ == "__main__":
    unittest.main()
