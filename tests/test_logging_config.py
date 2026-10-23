"""Startup logging configuration.

uvicorn configures only its own ``uvicorn.*`` loggers, so the application has to
give the root logger a handler itself. Without it every ``logger.info()`` in
core and the routers is dropped -- which is how the assembly matcher build time
went missing.
"""
import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main


class TestConfigureLogging(unittest.TestCase):
    def setUp(self):
        self._saved_level = os.environ.get("HOOPS_AI_LOG_LEVEL")
        root = logging.getLogger()
        self._saved_root_level = root.level
        self._saved_handlers = list(root.handlers)

    def tearDown(self):
        if self._saved_level is None:
            os.environ.pop("HOOPS_AI_LOG_LEVEL", None)
        else:
            os.environ["HOOPS_AI_LOG_LEVEL"] = self._saved_level
        root = logging.getLogger()
        root.handlers[:] = self._saved_handlers
        root.setLevel(self._saved_root_level)

    def _apply(self, value):
        if value is None:
            os.environ.pop("HOOPS_AI_LOG_LEVEL", None)
        else:
            os.environ["HOOPS_AI_LOG_LEVEL"] = value
        main._configure_logging()
        return logging.getLogger()

    def test_defaults_to_info(self):
        self.assertEqual(self._apply(None).level, logging.INFO)

    def test_core_info_records_are_emitted_by_default(self):
        self._apply(None)
        self.assertTrue(logging.getLogger("core").isEnabledFor(logging.INFO))

    def test_root_gets_a_handler(self):
        self.assertTrue(self._apply(None).handlers)

    def test_explicit_level_is_honoured(self):
        self.assertEqual(self._apply("WARNING").level, logging.WARNING)

    def test_level_is_case_insensitive_and_trimmed(self):
        self.assertEqual(self._apply("  debug  ").level, logging.DEBUG)

    def test_unknown_level_falls_back_to_info(self):
        self.assertEqual(self._apply("NOT_A_LEVEL").level, logging.INFO)

    def test_empty_level_falls_back_to_info(self):
        self.assertEqual(self._apply("").level, logging.INFO)

    def test_numeric_level_string_is_rejected_rather_than_crashing(self):
        # logging.getLevelName("10") is not an int, so this takes the fallback.
        self.assertEqual(self._apply("10").level, logging.INFO)

    def test_repeated_calls_do_not_stack_handlers(self):
        first = len(self._apply(None).handlers)
        self.assertEqual(len(self._apply(None).handlers), first)


if __name__ == "__main__":
    unittest.main()
