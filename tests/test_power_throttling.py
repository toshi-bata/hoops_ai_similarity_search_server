"""The Windows power-throttling (EcoQoS) opt-out.

Two things are covered: what power_throttling.opt_out_of_power_throttling()
asks the OS for, and -- just as important -- WHERE main.py calls it. The
opt-out is not inherited by spawned processes, so it only reaches the embedding
workers because the call sits at module level in the server's main script and a
spawned worker re-imports that script as ``__mp_main__``. Moved inside a
function, or behind ``if __name__ == "__main__":``, the workers would stay
throttled and the measured win (386 s against 601 s) would be lost silently.
"""
import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import power_throttling  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = REPO_ROOT / "main.py"

OPT_OUT = "opt_out_of_power_throttling"


class _FakeEntryPoint:
    """A kernel32 export that accepts the ctypes signature declarations.

    A bound method cannot carry ``restype``/``argtypes``, and those
    declarations are the subject of half the tests here, so each export is its
    own attribute-bearing callable.
    """

    def __init__(self, body):
        self.body = body
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self.body(*args)


class _FakeKernel32:
    """Stands in for kernel32 and records the SetProcessInformation call."""

    def __init__(self, result=1, last_error=None):
        self.result = result
        self.last_error = last_error
        self.calls = []
        self.GetCurrentProcess = _FakeEntryPoint(self._get_current_process)
        self.SetProcessInformation = _FakeEntryPoint(self._set_process_information)

    @staticmethod
    def _get_current_process():
        return -1

    def _set_process_information(self, handle, info_class, info, size):
        self.calls.append((handle, info_class, info, size))
        if self.last_error is not None:
            # Real Win32 APIs leave the last error behind whether they succeed
            # or fail; the fake does the same so the module's handling of it
            # can be tested.
            import ctypes

            ctypes.set_last_error(self.last_error)
        return self.result


class OffWindowsNothingHappens(unittest.TestCase):
    def test_a_non_windows_platform_reports_no_opt_out(self):
        with patch.object(sys, "platform", "linux"):
            self.assertFalse(power_throttling.opt_out_of_power_throttling())

    def test_a_non_windows_platform_never_touches_ctypes(self):
        # Nothing to load and nothing to fail: the guard returns before any
        # Windows-only import is attempted.
        fake = _FakeKernel32()
        with patch.object(sys, "platform", "darwin"), patch(
            "ctypes.WinDLL", return_value=fake, create=True
        ):
            self.assertFalse(power_throttling.opt_out_of_power_throttling())
        self.assertEqual(fake.calls, [])


class TheRequestMadeOfWindows(unittest.TestCase):
    def setUp(self):
        try:
            import ctypes.wintypes  # noqa: F401
        except (ImportError, ValueError):
            self.skipTest("ctypes.wintypes is Windows-only")

    def _call(self, result=1, last_error=None):
        fake = _FakeKernel32(result, last_error)
        with patch.object(sys, "platform", "win32"), patch(
            "ctypes.WinDLL", return_value=fake, create=True
        ):
            returned = power_throttling.opt_out_of_power_throttling()
        return returned, fake

    def test_it_declares_the_signatures_before_calling(self):
        # Left undeclared, ctypes takes the return value of
        # GetCurrentProcess() for a 32-bit int and truncates the 64-bit pseudo
        # handle, and Windows refuses the call with ERROR_INVALID_HANDLE. That
        # shipped once: the call reported success nowhere and the workers
        # stayed on the efficiency cores.
        from ctypes import wintypes

        _, fake = self._call()
        self.assertIs(fake.GetCurrentProcess.restype, wintypes.HANDLE)
        self.assertIsNotNone(fake.SetProcessInformation.argtypes)
        self.assertIs(fake.SetProcessInformation.argtypes[0], wintypes.HANDLE)
        self.assertIs(fake.SetProcessInformation.restype, wintypes.BOOL)

    def test_it_asks_for_the_power_throttling_information_class(self):
        returned, fake = self._call()
        self.assertTrue(returned)
        self.assertEqual(len(fake.calls), 1)
        _, info_class, _, _ = fake.calls[0]
        self.assertEqual(info_class, 4)

    def test_it_sends_a_twelve_byte_structure(self):
        _, fake = self._call()
        _, _, _, size = fake.calls[0]
        self.assertEqual(size, 12)

    def test_the_state_says_never_throttle_this_process(self):
        _, fake = self._call()
        state = fake.calls[0][2].contents
        self.assertEqual(state.Version, 1)
        # EXECUTION_SPEED is the bit being controlled...
        self.assertEqual(state.ControlMask, 0x1)
        # ...and a cleared state bit is what means "always HighQoS". Setting it
        # would ask to be throttled permanently -- the opposite of the point.
        self.assertEqual(state.StateMask, 0)

    def test_a_refusal_is_reported_without_raising(self):
        returned, fake = self._call(result=0)
        self.assertFalse(returned)
        self.assertEqual(len(fake.calls), 1)

    def test_a_refusal_keeps_the_windows_error_code(self):
        returned, _ = self._call(result=0, last_error=6)
        self.assertFalse(returned)
        self.assertEqual(power_throttling.LAST_ERROR, 6)

    def test_success_does_not_pick_up_a_stale_error(self):
        # Win32 does not clear the last error on success, so reading it after
        # an accepted call reports whatever failed before it.
        returned, _ = self._call(result=1, last_error=6)
        self.assertTrue(returned)
        self.assertIsNone(power_throttling.LAST_ERROR)

    def test_a_kernel32_that_will_not_load_is_reported_without_raising(self):
        with patch.object(sys, "platform", "win32"), patch(
            "ctypes.WinDLL", side_effect=OSError("boom"), create=True
        ):
            self.assertFalse(power_throttling.opt_out_of_power_throttling())


class TheRealApiAcceptsIt(unittest.TestCase):
    """No mock: this asks the running Windows for real.

    The fakes check what is sent, not whether Windows accepts it -- which is
    exactly how a call that always failed passed its tests once. The subject
    is the pytest process itself and the declaration has no side effect worth
    undoing.
    """

    def test_windows_accepts_the_declaration(self):
        if sys.platform != "win32":
            self.skipTest("the real API only exists on Windows")
        self.assertTrue(
            power_throttling.opt_out_of_power_throttling(),
            f"Windows refused the declaration with error "
            f"{power_throttling.LAST_ERROR}",
        )


class MainCallsItWhereTheWorkersWillSeeIt(unittest.TestCase):
    """Parsed with ast, not matched with a regex.

    A previous source-inspection helper in this repository mistook a default
    argument's ``= {}`` for a function body and drew the wrong boundary. The
    syntax tree answers "is this statement at module level?" exactly.
    """

    def setUp(self):
        self.tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))

    @staticmethod
    def _calls_the_opt_out(node):
        """True when this statement itself makes the call.

        Only the statement's own expressions are examined; statements nested
        inside it are not, so an enclosing ``if``/``try``/``def`` is never
        credited with a call its body makes -- that nested statement is found
        on its own turn through the walk instead.
        """
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.stmt, ast.excepthandler)):
                continue
            for inner in ast.walk(child):
                if isinstance(inner, ast.Call):
                    func = inner.func
                    name = getattr(func, "attr", None) or getattr(func, "id", None)
                    if name == OPT_OUT:
                        return True
        return False

    def test_the_call_is_a_top_level_statement(self):
        at_top_level = [node for node in self.tree.body if self._calls_the_opt_out(node)]
        self.assertTrue(
            at_top_level,
            f"main.py never calls {OPT_OUT}() at module level; a spawned "
            "embedding worker re-imports main.py as __mp_main__ and would "
            "never reach it",
        )

    def test_the_call_is_nowhere_else(self):
        # Every statement that is not one of the module's own top-level
        # statements: function and class bodies, and the bodies of any
        # top-level if/try/with -- including `if __name__ == "__main__":`,
        # which a spawned worker does not enter.
        top_level = {id(node) for node in self.tree.body}
        buried = [
            type(node).__name__
            for node in ast.walk(self.tree)
            if isinstance(node, ast.stmt)
            and id(node) not in top_level
            and self._calls_the_opt_out(node)
        ]
        self.assertEqual(
            buried,
            [],
            f"main.py calls {OPT_OUT}() somewhere a spawned worker does not "
            "run; it must stay at module level",
        )


if __name__ == "__main__":
    unittest.main()
