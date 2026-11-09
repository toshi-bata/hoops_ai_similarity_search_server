"""Opt this process out of Windows' power throttling (EcoQoS).

Windows 11 puts a process whose windows are not visible into a power-saving
quality-of-service class. On a hybrid CPU that class parks the process on the
efficiency cores: the server's console window sits behind the browser during
normal use, so registration runs slower in exactly the situation the app is
designed for.

Measured on an i9-13900H (6 P-cores + 8 E-cores, 14 cores / 20 logical
processors) over an identical 500-file registration, 14 workers, a 120 s time
limit, calling index_registry.add_to_index() directly:

    console visible                                371.1 s
    console hidden behind the browser              634.1 s
    console hidden, parent process opts out        601.4 s
    console hidden, parent AND every worker        386.1 s

Two conclusions follow, and both shape how this module is used:

* The opt-out is NOT inherited by spawned child processes. A parent-only
  declaration recovers almost nothing (634 -> 601 s); every embedding worker
  has to declare it for itself (-> 386 s). On Windows, multiprocessing has
  only the spawn start method, and a spawned child re-imports the parent's
  main script as ``__mp_main__`` -- so the call must sit at module level in
  the server's main script to reach the workers at all.
* With every worker opted out, the native child processes those workers start
  (converter.exe) need no treatment of their own, and no OS-level setting has
  to be changed.

The declaration is made with SetProcessInformation(ProcessPowerThrottling).
``StateMask = 0`` with ``ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED``
means "this process is always managed as HighQoS" -- it is the documented way
to say *do not throttle me*, as distinct from leaving the mask unset (let the
system decide) or setting the state bit (throttle me always).
"""
import sys

# Information class and flags from processthreadsapi.h / winnt.h. Spelled out
# here rather than looked up, because ctypes has no header to read them from.
_PROCESS_INFORMATION_CLASS_POWER_THROTTLING = 4
_PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
_PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
# PROCESS_POWER_THROTTLING_STATE is three ULONGs: Version, ControlMask,
# StateMask.
_PROCESS_POWER_THROTTLING_STATE_SIZE = 12


def opt_out_of_power_throttling() -> bool:
    """Ask Windows never to throttle this process's execution speed.

    Safe to call from any process, including a spawned multiprocessing worker,
    and safe to call more than once.

    Returns True when the declaration was accepted, and False on any other
    outcome -- a non-Windows platform, a ctypes import failure, a kernel32 that
    cannot be loaded, or an API that refuses. No exception ever leaves this
    function: a server that cannot make this request must still start, and a
    worker that cannot make it must still embed.
    """
    if sys.platform != "win32":
        return False
    try:
        # Imported here, not at module scope, so this module stays importable
        # on a platform where ctypes.wintypes does not exist.
        import ctypes
        from ctypes import wintypes

        class _PowerThrottlingState(ctypes.Structure):
            _fields_ = [
                ("Version", wintypes.ULONG),
                ("ControlMask", wintypes.ULONG),
                ("StateMask", wintypes.ULONG),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        state = _PowerThrottlingState(
            Version=_PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask=_PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            StateMask=0,
        )
        ok = kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(),
            _PROCESS_INFORMATION_CLASS_POWER_THROTTLING,
            # pointer(), not byref(): both pass the same address to the API,
            # but a real pointer object can be read back by the tests that
            # check what this process actually asked for.
            ctypes.pointer(state),
            _PROCESS_POWER_THROTTLING_STATE_SIZE,
        )
        return bool(ok)
    except Exception:
        return False
