"""Activate the HOOPS AI license at startup.

``init_hoops_license`` prefers a compiled ``hoops_license_shim`` (built by
``tools/build_license_shim.py``) that restores its obfuscated key internally and
calls ``hoops_ai.set_license`` itself, so the key value never becomes a Python
string, config entry, log line or environment variable in this process. When the
shim is absent it falls back to the plaintext ``HOOPS_AI_LICENSE`` in ``.env``
(development only); when the shim is present but not importable (an ABI/Python
mismatch) it stops with a message naming the expected and running versions.

Dependency direction. This module depends only on the standard library and
``config`` (ENV_FILE_PATH, read_env_file, load_env_file, get_required_file_env,
EnvConfigError). It must never import ``core``: the dependency runs one way,
``core`` -> ``licensing`` -> ``config``.

Why the SDK imports stay function-local. ``hoops_license_shim`` and ``hoops_ai``
are imported inside ``init_hoops_license`` (and the shim is set to ``None`` on
ImportError so the fallback path can test it), so importing this module never
loads the license shim or the SDK -- and never materialises the key -- until the
license is actually being activated.

Reference convention (see "The re-export convention" in docs/architecture.md).
``_find_license_shim_file`` is the one (A) name: tests monkeypatch
``licensing._find_license_shim_file`` to
simulate a present / absent / ABI-mismatched shim. It is therefore NOT re-exported
by ``core``; re-exporting would bind a second object under ``core`` while
``init_hoops_license`` keeps calling the module's own, so a test swap and the live
path would diverge -- the double-truth trap. ``_shim_expected_python`` and
``init_hoops_license`` are stable and are re-exported from ``core`` by name
(``main.py`` calls ``core.init_hoops_license``).
"""


import logging
import os
import pathlib
import re
import sys
from typing import Optional

import config
from config import EnvConfigError


logger = logging.getLogger(__name__)


def _find_license_shim_file() -> Optional[pathlib.Path]:
    """Return the on-disk path of a compiled license shim, or ``None``.

    Distinguishes a redistributed package that ships the shim (file present)
    from a developer checkout that does not (file absent). This lets the caller
    tell an ABI/version mismatch -- shim present but not importable -- apart from
    the development fallback. The scan looks in the application directory and in
    the bundled site-packages that ``main.py`` prepends to ``sys.path``.
    """
    app_root = pathlib.Path(__file__).resolve().parent
    search_dirs = [app_root]
    if os.name == "nt":
        search_dirs.append(app_root / ".venv" / "Lib" / "site-packages")
    else:
        venv_lib = app_root / ".venv" / "lib"
        if venv_lib.is_dir():
            search_dirs.extend(venv_lib.glob("python3.*/site-packages"))
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for pattern in (
            "hoops_license_shim*.pyd",
            "hoops_license_shim*.so",
            "hoops_license_shim.py",
        ):
            match = next(iter(directory.glob(pattern)), None)
            if match is not None:
                return match
    return None


def _shim_expected_python(shim_path: pathlib.Path) -> Optional[str]:
    """Return the Python version a compiled shim was built for (e.g. ``3.12``).

    Parses the CPython ABI tag from the filename (``...cp312...`` -> ``3.12``).
    Returns ``None`` for a pure-Python shim or an untagged file.
    """
    match = re.search(r"cp3(\d+)", shim_path.name)
    return f"3.{match.group(1)}" if match else None


def init_hoops_license() -> None:
    """Activate the HOOPS AI license.

    Preferred path: a compiled ``hoops_license_shim`` module (built by
    ``tools/build_license_shim.py``) restores its obfuscated key internally and
    calls ``hoops_ai.set_license`` itself, so the key value never becomes a
    Python string in this process's own code, a config file, a log line, or the
    process environment.

    Development fallback: when that compiled module is absent, read the key from
    the ``.env`` file only (never from ``os.environ``) and set it here. This
    fallback logs a warning because a distributed deployment should always ship
    the compiled shim, not a plaintext key.

    A third case is handled explicitly for redistributed packages: the shim file
    is present but cannot be imported. That means the interpreter's ABI does not
    match the one the shim was compiled for (typically a Python version other
    than the one the package was built with). Falling back to ``.env`` there
    would surface a misleading "no license" error, so instead we stop with a
    message that names both the expected and the running Python version.
    """
    try:
        import hoops_license_shim
    except ImportError as exc:
        shim_file = _find_license_shim_file()
        if shim_file is not None:
            actual = f"{sys.version_info.major}.{sys.version_info.minor}"
            expected = _shim_expected_python(shim_file)
            expected_str = expected or "the version it was built for"
            msg = (
                "[LICENSE] The compiled license shim '%s' is present but could "
                "not be imported under Python %s; it was built for Python %s. "
                "Run this package with the matching Python version. "
                "Import error: %s" % (shim_file.name, actual, expected_str, exc)
            )
            logger.error(msg)
            raise EnvConfigError(msg)
        hoops_license_shim = None

    if hoops_license_shim is not None:
        if not hoops_license_shim.activate():
            msg = (
                "[LICENSE] The compiled license shim failed to activate the "
                "HOOPS AI license."
            )
            logger.error(msg)
            raise EnvConfigError(msg)
        # With the shim active, a plaintext key in .env is unused. Warn so an
        # ISV who accidentally ships .env alongside the shim can remove it.
        if config.read_env_file().get("HOOPS_AI_LICENSE"):
            logger.warning(
                "[LICENSE] The compiled license shim is active but %s still "
                "defines HOOPS_AI_LICENSE. That plaintext key is unused and "
                "should be removed from a distributed deployment.",
                config.ENV_FILE_PATH.name,
            )
        return

    logger.warning(
        "[LICENSE] No compiled license shim found; falling back to the "
        "plaintext HOOPS_AI_LICENSE in %s. This fallback is intended for "
        "development only and must not be used in a distributed deployment.",
        config.ENV_FILE_PATH.name,
    )
    import hoops_ai

    config.load_env_file()
    license_key = config.get_required_file_env("HOOPS_AI_LICENSE")
    hoops_ai.set_license(license_key, validate=True, silent=True)
