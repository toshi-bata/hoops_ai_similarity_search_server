"""Build the compiled HOOPS AI license shim with Nuitka.

The shim carries the HOOPS AI license key as an OBFUSCATED literal (XOR with a
random per-build key, then Base85). This is obfuscation, NOT encryption: the
de-obfuscation key ships inside the same binary, so anyone with the binary can
recover the value. The only goal is to keep the key from appearing as a
readable string in configuration files, logs, the process environment, or the
``strings`` output of the artifact. It does NOT stop someone who replaces the
``hoops_ai.set_license`` API to capture the key at call time.

The key is read from the ``HOOPS_AI_LICENSE_BUILD`` environment variable, never
from a command-line argument (which would linger in shell history and the
process list).

The operator runs this script; it is never run by tooling automatically. Typical
use on Windows PowerShell::

    $env:HOOPS_AI_LICENSE_BUILD = '<your-key>'
    python tools/build_license_shim.py

Use ``--dry-run`` to print the Nuitka command without writing the intermediate
source or invoking the compiler.

The generated intermediate source and the Nuitka build/dist trees both embed the
obfuscated literal, so they are always removed in a ``finally`` block and must
never be committed (see .gitignore).
"""

from __future__ import annotations

import argparse
import base64
import os
import pathlib
import secrets
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE_NAME = "hoops_license_shim"
GENERATED_SOURCE = REPO_ROOT / f"{MODULE_NAME}.py"
BUILD_DIR = REPO_ROOT / f"{MODULE_NAME}.build"
DIST_DIR = REPO_ROOT / f"{MODULE_NAME}.dist"

# Template for the generated shim. Every secret-bearing name is a local inside
# activate(), so the compiled module exposes activate() and nothing else: there
# is no module attribute or helper that hands back the key. The literals below
# hold the OBFUSCATED bytes (XOR + Base85), not the plaintext key.
SHIM_TEMPLATE = '''"""Compiled HOOPS AI license shim (generated - DO NOT COMMIT).

This module carries the HOOPS AI license key as an OBFUSCATED byte string, not
an encrypted one: the de-obfuscation key lives in this same binary. Its only
purpose is to keep the key out of config files, logs, the process environment,
and the ``strings`` output of this artifact. It does NOT protect against someone
who swaps out ``hoops_ai.set_license`` to read the key when activate() calls it.

Public surface: activate() only.
"""


def activate() -> bool:
    """Restore the obfuscated key, hand it to hoops_ai.set_license, return success.

    The key is never returned, logged, or embedded in an exception. On any
    failure this returns False with no secret in scope for the caller.
    """
    import base64
    import hoops_ai

    xor_key = {xor_key!r}
    payload = {payload!r}
    try:
        scrambled = base64.b85decode(payload)
        key = bytes(
            b ^ xor_key[i % len(xor_key)] for i, b in enumerate(scrambled)
        ).decode("ascii")
        hoops_ai.set_license(key, validate=True, silent=True)
        return True
    except Exception:
        return False
'''


def _obfuscate(key: str) -> tuple[bytes, bytes]:
    """Return (base85 payload, xor key) for *key*.

    Obfuscation only: the returned xor key is what un-scrambles the payload and
    is embedded next to it in the shim. This defeats ``strings``, nothing more.
    """
    xor_key = secrets.token_bytes(16)
    raw = key.encode("ascii")
    scrambled = bytes(b ^ xor_key[i % len(xor_key)] for i, b in enumerate(raw))
    return base64.b85encode(scrambled), xor_key


def render_shim_source(key: str) -> str:
    """Render the shim's Python source with the obfuscated key baked in."""
    payload, xor_key = _obfuscate(key)
    return SHIM_TEMPLATE.format(payload=payload, xor_key=xor_key)


def _nuitka_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "nuitka",
        "--module",
        str(GENERATED_SOURCE),
        f"--output-dir={REPO_ROOT}",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Nuitka command without writing the intermediate source "
        "or invoking the compiler.",
    )
    args = parser.parse_args(argv)

    command = _nuitka_command()

    if args.dry_run:
        print(" ".join(command))
        print("(dry run: intermediate source not written, compiler not invoked)")
        return 0

    key = os.environ.get("HOOPS_AI_LICENSE_BUILD")
    if not key:
        print(
            "HOOPS_AI_LICENSE_BUILD is not set; export the license key into that "
            "environment variable before building.",
            file=sys.stderr,
        )
        return 2

    try:
        GENERATED_SOURCE.write_text(render_shim_source(key), encoding="utf-8")
        result = subprocess.run(command)
        return result.returncode
    finally:
        # The intermediate source and the Nuitka build/dist trees embed the
        # obfuscated literal (the build tree even holds generated C). Remove
        # them so nothing key-bearing lingers on the build machine.
        GENERATED_SOURCE.unlink(missing_ok=True)
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
        shutil.rmtree(DIST_DIR, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
