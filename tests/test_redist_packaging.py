"""Guard: the redistributable package must contain every server module.

When core.py was split into per-domain modules, the two packaging scripts still
named five files by hand, so 17 modules stopped shipping and the package could
not start. The scripts now COLLECT ``*.py`` from the repository root and subtract
an explicit exclusion list; this test reads those exclusion lists out of the
script text and checks that what is left still covers the whole server.

The scripts are parsed, not executed: running either one needs a HOOPS AI
install, a compiled license shim, a built web client and several GB of disk, and
none of that says anything about which files the script *selects*. Parsing keeps
the check cheap enough to run on every commit, and it fails loudly if the
selection logic is rewritten into a shape this test no longer recognises.
"""

import ast
import pathlib
import re
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PS1 = _REPO_ROOT / "tools" / "build_redist_package.ps1"
_SH = _REPO_ROOT / "tools" / "build_redist_package.sh"

# The Nuitka input for the compiled license shim. It holds the key in plain
# text, so it must never be packaged, whatever else changes.
_MUST_STAY_EXCLUDED = "hoops_license_shim.py"


def _array_items(text: str, name: str) -> set:
    """Read the quoted items of ``$Name = @( ... )`` / ``NAME=( ... )``.

    One parser for both scripts: the two array syntaxes differ only in the
    sigil and the separator, and neither matters once the quoted strings are
    pulled out. Comment lines inside the array are skipped so the reasons can
    live next to the entries they explain.
    """
    match = re.search(
        r"(?:\$%s\s*=\s*@\(|%s=\()(.*?)\n\)" % (re.escape(name), re.escape(name)),
        text,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError(
            f"Could not find the array '{name}' in the packaging script. If the "
            "selection logic was rewritten, update this test to match -- do not "
            "delete it."
        )
    body = "\n".join(
        line for line in match.group(1).splitlines() if not line.strip().startswith("#")
    )
    return set(re.findall(r'"([^"]+)"', body))


def _exclusions(path: pathlib.Path) -> tuple:
    text = path.read_text(encoding="utf-8")
    names = _array_items(text, "ExcludedServerPy" if path.suffix == ".ps1" else "EXCLUDED_SERVER_PY")
    patterns = _array_items(
        text,
        "ExcludedServerPyPatterns" if path.suffix == ".ps1" else "EXCLUDED_SERVER_PY_PATTERNS",
    )
    return names, patterns


def _root_modules() -> set:
    """Every *.py at the repository root that is part of the checkout."""
    return {p.name for p in _REPO_ROOT.glob("*.py")}


def _packaged_modules(path: pathlib.Path) -> set:
    """The files a script's collect-minus-exclusions rule would copy."""
    names, patterns = _exclusions(path)
    kept = set()
    for candidate in _root_modules():
        if candidate in names:
            continue
        if any(pathlib.PurePath(candidate).match(pattern) for pattern in patterns):
            continue
        kept.add(candidate)
    return kept


def _imported_root_modules() -> set:
    """Root modules reachable from main.py by following root-level imports.

    A static walk, so nothing has to be imported: the server's dependencies
    (hoops_ai, torch, faiss) are not installed in the test environment. Only
    names that exist as a *.py at the root are followed; everything else is a
    third-party or stdlib import and is ignored.
    """
    root_names = {name[:-3] for name in _root_modules()}
    seen = set()
    queue = ["main"]
    # routers/ is copied wholesale by both scripts, but its imports reach back
    # into the root modules, so they have to be part of the closure too.
    queue += [f"routers/{p.stem}" for p in (_REPO_ROOT / "routers").glob("*.py")]
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        # utf-8-sig: some modules carry a BOM, which ast.parse rejects.
        source = (_REPO_ROOT / f"{current}.py").read_text(encoding="utf-8-sig")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                found = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                found = [node.module.split(".")[0]] if node.module else []
            else:
                continue
            for name in found:
                if name in root_names and name not in seen:
                    queue.append(name)
    return {f"{name}.py" for name in seen if "/" not in name}


class RedistPackaging(unittest.TestCase):
    def test_both_scripts_exclude_the_same_files(self):
        # A divergence would mean the Windows and Linux packages hold different
        # code, which is the kind of difference nobody notices until a customer
        # reports it on one platform only.
        self.assertEqual(
            _exclusions(_PS1),
            _exclusions(_SH),
            "build_redist_package.ps1 and .sh exclude different files.",
        )

    def test_every_module_the_server_imports_is_packaged(self):
        needed = _imported_root_modules()
        for script in (_PS1, _SH):
            with self.subTest(script=script.name):
                missing = needed - _packaged_modules(script)
                self.assertEqual(
                    missing,
                    set(),
                    f"{script.name} would not copy these server modules: "
                    f"{sorted(missing)}",
                )

    def test_the_license_shim_source_is_never_packaged(self):
        for script in (_PS1, _SH):
            with self.subTest(script=script.name):
                names, _ = _exclusions(script)
                self.assertIn(
                    _MUST_STAY_EXCLUDED,
                    names,
                    f"{script.name} must exclude {_MUST_STAY_EXCLUDED}: it is the "
                    "Nuitka input and carries the license key in plain text.",
                )

    def test_nothing_at_the_root_is_dropped_without_a_reason(self):
        # The exclusion lists are the only place a root module may be left out,
        # so anything else at the root is packaged by construction. This pins
        # the count so a rewrite that quietly narrows the selection is visible.
        packaged = _packaged_modules(_PS1)
        self.assertEqual(packaged, _root_modules() - _exclusions(_PS1)[0] - {
            name for name in _root_modules()
            if any(pathlib.PurePath(name).match(p) for p in _exclusions(_PS1)[1])
        })
        self.assertGreaterEqual(len(packaged), 20)


if __name__ == "__main__":
    unittest.main()
