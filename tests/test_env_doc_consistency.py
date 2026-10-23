"""Guard: the user-facing env vars in .env.example and the READMEs must match.

Only a small, curated set of environment variables (groups A and B) is meant to
appear in .env.example and in the README variable tables; every other knob is
catalogued in config.py instead. If someone documents a variable in one place
but forgets the others, or moves an internal knob into a user-facing table, this
test fails and names the offending variable. That keeps the two entry points a
new user reads -- .env.example and README -- from drifting apart.
"""

import os
import re
import pathlib
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# A table row whose first cell is a `HOOPS_..._` code span, e.g.
#   | `HOOPS_AI_DATA_DIR` | optional | ... |
_README_VAR = re.compile(r"^\|\s*`(HOOPS_[A-Z_]+)`\s*\|", re.MULTILINE)

# An assignment line in .env.example, commented (optional) or not (required):
#   HOOPS_AI_LICENSE=...
#   #HOOPS_AI_WARMUP_INDEXES=...
_ENV_VAR = re.compile(r"^#?\s*(HOOPS_[A-Z_]+)\s*=", re.MULTILINE)


def _read(name: str) -> str:
    return (_REPO_ROOT / name).read_text(encoding="utf-8")


def _env_example_vars() -> set:
    return set(_ENV_VAR.findall(_read(".env.example")))


def _readme_table_vars(name: str) -> set:
    return set(_README_VAR.findall(_read(name)))


class EnvDocConsistency(unittest.TestCase):
    def test_env_example_matches_readme_en(self):
        env = _env_example_vars()
        readme = _readme_table_vars("README.md")
        self.assertEqual(
            env,
            readme,
            "Variables documented in .env.example and README.md differ.\n"
            f"  only in .env.example: {sorted(env - readme)}\n"
            f"  only in README.md:    {sorted(readme - env)}",
        )

    def test_env_example_matches_readme_ja(self):
        env = _env_example_vars()
        readme = _readme_table_vars("README_ja.md")
        self.assertEqual(
            env,
            readme,
            "Variables documented in .env.example and README_ja.md differ.\n"
            f"  only in .env.example: {sorted(env - readme)}\n"
            f"  only in README_ja.md: {sorted(readme - env)}",
        )

    def test_both_readmes_document_the_same_vars(self):
        en = _readme_table_vars("README.md")
        ja = _readme_table_vars("README_ja.md")
        self.assertEqual(
            en,
            ja,
            "README.md and README_ja.md document different variable sets.\n"
            f"  only in README.md:    {sorted(en - ja)}\n"
            f"  only in README_ja.md: {sorted(ja - en)}",
        )


if __name__ == "__main__":
    unittest.main()
