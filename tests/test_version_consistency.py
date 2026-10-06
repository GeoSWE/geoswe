"""The version string lives in five files; a release where they disagree is a release bug.

`twine check` does not compare them, and the publish workflow builds from whatever is in the
tree, so nothing else catches a half-finished bump.
"""
import re
from pathlib import Path

import geoswe

ROOT = Path(__file__).resolve().parents[1]


def _search(relpath, pattern):
    text = (ROOT / relpath).read_text()
    m = re.search(pattern, text, re.M)
    assert m, f"no version found in {relpath} with {pattern!r}"
    return m.group(1)


def test_every_version_string_agrees():
    found = {
        "geoswe.__version__": geoswe.__version__,
        "pyproject.toml": _search("pyproject.toml", r'^version\s*=\s*"([^"]+)"'),
        "CITATION.cff": _search("CITATION.cff", r'^version:\s*"([^"]+)"'),
        "docs/conf.py": _search("docs/conf.py", r'^release\s*=\s*"([^"]+)"'),
        "docs/citing.md": _search("docs/citing.md", r"version = \{([^}]+)\}"),
    }
    assert len(set(found.values())) == 1, f"version strings disagree: {found}"
