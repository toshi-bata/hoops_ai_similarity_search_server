"""Shared pytest configuration for the similarity search server test suite.

Named-index management and default model/index settings are
regular product endpoints served to every deployment, so there is no feature
flag to enable here.
"""

import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _isolated_tag_vocabulary():
    """Keep the tag vocabulary out of the developer's real data root.

    The vocabulary is the one tag sidecar that is NOT per index, so redirecting
    ``config.INDEXES_DIR`` -- which every tag test already does -- does not move
    it. Any test that assigns a tag records the name, and without this fixture
    that write lands in the live DATA_ROOT and fills a real file with "t0",
    "t1", ... Redirected for the whole session, so a test module does not have
    to know the vocabulary exists to stay out of real data.
    """
    original = config.TAG_VOCABULARY_FILE
    with tempfile.TemporaryDirectory() as tmp:
        config.TAG_VOCABULARY_FILE = pathlib.Path(tmp) / "tag_vocabulary.json"
        try:
            yield
        finally:
            config.TAG_VOCABULARY_FILE = original
