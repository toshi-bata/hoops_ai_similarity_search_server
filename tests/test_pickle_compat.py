"""Regression tests for the cross-OS index-sidecar unpickling shim.

An index sidecar pickled on Windows stores ``WindowsPath`` objects (and a
Linux-authored one stores ``PosixPath``). The class that is foreign to the
running OS cannot be instantiated there, so unpickling it raises
"cannot instantiate 'WindowsPath'/'PosixPath' on your system". The shim must
temporarily alias that foreign class to the concrete local one for the duration
of the load, then restore it.
"""

import os
import pathlib

import core
import index_registry


FOREIGN_ATTR = "PosixPath" if os.name == "nt" else "WindowsPath"


def test_foreign_path_is_instantiable_during_load():
    """The foreign path class must construct without error while loading."""
    original = getattr(pathlib, FOREIGN_ATTR)
    captured = {}

    class FakeSearcher:
        def load_shape_index(self, path=None):
            foreign_cls = getattr(pathlib, FOREIGN_ATTR)
            # Would raise NotImplementedError with the pre-fix shim on POSIX.
            captured["reconstructed"] = foreign_cls("some", "parts")
            captured["path"] = path
            return "loaded"

    result = index_registry._load_shape_index_pickle_compat(
        FakeSearcher(), pathlib.Path("index.faiss")
    )

    assert result == "loaded"
    assert captured["reconstructed"] is not None
    assert captured["path"] == str(pathlib.Path("index.faiss"))


def test_foreign_path_class_is_restored_afterwards():
    """The global alias must be reverted once the load returns."""
    original = getattr(pathlib, FOREIGN_ATTR)

    class FakeSearcher:
        def load_shape_index(self, path=None):
            return "loaded"

    index_registry._load_shape_index_pickle_compat(FakeSearcher(), pathlib.Path("index.faiss"))

    assert getattr(pathlib, FOREIGN_ATTR) is original


def test_foreign_path_class_is_restored_on_error():
    """The alias must be reverted even when the load raises."""
    original = getattr(pathlib, FOREIGN_ATTR)

    class FakeSearcher:
        def load_shape_index(self, path=None):
            raise RuntimeError("boom")

    try:
        index_registry._load_shape_index_pickle_compat(
            FakeSearcher(), pathlib.Path("index.faiss")
        )
    except RuntimeError:
        pass

    assert getattr(pathlib, FOREIGN_ATTR) is original
