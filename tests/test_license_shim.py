"""SDK-free unit tests for the HOOPS AI license shim path.

These tests exercise core.init_hoops_license() and the shim source rendered by
tools/build_license_shim.py without the real HOOPS AI SDK or a real license.
Both hoops_ai and the compiled shim are replaced with fakes, and only dummy key
values are used. The tests assert, above all, that no key value ever reaches a
log record or an exception message on any path.
"""

import importlib.util
import logging
import os
import pathlib
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core
import licensing
import config

SHIM_KEY = "SHIM-DUMMY-0000-0000"
ENV_KEY = "ENVV-DUMMY-1111-1111"


def _fake_hoops_ai(calls):
    module = types.ModuleType("hoops_ai")

    def set_license(license_key, validate=True, silent=False):
        calls.append((license_key, validate, silent))

    module.set_license = set_license
    return module


def _fake_shim(activate_result, calls):
    module = types.ModuleType("hoops_license_shim")

    def activate():
        if activate_result:
            import hoops_ai

            hoops_ai.set_license(SHIM_KEY, validate=True, silent=True)
        return activate_result

    module.activate = activate
    return module


def _assert_no_key_in_logs(caplog):
    for record in caplog.records:
        message = record.getMessage()
        assert SHIM_KEY not in message
        assert ENV_KEY not in message
        for arg in record.args or ():
            assert SHIM_KEY != arg and ENV_KEY != arg


def test_shim_present_does_not_read_env_key(monkeypatch, caplog):
    calls = []
    monkeypatch.setitem(sys.modules, "hoops_ai", _fake_hoops_ai(calls))
    monkeypatch.setitem(sys.modules, "hoops_license_shim", _fake_shim(True, calls))
    monkeypatch.setattr(config, "read_env_file", lambda *a, **k: {})

    def _fail(*a, **k):
        raise AssertionError("get_required_file_env must not be called with the shim active")

    monkeypatch.setattr(config, "get_required_file_env", _fail)

    with caplog.at_level(logging.WARNING, logger="licensing"):
        core.init_hoops_license()

    assert calls == [(SHIM_KEY, True, True)]
    assert not any("falling back" in r.getMessage() for r in caplog.records)
    _assert_no_key_in_logs(caplog)


def test_shim_present_warns_about_stray_env_key(monkeypatch, caplog):
    calls = []
    monkeypatch.setitem(sys.modules, "hoops_ai", _fake_hoops_ai(calls))
    monkeypatch.setitem(sys.modules, "hoops_license_shim", _fake_shim(True, calls))
    monkeypatch.setattr(config, "read_env_file", lambda *a, **k: {"HOOPS_AI_LICENSE": ENV_KEY})

    with caplog.at_level(logging.WARNING, logger="licensing"):
        core.init_hoops_license()

    assert calls == [(SHIM_KEY, True, True)]
    assert any("unused" in r.getMessage() for r in caplog.records)
    _assert_no_key_in_logs(caplog)


def test_fallback_reads_env_and_warns(monkeypatch, caplog):
    calls = []
    monkeypatch.setitem(sys.modules, "hoops_ai", _fake_hoops_ai(calls))
    monkeypatch.setitem(sys.modules, "hoops_license_shim", None)
    # Development fallback only applies when no shim file exists on disk; a shim
    # that is present but unimportable is a version mismatch, handled elsewhere.
    monkeypatch.setattr(licensing, "_find_license_shim_file", lambda: None)
    monkeypatch.setattr(config, "load_env_file", lambda *a, **k: None)
    monkeypatch.setattr(config, "get_required_file_env", lambda name: ENV_KEY)

    with caplog.at_level(logging.WARNING, logger="licensing"):
        core.init_hoops_license()

    assert calls == [(ENV_KEY, True, True)]
    assert any("falling back" in r.getMessage() for r in caplog.records)
    _assert_no_key_in_logs(caplog)


def test_shim_present_but_unimportable_raises_without_key(monkeypatch, caplog):
    # A compiled shim on disk that cannot be imported (ABI/version mismatch)
    # must stop the line naming the Python versions -- never fall back to .env,
    # and never leak a key.
    calls = []
    monkeypatch.setitem(sys.modules, "hoops_ai", _fake_hoops_ai(calls))
    monkeypatch.setitem(sys.modules, "hoops_license_shim", None)
    monkeypatch.setattr(
        licensing,
        "_find_license_shim_file",
        lambda: pathlib.Path("hoops_license_shim.cp312-win_amd64.pyd"),
    )

    def _fail(*a, **k):
        raise AssertionError("must not fall back to .env on a version mismatch")

    monkeypatch.setattr(config, "get_required_file_env", _fail)

    with caplog.at_level(logging.ERROR, logger="licensing"):
        with pytest.raises(core.EnvConfigError) as excinfo:
            core.init_hoops_license()

    message = str(excinfo.value)
    assert "3.12" in message
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in message
    assert SHIM_KEY not in message and ENV_KEY not in message
    assert calls == []
    _assert_no_key_in_logs(caplog)


def test_shim_activate_failure_raises_without_key(monkeypatch, caplog):
    calls = []
    monkeypatch.setitem(sys.modules, "hoops_ai", _fake_hoops_ai(calls))
    monkeypatch.setitem(sys.modules, "hoops_license_shim", _fake_shim(False, calls))
    monkeypatch.setattr(config, "read_env_file", lambda *a, **k: {})

    with caplog.at_level(logging.ERROR, logger="licensing"):
        with pytest.raises(core.EnvConfigError) as excinfo:
            core.init_hoops_license()

    assert SHIM_KEY not in str(excinfo.value)
    assert ENV_KEY not in str(excinfo.value)
    assert calls == []
    _assert_no_key_in_logs(caplog)


def _load_build_module():
    path = pathlib.Path(__file__).resolve().parent.parent / "tools" / "build_license_shim.py"
    spec = importlib.util.spec_from_file_location("build_license_shim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rendered_shim_exposes_only_activate_and_hides_key(monkeypatch):
    build = _load_build_module()
    dummy = "RENDER-DUMMY-2222-2222"
    source = build.render_shim_source(dummy)

    # strings requirement: the plaintext key must not survive in the source.
    assert dummy not in source

    calls = []
    monkeypatch.setitem(sys.modules, "hoops_ai", _fake_hoops_ai(calls))
    namespace: dict = {}
    exec(compile(source, "<shim>", "exec"), namespace)

    public = [name for name in namespace if not name.startswith("__")]
    assert public == ["activate"]

    assert namespace["activate"]() is True
    assert calls == [(dummy, True, True)]
