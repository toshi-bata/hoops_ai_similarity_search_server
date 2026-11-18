"""Tests for mcp_server/server.py. HTTP is mocked; no server is started."""

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

SERVER_PY = Path(__file__).resolve().parents[1] / "server.py"


@pytest.fixture()
def srv(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("hoops_sim_mcp_server", SERVER_PY)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    mod.CONFIG = mod.Config(
        work_dir=tmp_path / "work", logs_dir=tmp_path / "logs",
        state_dir=tmp_path / "state", repo=tmp_path)
    (tmp_path / "main.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(mod, "_sleep", lambda s: None)
    return mod


def _transport(handler):
    return httpx.MockTransport(handler)


class FakeProc:
    pid = 4242
    returncode = None

    def poll(self):
        return None


# -- ensure_server -----------------------------------------------------------


def test_ensure_server_alive_does_not_launch(srv):
    srv._transport = _transport(lambda r: httpx.Response(200, json=[]))
    srv._popen = lambda *a, **k: pytest.fail("must not launch")
    srv.ensure_server()


def test_ensure_server_launches_main_py_directly(srv):
    calls = {"launched": None, "n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["launched"] is None:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json=[])

    def popen(cmd, **kw):
        calls["launched"] = (cmd, kw)
        return FakeProc()

    srv._transport = _transport(handler)
    srv._popen = popen
    srv.CONFIG.python = "venv-python"
    srv.CONFIG.url = "http://127.0.0.1:8123"
    srv.ensure_server()

    cmd, kw = calls["launched"]
    assert cmd[0] == "venv-python"
    assert Path(cmd[1]).name == "main.py"
    assert cmd[2:] == ["--host", "127.0.0.1", "--port", "8123"]
    assert kw["cwd"] == str(srv.CONFIG.repo)
    assert ("creationflags" in kw) == (sys.platform == "win32")
    assert kw.get("start_new_session") is (sys.platform != "win32" or None)
    state = json.loads(srv.CONFIG.state_file.read_text(encoding="utf-8"))
    assert state["pid"] == 4242
    assert Path(state["log"]).parent == srv.CONFIG.logs_dir


def test_ensure_server_autostart_off(srv):
    srv._transport = _transport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("x")))
    srv.CONFIG.python = "venv-python"
    srv.CONFIG.autostart = False
    srv._popen = lambda *a, **k: pytest.fail("must not launch")
    with pytest.raises(srv.SimError, match="AUTOSTART"):
        srv.ensure_server()


def test_ensure_server_without_python_says_not_running(srv):
    srv._transport = _transport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("x")))
    with pytest.raises(srv.SimError, match="not running"):
        srv.ensure_server()


def test_stop_server_ignores_servers_it_did_not_start(srv):
    srv._transport = _transport(lambda r: httpx.Response(200, json=[]))
    out = srv.stop_server()
    assert out["stopped"] is False


# -- make_transformed_copy ---------------------------------------------------

STEP = (
    "#1=CARTESIAN_POINT('',(1.,2.,3.));\n"
    "#2=CARTESIAN_POINT('',(0.5,0.25));\n"
    "#3=DIRECTION('',(1.,0.,0.));\n"
)


def test_translate_moves_3d_points_only(srv, tmp_path):
    src = tmp_path / "a.stp"
    src.write_text(STEP, encoding="latin-1")
    out = srv.make_transformed_copy(str(src), translate=[10, 20, 30])
    text = Path(out["path"]).read_text(encoding="latin-1")
    assert "( 11.0, 22.0, 33.0 )" in text
    assert "(0.5,0.25)" in text
    assert "( 1.0, 0.0, 0.0 )" in text
    assert Path(out["path"]).parent == srv.CONFIG.work_dir


def test_rotate_turns_direction_and_point(srv, tmp_path):
    src = tmp_path / "a.step"
    src.write_text(STEP, encoding="latin-1")
    out = srv.make_transformed_copy(str(src), rotate_deg=90, axis="z")
    text = Path(out["path"]).read_text(encoding="latin-1")
    new, *_ = srv.transform_step_text(STEP, rotate_deg=90, axis="z")
    assert text == new
    assert "(0.5,0.25)" in text
    # DIRECTION (1,0,0) becomes (~0, 1, 0)
    line = [l for l in text.splitlines() if "DIRECTION" in l][0]
    vals = [float(v) for v in line.split("(", 2)[2].split(")")[0].split(",")]
    assert vals[0] == pytest.approx(0.0, abs=1e-12)
    assert vals[1] == pytest.approx(1.0)


def test_only_step_files_are_accepted(srv, tmp_path):
    src = tmp_path / "a.igs"
    src.write_text(STEP, encoding="latin-1")
    assert "error" in srv.make_transformed_copy(str(src))


# -- output limits -----------------------------------------------------------


def test_compare_rejects_more_than_twenty_files(srv):
    out = srv.compare_parts([f"f{i}.stp" for i in range(21)])
    assert "error" in out


def test_compare_pairs_are_sorted_and_capped(srv):
    n = 20
    files = [{"index": i, "filename": f"f{i}.stp"} for i in range(n)]
    pairs = [{"a": a, "b": b, "score": 0.5 + (a * n + b) / 10000}
             for a in range(n) for b in range(a + 1, n)]

    def handler(request):
        if request.url.path == "/similarity/compare":
            return httpx.Response(200, json={"files": files, "pairs": pairs})
        if request.url.path == "/files/upload-from-path":
            return httpx.Response(200, json={"files": [{"file_id": "x", "filename": "f"}]})
        return httpx.Response(200, json=[])

    srv._transport = _transport(handler)
    out = srv.compare_parts([f"f{i}.stp" for i in range(n)])
    assert len(out["pairs"]) == srv.MAX_PAIRS
    assert out["total_pairs"] == 190 and out["truncated"] is True
    scores = [p["score"] for p in out["pairs"]]
    assert scores == sorted(scores, reverse=True)


def test_list_parts_limit_is_capped(srv):
    seen = {}

    def handler(request):
        if request.url.path.endswith("/parts"):
            seen["limit"] = request.url.params["limit"]
            return httpx.Response(200, json={"total": 0, "offset": 0, "limit": 100, "items": []})
        return httpx.Response(200, json=[])

    srv._transport = _transport(handler)
    srv.list_parts("idx", limit=100000)
    assert seen["limit"] == str(srv.MAX_LIST)


def test_http_error_is_summarised(srv):
    def handler(request):
        if request.url.path == "/similarity/index/list":
            return httpx.Response(200, json=[])
        return httpx.Response(404, json={"detail": "Index 'nope' not found"})

    srv._transport = _transport(handler)
    out = srv.index_stats("nope")
    assert "HTTP 404" in out["error"] and "not found" in out["error"]


def test_model_consistency_trims_checkpoint_and_reports_sdk(srv):
    body = {
        "name": "idx", "current_ckpt_sha256": "a" * 64, "total_files": 3,
        "matching_files": 3, "mismatching_files": 0, "unknown_files": 0,
        "status": "match", "current_hoops_ai_version": "1.2.0",
        "sdk_matching_files": 1, "sdk_mismatching_files": 1,
        "sdk_unknown_files": 1, "sdk_status": "partial_mismatch",
    }

    def handler(request):
        if request.url.path == "/similarity/index/list":
            return httpx.Response(200, json=[])
        assert request.url.path == "/similarity/index/idx/model-consistency"
        return httpx.Response(200, json=body)

    srv._transport = _transport(handler)
    out = srv.model_consistency("idx")
    assert out["current_ckpt_sha256"] == "a" * 12
    assert out["status"] == "match" and out["sdk_status"] == "partial_mismatch"
    assert out["current_hoops_ai_version"] == "1.2.0"
    assert out["sdk_mismatching_files"] == 1 and out["sdk_unknown_files"] == 1
