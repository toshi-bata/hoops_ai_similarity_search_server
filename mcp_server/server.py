"""MCP server that drives the HOOPS AI similarity search server.

Runs in its own light environment (``mcp`` and ``httpx`` only) and talks to the
similarity server over HTTP. It never imports HOOPS AI. When the similarity
server is not running it can start ``main.py`` itself (see ``ensure_server``).

Configuration comes from environment variables; see ``load_config``.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlparse

import httpx
from mcp.server.fastmcp import FastMCP, Image

HERE = Path(__file__).resolve().parent

# Output limits: every tool answers with a few KB at most.
MAX_FILES_COMPARE = 20
MAX_PAIRS = 100
MAX_LIST = 100
MAX_SEARCH_K = 50
MAX_IDS_PER_CALL = 500
MAX_REGISTER_FILES = 500
MAX_VARIANTS = 19
# Tool calls should return within about a minute; heavy work is a job.
DEFAULT_TIMEOUT = 50.0
PAGE = 2000
REPORT_CHARS = 6000

# Mirrors storage.CAD_ALLOWED_EXTENSIONS of the similarity server. It is only
# used to pick CAD files out of a folder; the server remains the authority.
CAD_EXTENSIONS = frozenset({
    ".3ds", ".3mf", ".3dxml", ".sat", ".sab", ".dgn", ".dwg", ".dxf", ".dwf",
    ".dwfx", ".model", ".dlv", ".exp", ".session", ".catpart", ".catproduct",
    ".catshape", ".catdrawing", ".cgr", ".dae", ".prt", ".neu", ".asm", ".xas",
    ".xpr", ".fbx", ".gltf", ".glb", ".arc", ".unv", ".mf1", ".pkg", ".ifc",
    ".ifczip", ".igs", ".iges", ".ipt", ".iam", ".jt", ".kmz", ".nwd", ".prc",
    ".x_t", ".xmt", ".x_b", ".xmt_txt", ".rvt", ".3dm", ".stp", ".step",
    ".stpz", ".stl", ".par", ".pwd", ".psm", ".sldprt", ".sldasm", ".sldfpp",
    ".u3d", ".vda", ".wrl", ".vrml", ".obj", ".xv3", ".xv0", ".hsf",
})

mcp = FastMCP("hoops-ai-similarity")


class SimError(RuntimeError):
    """A short, user-facing failure message."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    url: str = "http://127.0.0.1:8000"
    repo: Path = HERE.parent
    python: Optional[str] = None
    autostart: bool = True
    startup_timeout: float = 180.0
    work_dir: Path = HERE / "work"
    logs_dir: Path = HERE / "logs"
    state_dir: Path = HERE / ".state"

    @property
    def port(self) -> int:
        return urlparse(self.url).port or 8000

    @property
    def is_local(self) -> bool:
        return (urlparse(self.url).hostname or "") in ("127.0.0.1", "localhost", "::1")

    @property
    def state_file(self) -> Path:
        return self.state_dir / "server.json"


def load_config(env: Optional[dict] = None) -> Config:
    env = os.environ if env is None else env
    cfg = Config()
    cfg.url = (env.get("HOOPS_SIM_URL") or cfg.url).rstrip("/")
    if env.get("HOOPS_SIM_REPO"):
        cfg.repo = Path(env["HOOPS_SIM_REPO"])
    cfg.python = env.get("HOOPS_SIM_PYTHON") or None
    cfg.autostart = (env.get("HOOPS_SIM_AUTOSTART", "1").strip().lower()
                     not in ("0", "false", "no", "off"))
    try:
        cfg.startup_timeout = float(env.get("HOOPS_SIM_STARTUP_TIMEOUT") or 180)
    except ValueError:
        pass
    if env.get("HOOPS_SIM_WORK_DIR"):
        cfg.work_dir = Path(env["HOOPS_SIM_WORK_DIR"])
    return cfg


CONFIG = load_config()

# Test seams: a transport for httpx, the process launcher and the sleeper.
_transport: Optional[httpx.BaseTransport] = None
_popen = subprocess.Popen
_sleep = time.sleep


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _client(timeout: float) -> httpx.Client:
    return httpx.Client(base_url=CONFIG.url, timeout=timeout, transport=_transport)


def _detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        detail = body.get("detail", body) if isinstance(body, dict) else body
    except ValueError:
        detail = resp.text
    return str(detail)[:300]


def _http(method: str, path: str, *, timeout: float = DEFAULT_TIMEOUT, **kw) -> httpx.Response:
    """One request without the server check; raises SimError on any failure."""
    try:
        with _client(timeout) as c:
            resp = c.request(method, path, **kw)
    except httpx.TimeoutException as exc:
        raise SimError(f"Timed out calling {method} {path}") from exc
    except httpx.RequestError as exc:
        raise SimError(f"Network error calling {path}: {exc}") from exc
    if resp.status_code >= 400:
        raise SimError(f"HTTP {resp.status_code} from {path}: {_detail(resp)}")
    return resp


def api(method: str, path: str, *, timeout: float = DEFAULT_TIMEOUT, **kw) -> Any:
    """Check the server is up (starting it if allowed), then return the JSON."""
    ensure_server()
    return _http(method, path, timeout=timeout, **kw).json()


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


def _alive() -> bool:
    try:
        with _client(3.0) as c:
            return c.get("/similarity/index/list").status_code == 200
    except httpx.HTTPError:
        return False


def _read_state() -> Optional[dict]:
    try:
        return json.loads(CONFIG.state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _launch() -> tuple:
    """Start ``main.py`` directly (not through a wrapper) and record its PID.

    main.py opts itself and its embedding workers out of Windows EcoQoS at
    import time, so it has to be the process that is started.
    """
    main_py = CONFIG.repo / "main.py"
    if not main_py.is_file():
        raise SimError(f"main.py not found in HOOPS_SIM_REPO ({CONFIG.repo}).")
    CONFIG.logs_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.state_dir.mkdir(parents=True, exist_ok=True)
    log_path = CONFIG.logs_dir / f"server_{datetime.now():%Y%m%d_%H%M%S}.log"
    cmd = [CONFIG.python, str(main_py), "--host", "127.0.0.1", "--port", str(CONFIG.port)]
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True
    with open(log_path, "ab") as log:
        proc = _popen(cmd, cwd=str(CONFIG.repo), stdin=subprocess.DEVNULL,
                      stdout=log, stderr=subprocess.STDOUT, **kwargs)
    CONFIG.state_file.write_text(
        json.dumps({"pid": proc.pid, "url": CONFIG.url, "log": str(log_path),
                    "started_at": datetime.now().isoformat(timespec="seconds")}),
        encoding="utf-8")
    return proc, log_path


def ensure_server() -> None:
    """Return once the similarity server answers; start it when allowed."""
    if _alive():
        return
    if not CONFIG.autostart:
        raise SimError(f"The similarity server is not running at {CONFIG.url} "
                       "and HOOPS_SIM_AUTOSTART is off.")
    if not CONFIG.python:
        raise SimError(f"The similarity server is not running at {CONFIG.url}. "
                       "Set HOOPS_SIM_PYTHON to enable automatic start.")
    if not CONFIG.is_local:
        raise SimError(f"The similarity server at {CONFIG.url} is not running and "
                       "cannot be started automatically (the URL is not local).")
    proc, log_path = _launch()
    deadline = time.monotonic() + CONFIG.startup_timeout
    while time.monotonic() < deadline:
        if _alive():
            return
        if proc.poll() is not None:
            raise SimError(f"The server exited with code {proc.returncode} while "
                           f"starting. See {log_path}")
        _sleep(2.0)
    raise SimError(f"The server did not answer within {CONFIG.startup_timeout:g} s. "
                   f"It may still be starting. See {log_path}")


@mcp.tool()
def server_status() -> dict:
    """Report whether the similarity server answers, and what MCP started."""
    out: dict = {"url": CONFIG.url, "running": _alive()}
    state = _read_state()
    if state:
        out["started_by_mcp"] = {"pid": state.get("pid"), "log": state.get("log")}
    return out


@mcp.tool()
def start_server() -> dict:
    """Start the similarity server if it is not already running."""
    was_running = _alive()
    try:
        ensure_server()
    except SimError as exc:
        return {"running": False, "error": str(exc)}
    out = {"url": CONFIG.url, "running": True, "already_running": was_running}
    state = _read_state()
    if state and not was_running:
        out.update(pid=state.get("pid"), log=state.get("log"))
    return out


def _kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, check=False)
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


@mcp.tool()
def stop_server() -> dict:
    """Stop the server, but only one that this MCP server started itself."""
    state = _read_state()
    if not state or not state.get("pid"):
        return {"stopped": False,
                "message": "No server started by this MCP server is recorded. "
                           "A server started another way is left running."}
    if not _alive():
        CONFIG.state_file.unlink(missing_ok=True)
        return {"stopped": False, "message": "The recorded server is not running; "
                                             "the stale record was removed."}
    _kill_tree(int(state["pid"]))
    CONFIG.state_file.unlink(missing_ok=True)
    return {"stopped": True, "pid": state["pid"]}


# ---------------------------------------------------------------------------
# STEP transforms
# ---------------------------------------------------------------------------


def _real(x: float) -> str:
    s = repr(float(x))
    if "e" in s:
        mant, exp = s.split("e")
        s = f"{mant if '.' in mant else mant + '.'}E{exp}"
    return s


def _rotation(deg: float, axis: str):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if axis == "x":
        return lambda x, y, z: (x, c * y - s * z, s * y + c * z)
    if axis == "y":
        return lambda x, y, z: (c * x + s * z, y, -s * x + c * z)
    return lambda x, y, z: (c * x - s * y, s * x + c * y, z)


def transform_step_text(text: str, translate=(0.0, 0.0, 0.0), rotate_deg: float = 0.0,
                        axis: str = "z") -> tuple:
    """Rotate about an axis through the origin, then translate.

    Only 3-coordinate CARTESIAN_POINTs move and only 3-component DIRECTIONs
    turn. Two-coordinate points (pcurves in parameter space) are left alone.
    Returns ``(text, points_changed, points_kept, directions_changed)``.
    """
    rot = _rotation(rotate_deg, axis)
    tx, ty, tz = (float(v) for v in translate)
    counts = {"p": 0, "k": 0, "d": 0}

    def make(kind: str, moves: bool):
        pat = re.compile(r"\b(" + kind + r"\s*\(\s*'[^']*'\s*,\s*\()([^()]*)(\))", re.S)

        def rep(m):
            vals = [v.strip() for v in m.group(2).split(",")]
            if len(vals) != 3:
                counts["k"] += moves
                return m.group(0)
            x, y, z = rot(*(float(v) for v in vals))
            if moves:
                x, y, z = x + tx, y + ty, z + tz
                counts["p"] += 1
            else:
                counts["d"] += 1
            return f"{m.group(1)} {_real(x)}, {_real(y)}, {_real(z)} {m.group(3)}"

        return pat, rep

    out = text
    for kind, moves in (("CARTESIAN_POINT", True), ("DIRECTION", False)):
        pat, rep = make(kind, moves)
        out = pat.sub(rep, out)
    return out, counts["p"], counts["k"], counts["d"]


def _num(v: float) -> str:
    return f"{v:g}".replace("-", "m").replace(".", "p")


def make_copy(path: str, translate=(0, 0, 0), rotate_deg: float = 0.0,
              axis: str = "z") -> Path:
    src = Path(path)
    if src.suffix.lower() not in (".stp", ".step"):
        raise SimError("Only .stp / .step files can be transformed.")
    if not src.is_file():
        raise SimError(f"File not found: {src}")
    if axis not in ("x", "y", "z"):
        raise SimError("axis must be 'x', 'y' or 'z'.")
    if len(translate) != 3:
        raise SimError("translate must have three numbers.")
    parts = []
    if any(float(v) for v in translate):
        parts.append("t" + "_".join(_num(float(v)) for v in translate))
    if rotate_deg:
        parts.append(f"r{_num(float(rotate_deg))}{axis}")
    tag = "__".join(parts) or "copy"
    text = src.read_bytes().decode("latin-1")
    out, *_ = transform_step_text(text, translate, float(rotate_deg), axis)
    CONFIG.work_dir.mkdir(parents=True, exist_ok=True)
    dst = CONFIG.work_dir / f"{src.stem}__{tag}{src.suffix}"
    dst.write_bytes(out.encode("latin-1"))
    return dst


@mcp.tool()
def make_transformed_copy(path: str, translate: list[float] = [0, 0, 0],
                          rotate_deg: float = 0, axis: str = "z") -> dict:
    """Write a translated / rotated copy of a STEP file into the work folder.

    Rotation is about an axis (x, y or z) through the origin and happens
    before the translation. Only .stp / .step files are accepted.
    """
    try:
        dst = make_copy(path, translate, rotate_deg, axis)
    except SimError as exc:
        return {"error": str(exc)}
    return {"path": str(dst)}


# ---------------------------------------------------------------------------
# Verification tools
# ---------------------------------------------------------------------------


def upload_path(path: str, timeout: float = DEFAULT_TIMEOUT) -> tuple:
    """Register a file that is on the server machine; return (file_id, filename)."""
    data = api("POST", "/files/upload-from-path", params={"file_path": path},
               timeout=timeout)
    files = data.get("files") or []
    if not files:
        errs = data.get("errors") or []
        raise SimError(f"Upload failed for {path}: {str(errs)[:200]}")
    return files[0]["file_id"], files[0].get("filename") or Path(path).name


def _compare(ids: list, timeout: float) -> dict:
    return api("POST", "/similarity/compare", params={"file_ids": ",".join(ids)},
               timeout=max(timeout, 1.0))


def _summarise_pairs(data: dict) -> dict:
    names = [f["filename"] for f in data.get("files", [])]
    pairs = sorted(data.get("pairs", []), key=lambda p: -p["score"])
    out = [{"a": names[p["a"]], "b": names[p["b"]], "score": round(p["score"], 4)}
           for p in pairs[:MAX_PAIRS]]
    res: dict = {"files": names, "pairs": out, "total_pairs": len(pairs),
                 "truncated": len(pairs) > MAX_PAIRS}
    if data.get("errors"):
        res["errors"] = [str(e)[:200] for e in data["errors"][:5]]
    return res


@mcp.tool()
def compare_parts(paths: list[str], timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Compare CAD files (paths on the server machine) by embedding similarity.

    Returns the pairs, highest similarity first. Returns what it has, with
    ``timed_out`` set, when ``timeout`` seconds pass first.
    """
    if not 2 <= len(paths) <= MAX_FILES_COMPARE:
        return {"error": f"Give between 2 and {MAX_FILES_COMPARE} files."}
    deadline = time.monotonic() + timeout
    ids: list = []
    try:
        for p in paths:
            if time.monotonic() >= deadline:
                return {"timed_out": True, "uploaded": len(ids), "requested": len(paths)}
            ids.append(upload_path(p, deadline - time.monotonic())[0])
        data = _compare(ids, deadline - time.monotonic())
    except SimError as exc:
        if "Timed out" in str(exc):
            return {"timed_out": True, "uploaded": len(ids), "requested": len(paths)}
        return {"error": str(exc)}
    return _summarise_pairs(data)


@mcp.tool()
def embed_part(path: str, include_vector: bool = False) -> dict:
    """Embed one CAD file and return the model, dimension and body count.

    The embed endpoint reports no oriented bounding box; the vector is
    returned only when ``include_vector`` is true.
    """
    try:
        fid, name = upload_path(path)
        data = api("POST", "/similarity/embed",
                   params={"file_id": fid, "include_vector": str(include_vector).lower()})
    except SimError as exc:
        return {"error": str(exc)}
    out = {k: data.get(k) for k in ("filename", "model_name", "dim", "num_bodies", "cached")}
    out["filename"] = out["filename"] or name
    if include_vector and data.get("vector") is not None:
        out["vector"] = [round(v, 6) for v in data["vector"]]
    return out


@mcp.tool()
def translation_stability(path: str, shifts: list[list[float]] = [[1, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, 50]],
                          rotations_deg: list[float] = [],
                          timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Score the original STEP file against shifted / rotated (about z) copies."""
    variants = [("shift " + ",".join(f"{v:g}" for v in s), {"translate": s}) for s in shifts]
    variants += [(f"rotate {r:g} deg z", {"rotate_deg": r}) for r in rotations_deg]
    if not variants or len(variants) > MAX_VARIANTS:
        return {"error": f"Give between 1 and {MAX_VARIANTS} shifts and rotations in total."}
    deadline = time.monotonic() + timeout
    rows: list = []
    try:
        copies = [make_copy(path, **kw) for _, kw in variants]
        ids = [upload_path(path, deadline - time.monotonic())[0]]
        for c in copies:
            if time.monotonic() >= deadline:
                return {"timed_out": True, "uploaded": len(ids) - 1, "variants": len(copies)}
            ids.append(upload_path(str(c), deadline - time.monotonic())[0])
        data = _compare(ids, deadline - time.monotonic())
    except SimError as exc:
        if "Timed out" in str(exc):
            return {"timed_out": True, "variants": len(variants)}
        return {"error": str(exc)}
    score = {}
    for p in data.get("pairs", []):
        if p["a"] == 0:
            score[p["b"]] = p["score"]
        elif p["b"] == 0:
            score[p["a"]] = p["score"]
    for i, (label, _) in enumerate(variants, start=1):
        s = score.get(i)
        rows.append({"variant": label, "score": None if s is None else round(s, 4)})
    return {"file": Path(path).name, "scores": rows}


# ---------------------------------------------------------------------------
# Index search
# ---------------------------------------------------------------------------


def _ix(index: str) -> str:
    return f"/similarity/index/{quote(index, safe='')}"


def _limited(limit: int) -> int:
    return max(1, min(int(limit), MAX_LIST))


@mcp.tool()
def list_indexes() -> dict:
    """List the indexes with their part counts."""
    try:
        items = api("GET", "/similarity/index/list")
    except SimError as exc:
        return {"error": str(exc)}
    out = [{"name": i["name"], "count": i.get("count"), "last_modified": i.get("last_modified")}
           for i in items[:MAX_LIST]]
    return {"indexes": out, "total": len(items), "has_more": len(items) > MAX_LIST}


@mcp.tool()
def index_stats(index: str) -> dict:
    """File, body and assembly counts of one index."""
    try:
        return api("GET", f"{_ix(index)}/stats")
    except SimError as exc:
        return {"error": str(exc)}


@mcp.tool()
def model_consistency(index: str) -> dict:
    """Check whether an index was built with the current checkpoint and HOOPS AI version.

    The first call after the server starts hashes the checkpoint and can be slow.
    """
    try:
        d = api("GET", f"{_ix(index)}/model-consistency", timeout=120)
    except SimError as exc:
        return {"error": str(exc)}
    return {
        "index": index,
        "status": d.get("status"),
        "sdk_status": d.get("sdk_status"),
        "current_ckpt_sha256": str(d.get("current_ckpt_sha256") or "")[:12],
        "current_hoops_ai_version": d.get("current_hoops_ai_version"),
        "total_files": d.get("total_files"),
        "matching_files": d.get("matching_files"),
        "mismatching_files": d.get("mismatching_files"),
        "unknown_files": d.get("unknown_files"),
        "sdk_matching_files": d.get("sdk_matching_files"),
        "sdk_mismatching_files": d.get("sdk_mismatching_files"),
        "sdk_unknown_files": d.get("sdk_unknown_files"),
    }


@mcp.tool()
def search_similar(index: str, path: Optional[str] = None, part_id: Optional[str] = None,
                   k: int = 20, mode: str = "part") -> dict:
    """Find parts (mode "part") or assemblies (mode "assembly") similar to a query.

    Give either ``path`` (a file on the server machine) or ``part_id``.
    """
    if bool(path) == bool(part_id):
        return {"error": "Give exactly one of path or part_id."}
    if mode not in ("part", "assembly"):
        return {"error": "mode must be 'part' or 'assembly'."}
    k = max(1, min(int(k), MAX_SEARCH_K))
    try:
        fid = part_id or upload_path(path)[0]
        route = "search" if mode == "part" else "search-assembly"
        data = api("POST", f"{_ix(index)}/{route}", params={"file_id": fid, "top_k": k})
    except SimError as exc:
        return {"error": str(exc)}
    hits = []
    for h in data.get("hits", [])[:k]:
        row = {"part_id": h["id"], "filename": (h.get("metadata") or {}).get("filename"),
               "score": round(h["score"], 4)}
        if mode == "assembly":
            row["coverage"] = round(h.get("coverage", 0.0), 4)
        hits.append(row)
    return {"hits": hits, "count": len(hits)}


def _part_row(it: dict) -> dict:
    return {"part_id": it["id"], "filename": it.get("filename"), "kind": it.get("kind"),
            "bodies": it.get("bodies"), "tags": it.get("tags") or []}


def _scan_parts(index: str, match, extra: dict, deadline: float) -> tuple:
    """Page through the listing; return (matches, finished)."""
    found, offset = [], 0
    while True:
        if time.monotonic() >= deadline:
            return found, False
        page = api("GET", f"{_ix(index)}/parts", timeout=max(deadline - time.monotonic(), 1.0),
                   params={"offset": offset, "limit": PAGE, **extra})
        items = page.get("items", [])
        found.extend(i for i in items if match(i))
        offset += len(items)
        if not items or offset >= page.get("total", 0):
            return found, True


@mcp.tool()
def list_parts(index: str, tag: Optional[str] = None, filename: Optional[str] = None,
               limit: int = 50, offset: int = 0) -> dict:
    """List registered parts, optionally filtered by tag and by file name text."""
    limit, offset = _limited(limit), max(0, int(offset))
    extra = {"tags": tag} if tag else {}
    try:
        if not filename:
            page = api("GET", f"{_ix(index)}/parts",
                       params={"offset": offset, "limit": limit, **extra})
            total = page.get("total", 0)
            return {"parts": [_part_row(i) for i in page.get("items", [])],
                    "total": total, "offset": offset,
                    "has_more": offset + len(page.get("items", [])) < total}
        needle = filename.lower()
        found, done = _scan_parts(
            index, lambda i: needle in (i.get("filename") or "").lower(), extra,
            time.monotonic() + DEFAULT_TIMEOUT)
    except SimError as exc:
        return {"error": str(exc)}
    window = found[offset:offset + limit]
    out = {"parts": [_part_row(i) for i in window], "total": len(found), "offset": offset,
           "has_more": offset + len(window) < len(found)}
    if not done:
        out["note"] = "Scan stopped at the time limit; the total is a lower bound."
    return out


@mcp.tool()
def get_part(index: str, part_id: str) -> dict:
    """Metadata of one part: file name, kind, body count, tags and so on.

    The API has no single-part lookup, so this pages through the listing. The
    oriented bounding box is not exposed by the listing.
    """
    try:
        found, _ = _scan_parts(index, lambda i: i["id"] == part_id, {},
                               time.monotonic() + DEFAULT_TIMEOUT)
    except SimError as exc:
        return {"error": str(exc)}
    if not found:
        return {"error": f"Part {part_id} not found in index '{index}' (or scan timed out)."}
    it = found[0]
    return {**_part_row(it), "registered_at": it.get("registered_at"),
            "has_thumbnail": bool(it.get("thumbnail_url"))}


@mcp.tool()
def get_part_thumbnail(index: str, part_id: str):
    """The stored PNG thumbnail of a part."""
    try:
        ensure_server()
        resp = _http("GET", f"{_ix(index)}/parts/{quote(part_id, safe='')}/thumbnail")
    except SimError as exc:
        return {"error": str(exc)}
    return Image(data=resp.content, format="png")


# ---------------------------------------------------------------------------
# Tags (no irreversible operations are exposed)
# ---------------------------------------------------------------------------


def _ids(part_ids: list[str]) -> list:
    return [p.strip() for p in part_ids if p and p.strip()][:MAX_IDS_PER_CALL]


@mcp.tool()
def list_tags(index: str) -> dict:
    """Tags in use in an index with their part counts."""
    try:
        data = api("GET", f"{_ix(index)}/tags")
    except SimError as exc:
        return {"error": str(exc)}
    return {"tags": data["tags"][:MAX_LIST], "total": data["total"],
            "has_more": len(data["tags"]) > MAX_LIST}


@mcp.tool()
def parts_by_tag(index: str, tag: str, limit: int = 50, offset: int = 0) -> dict:
    """Parts carrying a tag."""
    return list_parts(index, tag=tag, limit=limit, offset=offset)


@mcp.tool()
def set_part_tags(index: str, part_id: str, tags: list[str]) -> dict:
    """Replace the whole tag set of one part."""
    try:
        return api("PUT", f"{_ix(index)}/parts/{quote(part_id, safe='')}/tags",
                   json={"tags": tags})
    except SimError as exc:
        return {"error": str(exc)}


def _bulk(method: str, index: str, tag: str, part_ids: list[str]) -> dict:
    ids = _ids(part_ids)
    if not ids:
        return {"error": "No part_ids given."}
    try:
        return api(method, f"{_ix(index)}/tags/{quote(tag, safe='')}/parts",
                   json={"part_ids": ids})
    except SimError as exc:
        return {"error": str(exc)}


@mcp.tool()
def tag_parts(index: str, tag: str, part_ids: list[str]) -> dict:
    """Add a tag to several parts."""
    return _bulk("POST", index, tag, part_ids)


@mcp.tool()
def untag_parts(index: str, tag: str, part_ids: list[str]) -> dict:
    """Remove one tag from several parts."""
    return _bulk("DELETE", index, tag, part_ids)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@mcp.tool()
def create_index(name: str) -> dict:
    """Create an empty index (lowercase letters, digits, "_" and "-")."""
    try:
        return api("POST", "/similarity/index/create", params={"name": name})
    except SimError as exc:
        return {"error": str(exc)}


def collect_paths(paths: Optional[list], folder: Optional[str], recursive: bool) -> list:
    found = [str(p) for p in (paths or [])]
    if folder:
        root = Path(folder)
        if not root.is_dir():
            raise SimError(f"Folder not found: {root}")
        it = root.rglob("*") if recursive else root.glob("*")
        found += sorted(str(p) for p in it
                        if p.is_file() and p.suffix.lower() in CAD_EXTENSIONS)
    return found


@mcp.tool()
def register_files(index: str, paths: Optional[list[str]] = None,
                   folder: Optional[str] = None, recursive: bool = False) -> dict:
    """Start a background job that registers files in an index.

    Files are paths on the server machine. At most 500 files are taken per
    call and the call stops uploading after about 50 s; call again for the
    rest (re-uploads are harmless). The job is not awaited: use job_status.
    """
    try:
        files = collect_paths(paths, folder, recursive)
        if not files:
            return {"error": "No files to register."}
        batch = files[:MAX_REGISTER_FILES]
        deadline = time.monotonic() + DEFAULT_TIMEOUT
        ids, failed = [], []
        for p in batch:
            if time.monotonic() >= deadline:
                break
            try:
                ids.append(upload_path(p, max(deadline - time.monotonic(), 1.0))[0])
            except SimError as exc:
                failed.append({"path": p, "error": str(exc)[:200]})
        if not ids:
            return {"error": "No file could be uploaded.", "failed": failed[:5]}
        job = api("POST", f"{_ix(index)}/jobs/add", json={"file_ids": ids})
    except SimError as exc:
        return {"error": str(exc)}
    done = len(ids) + len(failed)
    return {"job_id": job.get("job_id"), "status": job.get("status"),
            "submitted": len(ids), "upload_failed": len(failed), "failed": failed[:5],
            "not_processed": len(files) - done}


@mcp.tool()
def job_status(index: str, job_id: str) -> dict:
    """Status and progress of a registration job."""
    try:
        d = api("GET", f"{_ix(index)}/jobs/{quote(job_id, safe='')}")
    except SimError as exc:
        return {"error": str(exc)}
    keep = ("job_id", "status", "progress", "summary", "detail", "created_at",
            "started_at", "finished_at")
    out = {k: d.get(k) for k in keep}
    errs = d.get("errors") or []
    out["error_count"] = len(errs)
    out["errors"] = [str(e)[:200] for e in errs[:5]]
    return out


def _failed_section(text: str) -> str:
    keep, in_failed = [], False
    for line in text.splitlines():
        if line.startswith("["):
            in_failed = line.startswith("[FAILED")
        if line.startswith("#") or in_failed:
            keep.append(line)
    return "\n".join(keep)


@mcp.tool()
def job_report(index: str, job_id: str, only_failed: bool = True) -> dict:
    """The text report of a finished job (header and failed files by default)."""
    try:
        ensure_server()
        text = _http("GET", f"{_ix(index)}/jobs/{quote(job_id, safe='')}/report").text
    except SimError as exc:
        return {"error": str(exc)}
    if only_failed:
        text = _failed_section(text)
    return {"report": text[:REPORT_CHARS], "truncated": len(text) > REPORT_CHARS}


if __name__ == "__main__":
    mcp.run()
