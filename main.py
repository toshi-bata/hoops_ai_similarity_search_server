"""ASGI composition root for the HOOPS AI Similarity Search Server.

This module assembles the FastAPI application: it bootstraps a bundled SDK onto
``sys.path`` when present, configures logging, registers the exception handlers
and the routers, and mounts the static web client. It owns no request logic of
its own -- every endpoint lives under ``routers/`` and every behaviour under the
domain modules reached through ``core``.

The ``lifespan`` context manager drives startup and shutdown in this exact
order. On startup: ``core.clear_embedding_memory_cache()`` drops process-local
caches, ``core.log_resolved_configuration()`` records the effective settings,
``core.run_startup_maintenance()`` TTL-sweeps the transient directories,
``core.init_hoops_license()`` initialises the HOOPS license,
``embeddings.log_compute_environment("STARTUP_ENV")`` logs the interpreter and
math-library configuration (only when ``HOOPS_SEARCH_TIMING=1``), and
``core.start_startup_warmup()`` kicks off the opt-in background index warmup. On
shutdown a single step runs: ``jobs.get_index_job_store().shutdown(wait=False)``
stops accepting new registration jobs without waiting for in-flight work.

``main.py`` and ``core.py`` are the only two files permitted to ``import core``
(pinned by ``tests/test_module_layout.py::ImportHygiene._IMPORT_CORE_ALLOWED``);
every other module reaches its siblings directly. The registered exception
handlers translate ``core.EnvConfigError`` and ``core.PathConfigError`` into 503
responses so a misconfigured deployment fails cleanly instead of leaking a stack
trace, and the three routers (``files``, ``similarity``, ``index_jobs``) expose
the HTTP surface.
"""
from contextlib import asynccontextmanager
import logging
import mimetypes
import os
import pathlib
import sys


def _bootstrap_bundled_site_packages() -> None:
    """Make a bundled HOOPS AI SDK importable without a system install.

    A redistributed package ships the required site-packages under
    ``<app>/.venv/...`` (see tools/build_redist_package). When that directory is
    present next to this file we prepend it to sys.path so ``import hoops_ai``
    resolves against the bundled SDK. This is the server-side analogue of the
    native bridge's test_client, which auto-detects its bundled ``.venv``
    relative to its own location. In a developer checkout the directory is
    absent and this is a no-op, so the developer's own environment is used.
    """
    app_root = pathlib.Path(__file__).resolve().parent
    if os.name == "nt":
        candidate = app_root / ".venv" / "Lib" / "site-packages"
    else:
        candidate = (
            app_root / ".venv" / "lib" / f"python3.{sys.version_info.minor}"
            / "site-packages"
        )
    if candidate.is_dir():
        resolved = str(candidate)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


_bootstrap_bundled_site_packages()

mimetypes.init()
mimetypes.add_type("application/javascript", ".mjs")
# WebAssembly must be served as application/wasm or browsers refuse to
# instantiate it. Python's mimetypes database does not always know this type,
# so register it explicitly for the web client's engine files.
mimetypes.add_type("application/wasm", ".wasm")

import core
import jobs
import config
import embeddings
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from routers import client_config, files, index_jobs, similarity


def _configure_logging() -> None:
    """Give the application's own loggers a handler and a level.

    uvicorn only configures its own ``uvicorn.*`` loggers, so without this the
    root logger keeps its WARNING default and every ``logger.info()`` in core
    and the routers is silently dropped -- including the diagnostics that report
    how long expensive one-off work took. ``force=True`` so we win even when an
    embedded runner (or a library import) has already installed a root handler.
    """
    config.load_env_file()
    level = os.environ.get("HOOPS_AI_LOG_LEVEL", "INFO").strip().upper()
    if not isinstance(logging.getLevelName(level), int):
        level = "INFO"
    logging.basicConfig(
        level=level,
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )


_configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Process-local caches only. The on-disk directories are NOT wiped here:
    # uploads/ is the CAD store backing every registered index, and deleting
    # out/ or embeddings_cache/ would also clobber a second instance running
    # on another port. Transient directories are TTL-swept instead.
    core.clear_embedding_memory_cache()
    core.log_resolved_configuration()
    core.run_startup_maintenance()
    core.init_hoops_license()
    # Log this process's interpreter, SDK and math-library configuration once at
    # startup (HOOPS_SEARCH_TIMING=1 only), so two deployments can be compared
    # line for line. A no-op when the flag is unset.
    embeddings.log_compute_environment("STARTUP_ENV")
    # Optional background warmup (opt-in via HOOPS_AI_WARMUP_INDEXES). Started
    # after the license is initialised so the heavy builds can run; it does
    # nothing when no indexes are configured. Never blocks request serving.
    core.start_startup_warmup()
    yield
    # Stop accepting new jobs. Work already inside embed_shape_batch cannot be
    # interrupted, so we do not wait for it; the next startup marks whatever
    # was still in progress as interrupted.
    jobs.get_index_job_store().shutdown(wait=False)


app = FastAPI(
    title="HOOPS AI Similarity Search Server",
    lifespan=lifespan,
)


@app.exception_handler(core.EnvConfigError)
async def env_config_error_handler(request: Request, exc: core.EnvConfigError):
    return JSONResponse(
        status_code=503,
        content={"detail": "Service unavailable: server configuration error. Check the server logs for details."},
    )


@app.exception_handler(core.PathConfigError)
async def path_config_error_handler(request: Request, exc: core.PathConfigError):
    return JSONResponse(
        status_code=503,
        content={"detail": "Service unavailable: required server resource not found. Check the server logs for details."},
    )

app.include_router(files.router)
app.include_router(similarity.router)
app.include_router(index_jobs.router)
# The web client's one look at the server's own settings (see the module note
# for what may be exposed there).
app.include_router(client_config.router)

config.CAD_VIEWER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/out", StaticFiles(directory=str(config.CAD_VIEWER_OUTPUT_DIR)), name="out")

_static_dir = pathlib.Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Built web client. Sources live in web/ and are built into static/app/ (see
# README). The build output is NOT committed: the bundle and its engine wasm are
# >95% @ts3d-hoops (HOOPS Web Viewer) code, and a committed copy would carry none
# of the license text that ships with the npm packages. Users build it from web/
# with "npm ci && npm run build" (same rationale as bin/scs_points,
# decision #16). The API is unaffected whether or not the client is built.
_app_dir = _static_dir / "app"
_app_dir.mkdir(exist_ok=True)
_WEB_BUILD_HINT = (
    "The HOOPS AI Similarity Search web client has not been built yet.\n"
    "\n"
    "Build it from the repository root:\n"
    "    cd web\n"
    "    npm ci\n"
    "    npm run build\n"
    "\n"
    "That writes the bundle and the viewer engine into static/app/, after which\n"
    "this page (/app/) serves the client. The server API is unaffected and keeps\n"
    "working while the client is unbuilt.\n"
)


class _WebClientFiles(StaticFiles):
    """Serve the built web client, or an explanatory 503 when it is absent.

    static/app/ is a build output that is not committed (see the note above the
    mount). Without it a plain StaticFiles mount answers /app/ with a bare 404
    that tells the user nothing. When index.html is missing this turns the
    not-found response into a message pointing at the build command; when the
    client is built it behaves like an ordinary StaticFiles mount.
    """

    async def get_response(self, path, scope):
        built = (_app_dir / "index.html").is_file()
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and not built:
                return PlainTextResponse(_WEB_BUILD_HINT, status_code=503)
            raise
        if response.status_code == 404 and not built:
            return PlainTextResponse(_WEB_BUILD_HINT, status_code=503)
        return response


if not (_app_dir / "index.html").is_file():
    logging.getLogger(__name__).warning(
        "Web client not built: /app/ is unavailable. Build it from web/ with "
        "'npm ci && npm run build' (writes static/app/). The API is "
        "unaffected."
    )

app.mount("/app", _WebClientFiles(directory=str(_app_dir), html=True), name="app")


if __name__ == "__main__":
    import argparse
    import socket
    import sys
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the HOOPS AI Similarity Search Server.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        if _s.connect_ex(("127.0.0.1", args.port)) == 0:
            print(
                f"Error: port {args.port} is already in use. "
                f"Use --port <number> to specify a different port.",
                file=sys.stderr,
            )
            sys.exit(1)

    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)

