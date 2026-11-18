<!-- A Japanese translation is maintained in README_ja.md. When you change either README, update the other so both languages stay in sync. -->

# mcp_server — MCP server for the similarity search server

**Languages:** English | [日本語](README_ja.md)

A FastMCP server that lets Claude Desktop use the similarity search server
without the web app: compare parts, embed, search an index, tag and register.
If the similarity server is not running, this MCP server starts it.

It runs in its own light environment (`mcp` and `httpx` only) and talks to the
similarity server over HTTP. It does not import HOOPS AI, and nothing is added
to the HOOPS AI virtual environment.

## Setup

```bash
cd mcp_server
uv sync
```

## Register with Claude Desktop

Add an entry under `mcpServers` in `claude_desktop_config.json`
(Windows: `%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "hoops-ai-similarity": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\path\\to\\hoops_ai_similarity_search_server\\mcp_server",
        "run",
        "server.py"
      ],
      "env": {
        "HOOPS_SIM_URL": "http://127.0.0.1:8000",
        "HOOPS_SIM_PYTHON": "C:\\path\\to\\hoops_ai_venv\\Scripts\\python.exe"
      }
    }
  }
}
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `HOOPS_SIM_URL` | `http://127.0.0.1:8000` | URL of the similarity search server |
| `HOOPS_SIM_REPO` | the folder above `mcp_server` | Folder that contains `main.py` |
| `HOOPS_SIM_PYTHON` | (none) | `python.exe` of the HOOPS AI venv. Without it the server is never started automatically |
| `HOOPS_SIM_AUTOSTART` | `1` | `0` turns automatic start off |
| `HOOPS_SIM_STARTUP_TIMEOUT` | `180` | Seconds to wait for a started server to answer |
| `HOOPS_SIM_WORK_DIR` | `mcp_server/work` | Where `make_transformed_copy` writes its files |

All file paths you give a tool are paths **on the machine that runs the
similarity server** (the server reads them itself). Use this MCP server on that
machine, or with a server that can see the same paths.

## Tools

| Group | Tool | Calls |
|-------|------|-------|
| Server | `server_status`, `start_server`, `stop_server` | `GET /similarity/index/list` (liveness) |
| Verify | `compare_parts(paths, timeout)` | `POST /files/upload-from-path`, `POST /similarity/compare` |
| | `embed_part(path, include_vector)` | `POST /files/upload-from-path`, `POST /similarity/embed` |
| | `make_transformed_copy(path, translate, rotate_deg, axis)` | none (local file) |
| | `translation_stability(path, shifts, rotations_deg, timeout)` | as `compare_parts` |
| Search | `list_indexes`, `index_stats(index)`, `model_consistency(index)` | `GET /similarity/index/list`, `GET /similarity/index/{name}/stats`, `GET /similarity/index/{name}/model-consistency` |
| | `search_similar(index, path or part_id, k, mode)` | `POST /similarity/index/{name}/search` or `/search-assembly` |
| | `list_parts`, `get_part`, `get_part_thumbnail` | `GET /similarity/index/{name}/parts`, `.../parts/{id}/thumbnail` |
| Tags | `list_tags`, `parts_by_tag` | `GET .../tags`, `GET .../parts?tags=` |
| | `set_part_tags`, `tag_parts`, `untag_parts` | `PUT .../parts/{id}/tags`, `POST` / `DELETE .../tags/{tag}/parts` |
| Register | `create_index(name)` | `POST /similarity/index/create` |
| | `register_files(index, paths, folder, recursive)` | `POST /files/upload-from-path`, `POST .../jobs/add` |
| | `job_status`, `job_report` | `GET .../jobs/{id}`, `GET .../jobs/{id}/report` |

Irreversible operations (deleting tags, clearing every tag, deleting an index
or a part) are deliberately not offered; use the web app for them.

Every answer is a few KB at most. Lists carry a limit, a total and a
`has_more` flag. A call returns within about a minute: registration is a job
you poll, and `compare_parts` and `translation_stability` take a `timeout` and
return what they have, with `timed_out` set, when it passes.

Notes on what the API allows:

- `embed_part` returns the model, dimension and body count. The embed endpoint
  has no oriented bounding box.
- `get_part` and `list_parts(filename=...)` page through the listing, because
  the API has no single-part lookup and no file-name filter.
- `register_files` uploads each file by path (up to 500 per call) and then
  submits the job; call it again for the rest.

## Automatic start and `stop_server`

Every tool first sends `GET /similarity/index/list`. If nothing answers and
`HOOPS_SIM_AUTOSTART` is on, `HOOPS_SIM_PYTHON` is set and the URL is local,
the MCP server runs `main.py --host 127.0.0.1 --port <port>` from
`HOOPS_SIM_REPO` and waits for it to answer. `main.py` is started directly, so
its Windows power-throttling opt-out also covers the embedding workers.

- On Windows the process is created with `DETACHED_PROCESS`,
  `CREATE_NEW_PROCESS_GROUP` and `CREATE_NO_WINDOW`; on Linux with
  `start_new_session`. It keeps running after the MCP server exits.
- Output goes to `mcp_server/logs/server_<time>.log`; the PID is recorded in
  `mcp_server/.state/server.json`.
- `stop_server` stops only a server recorded there (the whole process tree).
  A server started any other way is left running, and the tool says so.

`logs/`, `.state/` and `work/` are ignored by git.

## Tests

```bash
python -m pytest mcp_server
```

HTTP is mocked; no server is started.
