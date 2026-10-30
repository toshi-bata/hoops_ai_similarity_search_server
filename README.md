<!-- A Japanese translation is maintained in README_ja.md. When you change either README, update the other so both languages stay in sync. -->

# HOOPS AI Similarity Search Server

**Languages:** English | [日本語](README_ja.md)

## How this repository fits in

![How this repository fits in: HOOPS AI, a Python machine-learning framework, sits at
the bottom. A desktop application reaches it through a native bridge. This repository
takes the other route: a FastAPI web service wraps HOOPS AI, and a browser client built
with HOOPS Visualize Web talks to it over HTTP. A dashed box marks the part this
repository provides -- the web app, the web API, and the HOOPS AI calls behind it. An AI
agent reaches the same endpoints through MCP.](images/how_to_use_hoops_ai.png)

HOOPS AI is a Python machine-learning framework, and there is more than one way to put
it in front of a user.

A desktop application calls it through a native bridge. This repository takes the web
route instead: a FastAPI service wraps HOOPS AI, and a browser client built with HOOPS
Visualize Web talks to that service over HTTP. The dashed box in the diagram is what
this repository provides -- the web app, the web API, and the HOOPS AI calls behind it.

Because the server speaks plain HTTP, the browser client is not the only one it can
have. An AI agent can reach the same endpoints through MCP, and so can anything else
that can send a request.

## Status: sample / proof of concept

This repository is a **sample and proof of concept**. It exists to show how HOOPS AI
similarity search can be assembled into an on-premises server with a HOOPS Visualize Web
front end, so that Tech Soft 3D partners can see the shape of such a system and take
ideas from it.

**It is not a product, and it is not something you would ship as is.** Treat the code
as a starting point rather than production-quality software. There is no warranty and
no support commitment.

### If you are considering distributing something based on this

The HOOPS SDK components used here — HOOPS AI, HOOPS Exchange, HOOPS Visualize Web
(including its authoring libraries, libsc / hps_core), and the trained
checkpoints — are governed by the Technology Partner Agreement and the other terms
that apply to your license. **Whether you may redistribute any of them, and in what
form, depends on your own agreement.** Please don't take this repository as the answer;
check the terms that apply to you.

The same goes for the obligations that come with distributing an integrated product —
the proprietary notices you must reproduce, the product markings, and the terms of your
own agreement with your end users. Those are yours to handle; this repository does not
do them for you.

### The packaging script and the license documents it produces are also samples

`tools/build_redist_package.ps1` / `.sh` and the files they generate —
`THIRD-PARTY-NOTICES.md`, `third_party_notices/`, `THIRD_PARTY_MANIFEST.txt` —
demonstrate *one way* to assemble a self-contained package and to record where each
bundled component came from.

**They are illustrative of a process, not a compliance artifact.** They were produced
for one dependency set, on one machine, at one point in time. They have not been
reviewed by counsel. The two HOOPS Visualize Web acknowledgment texts — one for the
browser viewer and its WebAssembly engine, one for the Stream Cache library `libsc` —
were taken from the public documentation site, which carries no version marker (see
`third_party_notices/README.md`). If you ship a product, generate and verify your own
notices for your own build.

A FastAPI-based REST server **specialized for CAD similarity search and
similarity comparison**, built on [HOOPS AI](https://www.techsoft3d.com/developers/products/hoops-ai/)
(Tech Soft 3D).

The server does one thing: it finds and compares CAD parts and assemblies by
shape. It exposes shape search, shape-to-shape comparison, named-index
management with bulk registration, tags, and per-part thumbnails/stream caches
(SCS) for gallery display. The intended audience is an ISV building a product of
their own on top of this design: the code here is a reference, not something to
ship as is, but the system it sketches is one an ISV would redistribute to their
end users.

**Two things at once.** It began as that practical sample for ISV partners
embedding HOOPS AI in a product — a REST web API, several clients against one
server, license-key obfuscation, and tooling to build a redistributable
package. It has since also become an experimental web app for a second
question: how embedding vectors should be clustered so that they fall into
kinds of part. The Shape Map, k-means, Outliers, Drill in, the Ward tree and
the part catalogue all come from that. Read it either way — as a design to take
ideas from, or as a workbench for the clustering itself.

---

## Architecture

The server is a FastAPI app whose HTTP handlers hold almost no logic. Each
top-level concern lives in its own single-purpose module. `core.py` is a thin
**facade**: it imports the sixteen domain modules below and re-exports their
stable names, so the routers and tools reach everything as `core.<name>`.

| Module | Responsibility | ~Lines |
|---|---|---|
| `config.py` | Server configuration: the data root, its derived paths, and `.env` helpers | 225 |
| `locks.py` | Central registry of the process-wide lock hierarchy and its ordering rules | 143 |
| `storage.py` | CAD file storage and ingest: the upload store, ZIP extraction, and lookups | 509 |
| `embeddings.py` | Shape embedding: the encoders, the two-tier embedding cache, and model resolution | 709 |
| `index_registry.py` | Named-index registry: incremental index management and the public index API | 1593 |
| `search.py` | Part-search entry point and its finished-response cache | 372 |
| `context.py` | Context Layer: predict production records from nearest neighbours | 359 |
| `assembly.py` | Assembly-to-assembly search over the vendored AssemblyMatcher | 606 |
| `shape_map.py` | Index shape map: a 2D cluster map and a point-cloud SCS export | 1256 |
| `tags.py` | Index tag sidecars: the per-index `tags.json`, the application-wide tag vocabulary, the tags cache, and the tag-filter API | 978 |
| `viewer.py` | SCS stream-cache export for HOOPS Visualize Web | 59 |
| `licensing.py` | Activate the HOOPS AI license at startup | 157 |
| `warmup.py` | Opt-in startup warmup for named indexes | 199 |
| `jobs.py` | Background index-registration jobs | 379 |
| `runtime.py` | Runtime directory lifecycle and startup configuration logging | 214 |
| `diagnostics.py` | Per-request phase timing, independent of search and embedding | 49 |

**Dependency direction.** It runs strictly one way:

```
main.py  ->  routers/  ->  core.py (facade)  ->  domain modules  ->  config / locks / storage
```

A lower module never imports a higher one, and **no module imports `core`**.

By default the runtime data lives under `<app>/data/` (the data root), keeping
the source tree and the generated `uploads/ indexes/ out/ …` directories
separate; `HOOPS_AI_DATA_DIR` overrides the location.

**Why `core.py` still holds three functions.** `remove_from_index`,
`list_index_parts` and `delete_index` each orchestrate work that spans sibling
domains (index + tags + context + assembly). Those siblings sit *above* any
single index-owning module, so pushing an orchestrator down into one of them
would force an upward call and create a cycle. Keeping the three in the facade
is what lets the dependency graph stay one-way.

This layout is enforced at runtime: `tests/test_module_layout.py` walks every
module and fails if one imports `core`, if a re-export drifts from its owning
module, or if a domain module shadows an imported module name.

For *why* the boundaries are shaped this way -- the re-export convention, the
cache-invalidation contract, and the traps the guards protect against -- see
[docs/architecture.md](docs/architecture.md).

---

## Tests

This chapter is for people who change this repository, not for people who run
it. The sample is meant to be read and adapted by ISV partners, and the suite is
what tells you whether an adaptation still holds together; running the server
needs none of it.

The test suite is SDK-free and needs no HOOPS AI license: install the test-only
dependencies on top of `requirements.txt`, then run pytest.

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

A GitHub Actions workflow (`.github/workflows/tests.yml`) runs this same SDK-free
pytest on a Windows runner with Python 3.12 for every push and pull request to
`main`. Windows matches the server's target platform, where the suite has been
validated. It installs no HOOPS AI SDK and no front-end packages, and requires no
secrets.

---

## Requirements

- Python 3.12
- A valid **HOOPS AI license key and a HOOPS Visualize Web license key** — the
  second one is needed to author stream caches for the web viewer and to build
  the Shape Map point-cloud tool
- HOOPS AI (CPU or GPU version) installed in the environment
- **HOOPS AI Tutorial Data Assets** — the pre-trained embedding model checkpoint
  is required to embed CAD shapes. Obtain the data assets from the Tech Soft 3D
  File Transfer service by following the
  [HOOPS AI evaluation guide](https://docs.techsoft3d.com/hoops/ai/getting_started/evaluate.html#tutorial-data-assets).
  The essential file is the checkpoint
  `packages/trained_ml_models/ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt`.

  > The [HOOPS-AI-tutorials](https://github.com/techsoft3d/HOOPS-AI-tutorials/tree/main)
  > GitHub repository holds only the tutorial *notebooks* (sample code). This
  > server does **not** depend on those notebooks — it depends on the tutorial
  > **data asset** (the model checkpoint) distributed separately via File Transfer.

  **Directory layout** — `packages/` must reside directly under
  `HOOPS_AI_SDK_DIR`:

  ```
  <HOOPS_AI_SDK_DIR>/
  └── packages/
      └── trained_ml_models/
          └── ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt
  ```

  **Indexing your own CAD is the primary workflow.** With the checkpoint in
  place, end users build their own index from their own CAD files (upload +
  `/similarity/index/*`; see [Bulk registration jobs](#bulk-registration-jobs)).

---

## Setup

### 1. Install dependencies

Install HOOPS AI (CPU or GPU version) separately according to your HOOPS AI
distribution instructions. Then install this server's dependencies into the
**HOOPS AI virtual environment**:

**Windows:**
```bat
<Path\to\HOOPS_AI\install\dir>\.venv\Scripts\pip.exe install -r requirements.txt
```

**Linux:**
```bash
/path/to/HOOPS_AI/install/dir/.venv/bin/pip install -r requirements.txt
```

> On Ubuntu 22.04+ the system Python is externally managed (PEP 668) and will
> reject a bare `pip install`. Using the HOOPS AI venv's pip avoids this and
> ensures the same Python that runs the server has all required packages.

#### Additional steps for headless Linux (Ubuntu)

For running HOOPS AI on a headless Ubuntu server (no GPU, no monitor), refer to
this community forum post for step-by-step instructions:

**[I tried running HOOPS AI v1.1 headless on Ubuntu 24.04 EC2 — Tech Soft 3D Forum](https://forum.techsoft3d.com/t/i-tried-running-hoops-ai-v1-1-headless-on-ubuntu-24-04-ec2/5165)**

#### Optional: the UMAP layout

Installing `umap-learn` adds a UMAP layout beside PCA in the map's **Layout**
control. It changes only where a point is drawn -- the clustering runs on the
full 2,048-dimensional vectors either way -- and it is tuned by
`HOOPS_AI_SHAPE_MAP_UMAP_N_NEIGHBORS`, `HOOPS_AI_SHAPE_MAP_UMAP_MIN_DIST` and
`HOOPS_AI_SHAPE_MAP_UMAP_SEED`. The installation procedure, what to check
before and after it, the cost, and how to roll it back are in
[docs/umap.md](docs/umap.md) (Japanese: [docs/umap_ja.md](docs/umap_ja.md)).

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
# Windows
copy .env.example .env

# Linux
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `HOOPS_AI_LICENSE` | dev fallback | Your HOOPS AI license key. Read **only** from the `.env` file, never from system environment variables. In a redistributed package the key is supplied by the compiled license shim instead; this `.env` value is a development-only fallback (see *License key handling*). |
| `HOOPS_AI_SDK_DIR` | optional | Absolute path to a HOOPS AI SDK install directory (must contain `packages/`). Lets a machine with a full SDK install reuse that install's bundled checkpoints. When unset (a redistributed package), checkpoints are resolved from the data root and app directory; named indexes still work. |
| `HOOPS_AI_DATA_DIR` | optional | Parent directory for `uploads/ indexes/ jobs/ out/ embeddings_cache/ models/`. When unset it defaults to `<app>/data` (see [Data directories](#data-directories) — the runtime directories live under `data/`, separate from the source). Set it to place the data on another volume. |
| `HOOPS_AI_EMBEDDINGS_MODEL_NAME` | optional | Name of the embeddings checkpoint `*.ckpt` file to use (e.g. `ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt`). When set, the file is looked up in `<DATA_ROOT>/models/`, then `<APP_ROOT>/models/`, then `<HOOPS_AI_SDK_DIR>/packages/trained_ml_models/` (first match wins; other `*.ckpt` next to it are ignored). When unset, exactly one `*.ckpt` is resolved from `<DATA_ROOT>/models/` then `<APP_ROOT>/models/`. |
| `HOOPS_AI_INDEX_CACHE_MAX_MB` | optional | Memory budget for the in-process **named-index write store** (used by create/add/remove, not by search). Least-recently-used indexes are evicted above this and transparently reloaded from disk on next write. Defaults to `1024` (about three `tmcad`-sized indexes). See [Memory sizing](#memory-sizing). |
| `HOOPS_AI_EMBED_CACHE_MAX_MB` | optional | Memory budget for the in-process **embedding cache**. It is disk-backed, so an evicted entry is reloaded, not recomputed. Defaults to `256`. |
| `HOOPS_AI_WARMUP_INDEXES` | optional | Comma-separated index names to warm in the background at startup (searcher, Shape Map, assembly matcher), so the first user does not pay the cold build cost. **Unset means no warmup** (the default). See [Startup warmup](#startup-warmup). |
| `HOOPS_AI_WARMUP_MAX` | optional | Cap on how many of the configured `HOOPS_AI_WARMUP_INDEXES` are actually warmed, so a many-index host still starts quickly (default `1`). |
| `HOOPS_AI_WARMUP_STAGES` | optional | Comma-separated list of the warmup stages to run: `assembly`, `map`, `scs`, `searcher`. Unset means all of them. The assembly matcher is the slowest stage, so a deployment that does not use assembly search can leave it out. See [Startup warmup](#startup-warmup). |
| `HOOPS_AI_DEMO` | optional | Show the webinar-demo-only parts of the web client — currently the **Estimate** panel, which predicts production records from the Context Layer. Off unless set (`1`/`true`/`yes`/`on`): those predictions only mean something once an index carries real production records, and the demo seeds fabricated ones. The context API itself stays available either way. |

The table above lists only what a user or operator normally sets. Every other
knob (upload/ZIP caps, TTLs, job limits, per-index cache and prewarm toggles,
worker sizing, embedding intra-op threads, search-cache and diagnostic flags)
keeps a sensible default and is catalogued in one place: the **Environment
variable reference** block near the top of `config.py`. Set any of them through
the environment or `.env` in the same way; they are simply not repeated here.

Example `.env`:

```
HOOPS_AI_LICENSE=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
HOOPS_AI_SDK_DIR=C:\hoops_ai
HOOPS_AI_EMBEDDINGS_MODEL_NAME=ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt
```

### License key handling

This server is redistributed by ISVs to end users, so the ISV's HOOPS AI license
key should not sit in plaintext on the end user's disk. The key can instead be
carried inside a compiled extension module (the *license shim*) built with
Nuitka, so it never appears in a config file, a log line, or the process
environment.

**How the key is loaded at startup**

`core.init_hoops_license()` first tries to `import hoops_license_shim`. When that
compiled module is present, its single public function `activate()` restores the
key internally and calls `hoops_ai.set_license()` itself — the key value never
becomes a Python string in the server's own code. When the compiled module is
absent, the server falls back to reading `HOOPS_AI_LICENSE` from the `.env` file
only (never from system environment variables) and logs a warning. That fallback
is for development; a distributed deployment should always ship the compiled
shim. If the shim is active but a plaintext `HOOPS_AI_LICENSE` is still present
in `.env`, the server also warns so the unused key can be removed.

**Building the shim (operator step)**

Build with the **same Python 3.12 that runs the server**, from inside the SDK
virtual environment. The steps differ per platform.

*Windows (PowerShell):*

```powershell
# inside the SDK virtual environment
python -m pip install nuitka                 # once, if not already installed
$env:HOOPS_AI_LICENSE_BUILD = '<your-key>'   # single quotes: PowerShell does not expand $ inside them
python tools/build_license_shim.py           # add --dry-run to preview only
Remove-Item Env:\HOOPS_AI_LICENSE_BUILD
```

*Linux (bash):*

```bash
# Nuitka needs a C toolchain; install once:
sudo apt-get install -y build-essential patchelf

source /path/to/your/.venv/bin/activate      # activate the SDK virtual environment
python -m pip install nuitka                  # once, if not already installed
# leading space keeps the key out of shell history (needs HISTCONTROL=ignorespace)
 export HOOPS_AI_LICENSE_BUILD='<your-key>'
python tools/build_license_shim.py            # add --dry-run to preview only
unset HOOPS_AI_LICENSE_BUILD
```

On Ubuntu the interpreter is `python3` (or `python3.12`); the `python` command
exists only inside an activated virtual environment. Build the `.so` with the
same interpreter that runs the server so the ABI tag matches.

The build script obfuscates the key, generates an intermediate source, invokes
Nuitka, and always deletes the intermediate source and Nuitka build trees in a
`finally` block. The result is `hoops_license_shim.<abi-tag>.pyd` (Windows) or
`.so` (Linux); place it next to `core.py` so it is importable.

**Why the shim is a release artifact, not a source file**

The `.pyd`/`.so`, the generated source, and the Nuitka build trees are all
`.gitignore`d, because a Git history cannot shed an embedded key once it lands —
even an obfuscated one. Build the shim as part of your release process and ship
it alongside the package, not inside the source tree.

**Python-version and platform coupling**

The compiled shim is ABI-tagged (for example `hoops_license_shim.cp312-win_amd64.pyd`).
Upgrading the Python runtime (e.g. 3.12 → 3.13) makes the shim un-importable, at
which point the server silently reverts to the `.env` fallback (with a warning).
Rebuild the shim after any Python upgrade. Linux distributions need their own
`.so` build; a Windows `.pyd` will not load there.

**What this protects, and what it does not**

This is **obfuscation, not encryption** — the de-obfuscation key ships inside the
same binary. It keeps the license key from being exposed through configuration
files, log output, and the process environment, and from appearing in the
`strings` output of the artifact. It does **not** stop a determined party who
replaces the `hoops_ai.set_license` API to capture the key at the moment it is
passed in; that is a Python-level interception the shim cannot prevent, and no
stronger guarantee is claimed.

### 3. Build the web client

The reference web client served at `/app/` is a Vite build output. Its bundle
and the viewer engine are content of the HOOPS Visualize Web SDK, delivered as
the `@ts3d-hoops` npm packages, so the build output is **not part of the
repository**; you build it locally. Each build pulls those packages — together
with the license text that ships inside them — from the registry. This mirrors
how the `bin/scs_points` binary is built locally rather than shipped here.

Requires Node.js (verified with v22.12.0 LTS). From the repository root:

```bash
cd web
npm ci
npm run build
```

This writes the client into `static/app/`. The server still starts and the API
still works without this step — only `/app/` is unavailable until the client is
built, and it returns a message telling you to run the commands above. See
[Web client](#web-client) for details.

### 4. Build the Shape Map point-cloud tool

`bin/scs_points[.exe]` is built locally for the same reason the web client is:
it links a HOOPS license key into the executable, so it is **not part of the
repository**. Download `hoops_license.h` for your account from the Tech Soft 3D
Developer Zone and place it in `tools/scs_points/` first. From the repository
root:

**Windows** (the MSVC compiler from Build Tools for Visual Studio 2022; no IDE):
```powershell
cd tools\scs_points
copy _win_env.bat.example _win_env.bat
# edit the SDK path, then:
.\build.bat
```

**Linux:**
```bash
cd tools/scs_points
cp _linux_env.sh.example _linux_env.sh
# edit the SDK path, then:
source _linux_env.sh
make
```

This step is optional, but the Shape Map needs it. Without the tool the server
still starts, the API still works and `/map` still answers — only `/map/scs`
returns **503**, so the client's Shape Map cannot draw its point cloud. See
[Shape Map point cloud](#shape-map-point-cloud) and
[`tools/scs_points/`](tools/scs_points/README.md) for details.

### 5. Start the server

From the repository root:

**Windows:**
```bat
<HOOPS_AI_INSTALL_DIR>\.venv\Scripts\python.exe main.py --port 8000
```

**Linux (headless server, no display):**
```bash
./start_server.sh --port 8000
```
`start_server.sh` starts an `Xvfb` virtual display on `:99` (HOOPS needs an
OpenGL context to author stream caches) and runs `main.py` with the HOOPS AI
venv. Set `HOOPS_AI_VENV` to override the venv path; install `xvfb` first
(`sudo apt-get install -y xvfb`).

**Linux (desktop session, display present):**
```bash
source /path/to/HOOPS_AI/.venv/bin/activate   # the HOOPS AI venv
python main.py --port 8000
```
On a machine with a real desktop (for example a VirtualBox GUI session) `$DISPLAY`
is already set, so `Xvfb` is not needed — run `main.py` directly instead of
`start_server.sh`.

`main.py` accepts `--host` (default `0.0.0.0`), `--port` (default `8000`) and
`--reload`. The server refuses to start if the port is already in use.

---

## Web client

A reference web client (gallery + 3D viewer) ships with the server. It is a
demo and a starting point ISVs can adapt. It is built on **HOOPS Visualize Web**,
and its screens are assembled from **HOOPS Web Components** — the
Web-Components-based user-interface framework that comes with that SDK. In npm
terms: `@ts3d-hoops/web-viewer-components` supplies the UI elements (the
`<hoops-web-viewer>` element and the toolbars around it),
`@ts3d-hoops/web-viewer` the viewer API the client scripts against, and
`@ts3d-hoops/sc-engine` the graphics engine. All three are pinned to `2026.2.0`
to match the SCS authored on the server.

**The SDK files arrive through npm, not by hand.** A HOOPS Visualize Web
deployment has traditionally meant copying `hoops-web-viewer.mjs` and
`engine.esm.wasm` out of the SDK install and into the web root yourself. Here
the pinned packages carry the same files: `npm ci` fetches them into
`web/node_modules/`, `npm run build` bundles the viewer into
`static/app/assets/`, and a small Vite plugin (`copyEngineAssets` in
`web/vite.config.js`) copies `engine.esm.wasm` verbatim from
`@ts3d-hoops/sc-engine` into `static/app/engine/`. Nothing is loaded from a CDN,
and nothing has to be copied by hand.

Once built, open it at `http://<host>:<port>/app/` (e.g.
`http://127.0.0.1:8000/app/`). It talks to the server on the same origin, so no
CORS configuration is needed.

### Managing indexes from the UI

The index picker in the side panel has **New…** and **Delete** buttons next to
it, so operators can create and delete named indexes without leaving the
browser:

- **New…** opens a small dialog that takes an index name
  (`^[a-z0-9_-]{1,64}$`) and creates an empty index
  (`POST /similarity/index/create`). The embeddings model is the single
  checkpoint the server resolved at startup; it is not chosen here.
  The server's message is shown inline when the name already exists (409) or is
  invalid (422).
- **Delete** removes the selected index and all of its data
  (`DELETE /similarity/index/{name}?confirm=true`) after a confirmation.

After a create or delete the picker re-lists the indexes and selects a usable
one. To add parts to an index, use **Register…** (see the bulk-registration
workflow below).

### Version pinning (keep three components in lock-step)

Three separate HOOPS components must stay on the **same version** (currently
`2026.2.0`) because they exchange the same StreamCache (SCS) data:

- **HOOPS AI**, which authors the per-part SCS on the server. The export runs
  through its CAD access layer (`hoops_ai.cadaccess.HOOPSTools.exportStreamCache()`,
  called from `viewer.py`); what that layer uses internally is inside the SDK
  and cannot be determined from this repository.
- **libsc**, statically linked into `tools/scs_points` from the
  `HOOPS_Visualize_Web_2026.2.0` SDK, which writes the Shape Map point-cloud
  SCS. This is a separate path from the per-part export above.
- the browser viewer — the `@ts3d-hoops/*` npm packages pinned in
  `web/package.json` / `web/package-lock.json`, which reads SCS in the client.

The first two write SCS and the third reads it. That relationship is not
visible in the code, so a version mismatch does not fail loudly — it surfaces
later as a viewer that cannot load a model. **Upgrade all three together**; never
bump the npm viewer on its own.

### Layout

- `web/` — client source (Vite project). This is what you read and edit.
- `static/app/` — the build output. It is **not part of the repository**; you
  build it locally with `npm ci && npm run build` (see [Setup](#setup) step 3).
  The redist package includes the client that you build.

### Building

Building the client is a **required setup step** — `static/app/` is not part of
this repository, so a fresh clone has no client until you build it. Rebuild after
changing anything in `web/`.

**Why the build output is not in the repository.** `engine.esm.wasm`, and the
viewer code that ends up in the `static/app/assets/` bundle, are content of the
**HOOPS Visualize Web SDK**. Committing them here would make this repository a
redistributor of that SDK. That is the same reason `libsc` has to be obtained
through the proper channel rather than copied around: SDK content reaches you
through your entitlement, not through a public repository. Building locally
routes every user through npm, so each one obtains the SDK packages — and the
license text bundled in them — from the registry. See
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). Third-party components that are
embedded inside the compiled HOOPS binaries (the wasm engine and browser viewer,
and the `libsc` linked into `bin/scs_points`) are not visible in npm metadata;
their acknowledgment texts are in
[`third_party_notices/`](third_party_notices/).

```bash
cd web
npm ci
npm run build
```

This writes the bundle and the viewer engine (`engine.esm.wasm`) into
`static/app/`, which `.gitignore` excludes.

Everything is served locally — the client has **no external CDN dependency** and
works fully offline once built.

### Console handles

The client publishes three handles on `window` for use from the browser
console:

- `window.Communicator` — the `@ts3d-hoops/web-viewer` namespace (`Camera`,
  `Point3`, `Color`, …).
- `window.hwv` — the viewer that became ready most recently.
- `window.hwvMap` — the map's viewer. All three viewers (part, comparison, map)
  share one host element and each disposes whatever is on screen before
  building its own, so `window.hwvMap` is set when the map's viewer becomes
  ready and nulled when it is disposed: it is non-null exactly while the map is
  on screen. That is what makes it useful next to `window.hwv`, which would
  silently be the part viewer once the map is gone.

A `Camera` is a plain value, so one captured while the map was up stays usable
after that viewer is replaced — which is how you compare two settings from the
same viewpoint:

```js
let cam = hwvMap.view.getCamera();   // with the map on screen
// change the Layout from PCA to UMAP -- the map is rebuilt
hwvMap.view.setCamera(cam);          // onto the new map viewer
```

This is a development convenience, not an API: nothing in the app reads these
handles, and they may change without notice.

### License key

If a viewer feature you add needs a browser-side key, put it in
`web/src/config.js`. Keep a real key out of the source tree.

---

## Shape Map point cloud

The projection is PCA by default; installing `umap-learn` makes a UMAP layout
selectable as well (see
[Optional: the UMAP layout](#optional-the-umap-layout)).

The toolbar's **Clustering** selector chooses how the parts are grouped: k-means
(the default), Ward or Average. All three run on the full-dimensional vectors,
so none of them moves a point -- they decide only which cluster a part belongs
to. The
difference is that Ward builds one agglomerative tree and cuts it at the
requested count, so a larger count **subdivides** the clusters a smaller one
produced rather than partitioning the parts afresh. Average linkage (UPGMA)
builds the same kind of tree from the same distances and differs only in the
merge rule: Ward joins whichever pair adds least to the within-cluster
variance, which prefers compact clusters of similar size, while Average joins
on the mean distance between members. Textbook advice is that this makes
Average more tolerant of long thin families that Ward tends to cut in half; on
this data it does not work out that way. Over 7,928 parts cut at a maximum of
2,048 clusters, Ward's catalogue finds 300 families that pass a 0.95 cohesion
threshold and cover 4,098 parts, against Average's 125 families covering 2,013.
More telling is what each leaves unresolved: Ward's largest mixed branch holds
14 parts, Average's holds 237, and the tree shows why -- at the third level
Average still has 7,863 of the 7,928 parts in a single node, which is the
signature of chaining. Average is kept as a selectable option so that the
comparison can be reproduced rather than taken on trust. **Ward is the one to
use.** Measured on 7,928 parts at
2,048 dimensions, the tree takes about 40 seconds to build and is then cut in a
couple of milliseconds at any count, so sweeping the cluster count is faster
under Ward than under k-means once the first build is done. It needs no extra
dependency: `scipy` is already required. Very large indexes are refused rather
than attempted -- the tree needs an N x N distance matrix -- and the cap is
`HOOPS_AI_SHAPE_MAP_WARD_MAX_POINTS` (default 20,000). Drill in follows the
hierarchy while Ward or Average is selected: instead of clustering the subset
afresh, the
server cuts the same tree lower down, so the clusters inside a drill-down are
an exact refinement of the one that was drilled.

`POST /similarity/index/{name}/map` returns the Shape Map as JSON points. To draw
that map in 3D in HOOPS Visualize Web, the server can also serve it as a
point-cloud `.scs`:

```
GET /similarity/index/{name}/map/scs?color_by=cluster
```

The query vocabulary matches `/map` (`n_clusters`, `dims`, `tags`, `tag_mode`,
`tagged`) plus `color_by` (`cluster` or `tag`) for the baked-in colours. The
`.scs` is authored from the same cached map payload `/map` returns, so a point's
index in the file equals its position in the map — that order is the link back to
each part's `file_id`.

The file is produced by a small C++ tool, `scs_points`, built from
[`tools/scs_points/`](tools/scs_points/README.md). The tool requires a HOOPS
Visualize Web license key, so it is **not** part of this repository; build it and
place the executable in
`bin/`. The commands are in
[Setup](#setup) step 4, *Build the Shape Map point-cloud tool*.
When the tool is absent, `/map` keeps working and only `/map/scs`
returns **503**. Generated files are cached under `scs_map/` with an LRU cap set
by `HOOPS_AI_SHAPE_MAP_SCS_CACHE_MAX` (default `32`).

### Working with the map in the client

The toolbar above the map changes what is drawn without changing the search:

- **View** — `Point cloud` draws every mapped part as a point; `3D models`
  replaces the points of the shown categories with the parts themselves, up to
  a budget, so a family can be recognised by shape.
- **Scale** / **True scale** — the size of those models. Each model is normally
  scaled to the same on-screen size, which is what the embedding sees (L2
  normalization discards absolute size and keeps proportion). **True scale**
  draws them at one common scale instead, so real size ratios show.

Right-clicking a legend row or a tree node offers **Pin parts**: every part of
that cluster is drawn as a model, in full, without hiding anything else. A
pinned cluster is taken whole or not at all, so a family is never shown
half-way. **Unpin all** returns to the ordinary spread of representatives.

The **Catalogue** tab cuts the Ward (or Average) tree into families at one cohesion
threshold (`POST /similarity/index/{name}/map/catalogue`) and shows one tile per
family. **Order** sorts the tiles by tree order or by part count — a client-side
sort of what is already loaded, so it costs no request. The ticked families can
be tagged in a single write.

The side panel on the right can be dragged wider or narrower by its left edge,
and collapsed entirely with the arrow on that edge — useful for giving the map
the whole window, or for a screenshot. The edge stays visible while collapsed;
clicking it again restores the panel at the width it had before.

---

## API Endpoints

> **Interactive API reference.** While the server is running, its full,
> always-current parameter list lives in the auto-generated OpenAPI docs:
> **`/docs`** (Swagger UI, try requests in the browser), **`/redoc`** (reference
> view) and **`/openapi.json`** (machine-readable schema) — e.g.
> `http://127.0.0.1:8000/docs`. Every query parameter is documented there with
> its default, allowed range and meaning. The examples below show only the most
> common options (such as `top_k`); endpoints like
> `/similarity/index/{name}/search` and `.../search-assembly` accept many more
> (result filtering, tag filters, assembly-matching tuning). Consult `/docs` for
> the complete set rather than relying on the summaries here.

> **Windows note:** in PowerShell, `curl` is an alias for `Invoke-WebRequest`,
> which does **not** understand the `curl` flags below. Use `curl.exe` (the real
> binary, shipped with Windows 10/11) in every example. On Linux/macOS use
> `curl`. The examples in this section use Linux-style `curl`; the copy-paste
> **Quickstart** just below is provided for both shells.

### Quickstart: build and query an index from the terminal

The redistributable ships with **no index**, so the flow always starts by
creating one, then registering a folder of CAD files into it as a background
job, then searching. Replace the folder path and `my-parts` as needed.

**Windows (PowerShell, `curl.exe`):**

```powershell
$base    = "http://127.0.0.1:8000"
$index   = "my-parts"
$folder  = "C:\cad\parts"          # folder of CAD files (searched recursively)
$workers = 12                       # parallel embedding workers
$timeLimit = 120                    # per-file embedding budget, seconds

# 1. Create the index (409 if it already exists, which is fine)
curl.exe -X POST "$base/similarity/index/create?name=$index"

# 2. Collect CAD files recursively (TMCAD-style trees keep files in subfolders)
#    This list mirrors the server's CAD_ALLOWED_EXTENSIONS (66 types); keep it
#    in sync so no valid CAD file is silently skipped.
$exts = @(".3dm",".3ds",".3dxml",".3mf",".arc",".asm",".catdrawing",".catpart",
          ".catproduct",".catshape",".cgr",".dae",".dgn",".dlv",".dwf",".dwfx",
          ".dwg",".dxf",".exp",".fbx",".glb",".gltf",".hsf",".iam",".ifc",
          ".ifczip",".iges",".igs",".ipt",".jt",".kmz",".mf1",".model",".neu",
          ".nwd",".obj",".par",".pkg",".prc",".prt",".psm",".pwd",".rvt",".sab",
          ".sat",".session",".sldasm",".sldfpp",".sldprt",".step",".stl",".stp",
          ".stpz",".u3d",".unv",".vda",".vrml",".wrl",".x_b",".x_t",".xas",
          ".xmt",".xmt_txt",".xpr",".xv0",".xv3")
$cad = Get-ChildItem $folder -Recurse -File |
       Where-Object { $exts -contains $_.Extension.ToLower() }
"CAD files found: $($cad.Count)"

# 3. Upload each file, collecting the returned file_ids
$fileIds = @(); $i = 0
foreach ($f in $cad) {
    $r = curl.exe -s -X POST "$base/files/upload" -F "file=@$($f.FullName)" | ConvertFrom-Json
    if ($r.file_id) { $fileIds += $r.file_id }
    if (++$i % 250 -eq 0) { "uploaded $i / $($cad.Count)" }
}
"collected file_ids: $($fileIds.Count)"

# 4. Queue a bulk-registration job with workers + time_limit.
#    Write the JSON to a file and post it with --data-binary "@file": passing
#    inline JSON with -d "{...}" lets PowerShell strip the inner double quotes.
#    Building the array by hand also avoids ConvertTo-Json collapsing a
#    single-element array to a scalar (which the API would reject).
$idsJson = ($fileIds | ForEach-Object { $_ | ConvertTo-Json }) -join ','
$payload = "{`"file_ids`":[$idsJson],`"workers`":$workers,`"time_limit`":$timeLimit}"
$tmp = Join-Path $env:TEMP "job_add.json"
Set-Content -Path $tmp -Value $payload -Encoding ascii
$job = curl.exe -s -X POST "$base/similarity/index/$index/jobs/add" `
    -H "Content-Type: application/json" --data-binary "@$tmp" | ConvertFrom-Json
Remove-Item $tmp
$jobId = $job.job_id
"job_id: $jobId"

# 5. Poll until the job leaves queued/running
do {
    Start-Sleep 3
    $st = curl.exe -s "$base/similarity/index/$index/jobs/$jobId" | ConvertFrom-Json
    "$($st.status): $($st.progress.done)/$($st.progress.total) (errors $($st.progress.errors))"
} while ($st.status -eq "queued" -or $st.status -eq "running")

# 6. Read the per-file registration report
curl.exe -s "$base/similarity/index/$index/jobs/$jobId/report"

# 7. Search the index for parts similar to one of the files
curl.exe -X POST "$base/similarity/index/$index/search?top_k=10" -F "file=@$($cad[0].FullName)"
```

**Linux / macOS (bash, `curl` + [`jq`](https://jqlang.github.io/jq/)):**

```bash
base=http://127.0.0.1:8000
index=my-parts
folder=/cad/parts                 # folder of CAD files (searched recursively)
workers=12                        # parallel embedding workers
time_limit=120                    # per-file embedding budget, seconds

# 1. Create the index (409 if it already exists, which is fine)
curl -X POST "$base/similarity/index/create?name=$index"

# 2. Collect CAD files recursively, then upload each, collecting the file_ids
#    The -iname list mirrors the server's CAD_ALLOWED_EXTENSIONS (66 types);
#    keep it in sync so no valid CAD file is silently skipped.
ids=()
while IFS= read -r -d '' f; do
  id=$(curl -s -X POST "$base/files/upload" -F "file=@$f" | jq -r .file_id)
  [ -n "$id" ] && ids+=("$id")
done < <(find "$folder" -type f \( \
  -iname '*.3dm' -o -iname '*.3ds' -o -iname '*.3dxml' -o -iname '*.3mf' \
  -o -iname '*.arc' -o -iname '*.asm' -o -iname '*.catdrawing' \
  -o -iname '*.catpart' -o -iname '*.catproduct' -o -iname '*.catshape' \
  -o -iname '*.cgr' -o -iname '*.dae' -o -iname '*.dgn' -o -iname '*.dlv' \
  -o -iname '*.dwf' -o -iname '*.dwfx' -o -iname '*.dwg' -o -iname '*.dxf' \
  -o -iname '*.exp' -o -iname '*.fbx' -o -iname '*.glb' -o -iname '*.gltf' \
  -o -iname '*.hsf' -o -iname '*.iam' -o -iname '*.ifc' -o -iname '*.ifczip' \
  -o -iname '*.iges' -o -iname '*.igs' -o -iname '*.ipt' -o -iname '*.jt' \
  -o -iname '*.kmz' -o -iname '*.mf1' -o -iname '*.model' -o -iname '*.neu' \
  -o -iname '*.nwd' -o -iname '*.obj' -o -iname '*.par' -o -iname '*.pkg' \
  -o -iname '*.prc' -o -iname '*.prt' -o -iname '*.psm' -o -iname '*.pwd' \
  -o -iname '*.rvt' -o -iname '*.sab' -o -iname '*.sat' -o -iname '*.session' \
  -o -iname '*.sldasm' -o -iname '*.sldfpp' -o -iname '*.sldprt' \
  -o -iname '*.step' -o -iname '*.stl' -o -iname '*.stp' -o -iname '*.stpz' \
  -o -iname '*.u3d' -o -iname '*.unv' -o -iname '*.vda' -o -iname '*.vrml' \
  -o -iname '*.wrl' -o -iname '*.x_b' -o -iname '*.x_t' -o -iname '*.xas' \
  -o -iname '*.xmt' -o -iname '*.xmt_txt' -o -iname '*.xpr' -o -iname '*.xv0' \
  -o -iname '*.xv3' \
  \) -print0)
echo "collected file_ids: ${#ids[@]}"

# 3. Queue a bulk-registration job with workers + time_limit
json=$(printf '%s\n' "${ids[@]}" | jq -R . \
  | jq -s --argjson w "$workers" --argjson t "$time_limit" \
       '{file_ids: ., workers: $w, time_limit: $t}')
job_id=$(curl -s -X POST "$base/similarity/index/$index/jobs/add" \
  -H "Content-Type: application/json" -d "$json" | jq -r .job_id)
echo "job_id: $job_id"

# 4. Poll until the job leaves queued/running
while :; do
  sleep 3
  s=$(curl -s "$base/similarity/index/$index/jobs/$job_id")
  st=$(echo "$s" | jq -r .status)
  echo "$st: $(echo "$s" | jq -r '.progress.done')/$(echo "$s" | jq -r '.progress.total')"
  [ "$st" = queued ] || [ "$st" = running ] || break
done

# 5. Read the per-file registration report
curl -s "$base/similarity/index/$index/jobs/$job_id/report"

# 6. Search the index for parts similar to a query file
curl -X POST "$base/similarity/index/$index/search?top_k=10" -F "file=@/cad/parts/sample.stp"
```

`workers` and `time_limit` are optional — omit them to size workers
automatically (capped by `HOOPS_AI_MAX_WORKERS`) and use the default per-file
budget. See *Bulk registration jobs* below for every option. Note the PowerShell
JSON-via-file idiom above: it avoids two Windows pitfalls (PowerShell stripping
quotes from inline `-d` JSON, and `ConvertTo-Json` collapsing a one-element array).

Other ways to feed a job instead of uploading file by file: upload a single ZIP
of the folder and pass its `zip_file_id`, or — for an operator seeding from a
folder already on the server machine — pass `server_paths` (requires
`HOOPS_AI_ALLOW_SERVER_PATHS=true`).

### File Management

CAD files are uploaded once and addressed afterwards by `file_id`, the SHA-256
hex digest of the file's content. Re-uploading the same content returns the same
`file_id` without re-storing the bytes.

```
POST   /files/upload                 # single file  -> { file_id, filename, already_existed }
POST   /files/upload-batch           # many files in one request
POST   /files/exists                 # which of the given file_ids are already stored
POST   /files/paths                  # record the client folder for files already stored
POST   /files/upload-from-path       # register a server-side CAD file or ZIP by path
GET    /files/                       # list stored CAD files
GET    /files/{file_id}/download     # download the stored CAD file (original filename)
POST   /files/{file_id}/scs          # author an SCS stream cache for the file (for viewing)
DELETE /files/{file_id}              # delete a stored CAD file
```

`GET /files/{file_id}/download` returns the raw CAD bytes under the original
filename so a client can render the model locally (for example in HOOPS Visualize Web).
Unknown ids return **404**.

In every response above, `filename` is **the name the client sent**, not the
name the bytes are stored under. The store names a file `<sha256>_<original
name>`, but that is this server's bookkeeping: no caller is handed it, and no
caller should implement the rule to undo it.

`POST /files/upload-batch` also accepts an optional `relative_paths` field: one
value per file, in the same order as the file parts, holding the folder path the
client uploaded it from (a browser sends `webkitRelativePath`). The web client
always sends it. What the server does with it is deliberately narrow:

- **It is display data only. It never decides where anything is written.** The
  field is attacker-controlled, so a value such as `..\..\Windows\x.stp` or
  `C:\x.stp` would be a directory traversal if it reached a save path. The
  stored name is still derived from the file's SHA-256 and the sanitised
  basename of the multipart filename, exactly as it was before this field
  existed. That is also why the path travels in its own field rather than as
  the multipart filename.
- A value that is absolute, drive-qualified, rooted, contains a `..` segment,
  carries a control character or is implausibly long is **refused whole**, not
  trimmed down. A folder picker never produces one.
- Accepted paths are kept in `DATA_ROOT/upload_paths.json`, keyed by `file_id`,
  several per id: identical bytes share a `file_id`, so the same part sitting in
  two assemblies is exactly what this records.
- Registration reports show these paths. **Files
  uploaded before this existed have none and fall back to their original
  filename**, which is already on disk — no blanks and no `unknown`.

`POST /files/paths` records the same thing for files the server *already*
holds: `{ "paths": { "<file_id>": "<folder path>" } }`, answering
`{ "recorded": n }`. Bulk registration uploads only what `POST /files/exists`
reports as missing, so on a server that already holds the corpus nothing would
be uploaded and no folder would ever be learnt. The client knows the folder
either way, so it sends the mapping for every file it skipped. The values go
through the same check as `relative_paths` — one gate, not two — and a value
that fails it is dropped and left out of `recorded` rather than failing the
request.

**Windows (PowerShell, `curl.exe`):**

```powershell
# Upload, then download by file_id
curl.exe -X POST "http://127.0.0.1:8000/files/upload" -F "file=@C:\path\to\model.stp"
curl.exe -o model.stp "http://127.0.0.1:8000/files/<file_id>/download"
```

**Linux / macOS (bash, `curl`):**

```bash
# Upload, then download by file_id
curl -X POST "http://127.0.0.1:8000/files/upload" -F "file=@/path/to/model.stp"
curl -o model.stp "http://127.0.0.1:8000/files/<file_id>/download"
```

### Shape Comparison and Embeddings

```
POST /similarity/embed      # compute the embedding vector for one file
POST /similarity/compare    # pairwise similarity across a set of files
```

Both accept uploads and/or `file_id` values and use the active embedding model.

### Named Index Management

Build your own persistent indexes and grow them over time.

```
POST   /similarity/index/create               # create an empty index
GET    /similarity/index/list                 # list indexes
POST   /similarity/index/add                   # register parts synchronously
POST   /similarity/index/{name}/search         # search an index for similar parts
POST   /similarity/index/{name}/warm           # pre-warm the search-response cache for a query
POST   /similarity/index/{name}/search-assembly# assembly-to-assembly search
GET    /similarity/index/{name}/stats          # file/body/assembly counts
POST   /similarity/index/{name}/map            # 2D/3D similarity map of the index
GET    /similarity/index/{name}/parts           # list registered parts (+ asset URLs)
DELETE /similarity/index/{name}/parts           # remove parts
DELETE /similarity/index/{name}?confirm=true    # delete the whole index
```

**Windows (PowerShell, `curl.exe`):**

```powershell
# Search a named index (parts), including the query itself
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/search?top_k=10&file_id=<file_id>"

# "Which assemblies contain this sub-assembly?"
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/search-assembly?top_k=10" `
  -F "file=@C:\path\to\assembly.stp"
```

**Linux / macOS (bash, `curl`):**

```bash
# Search a named index (parts), including the query itself
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/search?top_k=10&file_id=<file_id>"

# "Which assemblies contain this sub-assembly?"
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/search-assembly?top_k=10" \
  -F "file=@/path/to/assembly.stp"
```

Removing a part also deletes its `thumbnails/<id>.png` and `scs/<id>.scs`
assets and prunes its tags, so nothing is orphaned.

`POST /similarity/index/{name}/warm` runs the exact `search` path for a query and
discards the hits, leaving only the response cache populated so the first real
request is served warm. The parameters must match the later search **exactly**
(`top_k`, `kind`, `include_self` and the tag filter are all part of the cache
key), otherwise the warm writes a different key that the real search never hits.
It requires the cache to be enabled (`HOOPS_AI_SEARCH_CACHE`); the response
reports `cache_enabled` so a no-op warm is visible.

### Per-part assets (thumbnail / stream cache)

```
GET /similarity/index/{name}/parts/{part_id}/thumbnail   -> image/png
GET /similarity/index/{name}/parts/{part_id}/scs         -> application/octet-stream
```

Serve the per-part PNG thumbnail (for gallery display) or the SCS stream cache
(what the web client loads into the viewer) generated at registration
time. Returns **404** when the asset does not exist. Paths are validated to stay
within `indexes/<name>/`.

**Windows (PowerShell, `curl.exe`):**

```powershell
curl.exe -o thumb.png "http://127.0.0.1:8000/similarity/index/my-parts/parts/<file_id>/thumbnail"
curl.exe -o part.scs  "http://127.0.0.1:8000/similarity/index/my-parts/parts/<file_id>/scs"
```

**Linux / macOS (bash, `curl`):**

```bash
curl -o thumb.png "http://127.0.0.1:8000/similarity/index/my-parts/parts/<file_id>/thumbnail"
curl -o part.scs  "http://127.0.0.1:8000/similarity/index/my-parts/parts/<file_id>/scs"
```

### Shape Map

```
POST /similarity/index/{name}/map   -> 2D/3D similarity map of the index
```

Projects an index's parts into a similarity map. The stored per-body embeddings
are read back from the index (no CAD file is re-embedded), aggregated to one
point per file (mean of that file's bodies, renormalized), projected with PCA
and coloured by FAISS k-means. Similar parts land close together. Coordinates
are normalized into the `[-1, 1]` cube, and the layout is **deterministic** for
a given index and parameters (the k-means seed is fixed), so it is safe to use
as a stable rendering input.

Query parameters:

| Param | Default | Meaning |
|-------|---------|---------|
| `n_clusters` | `0` | Number of k-means clusters. `0` auto-picks `min(max(2, N // 50), 24)` (a point-count heuristic); `-1` scans `k` in `[2, min(24, N-1)]` and picks the elbow of the inertia curve (data-driven, heavier). A positive value forces that exact `k`. Cheap to change for a fixed count (reuses the cached projection). The web UI defaults to `8` and offers an **Optimal** option (`-1`). |
| `dims` | `3` | Projection dimensionality, `2` or `3`. The unused axis is `0`. Cheap to change. |
| `tags` | — | Comma-separated tags. Restricts the map to matching parts and recomputes the layout on that subset only. |
| `tag_mode` | `any` | `any` matches at least one tag, `all` requires every tag. |
| `tagged` | — | `true` = only tagged parts, `false` = only untagged. |
| `include_tags` | `false` | Add each part's tags to every point. Off by default to keep the response small. |

The response carries `points` (one `{file_id, filename, x, y, z, cluster}` per
file) and a server-side `legend` summarising the mapped subset by tag — or by
cluster when the subset has no tags. The first call for an index runs the
projection (a few seconds on a large corpus) and caches it; later calls,
including different `n_clusters` / `dims`, return quickly from the cache. See
`HOOPS_AI_SHAPE_MAP_PREWARM` above to warm it after a registration job.

**Windows (PowerShell, `curl.exe`):**

```powershell
# Default 3D map, auto cluster count
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/map"

# 2D map of tagged parts only, 12 clusters, with per-point tags
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/map?dims=2&n_clusters=12&tagged=true&include_tags=true"
```

**Linux / macOS (bash, `curl`):**

```bash
# Default 3D map, auto cluster count
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/map"

# 2D map of tagged parts only, 12 clusters, with per-point tags
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/map?dims=2&n_clusters=12&tagged=true&include_tags=true"
```

```
POST /similarity/index/{name}/map/catalogue  -> part families cut from the Ward tree
```

Cuts the index's Ward tree into families at one cohesion: every branch is cut at
the first node whose cohesion reaches `threshold` (default `0.95`), and a branch
that never reaches it is returned with `reached=false` rather than dropped, so
the part totals always cover the whole population. `k_max` (default `2048`) is
the depth the frontier is cut from, `min_parts` drops groups smaller than it
from `groups` (what was dropped stays visible in `omitted_by_min_parts`), and
`min_reached_parts` (default `5`) is the size at which a group that met the
threshold counts as *confident* rather than *small* — a classification only, it
never changes which nodes the frontier is made of. `kind`, `tags`, `tag_mode`
and `tagged` select the population exactly as `/map` does.

### Tags

Attach and query free-form tags on registered parts. Writes are optimistic-lock
protected with ETags.

```
GET    /similarity/index/{name}/tags                       # all tags in the index
GET    /similarity/index/{name}/parts/{part_id}/tags       # one part's tags
PUT    /similarity/index/{name}/parts/{part_id}/tags       # replace one part's tags
POST   /similarity/index/{name}/tags/{tag}/parts           # tag many parts at once
DELETE /similarity/index/{name}/tags/{tag}/parts           # untag many parts at once
DELETE /similarity/index/{name}/tags/{tag}?confirm=true    # retire a tag everywhere
```

The server also keeps one **tag vocabulary** for the whole application
(`tag_vocabulary.json` at the data root): the names in use, kept apart from
which parts carry them. A name is added when it is first assigned — in any
index — and stays after the last part carrying it is untagged, so the wording
never has to be retyped from memory. The file is a plain `{"tags": [...]}`
array and can be edited by hand. It is one list rather than one per index
because the wording of a tag is a vocabulary choice, not a property of an
index; offer it after the tags in use so names from other indexes sink to the
bottom of a picker.

```
GET    /similarity/tag-vocabulary                          # names to choose from
POST   /similarity/tag-vocabulary                          # register names ahead of use
DELETE /similarity/tag-vocabulary/{tag}                    # drop a name (parts keep the tag)
```

Search endpoints accept a tag filter so results can be narrowed to tagged parts.

The API supports `tag_mode=all` (AND across multiple tags). This reference UI
uses a single-select filter for simplicity and therefore always sends `any`;
wire your own UI to `tag_mode` if you need AND filtering.

### Context (production records)

Attach production records — material, process, cost, lead time and so on — to
registered parts, and predict the missing records of a new part from those of
its nearest neighbours. Like tags, records live in a per-index sidecar
(`indexes/<name>/context.json`), not in the `.faiss` file: writing them into the
vector store would call `save()`, move the index mtime and force a ~19.3 s
assembly-matcher rebuild on the next search. Reads carry an `ETag`; writes take
an optional `If-Match`.

The API below is always available. The web client's **Estimate** panel, which
drives it, is demo-only and appears only when `HOOPS_AI_DEMO` is set: the
predictions are worth reading once an index carries real production records, and
the demo runs on fabricated ones.

```
POST   /similarity/index/{name}/predict-context               # predict fields for a query part
GET    /similarity/index/{name}/context                       # which fields exist, and how many parts use each
GET    /similarity/index/{name}/parts/{part_id}/context       # one part's record
PUT    /similarity/index/{name}/parts/{part_id}/context       # replace one part's record
```

`predict-context` takes a JSON body: `file_id` (a previously uploaded/registered
part), the `keys` to predict, and optional `top_k` (nearest neighbours to poll),
a tag filter, a `query_context` of already-known fields, and `neighbor_ids`. The
query part is searched exactly as `/search` searches, the self hit is dropped,
the `top_k` nearest that carry a record are aggregated, and one value is returned
per key. When `neighbor_ids` is given, the neighbours are pinned to exactly those
ids (the web client sends the parts shown in its Results tab, already sliced by
its similarity threshold and tag filter); `top_k` should then match the search
that produced them so the server's pool still contains them, and the tag filter
and `top_k` cap are not re-applied. The response reports `candidates_examined`,
`hits_considered`, `hits_with_context`, `coverage`, `predictions`, and the
evidence `hits`.

**Windows (PowerShell, `curl.exe`):**

```powershell
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/predict-context" `
  -H "Content-Type: application/json" `
  -d '{"file_id":"<id>","keys":["Material","UnitCost","LeadTimeDays"],"top_k":50}'
```

**Linux / macOS (bash, `curl`):**

```bash
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/predict-context" \
  -H "Content-Type: application/json" \
  -d '{"file_id":"<id>","keys":["Material","UnitCost","LeadTimeDays"],"top_k":50}'
```

Aggregation is tunable per index with an optional `indexes/<name>/context_rules.json`
(categorical vs. numeric rules, per-key overrides, named normalizers only — never
code). Unknown options are ignored rather than failing a prediction.

### Bulk registration jobs

Register large corpora asynchronously. A client can hash its folder locally
(`POST /files/exists`), upload only what is missing, then submit a job.

```
POST   /similarity/index/{name}/jobs/add            # submit a bulk-registration job
GET    /similarity/index/{name}/jobs                # list jobs for an index
GET    /similarity/index/{name}/jobs/{job_id}       # poll a job
GET    /similarity/index/{name}/jobs/{job_id}/report# per-file registration report
DELETE /similarity/index/{name}/jobs/{job_id}       # cancel a job
GET    /similarity/embedding-plan?file_count=N      # the values a job would use
```

`GET /similarity/embedding-plan` resolves, without creating anything, the
`workers` and `time_limit` a job over `file_count` files would run with: a
`first_pass` block, a `retry_pass` block (fewer workers, a longer budget), and
for each a `basis` naming the figures the choice came from (`cap`, `cores`,
`cores_measured`, `ram`, `files`). Its `time_limit` is always the effective
budget — never the `0` that means "pass nothing and leave the SDK's own default
in place". The web client uses it to pre-fill its two fields; the sizing rule
itself stays on the server, so there is only ever one copy of it.

`file_count` is optional. Omit it and the answer is the ceiling the host alone
allows: every other input (the cap, the physical core count, free RAM) is
settled before any file is chosen, and the file count can only lower the
result. `file_count` and `basis.files` then come back `null`, which says the
count was not one of the figures the minimum was taken over.

`POST .../jobs/add` takes a JSON body. Supply **one** input source plus optional
tuning:

| Field | Type | Meaning |
|---|---|---|
| `file_ids` | `string[]` | `file_id`s of CAD files already uploaded to the store (the standard route). |
| `zip_file_id` | `string` | `file_id` of an already-uploaded ZIP archive to expand and register. |
| `server_paths` | `string[]` | Paths **on the server machine**. Operator seeding only; returns **403** unless `HOOPS_AI_ALLOW_SERVER_PATHS=true`. |
| `workers` | `int` | Parallel embedding workers. Omit to size automatically from CPU cores and free RAM (capped by `HOOPS_AI_MAX_WORKERS`, default 12). |
| `time_limit` | `int` | Per-file embedding budget in seconds. Omit to use `HOOPS_AI_EMBED_TIME_LIMIT`, else the SDK's 120 s. Files that exceed it are reported with `retryable=true`. |
| `selected_count` | `int` | How many files the caller actually picked. Display only — see below. |
| `duplicate_paths` | `object` | `{ "<file_id>": ["<relative path>", ...] }` for the picked files that turned out to be byte-identical to each other. Display only — see below. |

**Why a folder of 10,897 files registers 10,250.** A `file_id` is the SHA-256 of
the file's bytes, so two byte-identical CAD files are one entry; the copies have
nowhere to go. The two optional fields above let the job say so. The web client
fills them in from the selection it hashed, and the report then opens with

```
# Selection: selected=10897 submitted=10250 collapsed=647
```

and lists the relative paths that collapsed under `[DUPLICATES]`. **The figures
describe that one submission.** No other job, index or previously stored file is
consulted, and a job that sends neither field simply reports no selection. The
paths go through the same refuse-don't-repair sanitiser as uploads and remain
display data: they never reach the filesystem.

The job runs one embedding pass over all inputs and returns immediately with a
`job_id`; poll `GET .../jobs/{job_id}` for `progress` and, when finished, a
`report_url`. Jobs run one at a time by default (`HOOPS_AI_JOB_MAX_CONCURRENCY`),
so a job may sit in `queued` for a while. Statuses are `queued` → `running` →
`done` / `failed` / `canceled`.

**Windows (PowerShell, `curl.exe`):**

```powershell
# Register already-uploaded files, 6 workers, 300 s per file
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/jobs/add" `
  -H "Content-Type: application/json" `
  -d '{"file_ids":["<id1>","<id2>"],"workers":6,"time_limit":300}'

# Register an uploaded ZIP of a folder
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/jobs/add" `
  -H "Content-Type: application/json" `
  -d '{"zip_file_id":"<zip_file_id>"}'
```

Single-quote the JSON so PowerShell passes it to `curl.exe` verbatim. For a
payload built from variables (e.g. a long `file_ids` array), write it to a file
and post it with `--data-binary "@file"` instead — see the Quickstart above.

**Linux / macOS (bash, `curl`):**

```bash
# Register already-uploaded files, 6 workers, 300 s per file
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/jobs/add" \
  -H "Content-Type: application/json" \
  -d '{"file_ids":["<id1>","<id2>"],"workers":6,"time_limit":300}'

# Register an uploaded ZIP of a folder
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/jobs/add" \
  -H "Content-Type: application/json" \
  -d '{"zip_file_id":"<zip_file_id>"}'
```

Failed files are annotated with a `retryable` flag. For a large corpus, run the
two-pass strategy: submit everything with many `workers` and a short
`time_limit`, then resubmit the failures whose `retryable` is true as a second
job with few `workers` and a long `time_limit`. Non-retryable failures are
deterministic CAD errors that will never succeed, however long the budget.
Cancellation only takes effect **before** a job starts embedding, so split a
large corpus into several jobs to keep cancellation and commit granularity under
your control.

In the browser, **Register…** runs both passes for you. Opening the dialog
already fills **Workers** and **Time limit** with the numbers this host allows,
and picking files refines them with the file count (a small selection lowers
the worker count, which is correct). The values come from
`GET /similarity/embedding-plan`; hovering either field shows what the worker
count was derived from, and says so plainly while no file count is known yet.
Edit a value to override it, or
clear a field to send nothing and let the server size that half of the run —
which is what every non-browser caller does. When the job finishes with
retryable failures, a **Resubmit** button appears and the two fields change to
the retry-pass values, so the second pass is one click rather than a lookup.

---

## Data directories

| Directory | Contents | Cleared by the server? |
|---|---|---|
| `uploads/` | The CAD store — the only copy of every registered file, keyed by `file_id`. | Never. |
| `indexes/` | Named FAISS indexes plus per-part `thumbnails/` and `scs/` assets, and `tags.json`. | Never (per-part assets are removed with their part). |
| `embeddings_cache/` | Cached embeddings. | Swept by TTL (`HOOPS_AI_EMBEDDINGS_CACHE_TTL_DAYS`, default keep forever). |
| `out/` | Transient result images and shape maps. | Swept by TTL (`HOOPS_AI_OUT_TTL_HOURS`, default 24 h). |
| `jobs/` | Registration job records. | Bounded by `HOOPS_AI_JOB_TTL_DAYS` / `HOOPS_AI_JOB_MAX_RECORDS`. |
| `models/` | Embedding checkpoints. This is where an operator drops their own `*.ckpt`; it is searched before the default bundled in `<APP_ROOT>/models/` and shadows it (see [Custom (fine-tuned) models](#custom-fine-tuned-models)). | Never (the server only reads it). |
| `scs_map/` | Cached Shape Map point-cloud `.scs` files. | LRU-capped by `HOOPS_AI_SHAPE_MAP_SCS_CACHE_MAX` (default 32). |
| `tag_vocabulary.json` | The tag names available to choose from, shared by every index (a file, not a directory). | Never (names are only removed on request). |
| `upload_paths.json` | The folder each file was uploaded from, as the client reported it — display data only, never used to build a path (a file, not a directory). | Never. |

All of these live under a single **data root**. When `HOOPS_AI_DATA_DIR` is not
set, the data root defaults to `<app>/data` — a `data/` subdirectory of the
application — so the runtime directories (`uploads/`, `indexes/`, …) stay
separate from the source tree and a fresh checkout works with no configuration.
Set `HOOPS_AI_DATA_DIR` to place the data on another volume (see *Deployment*
below).

---

## Deployment

This server can be packaged for redistribution to end users who do **not** have
HOOPS AI installed. The build scripts bundle the SDK, the compiled license shim,
a default embedding model, and only the Python packages the server actually
imports.

### Package layout

The build script writes two things into its output directory: the staging
directory it assembles, and a zip of that directory. The zip is what you hand
over; the staging directory is kept so you can inspect what went in.

```
<out-dir>/
├── redist_package_<yyyyMMdd-HHmmss>.zip     # the artifact to ship (expands to redist_package/)
└── redist_package/                          # staging directory, kept for inspection
    ├── *.py                                 # every server module in the repository root
    │                                        #   (main.py, core.py, search.py, assembly.py, …)
    ├── requirements.txt, start_server.sh
    ├── routers/  vendor/  static/           # static/ carries the built web client (static/app/)
    ├── bin/
    │   └── scs_points.exe                   # or scs_points on Linux; Shape Map point-cloud tool
    ├── hoops_license_shim.<abi-tag>.pyd     # or .so on Linux
    ├── .env                                 # template; all settings optional, no license key
    ├── models/
    │   └── <default>.ckpt                   # bundled default embedding checkpoint
    ├── .venv/
    │   └── Lib/site-packages/ (win) | lib/python3.12/site-packages/ (linux)
    │                                        # only the traced HOOPS AI + dependency packages
    │                                        # (site-packages only — NO python interpreter)
    ├── samples/                             # optional sample CAD files
    ├── third_party_notices/                 # notices for components inside the HOOPS binaries
    ├── DISCLAIMER.txt                       # sample/PoC disclaimer
    ├── LICENSE                              # MIT license for the code written here
    ├── THIRD-PARTY-NOTICES.md               # web client npm dependency license notices
    └── THIRD_PARTY_MANIFEST.txt             # bundled package origin / redistribution category
```

The server modules are **collected**, not listed: the script copies every `*.py`
in the repository root except an explicit exclusion list (test/scratch files and
the license-shim source, which holds the key in plain text). A module added by a
later refactor is packaged without touching the script, and
`tests/test_redist_packaging.py` fails if anything the server imports would be
left out.

There are two `models/` directories and they have different roles. `<APP_ROOT>/models/`
— the one inside the package above — holds the **default checkpoint bundled by
the build script**. `<DATA_ROOT>/models/` (by default `<app>/data/models/`) is
where an **operator drops their own checkpoint**; it is searched first and
shadows the bundled one, and the server logs when it does. See *Custom
(fine-tuned) models* below.

`main.py` prepends the bundled `.venv` site-packages to `sys.path`, so the server
imports the bundled HOOPS AI without a system install. In a developer checkout
that directory is absent and your own environment is used instead.

The package does **not** ship a Python interpreter — only the site-packages. The
target machine must have a **Python 3.12** runtime (the same minor version the
bundled packages and the license shim were built for; the native extensions and
the ABI-tagged `.pyd`/`.so` will not load on a different minor version).

The bundled `.venv` carries only the **traced HOOPS AI SDK and its native
dependencies** (`hoops_ai`, `faiss`, `torch`, …). The lightweight web-server
dependencies listed in `requirements.txt` (`fastapi`, `uvicorn`,
`python-multipart`, `scipy`) are **not** bundled and must be installed once into
the Python 3.12 runtime that will run the server. `main.py` then imports the web
stack from the runtime and the SDK from the bundled `.venv`.

```powershell
# Windows: use any Python 3.12 on the machine (e.g. the py launcher)
cd redist_package
py -3.12 -m pip install -r requirements.txt   # one-time: install web-server deps
py -3.12 main.py --port 8000
```

```bash
# Linux: create a Python 3.12 venv for the web-server deps and run from it.
# On Ubuntu 24.04 the system Python is externally managed (PEP 668) and often
# ships without pip/venv, so a venv is the clean path. If venv creation fails
# with "No module named venv/pip": sudo apt install -y python3-pip python3.12-venv
cd redist_package
python3.12 -m venv .webvenv
source .webvenv/bin/activate
pip install -r requirements.txt              # one-time: install web-server deps
python main.py --port 8000
#
# Headless server (no display): HOOPS needs an OpenGL context, so running
# main.py directly segfaults ("HOOPS Converter failed with exit code -11").
# Start under a virtual display via start_server.sh instead (install the deps
# once):
#   sudo apt-get install -y xvfb libgl1 libglu1-mesa libxrender1 libxext6 libsm6
#   HOOPS_AI_VENV="$PWD/.webvenv" ./start_server.sh --port 8000
```

The bundled SDK `.venv` never needs to be activated: `main.py` adds its
`.venv/.../site-packages` to `sys.path` itself, so `import hoops_ai` resolves
against the bundle while `import fastapi` resolves against the runtime (on
Windows the system Python 3.12; on Linux the `.webvenv` above).

### Configuration (`.env`)

Every setting is **optional**. `.env` is where an installer overrides a default;
with no `.env` at all the server still starts with derived defaults. Precedence
is `os.environ` > `.env` > derived default — except `HOOPS_AI_LICENSE`, which is
read from `.env` only and never from the process environment.

| Variable | When unset, it is derived as… |
|---|---|
| `HOOPS_AI_DATA_DIR` | `<app>/data` (the `uploads/ indexes/ …` runtime directories live under `data/`, separate from the source). |
| `HOOPS_AI_SDK_DIR` | unset; bundled checkpoints are not used and named indexes under the data root still work. |
| `HOOPS_AI_EMBEDDINGS_MODEL_NAME` | the single `*.ckpt` found in `<DATA_ROOT>/models/`, else in `<APP_ROOT>/models/`. |
| `HOOPS_AI_LICENSE` | supplied by the compiled shim; a development-only `.env` fallback exists. |

At startup the server logs, at INFO, which value came from where, the resolved
data root, and the resolved embedding checkpoint's absolute path and size — use
these lines to diagnose a deployment.

### Data on another volume

Set `HOOPS_AI_DATA_DIR` to a writable directory on the target volume. The server
checks the data root is writable at startup and stops with a clear error if it is
not (for example when the app is installed under `Program Files` and no data dir
is set), rather than failing on the first upload. Existing data is never moved or
deleted; to reuse an existing store, point `HOOPS_AI_DATA_DIR` at it.

### Custom (fine-tuned) models

Drop a single `*.ckpt` into `<DATA_ROOT>/models/` to override the bundled default
in `<APP_ROOT>/models/`. When the data-root model shadows the bundled one, the
server logs that fact. Keep exactly one `*.ckpt` per directory: two or more in
the same directory is a deterministic startup error listing the candidates.

> Model/index consistency: each index records the identity of the checkpoint it
> was built with (SHA-256, size, path, embedding dimension) in its `index.json`,
> and every record carries the checkpoint hash. This release only *records* that
> identity; it does not yet warn on a mismatch. Confirm the logged checkpoint
> path/size matches the model an index was built with before swapping models.

### Building the package (operator steps)

The compiled license shim must exist first (see *License key handling*), the web
client must be built (`cd web; npm ci; npm run build`) and `bin/scs_points[.exe]`
must be present — the scripts stop if either is missing. On Linux, `zip` must be
installed; the script checks for it before it starts copying, so a multi-gigabyte
build is never thrown away at the last step. Then trace the used modules and
build the package:

```powershell
# Windows PowerShell
$env:HOOPS_AI_HOME = "C:\SDK\HOOPS_AI\V1.1"
$env:HOOPS_AI_LICENSE = '<your-key>'   # single quotes: PowerShell does not expand $
python tools/trace_modules.py `
    --cad-file C:\path\to\sample.step `
    --embed-ckpt "$env:HOOPS_AI_HOME\packages\trained_ml_models\ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" `
    --out C:\temp\used_top_level_modules.txt
.\tools\build_redist_package.ps1 `
    -ModulesFile C:\temp\used_top_level_modules.txt `
    -Ckpt "$env:HOOPS_AI_HOME\packages\trained_ml_models\ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" `
    -OutDir C:\temp\redist_out
```

```bash
# Linux (bash)
export HOOPS_AI_HOME="$HOME/SDK/HOOPS_AI/V1.1"
export HOOPS_AI_LICENSE='<your-key>'   # single quotes: the shell does not expand $
python3 tools/trace_modules.py \
    --cad-file /path/to/sample.step \
    --embed-ckpt "$HOOPS_AI_HOME/packages/trained_ml_models/ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" \
    --out /tmp/used_top_level_modules.txt
tools/build_redist_package.sh \
    --modules-file /tmp/used_top_level_modules.txt \
    --ckpt "$HOOPS_AI_HOME/packages/trained_ml_models/ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" \
    --out-dir /tmp/redist_out
```

The build scripts require the compiled shim and refuse to stage a `.env` that
contains a `HOOPS_AI_LICENSE` line, so a plaintext key can never ship in the
package. The run ends with `<out-dir>/redist_package_<yyyyMMdd-HHmmss>.zip` —
that is the artifact to ship. The timestamp means two builds never silently
overwrite each other, and the `redist_package/` staging directory is kept next to
the zip so you can check what went in. (On Windows PowerShell 5.1,
`Compress-Archive` cannot handle a member file larger than 2 GB; if compression
fails for that reason the staging directory is still complete — re-run under
PowerShell 7+ or compress it with your own archiver.) The
`THIRD_PARTY_MANIFEST.txt` lists bundled packages; redistribution of Tech Soft 3D
packages must be confirmed individually under your TPA.

`run_traced_workload()` in `tools/trace_modules.py` must exercise every feature
the server can invoke — including `predict-context`, which imports
`hoops_ai.ml.context_layer` — so the package bundles the modules those features
need; if you change `run_traced_workload()`, re-trace and rebuild the package.

### Python version and platform

The bundled `.venv` and the license shim are built for one Python minor version
and platform. Upgrading the Python runtime (e.g. 3.12 → 3.13) makes the ABI-tagged
license shim un-importable — the server then reverts to the `.env` fallback with a
warning — so **rebuild the shim and re-trace/rebuild the package after any Python
upgrade**. Linux and Windows each need their own build.

---

## Operations

### Memory sizing

The server keeps several things resident in process memory. Size a host from the
read side first, then add the write cache:

| Component | Approx. resident size (a `tmcad`-class index: 10,230 files / 39,377 bodies / dim 2048) |
|-----------|----------------------------------------------------------------------------------------|
| Named searcher(s) | ~308 MB each (`index.faiss`), up to 2 held |
| Assembly matcher | ~616 MB (searcher + an L2-normalised copy of every body vector) |
| Shape Map caches | ~84 MB (one aggregate matrix; results are a few MB) |
| Embedding model | a few hundred MB (the single checkpoint, loaded once) |
| **Read-side subtotal** | **~1.9 GB** |
| Named-index **write** store | up to `HOOPS_AI_INDEX_CACHE_MAX_MB` (default **1024 MB**, ≈ three `tmcad` indexes) |

So a single `tmcad`-class index needs roughly **3 GB** in steady state, of which
about **1 GB is the write cache** and is tunable. On a memory-constrained host
lower `HOOPS_AI_INDEX_CACHE_MAX_MB` (the evicted index is reloaded from disk on
the next create/add/remove, a few seconds — negligible inside a minutes-long
registration job); raise it when several indexes are updated concurrently.

Both in-process caches are bounded by **memory**, not entry count, because index
sizes vary widely between deployments:

- `HOOPS_AI_INDEX_CACHE_MAX_MB` (default 1024) — named-index write store.
- `HOOPS_AI_EMBED_CACHE_MAX_MB` (default 256) — embedding cache (disk-backed, so
  eviction reloads rather than recomputes).

Every eviction is logged at `INFO` under `[CACHE]` (which index/entry was
dropped and the new resident total), so a memory or latency problem is
diagnosable from the log alone.

### Startup warmup

The first request that touches an index pays a one-time build cost: the named
searcher (~0.3 s), the Shape Map SVD (~15 s) and the assembly matcher k-means
(~20 s). After a restart the first user pays it again. Startup warmup does these
builds in the background so that user does not.

It is **opt-in**. With nothing configured the server does nothing extra at
startup (no surprise "started but the CPU is pegged for a minute" on a
redistributed deployment). To enable it, name the indexes:

```dotenv
# .env
HOOPS_AI_WARMUP_INDEXES=tmcad
HOOPS_AI_WARMUP_MAX=1
```

The map stages warm exactly the population the web client asks for on a fresh
page (parts only, 3-D, coloured by cluster, PCA projection, tags included, and
the cluster count the index's own part count implies), because the map is
cached per parameter set: warming anything else warms a cache key no request
will hit. Two of the three clusterings are warmed by default — k-means and Ward —
since each is a separate cache key. Average is accepted by
`HOOPS_AI_WARMUP_CLUSTERING` but left cold: it is there to be compared
against Ward rather than used daily, so a third build at every startup is
not worth paying for, and the user who selects it waits once instead. The
finest cluster count Ward's tree descends to is
worked out per index from its part count, about 32 parts to a cluster (tmcad's
7,928 single parts give 256). That is the same number `/index/{name}/stats`
hands the client as `default_clusters`, for both k-means' **Clusters (k)** and
Ward's **Max clusters**, so what the user opens is what was warmed.

`HOOPS_AI_WARMUP_STAGES` narrows the work by picking the stages — `assembly`,
`map`, `scs`, `searcher`, all of them when unset — which matters because the
assembly matcher is the slowest of the four:

```dotenv
# Skip assembly search on a deployment that does not use it
HOOPS_AI_WARMUP_STAGES=map,scs,searcher
```

The web client's Shape Map opens on one of the warmed clusterings: it asks
`/client-config` for the warm list at startup and keeps its own remembered
choice when that choice is in the list, falling back to the first entry when it
is not. Either way a user who opens the map gets a clustering that was warmed
rather than a cold build. The remembered choice stands on its own only when
nothing is warm (warmup is off, or its stages exclude `map`).

Warmup runs on a background thread, so the server **accepts requests
immediately**; anything that arrives before warmup finishes is served cold, as
before. It shares the same build locks as normal requests, so a concurrent
request never triggers a second build of the same thing, and a failure (e.g. a
corrupt index) is logged and skipped without taking the server down. Legacy
(schema v1) indexes only warm the searcher. Progress is logged under `[WARMUP]`:

```
[WARMUP] warming 1 index(es): tmcad
[WARMUP] 'tmcad' searcher ready in 0.31s
[WARMUP] 'tmcad' shape map (kmeans) ready in 14.8s
[WARMUP] 'tmcad' assembly matcher ready in 21.2s
[WARMUP] complete in 36.4s
```

---

## License

The code in this repository is released under the MIT License; see
[`LICENSE`](LICENSE) for the full text.

The MIT License covers only the code written by this repository's contributors.
It does **not** relicense any third-party or HOOPS SDK component. The bundled and
depended-on third-party components, and the HOOPS SDK components (HOOPS AI, HOOPS
Exchange, and HOOPS Visualize Web including its authoring libraries),
remain under their own terms; those are recorded in
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md) and
[`third_party_notices/`](third_party_notices/).

Whether you may redistribute any of the HOOPS SDK components, and in what form,
depends on your own Technology Partner Agreement — see the
*Status: sample / proof of concept* section at the top of this README.
