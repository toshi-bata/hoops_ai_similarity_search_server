// API client for the HOOPS AI Similarity Search Server.
//
// This is the first file an ISV should read: every server call the client
// makes is defined here, with its endpoint and arguments in one place. Nothing
// else in the client calls `fetch` directly.
//
// Only the endpoints used by this slice (index list + parts + per-part assets)
// are wrapped. Add new endpoints here, one function each, as later slices need
// them.

import { SERVER_BASE, SEARCH_TOP_K } from './config.js';

// How many file_ids to send per POST /files/exists. The server rejects a
// larger batch with 422; its own limit defaults to 1000 (storage.py,
// _EXISTS_MAX_IDS_DEFAULT) but an operator can lower it with
// HOOPS_AI_EXISTS_MAX_IDS, so this stays well under the default.
const EXISTS_CHUNK = 500;

// Thrown for any non-2xx response. `status` is the HTTP code and `detail` is
// the server's human-readable message (FastAPI returns it in a "detail"
// field). The UI surfaces both, so server errors like 404/409/412/422 reach
// the user instead of being swallowed.
export class ApiError extends Error {
  constructor(status, detail) {
    super(`${status}: ${detail}`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

// Build a full URL for a server path, appending defined query parameters.
function buildUrl(path, query) {
  const url = new URL(SERVER_BASE + path, window.location.origin);
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined && value !== null && value !== '') {
        url.searchParams.set(key, String(value));
      }
    }
  }
  return url.toString();
}

// Low-level request helper. Every other function goes through this so error
// handling and JSON parsing live in exactly one place. Pass `body` for a JSON
// request or `form` (a FormData) for a multipart upload; never both. `headers`
// adds request headers (e.g. If-Match, X-Client-Id for tag writes).
//
// By default it returns the parsed JSON body (or null for 204). Pass
// `wantEtag: true` to get `{ data, etag }` instead, where `etag` is the
// response's ETag header -- the value to send back as If-Match on the next
// write. The tag endpoints use this for optimistic concurrency.
async function request(path, { method = 'GET', query, body, form, headers, wantEtag = false, rawText = false } = {}) {
  const options = { method, headers: { ...(headers || {}) } };
  if (form !== undefined) {
    // Let the browser set the multipart Content-Type (with its boundary).
    options.body = form;
  } else if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(buildUrl(path, query), options);
  } catch (networkError) {
    // fetch only rejects on network failure (server down, DNS, CORS).
    throw new ApiError(0, `Cannot reach the server: ${networkError.message}`);
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      if (data && data.detail) detail = data.detail;
    } catch {
      // Body was not JSON; keep the status text.
    }
    throw new ApiError(response.status, detail);
  }

  const etag = response.headers.get('ETag');
  if (rawText) {
    const text = response.status === 204 ? null : await response.text();
    return wantEtag ? { data: text, etag } : text;
  }
  const data = response.status === 204 ? null : await response.json();
  return wantEtag ? { data, etag } : data;
}

// GET /client-config -> { demo }
// The server's own settings that this client must act on. Fetched once at
// startup (see main.js); everything else the client knows is baked in at build
// time by config.js.
export function getClientConfig() {
  return request('/client-config');
}

// GET /similarity/index/list ->
//   [{ name, count, last_modified, model, warm }]
// `warm` is true when the index is a startup-warmup target
// (HOOPS_AI_WARMUP_INDEXES); the client prefers it as the initial selection.
export function listIndexes() {
  return request('/similarity/index/list');
}

// POST /similarity/index/create?name=... -> { name, count, dim, model }.
// Creates a new empty named index. The embeddings model is NOT chosen here:
// the server uses its own active model (the default-model setting). The name
// must match ^[a-z0-9_-]{1,64}$. Returns 409 if an index with that name
// already exists and 422 if the name is invalid.
export function createIndex(name) {
  return request('/similarity/index/create', { method: 'POST', query: { name } });
}

// DELETE /similarity/index/{name}?confirm=true -> { name, ... }. Deletes a
// named index and all of its on-disk data. The server requires confirm=true
// (409 without it), so the caller must confirm with the user first. Returns
// 404 when the index does not exist.
export function deleteIndex(name, { confirm = false } = {}) {
  return request(`/similarity/index/${encodeURIComponent(name)}`, {
    method: 'DELETE',
    query: { confirm },
  });
}

// DELETE /similarity/index/{name}/parts?part_ids=<comma-separated> ->
// { name, removed, index_count }. Removes registered parts (by their file_ids)
// from a named index and drops their on-disk assets and tag entries. Returns
// 404 when the index does not exist, and 422 when no ids are supplied. Unlike
// deleteIndex there is no confirm flag on the server, so the caller confirms
// with the user first.
export function removeParts(name, partIds) {
  const ids = (partIds || []).map((id) => String(id)).filter(Boolean);
  return request(`/similarity/index/${encodeURIComponent(name)}/parts`, {
    method: 'DELETE',
    query: { part_ids: ids.join(',') },
  });
}

// GET /similarity/index/{name}/stats -> index statistics (used for totals).
export function getIndexStats(name) {
  return request(`/similarity/index/${encodeURIComponent(name)}/stats`);
}

// GET /similarity/embedding-plan -> the worker count and per-file budget a
// registration job would really use for `fileCount` files, for both the first
// pass and the retry pass. The sizing rule lives on the server and is NOT
// mirrored here: it is re-measured per host (cores, free RAM) and has changed
// several times. Returns { file_count, first_pass, retry_pass } where each pass
// is { workers, time_limit, basis: { cap, cores, cores_measured, ram, files,
// model_mb } }. time_limit is always an effective value, never 0.
export function getEmbeddingPlan(fileCount) {
  return request('/similarity/embedding-plan', {
    query: { file_count: Math.max(0, Number(fileCount) || 0) },
  });
}

// GET /similarity/index/{name}/duplicates -> the geometries in the index that
// were registered under more than one original filename. file_ids are content
// hashes, so these are NOT duplicate index entries (there is one entry each);
// only the filenames the copies arrived with differ. Returns { name, total,
// extra_copies, offset, limit, items: [{ file_id, names }] }, where total and
// extra_copies cover the whole index regardless of the page.
export function getIndexDuplicates(name, { offset = 0, limit = 500 } = {}) {
  return request(`/similarity/index/${encodeURIComponent(name)}/duplicates`, {
    query: { offset, limit },
  });
}

// GET /similarity/index/{name}/parts -> paginated file-level listing.
// Returns { total, offset, limit, items: [{ id, filename, kind, bodies,
// thumbnail_url, scs_url, registered_at, tags }] }.
export function listParts(name, { offset = 0, limit = 60, kind, tags, tagMode, tagged } = {}) {
  return request(`/similarity/index/${encodeURIComponent(name)}/parts`, {
    query: { offset, limit, kind, tags, tag_mode: tagMode, tagged },
  });
}

// Collect the file ids of every part matching a tag filter, paging through the
// /parts listing until all of them are gathered. Used by the Results tab's
// results filter: search hits carry only the vector-store metadata (no tags,
// which live in a separate server-side sidecar), so the client resolves tag
// membership here and intersects it with the shown hits. `tags` is a single tag
// name; `tagged` true restricts to any-tagged parts. Returns an array of ids.
export async function listPartIdsByTag(name, { tags, tagged } = {}) {
  const pageSize = 2000; // the server's maximum page size
  const ids = [];
  let offset = 0;
  for (;;) {
    const page = await listParts(name, {
      offset,
      limit: pageSize,
      tags,
      tagged,
    });
    for (const item of page.items || []) ids.push(item.id);
    offset += pageSize;
    if (offset >= (page.total || 0) || (page.items || []).length === 0) break;
  }
  return ids;
}

// Absolute URL of a part's PNG thumbnail (for an <img> src). The listing
// already returns thumbnail_url; this helper is here for completeness and for
// callers that only have the ids.
export function thumbnailUrl(name, partId) {
  return buildUrl(
    `/similarity/index/${encodeURIComponent(name)}/parts/${encodeURIComponent(partId)}/thumbnail`,
  );
}

// Absolute URL of a part's SCS stream cache (passed to the viewer as its
// endpointUri).
export function scsUrl(name, partId) {
  return buildUrl(
    `/similarity/index/${encodeURIComponent(name)}/parts/${encodeURIComponent(partId)}/scs`,
  );
}

// -- Shape Map --------------------------------------------------------------
//
// The Shape Map projects an index's parts into a 2D/3D scatter. Two calls make
// it up, and they MUST share the same parameters so their point orders line up:
//   - shapeMap()   POST /map  -> the point array (file_id, x, y, z, cluster,
//                  optional tags) plus a server-computed legend. This drives
//                  the info panel: a viewer pick is resolved back to a point in
//                  `points` by nearest world position within the picked node's
//                  group range, not by getPointId() -- see shape-map.js's
//                  resolvePoint() for why.
//   - mapScsUrl()  GET  /map/scs -> the same points baked into a point-cloud
//                  SCS the viewer loads by URL. Only `color_by` differs from
//                  /map; it changes the baked colours, never the point order.

// POST /similarity/index/{name}/map -> { index, count, clusters, dims,
// points: [{ file_id, filename, x, y, z, cluster, tags? }], legend }.
// `includeTags` adds each part's tags to every point (used for the info panel);
// it does not change the point order. n_clusters=0 lets the server pick.
// The endpoint also accepts n_clusters=-1, meaning "scan a range of k and keep
// the best-scoring one". This client never sends it: the "Optimal" option was
// removed from the UI because the elbow of the inertia curve answers a
// different question from "is this cluster taggable". The sentinel remains part
// of the API for hosts that do want the server to choose k.
// `colorBy` selects which colour scheme the legend describes (cluster or tag);
// pass the same value used for the SCS so the legend matches the baked colours.
// `onlyFileIds` drills into an explicit subset: the server re-projects and
// re-clusters within it, so the axes are redrawn rather than zoomed. It is sent
// as a body because it can hold thousands of ids; omitting it sends no body at
// all, which is the pre-drill-down request byte for byte. The response then
// carries `subset_id`, to be passed to mapScsUrl() as `subset`.
// `projection` selects the layout ('umap'); leaving it undefined asks for the
// server's default PCA and keeps the request byte for byte the one sent before
// this parameter existed.
// `clustering` selects how the parts are grouped ('ward'); leaving it undefined
// asks for the server's default k-means and keeps the request byte for byte the
// one sent before this parameter existed.
// `granularity` (0..1) cuts Ward's tree at a height instead of to a count, so
// each branch splits as far as its own spread warrants. It means nothing unless
// `clustering` is 'ward'; leaving it undefined keeps the request byte for byte
// the one sent before it existed.
// `wardMax` is the ceiling that granularity is read against, so it decides how
// fine the deepest cut is. It must equal the `kMax` given to shapeMapTree(), or
// the tree's leaves and the map's clusters stop being the same families.
// Leaving it undefined asks for the server's default (256) and keeps the
// request byte for byte the one sent before it existed.
export function shapeMap(
  name,
  { nClusters = 0, dims = 3, tags, tagMode, tagged, includeTags = false, colorBy = 'cluster', kind, onlyFileIds, outlierThreshold, projection, clustering, granularity, wardMax } = {},
) {
  const opts = {
    method: 'POST',
    query: {
      n_clusters: nClusters,
      dims,
      tags,
      tag_mode: tagMode,
      tagged,
      include_tags: includeTags,
      color_by: colorBy,
      kind,
      outlier_threshold: outlierThreshold,
      projection,
      clustering,
      granularity,
      ward_max: wardMax,
    },
  };
  if (onlyFileIds && onlyFileIds.length) opts.body = { only_file_ids: [...onlyFileIds] };
  return request(`/similarity/index/${encodeURIComponent(name)}/map`, opts);
}

// POST /map/tree -> Ward's hierarchy for the same population /map covers.
// Only meaningful with Ward: the tree IS the clustering there. No projection,
// no colouring and no cluster count, because a tree has no coordinates and no
// chosen cut -- it holds every cut at once.
export function shapeMapTree(
  name,
  // `kMax` must equal the `wardMax` given to shapeMap() and mapScsUrl(): all
  // three read the same ceiling, so a mismatch leaves the tree's leaves and
  // the map's clusters describing different families.
  { tags, tagMode, tagged, kind, onlyFileIds, kMax = 256, clustering } = {},
) {
  const opts = {
    method: 'POST',
    query: {
      tags,
      tag_mode: tagMode,
      tagged,
      kind,
      k_max: kMax,
      // Which linkage builds the tree. Left undefined at the server's default
      // ('ward'), so a Ward request is byte for byte the one this client sent
      // before the choice existed.
      clustering: clustering === 'ward' ? undefined : clustering,
    },
  };
  if (onlyFileIds && onlyFileIds.length) opts.body = { only_file_ids: [...onlyFileIds] };
  return request(`/similarity/index/${encodeURIComponent(name)}/map/tree`, opts);
}

// POST /map/legend-order -> the same cluster ids the map returned, in the leaf
// order of a Ward dendrogram over their centroids, so clusters that are alike
// end up next to each other in the legend.
//
// Give it the arguments of the map on screen; a different population is a
// different clustering and so a different order. Only for k-means: Ward draws
// its clusters in the tree's own order already, and the server returns 400.
//
// request() wraps its result in `{ data, etag }` ONLY when wantEtag is set
// (see #226, where reading a plain body as `{ data }` quietly produced
// undefined). This route has no ETag, so the resolved value is the body.
export function shapeMapLegendOrder(
  name,
  { nClusters = 0, tags, tagMode, tagged, kind, onlyFileIds, outlierThreshold } = {},
) {
  const opts = {
    method: 'POST',
    query: {
      n_clusters: nClusters,
      tags,
      tag_mode: tagMode,
      tagged,
      kind,
      outlier_threshold: outlierThreshold,
    },
  };
  if (onlyFileIds && onlyFileIds.length) opts.body = { only_file_ids: [...onlyFileIds] };
  return request(`/similarity/index/${encodeURIComponent(name)}/map/legend-order`, opts);
}

// POST /map/catalogue -> one entry per family, cut from the same Ward tree
// /map/tree returns. Each group names a tree node id, its member file ids and
// its centre part, plus a stats block describing the cut as a whole.
//
// `kMax` must equal the `wardMax` given to shapeMap() and the `kMax` given to
// shapeMapTree(). The tree's levels are 1, 2, 4, ... up to k_max, so a
// different value renumbers the nodes and the ids returned here name nothing
// in the tree on screen. It is always sent explicitly rather than left to the
// server's default, because the default and the tree's current ceiling are not
// the same thing.
//
// `minReachedParts` is the size at which a group that reached the threshold is
// called "confident" rather than "small". It only labels: the frontier itself
// is the same groups whatever it is set to.
export function shapeMapCatalogue(
  name,
  {
    threshold,
    kMax,
    kind = 'part',
    minParts,
    minReachedParts,
    tags,
    tagMode,
    tagged,
    clustering,
  } = {},
) {
  return request(`/similarity/index/${encodeURIComponent(name)}/map/catalogue`, {
    method: 'POST',
    query: {
      threshold,
      k_max: kMax,
      kind,
      min_parts: minParts,
      min_reached_parts: minReachedParts,
      tags,
      tag_mode: tagMode,
      tagged,
      // Which linkage the tree was built with -- it must be the one the map on
      // screen is showing, or the node ids name families the tree panel is not
      // drawing. undefined at the server's default ('ward').
      clustering: clustering === 'ward' ? undefined : clustering,
    },
  });
}

// Absolute URL of the Shape Map point-cloud SCS (passed to the viewer as its
// endpointUri). Same query vocabulary as shapeMap() plus `color_by`
// ("cluster" | "tag"), which selects the colours baked into the file. The
// server returns 503 here when the point-cloud build tool is not installed,
// while POST /map keeps working. `subset` is the subset_id a drill-down /map
// returned; the server answers 409 once it has forgotten that id, which the
// caller should read as "re-post /map".
// `projection` selects the layout ('umap'); leaving it undefined asks for the
// server's default PCA and keeps the request byte for byte the one sent before
// this parameter existed. It must match what shapeMap() was given: the cloud is
// authored from that payload, so a different value breaks the point order.
// `clustering` selects how the parts are grouped ('ward'); leaving it undefined
// asks for the server's default k-means and keeps the request byte for byte the
// one sent before this parameter existed.
// `granularity` (0..1) is the height Ward's tree is cut at, and means nothing
// unless `clustering` is 'ward'. It must match what shapeMap() was given: it
// decides how many baked nodes there are, and the cloud is authored from that
// payload, so a different value breaks the point order the viewer relies on.
// `wardMax` is the ceiling that granularity is read against. It must match what
// shapeMap() was given, for the same reason.
export function mapScsUrl(
  name,
  { nClusters = 0, dims = 3, tags, tagMode, tagged, colorBy = 'cluster', kind, subset, outlierThreshold, projection, clustering, granularity, wardMax } = {},
) {
  return buildUrl(`/similarity/index/${encodeURIComponent(name)}/map/scs`, {
    n_clusters: nClusters,
    dims,
    tags,
    tag_mode: tagMode,
    tagged,
    color_by: colorBy,
    kind,
    subset,
    outlier_threshold: outlierThreshold,
    projection,
    clustering,
    granularity,
    ward_max: wardMax,
  });
}

// POST /files/upload -> { file_id, filename, already_existed }.
// Stores a CAD file and returns its content-addressed id. The web client uses
// this to bring an arbitrary file into the app (view it, then search with it).
export function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  return request('/files/upload', { method: 'POST', form });
}

// POST /files/exists -> { known, unknown, invalid }.
// Given client-computed SHA-256 file_ids, reports which the server already
// holds. Lets the bulk-registration client hash a folder locally and upload
// only what is missing instead of re-sending everything. Malformed ids come
// back under `invalid` (distinct from `unknown`).
//
// Sends the ids in EXISTS_CHUNK-sized requests, sequentially, and concatenates
// the three arrays, so a caller can pass a whole folder's worth of hashes. The
// chunking lives here rather than in the caller so every caller is safe, and
// the requests are sequential to keep the server's exists cost bounded the way
// the upload loop already does.
// `onProgress(done, total)` is optional and is called after each chunk.
export async function existsFiles(fileIds, onProgress) {
  const ids = fileIds || [];
  const out = { known: [], unknown: [], invalid: [] };
  if (ids.length === 0) return out;
  for (let i = 0; i < ids.length; i += EXISTS_CHUNK) {
    const chunk = ids.slice(i, i + EXISTS_CHUNK);
    const res = await request('/files/exists', {
      method: 'POST',
      body: { file_ids: chunk },
    });
    for (const key of ['known', 'unknown', 'invalid']) {
      for (const id of res[key] || []) out[key].push(id);
    }
    if (onProgress) onProgress(Math.min(i + chunk.length, ids.length), ids.length);
  }
  return out;
}

// POST /files/upload-batch -> { files: [{ file_id, filename, already_existed }],
// errors: [{ filename, detail }], count }. Uploads several CAD files in one
// request; one failing file does not abort the others (every input is reported
// in `files` or `errors`). Used by bulk registration to bring the not-yet-known
// files onto the server before a job is queued.
export function uploadBatch(files) {
  const form = new FormData();
  for (const file of files) form.append('files', file);
  return request('/files/upload-batch', { method: 'POST', form });
}

// -- Bulk registration jobs -------------------------------------------------
//
// Registration also has an asynchronous job form (the synchronous add holds the
// connection open with no progress and no way to stop). The client uploads
// files first (upload-batch, skipping what /files/exists knows), then queues a
// job with their file_ids and polls it. See routers/index_jobs.py.

// POST /similarity/index/{name}/jobs/add -> 202 { job_id, status, accepted,
// errors }. Queues a bulk registration job for the already-uploaded file_ids
// and returns at once. `workers` and `timeLimit` are optional tuning: omit for
// the server's automatic sizing, or use few workers + a long timeLimit for a
// retry pass over heavy assemblies (the documented two-pass loop).
export function createIndexJob(name, fileIds, { workers, timeLimit } = {}) {
  return request(`/similarity/index/${encodeURIComponent(name)}/jobs/add`, {
    method: 'POST',
    body: { file_ids: fileIds, workers, time_limit: timeLimit },
  });
}

// GET /similarity/index/{name}/jobs -> { jobs: [JobStatus], count, total }.
// Registration jobs for an index, newest first.
export function listIndexJobs(name, { limit = 50, offset = 0 } = {}) {
  return request(`/similarity/index/${encodeURIComponent(name)}/jobs`, {
    query: { limit, offset },
  });
}

// GET /similarity/index/{name}/jobs/{job_id} -> JobStatus. Its `progress`
// (phase/done/total/errors/heavy) advances live; `summary`, `errors`,
// `report_url` and the finished_* timestamps fill in when it settles. Poll
// this to drive the progress UI.
export function getIndexJob(name, jobId) {
  return request(
    `/similarity/index/${encodeURIComponent(name)}/jobs/${encodeURIComponent(jobId)}`,
  );
}

// GET /similarity/index/{name}/jobs/{job_id}/report -> plain text. A three-part
// breakdown (ADDED / FAILED / HEAVY-FLAGGED). Returns 404 until the job has
// finished and written one, so callers fetch it only after the job settles.
export function getIndexJobReport(name, jobId) {
  return request(
    `/similarity/index/${encodeURIComponent(name)}/jobs/${encodeURIComponent(jobId)}/report`,
    { rawText: true },
  );
}

// DELETE /similarity/index/{name}/jobs/{job_id} -> JobStatus. Requests
// cancellation: a queued job stops at once, a running job stops after the batch
// currently inside embed_shape_batch finishes (so the response may still report
// running -- keep polling). Files already registered stay registered.
export function cancelIndexJob(name, jobId) {
  return request(
    `/similarity/index/${encodeURIComponent(name)}/jobs/${encodeURIComponent(jobId)}`,
    { method: 'DELETE' },
  );
}

// POST /files/{file_id}/scs -> { file_id, scs_url }.
// Authors an SCS stream cache for a stored (non-indexed) file so the viewer can
// display it. Indexed parts already have an SCS (see scsUrl); this is only for
// uploaded query files.
export function exportScs(fileId) {
  return request(`/files/${encodeURIComponent(fileId)}/scs`, { method: 'POST' });
}

// POST /similarity/compare?file_ids=a,b,... -> pairwise cosine similarity of
// the parts' shape embeddings. This endpoint needs NO FAISS index (it embeds
// the given files on the fly), which is exactly what demo 1 shows: two parts
// can be judged with no project prepared. `fileIds` are already-uploaded
// file_ids (see uploadBatch). Returns { count, model_name,
// files: [{ index, file_id, filename, num_bodies }], matrix,
// pairs: [{ a, b, score }], errors: [{ filename, detail }] }. The server
// returns 422 when fewer than two valid parts are supplied.
export function compareFiles(fileIds) {
  return request('/similarity/compare', {
    method: 'POST',
    query: { file_ids: fileIds.join(',') },
  });
}

// POST /similarity/index/{name}/search -> part similarity search.
// The query is either an uploaded File ({ file }) or the id of a part already
// in the index ({ fileId }); an indexed part's id doubles as its file_id, so a
// part selected in the gallery can be the query without re-uploading. Returns
// { hits: [{ id, score, metadata }], count, image_url, tag_filter }.
//
// include_image is always false: the server can render a result-grid PNG, but
// this client draws its own tiles, and rendering that grid dominates the
// response time (seconds vs. tens of seconds for a large top_k).
//
// kind is pinned to "part" so the results contain only single-body parts,
// matching the hoops_ai_native_bridge Qt sandbox (run_part_search passes
// filters={"kind": "part"}). Without it the endpoint defaults to kind="any"
// and an assembly's whole-shape embedding can outrank every part.
export function searchParts(name, query, options) {
  return runSearch(`/similarity/index/${encodeURIComponent(name)}/search`, query, {
    ...options,
    kind: 'part',
  });
}

// POST /similarity/index/{name}/search-assembly -> assembly similarity search.
// Same query/return shape as searchParts, but each hit also carries assembly
// diagnostics: geom_score, coverage, matched_parts, candidate_parts,
// query_parts. Returns 409 for a legacy (schema v1) index.
export function searchAssembly(name, query, options) {
  // Assembly search is assembly-only by construction (the AssemblyMatcher
  // endpoint has no kind parameter). Drop any `kind` the shared filter may
  // carry so it is never forwarded here; only searchParts pins a kind.
  const { kind: _ignoredKind, ...rest } = options || {};
  return runSearch(
    `/similarity/index/${encodeURIComponent(name)}/search-assembly`,
    query,
    rest,
  );
}

// Shared search POST: send the query as a file_id (indexed part) or a multipart
// upload (a new file). Both search endpoints accept either. `tags` (a comma-
// separated string), `tagMode` ("any"|"all") and `tagged` are the optional tag
// filter, the same one the parts listing uses: it restricts the searched set to
// parts carrying the tags before ranking.
function runSearch(
  path,
  { file, fileId } = {},
  { topK = SEARCH_TOP_K, includeSelf = false, tags, tagMode, tagged, kind } = {},
) {
  const query = {
    top_k: topK,
    include_self: includeSelf,
    include_image: false,
    tags,
    tag_mode: tagMode,
    tagged,
  };
  if (kind) query.kind = kind;
  if (fileId) {
    query.file_id = fileId;
    return request(path, { method: 'POST', query });
  }
  const form = new FormData();
  form.append('file', file);
  return request(path, { method: 'POST', query, form });
}

// -- Tags -------------------------------------------------------------------
//
// The tag endpoints use optimistic concurrency: a GET returns an ETag; a write
// sends it back as If-Match and the server returns 412 if the tags changed in
// between. The ETag covers the whole index's tags (one tags.json), so any tag
// edit anywhere advances it -- an important detail for the caller handling 412.
// Every write also sends X-Client-Id, recorded as updated_by.

// Build the If-Match / X-Client-Id headers for a tag write. If-Match is only
// sent when we hold an ETag, so a first write without one still works (last
// write wins), while an edit that read first is guarded.
function tagWriteHeaders({ ifMatch, clientId } = {}) {
  const headers = {};
  if (ifMatch) headers['If-Match'] = ifMatch;
  if (clientId) headers['X-Client-Id'] = clientId;
  return headers;
}

// GET /similarity/index/{name}/tags -> { data: { tags: [{tag, count}], total },
// etag }. Ordered by descending count, then tag name. The etag is what the
// caller sends as If-Match when deleting a tag.
export function getIndexTags(name) {
  return request(`/similarity/index/${encodeURIComponent(name)}/tags`, { wantEtag: true });
}

// GET /similarity/tag-vocabulary -> { data: { tags: [...], total } }. The
// remembered tag names, which survive a tag being taken off every part. There
// is ONE vocabulary for the whole application -- no index name in the path --
// so a client fetches it once per session rather than on every index switch.
// Not the same thing as getIndexTags(), which reports the tags an index is
// actually using, with counts.
export function getTagVocabulary() {
  return request('/similarity/tag-vocabulary');
}

// GET /similarity/index/{name}/parts/{id}/tags ->
// { data: { part_id, tags }, etag }. The etag is held and sent as If-Match on
// the matching PUT so a concurrent edit is detected.
export function getPartTags(name, partId) {
  return request(
    `/similarity/index/${encodeURIComponent(name)}/parts/${encodeURIComponent(partId)}/tags`,
    { wantEtag: true },
  );
}

// PUT /similarity/index/{name}/parts/{id}/tags -> { data: { part_id, tags },
// etag }. Replaces the WHOLE tag set: callers pass the full desired set (add =
// old set plus one, remove = old set minus one), not a delta. Sends If-Match
// (from a prior GET) so a stale write is rejected with 412.
export function putPartTags(name, partId, tags, { ifMatch, clientId } = {}) {
  return request(
    `/similarity/index/${encodeURIComponent(name)}/parts/${encodeURIComponent(partId)}/tags`,
    {
      method: 'PUT',
      body: { tags },
      headers: tagWriteHeaders({ ifMatch, clientId }),
      wantEtag: true,
    },
  );
}

// POST /similarity/index/{name}/tags/{tag}/parts -> { data: { tag, updated,
// skipped: [{part_id, reason}] }, etag }. Adds one tag to many parts at once.
// Unregistered ids are reported in `skipped` rather than failing the request.
export function assignTag(name, tag, partIds, { ifMatch, clientId } = {}) {
  return request(
    `/similarity/index/${encodeURIComponent(name)}/tags/${encodeURIComponent(tag)}/parts`,
    {
      method: 'POST',
      body: { part_ids: partIds },
      headers: tagWriteHeaders({ ifMatch, clientId }),
      wantEtag: true,
    },
  );
}

// DELETE /similarity/index/{name}/tags/{tag}/parts -> same shape as assignTag.
// Removes one tag from many parts at once.
export function unassignTag(name, tag, partIds, { ifMatch, clientId } = {}) {
  return request(
    `/similarity/index/${encodeURIComponent(name)}/tags/${encodeURIComponent(tag)}/parts`,
    {
      method: 'DELETE',
      body: { part_ids: partIds },
      headers: tagWriteHeaders({ ifMatch, clientId }),
      wantEtag: true,
    },
  );
}

// DELETE /similarity/index/{name}/tags/{tag} -> { data: { tag, removed_from },
// etag }. Removes the tag from EVERY part. The server requires confirm=true
// (409 without it), so the caller must confirm with the user first.
export function deleteTag(name, tag, { confirm = false, ifMatch, clientId } = {}) {
  return request(`/similarity/index/${encodeURIComponent(name)}/tags/${encodeURIComponent(tag)}`, {
    method: 'DELETE',
    query: { confirm },
    headers: tagWriteHeaders({ ifMatch, clientId }),
    wantEtag: true,
  });
}

// -- Context (production records) -------------------------------------------
//
// The Context Layer predicts a query part's missing production records
// (material, process, cost, lead time, ...) from the records its nearest
// neighbours carry. Two calls back the Estimate panel: getIndexContext() lists
// which fields the index holds and their units; predictContext() runs a
// prediction for one query part.

// GET /similarity/index/{name}/context -> { keys: [{key, count, type}],
// parts_with_context, units }. `units` maps a numeric field to its unit label
// (e.g. { UnitCost: "USD", LeadTimeDays: "days" }), so the client can label the
// numbers it shows. Used to choose which keys to predict and how to render them.
export function getIndexContext(name) {
  return request(`/similarity/index/${encodeURIComponent(name)}/context`);
}

// POST /similarity/index/{name}/predict-context -> { keys, candidates_examined,
// hits_considered, hits_with_context, coverage, predictions, hits }.
// `fileId` is the query part (the model open in the viewer). `keys` are the
// fields to predict; `kind` defaults to "part" so an estimate is not diluted by
// assemblies. `lotSize` is sent as query_context { LotSize } so the numeric
// rules can weight by order quantity. `neighborIds`, when given, pins the
// neighbours to exactly those ids (the Results-tab shown tiles) instead of a
// fresh top_k neighbourhood; `topK` should then match the search that produced
// them so the server's search pool still contains them. `predictions[key]`
// carries value / confidence / status / alternatives / reasons /
// injected_context; each hit carries id / score / filename / thumbnail_url /
// values (the evidence record).
export function predictContext(
  name,
  fileId,
  { keys, topK = 15, kind = 'part', neighborIds } = {},
) {
  const body = { file_id: fileId, keys, top_k: topK, kind };
  if (Array.isArray(neighborIds)) {
    body.neighbor_ids = neighborIds;
  }
  return request(`/similarity/index/${encodeURIComponent(name)}/predict-context`, {
    method: 'POST',
    body,
  });
}
