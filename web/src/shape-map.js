// Shape Map mode: shows the index as a 3D point cloud in the shared viewer.
//
// The server does the maths. This module only: fetches the point array + legend
// (POST /map), tells the viewer to load the matching point-cloud SCS
// (GET /map/scs), renders the legend, and turns a picked point back into a part
// (via its index in the /map array). Colours are baked into the SCS by the
// server, so the client never decides a colour -- changing the scheme re-fetches
// the SCS with a different `color_by`.
//
// resolvePoint() below never guesses: a pick that cannot be resolved to a
// /map point is reported to the caller as `null` rather than falling back to
// a plausible-looking index. It resolves by nearest picked-position match
// within the picked node's group range rather than "group.offset + pointId"
// -- see resolvePoint() for why. A resolved point's marker is drawn from its
// /map coordinates via the viewer's showPointMarker(), which is the only
// visual indicator of a pick (see viewer.js -- node-level selection
// highlighting is turned off in map mode).

import * as api from './api.js';
import { showError } from './ui.js';
import { legendRenderOrder } from './legend-order.js';
import { splitFocusedCandidates } from './model-focus.js';
import { clientId } from './client-id.js';
import { openTagPopover } from './tag-picker.js';
import {
  showMap,
  addMapModels,
  addMapModel,
  removeMapModels,
  setMapModelScale,
  setMapModelTrueScale,
  setMapModelGroupVisible,
  setMapModelGroupsVisible,
  peekMapModelGroups,
  highlightMapModel,
  setMapModelHighlighted,
  setMapModelColors,
  clearMapModelHighlight,
  getMapCamera,
} from './viewer.js';

// Fixed projection dimensionality. The map coordinates are the server's
// business (seed=42, deterministic); the client asks for a 3D layout and draws
// whatever comes back.
const MAP_DIMS = 3;

// Fallback cluster count for the map colouring, used only when the host has no
// `default_clusters` to apply -- that per-index count, from
// /index/{name}/stats, is the real default and is what the host sets through
// setNClusters(). This matches the option marked `selected` in index.html. The
// client chooses from a few discrete counts rather than a slider, because each
// change re-runs k-means.
const DEFAULT_N_CLUSTERS = 8;

// Where Ward's tree is cut. Always the deepest level: the tree view shows the
// whole hierarchy, so a cut height is no longer something to choose -- the
// deepest cut is simply the tree's leaves, and every coarser grouping is a
// node above them. 1.0 is the top of the server's 0..1 scale.
const WARD_GRANULARITY = 1.0;

// Default ceiling of Ward's tree: the cluster count its deepest level reaches.
// This matches the option marked `selected` on #map-wardmax in index.html, and
// it is the server's own default for both /map's ward_max and /map/tree's
// k_max, so the first request is the one this client sent before the control
// existed.
const DEFAULT_WARD_MAX = 256;

// The categorical palette the server paints its point clouds with. KEEP THESE
// VALUES IDENTICAL TO _SHAPE_MAP_PALETTE IN shape_map.py: the tree recolours a
// cloud the server already coloured, so a drifted entry would show as one
// family changing shade for no reason. 24 colours -- Okabe-Ito's eight
// colour-blind-safe qualitative colours first (their black replaced by a
// bright neutral), then Kelly's maximum-contrast colours with the dark and
// near-grey entries dropped.
const SHAPE_MAP_PALETTE = [
  [230, 159, 0], [86, 180, 233], [0, 158, 115], [240, 228, 66],
  [0, 114, 178], [213, 94, 0], [204, 121, 167], [222, 222, 222],
  [255, 179, 0], [255, 104, 0], [193, 0, 32], [0, 125, 52],
  [246, 118, 142], [0, 83, 138], [255, 122, 92], [255, 142, 0],
  [179, 40, 81], [244, 200, 0], [147, 170, 0], [241, 58, 19],
  [166, 189, 215], [206, 162, 98], [128, 62, 117], [129, 197, 122],
];

// How big an overlaid 3D model should be, as a multiple of the distance to the
// nearest other point in the SAME layout. Sizing models against the [-1, 1]
// cube instead assumes every layout spaces its points alike, which they do not:
// measured on tmcad (7,928 parts), the median nearest-neighbour distance is
// 0.0276 under PCA and 0.0023 under UMAP -- a factor of twelve. At a fixed
// size the UMAP map draws every model about fifty times its neighbour spacing,
// which is why it reads as one lump until the Scale slider is pulled down.
//
// 4.3 is not a taste: it is 0.12 / 0.0276, the ratio the PCA map has always
// been drawn at, so PCA keeps the look it has today while UMAP shrinks to match
// its own spacing.
const MODEL_SPACING_RATIO = 4.3;

// Bounds on the measured size, so a degenerate layout (every point on top of
// another, or two points at opposite corners) cannot produce an invisible or a
// viewport-filling model.
const MODEL_TARGET_DIAG_MIN = 0.008;
const MODEL_TARGET_DIAG_MAX = 0.3;

// Points sampled when measuring the spacing. Each sampled point is compared
// against every point, so this is SAMPLE x N distance computations -- 256 x
// 8,000 is a few tens of milliseconds and runs once per map build.
const SPACING_SAMPLE = 256;

// How long the pointer must rest on a legend row before its category is
// previewed alone on the map.
const PEEK_DELAY_MS = 250;

// How long it must rest before that category's parts are additionally LOADED
// as 3D models. Longer than PEEK_DELAY_MS because this one fetches geometry:
// showing/hiding what is already there is free, pulling in a budget's worth of
// SCS files is not, so sweeping down the legend must not trigger it.
const PEEK_MODEL_DELAY_MS = 600;

// How much room to leave around a hovered cluster when the camera frames it
// (see focusGroup). A FRACTION of the cluster's own size, not an amount added
// to it: a cluster has no fixed size in map units, and adding a constant means
// framing the constant rather than the cluster whenever the layout packs its
// points tightly. Measured on tmcad (k=64, Outliers 0.92), the median cluster's
// largest half-extent is 0.1723 under PCA but 0.0272 under UMAP, so the 0.12
// this used to add filled 14% of the UMAP frame with cluster and the rest with
// margin.
//
// 0.15 is small on purpose. The margin was originally there so that
// neighbouring clusters stayed visible around the one being framed -- but the
// preview this runs inside hides every other category while the pointer rests
// on a row (see schedulePeek), so there is nothing in that margin to see. What
// is left is only the room needed not to clip the cluster, and view.fitBounding
// adds some of that itself. The median cluster is framed at 0.172 x 1.15 =
// 0.198 under PCA (it was 0.292 before any of this) and at 0.0272 x 1.15 =
// 0.031 under UMAP.
const CLUSTER_FOCUS_PAD_FRACTION = 0.15;

// Floor on the half-extent, as a multiple of the layout's own nearest-neighbour
// spacing, so a cluster of two points still gets a frame with something in it
// rather than a camera pressed against one part. Three spacings is 0.083 under
// PCA -- the 0.08 this used to be a constant for -- and 0.007 under UMAP.
const CLUSTER_FOCUS_MIN_SPACINGS = 3;

// Absolute floor, for the degenerate case where every sampled point sits on
// top of another and the measured spacing is ~0.
const CLUSTER_FOCUS_MIN_HALF_FLOOR = 0.002;

// A cluster with at least this many members is counted as "worth one tag" in
// the legend summary. Below it, a cluster is a stray pair rather than a family
// of parts someone would name.
const LEGEND_SUMMARY_MIN = 10;

// How many steps the heat-map gradient is cut into. The gradient is quantised
// so that recolouring a cluster costs a fixed number of setNodesFaceColor
// calls (one per band) rather than one per model: a cluster can hold hundreds
// of parts, and the eye cannot separate more shades than this anyway.
const HEAT_BANDS = 32;

// Hits below the Min similarity slider keep their place on the map but stop
// carrying a reading: grey says "outside the set you are looking at" without
// implying a value on the heat scale.
const SEARCH_BELOW_RGB = [120, 120, 120];
// Recolouring 301 models is cheap but not free, and the Min similarity slider
// fires on every tick of a drag. Coalesce.
const SEARCH_COLOR_DELAY_MS = 120;

// A warm map returns in ~0.2 s, but changing the filter or cluster count makes
// the server re-project (seconds). Revealing the progress hint is now the
// host's job, because onBusy fires immediately (it also locks the controls);
// see MAP_PROGRESS_DELAY_MS in main.js for the "only show a spinner when the
// build is actually slow" part.

// The 3D-model map shows real geometry instead of points, which is far heavier
// per part, so it is capped. When the mapped set is larger than this, a
// representative subset is chosen (see selectRepresentatives): one part per
// cluster first, spreading outward, so every cluster is visible.
//
// Measured on tmcad (10,230 parts, 2026-09-11): loading 800 parts with the
// bounded parallel pool in addMapModels (concurrency 4) takes ~4 s, downloads
// ~45.6 MiB of SCS, and settles at ~161 MiB of JS heap -- comfortable for the
// demo, and enough that every visible cluster carries several real parts.
// Smaller indexes (e.g. my-index's 50 parts) still fit entirely, so they look
// exactly as before.
//
// This is a cap on the TOTAL number of models alive in the viewer at once --
// the ordinary selection plus anything the legend's hover preview has overlaid
// on top of it -- not a budget each of them gets separately. One rule, so the
// number on screen does not depend on whether a row happens to be hovered.
const MAX_MODELS = 800;

// Assemblies carry many more bodies per node than a single part, so each
// overlaid assembly is far heavier to load and render than a part. Cap the
// assembly-kind map at half the part budget to keep it responsive; parts (and
// the mixed "any" kind) keep the full MAX_MODELS budget.
const MAX_MODELS_ASSEMBLY = 400;

// The slice of MAX_MODELS held back for the legend's hover preview. The
// ordinary selection loads up to MAX_MODELS minus this, leaving the reserve
// empty, and the preview spends only what is free -- so the total on screen
// never exceeds MAX_MODELS, hovered or not.
//
// That total is what the viewer has to hold, and it has a ceiling: ~1,078
// models load and render cleanly, while ~1,600 makes feedScsBuffer fail with
// "memory access out of bounds" and leaves the frame buffer with zero-size
// attachments. Capping the total at MAX_MODELS keeps it well inside the range
// that is known to work, because that is the number this has always loaded.
const MODEL_PEEK_RESERVE = 200;
const MODEL_PEEK_RESERVE_ASSEMBLY = 100;

// How close (in the map's normalized [-1, 1] space) a /map point must be to
// the picked world position to be trusted as that pick (see resolvePoint()).
// A genuine match differs from its /map coordinates by float32 rounding only
// (the SCS stores points as float32; testing shows real matches land at
// 0.0000 once rounded to 4 decimals) -- this is a sanity backstop against a
// group/offset that resolved to the wrong territory entirely, not a tight
// matching tolerance.
const NEAREST_POINT_TOLERANCE = 0.05;

// Every layout the client may ask for. The <select> in index.html offers
// exactly these, and tests/test_map_select_options.py pins that the two agree
// -- together with the server, which is the side that was already right when
// this set did not exist and a layout was being dropped here. An unknown value
// falls back to PCA rather than failing: a bad value must not leave the map
// unable to draw.
//
// A set rather than a chain of comparisons on purpose. The chain is what
// silently turned a third layout into 'pca' while that layout existed: adding
// a layout meant remembering to extend an expression, and forgetting cost
// nothing at the time.
const MAP_PROJECTIONS = new Set(['pca', 'umap']);

// Default size multiplier for the 3D-model map (the "Part" slider). Each model
// is drawn at the size measured for the current layout (see
// MODEL_SPACING_RATIO) times this,
// so at 1.0 the 800 overlaid parts overlap heavily; 0.5 spaces them so their
// shapes read individually while still filling the clusters. The slider (see
// index.html) runs from 0.1 (well below this, so parts can be shrunk clear of
// each other) up to 5 (plenty of room to enlarge a sparse map).
const DEFAULT_PARTSCALE = 0.5;

// Smallest cluster worth drilling into. Below this, re-projecting 2048-dim
// vectors is meaningless: the SVD has fewer samples than it needs to find a
// direction and k-means splits noise. A module constant rather than a server
// setting so it can be tuned from here while the experiment runs.
const MIN_DRILL_PARTS = 40;

// Create the Shape Map controller.
//
//   legendEl        the legend PANEL. Passed to showMap(), whose Legend button
//                   shows and hides it; never emptied by this module
//   legendBodyEl    element the legend rows are rendered into. Separate from
//                   the panel because the panel also holds the clustering
//                   controls, which a rebuild must not wipe out
//   treeBodyEl      element Ward's hierarchy is rendered into, the legend's
//                   counterpart. The host switches which of the two is on
//                   screen; what goes in this one is drawn from the payload
//                   this module holds, so it is drawn here
//   onBusy(busy)    a rebuild started / finished. Fires IMMEDIATELY (no delay)
//                   because the host locks the map controls from it; it is the
//                   host that delays the visible spinner
//   onPointInfo(pt) a point was single-clicked -> show its part in the info area
//   onPointClear()  a click hit no point -> clear the info area (a normal
//                   deselect, not an error, so this is separate from
//                   onPointInfo(null), which reports a pick that could not be
//                   resolved)
//   onPointOpen(pt) a point was double-clicked -> open its part in 3D
//   getIndexName()  the current index name
//   onClusterTagged(tag, { reveal }) a set of parts was bulk-tagged from the
//                   map (a legend row, a tree node, or the Catalogue tab); the
//                   host refreshes the tag filters and gallery so the new tag
//                   shows everywhere. With `reveal` false the host must not
//                   navigate: the catalogue is still being worked in.
//   onClustersSet(v) the drill-down changed the cluster count (reset to the
//                   default on the way down, restored to the count remembered
//                   for that level on the way up); the host syncs the Clusters
//                   control. `v` is the internal value: the cluster count.
//                   DOM only -- the map has already been rebuilt.
//   onClusters(r)   a map finished loading. `r` is
//                   `{ k, colorBy, ward }`: the cluster count the server used,
//                   the colouring it was built for, and -- when Ward was cut by
//                   height -- the `{cut_height, height_lo, height_hi, k_max,
//                   granularity}` block it was cut with.
//   onOutlierThreshold(v) a map finished loading; `v` is the outlier threshold
//                   the server actually applied, which is below the requested
//                   one when the request exceeded what the clustering could
//                   support. Fires on success only, so the host never reports a
//                   cap from a map that was never drawn.
//   onUntaggedStale(n) in Untagged-only mode, `n` parts have been tagged since
//                   the map on screen was built (0 right after a build). The
//                   host puts it on the Reload button; the map is NOT rebuilt
//                   by itself, because a re-projection takes seconds and when
//                   to pay for it is the user's call.
//   getSearchContext() the last similarity search as { queryFileId, kind,
//                   hits: [{ id, score }] }, or null when none has run. Read
//                   on demand, so the search-results map always draws the hit
//                   set that is in the results panel right now.
//
// The map always projects the FULL corpus so a part's position can be read
// against every other part. The gallery's tag filter is deliberately NOT
// applied here: narrowing the map by tag would re-project on that subset and
// hide where a part sits among all shapes. Tags are still available as a
// colouring (setColorBy('tag')), which keeps the full projection.
//
// Returns { active(), enter(), exit(), refresh(), setColorBy(v),
//   setNClusters(v) }.
export function createShapeMap({
  legendEl,
  legendBodyEl,
  treeBodyEl,
  onBusy,
  onClusters,
  onPointInfo,
  onPointClear,
  onPointOpen,
  onPartFound,
  getIndexName,
  onClusterTagged,
  onClustersSet,
  onOutlierThreshold,
  onUntaggedStale,
  getSearchContext,
}) {
  let active = false;
  let colorBy = 'cluster';
  let kind = 'part';
  let nClusters = DEFAULT_N_CLUSTERS;
  // Cosine similarity below which a point is split out of its cluster into the
  // Outliers group. 0 = off, which is the default: the split changes what a
  // cluster means, so it is opt-in. Unlike the cluster count this is NOT reset
  // when drilling -- it is an absolute similarity, so the same value means the
  // same thing at every level of the drill stack.
  let outlierThreshold = 0;
  // The threshold that actually applies, which is 0 under Ward whatever the
  // slider holds. Outliers carves a category out of each cluster by distance
  // from its centroid -- a k-means idea. Ward cuts by height, where every
  // node's cohesion already reports how tight it is, and the extra category
  // would stop the map's categories matching the tree's nodes one for one,
  // which is what the tree's colours, checkboxes and tagging stand on. The
  // slider's own value is left alone so that going back to k-means brings it
  // back; the host hides the control under Ward.
  function effectiveOutlierThreshold() {
    if (clustering === 'ward') return 0;
    return outlierThreshold || 0;
  }
  // Which projection the server should use for the three drawn dimensions.
  // 'pca' (the default) or 'umap'. Only the point POSITIONS change: k-means
  // runs on the full-dimensional aggregate, so cluster membership, ids, colours
  // and the legend are identical under both.
  let projection = 'pca';
  // How the server should group the parts: 'kmeans' (the default) or 'ward'.
  // Unlike the projection this moves no point -- both run on the same
  // full-dimensional aggregate and only change which cluster a part lands in.
  let clustering = 'kmeans';
  // The ceiling of Ward's tree: the cluster count its deepest level reaches.
  // ONE value, read by all three requests that depend on it -- /map's and
  // /map/scs's ward_max and /map/tree's k_max. They have to agree: the map's
  // clusters are the tree's leaves, and a mismatch leaves two different sets
  // of families on screen at once, which is what the tree's colours,
  // checkboxes and tagging all rest on.
  let wardMax = DEFAULT_WARD_MAX;
  // The `ward` block from the last map the server built: the height it cut at
  // and the band that height came from. null whenever the server did not send
  // one (k-means, or Ward cut to a count).
  let wardBand = null;
  // The model size measured from the current layout, in map units. Recomputed
  // on every map build and handed to addMapModels(); null until the first one.
  let modelTargetDiag = null;
  // The median nearest-neighbour distance of the current layout, in map units.
  // Measured once per build (see measureLayoutSpacing) and used by anything
  // that needs to know how tightly this layout packs its points: the model
  // size, and how closely the camera frames a cluster. 0 until the first build.
  let layoutSpacing = 0;
  // Cluster only the parts that carry no tag yet. The filter is applied before
  // the projection, so the SVD and k-means run on the remainder alone: tag the
  // clusters that are obvious, switch this on, and what is left gets its own
  // centroids. false means "every part", which is the pre-existing behaviour.
  let untaggedOnly = false;
  // How many parts have been tagged since the map on screen was built, while
  // Untagged-only mode is on. Those parts are still drawn, because the server
  // filtered them in before they carried a tag; the count tells the user how
  // far the map has drifted from the index so they can decide when the ~7.6 s
  // rebuild is worth it. Rebuilding on every tag took that choice away.
  let untaggedStale = 0;
  // The file ids the Untagged-only map on screen covers, pinned so that
  // rebuilding it (View toggle, cluster count, returning from a part) asks for
  // the same parts instead of re-querying what is untagged by then. Cleared by
  // Reload, and whenever the population is redefined (the checkbox, the kind
  // filter, a new index). null means "ask the index".
  let untaggedPin = null;
  // The search-results map: the overlaid models are the hits of the last
  // similarity search instead of a per-cluster sample. The point cloud
  // underneath is unchanged -- it is the k-means picture being compared
  // against -- which is why this mode forces Cluster colouring and the
  // 3D-model view.
  // Legend hover moves the camera to the hovered cluster as well as previewing
  // it. Off by default, and client-side only -- it changes nothing about the
  // clustering, the projection or what is loaded.
  let zoomOnHover = false;
  let searchMode = false;
  let searchFileIds = null;   // Set of hit file ids, or null
  // True while the overlaid models carry the search colouring instead of their
  // cluster colours, so they can be put back when the search map is switched
  // off.
  let searchColored = false;
  let searchColorTimer = null;
  // 'cloud' shows the full corpus as a point cloud; 'models' shows up to
  // MAX_MODELS representative parts as real 3D geometry. Persisted across enters.
  let displayMode = 'cloud';
  // Per-part size multiplier for the 3D-model map (the "Part" slider).
  let partScale = DEFAULT_PARTSCALE;
  // Draw every model at one common scale (real size ratios visible) instead of
  // each at the same on-screen size. Off by default: the embedding discards
  // absolute size, so equal on-screen size is what the map's coordinates
  // actually encode. Persisted across enters, like the rest of the controls.
  let trueScale = false;
  // File ids of the parts actually shown as 3D models (the representatives), so
  // locate() knows whether a part is on the current model map.
  let modelFileIds = new Set();
  // The heat-map overlay, when it is on: the file ids that were recoloured and
  // the cluster's own colour, so clearHeat() can put them back without having
  // to find the group again (the group it came from may already be gone when
  // the map is rebuilt).
  let heatState = null;
  // The colour bar overlaid on the viewer while the heat map is on, built once
  // and reused (like #viewer-title).
  let heatBarEl = null;
  // The Heat map toolbar button of the current legend, so clearHeat() can put
  // its label back from anywhere.
  let heatButton = null;
  // The last /map payload; its `points` array (ordered by group) plus `groups`
  // are the source of truth for turning a picked (node, in-node id) into a part.
  let mapData = null;
  // The last /map/tree payload, or null for a non-Ward map and for a tree that
  // could not be fetched. Auxiliary throughout: the map never depends on it.
  let treeData = null;
  // Why there is no tree, when there is none: the fetch's message. Shown in the
  // panel, because an empty panel says nothing about whether a tree was asked
  // for, refused or lost.
  let treeError = null;
  // Which nodes are expanded. The expansion survives a map rebuild on purpose
  // -- the same tree comes back and the reader should not have to re-open their
  // way down to where they were -- but not an index switch, where the ids mean
  // something else entirely.
  let treeOpen = new Set();
  // Whether the default expansion has been applied. A flag rather than "the set
  // is empty", so collapsing every row by hand stays collapsed across the next
  // rebuild instead of springing open again.
  let treeSeeded = false;
  // node id -> palette index, for the nodes the tree is currently colouring.
  // Kept between draws so expanding one branch does not repaint the rest of the
  // map: a node that stays on the frontier keeps the colour it had.
  let treeColorOf = new Map();
  // The same assignment keyed by CATEGORY (node_name -> [r,g,b]), rebuilt by
  // assignTreeColors(). Read when a model is placed, so it arrives wearing its
  // family's colour instead of the cluster colour baked into its SCS. Empty
  // outside Ward's cluster colouring, where the baked-in colour is correct.
  let treeColorByCategory = new Map();
  // The part the reader last clicked in the viewer, and the panel rows that can
  // point at it. A click on a part says "this one", and the panel is where its
  // family can be read; without this the two halves of the screen never refer
  // to each other. Cleared on every rebuild: the id may not even be in the next
  // population.
  let revealedFileId = null;
  // The part the Part name search last found. Kept apart from revealedFileId:
  // that one marks a row in the panel, this one owns a marker in the viewer
  // that has to survive the hover previews which clear it (see endPeek).
  let foundFileId = null;
  // The query the cursor below belongs to, and which of its matches is on
  // screen. File names are base names, so one name can belong to several
  // parts; pressing Enter again walks them instead of insisting on the first.
  let findPartQuery = null;
  let findPartCursor = 0;
  // Every point the last query matched, in the order the cursor walks them.
  // Kept so the caller can offer the candidates as a list and jump straight to
  // one: point order differs between clusterings, so "the first match" is not
  // the same part on a k-means map and a Ward map.
  let findPartMatches = [];
  // file_id -> file name, for every point of the current map. Built once per
  // map because the lookups are per row: scanning the points array for each of
  // 256 rows over 7,928 points is two million comparisons for a name.
  let filenameByFileId = new Map();
  // node_name -> the legend row drawing that category (k-means).
  const legendRowOf = new Map();
  // tree node id -> the row drawing it, for the nodes currently on screen. A
  // collapsed branch has no row, which is why the walk below climbs.
  const treeRowOf = new Map();
  // node_name -> the tree node that IS that category, and child -> parent for
  // the whole tree. Both are rebuilt by renderTree(), so the highlight can be
  // recomputed outside it (after a legend draw, or on a click).
  const treeCutNodeOf = new Map();
  const treeParentOf = new Map();
  // The controller returned by showMap(), used to toggle a category node's
  // visibility from a legend checkbox.
  let mapViewer = null;
  // Camera to restore on the next load(), set only for a View toggle so the
  // point-cloud <-> 3D-models switch keeps the current viewpoint (see
  // setDisplayMode). Consumed and cleared by load(); null for every other
  // rebuild, which frames the iso overview as before.
  let pendingCamera = null;
  // node_name -> whether its checkbox is on. "Untagged" defaults off so the grey
  // majority does not bury the tagged islands; the user can switch it on.
  const groupChecked = new Map();
  // The legend's toolbar and its remembered checkbox state, kept here because
  // the rows that read them are now built for two panels: the legend and the
  // cut rows of Ward's tree. renderLegend() sets both before it builds a row.
  let legendTools = null;
  let legendRemembered = null;
  // How the legend's rows are ORDERED: 'size' (the groups array's own order,
  // which is member count descending) or 'similarity' (the leaf order of a
  // Ward dendrogram over the cluster centroids, from /map/legend-order).
  //
  // The setting is deliberately not remembered across maps: it is a way of
  // looking at one legend, and the order itself has to be fetched for each.
  let legendOrderMode = 'size';
  // legend signature -> the cluster id order the server returned for it. One
  // map, one order: the signature already names the index, the kind, the
  // colouring, the drill depth, the subset and the node names, so an entry can
  // only be reused for the legend it was computed from.
  const legendOrders = new Map();
  // The signature renderLegend() is currently drawing, so the fetch that comes
  // back later can tell whether it still belongs to what is on screen.
  let legendOrderSignature = '';
  // The arguments the map on screen was built with. The order must be asked
  // for with the same ones, or it would describe a different clustering.
  let lastMapRequest = null;
  // Categories whose parts are shown as models IN FULL, on top of the ordinary
  // representative spread. This is NOT a visibility state: it never hides
  // anything, and groupChecked is never switched off for it. The point cloud
  // behind stays exactly as the checkboxes left it -- which is the whole
  // reason this exists rather than Isolate.
  let focusNames = new Set();
  // What the last model selection actually did with them: the names that fit
  // the budget, and the names that did not. Kept so the rows can say which is
  // which -- a pinned category that is silently absent from the viewer reads
  // as a broken control.
  let focusUsed = new Set();
  let focusDropped = [];

  // Forget every pin. Called from the paths that REPLACE the categories --
  // a different index, clustering, cluster count, drill level, Outliers
  // threshold, colouring or kind -- because a name from the old map either
  // does not exist in the new one or names a different set of parts, and
  // either way the pin would aim the models at nothing in particular.
  //
  // Deliberately NOT called from: the View toggle (the categories are
  // unchanged, so going back to 3D models should give back the same picture),
  // entering search mode (selectVisibleModelParts returns early there, so the
  // pins are merely inert until the search is left), unchecking a pinned
  // category (the pin stops having an effect and starts again when the box is
  // ticked), Isolate / Hide / Show all (they write groupChecked only) and the
  // legend's Order control (it changes the order of the rows, nothing else).
  function clearModelFocus() {
    if (!focusNames.size && !focusUsed.size && !focusDropped.length) return;
    focusNames = new Set();
    focusUsed = new Set();
    focusDropped = [];
  }

  // Pin every category in *cats* and refresh the models. A pinned category
  // that is switched off would show nothing, so the box is ticked here too --
  // switched ON only, never off: the point cloud must not lose a category to
  // an action whose name says nothing about hiding.
  function pinCategories(cats) {
    let changed = false;
    for (const group of cats) {
      if (!group || !group.node_name) continue;
      if (!focusNames.has(group.node_name)) {
        focusNames.add(group.node_name);
        changed = true;
      }
      if (groupChecked.get(group.node_name) === false) {
        setTreeFamilyVisible([group], true);
      }
    }
    if (changed) markFocusedRows();
    scheduleModelRefresh();
  }

  function unpinCategories(cats) {
    let changed = false;
    for (const group of cats) {
      if (group && group.node_name && focusNames.delete(group.node_name)) changed = true;
    }
    if (changed) {
      focusUsed = new Set([...focusUsed].filter((n) => focusNames.has(n)));
      focusDropped = focusDropped.filter((n) => focusNames.has(n));
      markFocusedRows();
    }
    scheduleModelRefresh();
  }

  function unpinAll() {
    if (!focusNames.size) return;
    clearModelFocus();
    markFocusedRows();
    scheduleModelRefresh();
  }

  // Whether every category under a row is pinned, and whether any of them was
  // left out of the last selection for want of budget.
  function focusStateOf(cats) {
    const names = cats.map((g) => g && g.node_name).filter(Boolean);
    if (!names.length) return { pinned: false, dropped: false };
    return {
      pinned: names.every((n) => focusNames.has(n)),
      dropped: names.some((n) => focusDropped.includes(n)),
    };
  }

  // Put the pin marks on the rows that are on screen. Classes only: the rows
  // are not rebuilt, so this can run after a model refresh has decided which
  // pins actually fit without disturbing anything else on the panel.
  function markFocusedRows() {
    const mark = (row, cats) => {
      if (!row) return;
      const { pinned, dropped } = focusStateOf(cats);
      row.classList.toggle('map-row-pinned', pinned && !dropped);
      row.classList.toggle('map-row-pinned-dropped', pinned && dropped);
      if (pinned && dropped) {
        row.title = `Not shown: the model budget (${baseModelBudget()}) is already full`;
      } else if (row.title && row.title.startsWith('Not shown:')) {
        row.title = '';
      }
    };
    for (const [nodeName, row] of legendRowOf) {
      mark(row, [{ node_name: nodeName }]);
    }
    const catsOf = treePaint ? treePaint.catsOf : null;
    if (!catsOf) return;
    for (const [nodeId, row] of treeRowOf) {
      mark(row, catsOf.get(nodeId) || []);
    }
  }

  // The pin / unpin entries a row's right-click menu offers. Shared by the
  // legend's rows and the tree's nodes, so the two panels cannot drift apart.
  // Only in the 3D-model view: in the point cloud there are no models to aim.
  function focusMenuItems(cats) {
    if (displayMode !== 'models' || !cats.length) return [];
    const items = [];
    const { pinned } = focusStateOf(cats);
    items.push({
      label: pinned ? 'Unpin parts' : 'Pin parts',
      title: 'Show every part in this cluster as a model, without hiding anything else',
      run: () => (pinned ? unpinCategories(cats) : pinCategories(cats)),
    });
    if (focusNames.size) {
      items.push({
        label: 'Unpin all',
        title: 'Go back to the ordinary spread of representative models',
        run: () => unpinAll(),
      });
    }
    return items;
  }
  // node_name -> every checkbox currently showing that category. The same
  // category can appear twice (legend row and tree cut row), and a category is
  // either on or off -- one truth, so both boxes must read the same.
  const groupCheckInputs = new Map();
  // Checkbox state per legend signature (index + kind + colour scheme + depth +
  // subset + node names). Keeping only the state of the last render meant that
  // drilling in always cleared it -- the depth and subset are part of the
  // signature -- so coming back up lost the user's filtering. With one entry
  // per signature the state of the level we left is still there when we return,
  // and #90 is what makes returning land on the same signature: the cluster
  // count is restored, so the node names match too.
  //
  // Bounded because the key space is open-ended (index x kind x colouring x
  // depth x subset x cluster count x outlier threshold); the oldest insertion
  // is dropped first and a signature is re-inserted on use so it counts as
  // recent. No explicit clearing anywhere: the index name is part of the
  // signature, so another index can never restore into this one.
  const LEGEND_STATE_LIMIT = 16;
  const legendStates = new Map();

  function legendStateFor(signature) {
    let state = legendStates.get(signature);
    if (state) {
      legendStates.delete(signature); // re-insert so it ranks as most recent
    } else {
      state = new Map();
    }
    legendStates.set(signature, state);
    while (legendStates.size > LEGEND_STATE_LIMIT) {
      legendStates.delete(legendStates.keys().next().value);
    }
    return state;
  }

  // Drill-down stack. Entry 0 is always the whole index ({ fileIds: null }); a
  // "Drill in" push adds { label, fileIds, subsetId } holding the file ids of
  // the cluster that was drilled into. The top entry's fileIds are sent as
  // only_file_ids, so the server re-projects and re-clusters WITHIN them --
  // the axes are redrawn, which is the point; this is not a zoom. subsetId is
  // filled in from the /map response and then handed to /map/scs so the point
  // cloud describes the same points as the payload. nClusters is the control
  // chosen AT that level, remembered on the way down and put back by drillTo()
  // on the way up.
  let drillStack = [{
    label: 'All', fileIds: null, subsetId: null, count: 0,
    nClusters: null,
  }];

  function drillTop() {
    return drillStack[drillStack.length - 1];
  }

  function resetDrill() {
    drillStack = [{
      label: 'All', fileIds: null, subsetId: null, count: 0,
      nClusters: null,
    }];
  }

  // --- busy locks ----------------------------------------------------------
  //
  // #89 stopped two rebuilds from running at once, but it only disabled the
  // toolbar controls; the legend's "Drill in" / "Tag..." actions and the
  // breadcrumbs stayed live. Pressing "Drill in" twice during a rebuild made
  // the second press read groupFileIds() off the OLD mapData, pushing the wrong
  // file ids onto drillStack -- no crash after #89's coalescing, just a wrong
  // answer. These helpers close that half. (Those two actions are now items on
  // the row's right-click menu, which reads mapBusy when it is opened; what is
  // left here are the toolbar buttons and the breadcrumbs.)
  //
  // A blanket `disabled = false` on release would also clear each button's own
  // reason for being disabled (a cluster below MIN_DRILL_PARTS, the crumb for
  // the level we are already on, a tag request in flight). So every lockable
  // button carries that reason in data-lock-off and is always evaluated as
  // `busy || own reason`.
  let mapBusy = false;

  function setLockable(el, ownDisabled) {
    el.dataset.lockOff = ownDisabled ? '1' : '';
    el.disabled = mapBusy || Boolean(ownDisabled);
  }

  function applyBusyLocks() {
    if (!legendBodyEl) return;
    for (const el of legendBodyEl.querySelectorAll('[data-lock-off]')) {
      el.disabled = mapBusy || el.dataset.lockOff === '1';
    }
  }

  // --- legend hover preview -----------------------------------------------
  //
  // Past 24 clusters the palette repeats (#87), so a row's colour no longer
  // identifies it on the map. Hovering a row instead shows only that category
  // for as long as the pointer stays on it. This is a way of LOOKING, not a
  // setting: groupChecked is never written, so leaving the row restores exactly
  // what the checkboxes say. In model mode the preview does two things: it
  // hides the models of the other categories (they stay loaded), and after a
  // longer delay it LOADS the hovered category's own parts up to the model
  // budget, so a cluster can be inspected even when nothing of it was loaded --
  // after the None button, say. It deliberately does not go through
  // scheduleModelRefresh(), which would unload and reload hundreds of parts on
  // every hover; only the models the preview itself added are unloaded when the
  // pointer leaves.
  // The preview's subject. `peekKey` identifies it -- a legend row uses the
  // category's node_name, a tree row uses `tree:<node.id>` -- and `peekNames`
  // is the set of categories it covers (one for a legend row, up to all of them
  // for the root of Ward's tree). Two names rather than one because a tree node
  // is not a category: it has no node_name of its own to be identified by.
  let peekKey = null;
  let peekNames = new Set();
  let peekTimer = null;
  // file id of the centre part currently pointed out, so ending the preview
  // knows what to un-highlight.
  let peekCentreFileId = null;
  // The second, longer timer: the one that LOADS the hovered cluster's parts.
  let peekModelTimer = null;
  // The models that load added, and only those -- leaving the preview unloads
  // exactly this set. The models the ordinary selection loaded stay put (they
  // are merely hidden), because unloading and re-fetching them on every hover
  // would make walking down 24 clusters unusable. The cost is that during a
  // hover both sets are in the viewer at once, which is accepted here.
  let peekModelFileIds = new Set();
  // Bumped to cancel a hover load that has not run yet, or whose turn on the
  // chain comes after the pointer has already moved on.
  let peekModelSeq = 0;

  function clearPeekTimer() {
    if (peekTimer !== null) {
      window.clearTimeout(peekTimer);
      peekTimer = null;
    }
    if (peekModelTimer !== null) {
      window.clearTimeout(peekModelTimer);
      peekModelTimer = null;
    }
  }

  // `names` is the set of categories to show, `key` identifies the row that
  // asked (so a load can tell whether the pointer is still on it), and
  // `centreFileId` is the part to point out as the middle of them.
  function schedulePeek(names, row, { key, centreFileId = null } = {}) {
    clearPeekTimer();
    // The search map's overlaid models are the search hits, not a legend
    // category: a per-category preview has nothing meaningful to show, it
    // would load extra models that the search colouring does not cover, and
    // applyGroupVisibility() deliberately does not restore models from the
    // checkboxes there. No preview at all is the honest answer.
    if (searchMode) return;
    if (!names || names.size === 0) return;
    // A short delay so sweeping the pointer down the legend does not fire a
    // preview for every row it crosses.
    peekTimer = window.setTimeout(() => {
      peekTimer = null;
      if (!active || !mapViewer || mapBusy) return;
      peekKey = key;
      peekNames = names;
      // groupChecked's keys are exactly the categories of the current map,
      // which is the range the viewer is allowed to touch. Anything else named in the
      // SCS -- in particular the categories' ancestors -- must be left alone;
      // hiding one of those would take its whole subtree, including the
      // category being previewed, off the screen.
      mapViewer.setGroupsVisible(names, new Set(groupChecked.keys()));
      if (displayMode === 'models') peekMapModelGroups(names);
      if (row) row.classList.add('map-legend-row-peek');
      markPeekCentre(centreFileId);
      if (zoomOnHover) focusGroups(names);
    }, PEEK_DELAY_MS);
    // A second, independent timer for the heavy half. The visibility preview
    // above still fires at 250 ms whatever this one does.
    if (displayMode === 'models') {
      peekModelTimer = window.setTimeout(() => {
        peekModelTimer = null;
        schedulePeekModels(key);
      }, PEEK_MODEL_DELAY_MS);
    }
  }

  // Put the hovered row's load on the model chain. Everything is re-checked
  // when the turn actually comes, because the pointer may have moved on while
  // an earlier task was still running.
  function schedulePeekModels(key) {
    const seq = ++peekModelSeq;
    enqueueModelTask(async () => {
      if (seq !== peekModelSeq) return; // overtaken by a newer hover
      if (!active || displayMode !== 'models') return;
      if (mapBusy || !mapData || peekKey !== key) return;
      // peekNames, not the captured `names`: the visibility half of the preview
      // has already run by now (it fires at the shorter delay), and it is the
      // one that decided what is on screen. Loading for anything else would put
      // models into a scene that is hiding their categories.
      await loadPeekModels(peekNames, () => seq === peekModelSeq && peekKey === key);
    });
  }

  // Pick representatives from the hovered category alone and load whatever is
  // not on screen yet. Without this, hovering a row after switching every
  // category off (the None button) left the map as bare points, because the
  // preview only ever toggled models that some earlier selection happened to
  // have loaded.
  //
  // There is one cap for the whole scene, modelBudget(), and this stays inside
  // it: the ordinary selection stops at baseModelBudget() and so leaves
  // peekReserve() of room, and what is loaded here is measured against what is
  // actually in the viewer rather than given a budget of its own. Hovering on
  // does not accumulate either -- moving to another row runs endPeek() on the
  // same task chain, so the previous row's models are removed before these are
  // counted.
  async function loadPeekModels(names, stillWanted) {
    const name = getIndexName ? getIndexName() : null;
    if (!name || !mapData) return;
    const candidates = (mapData.points || [])
      .map((p, i) => ({ p, i }))
      .filter(({ p, i }) => {
        const group = groupOfIndex(i);
        if (!group || !names.has(group.node_name)) return false;
        // Drop what is already in the scene BEFORE picking, not after. The
        // picker is deterministic (centre-first), so picking from the full
        // cluster hands back exactly the parts the ordinary selection already
        // loaded, and filtering afterwards leaves nothing to add.
        return !modelFileIds.has(p.file_id) && !peekModelFileIds.has(p.file_id);
      });
    if (candidates.length === 0) return;
    // One cap for the whole scene: never hold more than modelBudget() models,
    // before or after a drill. The ordinary selection stops at
    // baseModelBudget(), so there is normally peekReserve() of room here; the
    // subtraction is what guarantees the cap even when a previous preview's
    // removal has not finished yet.
    const headroom = modelBudget() - modelFileIds.size - peekModelFileIds.size;
    if (headroom <= 0) return;
    // Same picker as the ordinary selection, so the Outliers setting keeps
    // choosing between centre-first and spread sampling here too. No filter
    // afterwards: the candidates were already narrowed to what is absent.
    const picked = selectRepresentatives(candidates, headroom, centreByClusterMap());
    const added = modelPartsFromEntries(name, picked);
    if (added.length === 0) return;
    if (!stillWanted()) return;
    // Recorded BEFORE the await: if the pointer leaves while the models are in
    // flight, endPeek() must still find these ids to unload. It runs after this
    // task on the same chain, so the removal cannot overtake the load.
    const ids = added.map((part) => part.fileId);
    for (const fileId of ids) peekModelFileIds.add(fileId);
    await addMapModels(added, {
      onModelClick,
      onModelActivate,
      onPointClick: onModelPointClick,
      onPointActivate: onModelPointActivate,
      targetDiag: modelTargetDiag || undefined,
      append: true,
    });
    // These models arrive in the colours the server baked into their SCS, so
    // the repaint below must send their family's colour even though nothing
    // about the tree changed.
    forgetPaintedModelColors();
    // New models arrive visible, so re-assert "these categories only" now that
    // they exist.
    peekMapModelGroups(names);
    // They also arrive in the colours the server baked into the SCS, one per
    // cluster, so without this the preview's own parts would be a different
    // colour from the rest of the family already on screen. Same conditions as
    // every other repaint; the call itself checks them.
    repaintTreeModelColors();
    // markPeekCentre() may have had to ring the point because the centre part
    // was not loaded. If this load brought it in, highlight the model instead.
    if (peekCentreFileId !== null && ids.includes(peekCentreFileId)) {
      if (setMapModelHighlighted(peekCentreFileId, true) && mapViewer) {
        mapViewer.clearCentreMarker();
      }
    }
  }

  // Fit the camera to the hovered cluster's own points. Runs on the SAME timer
  // as the visibility preview, so sweeping the pointer down the legend does not
  // send the camera chasing every row it crosses.
  //
  // The box is computed from the /map coordinates this module already holds,
  // not from the scene: the points are baked into one node per category, so the
  // viewer has no per-cluster bounding to ask for.
  //
  // Deliberately does not restore the camera afterwards. Returning to the
  // overview on mouseleave would bounce the view through the overview between
  // every two rows, and would also snatch the camera back the moment the
  // pointer is moved off the legend to orbit what was just zoomed to. Home
  // goes back to the overview in one click.
  //
  // The frame is sized from the cluster and from this layout's spacing, never
  // from a constant share of the [-1, 1] cube -- see CLUSTER_FOCUS_PAD_FRACTION.
  function focusGroups(names, { onlyFileIds = null } = {}) {
    if (!mapViewer || !mapData) return;
    const points = mapData.points || [];
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    let found = 0;
    for (let i = 0; i < points.length; i++) {
      // Two ways of naming the same thing: a set of categories (a hover), or a
      // set of parts (a search hit). The framing below is identical either way
      // -- the minimum half-size floor is what makes a single point land at a
      // sensible distance instead of filling the screen with one part.
      if (onlyFileIds) {
        if (!points[i] || !onlyFileIds.has(points[i].file_id)) continue;
      } else {
        const group = groupOfIndex(i);
        if (!group || !names.has(group.node_name)) continue;
      }
      const p = points[i];
      const xyz = [p.x, p.y, p.z];
      for (let a = 0; a < 3; a++) {
        if (!Number.isFinite(xyz[a])) return;
        if (xyz[a] < min[a]) min[a] = xyz[a];
        if (xyz[a] > max[a]) max[a] = xyz[a];
      }
      found += 1;
    }
    if (found === 0) return;
    const lo = [0, 0, 0];
    const hi = [0, 0, 0];
    // The floor is in this layout's own units, and the margin is a share of the
    // cluster, so the same rule frames a long PCA arc and a tight UMAP knot
    // alike.
    const minHalf = Math.max(
      CLUSTER_FOCUS_MIN_HALF_FLOOR,
      CLUSTER_FOCUS_MIN_SPACINGS * (layoutSpacing > 0 ? layoutSpacing : 0),
    );
    for (let a = 0; a < 3; a++) {
      const centre = (min[a] + max[a]) / 2;
      const half =
        Math.max((max[a] - min[a]) / 2, minHalf) * (1 + CLUSTER_FOCUS_PAD_FRACTION);
      lo[a] = centre - half;
      hi[a] = centre + half;
    }
    // Not awaited: the camera animation must not hold up the rest of the
    // preview, and a newer hover is allowed to overtake this one.
    Promise.resolve(mapViewer.focusBounding(lo, hi)).catch(() => {});
  }

  // Point out the real part closest to the hovered thing's centroid, so the
  // hover says not just "these parts" but "and this is the one at the middle of
  // them". A loaded model is highlighted in place; otherwise -- cloud mode, or
  // a part that did not make the model budget -- the point itself is ringed.
  // The caller passes the file id, because a legend row and a tree node carry
  // it under different names (centre_file_id / centre).
  function markPeekCentre(fileId) {
    if (!fileId || !mapData) return;
    const point = (mapData.points || []).find((p) => p.file_id === fileId);
    if (!point) return;
    let marked = false;
    if (displayMode === 'models') marked = setMapModelHighlighted(fileId, true);
    if (!marked) mapViewer.showCentreMarker(point.x, point.y, point.z);
    peekCentreFileId = fileId;
  }

  // `restore: false` is for the cases where the viewer the preview applied to
  // is about to be thrown away (a rebuild, or going busy): the new map loads
  // with everything visible and applyGroupVisibility() then sets the real
  // state, so restoring against the old viewer would be meaningless at best.
  function endPeek({ restore = true } = {}) {
    clearPeekTimer();
    // Cancel any hover load that has not run yet, and take ownership of the
    // models this preview added. Taking the set before the restore below means
    // a load that is still in flight cannot add to a set nobody will unload.
    peekModelSeq += 1;
    const peeked = [...peekModelFileIds];
    peekModelFileIds = new Set();
    if (legendBodyEl) {
      for (const el of legendBodyEl.querySelectorAll('.map-legend-row-peek')) {
        el.classList.remove('map-legend-row-peek');
      }
    }
    // The tree's cut rows preview the same way, so the highlight can be there
    // instead; clearing only the legend would leave a row lit for good.
    if (treeBodyEl) {
      for (const el of treeBodyEl.querySelectorAll('.map-legend-row-peek')) {
        el.classList.remove('map-legend-row-peek');
      }
    }
    if (peekKey !== null && restore) {
      // The one definition of "what should be visible" -- it restores both the
      // point cloud and the overlaid models from groupChecked, so a checkbox
      // toggled DURING the preview is honoured, and there is no second copy of
      // this rule to drift out of step. It guards on mapViewer/mapData itself.
      applyGroupVisibility();
      if (peekCentreFileId !== null) {
        // Both are harmless when the thing they address is not there, so there
        // is no need to remember which of the two was used.
        setMapModelHighlighted(peekCentreFileId, false);
        if (mapViewer) mapViewer.clearCentreMarker();
      }
    }
    // With `restore: false` the viewer itself is about to go, so detaching the
    // models one by one would be work for nothing; dropping the ids is enough.
    if (restore && peeked.length) {
      enqueueModelTask(async () => {
        await removeMapModels(peeked);
        // Only restore if no NEWER preview has taken over while the removal was
        // in flight. Sweeping down the legend fires mouseleave on one row and
        // mouseenter on the next, and the next row's preview applies its own
        // setGroupsVisible() after ~250 ms -- long before this task's turn on
        // the model chain comes up. Restoring here regardless would put every
        // checked category back on screen underneath an active preview. The
        // overlaid models recover (loadPeekModels re-asserts the group at the
        // end) but the point cloud has no such path, so it is the points that
        // were left showing. When a preview IS active, its own setGroupsVisible
        // is the current truth; its endPeek() restores synchronously later.
        if (peekKey === null) applyGroupVisibility();
      });
    }
    peekKey = null;
    peekNames = new Set();
    peekCentreFileId = null;
    // A search result outlives a hover. The restore above clears the centre
    // marker and the model highlight without knowing who put them there, so a
    // single sweep over the panel would otherwise lose the found part.
    if (restore) markFoundPart();
  }

  // Fetch the map JSON and load the matching SCS, both with the same
  // parameters so a point's index is consistent between them.
  // -- Ward's tree ---------------------------------------------------------

  // How far the tree is open when it first arrives, counted in depth from the
  // root rather than in Ward levels. Depth 0 and 1 open means the frontier --
  // the deepest nodes on screen -- starts at depth 2: a handful of large
  // families, which is what the map should be coloured by before anything has
  // been expanded.
  const TREE_OPEN_TO_DEPTH = 1;

  // One line of plain text in the tree panel. Reuses .map-tree-row rather than
  // introducing a class for it: these lines sit in the same list as the nodes
  // and want the same spacing.
  function treeNote(body, text) {
    const row = document.createElement('div');
    row.className = 'map-tree-row';
    row.style.opacity = '0.6';
    row.textContent = text;
    body.appendChild(row);
    // Returned so a caller can add a tooltip; the note itself needs none.
    return row;
  }

  // Which tree nodes ARE the categories the legend is showing, as a Map from
  // node id to the legend group it equals -- or null when the cut cannot be
  // identified.
  //
  // The legend is one cut of this tree, so each of its categories should be a
  // whole node. That is checked rather than assumed: a node qualifies only if
  // every one of its members carries the same category, it holds all of that
  // category's parts, and the category holds no others. Any mismatch gives up
  // on the whole mapping, which is normal -- an Outliers split, for one, makes
  // a category that is no node of the tree.
  //
  // Cost is members + nodes * categories, because the membership is tallied
  // from the leaves and rolled up through the parents in one descending pass
  // (a node's id is always larger than its parent's). No node's member set is
  // ever materialised.
  function treeCutNodes() {
    // Tag colouring draws categories that are not cuts of this tree at all.
    if (colorBy !== 'cluster') return null;
    if (!treeData || !mapData) return null;
    const groups = mapData.groups || [];
    const points = mapData.points || [];
    const fileIds = treeData.file_ids || [];
    const leaf = treeData.leaf || [];
    const nodes = treeData.nodes || [];
    if (!groups.length || !points.length || !fileIds.length || !nodes.length) return null;
    if (leaf.length !== fileIds.length) return null;

    const catOfGroup = new Map(groups.map((group, i) => [group, i]));
    const memberOfFileId = new Map(fileIds.map((fid, i) => [fid, i]));
    const catOfMember = new Int32Array(fileIds.length).fill(-1);
    for (let pi = 0; pi < points.length; pi += 1) {
      const member = memberOfFileId.get(points[pi].file_id);
      if (member === undefined) continue;
      const group = groupOfIndex(pi);
      if (!group) continue;
      const cat = catOfGroup.get(group);
      if (cat === undefined) continue;
      catOfMember[member] = cat;
    }

    const G = groups.length;
    const ordOfId = new Map(nodes.map((node, i) => [node.id, i]));
    const tally = new Int32Array(nodes.length * G);
    for (let member = 0; member < catOfMember.length; member += 1) {
      const cat = catOfMember[member];
      if (cat < 0) return null;
      const ord = ordOfId.get(leaf[member]);
      if (ord === undefined) return null;
      tally[ord * G + cat] += 1;
    }
    // Descending by id, so a node is complete before its parent is reached.
    const byIdDesc = [...nodes].sort((a, b) => b.id - a.id);
    for (const node of byIdDesc) {
      if (node.parent === -1) continue;
      const ord = ordOfId.get(node.id);
      const parentOrd = ordOfId.get(node.parent);
      if (ord === undefined || parentOrd === undefined) return null;
      for (let cat = 0; cat < G; cat += 1) {
        tally[parentOrd * G + cat] += tally[ord * G + cat];
      }
    }

    const cutOf = new Map();
    const nodeOfCat = new Map();
    for (const node of nodes) {
      const ord = ordOfId.get(node.id);
      let only = -1;
      let total = 0;
      for (let cat = 0; cat < G; cat += 1) {
        const n = tally[ord * G + cat];
        if (!n) continue;
        total += n;
        if (only >= 0) {
          only = -1;
          break;
        }
        only = cat;
      }
      if (only < 0) continue;
      if (total !== node.count) continue;
      if (node.count !== groups[only].count) continue;
      // A category that matched twice is no cut at all: one of the two is an
      // ancestor carrying the same members, so neither identifies the cut.
      if (nodeOfCat.has(only)) return null;
      nodeOfCat.set(only, node.id);
      cutOf.set(node.id, groups[only]);
    }
    // Every category has to be accounted for. A partial match means the legend
    // is not this tree's cut, and half a mapping would be worse than none.
    if (nodeOfCat.size !== G) return null;
    return cutOf;
  }

  // Which legend categories sit under each node, as a Map from node id to an
  // array of groups. Built by the same descending-id roll-up treeCutNodes()
  // uses, so the cost is one pass over the nodes plus the categories carried
  // through them -- no node's membership is ever expanded into file ids.
  //
  // This is what lets a row above the cut act on its whole family: a node's
  // parts are exactly the parts of the categories listed here.
  function treeCategoriesOf(cutOf, nodes) {
    const catsOf = new Map();
    for (const node of nodes) catsOf.set(node.id, []);
    const byIdDesc = [...nodes].sort((a, b) => b.id - a.id);
    for (const node of byIdDesc) {
      const own = catsOf.get(node.id);
      const cut = cutOf.get(node.id);
      // A cut node's children, if it has any, hold no categories of their own
      // (a category matches exactly one node), so this cannot double-count.
      if (cut) own.push(cut);
      if (node.parent === -1) continue;
      const up = catsOf.get(node.parent);
      if (up) up.push(...own);
    }
    return catsOf;
  }

  // The mean position of everything under a frontier node, in map coordinates.
  // Used only to keep repeated palette colours apart, so the groups' contiguous
  // point runs are averaged directly rather than resolved part by part.
  function treeCentroid(cats) {
    const points = mapData.points || [];
    let x = 0;
    let y = 0;
    let z = 0;
    let n = 0;
    for (const group of cats) {
      const start = group.offset || 0;
      const end = start + (group.count || 0);
      for (let i = start; i < end && i < points.length; i += 1) {
        const p = points[i];
        if (!p) continue;
        x += p.x;
        y += p.y;
        z += p.z;
        n += 1;
      }
    }
    return n ? [x / n, y / n, z / n] : null;
  }

  // Give every frontier node a palette colour, reusing the colour a node
  // already had so that expanding one branch leaves the rest of the map alone.
  //
  // New nodes are placed greedily: each takes the palette entry whose nearest
  // same-coloured neighbour is furthest away, ties going to the lower index.
  // With 24 colours and a frontier of 24 or fewer every node ends up with a
  // colour of its own; past that the palette repeats, and this is what keeps
  // the repeats at opposite ends of the map instead of side by side.
  function assignTreeColors(frontier, catsOf) {
    const centres = new Map();
    for (const node of frontier) {
      centres.set(node.id, treeCentroid(catsOf.get(node.id) || []));
    }
    const next = new Map();
    const held = new Map(); // palette index -> centres already using it
    for (let i = 0; i < SHAPE_MAP_PALETTE.length; i += 1) held.set(i, []);
    const fresh = [];
    for (const node of frontier) {
      const kept = treeColorOf.get(node.id);
      if (kept === undefined) {
        fresh.push(node);
        continue;
      }
      next.set(node.id, kept);
      const centre = centres.get(node.id);
      if (centre) held.get(kept).push(centre);
    }
    for (const node of fresh) {
      const centre = centres.get(node.id);
      let best = 0;
      let bestScore = -Infinity;
      for (let i = 0; i < SHAPE_MAP_PALETTE.length; i += 1) {
        const taken = held.get(i);
        let score = Infinity;
        if (centre) {
          for (const other of taken) {
            const dx = centre[0] - other[0];
            const dy = centre[1] - other[1];
            const dz = centre[2] - other[2];
            const d = dx * dx + dy * dy + dz * dz;
            if (d < score) score = d;
          }
        } else if (taken.length) {
          // No position to judge by: treat the colour as merely used, so an
          // unused one still wins.
          score = 0;
        }
        if (score > bestScore) {
          bestScore = score;
          best = i;
          if (score === Infinity) break;
        }
      }
      next.set(node.id, best);
      if (centre) held.get(best).push(centre);
    }
    // Dropped wholesale rather than merged: ids from a collapsed branch -- or
    // from another index's tree -- would otherwise keep colours reserved.
    treeColorOf = next;
    // The same assignment keyed by category, which is what everything OUTSIDE
    // the tree knows a part by. A model is placed in this colour from the
    // start, so it never appears in its baked-in cluster colour first.
    treeColorByCategory = new Map();
    for (const node of frontier) {
      const rgb = SHAPE_MAP_PALETTE[next.get(node.id) || 0];
      for (const group of catsOf.get(node.id) || []) {
        treeColorByCategory.set(group.node_name, rgb);
      }
    }
    return next;
  }

  // What the viewer was last told to wear, so a repaint can send only what
  // actually changed: category node_name -> index into SHAPE_MAP_PALETTE.
  // Two of them, because they colour different things -- the point cloud's
  // category nodes and the overlaid 3D models -- and each is invalidated by
  // its own events (see forgetPaintedCloudColors / forgetPaintedModelColors).
  // Getting an invalidation wrong shows as a colour that silently never
  // updates, so both are dropped generously rather than cleverly.
  let paintedCloudColor = new Map();
  let paintedModelColor = new Map();

  // Dropped whenever the viewer's category nodes are replaced or the set of
  // categories itself changes -- a new map build (load() assigns mapData), a
  // reloaded point-cloud SCS (showMap() builds fresh nodes), and with them
  // every control that rebuilds: Clustering, Max clusters, Show, Layout,
  // Colour by, View. All of those go through load(), which is why this has one
  // call site there and one at each showMap().
  function forgetPaintedCloudColors() {
    paintedCloudColor = new Map();
  }

  // Dropped whenever models are placed. A model that has just been loaded
  // arrives in the colour the server baked into its SCS, so what this map
  // remembers about it is wrong even when the value matches -- the cache says
  // "already sent", and the band would be skipped while the model stayed in
  // its baked-in colour.
  function forgetPaintedModelColors() {
    paintedModelColor = new Map();
  }

  // Repaint the point cloud so each category wears the colour of the frontier
  // node it belongs to. One call for every category at once: the viewer walks
  // its node map per entry, and 256 separate calls would be 256 walks. Only
  // the categories whose colour actually changed are sent, so a redraw that
  // moves nothing -- the common case, since colours are held across expand and
  // collapse -- touches the viewer not at all.
  function paintTreeColors(frontier, catsOf, colorOf) {
    if (!mapViewer || typeof mapViewer.setMapGroupColors !== 'function') return;
    const entries = [];
    const sent = [];
    for (const node of frontier) {
      const ci = colorOf.get(node.id) || 0;
      for (const group of catsOf.get(node.id) || []) {
        if (paintedCloudColor.get(group.node_name) === ci) continue;
        entries.push({ nodeName: group.node_name, rgb: SHAPE_MAP_PALETTE[ci] });
        sent.push([group.node_name, ci]);
      }
    }
    if (!entries.length) return;
    // Recorded only on success: a failed call leaves the viewer wearing the
    // old colours, and remembering the new ones would skip the retry.
    if (mapViewer.setMapGroupColors(entries)) {
      for (const [nodeName, ci] of sent) paintedCloudColor.set(nodeName, ci);
    }
  }

  // The 3D models overlaid on the map wear the same colours as the cloud. They
  // arrive from the SCS files in the colours the server baked in, which are one
  // cluster each -- several per frontier node -- so without this a frontier
  // family would be several colours in model view and one in cloud view.
  //
  // Skipped on the search map: there the models carry similarity to the query,
  // which is the whole point of that view.
  function paintTreeModelColors(frontier, catsOf, colorOf) {
    if (!mapViewer) return;
    if (displayMode !== 'models' || colorBy !== 'cluster') return;
    if (clustering !== 'ward' || searchMode) return;
    const bands = [];
    const sent = [];
    for (const node of frontier) {
      const ci = colorOf.get(node.id) || 0;
      const fileIds = [];
      for (const group of catsOf.get(node.id) || []) {
        // Diffed per category rather than per band: a frontier node keeps its
        // colour while gaining or losing categories, and only the newcomers
        // need sending.
        if (paintedModelColor.get(group.node_name) === ci) continue;
        fileIds.push(...groupFileIds(group));
        sent.push([group.node_name, ci]);
      }
      if (fileIds.length) bands.push({ rgb: SHAPE_MAP_PALETTE[ci], fileIds });
    }
    if (!bands.length) return;
    // Ids with no model placed are skipped by setMapModelColors(), so the whole
    // family can be listed whether or not it is on screen.
    setMapModelColors(bands);
    for (const [nodeName, ci] of sent) paintedModelColor.set(nodeName, ci);
  }

  // The frontier of the last draw, kept so that models loaded AFTER it -- every
  // model refresh brings in new ones in their baked-in colours -- can be
  // repainted without redrawing the panel.
  let treePaint = null;

  function repaintTreeModelColors() {
    if (!treePaint) return;
    paintTreeModelColors(treePaint.frontier, treePaint.catsOf, treePaint.colorOf);
  }

  // The right-click menu for a panel row, in the tree and in the legend. One
  // at a time, held here rather than in the row that opened it: the rows are
  // discarded on every redraw, and a menu left behind would act on a node that
  // is no longer on screen.
  let rowMenuEl = null;
  let lastRowMenuPoint = null;

  function closeRowMenu() {
    if (!rowMenuEl) return;
    rowMenuEl.remove();
    rowMenuEl = null;
    document.removeEventListener('mousedown', onRowMenuDismiss, true);
    document.removeEventListener('keydown', onRowMenuKey, true);
    // Capture, so scrolling the tree's own container closes it too: the menu is
    // positioned in viewport coordinates and would otherwise be left behind.
    window.removeEventListener('scroll', closeRowMenu, true);
    window.removeEventListener('resize', closeRowMenu);
  }

  function onRowMenuDismiss(event) {
    if (rowMenuEl && rowMenuEl.contains(event.target)) return;
    closeRowMenu();
  }

  function onRowMenuKey(event) {
    if (event.key === 'Escape') closeRowMenu();
  }

  function openRowMenu(event, items) {
    closeRowMenu();
    // Where the menu was raised, kept so an action that opens a panel of its
    // own -- the tag popover -- can appear in the same place instead of
    // somewhere the pointer has to travel to.
    lastRowMenuPoint = { x: event.clientX, y: event.clientY };
    const menu = document.createElement('div');
    menu.className = 'map-row-menu';
    for (const item of items) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'map-row-menu-item';
      button.textContent = item.label;
      // An item that cannot be run is shown rather than hidden, with the
      // reason in its tooltip: a reader who expects the action needs to be
      // told why it is not available, not left to wonder where it went.
      if (item.disabled) button.disabled = true;
      if (item.title) button.title = item.title;
      button.addEventListener('click', () => {
        closeRowMenu();
        item.run();
      });
      menu.appendChild(button);
    }
    menu.style.left = `${event.clientX}px`;
    menu.style.top = `${event.clientY}px`;
    document.body.appendChild(menu);
    rowMenuEl = menu;
    // Measured after it is in the document: a menu opened near the bottom of a
    // short window would otherwise run off the screen with no way to reach it.
    const rect = menu.getBoundingClientRect();
    if (rect.right > window.innerWidth) {
      menu.style.left = `${Math.max(0, window.innerWidth - rect.width - 4)}px`;
    }
    if (rect.bottom > window.innerHeight) {
      menu.style.top = `${Math.max(0, window.innerHeight - rect.height - 4)}px`;
    }
    document.addEventListener('mousedown', onRowMenuDismiss, true);
    document.addEventListener('keydown', onRowMenuKey, true);
    window.addEventListener('scroll', closeRowMenu, true);
    window.addEventListener('resize', closeRowMenu);
  }

  // Bring every drawn checkbox into line with groupChecked. The tree is
  // redrawn right after the two callers below, but the legend's rows are not:
  // without this, a row acted on from its own right-click menu would keep a
  // tick that no longer says what the map shows.
  function syncGroupCheckInputs() {
    for (const [nodeName, inputs] of groupCheckInputs) {
      const on = groupChecked.get(nodeName) !== false;
      for (const el of inputs) {
        if (el) el.checked = on;
      }
    }
  }

  // Switch a whole family of categories on or off from one tree checkbox. The
  // per-category work is the legend checkbox's, done in bulk: a node near the
  // root can hold every category there is, and one call per category would be
  // hundreds of round trips into the viewer.
  //
  // Also reached from a legend row's menu with a single category, which is the
  // same operation with a list of one.
  function setTreeFamilyVisible(cats, on) {
    // End any hover preview FIRST, for the same reason the legend's checkbox
    // does: its unload must join the model chain ahead of the refresh below.
    endPeek();
    clearHeat();
    for (const group of cats) {
      groupChecked.set(group.node_name, on);
      if (legendRemembered) legendRemembered.set(group.node_name, on);
    }
    const all = (mapData && mapData.groups ? mapData.groups : []).map((g) => g.node_name);
    const visible = new Set(all.filter((name) => groupChecked.get(name) !== false));
    if (mapViewer) mapViewer.setGroupsVisible(visible, all);
    if (displayMode === 'models') {
      // The whole visible set in one pass, not one call per category: the
      // per-category call walks every placed model each time.
      setMapModelGroupsVisible(visible);
      scheduleModelRefresh();
    }
    syncGroupCheckInputs();
    if (legendTools) legendTools.refresh();
    // Ancestors and descendants of the node just toggled show a different box
    // now, so the panel is rebuilt rather than patched in place.
    renderTree();
  }

  // Show one family and nothing else. `cats` null means "show every category
  // there is"; an EMPTY array means the opposite, "hide every category". The
  // two are deliberately different: a tree or legend row that isolates nothing
  // wants the whole map back, while a catalogue whose last tile was unticked
  // wants an empty viewer. Showing everything there costs a reload of up to
  // MAX_MODELS parts, which made Clear selection the slowest thing in the tab.
  //
  // The same path as the checkboxes above, written as one pass over the map's
  // categories rather than a hide-everything followed by a show-one: that
  // would be two viewer updates and two model refreshes.
  function setTreeFamilyIsolated(cats) {
    const keep = cats ? new Set(cats.map((group) => group.node_name)) : null;
    endPeek();
    clearHeat();
    const all = (mapData && mapData.groups ? mapData.groups : []).map((g) => g.node_name);
    const visible = new Set();
    for (const name of all) {
      const on = keep ? keep.has(name) : true;
      groupChecked.set(name, on);
      if (legendRemembered) legendRemembered.set(name, on);
      if (on) visible.add(name);
    }
    if (mapViewer) mapViewer.setGroupsVisible(visible, all);
    if (displayMode === 'models') {
      // One pass over the placed models, not one per category: at 2,048
      // clusters the per-category call was over a million iterations and 2,048
      // separate viewer calls for a single click.
      setMapModelGroupsVisible(visible);
      scheduleModelRefresh();
    }
    syncGroupCheckInputs();
    if (legendTools) legendTools.refresh();
    renderTree();
  }

  // Show exactly the families the catalogue has ticked, and open the tree down
  // to each of them.
  //
  // One way only: the catalogue drives the tree, never the other way round.
  // Reflecting the tree's own checkboxes back would both loop and make a tick
  // the user put on a tree row silently clear a catalogue tile.
  //
  // This NEVER rebuilds the map. It changes which categories are visible and
  // which rows are open, both of which are client-side; nothing here touches
  // the cluster count, the clustering or the ward ceiling. (See #194/#204: a
  // selection path that called a setter rebuilt the map on every click and
  // threw away the tree's expansion each time.)
  //
  // The node ids come from /map/catalogue, which is built from the same k_max
  // as the tree, so they name nodes here. Ids that do not are reported back
  // rather than ignored: that means the two are out of step, which the caller
  // has to say out loud.
  async function applyCatalogueSelection(nodeIds) {
    if (!active || clustering !== 'ward') return { ok: false, reason: 'not-ward' };
    if (mapBusy) return { ok: false, reason: 'busy' };
    if (!treeData) {
      // The tree is fetched with the map, but the catalogue can be ticked
      // before one has been drawn (a tag colouring, say). Fetch it at the same
      // ceiling the map was built with -- any other value renumbers the nodes.
      try {
        const level = drillTop();
        const tree = await api.shapeMapTree(getIndexName(), {
          kind,
          onlyFileIds: level.fileIds || null,
          kMax: wardMax,
        });
        if (!active) return { ok: false, reason: 'left' };
        treeData = tree && Array.isArray(tree.nodes) && tree.nodes.length ? tree : null;
        treeError = null;
      } catch (err) {
        treeError = (err && err.message) ? err.message : String(err);
        return { ok: false, reason: 'tree-failed', message: treeError };
      }
    }
    const nodes = treeData && Array.isArray(treeData.nodes) ? treeData.nodes : [];
    if (!nodes.length) return { ok: false, reason: 'no-tree' };
    return applyCatalogueSelectionInner(nodeIds, nodes);
  }

  // The stages of one catalogue selection, split out from the fetch above so
  // that the tree request and the work it enables stay separable.
  function applyCatalogueSelectionInner(nodeIds, nodes) {
    const resolved = (() => {
      const cutOf = treeCutNodes();
      if (!cutOf) return null;
      const catsOf = treeCategoriesOf(cutOf, nodes);
      const parentOf = new Map(nodes.map((node) => [node.id, node.parent]));

      const wanted = [];
      const missing = [];
      for (const raw of nodeIds || []) {
        const id = Number(raw);
        if (!parentOf.has(id)) missing.push(id);
        else wanted.push(id);
      }

      // Step 1 and 2 of the sync: nothing ticked, nothing open. Both are
      // replaced wholesale rather than patched, so a tick left over from a
      // previous selection cannot survive. treeSeeded stays set so the
      // first-draw seeding does not reopen the root's children underneath this.
      treeOpen = new Set();
      treeSeeded = true;
      // Step 3: open the ancestors of each ticked node, so the row is
      // reachable. The node ITSELF is left closed -- it stands for the whole
      // family, and expanding it would scatter that family over its children.
      for (const id of wanted) {
        let up = parentOf.get(id);
        while (up !== undefined && up !== -1) {
          if (treeOpen.has(up)) break; // this ancestor chain is already open
          treeOpen.add(up);
          up = parentOf.get(up);
        }
      }

      // The categories behind the ticked nodes, which the visibility change
      // below is driven from.
      const cats = [];
      const seen = new Set();
      for (const id of wanted) {
        for (const group of catsOf.get(id) || []) {
          if (seen.has(group.node_name)) continue;
          seen.add(group.node_name);
          cats.push(group);
        }
      }
      return { wanted, missing, cats };
    })();
    if (!resolved) return { ok: false, reason: 'no-cut' };
    const { wanted, missing, cats } = resolved;

    // Nothing ticked means an empty viewer here, not the whole map back.
    // Reloading up to MAX_MODELS parts is the single most expensive thing the
    // map does, so Clear selection used to be its slowest operation; emptying
    // it only unloads, and never loads anything back. The whole map is a
    // deliberate act from the tree or the legend, which keep their own
    // "Show all". (A tree or legend row isolating nothing still means "show
    // everything" -- that path passes null, not an empty list.)
    //
    // This also updates the visibility, the checkboxes, the legend and the
    // tree itself.
    setTreeFamilyIsolated(cats);
    // Frame what is left, under the same switch the hover preview obeys. The
    // camera routine is focusGroups(), unchanged and uncopied: a selection that
    // narrows the cloud to 80 points out of 7,928 leaves them a few pixels
    // across, so without this the sync appears to have done nothing. With
    // nothing ticked the map is whole again and there is nothing to frame.
    if (zoomOnHover && cats.length) {
      focusGroups(new Set(cats.map((group) => group.node_name)));
    }
    return { ok: true, groups: wanted.length, categories: cats.length, missing };
  }

  // Draw the tree.
  //
  // The server does NOT repeat each level's membership: a node's members are
  // the file ids its own leaf entries name plus those of all its descendants,
  // which is derived from the parallel `file_ids` / `leaf` arrays. Nothing here
  // needs them yet -- this draws structure only -- so the derivation arrives
  // with the selection, in the next step.
  //
  // Every path draws a line. A panel that is simply empty cannot be told from
  // one that was never filled, so each way of having no tree says which it was.
  function renderTree() {
    const body = treeBodyEl;
    if (!body) return;
    // Same as the legend's scroll container: leaving the list between two rows
    // can skip a row's own mouseleave. Assigned, not added, so redrawing the
    // tree on every expand cannot stack handlers.
    body.onmouseleave = () => endPeek();
    body.replaceChildren();
    // Any open menu belongs to a row that has just been discarded.
    closeRowMenu();
    // Dropped up front: every early return below means there is no frontier to
    // repaint models from, and a stale one would colour by a tree that is gone.
    // The category -> colour map goes with it, so a model placed while no tree
    // is drawn falls back to its cluster colour rather than a vanished family's.
    treePaint = null;
    treeColorByCategory = new Map();
    // The rows these pointed at have just been discarded; every early return
    // below leaves them empty, which is the truth -- there is no tree row to
    // point the revealed part at.
    treeRowOf.clear();
    treeCutNodeOf.clear();
    treeParentOf.clear();
    // Only Ward has a hierarchy, and only the cluster colouring is a cut of it:
    // a tag colouring crosses the hierarchy, so the tree describes nothing the
    // map is showing. Hidden rather than merely empty, so the panel does not
    // keep a box of blank space under the legend.
    const shown = clustering === 'ward' && colorBy === 'cluster';
    body.hidden = !shown;
    if (!shown) {
      return;
    }
    const tree = treeData;
    if (!tree) {
      treeNote(
        body,
        treeError
          ? `Tree unavailable: ${treeError}`
          : 'No hierarchy for this map.',
      );
      return;
    }
    const nodes = Array.isArray(tree.nodes) ? tree.nodes : [];
    if (!nodes.length) {
      treeNote(body, 'Tree is empty.');
      return;
    }

    // Built once per draw rather than searched per node: a 400-node tree would
    // otherwise be a scan of every node for every node.
    const byId = new Map(nodes.map((node) => [node.id, node]));
    const childrenOf = new Map();
    let root = null;
    for (const node of nodes) {
      treeParentOf.set(node.id, node.parent);
      if (node.parent === -1) {
        root = node;
        continue;
      }
      if (!childrenOf.has(node.parent)) childrenOf.set(node.parent, []);
      childrenOf.get(node.parent).push(node.id);
    }
    if (!root) {
      treeNote(body, `Tree has no root (${nodes.length} nodes).`);
      return;
    }

    // Which nodes the legend's categories are, if the legend is a cut of this
    // tree. Computed once per draw and read by every row.
    const cutOf = treeCutNodes();
    if (cutOf) {
      // Read the other way round, so a category can be found without scanning:
      // this is where the revealed part's family enters the tree.
      for (const [id, group] of cutOf) {
        if (group && group.node_name) treeCutNodeOf.set(group.node_name, id);
      }
    }
    // Only Ward's own clustering is a cut of this tree. A tag colouring crosses
    // it, so there is nothing to hang on the nodes: the tree is then structure
    // alone -- no swatches, no checkboxes, no recolouring of the cloud.
    const catsOf = cutOf ? treeCategoriesOf(cutOf, nodes) : null;

    // First draw of a tree: open the root and its children rather than
    // everything, which on a deep index would be hundreds of rows at once.
    if (!treeSeeded) {
      treeSeeded = true;
      const seed = (node, depth) => {
        if (depth > TREE_OPEN_TO_DEPTH) return;
        treeOpen.add(node.id);
        for (const id of childrenOf.get(node.id) || []) {
          const kid = byId.get(id);
          if (kid) seed(kid, depth + 1);
        }
      };
      seed(root, 0);
    }

    // The frontier: the deepest nodes on screen, being those with no children
    // and those whose children are collapsed. They tile the whole population
    // without overlapping, which is what makes them the thing to colour -- one
    // colour each covers every part exactly once.
    const frontier = [];
    const walkFrontier = (node) => {
      const kids = childrenOf.get(node.id) || [];
      if (!kids.length || !treeOpen.has(node.id)) {
        frontier.push(node);
        return;
      }
      for (const id of kids) {
        const kid = byId.get(id);
        if (kid) walkFrontier(kid);
      }
    };
    walkFrontier(root);
    const colorOf = catsOf ? assignTreeColors(frontier, catsOf) : new Map();
    const onFrontier = new Set(frontier.map((node) => node.id));
    treePaint = catsOf ? { frontier, catsOf, colorOf } : null;
    if (catsOf) {
      paintTreeColors(frontier, catsOf, colorOf);
      paintTreeModelColors(frontier, catsOf, colorOf);
    }

    const draw = (node, depth) => {
      const kids = childrenOf.get(node.id) || [];
      const open = treeOpen.has(node.id);
      const cut = cutOf ? cutOf.get(node.id) : null;
      const row = document.createElement('div');
      row.className = cut ? 'map-tree-row map-tree-row-cut' : 'map-tree-row';
      treeRowOf.set(node.id, row);
      row.style.marginLeft = `${depth * 10}px`;

      if (kids.length) {
        const caret = document.createElement('button');
        caret.type = 'button';
        caret.className = 'map-tree-caret';
        // Escaped rather than written literally: these two were literal
        // characters once and an editor that read this file as a legacy
        // Japanese code page and saved it back as UTF-8 destroyed them. An
        // escape survives that, and it names the character for the next
        // reader without requiring a font that can draw it.
        caret.textContent = open ? '\u25BE' : '\u25B8';   // down-pointing / right-pointing triangle
        caret.title = open ? 'Collapse' : 'Expand';
        caret.addEventListener('click', () => {
          if (treeOpen.has(node.id)) treeOpen.delete(node.id);
          else treeOpen.add(node.id);
          renderTree();
        });
        row.appendChild(caret);
      } else {
        // Same width as a caret, so the counts of childless nodes line up with
        // their siblings' instead of shifting left.
        const pad = document.createElement('span');
        pad.className = 'map-tree-caret';
        pad.textContent = '';
        row.appendChild(pad);
      }

      // Every drawn node can be switched off, not just the ones that are a
      // category: a node stands for its whole family, and hiding a family is
      // the question a hierarchy invites. Mixed families show the box as
      // indeterminate rather than guessing which way they lean.
      if (catsOf) {
        const cats = catsOf.get(node.id) || [];
        if (cats.length) {
          const check = document.createElement('input');
          check.type = 'checkbox';
          check.className = 'map-legend-check';
          const on = cats.filter((g) => groupChecked.get(g.node_name) !== false).length;
          check.checked = on > 0;
          check.indeterminate = on > 0 && on < cats.length;
          check.title =
            cats.length > 1
              ? `Show the ${cats.length} clusters under this node`
              : 'Show this cluster';
          check.addEventListener('change', () => {
            setTreeFamilyVisible(cats, check.checked);
          });
          row.appendChild(check);
        }
      }
      if (onFrontier.has(node.id) && colorOf.has(node.id)) {
        // The colour this family is wearing on the map. Only the frontier gets
        // one: a node above it holds several colours at once, and a single
        // swatch would claim otherwise.
        const [r, g, b] = SHAPE_MAP_PALETTE[colorOf.get(node.id)];
        const swatch = document.createElement('span');
        swatch.className = 'map-legend-swatch';
        swatch.style.background = `rgb(${r}, ${g}, ${b})`;
        row.appendChild(swatch);
      }
      // Every node with categories under it previews on hover, not just the
      // ones that ARE a category: the whole point of the tree is to ask "what
      // is in here?" of an inner node. The key keeps tree rows apart from
      // legend rows -- a tree node has no node_name of its own -- and node.centre
      // is the file id of the part whose name the row is showing.
      const peekCats = catsOf ? catsOf.get(node.id) || [] : [];
      if (peekCats.length) {
        const peekSet = new Set(peekCats.map((g) => g.node_name));
        row.addEventListener('mouseenter', () => schedulePeek(peekSet, row, {
          key: `tree:${node.id}`,
          centreFileId: node.centre,
        }));
        row.addEventListener('mouseleave', () => endPeek());
      }
      // How deep in the tree this row sits, root at L1. Dimmed inline rather
      // than through a class: it is a hint, not a column, and .map-row-centre
      // would let it be the first thing truncated away.
      const level = document.createElement('span');
      level.textContent = `L${depth + 1}`;
      level.style.opacity = '0.6';
      level.style.marginRight = '4px';
      row.appendChild(level);
      const parts = [`${node.count.toLocaleString()} parts`];
      // Four decimals, not two: 0.95 is where a user decides a family is
      // tight enough to accept without opening it, and 0.9499 rounded to
      // "0.95" would answer that question wrongly.
      if (typeof node.cohesion === 'number') parts.push(`coh ${node.cohesion.toFixed(4)}`);
      const label = document.createElement('span');
      label.textContent = parts.join(' \u00b7 ');
      row.appendChild(label);
      // The merge height stays off the row: it is a fact about the tree rather
      // than about the family, and in units nobody reads off a panel. It is
      // not in a tooltip either -- a tooltip over every row of a long list
      // follows the pointer around and covers the rows being read.

      if (node.filename) {
        // The centre part: what this family looks like, in one name.
        const centre = document.createElement('span');
        centre.className = 'map-row-centre';
        centre.textContent = `\u00B7 ${node.filename}`;   // middle dot separator, escaped for the reason above
        row.appendChild(centre);
      }
      // After the centre name, so the row reads identity, then structure. The
      // actions are on the right-click menu instead of buttons: every node can
      // be acted on, and a row of buttons on each of hundreds of rows is not a
      // panel. The menu is the same for every node, which is why Isolate, Hide
      // and Show all are here rather than in a toolbar above.
      row.addEventListener('contextmenu', (event) => {
        const cats = catsOf ? catsOf.get(node.id) || [] : [];
        if (!cats.length) return;
        event.preventDefault();
        const items = [
          {
            label: 'Isolate',
            run: () => setTreeFamilyIsolated(cats),
          },
          {
            label: 'Hide',
            run: () => setTreeFamilyVisible(cats, false),
          },
          {
            label: 'Show all',
            run: () => setTreeFamilyIsolated(null),
          },
        ];
        // The heat map measures one cluster's parts against ONE centre, so it
        // only means anything on a node that IS a cluster, and it colours the
        // overlaid models, so it needs the model view.
        if (heatState) {
          items.push({ label: 'Clear heat map', run: () => clearHeat() });
        } else if (cut && displayMode === 'models') {
          items.push({ label: 'Heat map', run: () => applyHeat(cut, null) });
        }
        items.push({
          label: `Tag these ${node.count} parts\u2026`,
          run: () => {
            const fileIds = [];
            for (const group of cats) fileIds.push(...groupFileIds(group));
            tagParts(
              `${node.count} parts at h ${Number(node.height).toFixed(2)}`,
              fileIds,
            );
          },
        });
        // A tree node stands for every category under it, so pinning it pins
        // them all. Partly pinned reads as not pinned, so the item offers to
        // add the rest rather than to take the pinned ones away.
        for (const item of focusMenuItems(cats)) items.push(item);
        openRowMenu(event, items);
      });

      body.appendChild(row);
      if (!open) return;
      for (const id of kids) {
        const kid = byId.get(id);
        if (kid) draw(kid, depth + 1);
      }
    };

    // Which population this tree describes, how finely the server cut it, and
    // how many families the map is currently coloured by. Permanent, not a
    // debugging aid: the panel otherwise gives no way to tell a drill-down's
    // tree from the whole index's, nor how far it has been opened. Under Ward
    // this is the map's only summary line -- the legend's is suppressed -- so
    // it carries the cluster counts that line used to report.
    const leaves = nodes.filter((node) => !childrenOf.has(node.id));
    const bigLeaves = leaves.filter((node) => (node.count || 0) >= LEGEND_SUMMARY_MIN);
    const note = treeNote(
      body,
      `${Number(tree.count).toLocaleString()} parts \u00b7 `
      + `${leaves.length.toLocaleString()} leaf clusters `
      + `(${bigLeaves.length.toLocaleString()} with ${LEGEND_SUMMARY_MIN}+ parts) \u00b7 `
      + `${frontier.length.toLocaleString()} groups shown`,
    );
    if (note) {
      note.title =
        'parts - how many parts this tree covers\n'
        + 'leaf clusters - the finest split the server baked in; the bracket '
        + 'counts those large enough to be worth one tag\n'
        + 'groups shown - the families currently open, one colour each\n'
        + 'Ln on a row is its depth in this tree (the root is L1), not the '
        + 'height Ward cut it at; two rows at the same depth can have been cut '
        + 'at different heights.';
    }
    draw(root, 0);
    // The rows exist now, so the revealed part can be pointed at. Expanding a
    // branch redraws and lands here again, which is how the mark follows the
    // part down to the deepest row that covers it.
    highlightRevealed({ scroll: false });
    // Same reason: the pin marks belong to rows that have just been rebuilt.
    markFocusedRows();
  }

  async function load() {
    const name = getIndexName();
    if (!name) return;

    // Take the pending camera for this rebuild (a View toggle set it); every
    // other caller leaves it null, so they frame the iso overview as before.
    const initialCamera = pendingCamera;
    pendingCamera = null;

    try {
      const level = drillTop();
      // In Untagged-only mode the population is pinned to the ids the last
      // build covered, and the pin -- not `tagged` -- is what selects them from
      // then on. Re-asking the server for "whatever is untagged now" would drop
      // the parts tagged since, so merely returning from a part, or toggling
      // the View, would silently apply the rebuild the Reload button exists to
      // defer. A drill level pins its own ids for the same reason: they were
      // chosen from the map as it stood.
      // Untagged-only is a cluster-colouring tool: in tag colouring the map
      // would be every part that carries no tag, coloured by the tag it does
      // not have. The host hides the control there, but the state survives the
      // switch, so the request has to refuse it too rather than trusting the
      // UI.
      const untaggedNow = untaggedOnly && colorBy === 'cluster';
      const pinned = untaggedNow && !level.fileIds ? untaggedPin : null;
      const onlyFileIds = level.fileIds || pinned;
      // Whether this load asks the index what is untagged NOW, which is the
      // only case in which the outstanding-tag count is answered.
      const refreshesPopulation = !untaggedNow || !onlyFileIds;
      // `tagged: false` asks the server for the untagged parts only; undefined
      // is dropped by api.js buildUrl(), so with the option off the request is
      // byte for byte the one that was sent before it existed.
      const params = {
        dims: MAP_DIMS,
        nClusters,
        kind,
        // Outliers is a k-means tool: it carves a category out of each cluster
        // by distance from its centroid. Ward cuts by height, where each
        // node's cohesion already answers the same question, and an extra
        // category would break the one-to-one match between the map's
        // categories and the tree's nodes that its colours, checkboxes and
        // tagging all stand on. The slider's value is kept, not cleared, so
        // switching back to k-means restores it.
        outlierThreshold: effectiveOutlierThreshold() || undefined,
        // undefined at the default, so a PCA request is byte for byte the one
        // this client sent before the layout could be chosen -- and so it still
        // hits every .scs already cached on the server.
        projection: projection === 'pca' ? undefined : projection,
        // undefined at the default, so a k-means request is byte for byte the
        // one this client sent before the method could be chosen.
        clustering: clustering === 'kmeans' ? undefined : clustering,
        // Ward's cut height. undefined for k-means, where the server ignores it
        // anyway, so a k-means request stays byte for byte what it was. Always
        // the deepest cut now: the tree view shows every coarser grouping, so
        // there is nothing left for a cut height to choose.
        granularity: clustering === 'ward' ? WARD_GRANULARITY : undefined,
        // The ceiling that granularity is read against, and the same value
        // /map/tree is asked for below. undefined at the default, so a request
        // that does not move it stays byte for byte what it was.
        wardMax: clustering === 'ward' && wardMax !== DEFAULT_WARD_MAX
          ? wardMax
          : undefined,
        tagged: untaggedNow && !onlyFileIds ? false : undefined,
      };
      // include_tags so the info panel can show a picked part's tags without a
      // second request. color_by matches the SCS below, so the legend colours
      // are the ones baked into the point cloud. Neither affects the point order.
      mapData = await api.shapeMap(name, {
        ...params,
        includeTags: true,
        colorBy,
        onlyFileIds,
      });
      // What this map was asked for, so the legend's similarity order can be
      // asked for the same clustering. Only the fields /map/legend-order takes.
      lastMapRequest = {
        nClusters: params.nClusters,
        kind: params.kind,
        outlierThreshold: params.outlierThreshold,
        tagged: params.tagged,
        onlyFileIds,
      };
      // A new map: new categories, new node names, new everything the two
      // colour caches are keyed by. Every control that changes the set of
      // categories -- Clustering, Max clusters, Show, Layout, Colour by, the
      // tag filter, a drill -- rebuilds through here, which is why this one
      // call covers them all.
      forgetPaintedCloudColors();
      forgetPaintedModelColors();
      // The clicked part belongs to the map that is being replaced; it may not
      // even be in this population, and a stale id would mark an unrelated row.
      revealedFileId = null;
      // The search result goes with it, and so does its marker: the viewer
      // below is rebuilt, so there is nothing left to clear it from. The
      // cursor goes too -- the same query over a new population is a new walk.
      foundFileId = null;
      findPartQuery = null;
      findPartCursor = 0;
      findPartMatches = [];
      // Rebuilt from the new points, once, for the rows below that show a
      // centre part's name.
      filenameByFileId = new Map();
      for (const p of (mapData && mapData.points) || []) {
        if (p && p.file_id) filenameByFileId.set(p.file_id, p.filename || null);
      }
      // Leaving the map does not abort this fetch -- nothing can -- so it runs
      // to completion and is dropped here instead. Everything below draws into
      // the viewer or the legend, both of which now show something else: the
      // part the user opened while this was in flight. Not an error, so it
      // returns rather than throwing.
      if (!active) return;
      // Measured here, once per build, because everything below that has to
      // match this layout's scale needs it and the points have just changed.
      layoutSpacing = measureLayoutSpacing(mapData.points);
      modelTargetDiag = targetDiagFor(layoutSpacing);
      // Remember the id the server minted for this subset: the SCS below and
      // every later loadModels() must ask for the same points.
      level.subsetId = mapData.subset_id || null;
      level.count = mapData.count;
      // The height Ward was cut at, when it was cut by height at all. Reported
      // through onClusters so a host can say what dissimilarity the map on
      // screen stands for.
      wardBand = mapData.ward || null;
      // Pin the population this map covers, so every later rebuild of it --
      // a View toggle, a cluster-count change, coming back from a part -- asks
      // for exactly these parts rather than for whatever is untagged by then.
      if (refreshesPopulation && untaggedNow && !level.fileIds) {
        untaggedPin = (mapData.points || []).map((p) => p.file_id).filter(Boolean);
      }
      // Report the cluster count the server actually used, and the Ward cut it
      // came from. It only means anything in cluster colouring, and neither is
      // required: the count is whatever the selector asked for, and Ward is now
      // always cut at its deepest. Kept because they describe what was really
      // built, which is the one thing a host cannot work out for itself.
      if (onClusters) {
        onClusters({
          k: mapData.clusters,
          colorBy,
          ward: wardBand,
        });
      }

      // The legend first, because the tree reads the category state it owns
      // (groupChecked), and because both draws below have to describe this
      // map rather than the one before it. The branches below render it again
      // once they know which view is being built; it is cheap and idempotent.
      renderLegend(mapData.groups);

      // The hierarchy, before anything is drawn. Only Ward has one -- there
      // the tree IS the clustering -- and it is auxiliary: it is read
      // alongside the map, never used to draw it. A failure therefore leaves
      // the map alone and is not shown as an error; the tree reports the
      // reason in its own panel, which is the difference between "no tree"
      // and silence.
      //
      // Fetched here rather than after the viewer is up because the tree
      // decides what colour each family wears, and a model placed before that
      // is known arrives in the cluster colour baked into its SCS and has to
      // be repainted on screen. The draw below assigns the colours; the models
      // loaded after it are placed in them (see modelRgbFor).
      treeError = null;
      treeData = null;
      if (clustering === 'ward') {
        try {
          const tree = await api.shapeMapTree(name, {
            kind,
            tagged: params.tagged,
            onlyFileIds,
            // The same ceiling /map was given as ward_max above: the leaves of
            // this tree have to BE the map's clusters.
            kMax: wardMax,
          });
          // A newer load owns the panel now. Returning rather than drawing:
          // this answer is stale, and showing it would replace the tree that
          // belongs to what is actually on screen.
          if (!active) return;
          treeData = tree && Array.isArray(tree.nodes) && tree.nodes.length
            ? tree
            : null;
        } catch (err) {
          if (!active) return;
          treeData = null;
          treeError = (err && err.message) ? err.message : String(err);
        }
      }
      // Draws the panel and assigns the colours. There is no viewer yet, so
      // the painting it attempts is a no-op; the draw after the map is up
      // repaints in full, because the caches were dropped with the viewer.
      renderTree();

      if (displayMode === 'models') {
        await loadModels(name, params, initialCamera);
      } else {
        renderLegend(mapData.groups);
        const scsUrl = api.mapScsUrl(name, { ...params, colorBy, subset: level.subsetId });
        mapViewer = await showMap(scsUrl, {
          onPointClick,
          onPointActivate,
          initialCamera,
          legendEl,
        });
        // Fresh viewer, fresh nodes: whatever colours were sent went to nodes
        // that no longer exist, and the models went with them.
        forgetPaintedCloudColors();
        forgetPaintedModelColors();
        applyGroupVisibility();
      }
      // After the map is really on screen, so a failed rebuild never reports a
      // threshold for a map nobody is looking at.
      if (typeof onOutlierThreshold === 'function') {
        onOutlierThreshold(mapData.outlier_threshold_effective);
      }
      // This map was built from the index as it stands now, so nothing is
      // outstanding. Done here rather than in the callers so EVERY rebuild
      // clears it: load() is the single path to a new map (see requestLoad()
      // below, which is what all ten entry points go through). A rebuild of a
      // PINNED population is not such a map -- it covers the same parts as
      // before, tags and all -- so the count has to survive it.
      if (refreshesPopulation) setUntaggedStale(0);

      // Drawn again now that the viewer exists: this is the draw that paints
      // the point cloud, and the only one that can.
      renderTree();
    } catch (err) {
      showError(err);
      throw err;
    }
  }

  // --- load() serialisation ------------------------------------------------
  //
  // load() is reached from ten places (setKind / setColorBy / setDisplayMode /
  // setNClusters / setOutlierThreshold / drillInto / drillTo / enter / reload
  // / setTagFilter) and had no concurrency guard. Two overlapping rebuilds each
  // build a viewer, and showMap() disposes the previous one while it is still
  // attaching its .scs, which the engine reports as
  // "Internal logic error" from _patchImplicitNodesByAttachment (and, from the
  // orphaned ResizeObserver, a TypeError on allowHighDpi). The server log shows
  // the overlap plainly: the same /map/scs URL fetched twice, from two
  // different client ports. This has always been possible; a <select> made it
  // unlikely because a click naturally waits for the rebuild it triggered,
  // while a slider fires "change" again a few hundred ms later.
  //
  // requestLoad() replaces every one of those call sites and guarantees:
  //
  // 1. At most one load() runs at a time -- a request arriving mid-build never
  //    starts a second one.
  // 2. At most ONE request waits. Later requests join the SAME pending run
  //    rather than queueing behind one another, so N rapid slider moves cost
  //    two rebuilds (the one in flight plus one final one), never N. Coalescing
  //    is correct here because a rebuild reads state, not a delta: the final
  //    run produces exactly what the last input asked for, and every
  //    intermediate result would have been overwritten anyway.
  // 3. The pending run calls load() when it STARTS, so it reads nClusters,
  //    outlierThreshold, drillStack, kind, colorBy and displayMode as they are
  //    then -- always the latest input, never a snapshot taken at request time.
  // 4. Each caller receives the promise of the run that will reflect ITS
  //    request: the in-flight one when idle, the pending one when busy. That
  //    promise settles with that run's outcome, which is what keeps
  //    drillInto()/drillTo() correct -- they roll drillStack and nClusters back
  //    in their catch, so handing them an earlier run's promise would either
  //    skip the rollback or fire it against the wrong state.
  // 5. A rejected run settles only its own waiters; the drain then starts the
  //    pending run regardless, so one failure never wedges the map.
  let loadInFlight = null;
  let loadPending = null;

  function requestLoad() {
    if (!loadInFlight) {
      const run = load();
      loadInFlight = run;
      setBusy(true);
      // Both outcomes drain, so a rejection never stops the chain; this also
      // handles `run` so a caller that ignores the promise cannot produce an
      // unhandled rejection.
      run.then(drainLoad, drainLoad);
      return run;
    }
    if (!loadPending) {
      let resolve;
      let reject;
      const promise = new Promise((res, rej) => {
        resolve = res;
        reject = rej;
      });
      // Swallow on this copy only; callers get `promise` and may still observe
      // the rejection through it.
      promise.catch(() => {});
      loadPending = { promise, resolve, reject };
    }
    // A run is in flight but busy may have been cleared by exit(): this request
    // will be served by a run that starts later, and that run still has to lock
    // the controls and raise the progress badge. Guarded because the host
    // re-arms its progress delay on every busy=true.
    if (!mapBusy) setBusy(true);
    return loadPending.promise;
  }

  function drainLoad() {
    // The map was left while this run was going. Do not start another one:
    // there is nothing on screen for it to build. loadInFlight is cleared so
    // the next enter() is seen as idle and starts a run of its own rather than
    // parking behind a promise that has already settled.
    if (!active) {
      const pending = loadPending;
      loadPending = null;
      if (pending) pending.resolve();
      loadInFlight = null;
      setBusy(false);
      return;
    }
    const pending = loadPending;
    loadPending = null;
    if (!pending) {
      loadInFlight = null;
      setBusy(false);
      return;
    }
    // Same reason as in requestLoad(): this follow-up run may be starting after
    // exit() cleared busy, so report it as a build in progress.
    if (!mapBusy) setBusy(true);
    const run = load();
    loadInFlight = run;
    run.then(pending.resolve, pending.reject);
    run.then(drainLoad, drainLoad);
  }

  // Busy is reported the moment a rebuild is requested, not after
  // PROGRESS_DELAY_MS: the host disables the map controls from it, and a
  // control that stays live for half a second is exactly how two rebuilds got
  // started. Delaying the visible spinner is the host's business (see main.js),
  // not a reason to delay the lock.
  function setBusy(busy) {
    mapBusy = Boolean(busy);
    // Going busy means a rebuild is coming, so drop any preview without trying
    // to restore it against a viewer that is about to be replaced.
    if (mapBusy) endPeek({ restore: false });
    // The legend is rebuilt asynchronously, so lock whatever is on screen now;
    // renderLegend()/renderCrumbs() apply the same rule to what they create.
    applyBusyLocks();
    if (onBusy) onBusy(mapBusy);
  }

  // Build and show the 3D-model map. The point cloud is loaded first (grid,
  // colours and camera exactly as in cloud mode), then a representative subset
  // of parts (capped at MAX_MODELS) is overlaid as real 3D geometry AT their
  // point positions -- the models sit on the cloud without moving the camera.
  // The legend is the ordinary cloud legend; its checkboxes show/hide both the
  // underlying point-cloud category AND that category's overlaid 3D models
  // together.
  //
  // *params* are load()'s, passed in rather than rebuilt here: the point cloud
  // underneath must be the one /map described, and a locally reconstructed set
  // of parameters drifts from it the moment one is added (the tag filter did
  // exactly that).
  async function loadModels(name, params, initialCamera = null) {
    // Same reason as the guard in load(): this must not draw into a viewer
    // that no longer belongs to the map.
    if (!active) return;
    renderLegend(mapData.groups);
    // The model map draws the SAME point cloud underneath, so it needs the same
    // subset. Forgetting it here would overlay the drilled models on the parent
    // map's points.
    const scsUrl = api.mapScsUrl(name, { ...params, colorBy, subset: drillTop().subsetId });
    // skipPointPicking: the model overlay wires its own Face picking, so leave
    // the cloud's Point picking off to avoid two competing handlers.
    // initialCamera keeps the viewpoint when arriving here from a View toggle.
    mapViewer = await showMap(scsUrl, {
      onPointClick,
      onPointActivate,
      skipPointPicking: true,
      overlayModels: true,
      initialCamera,
      legendEl,
    });
    // Same as the cloud path: the viewer's nodes have just been rebuilt, so
    // nothing this remembers applies to them.
    forgetPaintedCloudColors();
    forgetPaintedModelColors();
    applyGroupVisibility();

    // The cloud is up; the models take seconds more to arrive, which is ample
    // time to leave.
    if (!active) return;
    const parts = selectVisibleModelParts(name);
    modelFileIds = new Set(parts.map((part) => part.fileId));
    await addMapModels(parts, {
      onModelClick,
      onModelActivate,
      onPointClick: onModelPointClick,
      onPointActivate: onModelPointActivate,
      partScale,
      targetDiag: modelTargetDiag || undefined,
      trueScale,
    });
    // Baked-in colours again: forget what was sent before these existed.
    forgetPaintedModelColors();
    // Re-apply the legend state now that the models exist. Only visible
    // categories were loaded, so this is a no-op for them; it stays because the
    // user can toggle a category off while the models are on screen.
    if (!active) return;
    applyGroupVisibility();
    // The models arrived in their cluster colours. On the search-results map
    // they have to carry the similarity to the query instead.
    if (active && searchMode) applySearchColors();
    // Otherwise they arrive in their baked-in cluster colours, which would mix
    // with the tree's colouring of the models already on screen.
    else if (active) repaintTreeModelColors();
  }

  // The cap on the TOTAL number of models in the viewer at once -- the ordinary
  // selection plus any hover preview over it. Assemblies are much heavier per
  // node than parts, so the assembly-kind map caps lower (see
  // MAX_MODELS_ASSEMBLY).
  function modelBudget() {
    return kind === 'assembly' ? MAX_MODELS_ASSEMBLY : MAX_MODELS;
  }

  // The part of that cap kept free for the legend's hover preview.
  function peekReserve() {
    return kind === 'assembly' ? MODEL_PEEK_RESERVE_ASSEMBLY : MODEL_PEEK_RESERVE;
  }

  // What the ordinary selection may load: the on-screen cap minus the slice
  // held back for the legend's hover preview.
  function baseModelBudget() {
    return Math.max(1, modelBudget() - peekReserve());
  }

  // Median distance from a point to its nearest other point, over a sample.
  //
  // Sampled with a fixed stride rather than at random, so the same map always
  // measures the same -- a size that flickered between rebuilds would be worse
  // than one that is slightly off.
  function measureLayoutSpacing(points) {
    const n = points ? points.length : 0;
    if (n < 2) return 0;
    const step = Math.max(1, Math.floor(n / SPACING_SAMPLE));
    const found = [];
    for (let i = 0; i < n; i += step) {
      const a = points[i];
      if (!a) continue;
      let best = Infinity;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const b = points[j];
        if (!b) continue;
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dz = a.z - b.z;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best) best = d2;
      }
      if (Number.isFinite(best)) found.push(Math.sqrt(best));
    }
    if (!found.length) return 0;
    found.sort((p, q) => p - q);
    return found[Math.floor(found.length / 2)];
  }

  // The model size this layout calls for, clamped to the bounds above. Takes
  // the spacing already measured for this build rather than measuring again,
  // so the model size and the camera framing are derived from one number.
  function targetDiagFor(spacing) {
    if (!(spacing > 0)) return null; // let the viewer use its own fallback
    return Math.min(
      MODEL_TARGET_DIAG_MAX,
      Math.max(MODEL_TARGET_DIAG_MIN, MODEL_SPACING_RATIO * spacing),
    );
  }

  // cluster id -> file id of its centre part, from the payload the currently
  // displayed map was built from. Only cluster categories carry one.
  function centreByClusterMap() {
    const centreByCluster = new Map();
    for (const group of mapData.groups || []) {
      const key = String(group.key || '');
      if (!key.startsWith('cluster:') || !group.centre_file_id) continue;
      const c = Number(key.slice('cluster:'.length));
      if (Number.isFinite(c)) centreByCluster.set(c, group.centre_file_id);
    }
    return centreByCluster;
  }

  // The colour a model should be placed in: the tree's colour for its category
  // when the tree is colouring, else the cluster colour the server assigned
  // (which is what the point underneath wears). One source for every path that
  // builds a model descriptor, so no two can disagree.
  function modelRgbFor(group) {
    if (!group) return null;
    return treeColorByCategory.get(group.node_name) || group.rgb || null;
  }

  // Turn picked entries into the descriptors addMapModels() takes. Shared by
  // the ordinary selection and by the hover load, so the two cannot describe
  // the same part differently.
  function modelPartsFromEntries(name, entries) {
    return entries.map(({ p, i }) => {
      const group = groupOfIndex(i);
      return {
        fileId: p.file_id,
        filename: p.filename,
        scsUrl: api.scsUrl(name, p.file_id),
        x: p.x,
        y: p.y,
        z: p.z,
        rgb: modelRgbFor(group),
        groupNode: group ? group.node_name : null,
      };
    });
  }

  // Pick the parts that should be shown as 3D models right now: representatives
  // drawn from the categories the legend currently has switched ON, up to the
  // per-kind budget. In Tag colouring the Untagged group is off by default (the
  // grey majority), so filtering the candidates here means the budget goes to
  // what the user actually sees and no hidden part is ever loaded. It also
  // means the budget is re-spent on whatever is left when a category is
  // switched off: showing one cluster of eight fills the cap from that cluster
  // instead of keeping the eighth of the original sample that happened to
  // belong to it. Original point indices are kept so groupOfIndex() still
  // resolves each point's colour and category.
  function selectVisibleModelParts(name) {
    if (searchMode && searchFileIds && searchFileIds.size) {
      const candidates = (mapData.points || [])
        .map((p, i) => ({ p, i }))
        .filter(({ p }) => searchFileIds.has(p.file_id));
      // The hit set IS the selection. It is at most SEARCH_TOP_K (301), well
      // inside baseModelBudget(), so representative sampling never runs. The
      // legend checkboxes deliberately do not narrow it: they govern the point
      // cloud underneath, and switching every cluster off is how the hits are
      // seen on their own, still in the whole index's coordinates.
      return modelPartsFromEntries(name, candidates.slice(0, baseModelBudget()));
    }
    const candidates = (mapData.points || [])
      .map((p, i) => ({ p, i }))
      .filter(({ i }) => {
        const group = groupOfIndex(i);
        return !group || groupChecked.get(group.node_name) !== false;
      });
    if (focusNames.size) {
      // The focused categories take the budget first and IN FULL, because the
      // point of focusing one is to see every part in it, not a sample.
      //
      // Whatever budget is left over goes to the ordinary representative
      // spread over everything else, so the rest of the map does not go blank.
      // That is the difference from Isolate: nothing is hidden, the models are
      // merely aimed somewhere.
      //
      // groupChecked is read above exactly as it always was, so a category the
      // user switched off contributes nothing even if it is focused. Focusing
      // a hidden category is incoherent, which is why the menu item that pins
      // one also switches it on.
      const { head, rest, room, used, dropped } = splitFocusedCandidates(
        candidates,
        ({ i }) => (groupOfIndex(i) || {}).node_name || null,
        focusNames,
        baseModelBudget(),
      );
      focusUsed = used;
      focusDropped = dropped;
      markFocusedRows();
      const tail = room > 0
        ? selectRepresentatives(rest, room, centreByClusterMap())
        : [];
      return modelPartsFromEntries(name, [...head, ...tail]);
    }
    const picked = selectRepresentatives(candidates, baseModelBudget(), centreByClusterMap());
    return modelPartsFromEntries(name, picked);
  }

  // Bring the overlaid models in line with the legend after a checkbox change:
  // unload the models of the categories that are now off, and re-pick the
  // representatives among the categories that are still on so the budget is
  // spent on what is visible. Only the difference is touched -- a model that
  // survives the re-pick is never reloaded -- so toggling a category back and
  // forth costs nothing for the parts that stay.
  async function refreshModelSelection() {
    const name = getIndexName ? getIndexName() : null;
    if (!active || displayMode !== 'models' || !mapData || !name) return;
    const parts = selectVisibleModelParts(name);
    const wanted = new Set(parts.map((part) => part.fileId));
    const stale = [...modelFileIds].filter((fileId) => !wanted.has(fileId));
    const added = parts.filter((part) => !modelFileIds.has(part.fileId));
    if (stale.length === 0 && added.length === 0) return;
    modelFileIds = wanted;
    await removeMapModels(stale);
    if (added.length) {
      await addMapModels(added, {
        onModelClick,
        onModelActivate,
        onPointClick: onModelPointClick,
        onPointActivate: onModelPointActivate,
        targetDiag: modelTargetDiag || undefined,
        append: true,
      });
    }
    // Anything just added wears its baked-in cluster colour, and anything
    // removed takes its entry with it.
    forgetPaintedModelColors();
    applyGroupVisibility();
    // Models that have just been added arrive in their cluster colours, so
    // without this the search colouring and the cluster colouring would be
    // mixed on screen.
    if (active && searchMode) applySearchColors();
    // Otherwise they arrive in their baked-in cluster colours, which would mix
    // with the tree's colouring of the models already on screen.
    else if (active) repaintTreeModelColors();
  }

  // Serialize the refreshes above. A user can tick several categories faster
  // than the models load, and two overlapping runs would both diff against a
  // half-applied state and fight over the same nodes. Each request joins one
  // chain, and a request that is no longer the latest when its turn comes is
  // skipped: intermediate states are coalesced away, and the run that does
  // happen always sees the final checkbox state.
  let modelRefreshSeq = 0;
  let modelRefreshChain = Promise.resolve();
  // The one way onto that chain. The hover load and its matching unload join it
  // too, so a hover can never run addMapModels/removeMapModels concurrently
  // with a refresh: both sides mutate the same placed/nodeToFile maps in the
  // viewer, and a half-applied state is what two racing runs would each diff
  // against.
  function enqueueModelTask(task) {
    modelRefreshChain = modelRefreshChain
      .catch(() => {})
      .then(task)
      .catch((err) => {
        showError(err);
      });
    return modelRefreshChain;
  }

  function scheduleModelRefresh() {
    const seq = ++modelRefreshSeq;
    return enqueueModelTask(() =>
      seq === modelRefreshSeq ? refreshModelSelection() : undefined,
    );
  }

  // Choose which parts to show as 3D models. Takes candidate entries
  // ({ p, i } where i is the point's index in mapData.points), already filtered
  // to the visible categories by loadModels. If there are at most MAX_MODELS of
  // them, show them all. Otherwise pick representatives cluster by cluster and
  // round-robin across clusters, so every cluster is represented.
  //
  // Within a cluster there are two orders, and the Outliers setting picks
  // between them.
  //
  // * Outliers off (outlierThreshold === 0) -- nearest the centroid first, the
  //   behaviour this has always had. Farthest-point sampling cannot be used
  //   here: it goes for whatever is farthest from everything already taken, and
  //   with no outlier pass in effect the population is full of strays (measured
  //   at threshold 0.90, 6,250 of 7,928 parts fall outside their cluster), so
  //   it would draw the strays first and nothing about the cluster's character
  //   would be visible. Spread sampling assumes the strays have been removed.
  // * Outliers on -- farthest-point sampling: take the centre first, then
  //   repeatedly whichever remaining member is farthest from everything already
  //   taken. Nearest-first drew only the most central members, so the rim of a
  //   cluster -- above all where it meets its neighbour -- was never drawn, and
  //   that is exactly what you need to see to judge whether two adjacent
  //   clusters are really different things.
  //
  // Both orders start with the cluster's CENTRE PART when *centreByCluster*
  // names one: the real part closest to the cluster's k-means centroid, chosen
  // by the server in the full 2048-dimensional space. In nearest-first mode
  // that is not always the same part as the first after sorting, because the
  // sort is by distance to the centroid of the 3-D projection. They are aligned
  // in both modes so the hover highlight always has a model to point at,
  // whatever the Outliers setting is.
  //
  // The full FPS order is never computed. It costs O(n^2) per cluster, and
  // drilling in with Clusters=1 makes one cluster of every point in the level
  // -- 7,928 of them is over 60 million distance evaluations for an order of
  // which only the first few hundred entries are ever read. Each cluster
  // instead gets a picker that produces one more member on demand, which keeps
  // the whole thing at O(picked x cluster size). The two modes share the
  // round-robin below by presenting the same next() interface.
  //
  // Ties are broken deterministically, towards the earliest candidate, because
  // refreshModelSelection() re-runs this on every checkbox change and loads
  // only the difference from the previous selection. If the same candidate set
  // could yield a different answer, re-ticking a box would churn hundreds of
  // models for nothing.
  //
  // All distances are in the PROJECTED map space (x, y, z) -- the client has no
  // raw embeddings, so these are not the 2048-dimensional distances k-means
  // clustered on. Squared distances are used throughout; the comparisons are
  // monotonic, so the square root would change nothing.
  //
  // Returns [{ p, i }] with the original indices preserved.
  function selectRepresentatives(entries, cap, centreByCluster = null) {
    const indexed = entries;
    if (indexed.length <= cap) return indexed;

    // Read straight off the module state: it holds the threshold the map now on
    // screen was requested with, because load() sets it before the request and
    // the controls are locked for the whole rebuild.
    const spread = effectiveOutlierThreshold() > 0;
    const makePicker = spread ? makeSpreadPicker : makeCentrePicker;

    const byCluster = new Map();
    for (const entry of indexed) {
      const c = entry.p.cluster;
      const arr = byCluster.get(c);
      if (arr) arr.push(entry);
      else byCluster.set(c, [entry]);
    }
    const clusters = [];
    for (const [c, members] of byCluster) {
      let sx = 0;
      let sy = 0;
      let sz = 0;
      for (const { p } of members) {
        sx += p.x;
        sy += p.y;
        sz += p.z;
      }
      clusters.push(
        makePicker(
          members,
          sx / members.length,
          sy / members.length,
          sz / members.length,
          (centreByCluster && centreByCluster.get(c)) || null,
        ),
      );
    }
    // Larger clusters first, so when the cap is not a multiple of the cluster
    // count the busier clusters get the extra slots.
    clusters.sort((a, b) => b.size - a.size);

    const selected = [];
    while (selected.length < cap) {
      let addedAny = false;
      for (const picker of clusters) {
        const entry = picker.next();
        if (entry === null) continue;
        selected.push(entry);
        addedAny = true;
        if (selected.length >= cap) break;
      }
      if (!addedAny) break;
    }
    return selected;
  }

  // One cluster's farthest-point sampler. `pool` holds the members not yet
  // taken and `minDist[j]` is the squared distance from pool[j] to the nearest
  // member already taken (Infinity until the first one is). The two arrays are
  // created with the same length and are only ever changed together -- both are
  // written at the chosen slot and both are shortened by one -- so index j
  // always refers to the same member in both.
  function makeSpreadPicker(members, cx, cy, cz, centreFileId = null) {
    const pool = members.slice();
    const minDist = new Array(pool.length).fill(Infinity);
    let first = true;
    return {
      size: pool.length,
      next() {
        if (pool.length === 0) return null;
        let best = 0;
        if (first) {
          // Start at the cluster's centre part when the server named one, and
          // otherwise at the member nearest the projected centroid, so a
          // cluster's centre is always drawn.
          best = indexOfCentre(pool, centreFileId);
          if (best < 0) {
            let bestScore = Infinity;
            best = 0;
            for (let j = 0; j < pool.length; j++) {
              const { p } = pool[j];
              const d = (p.x - cx) ** 2 + (p.y - cy) ** 2 + (p.z - cz) ** 2;
              // Strictly less: on a tie the earlier candidate stays, which is
              // what makes repeated runs agree.
              if (d < bestScore) {
                bestScore = d;
                best = j;
              }
            }
          }
          first = false;
        } else {
          // The member whose nearest already-taken neighbour is farthest away.
          let bestScore = -Infinity;
          for (let j = 0; j < pool.length; j++) {
            if (minDist[j] > bestScore) {
              bestScore = minDist[j];
              best = j;
            }
          }
        }
        const chosen = pool[best];
        // Remove by swapping in the last element rather than splice(), which
        // would be O(n) on every pick.
        const last = pool.length - 1;
        pool[best] = pool[last];
        minDist[best] = minDist[last];
        pool.length = last;
        minDist.length = last;
        // Fold the new member into every remaining member's nearest-taken
        // distance, so the next call can read minDist directly.
        const { p: cp } = chosen;
        for (let j = 0; j < pool.length; j++) {
          const { p } = pool[j];
          const d = (p.x - cp.x) ** 2 + (p.y - cp.y) ** 2 + (p.z - cp.z) ** 2;
          if (d < minDist[j]) minDist[j] = d;
        }
        return chosen;
      },
    };
  }

  // Position of the cluster's centre part within *list*, or -1 when it is not
  // there (it can be missing: the candidates are filtered to the categories the
  // legend has switched on). First match wins, so the answer is deterministic.
  function indexOfCentre(list, centreFileId) {
    if (!centreFileId) return -1;
    for (let j = 0; j < list.length; j++) {
      if (list[j].p.file_id === centreFileId) return j;
    }
    return -1;
  }

  // One cluster's nearest-the-centroid sampler, used when the outlier pass is
  // off. The order is fixed up front by a single sort on squared distance to
  // the projected centroid, and next() just walks it; the cluster's centre part
  // is moved to the front so both modes start there.
  function makeCentrePicker(members, cx, cy, cz, centreFileId = null) {
    const order = members.slice();
    order.sort((a, b) => {
      const da = (a.p.x - cx) ** 2 + (a.p.y - cy) ** 2 + (a.p.z - cz) ** 2;
      const db = (b.p.x - cx) ** 2 + (b.p.y - cy) ** 2 + (b.p.z - cz) ** 2;
      // Equal distances fall back to the original index, so the order does not
      // depend on the sort's stability.
      return da === db ? a.i - b.i : da - db;
    });
    const centreAt = indexOfCentre(order, centreFileId);
    // The head of the sort is the member nearest the centroid of the 3-D
    // projection; the centre part is nearest the centroid of the space k-means
    // actually used, so they are not always the same member.
    if (centreAt > 0) order.unshift(order.splice(centreAt, 1)[0]);
    let at = 0;
    return {
      size: order.length,
      next() {
        return at < order.length ? order[at++] : null;
      },
    };
  }

  // Find the legend group a point (by its index in mapData.points) belongs to.
  // The points array is ordered by group, so each group owns the contiguous run
  // [offset, offset + count); this is the same mapping the server bakes into the
  // point cloud, so a model's colour matches the cloud exactly.
  function groupOfIndex(index) {
    for (const group of mapData.groups || []) {
      if (index >= group.offset && index < group.offset + group.count) return group;
    }
    return null;
  }

  // A part was picked in the viewer: remember it and point at the row that
  // covers it. Single clicks only -- a double click opens the part and leaves
  // the map, so there is no panel left to point at.
  function revealPart(fileId) {
    revealedFileId = fileId || null;
    highlightRevealed({ scroll: true });
  }

  // Put the found part's marker back on screen. A loaded model is highlighted
  // in place; otherwise -- cloud mode, or a part with no model on the overlay
  // -- the point itself is ringed, exactly as the hover centre is marked.
  function markFoundPart() {
    if (!foundFileId || !mapData || !mapViewer) return;
    const point = (mapData.points || []).find((p) => p && p.file_id === foundFileId);
    if (!point) return;
    let marked = false;
    if (displayMode === 'models') marked = setMapModelHighlighted(foundFileId, true);
    if (!marked) mapViewer.showCentreMarker(point.x, point.y, point.z);
  }

  // Drop the previous result's marker before a new search, so two searches in a
  // row never leave two parts marked.
  function clearFoundPart() {
    if (!foundFileId) return;
    // Both are harmless when the thing they address is not there, so there is
    // no need to remember which of the two was used.
    setMapModelHighlighted(foundFileId, false);
    if (mapViewer) mapViewer.clearCentreMarker();
    foundFileId = null;
  }

  // Find one part of this map by name and go to it. The panel lists families,
  // not parts, so a reader who knows the part's name has no other way in.
  //
  // Exact file name first, then the file id, then an id prefix, then a name
  // substring: an exact name is never what the reader meant to use as a prefix
  // of another, and ids are exact by nature.
  //
  // Repeating the same query steps to the NEXT match rather than insisting on
  // the first. A file id is the SHA-256 of the file's contents while a file
  // name is only its base name, so two parts in different folders can share a
  // name and still be two different points on this map; the cursor is how both
  // can be reached. A different query starts over.
  //
  // The result is reported through onPartFound as well as returned, including
  // the misses and the empty query, so the caller can clear what it is showing.
  // Which family a point belongs to, named the way the panel names it. Used to
  // say where a part was found rather than only that it was.
  function partLabelOf(point) {
    const group = groupOfIndex((mapData && mapData.points ? mapData.points : []).indexOf(point));
    return group ? (group.label || group.node_name || null) : null;
  }

  // The matches as the caller can list them: enough to tell two parts that
  // share a file name apart, which the name alone cannot do.
  function findPartList(matches) {
    return matches.map((p) => ({
      fileId: p.file_id,
      filename: p.filename || null,
      label: partLabelOf(p),
    }));
  }

  // Go to one point: mark it in the viewer, frame it, point at its row and
  // preview it -- exactly what clicking it would have done. Shared by the
  // query walk and by picking a candidate out of the list, so both leave the
  // map in the same state.
  async function goToPoint(point) {
    foundFileId = point.file_id;
    // In model view the part may be one of the many with no model placed; load
    // it so the answer is the shape itself rather than a dot.
    if (displayMode === 'models') await ensureModelForPoint(point);
    markFoundPart();
    // The same camera routine the hover preview uses, told to frame this one
    // part. Its minimum half-size floor is what keeps a single point from
    // filling the screen.
    focusGroups(null, { onlyFileIds: new Set([point.file_id]) });
    revealPart(point.file_id);
    if (onPointInfo) onPointInfo(point);
  }

  async function findPart(query) {
    const text = String(query == null ? '' : query).trim();
    clearFoundPart();
    const miss = {
      found: false, matches: 0, index: -1, fileId: null, filename: null, label: null, list: [],
    };
    if (!text || !mapData || !mapViewer) {
      findPartQuery = null;
      findPartCursor = 0;
      findPartMatches = [];
      if (onPartFound) onPartFound(miss);
      return miss;
    }
    const points = (mapData.points || []).filter(Boolean);
    const needle = text.toLowerCase();
    const byName = points.filter(
      (p) => String(p.filename || '').toLowerCase() === needle,
    );
    const byId = byName.length ? [] : points.filter((p) => p.file_id === text);
    // A file id is a 64-character hex hash, and the panel only ever shows its
    // first 8 characters, so a prefix has to be accepted for the id on screen
    // to be usable at all. Tried before the name substring because a hex string
    // of 8 characters or more is practically never part of a file name: taking
    // it for one would hide the id the reader actually typed.
    const isIdPrefix = /^[0-9a-f]{8,}$/i.test(text);
    const byIdPrefix = (byName.length || byId.length || !isIdPrefix)
      ? []
      : points.filter((p) => String(p.file_id || '').toLowerCase().startsWith(needle));
    const bySubstring = (byName.length || byId.length || byIdPrefix.length)
      ? []
      : points.filter((p) => String(p.filename || '').toLowerCase().includes(needle));
    let matches = byName;
    if (!matches.length) matches = byId;
    if (!matches.length) matches = byIdPrefix;
    if (!matches.length) matches = bySubstring;
    if (!matches.length) {
      findPartQuery = null;
      findPartCursor = 0;
      findPartMatches = [];
      showError(new Error(
        `No part named "${text}" in this map. It may be excluded by Show `
        + '(Parts / Assemblies) or by a tag filter.',
      ));
      if (onPartFound) onPartFound(miss);
      return miss;
    }
    if (findPartQuery === needle) findPartCursor += 1;
    else findPartCursor = 0;
    findPartQuery = needle;
    findPartCursor %= matches.length;
    findPartMatches = matches;
    const point = matches[findPartCursor];
    await goToPoint(point);
    const result = {
      found: true,
      matches: matches.length,
      index: findPartCursor,
      fileId: point.file_id,
      filename: point.filename || null,
      label: partLabelOf(point),
      list: findPartList(matches),
    };
    if (onPartFound) onPartFound(result);
    return result;
  }

  // Go to one of the matches the last query found, by id. The same state
  // change Enter makes -- the cursor moves to that candidate -- so clicking a
  // candidate and then pressing Enter continues from there rather than from
  // wherever the walk had got to.
  async function selectPartMatch(fileId) {
    const index = findPartMatches.findIndex((p) => p && p.file_id === fileId);
    if (index < 0) return null;
    const point = findPartMatches[index];
    clearFoundPart();
    findPartCursor = index;
    await goToPoint(point);
    const result = {
      found: true,
      matches: findPartMatches.length,
      index,
      fileId: point.file_id,
      filename: point.filename || null,
      label: partLabelOf(point),
      list: findPartList(findPartMatches),
    };
    if (onPartFound) onPartFound(result);
    return result;
  }

  // Mark the row the revealed part belongs to, in whichever panel is drawn.
  // Called again at the end of every legend and tree draw, so expanding a
  // branch moves the mark down to the deeper row that now covers the part.
  // Scrolling happens on the click only: a draw that yanked the panel around
  // would fight the reader scrolling it.
  function highlightRevealed({ scroll = false } = {}) {
    for (const el of [legendBodyEl, treeBodyEl]) {
      if (!el) continue;
      for (const prev of el.querySelectorAll('.map-row-revealed')) {
        prev.classList.remove('map-row-revealed');
      }
    }
    if (!revealedFileId || !mapData) return;
    const points = mapData.points || [];
    const index = points.findIndex((p) => p && p.file_id === revealedFileId);
    if (index < 0) return;
    const group = groupOfIndex(index);
    if (!group) return;
    let row = legendRowOf.get(group.node_name) || null;
    if (!row) {
      // Under Ward the category may be inside a collapsed branch; climb to the
      // deepest ancestor that IS on screen rather than expanding the tree,
      // which would move rows the reader did not ask to move.
      let id = treeCutNodeOf.has(group.node_name)
        ? treeCutNodeOf.get(group.node_name)
        : null;
      while (id !== null && id !== undefined && !treeRowOf.has(id)) {
        id = treeParentOf.has(id) ? treeParentOf.get(id) : null;
      }
      if (id !== null && id !== undefined) row = treeRowOf.get(id) || null;
    }
    if (!row || !row.isConnected) return;
    row.classList.add('map-row-revealed');
    if (scroll && typeof row.scrollIntoView === 'function') {
      row.scrollIntoView({ block: 'nearest' });
    }
  }

  // A model was single-clicked: resolve its file id back to a /map point and
  // preview it, mirroring the point-cloud onPointClick path. A click on empty
  // space (fileId null) is reported by the viewer so the panel closes here,
  // exactly like the point cloud's empty-click deselect.
  function onModelClick(fileId) {
    if (!fileId) {
      if (onPointClear) onPointClear();
      return;
    }
    const point = (mapData && mapData.points || []).find((p) => p && p.file_id === fileId);
    revealPart(fileId);
    if (onPointInfo) onPointInfo(point || null);
  }

  // A model was double-clicked: open its part.
  function onModelActivate(fileId) {
    const point = (mapData && mapData.points || []).find((p) => p && p.file_id === fileId);
    openPoint(point || null);
  }

  // In 3D-model mode most parts have no model loaded (only the representatives
  // do), but every part still has a cloud point. A click on such a bare point
  // is resolved to its /map point, that part's model is loaded on demand and
  // added at the point (camera unchanged, see addMapModel), and the part is
  // previewed -- so clicking anywhere on the cloud brings up its real shape.
  async function onModelPointClick(nodeName, pointId, position) {
    if (!position) {
      if (onPointClear) onPointClear();
      return;
    }
    const point = resolvePoint(nodeName, pointId, position);
    if (!point) {
      if (onPointInfo) onPointInfo(null);
      return;
    }
    await ensureModelForPoint(point);
    revealPart(point.file_id);
    if (onPointInfo) onPointInfo(point);
  }

  // A bare point was double-clicked in 3D-model mode: open its part (loading its
  // model first is unnecessary for opening, but harmless -- opening replaces the
  // map view anyway).
  function onModelPointActivate(nodeName, pointId, position) {
    const point = position ? resolvePoint(nodeName, pointId, position) : null;
    openPoint(point);
  }

  // Load and add the model for a /map point if it is not already on the overlay.
  // Builds the same part descriptor loadModels uses (colour and category from
  // the point's group) so the new model inherits the legend category and the
  // current Part scale. No-op if the part already has a model.
  async function ensureModelForPoint(point) {
    if (!point || modelFileIds.has(point.file_id)) return;
    const index = (mapData.points || []).indexOf(point);
    const group = index >= 0 ? groupOfIndex(index) : null;
    const part = {
      fileId: point.file_id,
      filename: point.filename,
      scsUrl: api.scsUrl(getIndexName(), point.file_id),
      x: point.x,
      y: point.y,
      z: point.z,
      rgb: modelRgbFor(group),
      groupNode: group ? group.node_name : null,
    };
    const ok = await addMapModel(part);
    if (ok) modelFileIds.add(point.file_id);
  }

  // Distance (in the map's normalized [-1, 1] space) between a /map point and
  // the picked world position it is supposed to be.
  function distanceTo(point, position) {
    const dx = point.x - position.x;
    const dy = point.y - position.y;
    const dz = point.z - position.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  // Turn a picked (node name, world position) into a /map point.
  //
  // In principle "group.offset + getPointId()" would give the index
  // directly, but real testing ruled that out: zoomed in, camera held still,
  // clicking the same point repeatedly always reported pointId 0 with an
  // identical picked position -- this SDK does not report a per-point id for
  // point-cloud picks, at least in this version. PointEntity.getPosition()
  // is trustworthy (the same testing found it lands exactly, to float32
  // rounding, on a real /map point's coordinates), so resolution is by
  // nearest position within the picked node's own group range instead.
  //
  // Never guesses: if the node/group can't be resolved, or no point in that
  // range is within NEAREST_POINT_TOLERANCE of the picked position, this
  // returns null rather than showing a plausible-looking but wrong part.
  function resolvePoint(nodeName, pointId, position) {
    if (!mapData) return null;
    if (!nodeName) {
      console.error(
        `Shape Map: pick had no resolvable node name (pointId=${pointId}); cannot look up a group.`,
      );
      return null;
    }
    const group = (mapData.groups || []).find((g) => g.node_name === nodeName);
    if (!group) {
      console.error(
        `Shape Map: node "${nodeName}" does not match any /map group; cannot resolve the picked point.`,
      );
      return null;
    }

    const start = group.offset;
    const end = start + group.count;
    let bestIndex = -1;
    let bestDistance = Infinity;
    for (let i = start; i < end; i++) {
      const p = mapData.points[i];
      if (!p) continue;
      const distance = distanceTo(p, position);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = i;
      }
    }
    if (bestIndex === -1 || bestDistance > NEAREST_POINT_TOLERANCE) {
      console.error(
        `Shape Map: no point in group "${nodeName}" (offset ${group.offset}, count ${group.count}, ` +
          `pointId=${pointId}) is near the picked position (${position.x}, ${position.y}, ${position.z}); ` +
          `nearest is ${bestDistance.toFixed(4)} away. Refusing to guess.`,
      );
      return null;
    }
    return mapData.points[bestIndex];
  }

  // A click was resolved to a point (or wasn't): map it to a part and preview
  // it. A click that hit no point at all (position is null) is a normal
  // deselect -- clear the marker and info area without an error. A click that
  // hit a point which still can't be resolved is different: say so instead of
  // showing an arbitrary part.
  function onPointClick(nodeName, pointId, position) {
    if (!position) {
      if (mapViewer) mapViewer.clearPointMarker();
      if (onPointClear) onPointClear();
      return;
    }
    const point = resolvePoint(nodeName, pointId, position);
    if (!point) {
      if (mapViewer) mapViewer.clearPointMarker();
      if (onPointInfo) onPointInfo(null);
      return;
    }
    if (mapViewer) mapViewer.showPointMarker(point.x, point.y, point.z);
    revealPart(point.file_id);
    if (onPointInfo) onPointInfo(point);
  }

  // The picked point was double-clicked: open it, or report that it couldn't
  // be identified rather than silently doing nothing.
  function onPointActivate(nodeName, pointId, position) {
    const point = position ? resolvePoint(nodeName, pointId, position) : null;
    openPoint(point);
  }

  // Blue -> green -> red, the usual CAE contour ramp: 0 is the least similar
  // part of the cluster, 1 the most. Two linear halves meeting at pure green,
  // so the midpoint is unambiguous to read off the colour bar.
  function heatColor(t) {
    const v = Math.min(1, Math.max(0, Number(t) || 0));
    if (v < 0.5) {
      return [0, Math.round(510 * v), Math.round(255 - 510 * v)];
    }
    return [Math.round(510 * v - 255), Math.round(510 - 510 * v), 0];
  }

  // The colour bar overlaid on the viewer, built on first use and reused after
  // that. A DOM overlay rather than a Communicator overlay: it is ordinary
  // chrome around the canvas, and the viewer is rebuilt on every map load,
  // which would take an engine-side overlay with it.
  function heatBar() {
    if (heatBarEl) return heatBarEl;
    const pane = document.querySelector('.viewer-pane');
    if (!pane) return null;
    const bar = document.createElement('div');
    bar.className = 'map-heat-bar';
    bar.hidden = true;
    const max = document.createElement('div');
    max.className = 'map-heat-label';
    const gradient = document.createElement('div');
    gradient.className = 'map-heat-gradient';
    const min = document.createElement('div');
    min.className = 'map-heat-label';
    const caption = document.createElement('div');
    caption.className = 'map-heat-caption';
    bar.append(max, gradient, min, caption);
    pane.appendChild(bar);
    heatBarEl = bar;
    // Kept on the element so showHeatBar() does not have to re-query them.
    heatBarEl._max = max;
    heatBarEl._min = min;
    heatBarEl._caption = caption;
    return heatBarEl;
  }

  function showHeatBar(min, max, caption) {
    const bar = heatBar();
    if (!bar) return;
    bar._max.textContent = max.toFixed(3);
    bar._min.textContent = min.toFixed(3);
    bar._caption.textContent = caption;
    bar.hidden = false;
  }

  function hideHeatBar() {
    if (heatBarEl) heatBarEl.hidden = true;
  }

  // Colour one cluster's models by their true cosine similarity to the cluster
  // centre. The positions on the map are a 3D projection of a 2048-dimensional
  // space, so they can only approximate the similarity; painting the real
  // number on top shows where the two agree and where they do not.
  function applyHeat(group, button) {
    if (!mapData || !group) return;
    const start = group.offset || 0;
    const points = (mapData.points || []).slice(start, start + (group.count || 0));
    const scored = points.filter(
      (p) => p && typeof p.file_id === 'string' && Number.isFinite(p.sim),
    );
    if (scored.length === 0) {
      showError(
        'This server did not return per-part similarities; restart the server '
        + 'to pick up the new build.',
      );
      return;
    }
    let min = Infinity;
    let max = -Infinity;
    for (const p of scored) {
      if (p.sim < min) min = p.sim;
      if (p.sim > max) max = p.sim;
    }
    const span = max - min;
    // byBand[i] holds the ids whose similarity falls in the i-th step.
    const byBand = new Map();
    for (const p of scored) {
      // A cluster whose members are all equally close has no range to show, so
      // everything gets the middle colour rather than a division by zero.
      const t = span > 0 ? (p.sim - min) / span : 0.5;
      const band = Math.min(HEAT_BANDS - 1, Math.floor(t * HEAT_BANDS));
      if (!byBand.has(band)) byBand.set(band, []);
      byBand.get(band).push(p.file_id);
    }
    const bands = [];
    for (const [band, fileIds] of byBand) {
      // The centre of the band, so the extremes are not clipped to pure blue
      // and pure red by the quantisation alone.
      bands.push({ rgb: heatColor((band + 0.5) / HEAT_BANDS), fileIds });
    }
    const colored = setMapModelColors(bands);
    // Something other than the tree now owns the models' colours, so what the
    // tree remembers sending is no longer what is on screen.
    forgetPaintedModelColors();
    heatState = {
      fileIds: scored.map((p) => p.file_id),
      rgb: group.rgb || [128, 128, 128],
    };
    heatButton = button || null;
    if (heatButton) heatButton.textContent = 'Heat map \u2713';
    showHeatBar(
      min,
      max,
      `${group.label} \u2014 ${colored} of ${scored.length} parts shown`,
    );
  }

  // Put the cluster back to its own colour and drop the bar. Safe to call at
  // any time: without an active heat map it does nothing.
  function clearHeat() {
    if (!heatState) return;
    setMapModelColors([{ rgb: heatState.rgb, fileIds: heatState.fileIds }]);
    forgetPaintedModelColors();
    heatState = null;
    hideHeatBar();
    if (heatButton) heatButton.textContent = 'Heat map';
    heatButton = null;
    // The line above restores the server's own cluster colour, which under Ward
    // is not what the models were wearing: the tree had painted them by family.
    repaintTreeModelColors();
  }

  // Open the part behind a double-clicked point. Every double-click path goes
  // through here, so the map is left in one place rather than three.
  //
  // The host does not call exit() for this: it sets mapMode to 'part-from-map'
  // so the Map button comes back to the map with its settings intact. But the
  // viewer underneath is replaced all the same, and two things would survive
  // that and should not. The colour bar is a DOM overlay on the viewer's
  // container, so it would hang over the opened part. And heatState would
  // outlive the map's viewer, which showMap() builds from scratch on the way
  // back: the models would be wearing cluster colours while "Clear heat map"
  // still offered to take heat colours off them.
  //
  // Not clearHeat(): that repaints the models and the Ward tree's colours,
  // which is per-node work on a scene disposeViewer() is about to throw away.
  // Only the state and the overlay need to go.
  function openPoint(point) {
    heatState = null;
    if (heatButton) heatButton.textContent = 'Heat map';
    heatButton = null;
    searchColored = false;
    hideHeatBar();
    if (onPointOpen) onPointOpen(point);
  }

  // Colour the overlaid models by each hit's similarity to the query. Reuses
  // the existing heat machinery: heatColor() for the gradient,
  // setMapModelColors() to apply it in one call per band, and the same colour
  // bar the cluster heat map shows. Hits below the Min similarity slider are
  // greyed rather than removed, so the shape of the neighbourhood that the
  // slider is cutting through stays visible.
  function applySearchColors() {
    if (!searchMode || displayMode !== 'models' || !mapData) return;
    const ctx = getSearchContext ? getSearchContext() : null;
    if (!ctx) return;
    // The search gradient replaces whatever the tree painted, here and in
    // every branch below, so the tree's record of it goes now.
    forgetPaintedModelColors();
    const rawThreshold = Number(ctx.threshold);
    // No usable slider value means nothing can be excluded, so the gradient
    // covers every hit.
    const threshold = Number.isFinite(rawThreshold) ? rawThreshold : -Infinity;
    const above = [];
    const below = [];
    for (const hit of ctx.hits || []) {
      const score = Number(hit.score);
      // A hit with no usable score has no place on the gradient; it is still
      // on the map, so it joins the greyed ones.
      if (Number.isFinite(score) && score >= threshold) above.push({ id: String(hit.id), score });
      else below.push(String(hit.id));
    }
    if (above.length === 0) {
      // The slider has excluded the whole set: one flat band, and no bar,
      // because there is no range left to read off it.
      setMapModelColors([{ rgb: SEARCH_BELOW_RGB, fileIds: below }]);
      hideHeatBar();
      searchColored = true;
      // Same reason as at the end of this function: the counts moved with the
      // slider, and this path returns before reaching it.
      renderSearchSummary();
      return;
    }
    let min = Infinity;
    let max = -Infinity;
    for (const hit of above) {
      if (hit.score < min) min = hit.score;
      if (hit.score > max) max = hit.score;
    }
    const bands = [];
    if (max <= min) {
      // One hit, or a set with no spread: there is nothing to grade, so show
      // it at the top of the scale rather than dividing by zero.
      bands.push({ rgb: heatColor(1), fileIds: above.map((hit) => hit.id) });
    } else {
      // byBand[i] holds the ids whose score falls in the i-th step, exactly as
      // applyHeat() quantises a cluster.
      const byBand = new Map();
      for (const hit of above) {
        const t = (hit.score - min) / (max - min);
        const band = Math.min(HEAT_BANDS - 1, Math.floor(t * HEAT_BANDS));
        if (!byBand.has(band)) byBand.set(band, []);
        byBand.get(band).push(hit.id);
      }
      for (const [band, fileIds] of byBand) {
        // The centre of the band, so the extremes are not clipped to pure blue
        // and pure red by the quantisation alone.
        bands.push({ rgb: heatColor((band + 0.5) / HEAT_BANDS), fileIds });
      }
    }
    if (below.length) bands.push({ rgb: SEARCH_BELOW_RGB, fileIds: below });
    // Ids that are not on the map are skipped by setMapModelColors(), so the
    // hits that fell outside the model budget need no filtering here.
    setMapModelColors(bands);
    showHeatBar(min, max, 'Similarity to the query part');
    searchColored = true;
    // The slider that just changed the colours also changes the counts.
    renderSearchSummary();
  }

  // Put the overlaid models back to their cluster colours after the search
  // colouring. The colours come from the same groups the legend and the point
  // cloud use (groupOfIndex), so a restored model matches the point underneath
  // it exactly as a freshly loaded one does: modelPartsFromEntries() hands
  // addMapModels() that same group.rgb, and a point with no group is left
  // alone here because loadOneMapModel() does not colour it either.
  function restoreModelClusterColors() {
    if (!searchColored) return;
    const byRgb = new Map();
    const points = (mapData && mapData.points) || [];
    points.forEach((p, i) => {
      if (!p || !modelFileIds.has(p.file_id)) return;
      const group = groupOfIndex(i);
      if (!group || !group.rgb) return;
      const key = group.rgb.join(',');
      if (!byRgb.has(key)) byRgb.set(key, { rgb: group.rgb, fileIds: [] });
      byRgb.get(key).fileIds.push(p.file_id);
    });
    setMapModelColors([...byRgb.values()]);
    forgetPaintedModelColors();
    hideHeatBar();
    searchColored = false;
  }

  // The Min similarity slider fires on every tick of a drag, and each repaint
  // walks the whole hit set; only the last one matters.
  function scheduleSearchColors() {
    if (searchColorTimer) clearTimeout(searchColorTimer);
    searchColorTimer = setTimeout(() => {
      searchColorTimer = null;
      if (!active) return;
      applySearchColors();
    }, SEARCH_COLOR_DELAY_MS);
  }

  // Report the outstanding count to the host, which puts it on the Reload
  // button. Only ever called with a value that differs from the current one.
  function setUntaggedStale(count) {
    if (count === untaggedStale) return;
    untaggedStale = count;
    if (typeof onUntaggedStale === 'function') onUntaggedStale(untaggedStale);
  }

  // Note that *updated* parts were just tagged, so the map on screen is that
  // much out of date. Counting only, deliberately: re-clustering costs a full
  // re-projection, and doing it after every tag made it impossible to tag a
  // few clusters in a row and then look at the result. The user asks for the
  // rebuild with the Reload button (see reloadUntagged()).
  //
  // A no-op outside Untagged-only mode: there, tagging does not change which
  // parts the map covers at all.
  function noteTaggedWhileUntaggedOnly(updated) {
    if (!untaggedOnly) return;
    const n = Number(updated);
    if (!Number.isFinite(n) || n <= 0) return;
    setUntaggedStale(untaggedStale + n);
  }

  // The toolbar above the legend: show/hide everything at once, and tag every
  // checked cluster in one go. Raising the cluster count splits one family of
  // parts across several clusters (at k=24 the same nuts land in more than
  // one), so tagging them one row at a time means typing the same tag over and
  // over; and isolating a single cluster meant unticking every other box.
  //
  // Returns { refresh } so the row handlers can re-evaluate the Tag button
  // whenever the checked set changes.
  // The legend's summary lines. There is no toolbar any more: All / None /
  // Tag checked / Heat map have become per-row right-click items, where the
  // row they act on is the one under the pointer (see groupMenuItems).
  //
  // Ward has no summary line of its own: the tree's heading says the same
  // things about the same population, and two lines of totals in one panel
  // invite the reader to look for the difference between them. Tag colouring
  // draws no tree, but its summary was already empty (that line counts
  // clusters), so only the search line is left to render either way.
  function renderLegendTools(groups) {
    if (clustering === 'ward') {
      if (colorBy === 'cluster') renderSearchSummary();
      else renderLegendSummary(groups);
      return null;
    }
    renderLegendSummary(groups);
    renderLegendOrderControl(groups);
    return null;
  }

  // The Order select: Size (the order the server built the groups in, which is
  // member count descending) or Similarity (the leaf order of a Ward
  // dendrogram over the cluster centroids).
  //
  // Drawn for k-means cluster colouring only. Ward already draws its clusters
  // in the order of the tree they are a cut of, and a tag colouring's rows are
  // tags rather than clusters, so neither has a second order to offer.
  function renderLegendOrderControl(groups) {
    if (!legendBodyEl || clustering !== 'kmeans' || colorBy !== 'cluster') return;
    const clusters = (groups || []).filter(
      (group) => typeof group.key === 'string' && group.key.startsWith('cluster:'),
    );
    // One row cannot be in the wrong order.
    if (clusters.length < 2) return;

    const row = document.createElement('div');
    row.className = 'map-legend-order';
    const label = document.createElement('label');
    label.textContent = 'Order:';
    const select = document.createElement('select');
    for (const [value, text] of [['size', 'Size'], ['similarity', 'Similarity']]) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = text;
      select.appendChild(option);
    }
    select.value = legendOrderMode;
    select.title =
      'Size lists the biggest clusters first. Similarity puts clusters whose '
      + 'centres are alike next to each other, so one family split across '
      + 'several clusters is read in one place instead of hunted down the list.';
    label.appendChild(select);
    row.appendChild(label);
    legendBodyEl.appendChild(row);

    select.addEventListener('change', async () => {
      const next = select.value === 'similarity' ? 'similarity' : 'size';
      if (next === 'size') {
        legendOrderMode = 'size';
        renderLegend(mapData && mapData.groups);
        return;
      }
      const signature = legendOrderSignature;
      if (!legendOrders.has(signature)) {
        // The map it describes is on screen and cannot change under this
        // request -- every control that would change it rebuilds the legend --
        // but the select must not take a second answer while the first is in
        // flight.
        select.disabled = true;
        try {
          const name = getIndexName ? getIndexName() : '';
          const result = await api.shapeMapLegendOrder(name, lastMapRequest || {});
          legendOrders.set(signature, (result && result.order) || []);
        } catch (err) {
          // Back to Size, and say so. Silently leaving the rows as they were
          // would look like a setting that does nothing.
          select.disabled = false;
          select.value = 'size';
          legendOrderMode = 'size';
          showError(err);
          return;
        }
        select.disabled = false;
      }
      legendOrderMode = 'similarity';
      renderLegend(mapData && mapData.groups);
    });
  }

  // One line above the legend's rows saying how much of the map is in clusters
  // big enough to be worth a tag.
  //
  // Raising the cluster count always rescues parts from Outliers -- more
  // centroids means more parts close enough to one of them -- but a lot of that
  // gain lands in clusters of one or two parts, which inflate the cluster count
  // without being anything you would tag. The surviving-part total alone
  // therefore flatters a high count; this splits it.
  //
  // Deliberately not refreshed when a checkbox is toggled: it describes the map
  // the server built, not what is currently ticked. renderLegend() rebuilds it
  // on every load, which is exactly when the numbers can change.
  function renderLegendSummary(groups) {
    // A tag colouring has no clusters to count; nor has a map whose parts all
    // fell out of clustering. Both used to return early, but the search line
    // below has to be rendered either way, so the ordinary line is now a
    // conditional block rather than a set of early returns.
    const clusters =
      colorBy === 'cluster'
        ? (groups || []).filter(
            (group) => typeof group.key === 'string' && group.key.startsWith('cluster:'),
          )
        : [];
    if (clusters.length > 0) {
      const parts = clusters.reduce((sum, group) => sum + (group.count || 0), 0);
      const bigGroups = clusters.filter((group) => (group.count || 0) >= LEGEND_SUMMARY_MIN);
      const bigParts = bigGroups.reduce((sum, group) => sum + (group.count || 0), 0);
      const line = document.createElement('div');
      line.className = 'map-legend-summary';
      line.textContent =
        `${clusters.length.toLocaleString()} clusters \u00b7 ${parts.toLocaleString()} parts `
        + `\u00b7 ${bigParts.toLocaleString()} in ${bigGroups.length.toLocaleString()} clusters `
        + `of ${LEGEND_SUMMARY_MIN}+`;
      line.title =
        'Parts in clusters large enough to be worth one tag. Raising the cluster '
        + 'count always moves more parts out of Outliers, but much of that gain '
        + 'lands in clusters of one or two parts; this says how much of it is '
        + 'usable.';
      legendBodyEl.appendChild(line);
    }
    // Last, so the search line sits under the ordinary one.
    renderSearchSummary();
  }

  // The number the search map exists to produce: how far the similarity
  // search's neighbourhood spreads across the k-means clustering. One cluster
  // means the two groupings agree; several means the search crosses boundaries
  // k-means drew. Rendered as a second summary line under the ordinary one,
  // and refreshed on its own whenever the Min similarity slider moves -- the
  // legend itself does not change, only this count.
  function renderSearchSummary() {
    if (!legendBodyEl) return;
    let el = legendBodyEl.querySelector('#map-search-summary');
    const ctx = searchMode && getSearchContext ? getSearchContext() : null;
    if (!searchMode || !ctx || !mapData || colorBy !== 'cluster') {
      if (el) el.remove();
      return;
    }
    const threshold = Number(ctx.threshold);
    const hits = ctx.hits || [];
    // file id -> point index, so each hit can be put back on the map it is
    // being measured against.
    const indexOfFile = new Map();
    (mapData.points || []).forEach((p, i) => {
      if (p && !indexOfFile.has(p.file_id)) indexOfFile.set(p.file_id, i);
    });
    let offMap = 0;
    let above = 0;
    let outliers = 0;
    // group key -> { label, count }, so clusters are counted by identity
    // rather than by the order the legend happens to list them in.
    const byCluster = new Map();
    for (const hit of hits) {
      const index = indexOfFile.get(String(hit.id));
      if (index === undefined) {
        // The map is a subset (Untagged only, a drill, a kind filter): this
        // hit has no point to be counted against.
        offMap += 1;
        continue;
      }
      const score = Number(hit.score);
      if (!Number.isFinite(score) || !(score >= threshold)) continue;
      above += 1;
      const group = groupOfIndex(index);
      const key = group && typeof group.key === 'string' ? group.key : '';
      if (!key.startsWith('cluster:')) {
        // Outliers, Untagged and Other are display categories, not clusters.
        outliers += 1;
        continue;
      }
      if (!byCluster.has(key)) byCluster.set(key, { label: group.label || key, count: 0 });
      byCluster.get(key).count += 1;
    }
    // Biggest first, ties by cluster number, so the order does not shuffle on
    // every tick of the Min similarity slider.
    const ranked = [...byCluster.entries()].sort((a, b) => {
      if (b[1].count !== a[1].count) return b[1].count - a[1].count;
      return Number(a[0].slice('cluster:'.length)) - Number(b[0].slice('cluster:'.length));
    });
    let text = `${hits.length.toLocaleString()} hits`;
    text += ` \u00b7 ${above.toLocaleString()} at or above ${
      Number.isFinite(threshold) ? threshold.toFixed(4) : '\u2014'
    }`;
    text += ` \u00b7 ${ranked.length.toLocaleString()} clusters`;
    if (ranked.length) {
      const top = ranked
        .slice(0, 3)
        .map(([, entry]) => `${entry.label} (${entry.count.toLocaleString()})`)
        .join(', ');
      text += ` \u00b7 top ${top}`;
    }
    if (outliers) text += ` \u00b7 ${outliers.toLocaleString()} in Outliers`;
    if (offMap) text += ` \u00b7 ${offMap.toLocaleString()} not on this map`;
    if (!el) {
      el = document.createElement('div');
      el.className = 'map-legend-summary';
      el.id = 'map-search-summary';
    }
    // Always (re)place it directly after the ordinary summary line, never by
    // appending: renderLegendSummary() runs before the legend's scroll
    // container exists while a slider repaint runs after it, so appending puts
    // this line above the rows in one case and below them in the other, and
    // changing the cluster count makes it jump. Both lines describe the map as
    // a whole, so they belong together, outside the scrolling rows.
    // Without an ordinary line (no clusters to summarise) the position is back
    // to whatever the caller's timing gives, which still beats not showing it.
    const ordinary = [...legendBodyEl.querySelectorAll('.map-legend-summary')]
      .find((node) => node.id !== 'map-search-summary');
    if (ordinary) ordinary.insertAdjacentElement('afterend', el);
    else if (el.parentNode !== legendBodyEl) legendBodyEl.appendChild(el);
    el.textContent = text;
    el.title =
      'How far the similarity search reaches across the k-means clustering. '
      + 'One cluster means the two groupings agree on this family; several '
      + 'means the search crosses boundaries k-means drew. Counted over the '
      + 'hits at or above the Min similarity slider.';
  }

  // Render the server-provided groups as a checkbox legend. Each row carries the
  // RGB the server baked into that category's node, so the swatch is drawn
  // straight from the response -- the client keeps no palette of its own, which
  // is what let the legend drift from the map before. Toggling a checkbox only
  // changes the node's visibility in the viewer; it never re-fetches.
  //
  // The legend is rebuilt on every load(), including the View toggle (point
  // cloud <-> 3D models), which rebuilds the viewer but shows the SAME
  // categories. Dropping the checkboxes there threw away the user's filtering
  // for no reason, so the state is remembered per signature: same index, kind,
  // colour scheme, depth, subset and node names. Anything else -- switching
  // Cluster/Tag colouring, changing the cluster count, changing kind or index
  // -- is a different set of categories and starts from the defaults, but it
  // now gets its own slot instead of overwriting the one it displaced, so
  // coming back to it restores what was there.
  // One category's identity: the checkbox, the swatch, the name, the count and
  // the cohesion. Built here rather than inside renderLegend() because Ward's
  // tree shows the SAME categories on its cut rows -- the legend is one row of
  // that tree -- and two copies of this would drift apart.
  //
  // `stretchLabel` is for the legend only. .map-legend-label is flex: 1, which
  // in a tree row would push everything after it to the right edge.
  function appendGroupIdentity(row, group, { stretchLabel = true } = {}) {
    const checked = groupChecked.has(group.node_name)
      ? groupChecked.get(group.node_name)
      : true;
    const check = document.createElement('input');
    check.type = 'checkbox';
    check.className = 'map-legend-check';
    check.checked = checked;
    if (!groupCheckInputs.has(group.node_name)) {
      groupCheckInputs.set(group.node_name, []);
    }
    groupCheckInputs.get(group.node_name).push(check);
    check.addEventListener('change', () => {
      groupChecked.set(group.node_name, check.checked);
      if (legendRemembered) legendRemembered.set(group.node_name, check.checked);
      // The other panel may be showing this category too. Rows are discarded
      // and rebuilt often (the tree redraws on every expand), so the ones no
      // longer in the document are dropped here rather than tracked.
      const peers = (groupCheckInputs.get(group.node_name) || []).filter(
        (el) => el === check || el.isConnected,
      );
      groupCheckInputs.set(group.node_name, peers);
      for (const el of peers) el.checked = check.checked;
      // End any hover preview FIRST. Its unload joins the model chain ahead
      // of the refresh below, so refreshModelSelection() diffs against a
      // viewer that no longer holds the hover's extra models -- it knows
      // nothing about them and would otherwise leave them loaded for good.
      endPeek();
      // The checked set decides which cluster the heat map describes, so a
      // change to it makes the current colouring stale.
      clearHeat();
      if (mapViewer) mapViewer.setGroupVisible(group.node_name, check.checked);
      // In model mode the same checkbox also toggles the overlaid 3D models
      // of this category, so hiding a category hides both its points and its
      // models (a no-op in cloud mode). Hiding is immediate feedback; the
      // refresh then unloads what is off and re-picks representatives among
      // the categories that are still on, so the model budget follows the
      // legend instead of staying spent on the original sample.
      if (displayMode === 'models') {
        setMapModelGroupVisible(group.node_name, check.checked);
        scheduleModelRefresh();
      }
      if (legendTools) legendTools.refresh();
    });
    const swatch = document.createElement('span');
    swatch.className = 'map-legend-swatch';
    const [r, g, b] = group.rgb || [128, 128, 128];
    swatch.style.background = `rgb(${r}, ${g}, ${b})`;
    const label = document.createElement('span');
    if (stretchLabel) label.className = 'map-legend-label';
    label.textContent = group.label;
    const count = document.createElement('span');
    count.className = 'map-legend-count';
    count.textContent = String(group.count);
    // Cohesion says how mixed a cluster is without having to look at the 3D
    // view, which is what tells the user where drilling is worth the wait.
    // Only k-means clusters have one, so rows without it show nothing rather
    // than a placeholder.
    const cohesion =
      typeof group.cohesion === 'number' && Number.isFinite(group.cohesion)
        ? document.createElement('span')
        : null;
    if (cohesion) {
      cohesion.className = 'map-legend-cohesion';
      // Four decimals for the same reason as the tree rows: 0.95 is a
      // decision boundary, and two decimals cannot tell 0.9499 from 0.9500.
      cohesion.textContent = group.cohesion.toFixed(4);
      cohesion.title =
        "Mean cosine similarity of this cluster's parts to its centre; " +
        'lower means the cluster is mixed and worth drilling into';
    }
    row.append(check, swatch, label, count);
    if (cohesion) row.appendChild(cohesion);
    // The centre part: what this family looks like, in one name. The tree
    // rows show theirs, so the legend shows its own and the two panels can be
    // read against each other. Nothing is drawn when the map has no centre for
    // the group, or when the centre is not a point of this map -- a
    // placeholder would only be a name-shaped gap.
    const centreId = group.centre_file_id || null;
    const centreName = centreId ? filenameByFileId.get(centreId) : null;
    if (centreName) {
      const centre = document.createElement('span');
      centre.className = 'map-row-centre';
      centre.textContent = `\u00b7 ${centreName}`;
      centre.title = `Centre part: ${centreName} (id ${String(centreId).slice(0, 8)})`;
      row.appendChild(centre);
    }
  }

  // The actions a category offers, as right-click menu items. There are no
  // buttons and no toolbar: every row should accept the same actions, and a
  // row of buttons on each of dozens of rows is not a panel -- it filled the
  // space the centre part's name now uses. Ward's tree rows already worked
  // this way, so the two clusterings no longer have two different ways of
  // being operated. An empty list means the row has nothing to offer and no
  // menu is opened.
  function groupMenuItems(group) {
    // In cluster colouring, every cluster row offers a one-click bulk tag over
    // its WHOLE membership (the k-means group spans the full index, not just
    // what a search returned), and a drill-in that re-projects that group on
    // its own. Tag colouring offers neither.
    const key = typeof group.key === 'string' ? group.key : '';
    // Tagging stays clusters-only: the outlier group is a display category,
    // not a family of parts, so one tag over it would be meaningless.
    const taggable = colorBy === 'cluster' && key.startsWith('cluster:');
    // Drilling in is offered for the outlier group too. Its parts are the
    // ones that were far from EVERY centroid, which is not the same as being
    // unlike each other; re-clustering them on their own gives them their own
    // centroids, and families that the global run could not reach may
    // separate there.
    //
    // Ward offers none of it: its tree already holds every sub-grouping of
    // every node, so re-projecting one of them would answer a question the
    // panel beside it has already answered.
    const drillable =
      colorBy === 'cluster'
      && clustering !== 'ward'
      && (key.startsWith('cluster:') || key === 'outliers');
    // Visibility first: it is what this menu is reached for most, and it is
    // the one thing every row can do whatever it holds. Single-category calls
    // into the same functions the tree's menu uses, so both panels change the
    // map by exactly one path.
    const items = [
      {
        label: 'Isolate',
        title: `Show ${group.label} and nothing else`,
        run: () => setTreeFamilyIsolated([group]),
      },
      {
        label: 'Hide',
        title: `Hide ${group.label}`,
        run: () => setTreeFamilyVisible([group], false),
      },
      {
        label: 'Show all',
        title: 'Show every category',
        run: () => setTreeFamilyIsolated(null),
      },
    ];
    // Same rule as the tree's menu: the heat map measures one cluster's parts
    // against ONE centre, so it means something only on a row that IS a
    // cluster, and it colours the overlaid models, so it needs the model view.
    // The toolbar's "exactly one cluster checked" condition does not come
    // across -- the row under the pointer is the cluster it acts on.
    if (heatState) {
      items.push({ label: 'Clear heat map', run: () => clearHeat() });
    } else if (colorBy === 'cluster' && key.startsWith('cluster:') && displayMode === 'models') {
      items.push({
        label: 'Heat map',
        title:
          "Colour this cluster's parts by their true cosine similarity to the "
          + 'cluster centre. Shows how much the positions on the map, which are '
          + 'a 3D projection, agree with the full-dimensional similarity.',
        run: () => applyHeat(group, null),
      });
    }
    if (drillable) {
      const tooSmall = (group.count || 0) < MIN_DRILL_PARTS;
      // Still barred during a rebuild, as the button it replaced was: a drill
      // started now would read groupFileIds() off the mapData being replaced
      // and push the wrong file ids onto the stack. Decided when the menu is
      // opened, which is the only moment it can be acted on.
      items.push({
        label: 'Drill in',
        disabled: tooSmall || mapBusy,
        title: tooSmall
          ? `Too few parts to re-project (${group.count} < ${MIN_DRILL_PARTS})`
          : `Re-project and re-cluster the ${group.count} parts in ${group.label}`,
        run: () => drillInto(group).catch(() => {}),
      });
    }
    if (taggable) {
      items.push({
        label: 'Tag\u2026',
        disabled: mapBusy,
        title: `Tag every part in ${group.label}`,
        // Opens the shared tag picker where this menu is; the picker reports
        // what the write did, so there is nothing for the row to report.
        run: () => tagCluster(group),
      });
    }
    // Last, because it is about what the viewer is showing rather than about
    // the category itself. Drawn only in the 3D-model view (see
    // focusMenuItems), like the heat map above.
    for (const item of focusMenuItems([group])) items.push(item);
    return items;
  }

  function renderLegend(groups) {
    const signature = [
      getIndexName ? getIndexName() : '',
      kind,
      colorBy,
      // Drilling produces a different set of clusters even when the node names
      // repeat (C1, C2, ...), so the depth belongs in the signature; without it
      // the checkboxes would carry over onto unrelated categories.
      drillStack.length,
      drillTop().subsetId || '',
      (groups || []).map((g) => g.node_name).join(','),
    ].join('|');
    const remembered = legendStateFor(signature);
    // The signature an order would belong to. Set before anything is drawn,
    // so the Order select built below asks about the legend it is part of.
    legendOrderSignature = signature;
    // A new map has no order yet, and drawing one legend in another's order is
    // not possible -- legendRenderOrder() would simply find no cluster it
    // names. Fall back to Size so the select says what is on screen; asking
    // for Similarity again fetches the order for this map.
    if (legendOrderMode === 'similarity' && !legendOrders.has(signature)) {
      legendOrderMode = 'size';
    }
    // The row builders are shared with Ward's tree, so they read these rather
    // than close over renderLegend()'s locals. Cleared first: an early return
    // below must not leave a toolbar from the previous render in place.
    legendRemembered = remembered;
    legendTools = null;
    // The models this heat map coloured are about to be replaced, so the
    // overlay cannot survive the rebuild; drop it before the rows that own its
    // button are discarded.
    clearHeat();
    // The rows about to be discarded may own a hover preview; drop it without
    // restoring, because this render belongs to a map that is being rebuilt.
    endPeek({ restore: false });
    legendBodyEl.innerHTML = '';
    // Ward hands its clusters to the tree panel, so the legend box holds only
    // the crumbs and the summary and must not stretch; the
    // tree takes the height the rows would have used. Tag colouring is not a
    // cut of the tree, so its rows stay here and the box stretches as usual.
    // Set before any early return below, so a map with no groups at all leaves
    // it right too.
    legendBodyEl.style.flex =
      clustering === 'ward' && colorBy === 'cluster' ? '0 0 auto' : '';
    groupChecked.clear();
    // The rows these pointed at have just been discarded.
    legendRowOf.clear();
    // Every registered checkbox has just been discarded with the rows.
    groupCheckInputs.clear();
    renderCrumbs();
    if (!groups || groups.length === 0) return;
    // Every category's visibility, decided here and not in the row builder:
    // Ward draws no rows at all, so leaving this to them left the state empty
    // and everything that reads it -- the hover preview's restore, Show all /
    // Isolate / Hide, the tree's checkboxes -- working from different ideas of
    // what exists. One map, one set of categories, whatever is drawn.
    for (const group of groups) {
      // Untagged is the large grey majority; hide it by default so tagged
      // islands stay visible. Every other category starts visible.
      const defaultOn = group.key !== 'untagged';
      const checked = remembered.has(group.node_name)
        ? remembered.get(group.node_name)
        : defaultOn;
      groupChecked.set(group.node_name, checked);
      remembered.set(group.node_name, checked);
    }
    // Outside the scroll container, like the breadcrumbs: at 64 clusters the
    // rows scroll, and a summary that scrolled away with them would only be
    // readable from the top of the list.
    const tools = renderLegendTools(groups);
    legendTools = tools;
    // Title and rows live in a scroll container so the breadcrumbs above stay
    // visible; at 64 clusters the rows alone overflow the panel.
    const scroll = document.createElement('div');
    scroll.className = 'map-legend-scroll';
    legendBodyEl.appendChild(scroll);
    const wardTree = clustering === 'ward' && colorBy === 'cluster';
    const title = document.createElement('div');
    title.className = 'map-legend-title';
    title.textContent = 'Legend';
    // No rows follow when the tree owns them, and a heading over nothing reads
    // as a legend that failed to draw.
    if (!wardTree) scroll.appendChild(title);
    // Leaving the list entirely (between two rows, or off the panel) ends the
    // preview too; mouseleave on a row can be missed when the pointer moves
    // fast enough to skip the gap.
    scroll.addEventListener('mouseleave', () => endPeek());
    // Ward draws its clusters in the tree panel instead: the legend would be
    // one cut of that tree, so listing the same families twice would be two
    // names for one thing. The breadcrumbs and the summary
    // above still apply to the whole map, so they stay. Tag colouring is not a
    // cut of the tree -- it crosses the hierarchy -- so its rows are drawn here
    // as always, and the tree shows structure only.
    if (wardTree) {
      // Always null now that there is no toolbar to refresh. The handle and
      // its guards stay because renderLegendTools() is still where anything
      // map-wide above the rows would be built.
      if (tools) tools.refresh();
      highlightRevealed({ scroll: false });
      return;
    }
    for (const index of legendRenderOrder(
      groups,
      // The groups array is positional -- the viewer resolves a pick through
      // groups[node].offset -- so it is never reordered. Only the order the
      // rows are built in changes, and only when the user asked for it.
      legendOrderMode === 'similarity' ? legendOrders.get(signature) : null,
    )) {
      const group = groups[index];
      const row = document.createElement('label');
      row.className = 'map-legend-row';
      legendRowOf.set(group.node_name, row);
      // mouseenter/mouseleave, NOT mouseover/mouseout: the row holds a
      // checkbox, a swatch and several spans, and the bubbling pair would fire
      // again every time the pointer crossed into or out of one of them.
      row.addEventListener('mouseenter', () => schedulePeek(
        new Set([group.node_name]),
        row,
        { key: group.node_name, centreFileId: group.centre_file_id },
      ));
      row.addEventListener('mouseleave', () => endPeek());
      // Drill in and Tag live here, as they do on the tree's rows. A row with
      // nothing to offer -- tag colouring, or Ward -- opens no menu at all
      // rather than an empty one.
      row.addEventListener('contextmenu', (event) => {
        const items = groupMenuItems(group);
        if (!items.length) return;
        event.preventDefault();
        openRowMenu(event, items);
      });
      appendGroupIdentity(row, group, { stretchLabel: true });
      scroll.appendChild(row);
    }
    // The rows own the Tag-checked button's enabled state through their
    // checkboxes, so it can only be decided once they all exist.
    if (tools) tools.refresh();
    highlightRevealed({ scroll: false });
    markFocusedRows();
  }

  // The file ids a legend group owns. Groups describe contiguous runs of the
  // (group-ordered) points array, so this is a slice -- the same trick
  // tagCluster uses.
  function groupFileIds(group) {
    const start = group.offset || 0;
    const end = start + (group.count || 0);
    return (mapData.points || [])
      .slice(start, end)
      .map((p) => p && p.file_id)
      .filter(Boolean);
  }

  // Push a cluster onto the drill stack and rebuild. The cluster count goes back
  // to DEFAULT_N_CLUSTERS: a count chosen against the parent population says
  // nothing about the child, which is a different set of parts with its own
  // axes, so carrying it over just got in the way. The count being left behind
  // is remembered on the level we are leaving, because coming back up returns
  // to the SAME population -- there the chosen count does mean something, and
  // drillTo() restores it.
  async function drillInto(group) {
    if (!mapData || !group) return;
    const fileIds = groupFileIds(group);
    if (fileIds.length < MIN_DRILL_PARTS) return;
    const previousClusters = nClusters;
    const from = drillTop();
    const fromRemembered = from.nClusters;
    from.nClusters = previousClusters;
    drillStack.push({
      label: group.label,
      fileIds,
      subsetId: null,
      count: fileIds.length,
      nClusters: DEFAULT_N_CLUSTERS,
    });
    nClusters = DEFAULT_N_CLUSTERS;
    // Drill in: the child map re-clusters the subset, so every category name
    // on screen is about to be replaced.
    clearModelFocus();
    try {
      await requestLoad();
    } catch (err) {
      drillStack.pop();
      from.nClusters = fromRemembered;
      nClusters = previousClusters;
      throw err;
    }
    // After the (single) load, so the control never shows a count the map on
    // screen was not built with.
    notifyClusters();
  }

  // Jump to a level already on the stack (a breadcrumb click). Everything below
  // it is dropped; level 0 is the whole index. The count chosen at that level
  // is restored (drillInto() recorded it on the way down): the population is
  // the one it was chosen for, so unlike drilling in, it still means
  // something.
  async function drillTo(level) {
    if (level < 0 || level >= drillStack.length - 1) return;
    const previousClusters = nClusters;
    const removed = drillStack.slice(level + 1);
    drillStack.length = level + 1;
    const remembered = drillStack[level].nClusters;
    nClusters = remembered === null || remembered === undefined
      ? DEFAULT_N_CLUSTERS
      : remembered;
    // Drill out: the level being returned to is re-clustered as well.
    clearModelFocus();
    try {
      await requestLoad();
    } catch (err) {
      drillStack.push(...removed);
      nClusters = previousClusters;
      throw err;
    }
    notifyClusters();
  }

  // Tell the host which count the map on screen was built with, so its control
  // can follow. Deliberately does NOT go through setNClusters(): that would run
  // a second load() and a second SVD for a map that is already on screen.
  function notifyClusters() {
    if (typeof onClustersSet === 'function') onClustersSet(nClusters);
  }

  // "All (10230) > C4 (862) > C2 (137)" above the legend. Only drawn once the
  // user has drilled at least once, so the root map looks exactly as before.
  function renderCrumbs() {
    if (drillStack.length < 2) return;
    const bar = document.createElement('div');
    bar.className = 'map-crumbs';
    drillStack.forEach((entry, i) => {
      if (i > 0) {
        const sep = document.createElement('span');
        sep.className = 'map-crumb-sep';
        sep.textContent = '\u203a';
        bar.appendChild(sep);
      }
      const crumb = document.createElement('button');
      crumb.type = 'button';
      crumb.className = 'map-crumb';
      crumb.textContent = entry.count ? `${entry.label} (${entry.count})` : entry.label;
      // The last crumb is where we already are.
      setLockable(crumb, i === drillStack.length - 1);
      crumb.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        drillTo(i).catch(() => {});
      });
      bar.appendChild(crumb);
    });
    legendBodyEl.appendChild(bar);
  }

  // Bulk-tag every part in a cluster legend row. The cluster owns a contiguous
  // run of the (group-ordered) points array, so its file ids are a slice --
  // this tags the whole auto-sorted group across the full index, independent of
  // any search or the point/model display mode.
  function tagCluster(group) {
    if (!mapData || !group) return;
    tagParts(group.label, groupFileIds(group));
  }

  // Tag an arbitrary set of parts under one name. Split out of tagCluster()
  // because the tree tags nodes, which are families of clusters rather than a
  // single legend row.
  //
  // The name comes from the shared tag picker, opened as a popover where the
  // row menu was. It used to come from a modal of this file's own; the control
  // is shared now so that the legend, the tree, the gallery, the results and
  // the catalogue all offer the same candidates and report a partial write the
  // same way. The write itself is untouched: applyTagToParts() is still what
  // runs, with everything that has to follow a successful write.
  function tagParts(label, fileIds) {
    const name = getIndexName ? getIndexName() : null;
    if (!name) return;
    if (!fileIds || fileIds.length === 0) return;

    // Stop the hover preview before the popover takes the pointer: the pointer
    // is about to leave the row, and a preview left running would keep
    // overriding the visibility the user is tagging against.
    endPeek();
    openTagPopover({
      at: lastRowMenuPoint || undefined,
      title: 'Tag parts',
      // The target is fixed when the menu is opened -- these are the parts the
      // row stood for -- so the count never moves under the user.
      getSelection: () => ({ ids: fileIds, groups: null, note: label }),
      onApply: (tag, selection) => applyTagToParts(tag, selection.ids),
    });
  }

  // The write half of tagParts(), split out so the Catalogue tab can reach the
  // same path instead of growing an endpoint of its own. Only the half that
  // asks for a name is left behind: the catalogue builds its selection up over
  // many tiles and keeps it across several tags, so it takes the name from a
  // field of its own rather than a modal raised per apply.
  //
  // What follows a successful write is the part that must not be duplicated --
  // the tag lists have to be refreshed and, in Untagged-only mode, the map has
  // to be marked stale -- so it lives here, once.
  //
  // `reveal` is what the two callers differ on. From a legend row or a tree
  // node, tagging is the end of the gesture and the host jumps to the new tag
  // in Browse to review it. The catalogue is still being worked in: navigating
  // away would throw out the selection the user is about to tag again.
  //
  // Returns the server's { updated, skipped } so the caller can report both.
  // Nothing is swallowed: skipped ids (a part already at max_tags_per_part, or
  // an id the index does not know) are the caller's to show.
  async function applyTagToParts(tag, fileIds, { reveal = true } = {}) {
    const name = getIndexName ? getIndexName() : null;
    if (!name) throw new Error('No index is open.');
    if (!tag) throw new Error('No tag name was given.');
    if (!fileIds || fileIds.length === 0) throw new Error('No parts were given.');
    const { data } = await api.assignTag(name, tag, fileIds, { clientId: clientId() });
    if (typeof onClusterTagged === 'function') onClusterTagged(tag, { reveal });
    noteTaggedWhileUntaggedOnly(data.updated);
    return data;
  }

  // Sync the viewer's node visibility to the legend checkboxes once the map has
  // loaded (the checkboxes are rendered before the viewer is ready). This is
  // what actually hides the Untagged node on first load.
  function applyGroupVisibility() {
    if (!mapViewer || !mapData) return;
    // Drive from groupChecked, whose keys are the SCS node names the current
    // legend rendered. In both cloud and model mode the legend is the cloud
    // legend, so these toggle the point-cloud category nodes; the overlaid 3D
    // models (model mode) are always-on colour-matched landmarks.
    for (const [nodeName, checked] of groupChecked) {
      mapViewer.setGroupVisible(nodeName, checked !== false);
      if (displayMode === 'models') {
        // On the search map the overlaid models are the search hits, not a
        // legend category, so they stay visible whatever the checkboxes say:
        // switching every cluster off is how the hits are seen on their own,
        // still in the whole index's coordinates. Writing `true` rather than
        // skipping the call also means anything that hid a model -- a preview
        // that was running when the mode changed, say -- is undone the next
        // time this runs, instead of staying hidden with nothing to restore it.
        setMapModelGroupVisible(nodeName, searchMode ? true : checked !== false);
      }
    }
  }

  return {
    active: () => active,
    // Enter map mode (or re-draw the map after opening a part from it).
    async enter() {
      active = true;
      await requestLoad();
    },
    // Leave map mode. The caller restores whatever the viewer should show.
    exit() {
      active = false;
      // The bar is overlaid on the viewer, which is about to show something
      // else entirely.
      clearHeat();
      // clearHeat() returns early without a heat map, so the search map's own
      // bar has to be dropped here.
      searchColored = false;
      hideHeatBar();
      // A build may still be in flight. It cannot be aborted, but it must not
      // draw into a viewer that now shows something else, and the host must
      // not be left with its controls locked and the progress badge up. The
      // guards in load() make the run itself harmless; this releases the host
      // and drops anything queued behind it.
      if (loadPending) {
        const pending = loadPending;
        loadPending = null;
        pending.resolve();
      }
      if (mapBusy) setBusy(false);
    },
    // Reset the map's per-index controls to their defaults. Called when the
    // index changes so the next index's map starts clean -- kind Parts, Colour
    // by Cluster, View Point cloud, Outliers off, and the default Part size --
    // rather than inheriting the previous index's choices. The map is dismissed
    // when this
    // runs (the caller left map mode first), so this only stores state; it does
    // not rebuild. Cluster count is reset separately (see main.js resetClusters).
    resetControls() {
      colorBy = 'cluster';
      kind = 'part';
      displayMode = 'cloud';
      partScale = DEFAULT_PARTSCALE;
      // Mirrors the host's resetMapControls(), which unticks both checkboxes.
      trueScale = false;
      zoomOnHover = false;
      pendingCamera = null;
      outlierThreshold = 0;
      // Mirrors the host's resetMapControls(), which puts the combo back to PCA.
      projection = 'pca';
      clustering = 'kmeans';
      wardMax = DEFAULT_WARD_MAX;
      wardBand = null;
      // The tree described the previous index: its node ids name nothing here,
      // so the expansion goes with it. Drawn empty straight away, so the panel
      // says what it is rather than keeping the old index's rows on screen.
      treeData = null;
      treeError = null;
      treeOpen = new Set();
      treeSeeded = false;
      treeColorOf = new Map();
      treeColorByCategory = new Map();
      renderTree();
      untaggedOnly = false;
      untaggedPin = null;
      setUntaggedStale(0);
      resetDrill();
      // A new index: not one of the pinned names describes anything in it.
      clearModelFocus();
    },
    // Clear the current pick: drop the point marker (if any) and let the host
    // close the property panel. Used by the panel's X button so the user can
    // dismiss the panel without clicking empty space in the viewer.
    clearSelection() {
      if (mapViewer) mapViewer.clearPointMarker();
      if (onPointClear) onPointClear();
    },
    // Re-load the map with the current filter / colour scheme (only while the
    // map is the thing on screen).
    async refresh() {
      if (active) await requestLoad();
    },
    // Highlight a specific part on the map. Every /map point already carries its
    // file_id, so this is a pure lookup: find the point and mark it. Called when
    // a thumbnail is selected, so it deliberately does NOT move the camera and
    // does NOT report anything when the part is absent -- selecting a tile is
    // not a request to go somewhere, and a part can be legitimately outside the
    // current map (filtered by tag, or not among the model-mode
    // representatives). A previous highlight is dropped either way so a stale
    // mark never points at the wrong part. Returns whether the part was found,
    // for callers that care.
    highlight(fileId) {
      if (!mapData || !mapViewer) return { found: false };
      const point = fileId ? (mapData.points || []).find((p) => p && p.file_id === fileId) : null;
      if (displayMode === 'models') {
        clearMapModelHighlight();
        if (!point || !modelFileIds.has(fileId)) return { found: false };
        return { found: highlightMapModel(fileId), point };
      }
      if (!point) {
        mapViewer.clearPointMarker();
        return { found: false };
      }
      mapViewer.showPointMarker(point.x, point.y, point.z);
      return { found: true, point };
    },
    // Switch between the point cloud ('cloud') and the 3D-model map ('models')
    // and rebuild. The projection, clusters and colours are unchanged; only how
    // they are drawn differs.
    async setDisplayMode(value) {
      const next = value === 'models' ? 'models' : 'cloud';
      // Carry the current viewpoint across the toggle: both modes draw the same
      // point cloud in the same [-1, 1] space, so switching View should keep
      // where the user is looking instead of resetting to the iso overview.
      if (active && next !== displayMode) pendingCamera = getMapCamera();
      displayMode = next;
      // The heat map only exists in model view, and leaving it unloads the
      // models it coloured.
      clearHeat();
      if (active) await requestLoad();
    },
    // Rescale the 3D-model map's parts (the "Part" slider). No reload: the
    // viewer just rebuilds each model's transform. A no-op in cloud mode.
    setPartScale(value) {
      const n = Number(value);
      partScale = Number.isFinite(n) && n > 0 ? n : partScale;
      if (displayMode === 'models') setMapModelScale(partScale);
    },
    // Draw the overlaid models at one common scale instead of each at the
    // same on-screen size. Client-side only.
    setTrueScale(on) {
      trueScale = Boolean(on);
      if (displayMode === 'models') setMapModelTrueScale(trueScale);
    },

    // Move the camera to the hovered cluster as part of the legend preview.
    // Client-side only: no rebuild, nothing reloaded.
    setZoomOnHover(on) {
      zoomOnHover = Boolean(on);
    },
    // Report the current display mode so the host can show/hide the Part slider.
    getDisplayMode() {
      return displayMode;
    },
    // Turn the search-results map on or off. Forces Cluster colouring and the
    // 3D-model view, because the point cloud has to stay the k-means picture
    // and there is nothing to overlay without models. A rebuild is only needed
    // when one of those two actually has to change; otherwise just the model
    // selection is re-picked, which is far faster.
    async setSearchMode(on) {
      const next = Boolean(on);
      if (next === searchMode) return;
      // Any preview running now was set up under the old mode's rules; end it
      // (and restore what it hid) while those rules still apply.
      endPeek();
      if (next) {
        const ctx = getSearchContext ? getSearchContext() : null;
        if (!ctx || !ctx.hits || !ctx.hits.length) return;
        searchFileIds = new Set(ctx.hits.map((hit) => String(hit.id)));
      } else {
        searchFileIds = null;
      }
      searchMode = next;
      // The heat map colours the same models against a cluster centre; it
      // cannot share them with the search colouring.
      clearHeat();
      if (next && (colorBy !== 'cluster' || displayMode !== 'models')) {
        // Set both and rebuild once. Going through setColorBy()/setDisplayMode()
        // would cost two rebuilds, and setColorBy() also drops the drill trail
        // and the Untagged pin -- correct when the user picks a scheme, wrong
        // here, where the scheme is being forced to the one already being
        // compared against. The viewpoint is carried across as both of them do:
        // the points do not move.
        if (active) pendingCamera = getMapCamera();
        colorBy = 'cluster';
        displayMode = 'models';
        await requestLoad();
        return;
      }
      // Switching off: repaint before the selection changes, while the models
      // carrying the search colouring are still the ones on screen.
      if (!next) {
        restoreModelClusterColors();
        // The line has nothing left to count; renderSearchSummary() removes it.
        renderSearchSummary();
      }
      await scheduleModelRefresh();
    },
    // The search ran again (or was cleared) while the map is up. Re-read the
    // hit set and re-pick the models; a no-op when the search map is off.
    async refreshSearchHits() {
      if (!searchMode) return;
      const ctx = getSearchContext ? getSearchContext() : null;
      if (!ctx || !ctx.hits || !ctx.hits.length) {
        searchMode = false;
        searchFileIds = null;
        // The host's own switch is turned off from setMapControls(), but
        // setSearchMode(false) will see the mode already gone and return, so
        // the colouring has to be undone here.
        restoreModelClusterColors();
        renderSearchSummary();
        await scheduleModelRefresh();
        return;
      }
      searchFileIds = new Set(ctx.hits.map((hit) => String(hit.id)));
      await scheduleModelRefresh();
    },
    // Whether the search-results map is on, so the host can reflect it.
    searchActive: () => searchMode,
    // The Min similarity slider moved (or the Results filter changed): the
    // models on screen are the same, only their colours are stale.
    refreshSearchColors() {
      if (!searchMode) return;
      scheduleSearchColors();
    },
    // Report the current kind filter ('part' / 'assembly' / 'any'), so the host
    // can open the matching map before highlighting an item (parts and
    // assemblies are projected on separate maps).
    getKind() {
      return kind;
    },
    // Switch the baked colour scheme and re-fetch the SCS.
    async setColorBy(value) {
      const next = value === 'tag' ? 'tag' : 'cluster';
      // Only the baked colours change; the projection, kind and cluster count
      // are the same, so the points sit at the same [-1, 1] positions. Keep the
      // current viewpoint across the switch instead of resetting to the iso
      // overview, exactly as the View toggle does.
      if (active && next !== colorBy) pendingCamera = getMapCamera();
      // With Untagged only on, the two schemes cover different populations:
      // cluster colouring sees the untagged remainder, tag colouring the whole
      // index. A trail drilled through one of them, and a pin taken from it,
      // say nothing about the other.
      if (next !== colorBy && untaggedOnly) {
        resetDrill();
        untaggedPin = null;
      }
      colorBy = next;
      // Cluster rows and tag rows are different categories entirely, so a pin
      // taken in one colouring names nothing in the other.
      clearModelFocus();
      // Before the rebuild, not after: the tree describes the cluster
      // hierarchy, so under tag colouring it should go the moment the user asks
      // for tags rather than seconds later when the new map arrives.
      renderTree();
      // A tag colouring has no cluster centres, and the parts are about to be
      // repainted by the server's scheme anyway.
      clearHeat();
      if (active) await requestLoad();
    },
    // Restrict the map to single parts / assemblies (or all) and re-project.
    // This re-runs the projection and k-means on the subset, so it is as heavy
    // as a filter change (the delayed progress hint covers the wait).
    async setKind(value) {
      // Parts and assemblies are separate populations, so a trail drilled
      // through one of them says nothing about the other: start again at the
      // whole index.
      const next = value === 'part' || value === 'assembly' ? value : 'any';
      // Parts and assemblies are different populations, so a pin taken from one
      // says nothing about the other.
      if (next !== kind) {
        resetDrill();
        untaggedPin = null;
      }
      kind = next;
      // Parts and assemblies are clustered separately, and the budget changes
      // with them, so the pins are void either way.
      clearModelFocus();
      // A different population is projected and clustered from scratch, so the
      // cluster the heat map described no longer exists.
      clearHeat();
      if (active) await requestLoad();
    },
    // Change the number of colour clusters and re-project (re-runs k-means on
    // the server; the delayed progress hint covers the wait).
    async setNClusters(value) {
      // A positive number asks for that exact count, clamped to the 1-1024
      // range the server accepts (the selector offers a subset of it);
      // anything else falls back to the default.
      //
      // 1 is a deliberate option, not a degenerate one. With a single cluster
      // the centroid is the mean of the whole population, so the Outliers
      // threshold picks out exactly what stands away from THIS population
      // rather than from one of several sub-centres. Drilling into that single
      // cluster then re-projects on the population minus its outliers, which is
      // how a level gets cleaned up and looked at again.
      const n = Number(value);
      const next =
        Number.isFinite(n) && n > 0
          ? Math.min(1024, Math.max(1, Math.trunc(n)))
          : DEFAULT_N_CLUSTERS;
      // Idempotent: a rebuild reads state rather than applying a delta, so
      // setting the count it already has would produce the same map at the
      // cost of another clustering run. Nothing calls this to force a reload;
      // refresh() and reload() are what that is for.
      if (next === nClusters) return;
      nClusters = next;
      // Another k means another partition: cluster 4 of 64 and cluster 4 of
      // 128 are different sets of parts under the same name.
      clearModelFocus();
      if (active) await requestLoad();
    },
    // Re-project the map. Unlike the cluster count, this MOVES every point, so
    // the viewpoint is deliberately not carried across: the old camera framed a
    // layout that no longer exists, and the rebuilt map opens at its own
    // overview instead. A server-side re-projection, so it is not free -- see
    // the Layout control's title in index.html.
    async setProjection(value) {
      const next = String(value || '').toLowerCase();
      projection = MAP_PROJECTIONS.has(next) ? next : 'pca';
      if (active) await requestLoad();
    },
    // Re-group the map. The POINTS DO NOT MOVE -- only which cluster each one
    // belongs to, and therefore the colours, the legend and the baked .scs.
    // That is why the viewpoint is carried across here but deliberately not in
    // setProjection(): comparing k-means against Ward is worth doing from the
    // same camera, and the layout under it is identical.
    //
    // Ward's first build on an index takes tens of seconds (it builds one tree
    // over every part); the busy badge covers it, and every later cluster count
    // is served from that tree.
    async setClustering(value) {
      clustering = String(value || '').toLowerCase() === 'ward' ? 'ward' : 'kmeans';
      // k-means clusters and Ward's tree nodes are two different partitions.
      clearModelFocus();
      if (active) {
        pendingCamera = getMapCamera();
        await requestLoad();
      }
    },
    // Set the ceiling of Ward's tree -- the cluster count its deepest level
    // reaches -- and rebuild. Same path as setClustering(), because this is
    // the same kind of change: the server re-cuts the tree, so every category
    // on screen is replaced.
    async setWardMax(value) {
      const n = Number(value);
      const next = Number.isFinite(n) && n >= 2 ? Math.round(n) : DEFAULT_WARD_MAX;
      // Idempotent, like setNClusters(): the same ceiling cuts the same tree,
      // so neither the rebuild nor the discarding below has anything to do.
      // resetControls() clears this state on its own, so the one caller that
      // follows a reset does not depend on this call to do it.
      if (next === wardMax) return;
      wardMax = next;
      // The tree is re-cut, so every category is replaced -- the same reason
      // the cluster count clears the pins.
      clearModelFocus();
      // Every id in the tree names a node of a tree cut to the OLD ceiling, so
      // nothing about the old one survives: not which nodes were open, not
      // which colour each wore, not the tree itself. Carrying any of it over
      // would reopen and recolour unrelated families.
      treeData = null;
      treeError = null;
      treeOpen = new Set();
      treeSeeded = false;
      treeColorOf = new Map();
      treeColorByCategory = new Map();
      if (active) {
        pendingCamera = getMapCamera();
        await requestLoad();
      }
    },
    // Find one part of this map by file name (or file id) and go to it: mark
    // it in the viewer, frame it, point at its row in the panel and preview it,
    // which is exactly what clicking it would have done. Reports when there is
    // no such part, because a silent no-op reads as a broken control.
    async findPart(query) {
      return findPart(query);
    },
    // Go to one of the matches the last findPart() reported, by file id. Used
    // by the candidate list: file names are base names, so several parts can
    // answer to one name and the reader has to be able to pick.
    async selectPartMatch(fileId) {
      return selectPartMatch(fileId);
    },
    // Set the cosine-similarity threshold below which a point is split out of
    // its cluster into the Outliers group, and rebuild. 0 (or anything
    // unparseable) turns the split off. Clamped to [0, 1], the range a cosine
    // similarity between L2-normalised vectors can take.
    async setOutlierThreshold(value) {
      const n = Number(value);
      outlierThreshold = Number.isFinite(n) && n > 0 ? Math.min(1, n) : 0;
      // The threshold carves a category out of every cluster and the clusters
      // are renumbered after it, so the same name covers other parts.
      clearModelFocus();
      if (active) await requestLoad();
    },
    // Restrict the map to the parts that carry no tag yet (or go back to every
    // part) and rebuild. The filter is applied before the projection, so this
    // is a different population: the SVD axes and the centroids are computed
    // from the remainder alone, which is the whole point -- the parts left over
    // after the obvious clusters have been tagged get centroids of their own.
    async setUntaggedOnly(value) {
      const next = Boolean(value);
      if (next === untaggedOnly) return;
      untaggedOnly = next;
      // Switching the mode defines a new population, so the old pin is void.
      untaggedPin = null;
      // A drilled subset was taken from the previous population, so its
      // subset_id and file ids mean nothing against this one.
      resetDrill();
      clearHeat();
      if (active) await requestLoad();
    },
    // Rebuild the map from the index as it stands now. Offered while
    // Untagged-only mode is on, where tagging a cluster leaves its parts on a
    // map that no longer describes the index: they were untagged when it was
    // built. Re-clustering gives the parts that are still untagged centroids
    // of their own, which is the whole workflow -- tag what is obvious, reload,
    // and look at what separates out of the remainder.
    async reloadUntagged() {
      if (!active) return;
      // Drop the pin: this is the one request to go back to the index and ask
      // what is untagged NOW, which is what makes the tagged parts leave the
      // map and gives the remainder fresh centroids.
      untaggedPin = null;
      // Both describe clusters that are about to be replaced.
      clearHeat();
      endPeek({ restore: false });
      // A drilled subset was chosen from the population as it was before the
      // tagging; it says nothing about the one being built now.
      resetDrill();
      await requestLoad();
    },
    // The ceiling the tree on screen was cut at. The catalogue MUST ask for
    // the same one: the levels are 1, 2, 4, ... up to it, so a different
    // ceiling renumbers every node and the ids a catalogue returns would point
    // at other families entirely.
    getWardMax() {
      return wardMax;
    },
    // Which clustering is on screen ('ward' | 'kmeans'). The catalogue is a
    // frontier of Ward's tree, so it does not exist for k-means.
    getClustering() {
      return clustering;
    },
    // Show exactly the families a catalogue has ticked, opening the tree down
    // to each. Does not rebuild the map. See applyCatalogueSelection().
    async setCatalogueSelection(nodeIds) {
      return applyCatalogueSelection(nodeIds);
    },
    // Write one tag across an arbitrary set of parts, through the same path as
    // the tree's "Tag these N parts..." -- see applyTagToParts(). `reveal` is
    // false here because the catalogue stays where it is after a write.
    // Resolves with the server's { updated, skipped }; rejects on failure.
    async tagParts(tag, fileIds) {
      return applyTagToParts(tag, fileIds, { reveal: false });
    },
  };
}
