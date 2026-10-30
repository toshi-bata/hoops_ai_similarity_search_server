// Application entry point: wires the index picker, the query panel, the
// gallery, the search results, and the shared 3D viewer. Read this top to
// bottom to follow the whole flow.
//
// The core loop mirrors the desktop sandbox: single-click a part (in the
// gallery or in the results) -> it becomes the current query and its properties
// are shown; double-click it -> it opens in the 3D viewer. "Find similar"
// searches with the current query -> click a result to inspect it and search
// again. A file can also be uploaded as the query, but an uploaded file has no
// SCS so it cannot be previewed (only searched).

import './styles.css';
import * as api from './api.js';
import { PAGE_SIZE } from './config.js';
import { withLoading, clearError, showError } from './ui.js';
import { clientId } from './client-id.js';
import { state, selectIndex } from './state.js';
import { createGallery } from './gallery.js';
import { initSearch } from './search.js';
import { initEstimate } from './estimate.js';
import { initCompare } from './compare.js';
import { createTagFilter } from './tags.js';
import { createTagEditor } from './tag-editor.js';
import { createTagPicker, openTagPopover, refreshTagVocabulary, setTagCandidateIndex } from './tag-picker.js';
import { showPart, showEmptyViewer, showViewerMessage } from './viewer.js';
import { createShapeMap } from './shape-map.js';
import { createRegister } from './register.js';

// The virtual-scroll Browse gallery for the current index (null until chosen).
let gallery = null;

// The current index's statistics, kept so the properties area can revert to
// them when a part is deselected (the query is cleared).
let currentIndexStats = null;

// The unfiltered part (file) count of the current index, captured when the
// gallery is built with no filter, so a filtered view can show "shown / total".
let indexFileTotal = 0;

// The gallery tag filter, the results tag filter (client-side slice of the
// search hits) and the per-part tag editor. All are created once and pointed at
// the current index/part as it changes.
let tagFilter = null;
let resultsFilter = null;
let tagEditor = null;
let search = null;

// The two-part comparison controller (created once). It owns the compare
// overlay and the comparison scene; `hide()` dismisses the overlay when another
// view takes the viewer.
let compare = null;

// The Shape Map controller (created once, pointed at the current index) and the
// viewer's map state. `mapMode` is one of:
//   'off'            a normal part (or nothing) is in the viewer
//   'map'            the point-cloud map is in the viewer
//   'part-from-map'  a part opened from the map is in the viewer (the Map
//                    button returns to the map)
let shapeMap = null;
let mapMode = 'off';

// The Catalogue tab: the Ward frontier fetched by its Reload button, the
// virtual-scroll gallery drawn from it, and the node ids ticked on its tiles.
// `catalogueKMax` records the Max clusters the fetch was made at, so the
// catalogue can be thrown away the moment the tree stops matching it.
let catalogueGallery = null;
let catalogueGroups = [];
let catalogueChecked = new Set();
let catalogueKMax = null;
let catalogueNote = '';
// The tile items the gallery is drawing, in the order it is drawing them.
// Held because a tag write patches them in place rather than refetching.
let catalogueItems = [];
// Whether the stats block is expanded. Remembered for the session so a Reload
// or a tab switch does not fold it back up on a user who opened it.
let catalogueStatsOpen = false;
// True while a tag write is in flight. Locks the tab the same way mapBusy
// does: the write is not transactional, so a second one started on top of it
// would report counts that belong to neither.
let catalogueTagBusy = false;

// Whether the server's demo-only UI is enabled (HOOPS_AI_DEMO, read once at
// startup through /client-config). False until that answer arrives, and false
// for good if it never does: a demo feature appearing late, or appearing
// because a request failed, is worse than not appearing at all.
let demoEnabled = false;

// Whether the server can lay the map out with UMAP (umap-learn installed,
// read once at startup through /client-config). False until that answer
// arrives, and false for good if it never does: offering a layout the server
// cannot build would only produce a 503.
let umapAvailable = false;

// The active gallery/results/estimate tab, and whether the user has dismissed
// the property panel with its X. The panel is hidden on the Estimate tab or
// while dismissed; any selection change (a part click, a deselect, or an
// empty-space click) clears the dismissal and reveals it again.
let currentMode = 'browse';
let infoDismissed = false;

// The part currently shown in the info panel (single-click / map pick), used by
// the panel's Compare and Delete buttons. Null when the index properties are
// shown.
let infoPart = null;

// The Shape Map cluster count to fall back on when the server has not said how
// many this index should open with. Matches the option marked `selected` in
// index.html and DEFAULT_N_CLUSTERS in shape-map.js. The real default is
// per-index: /index/{name}/stats answers `default_clusters`, the same rule
// startup warmup builds with, so the map the client opens is the map that was
// warmed. This constant only covers a server too old to answer it.
const DEFAULT_MAP_CLUSTERS = '8';

// `default_clusters` for the current index, from /index/{name}/stats; null
// until it answers (or for a server that does not).
let indexDefaultClusters = null;
// Set once the user picks a count by hand, for the same reason as
// mapClusteringTouched: a late stats answer must not undo an explicit choice.
// Cleared on an index switch, where the count starts over (see resetClusters).
let mapClustersTouched = false;

// The count the Clusters control should hold: this index's own default, or the
// built-in fallback.
function defaultMapClusters() {
  return indexDefaultClusters === null ? DEFAULT_MAP_CLUSTERS : String(indexDefaultClusters);
}

// Ward's ceiling the map should open with. The operator's fixed ceiling if
// /client-config named one, otherwise this index's own default count: Ward's
// leaves and k-means' clusters are the same granularity, and warmup sizes them
// from the one rule, so they must land on the same number here too.
function defaultMapWardMax() {
  if (serverWardMax) return serverWardMax;
  if (indexDefaultClusters === null) return DEFAULT_MAP_WARDMAX;
  return nearestOptionValue('map-wardmax', indexDefaultClusters, DEFAULT_MAP_WARDMAX);
}

// The offered option closest to `value`, as a string. The automatic counts are
// powers of two and so already sit on the Ward ceiling's ladder, but rounding
// rather than rejecting keeps the control usable if that rule is ever changed.
function nearestOptionValue(selectId, value, fallback) {
  const el = document.getElementById(selectId);
  if (!el) return fallback;
  let best = null;
  let bestGap = Infinity;
  for (const option of el.options) {
    const n = Number(option.value);
    if (!Number.isFinite(n)) continue;
    const gap = Math.abs(n - value);
    if (gap < bestGap) {
      best = option.value;
      bestGap = gap;
    }
  }
  return best === null ? fallback : best;
}

// Adopt this index's default cluster count, from /index/{name}/stats. Applies
// to both clusterings: k-means takes it as Clusters (k), and Ward takes it as
// the Max clusters unless the operator fixed one for every index.
//
// THIS FUNCTION CAN REBUILD THE MAP. Call it only where stats have just been
// FETCHED -- an index switch, a part removal, a finished job. It must not be
// reached from a path that merely re-displays cached stats, such as clearing
// the selection: that rebuilt the map on every empty-space click in the 3D
// view, losing the Ward tree's expansion and the camera with it.
function applyIndexDefaultClusters(stats) {
  const value = stats && stats.default_clusters;
  if (!Number.isFinite(value)) return;
  indexDefaultClusters = value;
  if (!mapClustersTouched) {
    const count = defaultMapClusters();
    // Read the control BEFORE writing it: setClustersControl() makes the two
    // equal, so a comparison taken afterwards can only ever say "unchanged".
    const changed = !sameNumber(currentControlValue('map-clusters'), count);
    setClustersControl(count);
    // Only when it actually changes. The setter rebuilds, and this function is
    // one call away from several paths; a second guard here keeps a future
    // caller from paying for a rebuild that would produce the same map.
    if (changed && shapeMap && shapeMap.setNClusters) {
      shapeMap.setNClusters(count).catch(() => {});
    }
  }
  if (!mapWardMaxTouched && !serverWardMax) {
    const wardmax = defaultMapWardMax();
    const el = document.getElementById('map-wardmax');
    const changed = !sameNumber(el && el.value, wardmax);
    if (el) el.value = wardmax;
    if (changed && shapeMap && shapeMap.setWardMax) {
      // The tree is about to be renumbered, so anything the catalogue holds
      // names other nodes from here on.
      invalidateCatalogue('Max clusters changed.');
      shapeMap.setWardMax(wardmax).catch(() => {});
    }
  }
}

// A <select>'s current value, or null when the control is absent.
function currentControlValue(id) {
  const el = document.getElementById(id);
  return el ? el.value : null;
}

// Compare two control values numerically: a <select>'s value is a string, so
// '256' and 256 are the same setting.
function sameNumber(a, b) {
  const x = Number(a);
  const y = Number(b);
  return Number.isFinite(x) && Number.isFinite(y) && x === y;
}

// Default Shape Map control values, matching index.html and shape-map.js. On an
// index switch the controls are returned to these so the next index's map does
// not inherit the previous one's Show / Colour by / View / Part choices (see
// resetMapControls).
const DEFAULT_MAP_KIND = 'part';
const DEFAULT_MAP_COLORBY = 'cluster';
const DEFAULT_MAP_DISPLAY = 'cloud';
const DEFAULT_MAP_PROJECTION = 'pca';
const DEFAULT_MAP_CLUSTERING = 'kmeans';
// Clustering is the one map control that is a working habit rather than a
// per-index choice: someone who reads the map as a Ward tree reads every index
// that way. Remember the last choice in localStorage (same key style as
// client-id.js) and restore it when the controls are reset, which is also how
// it is restored on a fresh page -- the initial index selection resets them.
const MAP_CLUSTERING_KEY = 'hoops-ai.map.clustering';
const MAP_CLUSTERINGS = ['kmeans', 'ward', 'average'];
// The clusterings that build a tree, as opposed to k-means, which partitions
// the points afresh for every count.
const MAP_TREE_CLUSTERINGS = ['ward', 'average'];

// Does the clustering currently chosen in the toolbar build a tree? Ward and
// average linkage do; k-means does not. Written as a predicate rather than a
// comparison with 'ward' because that comparison is exactly what was missed
// when average linkage was added -- the control stayed on the k-means branch
// and the deeper Max clusters values were unreachable.
// This reads the <select> rather than shapeMap.getClustering(), unlike
// catalogueWardActive(): the controls are laid out from the DOM, including
// before the choice has reached the controller.
function treeClusteringSelected() {
  const el = document.getElementById('map-clustering');
  return MAP_TREE_CLUSTERINGS.includes(el ? el.value : '');
}

// What the server's startup warmup makes hot, from /client-config; null when it
// has not answered (yet) or warms nothing in particular.
let serverClustering = null;
// Every clustering the server's warmup made hot, from /client-config. Empty
// until it answers, and empty for a server that warms nothing.
let serverWarmClusterings = [];
// Set once the user picks a clustering by hand. A late /client-config answer
// must not pull the control back out from under them.
let mapClusteringTouched = false;

// The clustering the map opens with. A remembered habit is honoured as long as
// the server warmed it; only when it did not does the warm one win. Opening on
// a clustering that was not warmed means the first user pays the full build --
// the very cost warmup exists to remove -- while overriding a warm habit would
// take away a choice that costs nothing. With nothing warm, there is no cost
// to weigh and the habit simply stands.
function defaultMapClustering() {
  const stored = storedMapClustering();
  if (!serverWarmClusterings.length) return stored;
  if (serverWarmClusterings.includes(stored)) return stored;
  return serverWarmClusterings[0];
}

function storedMapClustering() {
  try {
    const stored = window.localStorage.getItem(MAP_CLUSTERING_KEY);
    if (MAP_CLUSTERINGS.includes(stored)) return stored;
  } catch {
    // localStorage blocked (private mode, disabled): use the default.
  }
  return DEFAULT_MAP_CLUSTERING;
}

function rememberMapClustering(value) {
  if (!MAP_CLUSTERINGS.includes(value)) return;
  try {
    window.localStorage.setItem(MAP_CLUSTERING_KEY, value);
  } catch {
    // Nothing to do: the choice simply will not survive this page.
  }
}

// Ward's ceiling is the server's call rather than a habit: it has to match
// what startup warmup cut the tree to, or the first map build is paid in full.
// /client-config answers it; this is only the fallback for a server that does
// not (and the value of the `selected` option in index.html).
const DEFAULT_MAP_WARDMAX = '256';

// What the server's warmup cut Ward's tree to, from /client-config; null until
// it answers.
let serverWardMax = null;
// Set once the user picks a ceiling by hand, for the same reason as
// mapClusteringTouched: a late answer must not undo an explicit choice.
let mapWardMaxTouched = false;
// "Off": the outlier split is opt-in, because it changes what a cluster means
// and the default map should show what k-means actually produced. The slider's
// lowest position is that Off -- a sentinel rather than a threshold -- which
// lets the client send a literal 0, the value the server treats as "behave
// exactly as before". One position below the usable range is spent on it, so
// the usable thresholds are 0.80 to 0.99. Nothing lower is worth offering:
// measurements showed a threshold at or below 0.85 separates nothing, because
// the cosine similarities to a cluster's centre are squeezed into roughly
// 0.88 to 1.00 in practice.
const MAP_OUTLIERS_OFF = '0.79';
const DEFAULT_MAP_OUTLIERS = MAP_OUTLIERS_OFF;
// The Part slider is logarithmic so that equal travel to either side means the
// same multiplicative change, giving symmetric shrink/grow room (a linear slider
// left almost no room below the default while leaving a huge unused range above
// it). The slider position runs [-1, 1] with 0 at the centre; the actual model
// scale multiplier is MAP_PARTSCALE_BASE * MAP_PARTSCALE_RATIO^position, i.e.
// 0.5 at centre, 0.1 at the far left and 2.5 at the far right (x5 each way).
const MAP_PARTSCALE_BASE = 0.5; // multiplier at the slider centre (matches shape-map DEFAULT_PARTSCALE)
const MAP_PARTSCALE_RATIO = 5; // max shrink/grow factor reached at each slider end
const DEFAULT_MAP_PARTSCALE = '0'; // slider centre position -> MAP_PARTSCALE_BASE

// Convert a Part-slider position in [-1, 1] into the actual model scale multiplier.
function mapPartscalePosToScale(position) {
  const pos = Number(position);
  const clamped = Number.isFinite(pos) ? Math.max(-1, Math.min(1, pos)) : 0;
  return MAP_PARTSCALE_BASE * Math.pow(MAP_PARTSCALE_RATIO, clamped);
}

// The bulk-registration workspace (created once, acts on the current index).
let register = null;

// Populate the index dropdown from the server, without attaching handlers or
// choosing a selection. Returns the raw index list so callers can pick an
// initial (or post-create/delete) selection. Shared by initIndexPicker() and
// the create/delete admin actions, so the option list is built in one place.
async function populateIndexPicker() {
  const select = document.getElementById('index-select');
  const indexes = await api.listIndexes();

  select.innerHTML = '';
  for (const index of indexes) {
    const option = document.createElement('option');
    option.value = index.name;
    const count = index.count != null ? ` (${index.count})` : '';
    option.textContent = index.name + count;
    select.appendChild(option);
  }
  return indexes;
}

// Enable the "Delete" button only for a real, selected index: there is nothing
// to delete when no index is selected.
function updateIndexAdminButtons() {
  const select = document.getElementById('index-select');
  const name = select.value;
  document.getElementById('index-delete').disabled = !name;
}

// Populate the index dropdown from the server and select an index to browse.
async function initIndexPicker() {
  const select = document.getElementById('index-select');
  const indexes = await withLoading(() => populateIndexPicker());

  select.addEventListener('change', () => {
    onIndexChosen(select.value).catch(() => {});
  });

  if (indexes.length === 0) {
    document.getElementById('gallery-status').textContent = 'No indexes found.';
    updateIndexAdminButtons();
    return;
  }

  // Prefer a warm index -- one the operator named in HOOPS_AI_WARMUP_INDEXES --
  // as the initial selection, so the first interaction lands on a ready index
  // instead of a cold one. Fall back to the first index (the previous
  // behaviour) when nothing is warm.
  const initial =
    indexes.find((index) => index.warm) ||
    indexes[0];
  select.value = initial.name;
  await onIndexChosen(initial.name);
}

// Switch to an index: read its total, then build a fresh virtual-scroll gallery
// that fetches parts on demand as they scroll into view.
async function onIndexChosen(name) {
  clearError();
  // Switching indexes dismisses the Shape Map: its point cloud belongs to the
  // previous index, so leave map mode and return the viewer to a neutral state
  // rather than leaving a stale cloud on screen.
  const wasInMap = mapMode !== 'off';
  leaveMapMode();
  // Return the cluster selector to the default (8): the count is a deliberate,
  // per-index choice, so it must not silently carry over to the next index the
  // user opens the map for.
  resetClusters();
  resetMapControls();
  selectIndex(name);
  if (wasInMap) {
    setViewerTitle(null);
    showViewerMessage('Double-click a part on the right, or open a file, to view it in 3D.');
  }
  updateFindEnabled();
  showIndexProperties(null);

  // Registration acts on the current index; enable the bulk-registration
  // button (in the row under the index picker) once an index is selected.
  document.getElementById('register-open').disabled = !name;
  updateIndexAdminButtons();

  // Load this index's tags into both filters before building the gallery, so the
  // gallery is built with whatever filter carries over cleared for the new
  // index. setIndex clears any previous selection.
  await tagFilter.setIndex(name);
  if (resultsFilter) await resultsFilter.setIndex(name);
  // The tag candidates every tagging control offers: the tags this index uses
  // (refetched here, they are per index) plus the application's vocabulary
  // (fetched once, it is not). Never fatal: without them a name can still be
  // typed.
  setTagCandidateIndex(name).catch(() => {});
  // Drop any search results from the previous index so they cannot linger in
  // the Results tab or feed an Estimate against the new index.
  if (search) search.reset();
  // Same for the catalogue: its node ids belong to the previous index's tree.
  invalidateCatalogue('Index changed.');

  // Show the index's own properties below the gallery until a part is selected,
  // matching the desktop sandbox. Failures here are non-fatal: the gallery
  // still works, so just leave the index-properties list empty.
  api
    .getIndexStats(name)
    .then((stats) => {
      showIndexProperties(stats);
      // Freshly fetched stats: this is one of the three places allowed to move
      // the Shape Map's cluster controls.
      applyIndexDefaultClusters(stats);
    })
    .catch(() => showIndexProperties(null));

  await buildGallery(name);
}

// Build (or rebuild) the Browse gallery for `name`, applying the current
// filter. Called when the index changes and whenever the filter changes,
// because the filter changes both the total and which parts each page returns.
async function buildGallery(name) {
  const filter = tagFilter.query();
  const status = document.getElementById('gallery-status');
  const first = await withLoading(() =>
    api.listParts(name, { offset: 0, limit: PAGE_SIZE, ...filter }),
  );
  state.total = first.total;
  // Remember the full count when unfiltered, then show "shown / total" once a
  // filter is applied so the user sees how much the filter narrowed the index.
  const filtered = Object.keys(filter).length > 0;
  if (!filtered) indexFileTotal = first.total;
  const shown = filtered
    ? `${first.total} / ${indexFileTotal} part(s)`
    : `${first.total} part(s)`;
  status.textContent = `${shown} in "${name}".`;

  if (gallery) gallery.destroy();
  gallery = createGallery({
    viewport: document.getElementById('gallery-grid'),
    total: first.total,
    fetchPage: (offset, limit) =>
      api.listParts(name, { offset, limit, ...filter }).then((page) => page.items),
    onSelect: onGallerySelect,
    onActivate: onGalleryActivate,
    onEmptyClick: onGalleryDeselect,
    onSelectionChange: updateGalleryBulkBar,
  });
  // The gallery was recreated, so its multi-select set is empty again. Reset the
  // bulk-tag bar.
  updateGalleryBulkBar(0);
}

// -- Catalogue --------------------------------------------------------------
//
// The Catalogue tab lists the frontier of the Ward tree: every branch cut at
// the first node whose cohesion reaches the threshold, plus the branches that
// never reached it. It is a cut of that tree, so it only exists under Ward
// clustering, and it must be fetched at the tree's own Max clusters -- the
// levels are 1, 2, 4, ... k_max, so a different ceiling numbers the nodes
// differently and the ids the server returns would name other nodes here.
//
// Nothing in this section rebuilds the map. Ticking a tile changes which
// categories are visible and which tree rows are open, both client-side; no
// cluster count, clustering or ceiling setter is called from this path. (See
// #194/#204, where a selection path that did call one rebuilt the map on every
// click and threw away the tree's expansion with it.)
const CATALOGUE_DEFAULT_THRESHOLD = 0.95;
const CATALOGUE_DEFAULT_MIN_PARTS = 2;
// The size at which "reached the threshold" is worth repeating to a user.
// Cohesion is entangled with group size: the mean over a three-part group's
// three pairs clears 0.95 far more easily than the mean over a 180-part
// group's 16,110, and a three-part group at 0.96 has been seen holding a part
// that plainly does not belong. Five is where the catalogue stops moving with
// the tree's depth on the measured index, which is what makes it a line and
// not a round number.
const CATALOGUE_DEFAULT_MIN_REACHED = 5;
// The server returns at most this many distinct tag names per group, so the
// local update after a write keeps to the same ceiling.
const CATALOGUE_TAG_NAMES_SHOWN = 8;

// How each group kind is shown. The word carries the meaning; the colour only
// repeats it, so nothing here depends on being seen in colour.
const CATALOGUE_KINDS = {
  confident: { label: 'Confident', cls: 'catalogue-confident' },
  small: { label: 'Small', cls: 'catalogue-small' },
  mixed: { label: 'Mixed', cls: 'catalogue-mixed' },
};

// Is a clustering with a tree on screen? The catalogue is a cut of that tree,
// so it exists for Ward and average linkage alike and for neither k-means nor
// no map at all.
function catalogueWardActive() {
  const c = shapeMap ? shapeMap.getClustering() : null;
  return MAP_TREE_CLUSTERINGS.includes(c);
}

// Read a number out of one of the tab's inputs, falling back to its default
// when the box is empty or holds something that is not a number.
function catalogueNumber(id, fallback) {
  const el = document.getElementById(id);
  const value = Number(el ? el.value : NaN);
  return Number.isFinite(value) ? value : fallback;
}

// The one line of state under the controls. The Ward guard wins over whatever
// the last fetch left behind: with k-means on, the catalogue cannot be built at
// all, and saying anything else would invite a Reload that must fail.
function renderCatalogueNote() {
  const el = document.getElementById('catalogue-note');
  if (!el) return;
  if (!state.indexName) el.textContent = 'Select an index first.';
  else if (!catalogueWardActive()) el.textContent = 'Catalogue requires Ward or Average clustering.';
  else el.textContent = catalogueNote || 'Press Reload to build the catalogue.';
}

function setCatalogueNote(text) {
  catalogueNote = text || '';
  renderCatalogueNote();
}

// Drop the fetched catalogue and everything drawn from it, and release the tree
// ticks it put there: once the tree it was cut from is gone, a tick taken from
// it has nothing left to stand for.
function clearCatalogue(note) {
  catalogueGroups = [];
  catalogueItems = [];
  catalogueChecked = new Set();
  catalogueKMax = null;
  if (catalogueGallery) {
    catalogueGallery.destroy();
    catalogueGallery = null;
  }
  const grid = document.getElementById('catalogue-grid');
  if (grid) grid.innerHTML = '';
  const stats = document.getElementById('catalogue-stats');
  const statsBody = document.getElementById('catalogue-stats-body');
  if (statsBody) statsBody.innerHTML = '';
  if (stats) stats.hidden = true;
  if (shapeMap) shapeMap.setCatalogueSelection([]).catch(() => {});
  setCatalogueTagStatus('');
  renderCatalogueSelection();
  setCatalogueNote(note || '');
}

// Called when the tree the catalogue was cut from stops being the tree on
// screen: Max clusters changed, the clustering changed, or the index changed.
// Anything already fetched is now about other nodes, so it goes.
function invalidateCatalogue(reason) {
  if (!catalogueGroups.length && catalogueKMax === null && !catalogueNote) {
    renderCatalogueNote();
    return;
  }
  clearCatalogue(`${reason} Press Reload to rebuild the catalogue.`);
}

// One "label: value" row of the stats block.
function catalogueStatRow(parent, label, value, head) {
  const row = document.createElement('div');
  row.className = head ? 'catalogue-stat catalogue-stat-head' : 'catalogue-stat';
  const l = document.createElement('span');
  l.textContent = label;
  const v = document.createElement('span');
  v.className = 'catalogue-stat-value';
  v.textContent = value;
  row.appendChild(l);
  row.appendChild(v);
  parent.appendChild(row);
}

function cataloguePercent(part, total) {
  if (!total) return '0.0%';
  return `${((part / total) * 100).toFixed(1)}%`;
}

// How good this cut was, from the server's own summary. The point is to let a
// user try another Max clusters and compare, so the conditions the catalogue
// was built under are shown alongside the result, not assumed.
//
// Collapsed to one line by default. The full block runs to thirty rows, which
// was taking a third of the panel away from the tiles the tab exists to show;
// the summary line keeps the two figures a user compares cuts on. The open
// state is remembered for the session (catalogueStatsOpen), so a user who
// wants it open does not have to reopen it after every Reload or tab switch.
function renderCatalogueStats(payload) {
  const host = document.getElementById('catalogue-stats');
  const el = document.getElementById('catalogue-stats-body');
  const head = document.getElementById('catalogue-stats-summary');
  if (!host || !el) return;
  el.innerHTML = '';
  const stats = payload && payload.stats;
  if (!stats) {
    host.hidden = true;
    return;
  }
  const total = payload.total_parts || 0;
  catalogueStatRow(el, 'Built with', '', true);
  catalogueStatRow(el, 'Max clusters (k_max)', String(payload.k_max));
  catalogueStatRow(el, 'Threshold', Number(payload.threshold).toFixed(4));
  catalogueStatRow(el, 'Min parts for "confident"', String(stats.min_reached_parts));
  catalogueStatRow(el, 'Parts in index', String(total));

  // Three classes, one partition: confident + small + mixed is the whole
  // index. "Small" is split out of "reached" because cohesion over a handful
  // of pairs is a much weaker claim than the same number over thousands.
  catalogueStatRow(el, 'Confident (tight, big enough)', '', true);
  catalogueStatRow(el, 'Groups', String(stats.confident_groups));
  catalogueStatRow(
    el,
    'Parts',
    `${stats.confident_parts} (${cataloguePercent(stats.confident_parts, total)})`,
  );
  catalogueStatRow(el, 'Largest group', String(stats.confident_largest));
  catalogueStatRow(el, 'Median group', String(stats.confident_median));

  catalogueStatRow(el, 'Small (tight, too few to judge)', '', true);
  catalogueStatRow(el, 'Groups', String(stats.small_groups));
  catalogueStatRow(
    el,
    'Parts',
    `${stats.small_parts} (${cataloguePercent(stats.small_parts, total)})`,
  );
  catalogueStatRow(el, 'Largest group', String(stats.small_largest));
  catalogueStatRow(el, 'Median group', String(stats.small_median));

  catalogueStatRow(el, 'Mixed (never got tight)', '', true);
  catalogueStatRow(el, 'Groups', String(stats.mixed_groups));
  catalogueStatRow(
    el,
    'Parts',
    `${stats.mixed_parts} (${cataloguePercent(stats.mixed_parts, total)})`,
  );
  catalogueStatRow(el, 'Largest group', String(stats.mixed_largest));
  catalogueStatRow(el, 'Median group', String(stats.mixed_median));
  catalogueStatRow(el, 'Single parts', String(stats.mixed_singletons));

  // The buckets count the whole frontier, before Min parts hides anything, and
  // the confident groups only: they answer "was this a good cut", which must
  // move neither with a display setting nor with the groups too small to
  // judge. `groups` below is the number actually shown.
  catalogueStatRow(el, 'Confident by size (whole frontier)', '', true);
  for (const bucket of stats.buckets || []) {
    catalogueStatRow(
      el,
      `size >= ${bucket.min_parts}`,
      `${bucket.groups} group(s), ${bucket.parts} part(s)`,
    );
  }
  catalogueStatRow(el, 'Shown', '', true);
  catalogueStatRow(el, 'Groups listed', String(stats.groups));
  catalogueStatRow(el, 'Omitted by Min parts', String(payload.omitted_by_min_parts || 0));
  // The collapsed line: how much there is, and how much of it is worth
  // trusting. Those are the two numbers a user compares one cut against
  // another with.
  if (head) {
    head.textContent =
      `${stats.groups} group(s) \u00b7 Confident ${stats.confident_groups} / ` +
      `${stats.confident_parts.toLocaleString()} part(s) ` +
      `(${cataloguePercent(stats.confident_parts, total)})`;
  }
  host.open = catalogueStatsOpen;
  host.hidden = false;
}

// How much of a group has already been dealt with. Three states, told apart by
// their words first: the background colour only repeats what the line says.
function catalogueTagState(group) {
  const count = Number(group.count) || 0;
  const tagged = Number(group.tagged_parts) || 0;
  if (tagged <= 0) return { cls: 'catalogue-untagged', text: `Untagged (0/${count})` };
  if (tagged >= count) return { cls: 'catalogue-tagged', text: `Tagged (${count}/${count})` };
  return { cls: 'catalogue-partial', text: `Partly tagged (${tagged}/${count})` };
}

// The tile fields that change when a group is tagged, built in one place so
// the first draw and the patch after a write cannot drift apart.
function catalogueItemFields(group) {
  const state = catalogueTagState(group);
  const names = group.tags || [];
  const shown = names.slice(0, 2).join(', ');
  const more = names.length > 2 ? ` +${names.length - 2}` : '';
  return {
    tileClass: state.cls,
    lines: [
      `${group.count} part(s) \u00b7 id ${String(group.centre_file_id || '').slice(0, 8)}`,
      // Height is the Ward merge distance. Unlike cohesion it does not vary
      // with how many members a group has, so two groups' heights compare
      // where their cohesions do not.
      `${group.cohesion == null ? 'coh -' : `coh ${Number(group.cohesion).toFixed(4)}`}` +
        ` \u00b7 ${group.height == null ? 'h -' : `h ${Number(group.height).toFixed(4)}`}`,
      state.text,
      // Kept as a line even when empty, so a group gaining its first tag does
      // not make its tile a row taller than its neighbours.
      names.length ? `tags: ${shown}${more}` : '',
    ],
  };
}

// The groups in the order the tiles are drawn in. The server returns the
// tree's own order (a pre-order walk), which is the default because it puts
// similar parts next to each other and lines the tab up with the tree panel.
// "Part count" is a sort of what has already been fetched -- biggest first,
// and among equals the tighter one.
function catalogueOrderedGroups() {
  const select = document.getElementById('catalogue-sort');
  if (!select || select.value !== 'count') return catalogueGroups;
  return [...catalogueGroups].sort(
    (a, b) => b.count - a.count || (b.cohesion || 0) - (a.cohesion || 0) || a.id - b.id,
  );
}

// One tile per group, through the gallery the Browse tab uses: same virtual
// scroll, same tiles, same checkboxes.
function buildCatalogueGallery(name) {
  if (catalogueGallery) catalogueGallery.destroy();
  catalogueItems = catalogueOrderedGroups().map((group) => {
    const kind = CATALOGUE_KINDS[group.kind] || CATALOGUE_KINDS.mixed;
    return {
      // The gallery keys its selection by `id`; the node id is what the tree
      // needs back, so it is the id here and the part id only rides along.
      id: String(group.id),
      filename: group.centre_filename || '(unnamed)',
      thumbnail_url: group.centre_file_id
        ? api.thumbnailUrl(name, group.centre_file_id)
        : null,
      // Colour repeats the word; the word is what carries the meaning.
      badge: kind.label,
      badgeClass: kind.cls,
      tags: [],
      centreFileId: group.centre_file_id,
      ...catalogueItemFields(group),
    };
  });
  const items = catalogueItems;
  catalogueGallery = createGallery({
    viewport: document.getElementById('catalogue-grid'),
    total: items.length,
    // Already in memory: the catalogue arrives whole, so a "page" is a slice.
    fetchPage: (offset, limit) => Promise.resolve(items.slice(offset, offset + limit)),
    onSelect: () => {},
    onActivate: (item) => {
      // Double-click opens the group's centre part, the same gesture the
      // gallery uses. Single-click is left alone: it would otherwise fight the
      // checkbox, which is what this tab is for.
      if (item.centreFileId) openPart({ id: item.centreFileId, filename: item.filename });
    },
    onEmptyClick: () => {},
    onSelectionChange: onCatalogueSelectionChange,
    metaLines: 4,
  });
  // Re-ticking what was ticked before: a re-sort rebuilds the gallery, and the
  // selection belongs to the catalogue rather than to any one ordering of it.
  if (catalogueChecked.size) catalogueGallery.setSelectedIds([...catalogueChecked]);
}

// A tile was ticked or unticked. One way only: the catalogue drives the tree,
// never the reverse, so nothing here listens back for the tree's own ticks.
function onCatalogueSelectionChange() {
  if (!catalogueGallery) return;
  catalogueChecked = new Set(catalogueGallery.getSelectedIds());
  // The last write's report is about a selection that no longer exists.
  setCatalogueTagStatus('');
  renderCatalogueSelection();
  if (!shapeMap) return;
  shapeMap
    .setCatalogueSelection([...catalogueChecked])
    .then((result) => {
      if (!result || result.ok) return;
      if (result.reason === 'tree-failed') {
        setCatalogueNote(`Could not load the tree: ${result.message}`);
      } else if (result.reason === 'not-ward') {
        setCatalogueNote('Catalogue requires Ward or Average clustering.');
      }
    })
    .catch(() => {});
}

// The parts the ticked groups cover. Counted as a union although the frontier
// is a partition and no part should appear twice: the partition is a property
// of how the frontier is cut, not of this list, and a future option that
// returns overlapping groups should make the count wrong rather than the write.
function catalogueSelectedPartIds() {
  const byId = new Map(catalogueGroups.map((group) => [String(group.id), group]));
  const ids = new Set();
  for (const key of catalogueChecked) {
    const group = byId.get(key);
    if (!group) continue;
    for (const fileId of group.file_ids || []) ids.add(fileId);
  }
  return [...ids];
}

// Groups and parts, always on screen, plus the control that writes the tag.
// Both come from the shared tag picker, so the catalogue counts, validates and
// reports exactly as every other place that tags does.
let cataloguePicker = null;

function renderCatalogueSelection() {
  // setCatalogueControls() is what tells the picker to look at the selection
  // again, so the count and the enabled state always move together.
  setCatalogueControls();
}

function setCatalogueTagStatus(text) {
  if (cataloguePicker) cataloguePicker.setStatus(text || '');
}

// Tick every group of one kind, or every group the tab is showing. This runs
// through the gallery's own selection set, so the tree follows exactly as it
// does when a single checkbox is clicked.
function catalogueSelectAll(kind) {
  if (!catalogueGallery || mapBusy || catalogueTagBusy) return;
  const ids = catalogueGroups
    .filter((group) => !kind || group.kind === kind)
    .map((group) => String(group.id));
  catalogueGallery.setSelectedIds(ids);
}

// Offer the index's existing tags and the application's tag vocabulary under
// the input. Kept in the shared picker rather than fetched here, so a name
// invented in this tab is offered by the tree's popover straight away.
async function refreshCatalogueTagList(name) {
  if (!name) return;
  await setTagCandidateIndex(name);
}

// Write one tag across the union of the ticked groups, through the map's own
// bulk-tag path (the same one the tree's "Tag these N parts..." uses). The
// selection is kept afterwards so a second tag can follow, and nothing here
// rebuilds the map: only the tag lists are refreshed.
//
// The name, the count, the validation and the result line belong to the shared
// tag picker; what is left here is the write and the in-place patch of the
// tiles it changed.
async function applyCatalogueTag(tag, ids) {
  if (!shapeMap) throw new Error('The map is not available.');
  catalogueTagBusy = true;
  setCatalogueControls();
  try {
    const data = await withLoading(() => shapeMap.tagParts(tag, ids));
    patchCatalogueTagState(tag, (data && data.skipped) || []);
    return data;
  } finally {
    catalogueTagBusy = false;
    setCatalogueControls();
  }
}

// Bring the ticked tiles up to date after a write, in place. Refetching would
// be exact, but it would also throw away the selection and the scroll position
// -- which is most of the work a user has done by the time they tag -- so the
// counts are kept locally and the server's own figures take over on the next
// Reload.
//
// A skipped member was refused because it already holds the maximum number of
// tags, which means it was already tagged; the count therefore never goes
// down, hence the max() rather than a plain assignment.
function patchCatalogueTagState(tag, skipped) {
  const refused = new Set((skipped || []).map((entry) => String(entry && entry.id ? entry.id : entry)));
  for (const group of catalogueGroups) {
    if (!catalogueChecked.has(String(group.id))) continue;
    const ids = group.file_ids || [];
    let missed = 0;
    for (const id of ids) if (refused.has(String(id))) missed += 1;
    group.tagged_parts = Math.max(Number(group.tagged_parts) || 0, ids.length - missed);
    const names = group.tags || [];
    if (!names.includes(tag)) group.tags = [...names, tag].slice(0, CATALOGUE_TAG_NAMES_SHOWN);
    const fields = catalogueItemFields(group);
    // Both the tile the gallery is holding and the item it would hand back on
    // a later page fetch, so a scroll away and back does not bring the old
    // state with it.
    const item = catalogueItems.find((entry) => entry.id === String(group.id));
    if (item) Object.assign(item, fields);
    if (catalogueGallery) catalogueGallery.updateItem(String(group.id), fields);
  }
}

// Fetch the catalogue for the current index. The only place that fetches: the
// inputs are read here, so editing one does not start work on its own.
async function loadCatalogue() {
  const name = state.indexName;
  if (!name) {
    clearCatalogue('');
    return;
  }
  if (!catalogueWardActive()) {
    clearCatalogue('');
    return;
  }
  const threshold = catalogueNumber('catalogue-threshold', CATALOGUE_DEFAULT_THRESHOLD);
  if (!(threshold > 0) || threshold > 1) {
    setCatalogueNote('Threshold must be greater than 0 and at most 1.');
    return;
  }
  const minParts = Math.max(1, Math.round(catalogueNumber('catalogue-minparts', CATALOGUE_DEFAULT_MIN_PARTS)));
  const minReachedParts = Math.max(
    1,
    Math.round(catalogueNumber('catalogue-minreached', CATALOGUE_DEFAULT_MIN_REACHED)),
  );
  // Sent explicitly rather than left to the server's default: the catalogue is
  // only addressable against the tree it was cut from.
  const kMax = Number(shapeMap.getWardMax());
  setCatalogueNote('Building the catalogue\u2026');
  try {
    const payload = await withLoading(() =>
      api.shapeMapCatalogue(name, {
        threshold, kMax, kind: 'part', minParts, minReachedParts,
        // The catalogue is a cut of the tree on screen, so it has to be cut
        // from the same linkage the map was clustered with.
        clustering: shapeMap.getClustering(),
      }),
    );
    catalogueGroups = Array.isArray(payload.groups) ? payload.groups : [];
    catalogueKMax = kMax;
    // A new catalogue stands for a new set of nodes, so the previous ticks --
    // and the tree state they produced -- are dropped rather than carried over.
    catalogueChecked = new Set();
    if (shapeMap) shapeMap.setCatalogueSelection([]).catch(() => {});
    renderCatalogueStats(payload);
    buildCatalogueGallery(name);
    setCatalogueTagStatus('');
    renderCatalogueSelection();
    refreshCatalogueTagList(name).catch(() => {});
    setCatalogueNote(
      `${catalogueGroups.length} group(s) at threshold ${Number(payload.threshold).toFixed(4)}, ` +
        `k_max ${payload.k_max}, min parts ${minParts}, confident at ${minReachedParts}+.`,
    );
  } catch (err) {
    clearCatalogue('');
    setCatalogueNote(`Could not build the catalogue: ${err && err.message ? err.message : err}`);
    showError(err);
  }
}

function initCatalogue() {
  // The inline shape of the shared tag picker. The selection is read on every
  // sync rather than pushed in, so ticking a tile moves the count without this
  // tab telling it to.
  cataloguePicker = createTagPicker({
    container: document.getElementById('catalogue-tagbar'),
    variant: 'inline',
    getSelection: () => ({ ids: catalogueSelectedPartIds(), groups: catalogueChecked.size }),
    onApply: (tag, selection) => applyCatalogueTag(tag, selection.ids),
    // A selection that has lost its tree -- or a map build in progress -- must
    // not be able to write to the index.
    isBlocked: () => mapBusy || catalogueTagBusy || catalogueGroups.length === 0,
  });
  document.getElementById('catalogue-reload').addEventListener('click', () => {
    loadCatalogue().catch(() => {});
  });
  document.getElementById('catalogue-select-confident').addEventListener('click', () => {
    catalogueSelectAll('confident');
  });
  document.getElementById('catalogue-select-shown').addEventListener('click', () => {
    catalogueSelectAll(null);
  });
  document.getElementById('catalogue-select-clear').addEventListener('click', () => {
    if (catalogueGallery) catalogueGallery.clearSelection();
  });
  // Reordering is a view of what has already been fetched, so it rebuilds the
  // tiles without going back to the server; the ticks are restored by
  // buildCatalogueGallery.
  const sort = document.getElementById('catalogue-sort');
  if (sort) {
    sort.addEventListener('change', () => {
      if (state.indexName && catalogueGroups.length) buildCatalogueGallery(state.indexName);
    });
  }
  const stats = document.getElementById('catalogue-stats');
  if (stats) {
    stats.addEventListener('toggle', () => {
      catalogueStatsOpen = stats.open;
    });
  }
  renderCatalogueSelection();
  renderCatalogueNote();
}

// Lock the tab while a map build or a tag write runs. The checkboxes are
// blocked with a class on the grid rather than a `disabled` on each input: the
// tiles are recycled as the list scrolls, so a per-element flag would come back
// off on its own.
//
// With no catalogue fetched -- never built, or thrown away because the tree it
// was cut from changed -- the selection and tag controls are off as well: a
// selection that has lost its tree must not be able to write to the index.
function setCatalogueControls() {
  const busy = mapBusy || catalogueTagBusy;
  for (const id of ['catalogue-threshold', 'catalogue-minparts', 'catalogue-minreached', 'catalogue-reload', 'catalogue-sort']) {
    const el = document.getElementById(id);
    if (el) el.disabled = busy;
  }
  const hasCatalogue = catalogueGroups.length > 0;
  for (const id of [
    'catalogue-select-confident',
    'catalogue-select-shown',
    'catalogue-select-clear',
  ]) {
    const el = document.getElementById(id);
    if (el) el.disabled = busy || !hasCatalogue;
  }
  // The picker disables its own field and button from isBlocked(); this only
  // asks it to look again.
  if (cataloguePicker) cataloguePicker.sync();
  const grid = document.getElementById('catalogue-grid');
  if (grid) grid.classList.toggle('catalogue-locked', busy);
}

// -- Query = the model open in the viewer ----------------------------------
//
// The search always runs on whatever is currently shown in the 3D viewer, like
// the desktop sandbox. A part becomes the query only when it is opened in the
// viewer (double-click a tile, or open a file). Single-click just previews a
// part's properties; it does not change the viewer or the query.

// Open an indexed part in the viewer and make it the current query.
function openPart(part) {
  if (compare) compare.hide();
  state.query = { fileId: part.id, filename: part.filename, indexed: true };
  updateFindEnabled();
  setViewerTitle(part.filename);
  showPart(api.scsUrl(state.indexName, part.id));
}

// Open an arbitrary uploaded CAD file: store it, author an SCS so it can be
// shown, view it, and make it the current query. Its name is shown on the
// viewer overlay (like every open part), so the bottom properties panel is left
// on the neutral index view rather than repeating the file name there.
async function openFile(file) {
  leaveMapMode();
  if (compare) compare.hide();
  const uploaded = await withLoading(() => api.uploadFile(file));
  const scs = await withLoading(() => api.exportScs(uploaded.file_id));
  // `filename` is the name this client sent; the server no longer echoes the
  // name it stored the bytes under, so there is nothing to unwrap here.
  const filename = uploaded.filename;
  state.query = { fileId: uploaded.file_id, filename, indexed: false };
  updateFindEnabled();
  setViewerTitle(filename);
  showPartProperties(null);
  showPart(scs.scs_url);
}

// Enable "Find similar" once there is a model in the viewer to search with,
// and reveal "Register open file" when that model is an uploaded file not yet
// in the current (writable) index.
function updateFindEnabled() {
  document.getElementById('search-run').disabled = !state.query;
  updateEstimateEnabled();
  const registerCurrent = document.getElementById('register-current');
  const canRegister =
    !!state.query &&
    state.query.indexed === false &&
    !!state.indexName;
  registerCurrent.hidden = !canRegister;
}

// Estimate runs over the parts narrowed in the Results tab, so it is enabled
// only once a similarity search has produced shown results (search.js reports
// this through getEstimateContext()). Re-evaluated on every search, slider move
// and results-filter change (via the search onResultsChange callback) and on
// query/index changes (via updateFindEnabled).
function updateEstimateEnabled() {
  // Nothing to enable while the Estimate UI is hidden (see applyDemoFlag).
  if (!demoEnabled) return;
  const canEstimate = !!(search && search.getEstimateContext());
  document.getElementById('estimate-run').disabled = !canEstimate;
}

// Fill a <dl> with label/value rows, skipping empty values.
function fillProps(dl, rows) {
  dl.innerHTML = '';
  for (const [label, value] of rows) {
    if (value == null || value === '') continue;
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    dd.textContent = String(value);
    dl.appendChild(dt);
    dl.appendChild(dd);
  }
}

// Show or hide the whole property panel (index + part properties). It is hidden
// on the narrow Estimate tab, or while the user has dismissed it with the X.
function updateInfoPanelVisibility() {
  const infoPanel = document.querySelector('.info-panel');
  if (infoPanel) infoPanel.hidden = currentMode === 'estimate' || infoDismissed;
}

// Show the current index's properties (nothing selected). Occupies the same
// spot as the part properties; the two swap as the selection changes.
function showIndexProperties(stats) {
  currentIndexStats = stats;
  // The cluster count carried on these stats is deliberately NOT applied here.
  // This function is also the way back from a selection to the index view, and
  // is called with the CACHED stats every time the selection is cleared --
  // including an empty-space click in the 3D view. Applying the count there
  // rebuilt the map under the user. The fetch sites call
  // applyIndexDefaultClusters() themselves.
  document.getElementById('part-info').hidden = true;
  if (tagEditor) tagEditor.clear();
  const dl = document.getElementById('index-props');
  dl.hidden = false;
  if (!stats) {
    dl.innerHTML = '';
    return;
  }
  // The index name is already shown in the picker above, and the schema
  // version is an internal storage-format detail, so neither is listed here.
  fillProps(dl, [
    ['Files', stats.files],
    ['Bodies', stats.bodies],
    ['Assemblies', stats.assemblies],
    ['Single parts', stats.single_part],
    ['Dimension', stats.dim],
    ['Model', stats.model],
  ]);
}

// Show a selected part's properties (single-click). Swaps in over the index
// properties; pass null to revert to the index properties. An indexed part
// (one with an id) also gets an editable tag list; an uploaded query file has
// no id and so cannot be tagged.
function showPartProperties(part) {
  // Any selection change (a part click, a deselect, or an empty-space click that
  // lands here with null) clears an earlier X dismissal and reveals the panel.
  infoDismissed = false;
  updateInfoPanelVisibility();
  if (!part) {
    infoPart = null;
    const compareBtn = document.getElementById('part-compare');
    if (compareBtn) compareBtn.hidden = true;
    const deleteBtn = document.getElementById('part-delete');
    if (deleteBtn) deleteBtn.hidden = true;
    showIndexProperties(currentIndexStats);
    return;
  }
  setPartThumb(null);
  document.getElementById('index-props').hidden = true;
  document.getElementById('part-info').hidden = false;
  const dl = document.getElementById('part-props');
  fillProps(dl, [
    ['File', part.filename],
    ['Kind', part.kind],
    ['Bodies', part.bodies],
    ['Registered', part.registered_at],
  ]);
  // Only indexed parts (with an id) can be compared, deleted or highlighted on
  // the map. Carry the kind because parts and assemblies are projected on
  // separate maps.
  infoPart = part.id ? { id: part.id, filename: part.filename, kind: part.kind } : null;
  // Offer "Compare with current model" for any indexed part; comparing against
  // the open model, or the lack of one, is checked
  // when the button is clicked so the affordance stays visible and discoverable.
  const compareBtn = document.getElementById('part-compare');
  if (compareBtn) compareBtn.hidden = !part.id;
  // Offer "Delete" (remove the part from the index) for any indexed part. Like
  // the others it stays visible whether the part was picked in the gallery, the
  // results, or on the map.
  const deleteBtn = document.getElementById('part-delete');
  if (deleteBtn) deleteBtn.hidden = !part.id;
  if (part.id) {
    tagEditor.show(state.indexName, part.id);
  } else {
    tagEditor.clear();
  }
}

// Clear the current query and show a bare viewer with nothing loaded -- the
// "Open Empty Viewer" menu action.
function openEmptyViewer() {
  leaveMapMode();
  if (compare) compare.hide();
  state.query = null;
  updateFindEnabled();
  setViewerTitle(null);
  showPartProperties(null);
  showEmptyViewer();
}

// Show the friendly name of the model currently open in the viewer as a
// top-left overlay. The model tree's top node shows the storage id (parts are
// imported under their file id), which is unreadable, so the app supplies the
// name it already knows here instead. Pass a falsy name to hide the overlay --
// used whenever the viewer holds no single model (empty, the map's point cloud,
// or a two-part comparison).
function setViewerTitle(name) {
  const el = document.getElementById('viewer-title');
  if (!el) return;
  el.textContent = name || '';
  el.hidden = !name;
}

// Top-left menu: "Open Empty Viewer" and "Open local file..." (the file
// upload previously lived as its own button in the search panel; moved here
// to match the reference HOOPS demo app's layout).
function initAppMenu() {
  const menuButton = document.getElementById('app-menu-button');
  const popup = document.getElementById('app-menu-popup');
  const fileInput = document.getElementById('query-file');
  const addCompareInput = document.getElementById('add-compare-file');

  function closeMenu() {
    popup.hidden = true;
    menuButton.setAttribute('aria-expanded', 'false');
  }
  function toggleMenu() {
    popup.hidden = !popup.hidden;
    menuButton.setAttribute('aria-expanded', String(!popup.hidden));
  }

  menuButton.addEventListener('click', (event) => {
    event.stopPropagation();
    toggleMenu();
  });
  document.addEventListener('click', (event) => {
    if (!popup.hidden && !popup.contains(event.target) && event.target !== menuButton) closeMenu();
  });

  document.getElementById('menu-open-empty').addEventListener('click', () => {
    closeMenu();
    openEmptyViewer();
  });
  document.getElementById('menu-open-file').addEventListener('click', () => {
    closeMenu();
    fileInput.click();
  });
  document.getElementById('menu-add-compare').addEventListener('click', () => {
    closeMenu();
    if (!state.query) {
      showError(new Error('Open a model first, then add a local file to compare against it.'));
      return;
    }
    addCompareInput.click();
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) openFile(fileInput.files[0]).catch(() => {});
    fileInput.value = '';
  });
  // Add a local file as the "added" part (B) and compare it against the model
  // already open (A). B is uploaded first so it has a file_id and an SCS.
  addCompareInput.addEventListener('change', () => {
    const file = addCompareInput.files[0];
    addCompareInput.value = '';
    if (!file) return;
    if (!state.query) {
      showError(new Error('Open a model first, then add a local file to compare against it.'));
      return;
    }
    withLoading(() => api.uploadFile(file))
      .then((uploaded) =>
        compareWithCurrent({
          fileId: uploaded.file_id,
          filename: uploaded.filename,
          indexed: false,
        }),
      )
      .catch((err) => showError(err));
  });
}

// Compare the part described by `b` against the model currently open in the
// viewer (A). A is read from the current query BEFORE the comparison starts,
// because starting it clears the query (see initCompare's onStart).
function compareWithCurrent(b) {
  if (!state.query) {
    showError(new Error('Open a model first, then compare another part against it.'));
    return;
  }
  const a = {
    fileId: state.query.fileId,
    filename: state.query.filename,
    indexed: state.query.indexed,
  };
  if (compare) compare.startWith(a, b);
}

// -- Shape Map mode --------------------------------------------------------
//
// Map mode replaces the viewer content with the index's point cloud. A single
// click on a point previews its part here (the viewer stays on the map); a
// double-click opens that part in 3D, with "Back to map" to return.

// Show or hide the "map" thumbnail in the info panel. Gallery and result
// selections have their own tiles, so they clear it; a picked map point sets it
// (a map point has no tile of its own).
function setPartThumb(url) {
  const img = document.getElementById('part-thumb');
  if (url) {
    img.src = url;
    img.hidden = false;
  } else {
    img.removeAttribute('src');
    img.hidden = true;
  }
}

// Reflect the current map mode in the viewer overlay controls.
// Whether the previous call saw the map on screen, so a map session can be told
// apart from a rebuild within one (see the legend handling below).
let inMapSession = false;
// Parts tagged since the map on screen was built (Untagged-only mode only);
// shown on the Reload button so the user can see how stale the map is.
let untaggedStale = 0;
// Whether a map rebuild is in flight. The controller reports this the instant a
// rebuild is requested, and every control that can trigger one is disabled for
// the duration: leaving them live is how two rebuilds used to get started at
// once (see the requestLoad() comment in shape-map.js).
let mapBusy = false;
let mapProgressTimer = null;
// A warm map returns in ~0.2 s, so only reveal the spinner once a build is
// actually slow; the common warm case must not flash one. This delay applies to
// the HINT only -- the controls lock immediately.
const MAP_PROGRESS_DELAY_MS = 500;

// The query the Part name box last committed, so the blur handler can tell a
// value the user changed from one that was already searched by Enter or Tab.
let lastFindPartQuery = null;

// What the Part name search found, in one short line under the box. The id is
// shortened to its first 8 characters: it is a content hash, and that much is
// enough to tell two parts with the same file name apart. An empty line is
// hidden by CSS, so a miss and an idle box look the same -- which they are.
// When more than one part matched, the candidates are also drawn as a list of
// buttons: point order differs between clusterings, so the first match is not
// the same part on a k-means map and a Ward map, and the reader has to be able
// to pick. Clicking one goes to it through the same cursor Enter walks.
function renderFindPartStatus(result) {
  const el = document.getElementById('map-findpart-status');
  const listEl = document.getElementById('map-findpart-list');
  if (listEl) {
    listEl.textContent = '';
    // One candidate needs no list: the line above already names it, and a
    // single-item list would only ask to be clicked for no effect.
    listEl.hidden = !result || !result.found || !result.list || result.list.length <= 1;
    if (!listEl.hidden) {
      result.list.forEach((item, i) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        const where = [];
        if (item.label) where.push(item.label);
        if (item.fileId) where.push(`id ${String(item.fileId).slice(0, 8)}`);
        btn.textContent = `${i + 1}. ${item.filename || ''}\n${where.join(' \u00b7 ')}`;
        if (i === result.index) btn.setAttribute('aria-current', 'true');
        btn.addEventListener('click', () => {
          if (!shapeMap) return;
          shapeMap.selectPartMatch(item.fileId).then((next) => {
            if (next) renderFindPartStatus(next);
          }).catch(showError);
        });
        listEl.appendChild(btn);
      });
    }
  }
  if (!el) return;
  if (!result || !result.found) {
    el.textContent = '';
    return;
  }
  const parts = [];
  // Only when there is more than one: "1/1" would ask the reader to check a
  // number that cannot be anything else.
  if (result.matches > 1) parts.push(`${result.index + 1}/${result.matches}`);
  if (result.label) parts.push(result.label);
  if (result.fileId) parts.push(`id ${String(result.fileId).slice(0, 8)}`);
  el.textContent = parts.join(' \u00b7 ');
}

// Shape Map onBusy handler: lock the map controls now, show the hint later.
function setMapBusy(busy) {
  mapBusy = Boolean(busy);
  window.clearTimeout(mapProgressTimer);
  mapProgressTimer = null;
  const progress = document.getElementById('map-progress');
  if (mapBusy) {
    // A build is starting, so the map the last search reported on is being
    // replaced. Cleared here rather than in setMapControls(), which also runs
    // for changes that leave the map alone.
    renderFindPartStatus(null);
    // The next commit addresses a different map, so nothing committed against
    // the old one should suppress it.
    lastFindPartQuery = null;
    mapProgressTimer = window.setTimeout(() => {
      mapProgressTimer = null;
      progress.hidden = false;
    }, MAP_PROGRESS_DELAY_MS);
  } else {
    progress.hidden = true;
  }
  setMapControls();
  // The catalogue's Reload and its tile checkboxes lock with the map controls:
  // both act on the tree that the running build is about to replace.
  setCatalogueControls();
}

function setMapControls() {
  const inMap = mapMode === 'map';
  const searchEl = document.getElementById('map-search');
  const hasSearch = Boolean(search && search.getSearchMapContext());
  // The switch cannot be offered before a search has run, and it must not be
  // left on when the search that fed it was cleared.
  if (!hasSearch && searchEl.checked) {
    searchEl.checked = false;
    if (shapeMap) shapeMap.setSearchMode(false);
  }
  const inSearch = inMap && searchEl.checked;
  document.getElementById('map-toggle').textContent = inMap ? 'Exit map' : 'Map';
  // The search map pins Show, Colour by and View: the point cloud underneath
  // has to stay the k-means picture the hits are being compared against, and
  // without models there is nothing to overlay. Hidden rather than disabled --
  // they are on the toolbar, where nothing below them can shift.
  document.getElementById('map-kind-wrap').hidden = !inMap || inSearch;
  document.getElementById('map-colorby-wrap').hidden = !inMap || inSearch;
  document.getElementById('map-display-wrap').hidden = !inMap || inSearch;
  // Hidden outright, not greyed out, when the server cannot lay the map out
  // with UMAP: a disabled control with one option invites the question
  // "why is this here".
  // Unlike Show / Colour by / View it stays available on the search map --
  // re-laying-out the cloud under a fixed hit set is a comparison worth making,
  // the same argument that keeps Clusters (k) and Outliers live there.
  document.getElementById('map-projection-wrap').hidden = !inMap || !umapAvailable;
  document.getElementById('map-projection').disabled = !inMap || mapBusy;
  // Always offered: Ward needs only scipy, which is a hard dependency, so there
  // is no availability to check as there is for the UMAP layout. Locked while a
  // build runs, like every other control that starts one.
  document.getElementById('map-clustering-wrap').hidden = !inMap;
  document.getElementById('map-clustering').disabled = !inMap || mapBusy;
  const isTree = treeClusteringSelected();
  // A tree is cut at a height, not asked for a count: show one control or the
  // other, never both. #map-legend-controls already hides the whole block
  // outside the map.
  document.getElementById('map-clusters-wrap').hidden = isTree;
  // The tree's counterpart, in the same place and under the same rule: the
  // ceiling its tree descends to. Meaningless for k-means, which takes the
  // count above.
  document.getElementById('map-wardmax-wrap').hidden = !isTree;
  document.getElementById('map-wardmax').disabled = !inMap || mapBusy;
  // Part name search: offered under both clusterings -- the panel lists
  // families either way, so a named part has no other way in. Locked during a
  // build like the rest, because the map it would search is being replaced.
  document.getElementById('map-findpart-wrap').hidden = !inMap;
  document.getElementById('map-findpart').disabled = !inMap || mapBusy;
  // Outliers is a k-means tool too: it splits each cluster by distance from
  // its centroid, which under a tree is what every node's cohesion already
  // reports. It would also stop the map's categories matching the tree's nodes
  // one for one, and that match is what the tree's colours, checkboxes and
  // tagging are built on. Hidden under Ward and average, unchanged for k-means.
  document.getElementById('map-outliers-wrap').hidden = isTree;
  // The Part-size slider only applies to the 3D-model view; show it there only.
  const inModels = inMap && document.getElementById('map-display').value === 'models';
  document.getElementById('map-partscale-wrap').hidden = !inModels;
  document.getElementById('map-truescale-wrap').hidden = !inModels;
  // The legend is opened when a map session starts and hidden when it ends, but
  // while the map stays up the toolbar's Legend button owns it: a colour, kind
  // or view change also calls this, and must not reopen a legend the user
  // closed.
  const legend = document.getElementById('map-legend');
  if (!inMap) legend.hidden = true;
  else if (!inMapSession) legend.hidden = false;
  inMapSession = inMap;
  const byTag = document.getElementById('map-colorby').value === 'tag';
  // What a running build locks, and what it leaves alone. The rule is: an
  // input that could start a SECOND server-side rebuild is held shut; anything
  // that only looks at what is already on screen stays live.
  // Locked: Show, Colour by, View, Clusters (k), Outliers, Untagged only and
  // its Reload -- every one of them asks the server for a new projection or
  // clustering. The legend's own actions are on each row's right-click menu
  // now, and the ones that would start a rebuild (Drill in, Tag) refuse while
  // mapBusy is set rather than being locked here (see groupMenuItems).
  // Live: the gallery, the Scale slider (client-side transform only), the
  // True scale switch (the same -- it only changes the divisor the models are
  // placed with, so no projection or clustering is asked for), the
  // legend's per-category checkboxes (they toggle visibility of nodes that are
  // already loaded), and the Map button -- that one deliberately, as the way
  // out of a build that is taking too long.
  // This is also why the map does not use ui.js withLoading(): that full-screen
  // overlay is for short operations whose result replaces the whole viewer,
  // whereas a map build takes seconds during which browsing the gallery and
  // handling the current model must keep working.
  // Every control below triggers a server-side rebuild, so all of them are held
  // shut while one is running; re-entering during a build is the race this
  // guards against.
  for (const id of ['map-kind', 'map-colorby', 'map-display']) {
    document.getElementById(id).disabled = !inMap || mapBusy;
  }
  // The clustering controls live in the legend panel and describe its
  // categories, so they are shown only where those categories come from
  // k-means. In tag colouring the colours are per-tag: the cluster count, the
  // outlier threshold and the untagged population all mean nothing there, so
  // the block is hidden outright rather than greyed out with an explanation.
  // The search map locks what would redefine the population it is drawn
  // against -- Untagged only and its Reload -- and leaves Clusters (k) and
  // Outliers live, because re-clustering the cloud under a fixed hit set is
  // the whole comparison. Both are disabled rather than hidden, since removing
  // a row would shift every legend row below it.
  document.getElementById('map-legend-controls').hidden = !inMap || byTag;
  searchEl.disabled = !inMap || mapBusy || !hasSearch;
  searchEl.title = hasSearch
    ? 'Overlay the parts found by the last similarity search instead of the '
      + 'per-cluster sample. The point cloud underneath stays coloured by cluster.'
    : 'Run a similarity search first: this overlays its results on the map.';
  const clusters = document.getElementById('map-clusters');
  clusters.disabled = !inMap || mapBusy;
  const outliers = document.getElementById('map-outliers');
  outliers.disabled = !inMap || mapBusy;
  // Only moves the camera -- no projection, no clustering, nothing loaded --
  // so unlike the controls above it is NOT locked while a build is running.
  // It is disabled on the search map, where the legend's hover preview is
  // switched off entirely and there would be nothing to zoom to.
  const zoomhover = document.getElementById('map-zoomhover');
  zoomhover.disabled = !inMap || inSearch;
  const untagged = document.getElementById('map-untagged');
  untagged.disabled = !inMap || mapBusy || inSearch;
  // The Reload button belongs to Untagged-only mode: outside it, tagging does
  // not change which parts the map covers, so there is nothing to rebuild for.
  // It stays on screen there and is merely disabled, because removing it would
  // move every legend row below it whenever the checkbox is toggled.
  const reload = document.getElementById('map-untagged-reload');
  reload.disabled = !inMap || mapBusy || !untagged.checked || inSearch;
  renderUntaggedReload();
}

// Put the outstanding count next to the Reload icon. The number and the title
// carry the whole state -- no styling, so nothing is added to the stylesheet
// for it.
function renderUntaggedReload() {
  const reload = document.getElementById('map-untagged-reload');
  if (!reload) return;
  const count = document.getElementById('map-untagged-reload-count');
  if (count) count.textContent = untaggedStale > 0 ? String(untaggedStale) : '';
  if (!document.getElementById('map-untagged').checked) {
    reload.title = 'Switch on Untagged only to re-cluster the untagged parts.';
    return;
  }
  reload.title = untaggedStale > 0
    ? `${untaggedStale} part(s) have been tagged since this map was built. `
      + 'Re-cluster what is still untagged.'
    : 'Re-cluster the parts that are still untagged.';
}

// Preview a picked map point's part: its thumbnail, file name, and tags. The
// viewer keeps showing the map so the point cloud is never lost.
function showMapPointInfo(point) {
  if (!point) {
    showError(new Error('Could not identify the clicked point; see the console for details.'));
    return;
  }
  showPartProperties({ id: point.file_id, filename: point.filename });
  setPartThumb(api.thumbnailUrl(state.indexName, point.file_id));
}

// A click hit no point: clear the info area back to the index-level view. A
// normal deselect, not an error, so no error banner here.
function clearMapPointInfo() {
  showPartProperties(null);
  setPartThumb(null);
}

// Open the part behind a double-clicked map point in 3D, remembering we came
// from the map so "Back to map" can return.
function openFromMap(point) {
  if (!point) {
    showError(new Error('Could not identify the double-clicked point; see the console for details.'));
    return;
  }
  openPart({ id: point.file_id, filename: point.filename });
  mapMode = 'part-from-map';
  setMapControls();
}

async function enterMapMode() {
  if (compare) compare.hide();
  mapMode = 'map';
  setMapControls();
  setViewerTitle(null);
  try {
    await shapeMap.enter();
  } catch {
    // The map failed to load (e.g. the point-cloud tool is not installed and
    // /map/scs returned 503). The error is already shown; leave map mode so the
    // viewer message and the rest of the app stay usable.
    exitMapMode();
  }
}

function exitMapMode() {
  shapeMap.exit();
  mapMode = 'off';
  setMapControls();
  setPartThumb(null);
  // Return the viewer to the current query, if any, else a neutral message.
  if (state.query) {
    setViewerTitle(state.query.filename);
    showPart(api.scsUrl(state.indexName, state.query.fileId));
  } else {
    setViewerTitle(null);
    showViewerMessage('Double-click a part on the right, or open a file, to view it in 3D.');
  }
}

// Leave map mode because a part is about to be opened directly (from a tile or
// an uploaded file). The caller then shows the part, so this only resets the
// map state and its overlay controls.
function leaveMapMode() {
  if (mapMode === 'off') return;
  shapeMap.exit();
  mapMode = 'off';
  setMapControls();
  setPartThumb(null);
}

// Highlight a selected part on the shape map, if the map happens to be on
// screen and that part happens to be on it. Called on every thumbnail
// selection, so it is deliberately passive: it never enters map mode, never
// changes the map's kind or filter, never moves the camera and never reports a
// failure. Selecting a tile is not a request to go somewhere, and a part can be
// legitimately absent from the current map (which is built for one kind and can
// be filtered by tag), so the right answer there is to do nothing.
function highlightOnMap(part) {
  if (mapMode !== 'map' || !shapeMap) return;
  // The map is built for one kind and can be filtered by tag, so the lookup by
  // file_id is also the membership test: an assembly selected while the parts
  // map is up is simply not found, and nothing happens.
  shapeMap.highlight(part && part.id ? part.id : null);
}

// Remove the selected part from the current index. A destructive, irreversible
// server action, so it confirms first. On success it refreshes everything that
// showed the part -- the index stats, the gallery, the tag filters, and (when
// the map is up) the map -- and returns the property panel to the index view.
async function deleteInfoPart(part) {
  const name = state.indexName;
  if (!part || !part.id || !name) return;
  // A plain confirm(), matching the index and tag deletes: this cannot be undone.
  const ok = window.confirm(
    `Delete "${part.filename || part.id}" from index "${name}"? This cannot be undone.`,
  );
  if (!ok) return;
  clearError();
  try {
    // withLoading() surfaces any error in the page banner (404 missing) and
    // re-throws; the catch just stops the refresh below.
    await withLoading(() => api.removeParts(name, [part.id]));
  } catch {
    return;
  }
  // The part is gone: return the property panel to the index view.
  showPartProperties(null);
  setPartThumb(null);
  // Refresh the index total shown in the properties.
  api
    .getIndexStats(name)
    .then((stats) => {
      showIndexProperties(stats);
      // Removing a part changes the part count, and so can change the count
      // the map opens on.
      applyIndexDefaultClusters(stats);
    })
    .catch(() => {});
  // Repopulate the index selector so its "name (count)" label reflects the new
  // file count. Rebuilding replaces the options, so restore the current
  // selection afterwards (setting value fires no change event).
  try {
    const select = document.getElementById('index-select');
    await populateIndexPicker();
    select.value = name;
    updateIndexAdminButtons();
  } catch {
    // Non-fatal: a stale count in the selector is cosmetic; the listings below
    // are still refreshed.
  }
  // Reload the tag filters and rebuild the gallery/results so the removed part
  // disappears from every listing.
  tagFilter.reload();
  if (resultsFilter) resultsFilter.reload();
  await buildGallery(name);
  // If the Shape Map is on screen it still shows the old point cloud (which
  // includes the removed part); rebuild it so the map reflects the deletion.
  if (mapMode === 'map' && shapeMap) await shapeMap.refresh();
}

function initMapUi() {
  shapeMap = createShapeMap({
    legendEl: document.getElementById('map-legend'),
    legendBodyEl: document.getElementById('map-legend-body'),
    // Where the controller draws Ward's hierarchy, the same way it draws the
    // legend: it holds the payload, so it holds the rendering.
    treeBodyEl: document.getElementById('map-tree-body'),
    onBusy: setMapBusy,
    // The drill-down changed its cluster count (reset on the way down, restored
    // on the way up); bring the control in line. DOM only -- the controller has
    // already reloaded.
    onClustersSet: setClustersControl,
    // The map that just loaded may have been built at a lower threshold than
    // the slider asked for; show what was really applied.
    onOutlierThreshold: renderMapOutliersEffective,
    // Parts were tagged while Untagged-only mode is on, so the map on screen no
    // longer matches the index. Reported on the Reload button; the rebuild is
    // the user's to ask for.
    onUntaggedStale: (count) => {
      untaggedStale = count;
      renderUntaggedReload();
    },
    // Read on demand, not captured: initSearch() runs after this, so `search`
    // is still null here.
    getSearchContext: () => (search ? search.getSearchMapContext() : null),
    onPointInfo: showMapPointInfo,
    onPointClear: clearMapPointInfo,
    onPointOpen: openFromMap,
    // Say what the Part name search found, beside the box. Without it a hit
    // and a miss look the same from the keyboard. The count matters: file
    // names are base names, so one name can belong to several parts, and
    // pressing Enter again steps to the next of them.
    onPartFound: renderFindPartStatus,
    getIndexName: () => state.indexName,
    // A set of parts was bulk-tagged from the map. Refresh the tag filters so
    // the new tag is selectable. With `reveal` (a legend row or a tree node)
    // the gesture is over, so switch the right panel to Browse and select that
    // tag in the gallery filter -- this jumps the user straight to the
    // just-tagged parts so they can review them and prune any that do not fit
    // (untick / "Remove tag from selected"). The map overlay stays as-is.
    //
    // The Catalogue passes `reveal: false`: it keeps its selection so another
    // tag can follow, and navigating away would take that selection off screen
    // while leaving it live.
    onClusterTagged: (tag, opts) => {
      if (resultsFilter) resultsFilter.reload();
      if (opts && opts.reveal === false) {
        tagFilter.reload();
        return;
      }
      setMode('browse');
      if (tag) tagFilter.selectTag(tag).catch(() => {});
      else tagFilter.reload();
    },
  });

  document.getElementById('map-toggle').addEventListener('click', () => {
    // Leave only from the map itself. From 'off' and from 'part-from-map'
    // alike the button goes in, so a part opened out of the map returns to
    // the map as it was rather than closing it. The map keeps its settings,
    // so one button covers all three.
    if (mapMode === 'map') exitMapMode();
    else enterMapMode();
  });
  document.getElementById('map-search').addEventListener('change', (event) => {
    const on = event.target.checked;
    if (on) {
      // The two combos the search map fixes are hidden while it is on, so
      // put the DOM in the state the map is really in before hiding them.
      document.getElementById('map-colorby').value = 'cluster';
      document.getElementById('map-display').value = 'models';
    }
    // Switching off deliberately leaves them where they are: the ordinary map
    // comes back as a 3D-model map in cluster colouring, which is the picture
    // the hits were being compared against, rather than jumping back to
    // whatever was on screen before.
    setMapControls();
    if (shapeMap) shapeMap.setSearchMode(on);
  });
  document.getElementById('map-colorby').addEventListener('change', (event) => {
    setMapControls();
    shapeMap.setColorBy(event.target.value).catch(() => {});
  });
  document.getElementById('map-kind').addEventListener('change', (event) => {
    shapeMap.setKind(event.target.value).catch(() => {});
  });
  document.getElementById('map-clusters').addEventListener('change', (event) => {
    mapClustersTouched = true;
    shapeMap.setNClusters(event.target.value).catch(() => {});
  });
  // Slider, but on "change" (drag released) rather than "input": every step
  // costs a full server-side rebuild, so reacting mid-drag would fire a chain
  // of them. Unlike #search-threshold, which only re-filters what is already in
  // the browser and can safely follow "input".
  const outliersSlider = document.getElementById('map-outliers');
  outliersSlider.addEventListener('input', () => {
    renderMapOutliersValue();
  });
  outliersSlider.addEventListener('change', (event) => {
    renderMapOutliersValue();
    shapeMap.setOutlierThreshold(mapOutliersToThreshold(event.target.value)).catch(() => {});
  });
  // The map is rebuilt on the untagged remainder, which is a different
  // population: the projection and the centroids are recomputed from it.
  document.getElementById('map-untagged').addEventListener('change', (event) => {
    shapeMap.setUntaggedOnly(event.target.checked).catch(() => {});
    // Enable or disable the Reload button with the mode it belongs to.
    setMapControls();
  });
  document.getElementById('map-untagged-reload').addEventListener('click', () => {
    shapeMap.reloadUntagged().catch(() => {});
  });
  document.getElementById('map-display').addEventListener('change', (event) => {
    setMapControls();
    shapeMap.setDisplayMode(event.target.value).catch(() => {});
  });
  document.getElementById('map-projection').addEventListener('change', (event) => {
    shapeMap.setProjection(event.target.value).catch(() => {});
  });
  document.getElementById('map-clustering').addEventListener('change', (event) => {
    // Swap the count selector in or out before the rebuild starts, so the
    // control that is about to be used is the one on screen.
    setMapControls();
    mapClusteringTouched = true;
    rememberMapClustering(event.target.value);
    // The catalogue is a cut of the Ward tree: under any other clustering it
    // describes nodes that are no longer there.
    invalidateCatalogue('Clustering changed.');
    shapeMap.setClustering(event.target.value).catch(() => {});
  });
  document.getElementById('map-wardmax').addEventListener('change', (event) => {
    mapWardMaxTouched = true;
    // A different ceiling renumbers the tree's nodes, so the ids the catalogue
    // holds would resolve to other families.
    invalidateCatalogue('Max clusters changed.');
    shapeMap.setWardMax(event.target.value).catch(() => {});
  });
  document.getElementById('map-partscale').addEventListener('input', (event) => {
    shapeMap.setPartScale(mapPartscalePosToScale(event.target.value));
  });

  document.getElementById('map-truescale').addEventListener('change', (event) => {
    shapeMap.setTrueScale(event.target.checked);
  });

  document.getElementById('map-zoomhover').addEventListener('change', (event) => {
    shapeMap.setZoomOnHover(event.target.checked);
  });

  // Enter searches; the type="search" clear button fires `search` with an empty
  // value, which drops the previous result's marker. Typing alone does not
  // search: the map has thousands of parts and every keystroke would be a
  // camera move.
  const findPartEl = document.getElementById('map-findpart');
  // Enter, Tab and clicking away all commit the search: a box that only
  // answered to Enter read as broken to anyone who tabbed out of it.
  const commitFindPart = () => {
    if (!shapeMap) return;
    lastFindPartQuery = findPartEl.value;
    shapeMap.findPart(findPartEl.value).catch(showError);
  };
  findPartEl.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' && event.key !== 'Tab') return;
    // Enter has nothing else to do here; Tab must still move the focus, so it
    // searches on the way out rather than instead of leaving.
    if (event.key === 'Enter') event.preventDefault();
    // Not suppressed when the value is unchanged: repeating the same query is
    // how the reader steps to the next part sharing that name.
    commitFindPart();
  });
  // Only when the value actually changed, so the blur that follows an Enter or
  // a Tab does not search a second time and walk the cursor on by itself.
  findPartEl.addEventListener('blur', () => {
    if (findPartEl.value !== lastFindPartQuery) commitFindPart();
  });
  findPartEl.addEventListener('search', () => {
    commitFindPart();
  });
  // Typing one character at a time is still not a search -- this map has
  // thousands of points and the camera would move on every keystroke -- but a
  // value that arrives all at once is treated as committed. inputType is what
  // tells the two apart: picking from the browser's search history reports
  // 'insertReplacementText', a paste reports 'insertFromPaste'. Some autofill
  // paths report no inputType at all, so an empty one counts as committed too.
  findPartEl.addEventListener('input', (event) => {
    const kind = event.inputType || '';
    if (kind
      && kind !== 'insertReplacementText'
      && kind !== 'insertFromPaste'
      && kind !== 'insertFromDrop') return;
    commitFindPart();
  });

  // The info panel's X button is two things at once: it drops the current
  // selection, and with nothing selected it closes the panel. On the map a
  // selection is a picked point, so X clears the marker and the panel falls
  // back to the index properties; press it again, with nothing left to
  // deselect, and the panel itself goes away. Without that second half the
  // button looked broken on the index properties, where there was no selection
  // for clearSelection() to drop and so nothing happened at all.
  //
  // `infoPart` is the selection the panel already keeps -- null exactly when
  // the index properties are what is on screen -- rather than a new flag that
  // could disagree with it.
  document.getElementById('part-info-close').addEventListener('click', () => {
    if (mapMode === 'map' && shapeMap && infoPart) {
      shapeMap.clearSelection();
      return;
    }
    if (mapMode === 'part-from-map' && infoPart) {
      clearMapPointInfo();
      return;
    }
    infoDismissed = true;
    updateInfoPanelVisibility();
  });

  // The info panel's "Compare with current model" button overlays the selected
  // indexed part (B) onto the model already open (A) and runs the comparison --
  // the index-side entry point, alongside the menu's "Add local file".
  document.getElementById('part-compare').addEventListener('click', () => {
    if (infoPart) {
      compareWithCurrent({ fileId: infoPart.id, filename: infoPart.filename, indexed: true });
    }
  });

  // The info panel's "Delete" button removes the selected part from the index
  // (a destructive, irreversible server action, so it confirms first).
  document.getElementById('part-delete').addEventListener('click', () => {
    if (infoPart) deleteInfoPart(infoPart).catch(() => {});
  });
}

// Point the Clusters control at a value without touching the map controller.
// Split out from resetClusters() because the drill-down calls this while the
// map is on screen, where shapeMap.setNClusters() would trigger a second
// load() and so a second ~7.6 s projection on top of the one the drill is
// already running. `value` is the controller's internal form: the count.
function setClustersControl(value) {
  const select = document.getElementById('map-clusters');
  if (!select) return;
  select.value = String(value);
}

// Return the selector to the default count. Used when switching indexes so the
// previous index's chosen count does not carry over: the map is dismissed, and
// re-opening it starts from the next index's own default, which arrives with
// its stats a moment later (see applyIndexDefaultClusters).
function resetClusters() {
  indexDefaultClusters = null;
  mapClustersTouched = false;
  setClustersControl(DEFAULT_MAP_CLUSTERS);
  // Keep the controller's state in step with the reset control. The map is
  // dismissed here, so this only stores the value; it does not trigger a build.
  shapeMap.setNClusters(DEFAULT_MAP_CLUSTERS);
}

// Convert the Outliers slider position to the threshold the server takes. The
// lowest position is "Off" and must send a literal 0: the server guarantees
// outlier_threshold=0 behaves exactly as it did before outliers existed, and a
// near-zero float would not take that path.
//
// The sentinel is matched as a string, which is safe only because it is the
// slider's minimum: a range input reports the shortest string that round-trips
// to the stepped number, so the bottom position is exactly "0.79" (the min and
// the initial value attribute are that same literal). Positions with a trailing
// zero come back shortened -- 0.80 reports as "0.8" -- but none of those is the
// sentinel, so they simply fall through to Number().
function mapOutliersToThreshold(value) {
  return String(value) === MAP_OUTLIERS_OFF ? 0 : Number(value);
}

// Mirror the Outliers slider into its readout: "Off" at the lowest position,
// two decimals elsewhere. Any "(max)" annotation from the last map is dropped
// here, because the moment the slider moves the old map's cap says nothing
// about the one being requested.
function renderMapOutliersValue() {
  const slider = document.getElementById('map-outliers');
  const out = document.getElementById('map-outliers-value');
  if (!slider || !out) return;
  const threshold = mapOutliersToThreshold(slider.value);
  out.textContent = threshold ? threshold.toFixed(2) : 'Off';
  out.removeAttribute('title');
}

// Report the threshold the server actually applied. Above a certain point no
// threshold can separate any more without emptying the map, so the server caps
// the request there; say so rather than let the readout claim a setting that
// was not used.
//
// The comparison is exact, with no tolerance. The server returns
// min(requested, ceiling), so an uncapped request comes back as the requested
// value itself -- the same IEEE double Number(slider.value) produced, unchanged
// by the JSON round trip. `applied < requested` is therefore precisely
// "capped", and a tolerance would not merely be unnecessary: half a slider step
// of slack hid real caps, one measured case having the ceiling 0.003 below the
// request.
function renderMapOutliersEffective(effective) {
  const slider = document.getElementById('map-outliers');
  const out = document.getElementById('map-outliers-value');
  if (!slider || !out) return;
  const requested = mapOutliersToThreshold(slider.value);
  const applied = Number(effective);
  if (!requested || !Number.isFinite(applied) || applied <= 0) {
    renderMapOutliersValue();
    return;
  }
  if (applied >= requested) {
    renderMapOutliersValue();
    return;
  }
  // Three decimals here, unlike the slider's two: a ceiling of 0.987 shown as
  // 0.99 would read as the requested value and say nothing about the cap.
  out.textContent = `${applied.toFixed(3)} (max)`;
  out.title =
    'No higher threshold can separate outliers in this clustering without '
    + 'leaving fewer than two points in their clusters, so the map was built '
    + `at ${applied.toFixed(3)}.`;
}

// Return the rest of the map controls -- Show, Colour by, View, Outliers and the
// Part size slider -- to their defaults when switching indexes, so the new
// index's map does not inherit the previous one's choices (only Clusters was
// reset before). The map is dismissed here, so this only sets the controls and
// the controller's stored state; it does not rebuild. setMapControls() then
// re-hides the Part slider, which only applies to the (now reset away) 3D-model
// view.
function resetMapControls() {
  const kind = document.getElementById('map-kind');
  const colorby = document.getElementById('map-colorby');
  const display = document.getElementById('map-display');
  const projectionEl = document.getElementById('map-projection');
  const clusteringEl = document.getElementById('map-clustering');
  const wardmaxEl = document.getElementById('map-wardmax');
  const partscale = document.getElementById('map-partscale');
  const truescale = document.getElementById('map-truescale');
  const zoomhover = document.getElementById('map-zoomhover');
  const outliers = document.getElementById('map-outliers');
  const untagged = document.getElementById('map-untagged');
  if (kind) kind.value = DEFAULT_MAP_KIND;
  if (colorby) colorby.value = DEFAULT_MAP_COLORBY;
  if (display) display.value = DEFAULT_MAP_DISPLAY;
  if (projectionEl) projectionEl.value = DEFAULT_MAP_PROJECTION;
  const clustering = defaultMapClustering();
  if (clusteringEl) clusteringEl.value = clustering;
  // The operator's fixed ceiling if /client-config named one, else this
  // index's own default count, else the built-in default.
  const wardmax = defaultMapWardMax();
  if (wardmaxEl) wardmaxEl.value = wardmax;
  if (partscale) partscale.value = DEFAULT_MAP_PARTSCALE;
  if (truescale) truescale.checked = false;
  if (zoomhover) zoomhover.checked = false;
  if (outliers) outliers.value = DEFAULT_MAP_OUTLIERS;
  if (untagged) untagged.checked = false;
  untaggedStale = 0;
  renderUntaggedReload();
  renderMapOutliersValue();
  // Keep the controller's stored state in step with the reset controls.
  if (shapeMap && shapeMap.resetControls) shapeMap.resetControls();
  // resetControls() returns the controller to the built-in default, so push the
  // remembered clustering back into it. The map is not active here, so this
  // only stores the value; it does not rebuild.
  if (clustering !== DEFAULT_MAP_CLUSTERING && shapeMap && shapeMap.setClustering) {
    shapeMap.setClustering(clustering).catch(() => {});
  }
  // Same for the remembered Ward ceiling.
  if (wardmax !== DEFAULT_MAP_WARDMAX && shapeMap && shapeMap.setWardMax) {
    shapeMap.setWardMax(wardmax).catch(() => {});
  }
  setMapControls();
}

// -- Selection -------------------------------------------------------------

// A gallery tile was single-clicked: highlight it and preview its properties.
// The viewer and the query are left unchanged (double-click opens it).
function onGallerySelect(part) {
  state.selectedPartId = part.id;
  gallery.setSelected(part.id);
  showPartProperties(part);
  highlightOnMap(part);
}

// A gallery tile was double-clicked: open it in the viewer as the new query.
function onGalleryActivate(part) {
  onGallerySelect(part);
  leaveMapMode();
  openPart(part);
}

// The gallery background (not a tile) was clicked, or the selection was
// otherwise dropped: clear the highlighted tile and fall back to the index
// properties, so an empty click reveals the index summary again.
function onGalleryDeselect() {
  // Nothing selected and the panel already shows the index properties (not
  // dismissed): there is nothing to revive, so do not repaint.
  if (
    state.selectedPartId == null &&
    document.getElementById('part-info').hidden &&
    !infoDismissed
  ) {
    return;
  }
  state.selectedPartId = null;
  if (gallery) gallery.setSelected(null);
  showPartProperties(null);
}

// A result tile was single-clicked: preview its properties only.
function onResultSelect(part) {
  showPartProperties(part);
  highlightOnMap(part);
}

// The results background (not a tile) was clicked: fall back to the index
// properties, so an empty click reveals the index summary (and revives the
// panel if it was dismissed), matching the gallery's empty-click behaviour.
function onResultDeselect() {
  showPartProperties(null);
}

// A result tile was double-clicked: open it in the viewer as the new query.
function onResultActivate(part) {
  showPartProperties(part);
  leaveMapMode();
  openPart(part);
}

// -- Gallery / Results toggle ----------------------------------------------

// Show the gallery, the last search results, or the last estimate. The viewer
// is shared; switching does not clear any view.
function setMode(mode) {
  const modes = ['browse', 'search', 'estimate', 'catalogue'];
  // Estimate is demo-only: with the flag off its tab is not on screen, so a
  // request for it (including a stale one from before the flag was known)
  // falls back to the gallery rather than showing a pane with no tab.
  const wanted = modes.includes(mode) ? mode : 'browse';
  const active = wanted === 'estimate' && !demoEnabled ? 'browse' : wanted;
  document.getElementById('mode-browse').classList.toggle('active', active === 'browse');
  document.getElementById('mode-search').classList.toggle('active', active === 'search');
  document.getElementById('mode-estimate').classList.toggle('active', active === 'estimate');
  document.getElementById('mode-catalogue').classList.toggle('active', active === 'catalogue');
  document.getElementById('browse-view').hidden = active !== 'browse';
  document.getElementById('search-view').hidden = active !== 'search';
  document.getElementById('estimate-view').hidden = active !== 'estimate';
  document.getElementById('catalogue-view').hidden = active !== 'catalogue';
  // The Estimate tab is narrow and scrolls on its own, so the property panel
  // (index properties, or a part's properties from a double-click) is hidden
  // there; it returns when switching back to Browse/Results unless the user has
  // dismissed it with the X.
  currentMode = active;
  updateInfoPanelVisibility();
  // Entering the Catalogue only refreshes its note (whether Ward is on, what it
  // is waiting for). It never fetches: the fetch is the Reload button's, so
  // switching tabs cannot start work the user did not ask for.
  //
  // The tag vocabulary is the one exception, and it is not catalogue data: the
  // inline picker's datalist has just become visible, and the file behind it
  // can be edited by hand at any time. It is read in the background, so the
  // tab is usable immediately either way.
  if (active === 'catalogue') {
    renderCatalogueNote();
    refreshTagVocabulary();
  }
}

function initModeToggle() {
  document.getElementById('mode-browse').addEventListener('click', () => setMode('browse'));
  document.getElementById('mode-search').addEventListener('click', () => setMode('search'));
  document.getElementById('mode-estimate').addEventListener('click', () => setMode('estimate'));
  document.getElementById('mode-catalogue').addEventListener('click', () => setMode('catalogue'));
  setMode('browse');
}

// Show or hide the demo-only Estimate UI (its button in the query panel and its
// tab), once the server's HOOPS_AI_DEMO flag is known.
//
// The elements are hidden, not removed, so every getElementById() here and in
// estimate.js keeps working and there is no second code path to maintain. The
// global `[hidden] { display: none !important }` rule takes them out of both
// flex rows, so the toggle simply becomes two tabs.
function applyDemoFlag(enabled) {
  demoEnabled = !!enabled;
  document.getElementById('estimate-run').hidden = !demoEnabled;
  document.getElementById('mode-estimate').hidden = !demoEnabled;
  if (!demoEnabled) {
    // Never leave the Estimate pane showing behind a tab that is not there.
    document.getElementById('estimate-view').hidden = true;
    if (currentMode === 'estimate') setMode('browse');
  } else {
    // The button starts disabled; a search may already have made it usable.
    updateEstimateEnabled();
  }
}

// Build the Layout options from what the server actually offers. PCA is always
// there -- it needs nothing but numpy and it is what every cached map was built
// with -- and UMAP is added only when /client-config says the server can build
// it. Removing an option is better than disabling it: a 503,
// or a map nobody asked for, is all that choosing an unavailable layout could
// produce.
function applyProjectionOptions() {
  const select = document.getElementById('map-projection');
  if (!select) return;
  const allowed = new Set(['pca']);
  if (umapAvailable) allowed.add('umap');
  Array.from(select.options).forEach((option) => {
    const ok = allowed.has(option.value);
    option.hidden = !ok;
    option.disabled = !ok;
  });
  if (!allowed.has(select.value)) select.value = 'pca';
}

// Ask the server once for the settings the client must act on. A failure is
// not reported: the client falls back to its safe defaults (no demo UI), and
// everything else on screen still works.
async function initClientConfig() {
  let settings = null;
  try {
    settings = await api.getClientConfig();
  } catch {
    settings = null;
  }
  applyDemoFlag(!!(settings && settings.demo));
  // Boolean() rather than a truthiness test on the raw field: a server built
  // before this flag existed answers without it, and undefined must fold to
  // false so the Layout control simply never appears there.
  umapAvailable = Boolean(settings && settings.umap_available);
  applyProjectionOptions();
  applyServerClustering(
    settings && settings.default_clustering,
    settings && settings.warm_clusterings,
  );
  applyServerWardMax(settings && settings.default_ward_max);
  setMapControls();
}

// Adopt what the server warmed, as far as this client can offer it. A server
// built before these fields existed answers without them, so an unknown or
// missing value simply leaves the client's own order in place. `warm` is the
// whole warm list; `value` is its first entry, read only when the list is
// absent (a server older than it).
function applyServerClustering(value, warm) {
  const el = document.getElementById('map-clustering');
  const offers = (name) => (el
    ? Array.from(el.options).some((option) => option.value === name)
    : MAP_CLUSTERINGS.includes(name));
  const list = Array.isArray(warm) ? warm.filter(offers) : [];
  if (!list.length && value && offers(value)) list.push(value);
  if (!list.length) return;
  serverWarmClusterings = list;
  serverClustering = list[0];
  // The map is normally still unbuilt here (the first index selection waits for
  // this answer), but if the user has already chosen a clustering by hand,
  // leave it alone: an explicit action outranks a default.
  if (mapClusteringTouched) return;
  // Not list[0]: a remembered choice that is warm is kept, which is the whole
  // point of being told the list rather than one name.
  const choice = defaultMapClustering();
  if (el) el.value = choice;
  if (shapeMap && shapeMap.setClustering) shapeMap.setClustering(choice).catch(() => {});
}

// Adopt the Ward ceiling the operator fixed, if it is one this client offers.
// Same rules as applyServerClustering(): a server built before this field
// existed, or a value outside the offered counts, leaves the client on the
// per-index default, and an explicit choice already made by hand is not undone.
// A null is the normal answer and means exactly that: no fixed ceiling, so the
// index's own default_clusters decides (see defaultMapWardMax).
function applyServerWardMax(value) {
  const el = document.getElementById('map-wardmax');
  const text = (value === null || value === undefined) ? '' : String(value);
  const offered = el
    ? Array.from(el.options).some((option) => option.value === text)
    : false;
  if (!offered) return;
  serverWardMax = text;
  if (mapWardMaxTouched) return;
  el.value = text;
  if (shapeMap && shapeMap.setWardMax) shapeMap.setWardMax(text).catch(() => {});
}

// -- Panel resizer ---------------------------------------------------------

// Let the user drag the handle to change the gallery panel's width. The panel
// is on the right, so its width is the distance from the pointer to the right
// edge of the window. We only update one CSS variable (--panel-width); the
// gallery's own ResizeObserver then recomputes how many tile columns fit, so
// the column count follows the width automatically.
const MIN_PANEL_WIDTH = 200;
const MAX_PANEL_WIDTH = 1200;

function initPanelResizer() {
  const resizer = document.getElementById('panel-resizer');
  const main = document.querySelector('.app-main');
  const toggle = document.getElementById('panel-toggle');
  const panel = document.querySelector('.side-panel');
  // The width the panel had when it was collapsed, restored when it comes
  // back. Not the default: someone who dragged the panel to 900px to read the
  // catalogue wants those 900px back, not 420.
  let widthBeforeCollapse = '';

  function onPointerMove(event) {
    const fromRight = document.documentElement.clientWidth - event.clientX;
    const width = Math.min(MAX_PANEL_WIDTH, Math.max(MIN_PANEL_WIDTH, fromRight));
    main.style.setProperty('--panel-width', `${width}px`);
  }

  function onPointerUp() {
    resizer.classList.remove('dragging');
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', onPointerUp);
    document.body.style.userSelect = '';
  }

  resizer.addEventListener('pointerdown', (event) => {
    // Nothing to resize while the panel is hidden, and a drag would silently
    // rewrite the width that showing it again is supposed to restore.
    if (panel && panel.hidden) return;
    event.preventDefault();
    resizer.classList.add('dragging');
    // Stop text selection while dragging over the page.
    document.body.style.userSelect = 'none';
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
  });

  // Collapse by hiding the element, NOT by setting --panel-width to 0.
  //
  // The gallery has its own ResizeObserver that recomputes how many tile
  // columns fit from the panel's width (see the comment above
  // initPanelResizer). A width of 0 keeps the element laid out and fires that
  // observer with a zero box, which is a column count nobody wants to reason
  // about. Hiding the element stops the observer firing at all, and showing it
  // again fires once with the real width.
  //
  // The resizer itself stays on screen either way -- it carries the only
  // control that brings the panel back.
  //
  // The viewer needs nothing here: bindResize() in viewer.js watches
  // #viewer-host, .viewer-pane is `flex: 1`, so removing the fixed-width panel
  // from the row grows the host and fires that observer. The same path the
  // drag already relies on.
  if (!toggle || !panel) return;
  toggle.addEventListener('pointerdown', (event) => {
    // The knob sits inside the drag handle; without this, pressing it would
    // also start a resize.
    event.stopPropagation();
  });
  toggle.addEventListener('click', () => {
    const collapsing = !panel.hidden;
    if (collapsing) {
      widthBeforeCollapse = main.style.getPropertyValue('--panel-width');
    }
    panel.hidden = collapsing;
    resizer.classList.toggle('collapsed', collapsing);
    resizer.title = collapsing ? '' : 'Drag to resize';
    toggle.title = collapsing ? 'Show the panel' : 'Hide the panel';
    toggle.setAttribute('aria-expanded', collapsing ? 'false' : 'true');
    toggle.innerHTML = collapsing ? '&lsaquo;' : '&rsaquo;';
    if (!collapsing && widthBeforeCollapse) {
      main.style.setProperty('--panel-width', widthBeforeCollapse);
    }
  });
}

// -- Tag filter and tag editor ---------------------------------------------

// The tag filter changed (a tag toggled, the any/all mode switched, or a tag
// was deleted from the index): re-apply it to the gallery ONLY. The filter is a
// gallery listing tool -- it no longer re-runs the search (which would switch
// away to Results) or re-project the map (which would narrow it to the tag
// subset). Both of those views deliberately ignore the filter now.
async function onTagFilterChange() {
  if (state.indexName) await buildGallery(state.indexName);
}

// A part's tags were saved (in the editor): refresh the tag counts and repaint
// any tiles showing that part so nothing is left stale.
function onTagsChanged(partId, tags) {
  tagFilter.reload();
  if (resultsFilter) resultsFilter.reload();
  if (gallery && partId && tags) gallery.updatePartTags(partId, tags);
  if (search && partId && tags) search.updateResultTags(partId, tags);
}

// Several parts were bulk-tagged in the results: repaint each affected gallery
// tile, then refresh the tag counts once (not once per part).
function onBulkTagged(changed) {
  for (const { id, tags } of changed) {
    if (gallery) gallery.updatePartTags(id, tags);
  }
  tagFilter.reload();
  if (resultsFilter) resultsFilter.reload();
}

// -- Gallery multi-select bulk tag / untag ---------------------------------
//
// The gallery tiles carry a corner checkbox (see gallery.js). Ticking any tile
// reveals this bar so the user can add or remove one tag across the hand-picked
// selection at once. It is the pruning half of cluster tagging: tag a whole
// cluster from the map, jump here (filtered by that tag), then untick/remove
// the tag from the few parts that do not belong.

// Reflect the current selection count in the bulk bar and show or hide it.
function updateGalleryBulkBar(count) {
  const bar = document.getElementById('gallery-bulk');
  if (!bar) return;
  bar.hidden = count === 0;
  const label = document.getElementById('gallery-bulk-count');
  if (label) label.textContent = `${count} selected`;
  // The tag actions need a selection; grey them out when nothing is ticked.
  const add = document.getElementById('gallery-bulk-add');
  const remove = document.getElementById('gallery-bulk-remove');
  if (add) add.disabled = count === 0;
  if (remove) remove.disabled = count === 0;
}

// Add or remove one tag across the multi-selected parts. The name comes from
// the shared tag picker, opened next to the button that was pressed, so the
// gallery offers the same candidates and reports a partial write the same way
// as every other place that tags. Remove keeps its own button: the picker is
// told which write to run, and the wording follows.
function openGalleryBulkPopover(remove, button) {
  clearError();
  if (!gallery || !state.indexName) return;
  openTagPopover({
    anchor: button,
    title: remove ? 'Remove a tag from the selected parts' : 'Tag the selected parts',
    applyLabel: remove ? 'Remove tag' : 'Tag selected',
    // Prefilled with the tag the view is filtered by (if any), so "Remove tag"
    // defaults to that tag -- the exact action a user wants when pruning parts
    // out of a just-tagged cluster. The bar used to keep a field of its own
    // for this; the field moved into the picker, the default did not.
    initialValue: galleryFilterTag(),
    progressVerb: remove ? 'Untagging' : 'Tagging',
    doneVerb: remove ? 'Untagged' : 'Tagged',
    failVerb: remove ? 'untag' : 'tag',
    getSelection: () => ({ ids: gallery ? gallery.getSelectedIds() : [], groups: null }),
    onApply: (tag, selection) => bulkTagSelected(remove, tag, selection.ids),
  });
}

// The single tag the Browse view is filtered by, or '' when it is not filtered
// by exactly one.
function galleryFilterTag() {
  const filter = tagFilter ? tagFilter.query() : {};
  return typeof filter.tags === 'string' ? filter.tags : '';
}

// The write. Skipped ids (e.g. anything not registered) come back to the
// picker, which reports them; the tag filters and gallery are rebuilt here so
// the view reflects the change.
async function bulkTagSelected(remove, tag, ids) {
  const call = remove ? api.unassignTag : api.assignTag;
  const { data } = await withLoading(() =>
    call(state.indexName, tag, ids, { clientId: clientId() }),
  );
  // Rebuild so tiles show the change and, in a tag-filtered view, parts that
  // lost the filter tag drop out. This also clears the selection.
  tagFilter.reload();
  if (resultsFilter) resultsFilter.reload();
  await buildGallery(state.indexName);
  return data;
}

function initGalleryBulkUi() {
  const add = document.getElementById('gallery-bulk-add');
  const remove = document.getElementById('gallery-bulk-remove');
  const clear = document.getElementById('gallery-bulk-clear');
  if (add) add.addEventListener('click', () => openGalleryBulkPopover(false, add));
  if (remove) remove.addEventListener('click', () => openGalleryBulkPopover(true, remove));
  if (clear) clear.addEventListener('click', () => { if (gallery) gallery.clearSelection(); });
}

function initTagsUi() {
  tagFilter = createTagFilter({
    container: document.getElementById('tag-filter'),
    onChange: () => onTagFilterChange().catch(() => {}),
  });
  // The results filter slices the search hits on the client. It has no Delete
  // and no kind options (the search kind is already fixed); changing it just
  // re-renders the shown results.
  resultsFilter = createTagFilter({
    container: document.getElementById('results-filter'),
    onChange: () => {
      if (search) search.refilter().catch(() => {});
    },
    showDelete: false,
    showKind: false,
    labelText: 'Filter results:',
  });
  tagEditor = createTagEditor({
    container: document.getElementById('tag-editor'),
    onTagsCommitted: onTagsChanged,
  });
}

// -- Registration workspace ------------------------------------------------

// Wire the two registration buttons in the row under the index picker:
// "Register files..." (the bulk workspace) and "Register open file" (the
// uploaded file open in the viewer). When a job finishes, refresh the index
// stats and rebuild the gallery so the newly added parts appear without a
// manual reload.
function initRegisterUi() {
  register = createRegister({
    getIndexName: () => state.indexName,
    onJobComplete: async () => {
      const name = state.indexName;
      if (!name) return;
      try {
        const stats = await api.getIndexStats(name);
        showIndexProperties(stats);
        // A finished job adds parts, so the index's default count can have
        // moved; this is a fresh fetch, so applying it is allowed here.
        applyIndexDefaultClusters(stats);
      } catch {
        // Non-fatal: the gallery rebuild below still reflects the new parts.
      }
      // Refresh the picker so the "(count)" beside the index name reflects the
      // parts just added; repopulating rebuilds the options, so restore the
      // current selection afterwards (setting value fires no change event).
      try {
        const select = document.getElementById('index-select');
        await populateIndexPicker();
        select.value = name;
      } catch {
        // Non-fatal: a stale count is cosmetic; the gallery below is current.
      }
      await buildGallery(name);
    },
  });

  document.getElementById('register-open').addEventListener('click', () => {
    if (register) register.open();
  });
  document.getElementById('register-current').addEventListener('click', () => {
    if (register && state.query && state.query.indexed === false) {
      register.registerCurrentFile(state.query.fileId, state.query.filename);
    }
  });
}

// Create / delete the index next to the picker. "New..." opens a small modal
// that takes a name and creates an empty index (the server picks the model);
// "Delete" confirms, then removes the selected index. Both re-list the indexes
// and land on a sensible selection afterwards, so the gallery never points at a
// gone index.
function initIndexAdmin() {
  const select = document.getElementById('index-select');
  const modal = document.getElementById('index-create-modal');
  const nameInput = document.getElementById('index-create-name');
  const errorBox = document.getElementById('index-create-error');
  const submit = document.getElementById('index-create-submit');

  function showCreateError(text) {
    errorBox.textContent = text || '';
    errorBox.hidden = !text;
  }

  function openCreate() {
    nameInput.value = '';
    showCreateError('');
    modal.hidden = false;
    nameInput.focus();
  }

  function closeCreate() {
    modal.hidden = true;
  }

  // Re-list the indexes after a create/delete and select `preferred` if it is
  // present, else fall back the same way the initial pick does (a warm,
  // browsable index first). Leaves the app in a valid state when the last
  // browsable index was just deleted.
  async function reselect(preferred) {
    const indexes = await withLoading(() => populateIndexPicker());
    if (indexes.length === 0) {
      document.getElementById('gallery-status').textContent = 'No indexes found.';
      document.getElementById('register-open').disabled = true;
      selectIndex(null);
      updateIndexAdminButtons();
      return;
    }
    const pick =
      indexes.find((index) => index.name === preferred) ||
      indexes.find((index) => index.warm) ||
      indexes[0];
    select.value = pick.name;
    await onIndexChosen(pick.name);
  }

  async function submitCreate() {
    const name = nameInput.value.trim();
    if (!name) {
      showCreateError('Enter a name.');
      return;
    }
    submit.disabled = true;
    showCreateError('');
    try {
      // Not routed through withLoading(): keep the error inside the modal (409
      // exists, 422 invalid) so the user can fix the name, instead of also
      // flashing the page-level banner.
      const created = await api.createIndex(name);
      closeCreate();
      await reselect(created.name);
    } catch (err) {
      showCreateError(describeAdminError(err));
    } finally {
      submit.disabled = false;
    }
  }

  async function deleteSelected() {
    const name = select.value;
    if (!name) return;
    // A plain confirm(), matching the tag delete (tags.js): this is a
    // destructive, irreversible server action.
    const ok = window.confirm(
      `Delete index "${name}" and all of its data? This cannot be undone.`,
    );
    if (!ok) return;
    clearError();
    try {
      // withLoading() surfaces any error in the page banner (404 missing) and
      // re-throws; the catch just stops the reselect.
      await withLoading(() => api.deleteIndex(name, { confirm: true }));
      await reselect(null);
    } catch {
      // Already shown by withLoading(); nothing to add.
    }
  }

  document.getElementById('index-create').addEventListener('click', openCreate);
  document.getElementById('index-create-close').addEventListener('click', closeCreate);
  document.getElementById('index-create-cancel').addEventListener('click', closeCreate);
  submit.addEventListener('click', () => submitCreate().catch(() => {}));
  nameInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') submitCreate().catch(() => {});
  });
  document.getElementById('index-delete').addEventListener('click', () => {
    deleteSelected().catch(() => {});
  });
  // Close when clicking the backdrop (outside the dialog).
  modal.addEventListener('click', (event) => {
    if (event.target === modal) closeCreate();
  });
}

// Turn any thrown error into a one-line message (server detail for ApiError).
// Used for the create modal's inline error line.
function describeAdminError(err) {
  if (err instanceof api.ApiError) return `${err.status}: ${err.detail}`;
  return err && err.message ? err.message : String(err);
}

initModeToggle();
initCatalogue();
// Hide the demo-only UI until the server says otherwise, so it never flashes on
// screen while /client-config is in flight; initClientConfig() then reveals it
// if HOOPS_AI_DEMO is set.
applyDemoFlag(false);
// Kept as a promise: the initial index selection resets the map controls, and
// those controls must already know which clustering the server warmed. See the
// bottom of this file, where initIndexPicker() waits for it.
const clientConfigReady = initClientConfig();
initPanelResizer();
compare = initCompare({
  getIndexName: () => state.indexName,
  onStart: () => {
    // A comparison replaces the viewer, but its base (A) is still the model in
    // focus, so keep it as the query. That lets a second "Compare with current
    // model" run against the same base instead of failing with no current
    // model, and keeps Find/Estimate meaningful. Just leave map mode first.
    leaveMapMode();
    // Two parts share the scene now, so the single-model name overlay does not
    // apply; the comparison panel identifies them instead.
    setViewerTitle(null);
  },
  onClose: () => {
    // The base part (A) was reloaded alone and stays the current query, so
    // restore its name overlay and clear any stale error banner.
    setViewerTitle(state.query ? state.query.filename : null);
    clearError();
  },
});
initAppMenu();
initTagsUi();
initGalleryBulkUi();
initMapUi();
initRegisterUi();
initIndexAdmin();
search = initSearch({
  onResultSelect,
  onResultActivate,
  onResultDeselect,
  onSearchDone: () => setMode('search'),
  onSearchRun: () => {
    if (resultsFilter) resultsFilter.reset();
  },
  getResultsFilter: () => (resultsFilter ? resultsFilter.query() : {}),
  onBulkTagged,
  onResultsChange: () => {
    updateEstimateEnabled();
    // The similarity slider is the search map's colour scale; moving it
    // repaints the models without touching the map or the server.
    if (shapeMap) shapeMap.refreshSearchColors();
  },
  onSearchHits: () => {
    // The hit set changed under the map: bring it in line, and re-evaluate
    // whether the Search results switch can be offered at all.
    if (shapeMap) shapeMap.refreshSearchHits();
    setMapControls();
  },
});
initEstimate({
  getEstimateContext: () => (search ? search.getEstimateContext() : null),
  // The Estimate tab is narrow, so a single click on a neighbour does nothing;
  // a double-click opens the neighbour in the viewer as the new query.
  onEvidenceActivate: onResultActivate,
  onEstimateDone: () => setMode('estimate'),
});
// Wait for /client-config before touching an index: choosing one resets the map
// controls, which read the clustering the server warmed. initClientConfig()
// never rejects (it falls back to the client's defaults), so this only delays
// the index list by one round trip, and nothing on screen depends on it first.
clientConfigReady.then(() => initIndexPicker().catch(() => {}));
