// Search controller: runs a part or assembly similarity search on the current
// query and shows the hits.
//
// The query (state.query) is set elsewhere by clicking a gallery tile, clicking
// a result, or uploading a file. An indexed part is searched by its file_id (no
// re-upload); an uploaded file is sent as multipart. Results are at most top_k,
// so this is a plain grid -- the virtual-scroll machinery from the Browse
// gallery is deliberately not used here. Result tiles reuse the shared renderer
// (tile.js); clicking one calls onResultSelect so the caller can view it and
// make it the next query (search chaining, like the desktop sandbox).

import * as api from './api.js';
import { ApiError } from './api.js';
import { TILE_WIDTH, SEARCH_TOP_K } from './config.js';
import { state } from './state.js';
import { clientId } from './client-id.js';
import { renderTile } from './tile.js';
import { openTagPopover } from './tag-picker.js';
import { withLoading, clearError, showError } from './ui.js';

// Wire up the search controls once.
//
//   onResultSelect(part)   called when a result is single-clicked (select)
//   onResultActivate(part) called when a result is double-clicked (open 3D)
//   onSearchDone()         called after results render (e.g. to show the tab)
//   onSearchRun()          called at the start of each search, before results
//                          render, so the caller can reset the results filter
//   getResultsFilter()     returns the results slice as { tags?, tagged? } (or
//                          {}), applied CLIENT-SIDE to the shown hits only; it
//                          never changes what the search ranked
//   onBulkTagged(changed)  called after a bulk assignment with an array of
//                          { id, tags }, so the caller can refresh many tiles
//                          and the tag counts with a single pass
//   onSearchHits()         called when the HIT SET itself changes (a search ran,
//                          or the state was reset), so the caller can re-read
//                          getSearchMapContext()
//
// The similarity search always ranks over the WHOLE index. The gallery's tag
// filter is deliberately not applied. A separate results filter (getResultsFilter)
// only slices what is already on screen, so "the gears among these hits" is a
// view of the same ranking, not a re-scoped search.
//
// Returns { refilter(), updateResultTags(id, tags) }.
export function initSearch({
  onResultSelect,
  onResultActivate,
  onResultDeselect,
  onSearchDone,
  onSearchRun,
  getResultsFilter,
  onBulkTagged,
  onResultsChange,
  onSearchHits,
}) {
  const runButton = document.getElementById('search-run');
  const status = document.getElementById('search-status');
  const grid = document.getElementById('search-grid');
  const thresholdInput = document.getElementById('search-threshold');
  const thresholdValue = document.getElementById('threshold-value');
  const bulkTagButton = document.getElementById('bulk-tag-apply');

  let selectedId = null;
  // The last search result and kind, kept so the similarity slider can re-filter
  // the shown tiles without running the search again.
  let lastResponse = null;
  let lastKind = 'part';
  // The query of the last search, kept so an Estimate can reuse the exact same
  // neighbourhood: it pins its neighbours to the shown hits (below) and passes
  // this query (with the fixed SEARCH_TOP_K) so the server rebuilds the same
  // search pool.
  let lastQueryFileId = null;
  // The hits currently on screen (above the similarity threshold); bulk tagging
  // acts on exactly this set.
  let shownHits = [];
  // The set of file ids the active results filter allows, or null when the
  // filter is All (everything passes). Resolved from the /parts listing because
  // search hits do not carry tags. Recomputed whenever the results filter
  // changes; a new search resets it to null.
  let resultsMembership = null;
  // Timing of the last search, split into two parts so the server does not look
  // slow when the client is the bottleneck:
  //   serverMs  the server's own time, read from the browser's Resource Timing
  //             for the search request (responseStart - requestStart): request
  //             sent -> first response byte. This excludes any client-side
  //             queueing, so it matches a direct curl call to the API.
  //   totalMs   wall-clock from the click until the result tiles are painted.
  // The gap between them is client overhead (mainly the browser queueing the
  // request behind other connections). Showing both, honestly, tells the user
  // the server is fast even when the total wait is longer
  // (e.g. "Search: 2.7 s (server) / 3.1 s (total)").
  let lastServerMs = 0;
  let lastTotalMs = 0;

  runButton.addEventListener('click', () => {
    runSearch().catch(() => {});
  });

  bulkTagButton.addEventListener('click', () => {
    openBulkTagPopover();
  });

  // Dragging the slider re-renders the results filtered by score, live.
  thresholdInput.addEventListener('input', () => {
    thresholdValue.textContent = Number(thresholdInput.value).toFixed(4);
    if (lastResponse) renderResults(lastResponse, lastKind);
  });

  // A click that lands on the grid itself -- a gap or the area below the tiles,
  // not a tile -- is an "empty click" that deselects, mirroring the gallery.
  grid.addEventListener('click', (e) => {
    if (e.target !== grid) return;
    selectedId = null;
    for (const t of grid.querySelectorAll('.tile.selected')) {
      t.classList.remove('selected');
    }
    if (onResultDeselect) onResultDeselect();
  });

  function currentKind() {
    const checked = document.querySelector('input[name="search-kind"]:checked');
    return checked ? checked.value : 'part';
  }

  async function runSearch() {
    clearError();
    if (!state.query) {
      showError(new Error('Open a part or a file in the viewer first, then search.'));
      return;
    }
    // A new search starts from an unfiltered result set: reset the results
    // filter to All before the hits are rendered.
    if (onSearchRun) onSearchRun();
    resultsMembership = null;

    const kind = currentKind();
    // include_self is always on: the query is a part in (or just like) the
    // index, and seeing the exact match at the top is the expected behaviour.
    // The search always ranks over the whole index (no tag scope).
    const options = {
      topK: SEARCH_TOP_K,
      includeSelf: true,
    };
    // The query is always the model open in the viewer, referenced by file id.
    const query = { fileId: state.query.fileId };

    let response;
    const startedAt = performance.now();
    try {
      response = await withLoading(() =>
        kind === 'assembly'
          ? api.searchAssembly(state.indexName, query, options)
          : api.searchParts(state.indexName, query, options),
      );
    } catch (err) {
      // The search failed. Make the failure visible in the Results panel (not
      // just the top banner) so an empty grid always explains itself. Replace
      // the message for the one case that needs explaining: a legacy (schema
      // v1) index cannot do assembly search and returns 409.
      let message;
      if (err instanceof ApiError && err.status === 409 && kind === 'assembly') {
        message =
          'This index does not support assembly search. It was built with an ' +
          'older schema (v1) that has no per-body vectors; rebuild the index ' +
          'to enable assembly search.';
        showError(new Error(message));
      } else if (err instanceof ApiError) {
        message = err.status ? `Error ${err.status}: ${err.detail}` : err.detail;
      } else {
        message = err && err.message ? err.message : String(err);
      }
      showSearchError(message);
      if (onSearchDone) onSearchDone();
      return;
    }
    // Prefer the server's true time from Resource Timing; fall back to the
    // wall-clock fetch time if no entry for THIS search is available (e.g. the
    // buffer was full, see serverTimeMs).
    const wallFetchMs = performance.now() - startedAt;
    const pathFragment = kind === 'assembly' ? '/search-assembly?' : '/search?';
    lastServerMs = serverTimeMs(pathFragment, wallFetchMs, startedAt);

    selectedId = null;
    lastResponse = response;
    lastKind = kind;
    lastQueryFileId = query.fileId;
    // The hit set has changed. Announced here and in reset() only -- NOT from
    // renderResults() or refilter(), which run on every nudge of the similarity
    // slider and would make the map unload and reload hundreds of models for a
    // ranking that has not moved.
    if (onSearchHits) onSearchHits();
    lastTotalMs = 0; // not known until the tiles are built and painted
    applyThresholdRange(kind);
    renderResults(response, kind);
    if (onSearchDone) onSearchDone();

    // Measure the total time only after the browser has laid out and painted
    // the tiles, then refresh the status line so it shows the full breakdown.
    // (Thumbnails load lazily afterwards and are not counted here.)
    requestAnimationFrame(() => {
      lastTotalMs = performance.now() - startedAt;
      if (lastResponse) {
        status.textContent = '';
        status.appendChild(buildStatusLine(lastResponse, lastKind));
      }
    });
  }

  // Set the similarity slider's range for the search kind and reset it to the
  // default. Parts score higher on average than assemblies, so each kind gets a
  // range that makes the useful band easy to reach.
  function applyThresholdRange(kind) {
    const min = kind === 'assembly' ? 0.4 : 0.6;
    thresholdInput.min = String(min);
    thresholdInput.max = '1';
    thresholdInput.value = '0.75';
    thresholdValue.textContent = '0.7500';
  }

  // Does a hit pass the results filter (tag slice)? The filter narrows only what
  // is shown. Search hits do NOT carry tags (tags live in a separate server-side
  // sidecar, not in the vector-store metadata each hit exposes), so the allowed
  // file ids are resolved from the /parts listing into `resultsMembership` and
  // the hit is matched by id. `resultsMembership` is null when the filter is All
  // (everything passes).
  function matchesResultsFilter(hit) {
    if (resultsMembership === null) return true;
    return resultsMembership.has(hit.id);
  }

  function renderResults(response, kind) {
    status.textContent = '';
    grid.innerHTML = '';

    // Client-side filter by the similarity slider AND the results tag filter.
    // hit.score is the ranking score (0..1); hits at or above the threshold and
    // matching the tag slice are shown. The shown set is kept so bulk tagging
    // acts on exactly these parts.
    const threshold = Number(thresholdInput.value);
    shownHits = response.hits.filter(
      (hit) => hit.score >= threshold && matchesResultsFilter(hit),
    );

    status.appendChild(buildStatusLine(response, kind));

    for (const hit of shownHits) {
      grid.appendChild(buildResultTile(hit, kind));
    }
    if (shownHits.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'search-empty';
      empty.textContent = 'No matching parts.';
      grid.appendChild(empty);
    }
    // The shown set changed (new search, slider move, or results filter): let
    // the host re-evaluate whether Estimate can run over these tiles.
    if (onResultsChange) onResultsChange();
  }

  // Re-render the current results with the current filters (threshold + results
  // tag filter), without re-running the search. Used when the results filter
  // changes. Resolves the tag membership first (a server call) because search
  // hits do not carry tags; on failure it falls back to showing everything
  // rather than an empty grid. A no-op before the first search.
  async function refilter() {
    if (!lastResponse) return;
    const filter = getResultsFilter ? getResultsFilter() : {};
    if (!filter || (!filter.tags && !filter.tagged)) {
      resultsMembership = null;
    } else {
      try {
        const ids = await api.listPartIdsByTag(state.indexName, {
          tags: filter.tags || undefined,
          tagged: filter.tagged ? true : undefined,
        });
        resultsMembership = new Set(ids);
      } catch (err) {
        resultsMembership = null;
        showError(err);
      }
    }
    if (lastResponse) renderResults(lastResponse, lastKind);
  }

  // Add one tag to every result currently shown (those above the similarity
  // threshold) in a single request. The name comes from the shared tag picker,
  // opened next to the button; the target is NOT the selection of some grid
  // but the shown hits, which is why the count the picker displays is taken
  // from shownHits and why it is read again at Apply time -- the slider may
  // have moved while the popover was open.
  function openBulkTagPopover() {
    clearError();
    openTagPopover({
      anchor: bulkTagButton,
      title: 'Tag shown results',
      applyLabel: 'Tag shown',
      getSelection: () => ({
        ids: shownHits.map((hit) => hit.id),
        groups: null,
        note: 'shown results',
      }),
      onApply: (tag, selection) => applyTagToShown(tag, selection.ids),
    });
  }

  // The write. Skipped ids (e.g. anything not registered) come back to the
  // picker, which reports them; the tiles and counts are refreshed here.
  async function applyTagToShown(tag, partIds) {
    const { data } = await withLoading(() =>
      api.assignTag(state.indexName, tag, partIds, { clientId: clientId() }),
    );
    // Reflect the new tag on every part that actually got it (skipped ids did
    // not), both in the shown response and, via a single coalesced callback,
    // in the gallery tiles and tag counts -- not one reload per part.
    const skippedIds = new Set(data.skipped.map((s) => s.part_id));
    const changed = [];
    for (const hit of shownHits) {
      if (skippedIds.has(hit.id)) continue;
      const meta = hit.metadata || (hit.metadata = {});
      const tags = meta.tags || [];
      if (!tags.includes(tag)) meta.tags = [...tags, tag];
      changed.push({ id: hit.id, tags: meta.tags });
    }
    if (lastResponse) renderResults(lastResponse, lastKind);
    if (onBulkTagged) onBulkTagged(changed);
    return data;
  }

  // Show a failed search in the Results panel so an empty grid always explains
  // itself. lastResponse is cleared so dragging the similarity slider cannot
  // re-render stale hits from a previous, successful search.
  function showSearchError(message) {
    lastResponse = null;
    shownHits = [];
    status.textContent = '';
    grid.innerHTML = '';
    const box = document.createElement('div');
    box.className = 'search-empty';
    box.textContent = `Search failed. ${message}`;
    grid.appendChild(box);
    if (onResultsChange) onResultsChange();
  }

  // The status line: the split search timing and hit count, plus the tag-filter
  // note when present. Showing the server round-trip apart from the total makes
  // it clear the server is fast even when building many tiles takes a moment. A
  // tag filter with matching_parts_in_index greater than count means the tag
  // exists in the index but those parts did not rank into the results -- a
  // distinction the server provides so it does not look like a bug.
  function buildStatusLine(response, kind) {
    const wrap = document.createElement('div');
    const count = document.createElement('div');
    const label = kind === 'assembly' ? 'Assembly' : 'Part';
    // The total is only known after the tiles are painted; until then show just
    // the server time.
    const timing =
      lastTotalMs > 0
        ? `${formatDuration(lastServerMs)} (server) / ${formatDuration(lastTotalMs)} (total)`
        : `${formatDuration(lastServerMs)} (server)`;
    count.textContent = `${label} search: ${timing} \u2014 ${response.count} hit(s).`;
    wrap.appendChild(count);

    const filter = response.tag_filter;
    if (filter) {
      const note = document.createElement('div');
      note.className = 'tag-note';
      const inIndex = filter.matching_parts_in_index;
      if (inIndex > response.count) {
        note.textContent =
          `${inIndex} part(s) in the index carry the tag filter, but only ` +
          `${response.count} ranked into the results.`;
      } else {
        note.textContent = `Tag filter matched ${inIndex} part(s) in the index.`;
      }
      wrap.appendChild(note);
    }
    return wrap;
  }

  // Format a millisecond duration compactly: seconds (one decimal) at or above
  // one second, whole milliseconds below.
  function formatDuration(ms) {
    return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
  }

  // The server's own time for the most recent search request, taken from the
  // browser's Resource Timing. responseStart - requestStart is the span from
  // the request leaving the browser to the first response byte: server work
  // plus network, with no client-side queueing. That is why it matches a curl
  // call, while the wall-clock fetch time can be far larger when the browser is
  // busy (e.g. thumbnails competing for the few HTTP/1.1 connections). We match
  // on the endpoint path ('/search?' or '/search-assembly?') and take the last
  // entry recorded AFTER this search started. The startedAt filter is what
  // makes repeat searches correct: the Resource Timing buffer holds only ~250
  // entries and silently drops new ones once full, which thumbnails and map
  // resources manage quickly, so without it the last matching entry stays the
  // FIRST search forever -- we once showed '65.5 s (server) / 627 ms (total)',
  // a server time longer than the total. Falling back to the measured fetch
  // time when nothing matches is honest; repeating a stale number is not.
  function serverTimeMs(pathFragment, fallbackMs, startedAt) {
    const entries = performance
      .getEntriesByType('resource')
      .filter(
        (e) => e.name.includes(pathFragment) && e.responseStart > 0 && e.startTime >= startedAt,
      );
    if (entries.length === 0) return fallbackMs;
    const last = entries[entries.length - 1];
    return last.responseStart - last.requestStart;
  }

  function buildResultTile(hit, kind) {
    const el = document.createElement('div');
    el.className = 'tile';
    el.style.width = `${TILE_WIDTH}px`;

    const meta = hit.metadata || {};
    const filename = meta.filename || `${hit.id.slice(0, 12)}...`;

    // The part passed to the caller for viewing/properties. Search hits carry
    // their metadata inline, so no extra lookup is needed.
    const part = {
      id: hit.id,
      filename,
      kind: meta.kind,
      bodies: meta.bodies,
      registered_at: meta.registered_at,
      tags: meta.tags || [],
    };

    // Assembly hits carry diagnostics behind the composite score; show them so
    // it is clear what assembly search did (e.g. "16/22 parts matched").
    const lines = [];
    if (kind === 'assembly') {
      lines.push(`geom ${hit.geom_score.toFixed(4)}`);
      lines.push(`${hit.matched_parts}/${hit.query_parts} parts matched`);
      lines.push(`coverage ${Math.round(hit.coverage * 100)}%`);
    }

    function markSelected() {
      selectedId = hit.id;
      for (const t of grid.querySelectorAll('.tile.selected')) {
        t.classList.remove('selected');
      }
      el.classList.add('selected');
    }

    renderTile(el, {
      thumbUrl: api.thumbnailUrl(state.indexName, hit.id),
      filename,
      tags: meta.tags || [],
      selected: hit.id === selectedId,
      badge: hit.score.toFixed(4),
      lines,
      onClick: () => {
        markSelected();
        onResultSelect(part);
      },
      onDblClick: () => {
        markSelected();
        onResultActivate(part);
      },
    });
    return el;
  }

  // Patch a result's tags in place after an edit elsewhere and repaint it, so
  // the results do not show stale tags. A no-op if that part is not a hit.
  function updateResultTags(id, tags) {
    if (!lastResponse) return;
    let touched = false;
    for (const hit of lastResponse.hits) {
      if (hit.id === id) {
        hit.metadata = { ...(hit.metadata || {}), tags };
        touched = true;
      }
    }
    if (touched) renderResults(lastResponse, lastKind);
  }

  // The neighbourhood an Estimate should run over: the query and top_k of the
  // last search, plus the ids of the hits currently shown (after the similarity
  // slider and the results tag filter). Returns null when no search has run or
  // nothing is shown, which is exactly when Estimate must stay disabled. The
  // self hit is left in -- the server drops it -- so the caller need not know
  // which tile is the query itself.
  function getEstimateContext() {
    if (!lastResponse || !lastQueryFileId || shownHits.length === 0) return null;
    return {
      queryFileId: lastQueryFileId,
      kind: lastKind,
      topK: SEARCH_TOP_K,
      neighborIds: shownHits.map((hit) => hit.id),
    };
  }

  // The hit set the search-results map draws: the query, and EVERY hit with
  // its score -- not shownHits. The map greys the hits below the similarity
  // threshold rather than dropping them, so it needs the hits the threshold
  // has already removed from the grid as well. Returns null when no search
  // has run.
  function getSearchMapContext() {
    if (!lastResponse || !lastQueryFileId) return null;
    return {
      queryFileId: lastQueryFileId,
      kind: lastKind,
      // The live slider value. The map greys the hits below it and spreads the
      // heat gradient over the rest, so it must come from the one control the
      // Results grid is already filtered by -- not a second copy of it.
      threshold: Number(thresholdInput.value),
      hits: lastResponse.hits.map((hit) => ({ id: hit.id, score: hit.score })),
    };
  }

  // Clear all search state and the grid. Used when the index changes so a stale
  // result set from another index cannot linger or feed an Estimate.
  function reset() {
    lastResponse = null;
    lastQueryFileId = null;
    shownHits = [];
    resultsMembership = null;
    selectedId = null;
    status.textContent = '';
    grid.innerHTML = '';
    if (onResultsChange) onResultsChange();
    if (onSearchHits) onSearchHits();
  }

  return { refilter, updateResultTags, getEstimateContext, getSearchMapContext, reset };
}
