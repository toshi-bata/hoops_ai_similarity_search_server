// Gallery: a virtual-scrolling grid of registered parts with thumbnails.
//
// An index can hold tens of thousands of parts, so we never create one <img>
// per part. Instead we window the list (the same idea as Qt's QListView):
//
//   1. A tall inner "spacer" gives the scroll container its full height
//      (totalRows * rowHeight), so the scrollbar reflects every part from the
//      start -- even though almost none of them are in the DOM.
//   2. On scroll (and resize) we compute which rows are visible, then render
//      only those tiles (plus a little overscan) at absolute positions inside
//      the spacer.
//   3. Part data is fetched one page at a time as parts scroll into view and
//      cached, so each part is fetched at most once.
//
// Tiles are a fixed size. When the panel gets wider we add columns; we never
// shrink a tile to fit. Read this file top to bottom: createGallery() sets up
// state and listeners, layout() does the geometry, render() does the DOM.

import { PAGE_SIZE, TILE_WIDTH } from './config.js';
import { renderTile, renderPlaceholder } from './tile.js';

// Fixed spacing, in CSS pixels. These match the padding/gap used below.
const GAP = 12;
const PAD = 12;
// Height of the text area under the thumbnail (filename + optional tags).
const META_HEIGHT = 46;
// Thumbnails are 4:3, so the image area is TILE_WIDTH * 3 / 4 tall.
const TILE_HEIGHT = Math.round((TILE_WIDTH * 3) / 4) + META_HEIGHT;
// One line of the small muted text a tile can carry under its filename. A
// caller whose items have such lines adds this per line, so the rows keep
// their spacing instead of the text running into the tile below.
const LINE_HEIGHT = 15;
// Extra rows rendered above and below the viewport for smoother scrolling.
const OVERSCAN_ROWS = 2;

// Create a virtual-scrolling gallery inside `viewport` (the scroll container).
//
//   total      total number of parts in the index
//   fetchPage  (offset, limit) => Promise<items[]>; called to load a page
//   onSelect   (part) => void; called when a tile is clicked (select part)
//   onActivate (part) => void; called when a tile is double-clicked (open 3D)
//   onSelectionChange (count) => void; called when the multi-select set changes
//              (its size), so the host can show/update a bulk-tag action bar
//
// Returns { setSelected(id), destroy() }.
export function createGallery({
  viewport,
  total,
  fetchPage,
  onSelect,
  onActivate,
  onEmptyClick,
  onSelectionChange,
  // How many small text lines every item carries under its filename, so the
  // row height can make room for them. 0 for the parts listing, which has
  // none; the catalogue passes its own count.
  metaLines = 0,
}) {
  // Parts indexed by their absolute position in the index. Sparse: only loaded
  // positions are set. `pagesRequested` guards against fetching a page twice.
  const parts = new Array(total);
  const pagesRequested = new Set();

  // Bumped whenever the index changes (via a fresh createGallery). A fetch
  // started under an old generation is ignored when it resolves, so a slow
  // response can never fill a gallery that has since moved on.
  let generation = 0;

  // Live tile elements, keyed by part index, so we can add/remove only what
  // changed instead of rebuilding the whole window on every scroll tick.
  const tiles = new Map();
  let selectedId = null;

  // The multi-select set (file ids), independent of the single `selectedId`
  // (which drives the properties preview). Survives virtual-scroll recycling:
  // tiles are re-rendered from this set as they scroll back into view.
  const multiSelected = new Set();

  function notifySelection() {
    if (onSelectionChange) onSelectionChange(multiSelected.size);
  }

  function toggleMulti(id, checked) {
    if (checked) multiSelected.add(id);
    else multiSelected.delete(id);
    // Repaint just this part's tile so its checkbox state sticks.
    for (let i = 0; i < parts.length; i++) {
      if (parts[i] && parts[i].id === id) {
        const el = tiles.get(i);
        if (el) el.dataset.rendered = '';
      }
    }
    render();
    notifySelection();
  }

  // Current layout, recomputed by layout().
  let columns = 1;
  let rowHeight = TILE_HEIGHT + metaLines * LINE_HEIGHT + GAP;
  let totalRows = 0;

  // The scrollable spacer that holds the absolutely-positioned tiles.
  const spacer = document.createElement('div');
  spacer.className = 'gallery-spacer';
  viewport.innerHTML = '';
  viewport.appendChild(spacer);

  function layout() {
    const usable = viewport.clientWidth - 2 * PAD;
    columns = Math.max(1, Math.floor((usable + GAP) / (TILE_WIDTH + GAP)));
    rowHeight = TILE_HEIGHT + metaLines * LINE_HEIGHT + GAP;
    totalRows = Math.ceil(total / columns);
    spacer.style.height = `${PAD + totalRows * rowHeight}px`;
  }

  // The absolute position of the tile for part index `i`.
  function positionOf(i) {
    const row = Math.floor(i / columns);
    const col = i % columns;
    return {
      left: PAD + col * (TILE_WIDTH + GAP),
      top: PAD + row * rowHeight,
    };
  }

  // The inclusive range of part indexes currently in (or near) the viewport.
  function visibleRange() {
    const top = viewport.scrollTop;
    const height = viewport.clientHeight;
    const firstRow = Math.max(0, Math.floor((top - PAD) / rowHeight) - OVERSCAN_ROWS);
    const lastRow = Math.min(
      totalRows - 1,
      Math.floor((top + height - PAD) / rowHeight) + OVERSCAN_ROWS,
    );
    const first = firstRow * columns;
    const last = Math.min(total - 1, (lastRow + 1) * columns - 1);
    return { first, last };
  }

  // Ensure the page containing part index `i` is loaded (or being loaded).
  function ensurePage(i) {
    const pageStart = Math.floor(i / PAGE_SIZE) * PAGE_SIZE;
    if (pagesRequested.has(pageStart)) return;
    pagesRequested.add(pageStart);

    const myGeneration = generation;
    Promise.resolve(fetchPage(pageStart, PAGE_SIZE))
      .then((items) => {
        if (myGeneration !== generation) return; // index changed; drop it
        items.forEach((item, k) => {
          parts[pageStart + k] = item;
        });
        render();
      })
      .catch(() => {
        // Let the page be retried on a later scroll if it failed.
        if (myGeneration === generation) pagesRequested.delete(pageStart);
      });
  }

  function render() {
    const { first, last } = visibleRange();

    // Remove tiles that scrolled out of the window.
    for (const [index, el] of tiles) {
      if (index < first || index > last) {
        el.remove();
        tiles.delete(index);
      }
    }

    // Add/refresh tiles inside the window; request pages for missing data.
    for (let i = first; i <= last; i++) {
      const part = parts[i];
      if (!part) ensurePage(i);

      let el = tiles.get(i);
      if (!el) {
        el = document.createElement('div');
        el.className = 'tile';
        el.style.width = `${TILE_WIDTH}px`;
        tiles.set(i, el);
        spacer.appendChild(el);
      }
      const { left, top } = positionOf(i);
      el.style.left = `${left}px`;
      el.style.top = `${top}px`;
      fillTile(el, part, i);
    }
  }

  // Render one tile's contents. Rebuilt only when its loaded/selected state
  // changes, so scrolling does not re-request already-visible images. The tile
  // drawing itself is shared with the search grid (see tile.js).
  function fillTile(el, part, index) {
    const wanted = part
      ? `loaded:${part.id}:${part.id === selectedId}:${multiSelected.has(part.id)}`
      : `placeholder:${index}`;
    if (el.dataset.rendered === wanted) return;
    el.dataset.rendered = wanted;

    if (!part) {
      renderPlaceholder(el);
      return;
    }

    renderTile(el, {
      thumbUrl: part.thumbnail_url,
      filename: part.filename,
      tags: part.tags || [],
      // Optional extras an item may carry. The parts listing has neither, so
      // these are absent there and the tile looks exactly as it did; the
      // catalogue fills them with its group's size, cohesion and badge. Both
      // are fixed for the life of an item, so they need no place in the
      // repaint key above.
      badge: part.badge == null ? null : part.badge,
      badgeClass: part.badgeClass || null,
      tileClass: part.tileClass || null,
      lines: part.lines || [],
      selected: part.id === selectedId,
      onClick: () => onSelect(part),
      onDblClick: () => onActivate(part),
      selectable: true,
      selectChecked: multiSelected.has(part.id),
      onToggleSelect: (checked) => toggleMulti(part.id, checked),
    });
  }

  const onScroll = () => render();
  const resizeObserver = new ResizeObserver(() => {
    layout();
    render();
  });
  // A click that lands on the scroll container or its spacer background --
  // rather than on a tile -- is an "empty click" that deselects. We test the
  // click target directly instead of `closest('.tile')`, because selecting a
  // tile re-renders it: by the time the event bubbles up, the original target
  // is detached and would look like an empty click.
  const onViewportClick = (e) => {
    if (onEmptyClick && (e.target === viewport || e.target === spacer)) onEmptyClick();
  };
  viewport.addEventListener('scroll', onScroll);
  viewport.addEventListener('click', onViewportClick);
  resizeObserver.observe(viewport);

  layout();
  render();

  return {
    // Highlight the selected part without a full re-render.
    setSelected(id) {
      selectedId = id;
      render();
    },
    // The file ids currently multi-selected (for bulk tag / untag).
    getSelectedIds() {
      return [...multiSelected];
    },
    // How many parts are multi-selected.
    selectedCount() {
      return multiSelected.size;
    },
    // Replace the multi-select set outright. Added for the Catalogue, where
    // "Select all Confident" ticks hundreds of tiles at once and clicking each
    // checkbox is not a realistic alternative. Every tile is repainted rather
    // than only the changed ones: a whole-set replacement usually touches most
    // of them, so working out which is more work than redrawing the window.
    setSelectedIds(ids) {
      multiSelected.clear();
      for (const id of ids || []) multiSelected.add(String(id));
      for (const el of tiles.values()) el.dataset.rendered = '';
      render();
      notifySelection();
    },
    // Clear the multi-select set and repaint the affected tiles.
    clearSelection() {
      if (multiSelected.size === 0) return;
      multiSelected.clear();
      for (const el of tiles.values()) el.dataset.rendered = '';
      render();
      notifySelection();
    },
    // Update the tags of a loaded part in place (after a tag edit) and repaint
    // its tile, so the gallery does not show stale tags. A no-op if the part is
    // not currently loaded in the cache.
    updatePartTags(id, tags) {
      for (let i = 0; i < parts.length; i++) {
        if (parts[i] && parts[i].id === id) {
          parts[i] = { ...parts[i], tags };
          const el = tiles.get(i);
          if (el) el.dataset.rendered = ''; // force fillTile to rebuild it
        }
      }
      render();
    },
    // Patch a loaded item in place and repaint its tile. Added for the
    // Catalogue, which updates a group's tag counts after a write rather than
    // refetching: a refetch mid-task would lose the selection and the scroll
    // position, which is most of what the user has built up. A no-op if the
    // item is not currently in the cache.
    updateItem(id, patch) {
      for (let i = 0; i < parts.length; i++) {
        if (parts[i] && parts[i].id === id) {
          parts[i] = { ...parts[i], ...patch };
          const el = tiles.get(i);
          if (el) el.dataset.rendered = '';
        }
      }
      render();
    },
    // Detach listeners; call before creating a new gallery.
    destroy() {
      generation++;
      viewport.removeEventListener('scroll', onScroll);
      viewport.removeEventListener('click', onViewportClick);
      resizeObserver.disconnect();
      viewport.innerHTML = '';
    },
  };
}
