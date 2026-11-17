// Compare mode: overlay two CAD parts and judge their shape similarity.
//
// This is demo 1's central claim -- "two parts are enough, no project needed".
// POST /similarity/compare embeds the files on the fly and needs no FAISS index
// (see api.compareFiles). Two parts are shown in one scene as A (the model
// already open, the "before"/base) and B (the "after"/added part); the HOOPS
// Web Viewer shape comparison (viewer.showComparison) paints the unique and
// shared regions in three colours. The number alone is weak; the coloured
// difference is what the desktop sandbox showed, so the Web client shows it too.
//
// A part descriptor is `{ fileId, filename, indexed }`: `indexed` parts already
// have an SCS on the index (api.scsUrl); an uploaded file gets one on demand
// (api.exportScs). The similarity number is best-effort -- it needs both source
// CAD files in the upload store, which an indexed part may not have, so the
// visual comparison still runs when the number cannot be computed.

import * as api from './api.js';
import { withLoading, showError, clearError } from './ui.js';
import { showComparison, showPart, COMPARE_LEGEND } from './viewer.js';

// Wire the Compare feature. `onStart` is called just before a comparison takes
// over the viewer, so the host can clear the current query and leave map mode
// (the comparison scene is neither a part nor the map). `onClose` is called
// after the user ends a comparison and the base part has been restored alone,
// so the host can update any UI that tracks whether a comparison is on screen.
// `getIndexName` supplies the index an indexed part's SCS lives on. Returns
// { startWith(a, b), hide(), close() }.
export function initCompare({ onStart, onClose, getIndexName } = {}) {
  const panel = document.getElementById('compare-panel');
  const scoreEl = document.getElementById('compare-score');
  const legendEl = document.getElementById('compare-legend');
  const toggle = document.getElementById('compare-toggle');
  const alignToggle = document.getElementById('compare-align');
  const propsBody = document.querySelector('#compare-props tbody');
  const headA = document.getElementById('compare-props-a');
  const headB = document.getElementById('compare-props-b');
  const closeX = document.getElementById('compare-close-x');
  const closeBtn = document.getElementById('compare-close');

  // The controller returned by showComparison() for the scene on screen, used
  // to turn the coloured comparison on/off from the switch.
  let controller = null;

  // The base part (A) of the comparison on screen and its resolved SCS url, so
  // ending the comparison can drop the added part (B) and reload A alone.
  let baseScs = null;

  toggle.addEventListener('change', () => {
    if (controller) controller.setEnabled(toggle.checked);
  });

  alignToggle.addEventListener('change', () => {
    if (controller) controller.setAligned(alignToggle.checked);
  });

  // -- Properties table ------------------------------------------------------
  //
  // One row per figure, one column per part. The cells are filled as each
  // figure arrives (see fillProps), so a slow or missing one never holds back
  // the rest. A figure that cannot be read shows a dash and says why in its
  // tooltip; the table itself is always drawn.
  const REL_TOLERANCE = 1e-6;
  const CENTRE_TOLERANCE = 1e-6;
  const UNAVAILABLE = 'Not available in this SCS';

  const PROP_ROWS = [
    { key: 'faces', label: 'Faces', format: formatCount, same: (a, b) => a === b },
    { key: 'edges', label: 'Edges', format: formatCount, same: (a, b) => a === b },
    { key: 'area', label: 'Surface area', format: formatQuantity, same: sameQuantity },
    { key: 'volume', label: 'Volume', format: formatQuantity, same: sameQuantity },
    { key: 'centre', label: 'Bounding-box centre', format: formatCentre, same: sameCentre },
  ];

  function formatCount(n) {
    return String(n);
  }

  // About seven significant digits, with the unit when the SCS gave one.
  function formatQuantity(q) {
    const digits = String(Number(q.value.toPrecision(7)));
    return q.unit ? `${digits} ${q.unit}` : digits;
  }

  function formatCentre(c) {
    return `${c.x.toFixed(3)}, ${c.y.toFixed(3)}, ${c.z.toFixed(3)}`;
  }

  // Relative difference under 1e-6 counts as equal; two zeros are equal.
  function sameQuantity(a, b) {
    const scale = Math.max(Math.abs(a.value), Math.abs(b.value));
    return scale === 0 || Math.abs(a.value - b.value) / scale < REL_TOLERANCE;
  }

  function sameCentre(a, b) {
    return (
      Math.abs(a.x - b.x) < CENTRE_TOLERANCE
      && Math.abs(a.y - b.y) < CENTRE_TOLERANCE
      && Math.abs(a.z - b.z) < CENTRE_TOLERANCE
    );
  }

  // The scene whose figures the table is waiting for. A figure that arrives
  // after the comparison was replaced or closed must not be written into the
  // next one's table.
  let propsToken = 0;

  function renderPropsSkeleton(nameA, nameB) {
    headA.textContent = nameA;
    headA.title = nameA;
    headB.textContent = nameB;
    headB.title = nameB;
    propsBody.innerHTML = '';
    const rows = {};
    for (const spec of PROP_ROWS) {
      const tr = document.createElement('tr');
      const label = document.createElement('td');
      label.textContent = spec.label;
      const cellA = document.createElement('td');
      const cellB = document.createElement('td');
      cellA.textContent = '\u2026';
      cellB.textContent = '\u2026';
      tr.append(label, cellA, cellB);
      propsBody.appendChild(tr);
      rows[spec.key] = { spec, tr, cells: [cellA, cellB], values: [undefined, undefined] };
    }
    return rows;
  }

  function setProp(row, index, value) {
    const cell = row.cells[index];
    row.values[index] = value;
    if (value === null || value === undefined) {
      cell.textContent = '\u2014';
      cell.title = UNAVAILABLE;
    } else {
      cell.textContent = row.spec.format(value);
      cell.title = '';
    }
    const [a, b] = row.values;
    const both = a !== null && a !== undefined && b !== null && b !== undefined;
    row.tr.classList.toggle('compare-props-differs', both && !row.spec.same(a, b));
  }

  // Read both parts' figures and put each in its cell as it resolves. Not
  // awaited by the caller: the parts are already on screen and the comparison
  // already running by the time this starts.
  function fillProps(ctrl, nameA, nameB) {
    const token = ++propsToken;
    const rows = renderPropsSkeleton(nameA, nameB);
    ['a', 'b'].forEach((which, index) => {
      const figures = ctrl.readProperties(which);
      if (!figures) return;
      setProp(rows.centre, index, figures.centre);
      for (const key of ['faces', 'edges', 'area', 'volume']) {
        figures[key]
          .catch(() => null)
          .then((value) => {
            if (token === propsToken) setProp(rows[key], index, value);
          });
      }
    });
  }

  if (closeX) closeX.addEventListener('click', () => close());
  if (closeBtn) closeBtn.addEventListener('click', () => close());

  // Draw one legend row: an optional visibility checkbox, a colour swatch and a
  // label. `part` is 'a' or 'b' for the two source parts (whose visibility the
  // checkbox toggles), or null for the shared "In both" region, which is not a
  // node of its own and so cannot be shown/hidden.
  function legendRow(rgb, text, part) {
    const row = document.createElement('div');
    row.className = 'compare-legend-row';
    if (part) {
      const check = document.createElement('input');
      check.type = 'checkbox';
      check.checked = true;
      check.className = 'compare-legend-check';
      check.title = part === 'a' ? 'Show the base part' : 'Show the added part';
      check.addEventListener('change', () => {
        if (controller) controller.setPartVisible(part, check.checked);
      });
      row.appendChild(check);
    } else {
      const spacer = document.createElement('span');
      spacer.className = 'compare-legend-check';
      spacer.setAttribute('aria-hidden', 'true');
      row.appendChild(spacer);
    }
    const swatch = document.createElement('span');
    swatch.className = 'compare-legend-swatch';
    const [r, g, b] = rgb;
    swatch.style.background = `rgb(${r}, ${g}, ${b})`;
    const label = document.createElement('span');
    label.className = 'compare-legend-label';
    label.textContent = text;
    row.append(swatch, label);
    return row;
  }

  // Fill the panel: the big similarity number and the file-to-colour legend.
  // `nameA` (the model already open, the base) maps to only1Color and `nameB`
  // (the added part) to only2Color, matching the load order in showComparison
  // (file A is the first subtree). The base/added rows carry a checkbox that
  // shows or hides that part so the two can be inspected alone or overlaid.
  function renderPanel(score, nameA, nameB) {
    scoreEl.textContent = typeof score === 'number' ? score.toFixed(4) : '\u2014';
    legendEl.innerHTML = '';
    legendEl.append(
      legendRow(COMPARE_LEGEND.only1, `Only in ${nameA} (base)`, 'a'),
      legendRow(COMPARE_LEGEND.only2, `Only in ${nameB} (added)`, 'b'),
      legendRow(COMPARE_LEGEND.same, 'In both', null),
    );
  }

  // Resolve a part descriptor to the URL of its SCS stream cache. An indexed
  // part already has one on the index; an uploaded file gets one authored now.
  async function scsFor(desc) {
    if (desc.indexed) return api.scsUrl(getIndexName(), desc.fileId);
    const exported = await api.exportScs(desc.fileId);
    return exported.scs_url;
  }

  // Run a comparison of part `a` (base, the model already open) against part `b`
  // (added). Both are `{ fileId, filename, indexed }`. The visual comparison is
  // the point; the similarity number is best-effort (see the file header).
  async function startWith(a, b) {
    clearError();
    if (!a || !b) return;
    if (a.fileId === b.fileId) {
      showError(new Error('Pick a different part to compare against the current model.'));
      return;
    }
    if (onStart) onStart();
    try {
      // Author/resolve both SCS caches, then load them into one scene.
      const [scsA, scsB] = await withLoading(() => Promise.all([scsFor(a), scsFor(b)]));

      // Similarity number, best-effort: this needs both source CAD files in the
      // upload store. An indexed part may not have its source there, in which
      // case the endpoint fails; the visual comparison still stands, so fall
      // back to no number rather than aborting.
      let score = null;
      try {
        const result = await withLoading(() => api.compareFiles([a.fileId, b.fileId]));
        score = result.pairs && result.pairs[0] ? result.pairs[0].score : null;
      } catch (_scoreErr) {
        score = null;
      }

      controller = await showComparison({ scsA, scsB });
      baseScs = scsA;
      renderPanel(score, a.filename, b.filename);
      toggle.checked = true;
      alignToggle.checked = true;
      panel.hidden = false;
      fillProps(controller, a.filename, b.filename);
    } catch (err) {
      showError(err);
      hide();
    }
  }

  // Hide the compare panel without touching the viewer. Used when the host is
  // about to load something else into the viewer anyway (opening a part, a
  // file, or the map), so reloading the base here would be wasted work.
  function hide() {
    panel.hidden = true;
    controller = null;
    baseScs = null;
    propsToken += 1;
  }

  // End the comparison the user is looking at: drop the added part (B) and
  // reload the base (A) alone -- a fresh single-model viewer, which brings back
  // the model tree the comparison scene does not show -- then hide the panel.
  // A no-op when no comparison is on screen. The base stays the current query
  // (the host keeps it), so Find similar and a repeat Compare still work.
  function close() {
    if (panel.hidden) return;
    const scs = baseScs;
    hide();
    if (scs) showPart(scs);
    if (onClose) onClose();
  }

  return { startWith, hide, close };
}
