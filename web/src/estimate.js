// Estimate controller: predicts the query part's production records (material,
// process, cost, lead time, ...) from the records of its nearest neighbours --
// the HOOPS AI Context Layer. It runs on the same query as a similarity search
// (state.query, the model open in the 3D viewer) so the neighbours here are the
// same neighbours a search would show.
//
// The panel is read top-to-bottom as a piece of reasoning:
//   1. the neighbour parts whose records fed the prediction (with thumbnails),
//      titled with how thick the evidence is (e.g. "6 of 8"),
//   2. the predicted fields with a status chip each.
// When no neighbour has a production record, the panel says so plainly rather
// than drawing an empty table, so a data problem is obvious during a demo.

import * as api from './api.js';
import { state } from './state.js';
import { withLoading, clearError, showError } from './ui.js';

// Lifecycle values that should carry a visible warning on a neighbour row: a
// neighbour that is discontinued or on quality hold is weak evidence to copy a
// record from, and the demo narration calls this out explicitly.
const FLAGGED_LIFECYCLE = new Set(['Discontinued', 'Quality hold']);

// Wire up the Estimate controls once.
//
//   getEstimateContext()     returns the neighbourhood to estimate over --
//                            { queryFileId, kind, topK, neighborIds } from the
//                            last similarity search sliced to the shown tiles,
//                            or null when no search has run. Estimate pins its
//                            neighbours to exactly those shown parts.
//   onEvidenceActivate(part) called when a neighbour row is double-clicked (open in 3D)
//   onEstimateDone()         called after the panel renders (e.g. to show the tab)
//
// Returns { run() } so the caller can trigger an estimate from elsewhere.
export function initEstimate({
  getEstimateContext,
  onEvidenceActivate,
  onEstimateDone,
}) {
  const runButton = document.getElementById('estimate-run');
  const neighborsBox = document.getElementById('estimate-neighbors');
  const predictionsBox = document.getElementById('estimate-predictions');

  // The context summary (keys + units) for the current index, fetched lazily
  // and cached so switching indexes is the only thing that re-fetches it. Reset
  // when the index changes (tracked by name).
  let contextCacheName = null;
  let contextCache = null;

  runButton.addEventListener('click', () => {
    run().catch(() => {});
  });

  // Fetch (once per index) which fields the index holds and their units.
  async function loadContext(name) {
    if (contextCacheName === name && contextCache) return contextCache;
    const summary = await api.getIndexContext(name);
    contextCacheName = name;
    contextCache = summary;
    return summary;
  }

  async function run() {
    clearError();
    // Estimate runs over the parts narrowed in the Results tab, so it needs a
    // similarity search first. getEstimateContext() is null until then (and the
    // Estimate button is disabled to match), but guard here too in case run()
    // is triggered programmatically.
    const estimateContext = getEstimateContext ? getEstimateContext() : null;
    if (!estimateContext) {
      showError(
        new Error('Run a similarity search first; Estimate uses the parts shown in Results.'),
      );
      return;
    }
    const name = state.indexName;

    let context;
    try {
      context = await withLoading(() => loadContext(name));
    } catch (err) {
      showError(err);
      return;
    }
    const keys = (context.keys || []).map((k) => k.key);
    const units = context.units || {};
    if (keys.length === 0) {
      showEmptyContext();
      if (onEstimateDone) onEstimateDone();
      return;
    }

    let response;
    try {
      response = await withLoading(() =>
        api.predictContext(name, estimateContext.queryFileId, {
          keys,
          // Pin the neighbours to the shown Results tiles; pass the search's
          // top_k and kind so the server rebuilds the same pool that contains
          // them.
          neighborIds: estimateContext.neighborIds,
          topK: estimateContext.topK,
          kind: estimateContext.kind,
        }),
      );
    } catch (err) {
      showEstimateError(err);
      if (onEstimateDone) onEstimateDone();
      return;
    }

    renderEstimate(response, units);
    if (onEstimateDone) onEstimateDone();
  }

  function renderEstimate(response, units) {
    renderNeighborTable(response, units);
    renderPredictions(response, units);
  }

  // The predicted fields. When no neighbour carried a record there is nothing
  // to predict from, so say that plainly instead of drawing an empty table.
  function renderPredictions(response, units) {
    predictionsBox.innerHTML = '';
    const title = document.createElement('div');
    title.className = 'estimate-section-title';
    title.textContent = 'Context predictions';
    predictionsBox.appendChild(title);
    if (response.hits_with_context === 0) {
      const note = document.createElement('div');
      note.className = 'estimate-empty';
      note.textContent =
        'None of the narrowed results carried a production record, so nothing ' +
        'can be predicted.';
      predictionsBox.appendChild(note);
      return;
    }

    const table = document.createElement('table');
    table.className = 'estimate-table';
    const head = document.createElement('tr');
    for (const label of ['Field', 'Predicted', 'Confidence', 'Status']) {
      const th = document.createElement('th');
      th.textContent = label;
      head.appendChild(th);
    }
    table.appendChild(head);

    for (const key of response.keys) {
      const pred = response.predictions[key];
      if (!pred) continue;
      table.appendChild(buildPredictionRow(key, pred, units));
      const details = buildDetailsRow(key, pred, response.hits);
      if (details) table.appendChild(details);
    }
    predictionsBox.appendChild(table);
  }

  function buildPredictionRow(key, pred, units) {
    const row = document.createElement('tr');
    row.className = 'estimate-row';

    const fieldCell = document.createElement('td');
    fieldCell.textContent = key;
    row.appendChild(fieldCell);

    const valueCell = document.createElement('td');
    valueCell.appendChild(renderValue(key, pred.value, units));
    row.appendChild(valueCell);

    const confCell = document.createElement('td');
    confCell.textContent =
      typeof pred.confidence === 'number' ? `${Math.round(pred.confidence * 100)}%` : '\u2014';
    row.appendChild(confCell);

    const statusCell = document.createElement('td');
    statusCell.appendChild(statusChip(pred.status));
    row.appendChild(statusCell);

    // Clicking the row toggles the details row that follows it.
    row.addEventListener('click', () => {
      const next = row.nextElementSibling;
      if (next && next.classList.contains('estimate-details')) {
        next.hidden = !next.hidden;
        row.classList.toggle('expanded', !next.hidden);
      }
    });
    return row;
  }

  // Render a predicted value. ProcessRoute is the demo showpiece: split it on
  // "->" and draw it as a numbered list of steps; everything else is text, with
  // a unit appended for numeric fields.
  function renderValue(key, value, units) {
    if (value == null || value === '') {
      const span = document.createElement('span');
      span.className = 'estimate-muted';
      span.textContent = '\u2014';
      return span;
    }
    if (key === 'ProcessRoute' && typeof value === 'string' && value.includes('->')) {
      const ol = document.createElement('ol');
      ol.className = 'estimate-route';
      for (const step of value.split('->')) {
        const li = document.createElement('li');
        li.textContent = step.trim();
        ol.appendChild(li);
      }
      return ol;
    }
    const span = document.createElement('span');
    const unit = units[key];
    const text =
      typeof value === 'number' ? formatFieldNumber(key, value) : String(value);
    span.textContent = unit ? `${text} ${unit}` : text;
    return span;
  }

  // The expandable details for a prediction: the runners-up, why the rule
  // decided as it did, how many records it saw, and any classification values
  // that were forwarded into the numeric rules (injected_context).
  function buildDetailsRow(key, pred, hits) {
    // Numeric fields (lead time, lot size, unit cost) get a distribution
    // histogram built from the neighbours' own values instead of the long
    // "Alternatives"/"Candidates" text line, which was hard to read. The graph
    // conveys the spread and marks where the prediction sits; the text reasons
    // still follow it.
    const numericValues = numericFieldValues(hits, key);
    const isNumeric = numericValues.length >= 2;

    const parts = [];
    if (!isNumeric) {
      if (Array.isArray(pred.alternatives) && pred.alternatives.length > 0) {
        parts.push(`Alternatives: ${formatAlternatives(pred.alternatives, key)}`);
      } else if (pred.value == null || pred.value === '') {
        // The SDK reports alternatives only as runners-up to a chosen value;
        // when it cannot settle on one (the neighbours' values split below the
        // evidence threshold) it returns none and the row reads "insufficient
        // evidence". Rebuild the split from the neighbours' own values so the
        // row still shows what the competing candidates were -- the point is
        // that a single value could not be picked, not that there was no data.
        const dist = valueDistribution(hits, key);
        if (dist.length > 0) {
          parts.push(`Candidates: ${formatDistribution(dist)}`);
        }
      }
    }
    if (Array.isArray(pred.reasons) && pred.reasons.length > 0) {
      parts.push(`Reasons: ${pred.reasons.join('; ')}`);
    }
    if (typeof pred.evidence_count === 'number') {
      parts.push(`Evidence: ${pred.evidence_count} record(s)`);
    }
    if (parts.length === 0 && !isNumeric) return null;

    const row = document.createElement('tr');
    row.className = 'estimate-details';
    row.hidden = true;
    const cell = document.createElement('td');
    cell.colSpan = 4;
    if (isNumeric) {
      cell.appendChild(buildHistogram(numericValues, key, pred));
    }
    for (const text of parts) {
      const div = document.createElement('div');
      div.className = 'estimate-detail';
      div.textContent = text;
      cell.appendChild(div);
    }
    row.appendChild(cell);
    return row;
  }

  // The neighbours' numeric values for one field, in record order. Empty for a
  // categorical field, so its caller falls back to the text distribution.
  function numericFieldValues(hits, key) {
    const values = [];
    for (const hit of hits || []) {
      const v = hit.values ? hit.values[key] : undefined;
      if (typeof v === 'number' && Number.isFinite(v)) values.push(v);
    }
    return values;
  }

  // A compact histogram of the neighbours' values for a numeric field. A
  // labelled vertical marker sits at the exact predicted value (so it is not
  // mistaken for the median), and the axis shows the min/max in the field's own
  // precision. Fewer, wider bars read better for the handful of records a
  // prediction usually rests on.
  function buildHistogram(values, key, pred) {
    const wrap = document.createElement('div');
    wrap.className = 'estimate-hist';

    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    const binCount = Math.min(6, Math.max(4, Math.round(values.length / 1.5)));
    const counts = new Array(binCount).fill(0);
    for (const v of values) {
      const idx = Math.floor(((v - min) / span) * binCount);
      counts[Math.max(0, Math.min(binCount - 1, idx))] += 1;
    }
    const maxCount = Math.max(...counts, 1);

    const plot = document.createElement('div');
    plot.className = 'estimate-hist-plot';

    const bars = document.createElement('div');
    bars.className = 'estimate-hist-bars';
    for (const count of counts) {
      const bar = document.createElement('div');
      bar.className = 'estimate-hist-bar';
      bar.style.height = `${(count / maxCount) * 100}%`;
      bar.title = `${count} record(s)`;
      bars.appendChild(bar);
    }
    plot.appendChild(bars);

    if (typeof pred.value === 'number') {
      const pct = Math.max(0, Math.min(100, ((pred.value - min) / span) * 100));
      const marker = document.createElement('div');
      marker.className = 'estimate-hist-marker';
      marker.style.left = `${pct}%`;
      const label = document.createElement('span');
      label.className = 'estimate-hist-marker-label';
      label.textContent = 'predicted';
      marker.appendChild(label);
      plot.appendChild(marker);
    }
    wrap.appendChild(plot);

    const axis = document.createElement('div');
    axis.className = 'estimate-hist-axis';
    const lo = document.createElement('span');
    lo.textContent = formatFieldNumber(key, min);
    const hi = document.createElement('span');
    hi.textContent = formatFieldNumber(key, max);
    axis.appendChild(lo);
    axis.appendChild(hi);
    wrap.appendChild(axis);

    return wrap;
  }

  function statusChip(status) {
    const chip = document.createElement('span');
    chip.className = 'estimate-chip';
    const value = status || 'insufficient_evidence';
    if (value.includes('ready')) chip.classList.add('chip-ready');
    else if (value.includes('review')) chip.classList.add('chip-review');
    else chip.classList.add('chip-insufficient');
    chip.textContent = value.replace(/_/g, ' ');
    return chip;
  }

  // The neighbours whose records fed the prediction, shown with a thumbnail and
  // their fields side by side (like the tutorial's hits_table). Built from
  // response.hits, which the server limits to the record-bearing neighbours --
  // the full ranked list, including neighbours without records, stays in the
  // Results tab. The title carries the evidence depth so a thin basis reads as
  // thin. Double-clicking a row opens that neighbour in the 3D viewer.
  function renderNeighborTable(response, units) {
    neighborsBox.innerHTML = '';
    const hits = response.hits || [];

    const title = document.createElement('div');
    title.className = 'estimate-section-title';
    title.textContent =
      `Top neighbors \u2014 ${response.hits_with_context} of ${response.hits_considered} ` +
      `narrowed results had production records`;
    neighborsBox.appendChild(title);
    if (hits.length === 0) return;

    // ProcessRoute is a long, multi-step string already drawn in the
    // predictions table; a per-neighbour column of it would be unreadable, so
    // it is left out of this matrix.
    const cols = (response.keys || []).filter((key) => key !== 'ProcessRoute');

    const scroll = document.createElement('div');
    scroll.className = 'estimate-neighbors-scroll';
    const table = document.createElement('table');
    table.className = 'estimate-table estimate-neighbors-table';

    const head = document.createElement('tr');
    const headers = ['#', 'Thumbnail', 'File', 'Score'].concat(
      cols.map((key) => (units[key] ? `${key} (${units[key]})` : key)),
    );
    for (const label of headers) {
      const th = document.createElement('th');
      th.textContent = label;
      head.appendChild(th);
    }
    table.appendChild(head);

    hits.forEach((hit, i) => {
      const row = document.createElement('tr');
      row.className = 'estimate-neighbor-row';
      const filename = hit.filename || `${String(hit.id).slice(0, 12)}...`;
      const values = hit.values || {};
      // Flag a weak-evidence neighbour (discontinued / quality hold) so copying
      // its record is visibly questionable, as on the old evidence tiles.
      if (FLAGGED_LIFECYCLE.has(values.LifecycleStatus)) {
        row.classList.add('estimate-neighbor-flagged');
      }

      const rankCell = document.createElement('td');
      rankCell.textContent = String(i + 1);
      row.appendChild(rankCell);

      const thumbCell = document.createElement('td');
      const img = document.createElement('img');
      img.className = 'estimate-neighbor-thumb';
      img.loading = 'lazy';
      img.alt = filename;
      img.src = hit.thumbnail_url || api.thumbnailUrl(state.indexName, hit.id);
      thumbCell.appendChild(img);
      row.appendChild(thumbCell);

      const fileCell = document.createElement('td');
      fileCell.textContent = filename;
      row.appendChild(fileCell);

      const scoreCell = document.createElement('td');
      scoreCell.textContent = typeof hit.score === 'number' ? hit.score.toFixed(4) : '\u2014';
      row.appendChild(scoreCell);

      for (const key of cols) {
        const td = document.createElement('td');
        const value = values[key];
        if (value == null || value === '') {
          td.textContent = '\u2014';
          td.classList.add('estimate-muted');
        } else {
          td.textContent = typeof value === 'number' ? formatFieldNumber(key, value) : String(value);
        }
        row.appendChild(td);
      }

      // The properties panel stays closed on the narrow Estimate tab (a single
      // click does nothing here), but a double-click opens the neighbour in the
      // viewer, matching the results/gallery grids.
      row.title = 'Double-click to open in the viewer';
      row.addEventListener('dblclick', () => {
        if (onEvidenceActivate) onEvidenceActivate({ id: hit.id, filename });
      });
      table.appendChild(row);
    });

    scroll.appendChild(table);
    neighborsBox.appendChild(scroll);
  }

  // The index holds no context fields at all: predicting is not possible, so
  // clear the panel and say so, rather than posting an empty prediction.
  function showEmptyContext() {
    neighborsBox.innerHTML = '';
    predictionsBox.innerHTML = '';
    const note = document.createElement('div');
    note.className = 'estimate-empty';
    note.textContent =
      'This index has no production records to estimate from. Seed context ' +
      'data for the index first.';
    predictionsBox.appendChild(note);
  }

  function showEstimateError(err) {
    neighborsBox.innerHTML = '';
    predictionsBox.innerHTML = '';
    const box = document.createElement('div');
    box.className = 'estimate-empty';
    const message =
      err && err.status ? `Error ${err.status}: ${err.detail}` : err && err.message ? err.message : String(err);
    box.textContent = `Estimate failed. ${message}`;
    predictionsBox.appendChild(box);
  }

  // Per-field decimal precision for numeric predictions, so quantities read the
  // way a planner would write them: lead time to a tenth of a day, lot size as
  // a whole piece count, unit cost to the cent. Fields not listed keep the
  // generic formatNumber rule.
  const FIELD_DECIMALS = { LeadTimeDays: 1, LotSize: 0, UnitCost: 2 };

  function formatFieldNumber(key, value) {
    if (typeof value !== 'number') return String(value);
    if (Object.prototype.hasOwnProperty.call(FIELD_DECIMALS, key)) {
      return value.toFixed(FIELD_DECIMALS[key]);
    }
    return formatNumber(value);
  }

  function formatNumber(value) {
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }

  // Render the runners-up compactly. Each alternative is either a bare value or
  // an object carrying a value and a weight/confidence; show the value and, when
  // present, its share. A numeric field's runner-up is a predicted interval
  // ({lower, upper, std}) rather than a single value, so it is shown as a
  // rounded range instead of raw JSON. All numbers use the field's precision.
  function formatAlternatives(alternatives, key) {
    return alternatives
      .slice(0, 5)
      .map((alt) => {
        if (alt && typeof alt === 'object') {
          if (alt.value == null && (alt.lower != null || alt.upper != null || alt.std != null)) {
            return formatRange(key, alt);
          }
          const value =
            alt.value != null ? formatFieldNumber(key, alt.value) : JSON.stringify(alt);
          const weight = alt.confidence != null ? alt.confidence : alt.weight;
          return typeof weight === 'number' ? `${value} (${Math.round(weight * 100)}%)` : `${value}`;
        }
        return typeof alt === 'number' ? formatFieldNumber(key, alt) : String(alt);
      })
      .join(', ');
  }

  // A predicted numeric interval, rounded to the field's precision: the
  // lower-upper bound and its standard deviation.
  function formatRange(key, range) {
    const segments = [];
    if (range.lower != null && range.upper != null) {
      segments.push(`${formatFieldNumber(key, range.lower)}\u2013${formatFieldNumber(key, range.upper)}`);
    } else if (range.lower != null) {
      segments.push(`\u2265 ${formatFieldNumber(key, range.lower)}`);
    } else if (range.upper != null) {
      segments.push(`\u2264 ${formatFieldNumber(key, range.upper)}`);
    }
    if (range.std != null) segments.push(`std ${formatFieldNumber(key, range.std)}`);
    return segments.join(', ');
  }

  // Tally the neighbours' own values for one field, most common first, so a
  // prediction the SDK could not settle can still show what the competing
  // values were. Counts are over the neighbours that actually carry the field.
  function valueDistribution(hits, key) {
    const counts = new Map();
    let total = 0;
    for (const hit of hits || []) {
      const v = hit.values ? hit.values[key] : undefined;
      if (v == null || v === '') continue;
      const label = typeof v === 'number' ? formatNumber(v) : String(v);
      counts.set(label, (counts.get(label) || 0) + 1);
      total += 1;
    }
    return [...counts.entries()]
      .map(([value, count]) => ({ value, count, total }))
      .sort((a, b) => b.count - a.count);
  }

  // "Carburizing 33% (1), Nitriding 33% (1), ...": the share and record count of
  // each candidate value. Percentages are raw record counts (not the SDK's
  // score-weighted share), enough to show why no single value could be picked.
  function formatDistribution(dist) {
    return dist
      .slice(0, 6)
      .map(({ value, count, total }) => {
        const pct = total ? Math.round((count / total) * 100) : 0;
        return `${value} ${pct}% (${count})`;
      })
      .join(', ');
  }

  return { run };
}
