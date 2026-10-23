// One way to put a tag on a set of parts, used by every place that can.
//
// Tagging grew in six places -- the legend's right-click menu, the Ward tree's
// right-click menu, the single-part editor, the gallery's bulk bar, the search
// results bar and the catalogue tab -- and each had invented its own manner: a
// modal here, a bare text field there, a result line in one place and silence
// in another. The write itself was never the problem (it is one endpoint); the
// difference was only where the name came from and what the user was told
// afterwards. That is what this module owns, so the answer to "how do I tag
// something" is the same wherever the question is asked.
//
// Two shapes, one body:
//
//   inline    a bar that stays on screen (the catalogue tab), for a selection
//             that is built up over many tiles and usually outlives one tag.
//   popover   a small panel next to whatever was clicked, for the places where
//             tagging is the end of a gesture.
//
// The two differ in where they sit and nothing else: the same candidate list,
// the same count, the same validation, the same result line. Moving a control
// is not an excuse to change what it does.
//
// What this module does NOT do is write. The caller passes `onApply`, so the
// existing write paths (shape-map's applyTagToParts, api.assignTag and
// api.unassignTag) stay exactly where they are, with their own follow-up work
// -- refreshing tag lists, marking an Untagged-only map stale -- untouched.

import * as api from './api.js';
import { mergeTagCandidates } from './tag-candidates.js';
import { validateTagName } from './tags.js';

// -- The candidate list -----------------------------------------------------
//
// Candidates are "the tags this index already uses" plus "the application's tag
// vocabulary". The two come from different endpoints with different lifetimes:
//
//   in use      GET /index/{name}/tags        per index, refetched on a switch
//   vocabulary  GET /tag-vocabulary           one list for the whole app, so
//                                             fetching it once per session is
//                                             enough
//
// Order matters and is not cosmetic. The vocabulary is application-wide, so it
// carries names that were used in other indexes and mean nothing here. Listing
// the tags IN USE first, in the server's own descending-count order, keeps the
// names that belong to this index at the top; the vocabulary-only names follow
// alphabetically, available but out of the way.
//
// Both are refetched after a successful write: a name invented in the
// catalogue has to be offered by the tree's popover straight away, which is
// the whole reason the vocabulary exists.

// The two payloads, kept as the server sent them; mergeTagCandidates() is the
// only thing that interprets them.
let vocabularyPayload = null;
let vocabularyLoaded = false;
let candidateIndexName = null;
let inUsePayload = null;
const candidateListeners = new Set();

function notifyCandidates() {
  for (const listener of [...candidateListeners]) {
    try {
      listener();
    } catch {
      // A listener that throws must not stop the others from updating.
    }
  }
}

// Subscribe to candidate changes. Returns an unsubscribe function.
//
// This is the ONE path by which a datalist is rebuilt. Every picker subscribes
// when it is built, and the three things that can change the candidates --
// the vocabulary arriving, an index switch, and a successful write -- all end
// in notifyCandidates(), so inline and popover pickers follow together. A
// picker that filled its options once at construction would show whatever had
// arrived by then, which is how an answered request can leave an empty list.
export function onTagCandidatesChanged(listener) {
  candidateListeners.add(listener);
  return () => candidateListeners.delete(listener);
}

async function loadVocabulary(force) {
  if (vocabularyLoaded && !force) return;
  try {
    // GET /tag-vocabulary is not an ETag route, so api.getTagVocabulary()
    // resolves to the body itself -- { tags, total } -- and NOT to a
    // { data, etag } wrapper the way getIndexTags() does. Reading it as the
    // wrapper is the bug this comment exists to prevent: it fails silently,
    // because an empty candidate list looks exactly like an index with no
    // tags yet.
    vocabularyPayload = await api.getTagVocabulary();
    vocabularyLoaded = true;
  } catch {
    // A missing vocabulary costs completions, not the ability to type a name,
    // so it is never raised to the user. The flag is set either way: retrying
    // on every keystroke would turn one unreachable endpoint into a flood.
    //
    // The previous payload is deliberately kept. The file is meant to be
    // edited by hand, so a failed read is most likely a half-written file
    // caught mid-save; dropping the names already in hand would make an
    // editor's keystroke empty every datalist in the application.
    vocabularyLoaded = true;
  }
}

let vocabularyInFlight = null;

// Refetch the vocabulary without waiting for it.
//
// DATA_ROOT/tag_vocabulary.json is a hand-editable file and the server reads
// it on every GET, so the only thing between an edit and the datalist is when
// the client asks. Fetching once at startup meant an edit made during a
// session did not appear until a reload or a successful write, so a picker
// also asks whenever it becomes visible.
//
// Nothing waits on the result: the picker opens and accepts typing straight
// away, and the names land through the usual notifyCandidates() path when
// they arrive. Concurrent calls share one request, which is what makes this
// safe to call on every open -- the response is small, but a popover reopened
// in a loop should still issue one request at a time.
export function refreshTagVocabulary() {
  if (vocabularyInFlight) return vocabularyInFlight;
  vocabularyInFlight = loadVocabulary(true)
    .then(() => {
      notifyCandidates();
    })
    .finally(() => {
      vocabularyInFlight = null;
    });
  return vocabularyInFlight;
}

async function loadInUse(name) {
  if (!name) {
    inUsePayload = null;
    candidateIndexName = null;
    return;
  }
  try {
    // This one IS an ETag route, hence the wrapper.
    const { data } = await api.getIndexTags(name);
    inUsePayload = data;
  } catch {
    inUsePayload = null;
  }
  candidateIndexName = name;
}

// Point the candidate list at an index. Called on an index switch: the
// vocabulary is kept (it is application-wide), the in-use list is refetched.
export async function setTagCandidateIndex(name) {
  await Promise.all([loadVocabulary(false), loadInUse(name)]);
  notifyCandidates();
}

// Refetch both lists. Called after a successful write so a new name is offered
// everywhere, immediately.
export async function refreshTagCandidates() {
  await Promise.all([loadVocabulary(true), loadInUse(candidateIndexName)]);
  notifyCandidates();
}

// The merged list: in use first (descending count), then vocabulary-only names
// by name. See tag-candidates.js for why the order matters.
export function tagCandidateNames() {
  return mergeTagCandidates(inUsePayload, vocabularyPayload);
}

// Fill a <datalist> with the current candidates. Exported because the
// single-part tag editor keeps its own markup and only shares this list.
export function fillTagDatalist(list) {
  if (!list) return;
  list.innerHTML = '';
  for (const name of tagCandidateNames()) {
    const option = document.createElement('option');
    option.value = name;
    list.appendChild(option);
  }
}

// -- The control ------------------------------------------------------------

let pickerSeq = 0;

// Why a part was not tagged, in words. The server reports machine reasons; a
// skipped part is never dropped silently, so each one has to read as English.
function reasonText(reason) {
  if (reason === 'tag_limit_reached') return 'already at the per-part tag limit';
  if (reason === 'not_registered') return 'not known to the index';
  return String(reason || 'unknown reason');
}

function describeResult(tag, data, doneVerb) {
  const updated = Number(data && data.updated) || 0;
  const skipped = (data && data.skipped) || [];
  if (!skipped.length) return `${doneVerb} ${updated} part(s) with '${tag}'.`;
  const reasons = [...new Set(skipped.map((entry) => reasonText(entry && entry.reason)))];
  return (
    `${doneVerb} ${updated} part(s) with '${tag}'; ` +
    `${skipped.length} part(s) were skipped (${reasons.join('; ')}).`
  );
}

function errorText(err) {
  return err && err.message ? err.message : String(err);
}

// Build the control inside `container`.
//
//   variant       'inline' | 'popover' -- styling only; the body is identical.
//   initialValue  what the field starts with; a caller that knows the likely
//                 answer (the gallery's current tag filter, say) offers it
//                 rather than making the user retype it.
//   getSelection  () => { ids, groups } -- the parts to write to, and
//                 optionally how many groups they came from. Read on every
//                 sync, so a selection that changes elsewhere is reflected
//                 without the caller pushing it in.
//   onApply       async (tag, selection) => { updated, skipped } -- the write.
//                 Errors are shown, not swallowed.
//   onApplied     optional, after a successful write, for the caller's own
//                 refresh (tag lists, tiles). Keep it to what the catalogue
//                 already proved safe: a map rebuild from here is a
//                 regression, not a refresh.
//   isBlocked     optional, () => bool -- an outside reason to disable
//                 everything (a map build running, no catalogue fetched).
//
// Returns { element, sync, setStatus, focus, destroy }.
export function createTagPicker({
  container,
  variant = 'inline',
  label = 'Tag',
  placeholder = 'Tag name',
  applyLabel = 'Apply tag',
  initialValue = '',
  progressVerb = 'Tagging',
  doneVerb = 'Tagged',
  failVerb = 'tag',
  getSelection,
  onApply,
  onApplied,
  isBlocked,
}) {
  pickerSeq += 1;
  const listId = `tag-picker-list-${pickerSeq}`;
  let busy = false;

  const root = document.createElement('div');
  root.className = `tag-picker tag-picker-${variant}`;

  // The count that has to be read before Apply. Tagging writes to the index and
  // cannot be undone from here, so what is about to be written is on screen at
  // all times -- selection or no selection.
  const target = document.createElement('div');
  target.className = 'tag-picker-target';
  root.appendChild(target);

  const row = document.createElement('div');
  row.className = 'tag-picker-row';
  const caption = document.createElement('label');
  caption.className = 'tag-picker-label';
  caption.htmlFor = `${listId}-input`;
  caption.textContent = label;
  const input = document.createElement('input');
  input.type = 'text';
  input.id = `${listId}-input`;
  input.className = 'tag-picker-input';
  input.placeholder = placeholder;
  input.maxLength = 64;
  input.autocomplete = 'off';
  input.value = initialValue || '';
  input.setAttribute('list', listId);
  const list = document.createElement('datalist');
  list.id = listId;
  const applyButton = document.createElement('button');
  applyButton.type = 'button';
  applyButton.className = 'tag-picker-apply';
  applyButton.textContent = applyLabel;
  row.append(caption, input, list, applyButton);
  root.appendChild(row);

  const status = document.createElement('div');
  status.className = 'tag-picker-status';
  root.appendChild(status);

  if (container) container.appendChild(root);

  function selection() {
    const value = (getSelection && getSelection()) || {};
    const ids = Array.isArray(value.ids) ? value.ids : [];
    const groups = typeof value.groups === 'number' ? value.groups : null;
    return { ids, groups, note: value.note || '' };
  }

  function setStatus(text) {
    status.textContent = text || '';
  }

  // Count, enabled state and completions, from the current selection. Cheap
  // enough to run on every change: the alternative is a cached count that can
  // disagree with what is about to be written.
  function sync() {
    const { ids, groups, note } = selection();
    const parts = `${ids.length} part(s) selected`;
    const head = groups === null ? parts : `${groups} group(s) \u00b7 ${parts}`;
    target.textContent = note ? `${head} \u00b7 ${note}` : head;
    const blocked = busy || Boolean(isBlocked && isBlocked());
    input.disabled = blocked;
    applyButton.disabled = blocked || ids.length === 0 || !input.value.trim();
  }

  async function apply() {
    if (busy) return;
    const problem = validateTagName(input.value);
    if (problem) {
      setStatus(problem);
      return;
    }
    const tag = input.value.trim();
    const current = selection();
    if (current.ids.length === 0) return;
    busy = true;
    sync();
    setStatus(`${progressVerb} ${current.ids.length} part(s) with '${tag}'\u2026`);
    try {
      const data = await onApply(tag, current);
      setStatus(describeResult(tag, data, doneVerb));
      // The name may be new to this index, or new to the application; both
      // lists are asked again so the next picker -- anywhere -- offers it.
      await refreshTagCandidates();
      if (onApplied) onApplied(tag, data, current);
    } catch (err) {
      setStatus(`Could not ${failVerb}: ${errorText(err)}`);
    } finally {
      busy = false;
      sync();
    }
  }

  input.addEventListener('input', sync);
  input.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    if (!applyButton.disabled) apply().catch(() => {});
  });
  applyButton.addEventListener('click', () => {
    apply().catch(() => {});
  });

  const unsubscribe = onTagCandidatesChanged(() => fillTagDatalist(list));
  fillTagDatalist(list);
  sync();

  return {
    element: root,
    sync,
    setStatus,
    focus: () => input.focus(),
    destroy: () => {
      unsubscribe();
      root.remove();
    },
  };
}

// -- The popover shape ------------------------------------------------------

// One at a time: a second panel over the first would leave two counts on
// screen, and the whole point of the count is that it is unambiguous.
let openPopover = null;

function closeOpenPopover() {
  if (!openPopover) return;
  const panel = openPopover;
  openPopover = null;
  document.removeEventListener('pointerdown', panel.onPointerDown, true);
  document.removeEventListener('keydown', panel.onKeyDown, true);
  panel.picker.destroy();
  panel.element.remove();
}

export function closeTagPopover() {
  closeOpenPopover();
}

// Place the panel near what was clicked, then pull it back inside the window.
// A panel that opens half off-screen is worse than one that is not next to its
// trigger.
function placePanel(element, at) {
  const margin = 8;
  const rect = element.getBoundingClientRect();
  let left = at.x;
  let top = at.y;
  if (left + rect.width + margin > window.innerWidth) left = window.innerWidth - rect.width - margin;
  if (top + rect.height + margin > window.innerHeight) top = window.innerHeight - rect.height - margin;
  element.style.left = `${Math.max(margin, left)}px`;
  element.style.top = `${Math.max(margin, top)}px`;
}

// Open the picker as a small panel.
//
//   at      {x, y} in client coordinates (a right-click), or
//   anchor  an element to open under (a button).
//   title   what is being tagged, in words -- the popover's one addition to
//           the inline bar, because unlike a bar in a tab it is not obvious
//           from its surroundings what the target is.
//
// Every other option is passed straight to createTagPicker.
export function openTagPopover({ at, anchor, title = 'Tag parts', ...options }) {
  closeOpenPopover();

  // The vocabulary may have been edited by hand since it was last read. This
  // does not block the panel below -- it opens and accepts typing now, and
  // the names are rebuilt if and when a newer list arrives.
  refreshTagVocabulary();

  const element = document.createElement('div');
  element.className = 'tag-popover';

  const head = document.createElement('div');
  head.className = 'tag-popover-head';
  const heading = document.createElement('span');
  heading.className = 'tag-popover-title';
  heading.textContent = title;
  const close = document.createElement('button');
  close.type = 'button';
  close.className = 'tag-popover-close';
  close.textContent = '\u00d7';
  close.title = 'Close';
  close.addEventListener('click', () => closeOpenPopover());
  head.append(heading, close);
  element.appendChild(head);

  const picker = createTagPicker({ ...options, container: element, variant: 'popover' });
  document.body.appendChild(element);

  let point = at;
  if (!point && anchor && anchor.getBoundingClientRect) {
    const rect = anchor.getBoundingClientRect();
    point = { x: rect.left, y: rect.bottom + 4 };
  }
  placePanel(element, point || { x: 80, y: 80 });

  const onPointerDown = (event) => {
    if (element.contains(event.target)) return;
    closeOpenPopover();
  };
  const onKeyDown = (event) => {
    if (event.key !== 'Escape') return;
    event.stopPropagation();
    closeOpenPopover();
  };
  document.addEventListener('pointerdown', onPointerDown, true);
  document.addEventListener('keydown', onKeyDown, true);
  openPopover = { element, picker, onPointerDown, onKeyDown };

  picker.focus();
  return { close: closeOpenPopover, sync: picker.sync };
}
