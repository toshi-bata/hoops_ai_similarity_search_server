// The selected part's tag editor: the reference example of optimistic-locking
// conflict handling (If-Match / 412).
//
// A part can carry several tags at once, so tags are shown as a set of chips,
// each removable on its own; a text box adds one more. Every change -- adding
// or removing a single chip -- saves the WHOLE tag set with PUT .../tags (the
// server replaces the set, it does not take deltas). That is deliberate and
// visible in the code: `save(nextSet)` always receives the entire desired set.
//
// Concurrency is the point of this file. Each read returns an ETag; the save
// sends it back as If-Match. If someone else changed the index's tags in
// between, the server returns 412. We do NOT silently retry or overwrite:
// instead we tell the user, reload the current tags so they can see what
// changed, and let them redo their edit against the fresh state. Swallowing the
// 412 or auto-resending would defeat the lock the server put there on purpose.

import * as api from './api.js';
import { clientId } from './client-id.js';
import { ApiError } from './api.js';
import { showError, clearError } from './ui.js';
import { validateTagName } from './tags.js';
import { fillTagDatalist, onTagCandidatesChanged, refreshTagCandidates } from './tag-picker.js';

// Create the editor inside `container`.
//
//   onTagsCommitted(partId, tags)  called after a successful save, so the
//                                  caller can refresh the tile and tag counts.
//
// Returns { show(indexName, partId), clear() }.
export function createTagEditor({ container, onTagsCommitted }) {
  // The state for the part currently being edited.
  let indexName = null;
  let partId = null;
  // The last set the server confirmed, and the ETag that goes with it. `etag`
  // is what we send as If-Match on the next write.
  let committed = [];
  let etag = null;
  let busy = false;

  container.className = 'tag-editor';

  // The completions under the add field. One id per editor (there is one), and
  // a subscription so a name invented in another part of the app shows up here
  // without this editor being re-opened.
  const datalistId = 'tag-editor-tag-list';
  onTagCandidatesChanged(() => {
    const list = container.querySelector('datalist');
    if (list) fillTagDatalist(list);
  });

  // Show the editor for one part: read its current tags and ETag, then render.
  async function show(name, id) {
    indexName = name;
    partId = id;
    committed = [];
    etag = null;
    renderLoading();
    try {
      const result = await api.getPartTags(name, id);
      committed = result.data.tags;
      etag = result.etag;
      render();
    } catch (err) {
      renderMessage(`Could not load tags: ${errorText(err)}`, 'error');
    }
  }

  function clear() {
    indexName = null;
    partId = null;
    committed = [];
    etag = null;
    container.innerHTML = '';
  }

  function renderLoading() {
    container.innerHTML = '';
    const label = document.createElement('div');
    label.className = 'tag-editor-title';
    label.textContent = 'Tags';
    const loading = document.createElement('div');
    loading.className = 'tag-editor-hint';
    loading.textContent = 'Loading tags\u2026';
    container.append(label, loading);
  }

  // The whole editor: a title, the current chips (each removable), an add row,
  // and a message area reused for conflicts and errors.
  function render() {
    container.innerHTML = '';

    const title = document.createElement('div');
    title.className = 'tag-editor-title';
    title.textContent = 'Tags';
    container.appendChild(title);

    const chips = document.createElement('div');
    chips.className = 'tag-editor-chips';
    if (committed.length === 0) {
      const none = document.createElement('span');
      none.className = 'tag-editor-hint';
      none.textContent = 'No tags yet.';
      chips.appendChild(none);
    } else {
      for (const tag of committed) {
        chips.appendChild(buildChip(tag));
      }
    }
    container.appendChild(chips);

    const addRow = document.createElement('div');
    addRow.className = 'tag-editor-add';
    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'Add a tag\u2026';
    input.maxLength = 64;
    input.disabled = busy;
    // The same candidates every other tagging control offers: the tags this
    // index uses, then the application's vocabulary. Only the list is shared --
    // this editor keeps its chips, its whole-set PUT and its 412 handling,
    // because adding one tag to one part is a different act from writing one
    // tag across a selection.
    input.autocomplete = 'off';
    input.setAttribute('list', datalistId);
    const list = document.createElement('datalist');
    list.id = datalistId;
    fillTagDatalist(list);
    const addButton = document.createElement('button');
    addButton.type = 'button';
    addButton.textContent = 'Add';
    addButton.disabled = busy;

    const submit = () => addTag(input.value, input);
    addButton.addEventListener('click', submit);
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        submit();
      }
    });

    addRow.append(input, list, addButton);
    container.appendChild(addRow);

    const message = document.createElement('div');
    message.className = 'tag-editor-message';
    message.hidden = true;
    container.appendChild(message);
    input.focus();
  }

  // One removable tag chip. The × removes just this tag (saving the reduced
  // set), leaving the other tags untouched.
  function buildChip(tag) {
    const chip = document.createElement('span');
    chip.className = 'tag';
    const text = document.createElement('span');
    text.textContent = tag;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'tag-remove';
    remove.textContent = '\u00d7';
    remove.title = `Remove "${tag}"`;
    remove.disabled = busy;
    remove.addEventListener('click', () => {
      save(committed.filter((t) => t !== tag));
    });
    chip.append(text, remove);
    return chip;
  }

  // Add one tag: validate it, ignore an exact duplicate, otherwise save the
  // whole set plus the new tag. Adding never drops the existing tags.
  function addTag(raw, input) {
    const problem = validateTagName(raw);
    if (problem) {
      renderMessage(problem, 'error');
      return;
    }
    const tag = raw.trim();
    if (committed.includes(tag)) {
      renderMessage(`"${tag}" is already set.`, 'info');
      if (input) input.value = '';
      return;
    }
    save([...committed, tag]);
  }

  // Save the entire desired tag set. `nextSet` is the whole set the part should
  // have after this edit -- not a delta. On success the committed set and ETag
  // advance; on 412 we run the conflict flow; other errors are surfaced and the
  // editor is left on the last known-good set.
  async function save(nextSet) {
    if (busy) return;
    busy = true;
    clearError();
    setMessage('Saving\u2026', 'info');
    try {
      const result = await api.putPartTags(indexName, partId, nextSet, {
        ifMatch: etag,
        clientId: clientId(),
      });
      committed = result.data.tags;
      etag = result.etag;
      busy = false;
      render();
      // A name added here is a name the other tagging controls should offer.
      refreshTagCandidates().catch(() => {});
      if (onTagsCommitted) onTagsCommitted(partId, committed);
    } catch (err) {
      busy = false;
      if (err instanceof ApiError && err.status === 412) {
        await handleConflict();
        return;
      }
      // A rejected write (e.g. 422 over the per-part limit) leaves the server
      // unchanged; re-render the known-good set and show why it failed.
      render();
      renderMessage(errorText(err), 'error');
      showError(err);
    }
  }

  // 412: someone else changed the tags first. Do not overwrite and do not
  // retry automatically. Reload the current tags (fresh ETag) so the user sees
  // the latest state, and ask them to redo their change against it.
  async function handleConflict() {
    try {
      const result = await api.getPartTags(indexName, partId);
      committed = result.data.tags;
      etag = result.etag;
      render();
      renderMessage(
        'Someone else changed the tags first. The current tags are shown ' +
          'above; review them and make your change again.',
        'error',
      );
      if (onTagsCommitted) onTagsCommitted(partId, committed);
    } catch (err) {
      renderMessage(`Tags changed elsewhere and could not be reloaded: ${errorText(err)}`, 'error');
    }
  }

  // Rebuild the editor, then show a message. Used after add/remove so the
  // message survives the re-render.
  function renderMessage(text, level) {
    render();
    setMessage(text, level);
  }

  function setMessage(text, level) {
    const message = container.querySelector('.tag-editor-message');
    if (!message) return;
    message.textContent = text;
    message.className = `tag-editor-message ${level}`;
    message.hidden = false;
  }

  function errorText(err) {
    if (err instanceof ApiError) {
      return err.status ? `Error ${err.status}: ${err.detail}` : err.detail;
    }
    return err && err.message ? err.message : String(err);
  }

  return { show, clear };
}
