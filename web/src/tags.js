// The tag / kind filter combobox.
//
// A single combobox, like the desktop sandbox: "All" (no filter), "Part" and
// "Assembly" (by kind), "Tagged" (only parts that carry any tag), then one
// entry per tag in the index. A "Delete tag" button appears only when a tag
// entry is selected; it removes that tag from the whole index (a destructive
// action, so it confirms first).
//
// Two instances exist (see main.js): the Gallery instance narrows the gallery
// listing on the server; the Results instance slices the search hits on the
// client (showDelete / showKind off). Neither scopes the similarity search or
// the shape map.

import * as api from './api.js';
import { clientId } from './client-id.js';
import { showError, clearError } from './ui.js';

// Fixed choices that always precede the per-tag entries. `value` is what the
// combobox stores; query() below turns it into API parameters.
const FIXED_OPTIONS = [
  { value: 'all', label: 'All' },
  { value: 'kind:part', label: 'Part' },
  { value: 'kind:assembly', label: 'Assembly' },
  { value: 'tagged', label: 'Tagged' },
];

// Client-side check mirroring the server's rule, so an obviously bad tag is
// rejected before the request. The server is still the authority (it also
// enforces the per-part limit and reports its own errors).
export function validateTagName(name) {
  const trimmed = name.trim();
  if (!trimmed) return 'A tag cannot be empty.';
  if (trimmed.length > 64) return 'A tag can be at most 64 characters.';
  if (trimmed.includes('/') || trimmed.includes('\\')) {
    return 'A tag cannot contain "/" or "\\".';
  }
  return null;
}

// Create the filter inside `container`.
//
//   onChange()   called whenever the selection changes, so the caller can
//                re-apply the filter.
//   showDelete   whether to offer the destructive "Delete tag" button (default
//                true). The results filter is a view-only slice, so it omits it.
//   showKind     whether to offer the Part / Assembly (kind) options (default
//                true). The results filter's kind is already fixed by the
//                search, so it omits them and offers All / Tagged / per-tag.
//   labelText    the row label (default "Filter:").
//
// Returns { setIndex(name), reload(), reset(), query() }.
export function createTagFilter({
  container,
  onChange,
  showDelete = true,
  showKind = true,
  labelText = 'Filter:',
}) {
  let indexName = null;
  let tags = []; // tag names, sorted for the dropdown
  let selection = 'all'; // the current combobox value

  container.className = 'tag-filter';

  const label = document.createElement('label');
  label.className = 'tag-filter-row';
  label.append(document.createTextNode(labelText));

  const select = document.createElement('select');
  select.className = 'tag-filter-select';
  select.addEventListener('change', () => {
    selection = select.value;
    updateDeleteButton();
    onChange();
  });
  label.appendChild(select);

  // Shown only when a specific tag is selected; deletes that tag everywhere.
  const deleteButton = document.createElement('button');
  deleteButton.type = 'button';
  deleteButton.className = 'tag-filter-delete';
  deleteButton.textContent = 'Delete tag';
  deleteButton.hidden = true;
  deleteButton.addEventListener('click', () => deleteSelectedTag());

  container.appendChild(label);
  if (showDelete) container.appendChild(deleteButton);

  // The tag name behind the current selection, or null for a fixed option.
  function selectedTag() {
    return selection.startsWith('tag:') ? selection.slice('tag:'.length) : null;
  }

  function updateDeleteButton() {
    if (!showDelete) return;
    deleteButton.hidden = selectedTag() === null;
  }

  function renderOptions() {
    select.innerHTML = '';
    for (const opt of FIXED_OPTIONS) {
      // The results filter hides the kind options: within a single search the
      // kind is already fixed, so Part / Assembly would never narrow anything.
      if (!showKind && opt.value.startsWith('kind:')) continue;
      const el = document.createElement('option');
      el.value = opt.value;
      el.textContent = opt.label;
      select.appendChild(el);
    }
    for (const tag of tags) {
      const el = document.createElement('option');
      el.value = `tag:${tag}`;
      el.textContent = tag;
      select.appendChild(el);
    }
    // Keep the current selection if it still exists (e.g. after a reload), else
    // fall back to "All".
    if (![...select.options].some((o) => o.value === selection)) {
      selection = 'all';
    }
    select.value = selection;
    updateDeleteButton();
  }

  async function deleteSelectedTag() {
    const tag = selectedTag();
    if (!tag) return;
    // Confirm with wording that makes the scope explicit: this erases the tag
    // from every part, not just from the filter.
    const ok = window.confirm(
      `Delete the tag "${tag}" from every part in index "${indexName}"?\n\n` +
        'This removes the tag everywhere and cannot be undone.',
    );
    if (!ok) return;
    clearError();
    try {
      // Send the index-tags ETag as If-Match: if the tags changed since this
      // list was loaded, the delete is rejected (412) rather than acting on a
      // stale view. confirm=true is required by the server.
      const { etag } = await api.getIndexTags(indexName);
      await api.deleteTag(indexName, tag, {
        confirm: true,
        ifMatch: etag,
        clientId: clientId(),
      });
      selection = 'all';
      await reload();
      onChange();
    } catch (err) {
      showError(err);
    }
  }

  async function reload() {
    if (!indexName) return;
    try {
      const { data } = await api.getIndexTags(indexName);
      // Alphabetical by name for the dropdown, so a tag is easy to find.
      tags = data.tags.map((t) => t.tag).sort((a, b) => a.localeCompare(b));
      renderOptions();
    } catch {
      // A failed tag load leaves the previous options in place; the filter is a
      // convenience, not required for browsing.
    }
  }

  renderOptions();

  return {
    // Switch to a different index: reset to "All" and load its tags.
    async setIndex(name) {
      indexName = name;
      selection = 'all';
      await reload();
    },
    // Reload the tag list (after a tag edit added or removed tags).
    reload,
    // Reload the tag list, then select a specific tag and fire onChange, so a
    // caller can jump the view to "filtered by this tag" (e.g. after bulk-
    // tagging a cluster). A no-op if the tag is absent after the reload.
    async selectTag(tag) {
      await reload();
      const value = `tag:${tag}`;
      if ([...select.options].some((o) => o.value === value)) {
        selection = value;
        select.value = value;
        updateDeleteButton();
        onChange();
      }
    },
    // Reset the selection to "All" without a server round-trip. Setting the
    // value programmatically does not fire 'change', so this never calls
    // onChange -- the caller decides when to re-render.
    reset() {
      selection = 'all';
      if ([...select.options].some((o) => o.value === 'all')) {
        select.value = 'all';
      }
      updateDeleteButton();
    },
    // The current selection as filter arguments, or an empty object for "All":
    // { kind } and/or { tags } / { tagged }. The gallery instance sends these to
    // the server to narrow the listing; the results instance uses tags / tagged
    // to slice the already-returned hits on the client. Either way this never
    // scopes the similarity search or the shape map at the server.
    query() {
      switch (selection) {
        case 'kind:part':
          return { kind: 'part' };
        case 'kind:assembly':
          return { kind: 'assembly' };
        case 'tagged':
          return { tagged: true };
        default: {
          const tag = selectedTag();
          return tag ? { tags: tag } : {};
        }
      }
    },
  };
}
