// Shared tile rendering for both the Browse gallery and the Search results.
//
// A tile is one thumbnail with a filename, optional tags, an optional score
// badge (search) and optional diagnostic lines (assembly search). This module
// only fills a tile's contents; it knows nothing about layout. That lets the
// gallery position tiles absolutely for virtual scrolling while the search
// grid lays them out with plain CSS -- the look stays identical either way.

// Fill `el` (a .tile element) with a part's contents.
//
//   thumbUrl   absolute URL of the thumbnail PNG, or null/'' for none
//   filename   text under the thumbnail
//   tags       array of tag strings (may be empty)
//   selected   whether to show the selection highlight
//   badge      short text shown over the thumbnail (e.g. a score), or null
//   badgeClass extra class on that badge, for callers that distinguish kinds
//              of badge by colour. The badge always says in words what it
//              means as well, so the colour is never the only carrier.
//   tileClass  extra class on the tile itself, for a state worth seeing at a
//              glance across a grid (the catalogue's tagged/untagged). As with
//              badgeClass the state is also written in the lines below, so the
//              colour only repeats it.
//   lines      array of small muted text lines under the filename (diagnostics)
//   onClick    single-click handler, or null (select + show properties)
//   onDblClick double-click handler, or null (open in the 3D viewer)
//   selectable whether to show a multi-select checkbox in the thumbnail corner
//              (Gallery only; the search grid leaves this off)
//   selectChecked  whether that checkbox is currently ticked
//   onToggleSelect optional handler -> the checkbox was toggled; receives the
//              new checked state. Its click never bubbles, so ticking a tile
//              never also selects or opens it.
export function renderTile(el, {
  thumbUrl,
  filename,
  tags = [],
  selected = false,
  badge = null,
  badgeClass = null,
  tileClass = null,
  lines = [],
  onClick = null,
  onDblClick = null,
  selectable = false,
  selectChecked = false,
  onToggleSelect = null,
}) {
  el.classList.remove('placeholder');
  el.classList.toggle('selected', selected);
  // Tiles are recycled as the list scrolls, so last time's state class has to
  // come off before this one goes on.
  if (el.dataset.tileClass) {
    el.classList.remove(el.dataset.tileClass);
    delete el.dataset.tileClass;
  }
  if (tileClass) {
    el.classList.add(tileClass);
    el.dataset.tileClass = tileClass;
  }
  el.onclick = onClick;
  el.ondblclick = onDblClick;
  el.innerHTML = '';

  const thumb = document.createElement('div');
  thumb.className = 'thumb';
  if (thumbUrl) {
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.alt = filename;
    img.src = thumbUrl;
    img.addEventListener('error', () => {
      thumb.classList.add('thumb-missing');
      thumb.textContent = 'No thumbnail';
    });
    thumb.appendChild(img);
  } else {
    thumb.classList.add('thumb-missing');
    thumb.textContent = 'No thumbnail';
  }
  if (badge != null) {
    const b = document.createElement('div');
    b.className = badgeClass ? `score-badge ${badgeClass}` : 'score-badge';
    b.textContent = badge;
    thumb.appendChild(b);
  }
  if (selectable) {
    // A multi-select checkbox in the corner. Its own click is stopped so it
    // never triggers the tile's select (properties) or open (3D) handlers.
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.className = 'tile-select';
    box.checked = selectChecked;
    box.title = 'Select for bulk tagging';
    box.addEventListener('click', (event) => event.stopPropagation());
    box.addEventListener('dblclick', (event) => event.stopPropagation());
    box.addEventListener('change', (event) => {
      event.stopPropagation();
      if (onToggleSelect) onToggleSelect(box.checked);
    });
    thumb.appendChild(box);
  }
  el.appendChild(thumb);

  const meta = document.createElement('div');
  meta.className = 'meta';

  const name = document.createElement('div');
  name.className = 'filename';
  name.textContent = filename;
  name.title = filename;
  meta.appendChild(name);

  for (const line of lines) {
    const row = document.createElement('div');
    row.className = 'metric';
    row.textContent = line;
    meta.appendChild(row);
  }

  if (tags.length > 0) {
    const tagRow = document.createElement('div');
    tagRow.className = 'tags';
    for (const tag of tags) {
      const chip = document.createElement('span');
      chip.className = 'tag';
      chip.textContent = tag;
      tagRow.appendChild(chip);
    }
    meta.appendChild(tagRow);
  }

  el.appendChild(meta);
}

// Fill `el` as an empty placeholder (data not loaded yet). Used by the gallery
// while a page is still being fetched.
export function renderPlaceholder(el) {
  el.classList.add('placeholder');
  el.classList.remove('selected');
  el.onclick = null;
  el.ondblclick = null;
  el.innerHTML = '<div class="thumb"></div>';
}
