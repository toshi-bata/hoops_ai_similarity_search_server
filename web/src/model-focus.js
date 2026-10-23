// Splitting the model candidates into "the pinned categories, in full" and
// "everything else".
//
// Import-free on purpose, like tag-candidates.js and legend-order.js: the
// decision this makes is the whole feature, and a module with no imports can
// be loaded straight into node and tested without a DOM.

// Divide *candidates* into the entries of the pinned categories and the rest.
//
// `candidates` is an array of entries, `keyOf(entry)` returns the category
// name that entry belongs to (or null), `focusNames` is a Set of pinned
// category names and `budget` is how many models may be loaded in total.
//
// Returns `{ head, rest, room, used, dropped }`:
//   head     the entries of the pinned categories that fit, original order
//   rest     every other entry, original order (the ordinary spread's input)
//   room     budget - head.length, never negative
//   used     Set of the pinned category names that actually contributed
//   dropped  the pinned names that did not fit, in the order they were pinned
//
// A pinned category is taken WHOLE or not at all. A cluster shown half-way is
// worse than one not shown at all -- the user cannot tell the difference
// between "this is the cluster" and "this is part of the cluster", and a
// screenshot taken from it would misrepresent the data. The categories are
// tried in the order they were pinned (a Set keeps insertion order), and the
// first one that does not fit stops the loop; the ones after it are not tried
// either, so which categories appear depends only on the order they were
// pinned in and not on their sizes.
//
// A pinned name with no candidates at all -- a category switched off, or one
// left over from another map -- is in neither `used` nor `dropped`: it was not
// refused for want of budget, there was simply nothing of it to show.
//
// `candidates` is never modified; every returned array is new.
export function splitFocusedCandidates(candidates, keyOf, focusNames, budget) {
  const all = Array.isArray(candidates) ? candidates : [];
  const cap = Math.max(0, Math.trunc(Number(budget) || 0));
  const names = focusNames && focusNames.size ? focusNames : null;
  if (!names) {
    return {
      head: [], rest: all.slice(), room: cap, used: new Set(), dropped: [],
    };
  }
  const byName = new Map();
  for (const entry of all) {
    const key = keyOf(entry);
    if (key === null || key === undefined || !names.has(key)) continue;
    const bucket = byName.get(key);
    if (bucket) bucket.push(entry);
    else byName.set(key, [entry]);
  }
  const head = [];
  const used = new Set();
  const dropped = [];
  let full = false;
  for (const name of names) {
    const bucket = byName.get(name);
    if (!bucket || !bucket.length) continue;
    if (full || head.length + bucket.length > cap) {
      full = true;
      dropped.push(name);
      continue;
    }
    for (const entry of bucket) head.push(entry);
    used.add(name);
  }
  const taken = new Set(head);
  const rest = all.filter((entry) => !taken.has(entry));
  return { head, rest, room: Math.max(0, cap - head.length), used, dropped };
}
