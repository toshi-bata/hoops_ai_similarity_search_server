// Turning tag responses into a list of candidate names.
//
// Split out of tag-picker.js, and kept free of imports, because this is the
// part that can be wrong without anything looking wrong: a datalist with no
// options is indistinguishable from an index with no tags. It was wrong once
// already -- the vocabulary response was read as { data: { tags } } when the
// endpoint returns { tags, total } directly, so 62 names arrived and none were
// ever offered -- so the shape handling lives in one place with tests on it
// (tests/test_tag_candidates.py, which runs these functions under Node).
//
// Two endpoints feed the list:
//
//   GET /index/{name}/tags  -> { tags: [{tag, count}], total }   per index
//   GET /tag-vocabulary     -> { tags: ["name", ...], total }    application
//
// Both are accepted in either form -- records or plain strings -- because the
// two differ only in what they carry, not in what a candidate list needs.

// The names in a tags payload, in the order they appear. Accepts the wrapped
// object, a bare array, or anything else (which yields no names rather than
// throwing: a candidate list is an aid, never a reason to fail a panel).
export function namesFromTagsPayload(payload) {
  const entries = Array.isArray(payload) ? payload : payload && payload.tags;
  const names = [];
  for (const entry of Array.isArray(entries) ? entries : []) {
    const name = typeof entry === 'string' ? entry : entry && entry.tag;
    if (typeof name === 'string' && name) names.push(name);
  }
  return names;
}

// The same payload as {name, count} records, for the ordering below. A missing
// or unusable count sorts as 0.
function recordsFromTagsPayload(payload) {
  const entries = Array.isArray(payload) ? payload : payload && payload.tags;
  const records = [];
  for (const entry of Array.isArray(entries) ? entries : []) {
    if (typeof entry === 'string') {
      if (entry) records.push({ name: entry, count: 0 });
      continue;
    }
    const name = entry && entry.tag;
    if (typeof name !== 'string' || !name) continue;
    const count = Number(entry.count);
    records.push({ name, count: Number.isFinite(count) ? count : 0 });
  }
  return records;
}

// The merged candidate list: the tags this index uses first, by descending
// count and then by name, followed by the vocabulary-only names by name.
//
// The order is not cosmetic. The vocabulary is application-wide, so it carries
// names used in other indexes that mean nothing here; putting the in-use names
// first keeps the ones that belong to this index at the top. The in-use order
// is re-established here rather than trusted from the server, so the list is
// the same whoever produced the payload.
//
// `inUse` may be empty -- a fresh index has no tags yet -- and the vocabulary
// names still come through: an empty index is exactly when a remembered name
// is worth the most.
export function mergeTagCandidates(inUsePayload, vocabularyPayload) {
  const records = recordsFromTagsPayload(inUsePayload);
  records.sort((a, b) => (b.count - a.count) || a.name.localeCompare(b.name));

  const seen = new Set();
  const names = [];
  for (const record of records) {
    if (seen.has(record.name)) continue;
    seen.add(record.name);
    names.push(record.name);
  }

  // A name in both lists is one name: it is in use here AND remembered.
  const rest = namesFromTagsPayload(vocabularyPayload).filter((name) => {
    if (seen.has(name)) return false;
    seen.add(name);
    return true;
  });
  rest.sort((a, b) => a.localeCompare(b));
  return [...names, ...rest];
}
