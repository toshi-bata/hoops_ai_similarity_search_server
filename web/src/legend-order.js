// Reading the legend's rows in an order that is not the groups array's order.
//
// The groups array is POSITIONAL: the viewer resolves a pick to
// `groups[node].offset + index`, and the baked .scs holds one node per
// category in that same order. Reordering it would silently point every pick
// at the wrong part. So the array stays as the server built it, and the
// legend's rows are drawn through a list of indices into it.
//
// Import-free on purpose, like tag-candidates.js: a module with no imports can
// be run straight from node, which is how tests/test_legend_order.py exercises
// the function below without a browser or a build step.

// The indices of `groups`, in the order the legend should draw them.
//
// `order` is a list of cluster ids from /map/legend-order. Categories whose
// key is `cluster:<id>` are placed in that order; everything else -- Outliers,
// a tag category, a cluster the order was computed before -- keeps its
// original relative position and goes last.
//
// Nothing is dropped and nothing is repeated: the result is always a
// permutation of `[0 .. groups.length - 1]`. A legend that quietly lost a row
// would lose the parts in it too, because the row owns the category's
// visibility checkbox.
export function legendRenderOrder(groups, order) {
  const items = Array.isArray(groups) ? groups : [];
  const total = items.length;
  if (!Array.isArray(order) || order.length === 0) {
    // No order to apply is not an error: it is the legend as it has always
    // been drawn, which is also what the Size setting asks for.
    return items.map((_group, i) => i);
  }
  // cluster id -> index in groups. First occurrence wins, so a malformed
  // payload with two rows for one cluster still yields each index once.
  const indexOfCluster = new Map();
  items.forEach((group, i) => {
    const key = group && typeof group.key === 'string' ? group.key : '';
    if (!key.startsWith('cluster:')) return;
    const id = Number(key.slice('cluster:'.length));
    if (!Number.isInteger(id) || indexOfCluster.has(id)) return;
    indexOfCluster.set(id, i);
  });
  const out = [];
  const taken = new Array(total).fill(false);
  for (const id of order) {
    const i = indexOfCluster.get(Number(id));
    if (i === undefined || taken[i]) continue;
    taken[i] = true;
    out.push(i);
  }
  // Everything the order did not name, in the order the server sent it.
  for (let i = 0; i < total; i += 1) {
    if (!taken[i]) out.push(i);
  }
  return out;
}
