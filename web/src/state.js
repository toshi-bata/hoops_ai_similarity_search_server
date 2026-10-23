// Application state.
//
// A single plain object, read and written directly. No framework and no
// reactivity: screens call the render functions themselves after they change
// state. This keeps the data flow easy to follow top to bottom.

export const state = {
  // Name of the index currently being browsed (null until one is chosen).
  indexName: null,
  // Total number of parts in the current index (from the parts endpoint).
  total: 0,
  // The part currently shown in the viewer (null when none).
  selectedPartId: null,
  // The current similarity-search query: the model open in the 3D viewer. It
  // always references a stored file by its id (an indexed part's id, or the id
  // returned when an arbitrary file is uploaded and converted for viewing).
  // `filename` is for display. Null until a model is opened.
  query: null,
};

// Switch to a different index and reset the total/selection. The gallery owns
// its own part cache and is recreated per index, so there is nothing else to
// reset here.
export function selectIndex(name) {
  state.indexName = name;
  state.total = 0;
  state.selectedPartId = null;
  state.query = null;
}
