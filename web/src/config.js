// Client configuration.
//
// Keep this file tiny and obvious: it is the one place an ISV changes to point
// the client at a different server or engine location.

// Base URL of the similarity-search server. Empty means "same origin", which
// is the normal case because the server also serves this client (under /app).
// Set an absolute origin (e.g. "http://localhost:8000") only when developing
// the client with `npm run dev` against a separately running server.
export const SERVER_BASE = '';

// Directory that holds the graphics engine WebAssembly file
// (engine.esm.wasm). The Vite build copies the engine here, and the server
// serves it from the same location, so an absolute path works whether the page
// is opened at /app/ or /static/app/.
export const ENGINE_PATH = '/static/app/engine';

// How many parts to request per fetch. The parts endpoint accepts 1..2000.
// The virtual-scroll gallery fetches parts one page at a time as they scroll
// into view, so this is the chunk size, not a fixed on-screen page. A page of
// 60 keeps each response small while covering several rows at once.
export const PAGE_SIZE = 60;

// Width, in CSS pixels, of a gallery tile. This is the first value an ISV tunes
// to make thumbnails larger or smaller. Tiles keep a fixed size: the gallery
// changes the number of columns to fit the panel width, but never shrinks a
// tile. The thumbnail area uses a 4:3 aspect ratio to match the server's
// 1280x960 thumbnails, so its height is TILE_WIDTH * 3 / 4.
export const TILE_WIDTH = 150;

// How many hits to request per similarity search. This is fixed on purpose:
// there is no Top K field in the UI. Two reasons the client owns this value
// instead of the user:
//   - Search cost barely depends on the count. Measured on one query, 10 hits
//     was actually slower than 301, and 600 was in between: the cost is
//     dominated by encoding the query shape, so the count has no useful knob
//     to turn.
//   - The results are narrowed client-side by the Min similarity slider, so the
//     client fetches a wide set once and slices it down. A small top_k would
//     cap that set, and lowering the slider then could not reveal more hits.
// 301 is 300 neighbours plus the query itself (include_self is always true).
// Keeping it constant also keeps the server's search-response cache key stable
// (top_k is part of that key), so repeated searches keep hitting the cache.
export const SEARCH_TOP_K = 301;

// Note on licensing: the HOOPS Web Viewer loads pre-authored SCS files, which
// are produced server-side by HOOPS AI. Verified against
// @ts3d-hoops/web-viewer-components 2026.2.0, the <hoops-web-viewer> element
// exposes no browser license-key attribute and none is required to display an
// SCS file, so this client intentionally supplies no key.
