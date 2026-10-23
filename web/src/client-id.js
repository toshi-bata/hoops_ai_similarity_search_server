// A stable identifier for this browser, sent as X-Client-Id on tag writes so
// the server can record who made a change (updated_by).
//
// The value only has to be unique and stable per browser; it is not a login.
// We generate one the first time it is needed and keep it in localStorage so
// the same browser reports the same id across reloads. If localStorage is
// unavailable (private mode, disabled), we fall back to a per-session id held
// in memory, which still lets the header be sent.

const STORAGE_KEY = 'hoops-ai-client-id';

let cached = null;

function makeId() {
  // A short random token is enough; crypto.randomUUID exists in every browser
  // this client targets, with a Math.random fallback just in case.
  if (window.crypto && typeof window.crypto.randomUUID === 'function') {
    return `web-${window.crypto.randomUUID()}`;
  }
  return `web-${Math.random().toString(36).slice(2)}-${Date.now().toString(36)}`;
}

// Return this browser's client id, creating and persisting one on first use.
export function clientId() {
  if (cached) return cached;
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored) {
      cached = stored;
      return cached;
    }
    cached = makeId();
    window.localStorage.setItem(STORAGE_KEY, cached);
  } catch {
    // localStorage blocked: keep an in-memory id for this session only.
    cached = makeId();
  }
  return cached;
}
