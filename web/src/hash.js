// Client-side SHA-256, used by bulk registration to compute each file's id
// locally so it can ask the server which files it already holds (POST
// /files/exists) and upload only the missing ones -- instead of re-sending
// hundreds of megabytes on every run.
//
// The digest must match the server's file_id scheme exactly: the server derives
// file_id from the raw file bytes' SHA-256 hex (see core.upload_CAD_file_*), so
// this hashes the same bytes with the same algorithm and lower-cases the hex.

// Hash one File/Blob to a lower-case SHA-256 hex string. Reads the whole file
// into memory; CAD parts are small enough that this is fine, and it keeps the
// code to one obvious call. Returns null if the browser exposes no
// crypto.subtle (e.g. a non-secure origin), so the caller can fall back to
// uploading without the exists() skip rather than failing outright.
export async function sha256Hex(file) {
  if (!window.crypto || !window.crypto.subtle) return null;
  const buffer = await file.arrayBuffer();
  const digest = await window.crypto.subtle.digest('SHA-256', buffer);
  const bytes = new Uint8Array(digest);
  let hex = '';
  for (const byte of bytes) hex += byte.toString(16).padStart(2, '0');
  return hex;
}
