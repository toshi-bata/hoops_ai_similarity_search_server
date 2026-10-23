// The one loading/error convention for the whole client.
//
// Every screen uses withLoading() for async work and showError() for failures,
// so there is a single, predictable way things behave. These helpers talk to a
// fixed loading overlay and error banner defined in index.html.

import { ApiError } from './api.js';

function overlay() {
  return document.getElementById('loading-overlay');
}

function banner() {
  return document.getElementById('error-banner');
}

function bannerText() {
  return document.getElementById('error-banner-text');
}

// The message currently on screen. Kept so an identical error repeated while
// the banner still shows it (a retry that keeps failing the same way, or one
// action that reports the same failure twice) does not rewrite the banner
// underneath the cursor reaching for its x. Dismissing resets it, so the same
// error raised by a later action is shown again.
let shownMessage = null;
let closeWired = false;

// Wire the banner's close button once, lazily: ui.js has no init step, and
// every path that can show the banner goes through showError() first.
function wireClose() {
  if (closeWired) return;
  const button = document.getElementById('error-banner-close');
  if (!button) return;
  button.addEventListener('click', clearError);
  closeWired = true;
}

// Show the full-window loading overlay while `task` runs, then hide it. Any
// error is turned into a user-visible banner and re-thrown so callers can also
// react if they need to.
export async function withLoading(task) {
  const el = overlay();
  if (el) el.hidden = false;
  try {
    return await task();
  } catch (err) {
    showError(err);
    throw err;
  } finally {
    if (el) el.hidden = true;
  }
}

// Display an error to the user. ApiError shows its status and the server's
// detail message verbatim (so 404 / 409 / 412 / 422 are never hidden); any
// other error shows its message. The banner can be dismissed with its x, and
// repeating the same message never revives a dismissed banner.
export function showError(err) {
  const el = banner();
  if (!el) return;
  let text;
  if (err instanceof ApiError) {
    text = err.status ? `Error ${err.status}: ${err.detail}` : err.detail;
  } else {
    text = err && err.message ? err.message : String(err);
  }
  if (text === shownMessage) return;
  shownMessage = text;
  const textEl = bannerText();
  if (textEl) textEl.textContent = text;
  else el.textContent = text;
  wireClose();
  el.hidden = false;
}

// Clear the error banner (also the close button's handler).
export function clearError() {
  const el = banner();
  if (!el) return;
  const textEl = bannerText();
  if (textEl) textEl.textContent = '';
  shownMessage = null;
  el.hidden = true;
}
