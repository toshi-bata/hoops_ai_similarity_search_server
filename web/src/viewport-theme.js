// Light/dark background for the viewport (.viewer-pane and the overlay
// chrome drawn on top of it). The 3D viewer and the Shape Map share that
// pane, so a single switch serves both.
//
// This is deliberately a CSS-only affair: the WebGL canvas is transparent and
// the pane's background shows through it, which is also how the HOOPS
// reference demo-app does it. Hooking view.setBackgroundColor() there and
// working its own light/dark switch records zero calls, and
// view.getBackgroundColor() returns {top: null, bottom: null} in both states.
// Do NOT "improve" this by calling view.setBackgroundColor(): a painted
// background would sit on top of the CSS one, the two would disagree, and the
// switch would then have to be applied again on every model load.

const STORAGE_KEY = 'hoops-ai.viewport.theme';
const THEMES = ['dark', 'light'];
// Dark is the default: it is what the app has always looked like, so an
// existing user sees no change until they ask for one.
const DEFAULT_THEME = 'dark';

// A sun (shown while dark is active: pressing it brings the light livery) and
// a moon (shown while light is active). Drawn inline rather than loaded so
// the header has no extra request and no icon dependency.
const SUN_ICON =
  '<circle cx="10" cy="10" r="4" fill="currentColor"></circle>' +
  '<g stroke="currentColor" stroke-width="1.6" stroke-linecap="round">' +
  '<line x1="10" y1="1" x2="10" y2="3.2"></line>' +
  '<line x1="10" y1="16.8" x2="10" y2="19"></line>' +
  '<line x1="1" y1="10" x2="3.2" y2="10"></line>' +
  '<line x1="16.8" y1="10" x2="19" y2="10"></line>' +
  '<line x1="3.7" y1="3.7" x2="5.3" y2="5.3"></line>' +
  '<line x1="14.7" y1="14.7" x2="16.3" y2="16.3"></line>' +
  '<line x1="3.7" y1="16.3" x2="5.3" y2="14.7"></line>' +
  '<line x1="14.7" y1="5.3" x2="16.3" y2="3.7"></line>' +
  '</g>';
const MOON_ICON =
  '<path d="M15.5 12.4A6.6 6.6 0 0 1 7.6 4.5a6.6 6.6 0 1 0 7.9 7.9Z" fill="currentColor"></path>';

function readStored() {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (THEMES.includes(stored)) return stored;
  } catch (err) {
    // localStorage blocked (private mode, disabled): use the default.
  }
  return DEFAULT_THEME;
}

function store(theme) {
  try {
    window.localStorage.setItem(STORAGE_KEY, theme);
  } catch (err) {
    // Blocked: the choice still applies for this page, it just is not kept.
  }
}

/**
 * Current viewport theme, as it is written on the pane. Exported so other
 * modules (the portalled row menu in shape-map.js) can carry it across to
 * elements that are not descendants of .viewer-pane and therefore do not
 * inherit the theme tokens.
 *
 * @returns {string} 'dark' or 'light'.
 */
export function viewportTheme() {
  const pane = document.querySelector('.viewer-pane');
  const theme = pane && pane.dataset.viewportTheme;
  return THEMES.includes(theme) ? theme : DEFAULT_THEME;
}

function apply(theme, button, icon) {
  const pane = document.querySelector('.viewer-pane');
  if (pane) pane.dataset.viewportTheme = theme;
  if (icon) icon.innerHTML = theme === 'light' ? MOON_ICON : SUN_ICON;
  if (button) {
    // The label names what the button will DO, not what is showing now: that
    // is the question a reader hovering over it is asking.
    const label =
      theme === 'light'
        ? 'Switch the viewport to a dark background'
        : 'Switch the viewport to a light background';
    button.title = label;
    button.setAttribute('aria-label', label);
    button.setAttribute('aria-pressed', theme === 'light' ? 'true' : 'false');
  }
}

/**
 * Wire the header's light/dark switch and restore the last choice.
 */
export function initViewportTheme() {
  const button = document.getElementById('viewport-theme-toggle');
  const icon = document.getElementById('viewport-theme-icon');
  apply(readStored(), button, icon);
  if (!button) return;
  button.addEventListener('click', () => {
    const next = viewportTheme() === 'light' ? 'dark' : 'light';
    apply(next, button, icon);
    store(next);
  });
}
