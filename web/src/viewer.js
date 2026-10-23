// 3D viewer: shows one part's SCS stream cache using the HOOPS Web Viewer
// component with a standard toolbar and model tree.
//
// Importing the components package registers the custom elements
// (<hoops-web-viewer>, <hoops-web-viewer-context-manager>, <hoops-toolbar-*>,
// <hoops-model-tree>). A fresh <hoops-web-viewer> is created for each part:
// the documented component API loads the SCS from its endpoint-uri at
// initialization, and recreating the element gives every part a clean scene
// and a clean error state. The toolbar and tree are siblings of it, not
// slotted children -- see the comment in showPart() -- wired up explicitly
// rather than through any automatic slot/context mechanism. tryGetService()
// reaches the components package's service registry, used by
// setPerspectiveProjection() below.

import { tryGetService } from '@ts3d-hoops/web-viewer-components';
import {
  Markup,
  Point2,
  Point3,
  Color,
  InvalidNodeId,
  PickConfig,
  SelectionMask,
  MeshData,
  MeshInstanceData,
  MeshInstanceCreationFlags,
  Projection,
  OverlayAnchor,
  ViewOrientation,
  Matrix,
  Box,
  SelectionMode,
  OperatorId,
  LoadSubtreeConfig,
} from '@ts3d-hoops/web-viewer';
import * as Communicator from '@ts3d-hoops/web-viewer';
import { ENGINE_PATH } from './config.js';

// If the graphics engine never reports readiness within this window, treat it
// as an initialization failure (wrong engine path, missing wasm, wrong MIME).
const ENGINE_INIT_TIMEOUT_MS = 20000;

// Most entries are <hoops-toolbar-*> tag names; Cutting Planes has no
// official toolbar-button component of its own (confirmed by reading the
// full hoops-toolbar-buttons/ directory listing -- no cutting/section
// button exists there), so it is a plain object describing a custom button
// built by buildCustomToolbarButton() instead of document.createElement().
const TOOLBAR_BUTTONS = [
  'hoops-toolbar-model-tree',
  'hoops-toolbar-home',
  'hoops-toolbar-camera',
  'hoops-toolbar-drawmode',
  'hoops-toolbar-explode',
  { key: 'cutting-planes', icon: 'cuttingPlaneSection', title: 'Cutting Planes' },
  'hoops-toolbar-tools',
  'hoops-toolbar-snapshot',
];

// The Shape Map is a point cloud, so it only needs navigation buttons (no draw
// modes or model tree). Two of its three buttons are custom ones built by
// buildCustomToolbarButton() -- the same mechanism Cutting Planes uses above:
//
//   home    the stock <hoops-toolbar-home> calls ContextManager.reset(), i.e.
//           WebViewer.reset(), documented as resetting "the camera, visibility,
//           and transforms". On the map that also undoes the legend's
//           filtering: the hidden categories come back while their checkboxes
//           stay unticked, and the overlaid models are not re-picked, so the
//           scene and the legend disagree. On the map, Home means "put the
//           viewpoint back", not "undo my filtering", so this one only
//           re-applies the iso view orientation. The part and comparison
//           toolbars keep the stock button, where resetting visibility and
//           transforms is the right behaviour.
//   legend  the legend is a panel of the map like the model tree is a panel of
//           a part, but has no official <hoops-toolbar-*> component.
//
// This list is used only by showMap(), so neither button appears on the part or
// comparison toolbars.
const MAP_TOOLBAR_BUTTONS = [
  { key: 'map-home', icon: 'home', title: 'Home' },
  'hoops-toolbar-camera',
  { key: 'map-legend', icon: 'paletteIcon', title: 'Legend' },
];

// The two-part comparison scene needs navigation and a draw-mode toggle (to
// switch between shaded and wireframe while inspecting the coloured difference),
// but no model tree or explode.
const COMPARE_TOOLBAR_BUTTONS = [
  'hoops-toolbar-home',
  'hoops-toolbar-camera',
  'hoops-toolbar-drawmode',
];

// A pointerup fires a pick only if the pointer moved less than this many CSS
// pixels since the matching pointerdown -- otherwise it is a camera drag
// (orbit/pan), not a click.
const CLICK_MOVE_TOLERANCE_PX = 4;

// Point-cloud display tuning. The numeric values are the enum members from
// @ts3d-hoops/web-viewer (PointShape, PointSizeUnit); they are written here as
// literals so this reference file needs no extra import. A round splat with
// eye-dome lighting reads as depth even though the SCS carries flat points.
// Point diameter, in CSS pixels. Two values because the map is shown two ways:
// as a bare point cloud, and with real 3D models overlaid at the point
// positions. At 8 px the points read well on their own, but when models are
// overlaid those same fat points bury the small parts sitting on them; 4 px
// keeps the cloud legible as a backdrop while letting the overlaid geometry
// show through. Switching back to the point-cloud view restores 8 px.
const POINT_SIZE_CLOUD = 8; // point diameter, point-cloud-only view
const POINT_SIZE_MODELS = 4; // point diameter when 3D models are overlaid
const POINT_SIZE_UNIT_CSS_PIXELS = 1; // PointSizeUnit.CSSPixels
const POINT_SHAPE_DISK = 1; // PointShape.Disk (a round splat)

// A picked map point is marked with a 2D ring drawn over its projected screen
// position (see PointMarker below), not by highlighting its node -- the whole
// category shares one node, so a node-level highlight would light up every
// point in the category instead of the one that was clicked.
const MARKER_RADIUS_PX = 6;
const MARKER_STROKE_WIDTH_PX = 2;
const MARKER_STROKE_COLOR = new Color(255, 140, 0);

// The hovered cluster's centre part gets a second, distinct ring: wider and in
// a different colour, so it cannot be mistaken for the selection ring above.
// Two markers rather than one reused for both, because the selection is not
// dismissed while hovering the legend and must stay where it is.
const CENTRE_MARKER_RADIUS_PX = 11;
const CENTRE_MARKER_STROKE_WIDTH_PX = 3;
const CENTRE_MARKER_STROKE_COLOR = new Color(80, 200, 255);

// Reference grids drawn around the Shape Map point cloud, for spatial
// context only (see addGridPlane() below). GRID_HALF_SIZE matches the
// server's projection cube (points are normalised to [-1, 1]), so the grids
// roughly bound the point cloud without needing to be recomputed per index.
const GRID_HALF_SIZE = 1;
const GRID_DIVISIONS = 20;
const GRID_LINE_COLOR = new Color(150, 150, 150);

// Colours for the two-part shape comparison (see showComparison). A
// colourblind-safe orange/blue pair for the "only in one file" regions (never a
// red/green pair, which is hard to tell apart for many viewers and washes out
// when projected), and a neutral grey for the shared region so the eye reads
// the two coloured shells as "the difference".
const COMPARE_ONLY1_COLOR = new Color(240, 140, 0); // regions only in file A
const COMPARE_ONLY2_COLOR = new Color(50, 120, 230); // regions only in file B
const COMPARE_SAME_COLOR = new Color(170, 170, 170); // regions in both

// Half-extent (in the map's normalized [-1, 1] space) of the box framed around
// a located point (see focusPoint). Large enough that neighbouring clusters
// stay in view -- the point is meaningless without the cluster around it -- but
// small enough that the camera actually moves in to it.
const FOCUS_HALF_EXTENT = 0.45;

let currentViewer = null;
// Set while a two-part comparison is on screen (see showComparison), so
// disposeViewer() knows to call View.endComparison() before tearing the viewer
// down -- leaving a comparison active would corrupt the next model's colours.
let comparisonActive = false;
// The element actually appended to #viewer-host and removed on dispose --
// the <hoops-web-viewer-context-manager> wrapping currentViewer (see
// showPart()/showMap()), not currentViewer itself, since removing just the
// inner <hoops-web-viewer> would leave its toolbar/tree siblings behind.
let currentViewerRoot = null;
let initTimer = null;
let servicesRegistered = false;
// Watches #viewer-host for the current viewer, so its canvas resolution
// stays in sync with its container's actual size. A `window` 'resize'
// listener is not enough: dragging the gallery panel resizer (see
// initPanelResizer() in main.js) changes #viewer-host's width by updating a
// CSS variable, which never changes the window's own size and so never fires
// `window`'s 'resize' event. Kept at module scope so disposeViewer() can
// disconnect it.
let resizeObserver = null;

// Console handles for the HOOPS Web Viewer.
//
// This is a sample project, and the first thing a reader wants from a viewer
// they did not write is to poke at it: read the camera, set it back, list the
// nodes. Exposing them saves adding a breakpoint to do what one line in the
// console does.
//
//   window.Communicator  the web-viewer namespace (Camera, Point3, Color, ...)
//   window.hwv           the viewer that became ready most recently
//   window.hwvMap        the map's viewer, while the map is on screen
//
// All three viewer flavours (part, comparison, map) share the single
// #viewer-host, and showPart()/showComparison()/showMap() each dispose
// whatever is there before building their own, so only one viewer is ever
// alive. window.hwvMap is set whenever the map's viewer becomes ready and
// nulled when it is disposed; it is therefore non-null exactly while the map
// is on screen. It is worth having beside window.hwv not because it outlives
// the map, but because it says so when the map is gone: window.hwv would
// silently be the part viewer, and a camera restored onto it would act on the
// wrong scene.
//
// A Camera is a plain value, so one captured while the map was up stays usable
// after that viewer is gone -- read it first, then restore it onto the map
// viewer that replaces it:
//
//   let cam = hwvMap.view.getCamera();   // with the map on screen
//   // change the layout, the clustering, k, ... -- the map is rebuilt
//   hwvMap.view.setCamera(cam);          // onto the new map viewer
//
// Cleared on teardown. A handle to a shut-down viewer is worse than no handle:
// its methods either throw or act on a dead scene.
//
// Development convenience only. Nothing in the app reads these, and they are
// not an API.
if (typeof window !== 'undefined') {
  window.Communicator = Communicator;
}

function publishViewerHandle(hwv, { map = false } = {}) {
  if (typeof window === 'undefined' || !hwv) return;
  window.hwv = hwv;
  if (map) window.hwvMap = hwv;
}

function clearViewerHandle({ map = false } = {}) {
  if (typeof window === 'undefined') return;
  window.hwv = null;
  if (map) window.hwvMap = null;
}

// Register the viewer's services once. The toolbar buttons look their services
// up in a shared registry that is only populated while a <hoops-service-registry>
// element is in the DOM; without it, buttons like the camera control throw.
function ensureServices() {
  if (servicesRegistered) return;
  const registry = document.createElement('hoops-service-registry');
  registry.hidden = true;
  document.body.appendChild(registry);
  servicesRegistered = true;
}

// Keep the canvas resolution in sync with #viewer-host's actual size so the
// model does not look stretched/skewed after the container is resized --
// whatever the cause (window resize, the gallery panel resizer, a future
// layout change). Bound once the engine is ready (`hwvReady`, so this also
// works while a model is still streaming in), and disconnected in
// disposeViewer() when this viewer goes away. Only one observer is ever live:
// binding a new one supersedes the previous one, and a viewer that was disposed
// while starting up is not bound at all.
//
// `viewerEl` is the <hoops-web-viewer> that produced this hwv, used to tell a
// live viewer from one that is already gone.
function bindResize(hwv, host, viewerEl) {
  // A viewer whose element has left the document is one disposeViewer() tore
  // down while it was still starting up: its 'hwvReady' arrives after the
  // teardown, and binding to it would leave an observer calling resizeCanvas()
  // on a dead engine. Checked FIRST -- disconnecting before this test would
  // kill the live viewer's observer on the way out.
  if (viewerEl && !viewerEl.isConnected) return;
  // Supersede whatever was bound before. Overwriting the module variable
  // without this left the previous observer watching `host` forever, which is
  // how a disposed viewer kept being resized -- and, because the observer held
  // a reference to it, kept it (and its WebGL context) from being collected.
  if (resizeObserver) resizeObserver.disconnect();
  const observer = new ResizeObserver(() => {
    // A later viewer has taken over; this callback belongs to a previous one.
    if (resizeObserver !== observer) return;
    try {
      hwv.resizeCanvas();
    } catch {
      // The engine can be torn down between the notification and this call.
      // A canvas that cannot be resized is not worth an uncaught exception.
    }
  });
  resizeObserver = observer;
  observer.observe(host);
}

// Show the part whose stream cache is at `scsUrl`.
export function showPart(scsUrl) {
  buildViewer({ scsUrl });
}

// Show the viewer with no model loaded -- the "Open Empty Viewer" menu
// action. `empty` is the SDK's own <hoops-web-viewer> property for this
// (confirmed by reading hoops-web-viewer.js: it passes straight through to
// the WebViewer engine's own `empty` constructor option), not something
// improvised on top of showPart().
export function showEmptyViewer() {
  buildViewer({ empty: true });
}

// The three comparison colours, exported so the caller's on-screen legend uses
// exactly the values baked into the scene by showComparison() (rgb triples).
export const COMPARE_LEGEND = {
  only1: [COMPARE_ONLY1_COLOR.r, COMPARE_ONLY1_COLOR.g, COMPARE_ONLY1_COLOR.b],
  only2: [COMPARE_ONLY2_COLOR.r, COMPARE_ONLY2_COLOR.g, COMPARE_ONLY2_COLOR.b],
  same: [COMPARE_SAME_COLOR.r, COMPARE_SAME_COLOR.g, COMPARE_SAME_COLOR.b],
};

// Resolve once the viewer's scene has finished its initial (empty) load. At
// `hwvReady` the engine is up but the empty scene may still be loading, and its
// session only becomes an SCS session when that load completes; calling
// Model.loadSubtreeFromScsFile before then fails with "Incompatible load
// types". getSceneReady()/the sceneReady callback are the SDK's own signal for
// this (see IWebViewer.d.ts).
function waitForSceneReady(hwv) {
  return new Promise((resolve) => {
    if (typeof hwv.getSceneReady === 'function' && hwv.getSceneReady()) {
      resolve();
      return;
    }
    const callbacks = {
      sceneReady() {
        hwv.unsetCallbacks(callbacks);
        resolve();
      },
    };
    hwv.setCallbacks(callbacks);
  });
}

// Show two parts in one scene and run the HOOPS Web Viewer shape comparison
// (View.startComparison): the regions unique to each part and the shared region
// are painted in three colours (see COMPARE_LEGEND). This is demo 1's "Compare
// two files..." action and uses no index.
//
// Unlike showPart(), this loads TWO subtrees into one empty scene
// (Model.loadSubtreeFromScsFile) and waits for BOTH before comparing, then
// aligns them by TRANSLATION ONLY: the second part is moved so its bounding-box
// centre coincides with the first's. No rotation or scaling -- scaling would
// erase a genuine size difference, which the comparison is meant to reveal.
//
// Returns a promise that resolves, once the comparison is on screen, to a small
// controller ``{ setEnabled(on) }`` the caller drives from the "Show
// differences" switch; it rejects (after showing the cause) if either part
// fails to load.
export function showComparison({ scsA, scsB }) {
  const host = document.getElementById('viewer-host');
  const message = document.getElementById('viewer-message');

  ensureServices();
  disposeViewer();
  setMessage(message, 'Loading both parts\u2026', false);

  // See the comment in showPart() on why the toolbar needs this wrapper.
  const contextManager = document.createElement('hoops-web-viewer-context-manager');
  const viewer = document.createElement('hoops-web-viewer');
  viewer.enginePath = ENGINE_PATH;
  viewer.empty = true;
  currentViewer = viewer;
  currentViewerRoot = contextManager;

  const config = {
    only1Color: COMPARE_ONLY1_COLOR,
    only2Color: COMPARE_ONLY2_COLOR,
    sameColor: COMPARE_SAME_COLOR,
  };

  const engineText =
    `The 3D engine did not start. Check that the viewer engine files are ` +
    `present under ${ENGINE_PATH} and served as application/wasm.`;

  const ready = new Promise((resolve, reject) => {
    initTimer = window.setTimeout(() => {
      initTimer = null;
      setMessage(message, engineText, true);
      reject(new Error(engineText));
    }, ENGINE_INIT_TIMEOUT_MS);

    viewer.addEventListener('hwvReady', async (event) => {
      clearInitTimer();
      const hwv = event.detail;
      publishViewerHandle(hwv);
      bindResize(hwv, host, viewer);
      const model = hwv.model;
      const view = hwv.view;
      try {
        // The empty scene must finish loading before subtrees can be added,
        // otherwise the session is not yet an SCS session (see
        // waitForSceneReady).
        await waitForSceneReady(hwv);
        // Create one container node per part under the assembly root and load
        // each SCS as a subtree beneath it. This runtime requires the link node
        // to be a real node (AssemblyNode/PartInstance/DrawingSheet); a null
        // parent is rejected. The container nodes are what startComparison and
        // the alignment below operate on -- everything a part loads lands under
        // its container. Load sequentially: two concurrent subtree loads into
        // the same fresh scene are not reliable.
        const root = model.getAbsoluteRootNode();
        const nodeA = model.createNode(root, 'compare-a');
        const nodeB = model.createNode(root, 'compare-b');
        // Attach both parts INVISIBLY. nodeB is drawn the instant it loads, but
        // its alignment shift is not applied until after both boxes are
        // measured, so with visible geometry it would first appear at its own,
        // un-aligned position and then jump onto nodeA -- a flash that lands in
        // demo 1, clip 1. Loading invisibly means the un-aligned state is never
        // drawn: we measure, shift nodeB, fit, reveal both, and only then start
        // the comparison. Load sequentially: two concurrent subtree loads into
        // the same fresh scene are not reliable.
        const cfgA = new LoadSubtreeConfig();
        cfgA.attachInvisibly = true;
        const cfgB = new LoadSubtreeConfig();
        cfgB.attachInvisibly = true;
        const rootsA = await model.loadSubtreeFromScsFile(nodeA, scsA, cfgA);
        const rootsB = await model.loadSubtreeFromScsFile(nodeB, scsB, cfgB);
        if (!rootsA || !rootsA.length || !rootsB || !rootsB.length) {
          throw new Error('One of the parts produced no geometry to compare.');
        }
        const setA = [nodeA];
        const setB = [nodeB];
        // Align by bounding-box centre, translation only (see the header): shift
        // the whole second subtree so its centre lands on the first's.
        // getNodesBounding is the world-space geometric box and does not depend
        // on visibility, so it is valid for the attach-invisibly geometry above.
        const [boxA, boxB] = await Promise.all([
          model.getNodesBounding(setA),
          model.getNodesBounding(setB),
        ]);
        const ca = boxA.center();
        const cb = boxB.center();
        const shift = new Matrix().setTranslationComponent(
          ca.x - cb.x,
          ca.y - cb.y,
          ca.z - cb.z,
        );
        await model.setNodeMatrix(nodeB, shift);
        await view.fitWorld();
        // Reveal both parts BEFORE startComparison. The alignment is now applied,
        // so the first drawn frame already shows the parts overlapped, and
        // startComparison's 3-colour output (only1 / only2 / same) is defined on
        // visible geometry -- doing it after the reveal keeps that output intact.
        // Pass false for initiallyHiddenStayHidden so the attach-invisibly
        // geometry actually shows.
        await model.setNodesVisibility([nodeA, nodeB], true, false);
        view.startComparison(setA, setB, config);
        comparisonActive = true;
        clearMessage(message);
        resolve({
          // Turn the coloured comparison on or off (the "Show differences"
          // switch) without rebuilding the scene.
          setEnabled(on) {
            if (!currentViewer || !currentViewer.viewer) return;
            const v = currentViewer.viewer.view;
            if (on) {
              v.startComparison(setA, setB, config);
              comparisonActive = true;
            } else {
              v.endComparison();
              comparisonActive = false;
            }
          },
          // Show or hide one of the two parts (`which` is 'a' or 'b'), so the
          // base and the added part can be inspected alone or overlaid. Toggling
          // visibility hides the container node's whole subtree, including its
          // comparison colouring.
          setPartVisible(which, on) {
            if (!currentViewer || !currentViewer.viewer) return;
            const m = currentViewer.viewer.model;
            m.setNodesVisibility(which === 'a' ? setA : setB, !!on);
          },
        });
      } catch (err) {
        const detail = err && err.message ? err.message : '';
        const text = `Failed to load the two parts for comparison. ${detail}`.trim();
        setMessage(message, text, true);
        reject(new Error(text));
      }
    });

    bindFailure(viewer, message, 'hwvWebGlContextLost', 'The WebGL context was lost.');
    bindFailure(viewer, message, 'hwvTimeout', 'The viewer session timed out.');
  });

  const toolbar = buildToolbar(COMPARE_TOOLBAR_BUTTONS);
  contextManager.append(viewer, toolbar);
  host.appendChild(contextManager);
  return ready;
}

// Enable the view's axis triad in the lower-right corner. Shared by the loaded
// model view (buildViewer) and the point cloud (applyPointCloudView), which
// both anchor it there, clear of the map legend and controls.
async function enableAxisTriad(hwv) {
  if (!hwv) return;
  const view = hwv.view;
  await view.axisTriad.enable();
  await view.axisTriad.setAnchor(OverlayAnchor.LowerRightCorner);
}

function buildViewer({ scsUrl, empty }) {
  const host = document.getElementById('viewer-host');
  const message = document.getElementById('viewer-message');

  ensureServices();
  disposeViewer();
  if (!empty) setMessage(message, 'Loading\u2026', false);

  // Toolbar buttons and the model tree need a shared
  // <hoops-web-viewer-context-manager> ancestor: it is what actually
  // receives the WebViewer instance and hands it to service-backed buttons
  // like camera/drawmode (confirmed by reading hoops-web-viewer.js --
  // handleReady() does `this.contextManager && (this.contextManager.webViewer
  // = this.hwv)`, and without an ancestor context manager this.contextManager
  // stays undefined, so those buttons silently no-op). This wrapper, not
  // <hoops-web-viewer> itself, is what gets appended to #viewer-host and
  // removed on dispose (see currentViewerRoot).
  const contextManager = document.createElement('hoops-web-viewer-context-manager');

  const viewer = document.createElement('hoops-web-viewer');
  viewer.enginePath = ENGINE_PATH;
  if (empty) {
    viewer.empty = true;
  } else {
    viewer.endpointUri = scsUrl;
  }
  currentViewer = viewer;
  currentViewerRoot = contextManager;

  initTimer = window.setTimeout(() => {
    initTimer = null;
    setMessage(
      message,
      `The 3D engine did not start. Check that the viewer engine files are ` +
        `present under ${ENGINE_PATH} and served as application/wasm.`,
      true,
    );
  }, ENGINE_INIT_TIMEOUT_MS);

  viewer.addEventListener('hwvReady', (event) => {
    clearInitTimer();
    publishViewerHandle(event.detail);
    bindResize(event.detail, host, viewer);
    // Unlike every other toolbar button here, <hoops-toolbar-explode> does not
    // pick up the viewer from the ancestor context manager -- it declares its
    // own plain `webViewer` property that must be assigned directly (per its
    // own JSDoc example and the web-components API reference). `toolbar` is
    // assigned further down in this function but is already set by the time
    // this listener runs, since hwvReady can only fire after this function's
    // synchronous body (which creates it) has finished.
    const explodeButton = toolbar.querySelector('hoops-toolbar-explode');
    if (explodeButton) explodeButton.webViewer = event.detail;
    // An empty viewer has no model, so hwvModelStructureReady (below) never
    // fires -- clear the "Loading..." state here instead. There is none to
    // clear since setMessage() was skipped above, but this keeps the message
    // element in its default idle text rather than a stale one.
    if (empty) clearMessage(message);
  });
  viewer.addEventListener('hwvModelStructureReady', () => {
    clearInitTimer();
    clearMessage(message);
    // Show the axis triad in the lower-right corner, as the Shape Map view does
    // (see applyPointCloudView). A loaded part leaves that corner free, and the
    // triad gives an orientation cue that the plain model view otherwise lacks.
    const hwv = viewer.viewer;
    if (hwv) enableAxisTriad(hwv);
  });

  // Never leave a blank canvas: every failure path shows its cause.
  bindFailure(viewer, message, 'hwvModelLoadFailure', 'Failed to load the model.');
  bindFailure(viewer, message, 'hwvMissingModel', 'The model could not be found.');
  bindFailure(viewer, message, 'hwvXHRonerror', 'Network error while fetching the model.');
  bindFailure(viewer, message, 'hwvWebGlContextLost', 'The WebGL context was lost.');
  bindFailure(viewer, message, 'hwvTimeout', 'The viewer session timed out.');

  // The toolbar and model tree render as siblings of <hoops-web-viewer>, not
  // as slotted children of it: this version of <hoops-web-viewer> has no
  // named <slot>s in its shadow DOM (its render() is just an empty
  // container div -- confirmed by reading the compiled component), so a
  // `slot="..."` attribute on a child does not compose it into anything.
  // They reach the viewer instead through the context manager (toolbar
  // buttons) or direct wiring (the tree; see setupModelTree()).
  const toolbar = buildToolbar(TOOLBAR_BUTTONS);
  wireSnapshotButton(toolbar, viewer);

  const tree = document.createElement('hoops-model-tree');
  setupModelTree(tree, viewer);
  const treePanel = document.createElement('div');
  treePanel.className = 'part-tree';
  treePanel.appendChild(tree);

  const cuttingPanel = buildCuttingPanel();
  const toolsPanel = buildToolsPanel();

  // The model-tree, cutting-planes, and tools rail buttons share one docked
  // slot to their right (matching the reference demo app): opening one
  // closes the others, and clicking the currently-open one's icon closes it,
  // leaving the full canvas visible. All three start closed so the canvas is
  // unobstructed on load; the user opens the tree from its toolbar button.
  const setActiveSidePanel = wireSidePanelButtons(toolbar, [
    { selector: 'hoops-toolbar-model-tree', panel: treePanel },
    { selector: '[data-toolbar-key="cutting-planes"]', panel: cuttingPanel },
    { selector: 'hoops-toolbar-tools', panel: toolsPanel },
  ]);
  setActiveSidePanel(null);

  contextManager.append(viewer, toolbar, treePanel, cuttingPanel, toolsPanel);
  host.appendChild(contextManager);
}

// <hoops-tools-panel> bundles selection/measurement/redline/markup groups.
// It picks up the WebViewer automatically from the ancestor
// <hoops-web-viewer-context-manager> via @lit/context, unlike
// <hoops-toolbar-explode> -- no direct .webViewer assignment needed here.
function buildToolsPanel() {
  const panel = document.createElement('div');
  panel.className = 'part-tools';
  panel.hidden = true;
  panel.appendChild(document.createElement('hoops-tools-panel'));
  return panel;
}

// <hoops-cutting-plane-panel> gets its own rail button and docked slot
// (matching the reference demo app, which never nests it inside Tools).
// Like Measurement/Redline, it reads its service (CuttingService) straight
// from the shared service registry singleton in its own connectedCallback()
// rather than via @lit/context, so it does not need to live inside
// <hoops-tools-panel> or even under the context manager -- confirmed by
// reading hoops-cutting-plane-panel.js.
function buildCuttingPanel() {
  const panel = document.createElement('div');
  panel.className = 'part-cutting';
  panel.hidden = true;
  panel.appendChild(document.createElement('hoops-cutting-plane-panel'));
  return panel;
}

// hoops-toolbar-model-tree and hoops-toolbar-tools both have no behavior of
// their own beyond a plain `click` event (their render() is just an icon
// button, confirmed by reading the compiled components) -- the docs say
// toggling a panel's visibility is the application's job, and each button's
// `color` property ('default'/'accent') is meant to reflect open/closed
// state (the custom Cutting Planes button built by buildCustomToolbarButton()
// is a plain <hoops-icon-button>, which has the same `color` property, so it
// fits this same wiring). Since the panels share one docked slot (see
// buildViewer()), this wires all of them together rather than independently.
//
// `selector` is a CSS selector for the button within the toolbar -- a
// <hoops-toolbar-*> tag name, or `[data-toolbar-key="..."]` for a custom one.
//
// Returns a setActive(panel) function so the caller can pick the initial
// open panel; pass null to close all of them.
function wireSidePanelButtons(toolbar, entries) {
  const wired = entries
    .map(({ selector, panel }) => ({ button: toolbar.querySelector(selector), panel }))
    .filter(({ button }) => button);

  function setActive(activePanel) {
    for (const { button, panel } of wired) {
      const open = panel === activePanel;
      panel.hidden = !open;
      button.color = open ? 'accent' : 'default';
    }
  }

  for (const { button, panel } of wired) {
    button.addEventListener('click', () => {
      setActive(panel.hidden ? panel : null);
    });
  }

  return setActive;
}

// Build a toolbar overlay (not slotted children -- see showPart()/showMap())
// with one button per tag name.
function buildToolbar(items) {
  const toolbar = document.createElement('div');
  toolbar.className = 'part-toolbar';
  for (const item of items) {
    toolbar.appendChild(
      typeof item === 'string' ? document.createElement(item) : buildCustomToolbarButton(item),
    );
  }
  return toolbar;
}

// A rail button with no official <hoops-toolbar-*> component (e.g. Cutting
// Planes): a plain <hoops-icon-button> wrapping a named <hoops-icon>, built
// to match the SDK buttons' own look (both are already globally registered
// custom elements, transitively imported by '@ts3d-hoops/web-viewer-
// components'). `data-toolbar-key` lets wireSidePanelButtons() find it the
// same way it finds a <hoops-toolbar-*> tag name.
function buildCustomToolbarButton({ key, icon, title }) {
  const button = document.createElement('hoops-icon-button');
  button.size = 'sm';
  button.title = title;
  button.dataset.toolbarKey = key;
  const iconEl = document.createElement('hoops-icon');
  iconEl.setAttribute('icon', icon);
  button.appendChild(iconEl);
  return button;
}

// hoops-toolbar-snapshot has no built-in click behavior of its own (its
// render() is just an icon button, confirmed by reading the compiled
// component -- no click handler, no dispatched event), so it does nothing
// unless wired up. WebViewer.takeSnapshot() is the SDK's own capture API;
// some reference apps pull in a third-party canvas-capture library for this
// instead, but that is unnecessary here.
function wireSnapshotButton(toolbar, viewer) {
  const button = toolbar.querySelector('hoops-toolbar-snapshot');
  if (!button) return;
  button.addEventListener('click', async () => {
    const hwv = viewer.viewer;
    if (!hwv) return;
    const image = await hwv.takeSnapshot();
    const link = document.createElement('a');
    link.href = image.src;
    link.download = `snapshot-${Date.now()}.png`;
    link.click();
  });
}

// <hoops-model-tree> is a purely passive component (confirmed by reading its
// source): it stays empty until .model is set, and clicking a node or its
// visibility icon does nothing until those events are bridged back to the
// viewer. This follows the wiring in the HOOPS Web Viewer web-components
// programming guide ("Using the Model Tree").
function setupModelTree(tree, viewer) {
  viewer.addEventListener('hwvModelStructureReady', () => {
    const hwv = viewer.viewer;
    if (hwv) tree.model = hwv.model;
  });
  tree.addEventListener('hoops-model-tree-node-click', (event) => {
    const hwv = viewer.viewer;
    if (hwv) hwv.selectionManager.selectNode(event.detail.nodeId, SelectionMode.Set);
  });
  viewer.addEventListener('hwvSelectionArray', (event) => {
    const hwv = viewer.viewer;
    const selectionEvents = (event.detail && event.detail.selectionEvents) || [];
    const nodeIds = selectionEvents.map((selectionEvent) =>
      selectionEvent.getSelection().getNodeId(),
    );
    // Per the SDK docs ("Model Tree"): selections from the 3D view always
    // resolve to leaf nodes, which the tree only renders once their ancestor
    // rows have been expanded (it lazily loads children on expand). Without
    // this, clicking a part in the viewer sets `tree.selected` to a node
    // whose row doesn't exist yet, so nothing visibly highlights.
    if (hwv && !event.detail.removed) {
      for (const nodeId of nodeIds) revealTreeNode(tree, hwv.model, nodeId);
    }
    tree.selected = nodeIds;
  });
  tree.addEventListener('hoops-model-tree-node-visibility-change', (event) => {
    const hwv = viewer.viewer;
    if (!hwv) return;
    const { nodeId, visibility } = event.detail;
    hwv.model.setNodesVisibility([nodeId], visibility);
    tree.updateNodeData(nodeId, { visible: visibility });
  });
}

// Expand <hoops-model-tree>'s internal <hoops-tree> along nodeId's ancestor
// chain so its row is loaded and can render (the tree only ever loads a
// node's children once its parent row has been expanded -- see resetTree()
// in the compiled hoops-model-tree.js, which does the same walk for the
// model's initial single-child chain). No-op if nodeId is already loaded, or
// is the root itself (empty path).
function revealTreeNode(tree, model, nodeId) {
  const treeElement = tree.treeElement;
  if (!treeElement) return;
  const path = [];
  let current = nodeId;
  for (;;) {
    const parent = model.getNodeParent(current);
    if (parent === null || parent === undefined) break;
    path.unshift(parent);
    current = parent;
  }
  if (!path.length) return;
  try {
    treeElement.expandPath(path);
  } catch {
    // Selection can race a model swap; a stale nodeId here should not break
    // the rest of the selection-sync handler.
  }
}

function bindFailure(viewer, message, eventName, text) {
  viewer.addEventListener(eventName, (event) => {
    clearInitTimer();
    const detail = detailText(event);
    setMessage(message, detail ? `${text} ${detail}` : text, true);
  });
}

function disposeViewer() {
  // Teardown is fully synchronous, so callers do not need to await it before
  // building the next viewer: removing currentViewerRoot runs
  // <hoops-web-viewer>'s disconnectedCallback, which drops its resize listener
  // and calls hwv.shutdown() -- declared `shutdown(): void` in
  // @ts3d-hoops/web-viewer, i.e. not a promise (verified in
  // node_modules/@ts3d-hoops/web-viewer-components/lib/hoops-web-viewer.js and
  // .../web-viewer/lib/WebViewer.d.ts). What it cannot clean up is a viewer
  // that has not produced an hwv yet, because `this.hwv` is still undefined at
  // that point; that is why overlapping rebuilds crashed the engine, and it is
  // fixed by never starting one (see requestLoad() in shape-map.js).
  clearInitTimer();
  const dying = currentViewer && currentViewer.viewer;
  // End any active comparison before the viewer goes away, so a stale
  // comparison state never bleeds into the next model (see showComparison).
  if (comparisonActive && currentViewer && currentViewer.viewer) {
    try {
      currentViewer.viewer.view.endComparison();
    } catch {
      // The engine may already be gone; nothing more to end.
    }
  }
  comparisonActive = false;
  mapModelState = null;
  if (resizeObserver) {
    resizeObserver.disconnect();
    resizeObserver = null;
  }
  if (currentViewerRoot) {
    currentViewerRoot.remove();
    currentViewerRoot = null;
  }
  // The console handles must not outlive the viewer they point at. `map` is
  // decided by identity rather than by which lineage is tearing down: the one
  // on screen is the map's exactly when window.hwvMap is it, and every other
  // case leaves window.hwvMap as it already was (null).
  clearViewerHandle({ map: typeof window !== 'undefined' && !!dying && window.hwvMap === dying });
  currentViewer = null;
}

function clearInitTimer() {
  if (initTimer) {
    window.clearTimeout(initTimer);
    initTimer = null;
  }
}

function detailText(event) {
  const detail = event && event.detail;
  if (!detail) return '';
  if (typeof detail === 'string') return detail;
  if (detail.message) return detail.message;
  return '';
}

function setMessage(el, text, isError) {
  el.textContent = text;
  el.classList.toggle('error', isError);
  el.hidden = false;
}

function clearMessage(el) {
  el.textContent = '';
  el.classList.remove('error');
  el.hidden = true;
}

// A single markup circle over the last picked map point's world position.
// Markup is a 2D overlay drawn after the 3D scene, so it always renders in
// front of the point cloud; re-projecting the stored world position on every
// draw() (rather than computing a screen position once) is what makes it
// track the point as the camera orbits/zooms/pans.
class PointMarker extends Markup.MarkupItem {
  constructor({
    radiusPx = MARKER_RADIUS_PX,
    strokeColor = MARKER_STROKE_COLOR,
    strokeWidthPx = MARKER_STROKE_WIDTH_PX,
  } = {}) {
    super();
    this._worldPosition = null;
    this._radiusPx = radiusPx;
    this._strokeColor = strokeColor;
    this._strokeWidthPx = strokeWidthPx;
  }

  setPosition(position) {
    this._worldPosition = position;
  }

  clear() {
    this._worldPosition = null;
  }

  draw(renderer, view) {
    if (!this._worldPosition) return;
    const projected = view.projectPoint(this._worldPosition);
    // Skip the draw if the projection is not a finite screen position. draw()
    // runs on every rendered frame, so one NaN projection otherwise means a
    // pair of "<circle> attribute cx: Expected length, NaN" errors per frame --
    // observed while orbiting a rescaled 3D-model map -- which buries every
    // other console message and makes the real problem impossible to see. The
    // marker simply disappears for those frames and comes back when the
    // projection is valid again.
    if (!projected || !isFinitePoint(projected)) return;
    const circle = new Markup.Shapes.Circle();
    circle.set(Point2.fromPoint3(projected), this._radiusPx);
    circle.setStrokeColor(this._strokeColor);
    circle.setStrokeWidth(this._strokeWidthPx);
    circle.setFillOpacity(0);
    renderer.drawCircle(circle);
  }
}

function isFinitePoint(p) {
  return Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z);
}

// Show the current index's Shape Map point cloud, loaded from `scsUrl`.
//
// The model swap is the same mechanism showPart() uses: a fresh
// <hoops-web-viewer> element is created (its `viewer` getter exposes the
// underlying WebViewer once ready), so switching between the map and a single
// part never leaves stale state behind. Returns a promise that resolves, once
// the point cloud is on screen, to a small controller
// Read the current map camera, or null if no map is on screen. Used to carry
// the user's viewpoint across a View toggle (point cloud <-> 3D models), which
// disposes the viewer and builds a fresh one: capture this before the toggle
// and pass it back in as showMap()'s initialCamera so the new map keeps the
// same viewpoint instead of snapping to the iso overview.
export function getMapCamera() {
  const hwv = currentViewer && currentViewer.viewer;
  if (!hwv) return null;
  try {
    return hwv.view.getCamera();
  } catch {
    return null;
  }
}

// ``{ setGroupVisible(nodeName, visible), showPointMarker(x, y, z), clearPointMarker(), focusPoint(x, y, z) }``
// the caller uses to show/hide a category node from the legend and to mark/
// unmark the picked point; it rejects if the map fails to load (e.g. the
// server returns 503 because the point-cloud tool is not installed) -- callers
// show that error without taking down the rest of the app.
//
// The SCS authors one node per category (cluster / tag / untagged), so a picked
// point is identified by its NODE plus its index WITHIN that node. The callbacks
// therefore report the node name, the in-node point id and the picked world
// position; the caller resolves that back to a part (see shape-map.js's
// resolvePoint()). Picking itself is set up by setupPointPicking() below.
//
//   onPointClick(nodeName, pointId, position)  a single click; all three are
//                                               null when the click hit no point
//   onPointActivate(nodeName, pointId, position)  the picked point was double-clicked
//   legendEl  the map legend element (an overlay outside the viewer, see
//             index.html); the toolbar's Legend button shows and hides it
export function showMap(scsUrl, { onPointClick, onPointActivate, skipPointPicking = false, overlayModels = false, initialCamera = null, legendEl = null } = {}) {
  const host = document.getElementById('viewer-host');
  const message = document.getElementById('viewer-message');

  ensureServices();
  disposeViewer();
  setMessage(message, 'Loading map\u2026', false);

  // See the comment in showPart() on why the toolbar needs this wrapper.
  const contextManager = document.createElement('hoops-web-viewer-context-manager');

  const viewer = document.createElement('hoops-web-viewer');
  viewer.enginePath = ENGINE_PATH;
  viewer.endpointUri = scsUrl;
  currentViewer = viewer;
  currentViewerRoot = contextManager;

  // node name -> [NodeId, ...]. A category may be authored as a named child node
  // plus a named body-instance node, so a name can map to several node ids;
  // hiding the whole category means hiding all of them.
  let nodeMap = new Map();
  // nodeName -> the visibility last APPLIED to that category's nodes. The
  // viewer walks a node's whole subtree on every setNodesVisibility call
  // (traverse -> _walkNode -> _applyVisibilities), which at 2,048 categories
  // was a two-second task for a click that usually changes a handful of them.
  // Comparing against this means only the categories that actually changed are
  // passed in, and an unchanged request costs no viewer call at all.
  //
  // A category absent from this map has never been set, so its state is
  // unknown and the next request for it is always applied. Dropped with
  // nodeMap whenever the cloud is rebuilt -- the node ids themselves are new
  // then, so nothing from the old map can be believed.
  let groupVisible = new Map();
  // The single on-screen point marker (see PointMarker). Created once the map
  // is ready; showPointMarker()/clearPointMarker() just move or hide it, so
  // there is never more than one.
  let pointMarker = null;
  // The hovered cluster's centre part (see showCentreMarker). Separate from
  // pointMarker on purpose: the selection ring must survive a legend hover.
  let centreMarker = null;

  const controller = {
    // Recolour whole categories of the point cloud at runtime, so a legend
    // change does not need the map rebuilt and re-fetched. `entries` is
    // [{ nodeName, rgb: [r, g, b] }]; names nodeMap does not know are skipped,
    // exactly as setGroupsVisible does -- the caller's list comes from the
    // legend, which may name categories this cloud was not authored with.
    //
    // Returns whether anything was coloured, rather than throwing: colour is
    // best-effort, and the caller decides whether to fall back to a rebuild.
    setMapGroupColors(entries) {
      const hwv = currentViewer && currentViewer.viewer;
      const model = hwv && hwv.model;
      if (!model || !Array.isArray(entries) || !entries.length) return false;
      let touched = 0;
      try {
        for (const entry of entries) {
          if (!entry) continue;
          const ids = nodeMap.get(entry.nodeName);
          if (!ids || !ids.length) continue;
          const [r, g, b] = entry.rgb || [128, 128, 128];
          const color = new Color(r, g, b);
          // setNodesMaterial is the current API; setNodesPointColor is the
          // deprecated one it replaced. Points are their own geometry here, so
          // it is pointColor that matters -- a face colour would leave a point
          // cloud looking untouched.
          if (typeof model.setNodesMaterial === 'function') {
            model.setNodesMaterial(ids, { pointColor: color });
          } else if (typeof model.setNodesPointColor === 'function') {
            model.setNodesPointColor(ids, color);
          } else {
            return false;
          }
          touched += ids.length;
        }
      } catch {
        return false;
      }
      return touched > 0;
    },
    // Show or hide every node authored for `nodeName` (a legend checkbox).
    setGroupVisible(nodeName, visible) {
      const hwv = currentViewer && currentViewer.viewer;
      const model = hwv && hwv.model;
      const ids = nodeMap.get(nodeName);
      if (!model || !ids || !ids.length) return;
      const want = Boolean(visible);
      // Already in that state: the viewer would walk the subtree to arrive
      // where it already is. applyGroupVisibility() calls this once per
      // category, so at 2,048 categories this is the difference between 2,048
      // subtree walks and none.
      if (groupVisible.get(nodeName) === want) return;
      try {
        model.setNodesVisibility(ids, want);
        groupVisible.set(nodeName, want);
      } catch {
        // Visibility is best-effort, and the state stays unrecorded so the
        // next request for this category is applied rather than skipped.
      }
    },
    // Set the visibility of a KNOWN set of categories in one go. `allNames` is
    // the range this is allowed to touch -- the categories the legend drew --
    // and every nodeMap entry outside it is left exactly as it is. That
    // restriction is the whole point: nodeMap also holds the ancestors of the
    // category nodes (see buildNodeNameMap), and hiding an ancestor hides its
    // subtree, so a blanket "hide everything not in visibleNames" would hide
    // the very category it was asked to show. Batched into at most two
    // setNodesVisibility calls, because one per category would be 64 calls per
    // hover at 64 clusters.
    //
    // Only the categories whose state CHANGES are passed to the viewer. The
    // call cost is in the subtree walk per node, not in the number of calls,
    // so isolating four categories out of 2,048 should hand over four
    // categories' nodes and not all of them. With nothing changed neither call
    // is made. Returns how many categories moved each way, for the log.
    setGroupsVisible(visibleNames, allNames) {
      const hwv = currentViewer && currentViewer.viewer;
      const model = hwv && hwv.model;
      if (!model || !allNames) return { shown: 0, hidden: 0, total: 0 };
      const show = [];
      const hide = [];
      const shownNames = [];
      const hiddenNames = [];
      let total = 0;
      for (const name of allNames) {
        const ids = nodeMap.get(name);
        if (!ids || !ids.length) continue;
        total += 1;
        const want = Boolean(visibleNames && visibleNames.has(name));
        if (groupVisible.get(name) === want) continue;
        if (want) {
          shownNames.push(name);
          for (const id of ids) show.push(id);
        } else {
          hiddenNames.push(name);
          for (const id of ids) hide.push(id);
        }
      }
      try {
        if (show.length) {
          model.setNodesVisibility(show, true);
          for (const name of shownNames) groupVisible.set(name, true);
        }
        if (hide.length) {
          model.setNodesVisibility(hide, false);
          for (const name of hiddenNames) groupVisible.set(name, false);
        }
      } catch {
        // Visibility is best-effort. Whatever was not recorded above is simply
        // applied again next time.
      }
      return { shown: shownNames.length, hidden: hiddenNames.length, total };
    },
    // Mark the picked point at its authoritative /map coordinates, resolved
    // by shape-map.js from the raw pick position this module reports.
    showPointMarker(x, y, z) {
      const hwv = currentViewer && currentViewer.viewer;
      if (!hwv || !pointMarker) return;
      // Refuse a non-finite position rather than storing it: it would project
      // to NaN on every frame from then on (see PointMarker.draw).
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;
      pointMarker.setPosition(new Point3(x, y, z));
      hwv.markupManager.refreshMarkup(hwv.view);
    },
    clearPointMarker() {
      const hwv = currentViewer && currentViewer.viewer;
      if (!hwv || !pointMarker) return;
      pointMarker.clear();
      hwv.markupManager.refreshMarkup(hwv.view);
    },
    // Ring the hovered cluster's centre part. Its own marker, so the selection
    // ring set by showPointMarker() stays where the user put it.
    showCentreMarker(x, y, z) {
      const hwv = currentViewer && currentViewer.viewer;
      if (!hwv || !centreMarker) return;
      // Same rule as showPointMarker: a non-finite position would project to
      // NaN on every frame from then on (see PointMarker.draw).
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;
      centreMarker.setPosition(new Point3(x, y, z));
      hwv.markupManager.refreshMarkup(hwv.view);
    },
    clearCentreMarker() {
      const hwv = currentViewer && currentViewer.viewer;
      if (!hwv || !centreMarker) return;
      centreMarker.clear();
      hwv.markupManager.refreshMarkup(hwv.view);
    },
    // Move the camera so the point at (x, y, z) is framed together with the
    // area around it. A padded box (FOCUS_HALF_EXTENT) is framed instead of the
    // bare point so neighbouring clusters stay visible -- "this part sits in the
    // right cluster" only reads if the cluster is on screen too. Translation of
    // the camera only; the point cloud's own coordinates are never touched.
    async focusPoint(x, y, z) {
      const hwv = currentViewer && currentViewer.viewer;
      if (!hwv) return;
      const pad = FOCUS_HALF_EXTENT;
      const min = new Point3(x - pad, y - pad, z - pad);
      const max = new Point3(x + pad, y + pad, z + pad);
      try {
        await hwv.view.fitBounding(new Box(min, max));
      } catch {
        // Framing is best-effort; the marker is still shown at the point.
      }
    },

    // Frame an axis-aligned box given in map coordinates. The box comes from
    // shape-map.js, which owns the point positions and so is the only place
    // that can know where a cluster is; this end just moves the camera. Like
    // focusPoint() above, the point cloud's own coordinates are never touched.
    async focusBounding(min, max) {
      const hwv = currentViewer && currentViewer.viewer;
      if (!hwv || !min || !max) return;
      // A non-finite bound would put the camera somewhere it cannot come back
      // from, so refuse the whole call rather than fit half of it.
      if (![...min, ...max].every((v) => Number.isFinite(v))) return;
      try {
        await hwv.view.fitBounding(
          new Box(
            new Point3(min[0], min[1], min[2]),
            new Point3(max[0], max[1], max[2]),
          ),
        );
      } catch {
        // Framing is best-effort; the preview is still shown.
      }
    },
  };

  const ready = new Promise((resolve, reject) => {
    initTimer = window.setTimeout(() => {
      initTimer = null;
      const text =
        `The 3D engine did not start. Check that the viewer engine files are ` +
        `present under ${ENGINE_PATH} and served as application/wasm.`;
      setMessage(message, text, true);
      reject(new Error(text));
    }, ENGINE_INIT_TIMEOUT_MS);

    viewer.addEventListener('hwvReady', (event) => {
      publishViewerHandle(event.detail, { map: true });
      bindResize(event.detail, host, viewer);
    });

    viewer.addEventListener('hwvModelStructureReady', async () => {
      clearInitTimer();
      clearMessage(message);
      const hwv = viewer.viewer;
      try {
        // Smaller points when 3D models will be overlaid, so the cloud does not
        // bury the parts sitting on it (see POINT_SIZE_MODELS). initialCamera,
        // when given, restores the viewpoint from before a View toggle.
        await applyPointCloudView(hwv, overlayModels, initialCamera);
      } catch {
        // Display tuning is best-effort; the map is still usable without it.
      }
      try {
        await addGridPlane(hwv);
      } catch {
        // Decorative only; the map is still usable without the grid.
      }
      try {
        nodeMap = buildNodeNameMap(hwv.model);
      } catch {
        nodeMap = new Map();
      }
      // New node ids, so nothing remembered about the old ones holds.
      groupVisible = new Map();
      try {
        pointMarker = new PointMarker();
        hwv.markupManager.registerMarkup(pointMarker, hwv.view);
      } catch {
        pointMarker = null;
      }
      try {
        centreMarker = new PointMarker({
          radiusPx: CENTRE_MARKER_RADIUS_PX,
          strokeColor: CENTRE_MARKER_STROKE_COLOR,
          strokeWidthPx: CENTRE_MARKER_STROKE_WIDTH_PX,
        });
        hwv.markupManager.registerMarkup(centreMarker, hwv.view);
      } catch {
        centreMarker = null;
      }
      try {
        // In 3D-models mode the caller overlays real models and wires Face
        // picking itself, so leave Point picking off to avoid two handlers.
        if (!skipPointPicking) setupPointPicking(hwv, onPointClick, onPointActivate);
      } catch {
        // If picking cannot be wired up, clicks simply do nothing; the rest
        // of the map (legend, navigation) still works.
      }
      resolve(controller);
    });

    const fail = (text) => (event) => {
      clearInitTimer();
      const detail = detailText(event);
      const full = detail ? `${text} ${detail}` : text;
      setMessage(message, full, true);
      reject(new Error(full));
    };
    // The viewer fetches the map .scs itself, so a load failure carries no
    // reason from the server -- e.g. the 503 body sent when the point-cloud
    // authoring tool is missing under bin/. Only when the load fails do we
    // re-request the same URL to recover that message; a successful load adds no
    // extra round trip. If the re-request unexpectedly succeeds, the .scs was
    // served but the engine could not use it, so keep the generic message.
    const failMapLoad = async (event) => {
      clearInitTimer();
      let text = 'Failed to load the map.';
      let fromServer = false;
      try {
        const response = await fetch(scsUrl);
        if (!response.ok) {
          fromServer = true;
          const body = await response.text();
          let reason = '';
          try {
            reason = (JSON.parse(body).detail || '').toString().trim();
          } catch {
            reason = body.trim();
          }
          text = reason || `Failed to load the map. (HTTP ${response.status})`;
        }
      } catch {
        // Could not re-request the map; fall back to the generic message.
      }
      if (!fromServer) {
        const detail = detailText(event);
        if (detail) text = `${text} ${detail}`;
      }
      setMessage(message, text, true);
      reject(new Error(text));
    };
    viewer.addEventListener('hwvModelLoadFailure', failMapLoad);
    viewer.addEventListener('hwvMissingModel', fail('The map could not be found.'));
    viewer.addEventListener('hwvXHRonerror', fail('Network error while fetching the map.'));
    viewer.addEventListener('hwvWebGlContextLost', fail('The WebGL context was lost.'));
    viewer.addEventListener('hwvTimeout', fail('The viewer session timed out.'));
  });

  const toolbar = buildToolbar(MAP_TOOLBAR_BUTTONS);
  wireMapHomeButton(toolbar);
  wireLegendButton(toolbar, legendEl);

  contextManager.append(viewer, toolbar);
  host.appendChild(contextManager);
  return ready;
}

// Wire the map's own Home button: put the viewpoint back and nothing else. See
// MAP_TOOLBAR_BUTTONS for why the map does not use the stock
// <hoops-toolbar-home>, which resets visibility and transforms too and so
// silently undoes the legend's filtering.
//
// This applies the SAME call the Camera menu's Iso View button makes, with the
// same defaults, so the two buttons land on exactly the same frame. It is used
// in preference to view.resetCamera() because the two can drift apart: the
// initial camera is captured once, when the map is first revealed, while this
// is recomputed from the model bounding every time -- and because having Home
// and Iso View visibly disagree is confusing in a toolbar that offers both.
function wireMapHomeButton(toolbar) {
  const button = toolbar.querySelector('[data-toolbar-key="map-home"]');
  if (!button) return;
  button.addEventListener('click', () => {
    const hwv = currentViewer && currentViewer.viewer;
    if (!hwv) return;
    Promise.resolve(hwv.view.setViewOrientation(ViewOrientation.Iso)).catch(() => {
      // Returning to the initial viewpoint is best-effort; a failure leaves the
      // camera where it is rather than disturbing anything else.
    });
  });
}

// Wire the map toolbar's Legend button to the legend overlay. The legend is not
// a child of the viewer (it is positioned over the viewer pane by index.html),
// so it cannot join the docked side-panel rail that wireSidePanelButtons()
// drives; it gets the same convention instead -- click toggles, and the
// button's `color` ('accent' when open, 'default' when closed) reports the
// current state, exactly like the model-tree and Cutting Planes buttons.
function wireLegendButton(toolbar, legendEl) {
  const button = toolbar.querySelector('[data-toolbar-key="map-legend"]');
  if (!button || !legendEl) return;
  const sync = () => {
    button.color = legendEl.hidden ? 'default' : 'accent';
  };
  button.addEventListener('click', () => {
    legendEl.hidden = !legendEl.hidden;
    sync();
  });
  // The map may be rebuilt (a colour, kind or view change) while the legend is
  // hidden, so start from whatever state the legend is actually in rather than
  // assuming it is open.
  sync();
}

// Target on-screen size of each overlaid model, in the map's normalized [-1, 1]
// coordinates: every model is scaled so its bounding-box diagonal is about this
// long regardless of the part's real millimetre size, so a tiny washer and a
// large gear are both legible at the same map scale. The Part slider multiplies
// this. The grid divides the [-1, 1] cube into 0.1 cells, so ~0.12 spans just
// over one cell -- visible without burying its neighbours.
//
// This is now the FALLBACK, used when the caller does not measure the layout.
// It assumes the points are spaced roughly as the PCA projection spaces them;
// a layout that packs them more tightly (UMAP does, by an order of magnitude)
// buries every model in its neighbours at this size. addMapModels() takes a
// measured `targetDiag` instead -- see shape-map.js's measureLayoutSpacing().
const MODEL_TARGET_DIAG = 0.12;

// How many part subtrees to load at once when overlaying models on the map.
// Loading is otherwise per part (createNode -> attach-invisibly -> measure ->
// place -> reveal), and doing that strictly sequentially takes ~18 s for 800
// parts, which is unusable for the demo. Measured on tmcad (10,230 parts,
// 2026-09-11): 4 concurrent subtree loads finished 400 parts in 1.72 s vs
// 8.23 s sequential (~4.8x faster) and 800 parts in ~4 s, all rendering
// correctly. The old "concurrent subtree loads into one scene are not
// reliable" caution did not reproduce on this SCS-file path; 4 is kept modest
// so the scene is never flooded with in-flight loads.
const MAP_MODEL_CONCURRENCY = 4;

// Per-viewer state for the models overlaid on the point-cloud map (see
// addMapModels). Reset by disposeViewer() when the scene goes away.
let mapModelState = null;

// Overlay real 3D models onto the CURRENT point-cloud map, at their point
// positions, WITHOUT disturbing the point cloud, the grid, or the camera. Call
// this after showMap() has resolved. Each part's SCS is loaded as its own
// container node under the scene root, scaled so its diagonal is about
// MODEL_TARGET_DIAG in the map's [-1, 1] space (times `partScale`), unless the
// caller passes a `targetDiag` measured from the layout, and
// translated so that scaled centre lands on its map point. The camera is left
// exactly where the point cloud put it -- the models appear in place.
//
//   parts  [{ fileId, filename, scsUrl, x, y, z, rgb? }] -- x/y/z are the
//          normalized map coordinates; rgb (0-255) colours the model.
//   onModelClick(fileId) / onModelActivate(fileId)  model pick callbacks;
//          onModelClick(null) reports a click on empty space (close the panel)
//   onPointClick / onPointActivate  cloud-point pick callbacks (same signature
//          as showMap's), so a click on a part that has no model yet is still
//          reported and can be loaded on demand
//   partScale  initial size multiplier
//   targetDiag  the on-screen diagonal each model should be drawn at, in map
//          units. Omitted falls back to MODEL_TARGET_DIAG. Ignored when
//          appending, which keeps the size the models on screen already use.
//   append     add to the models already on the map instead of replacing them
//              (used when the legend changes which categories are shown). The
//              existing placement state and pick wiring are kept, so only the
//              new parts are loaded.
export async function addMapModels(
  parts,
  {
    onModelClick,
    onModelActivate,
    onPointClick,
    onPointActivate,
    partScale = 1,
    targetDiag,
    trueScale = false,
    append = false,
  } = {},
) {
  const hwv = currentViewer && currentViewer.viewer;
  if (!hwv) return;
  const model = hwv.model;
  const message = document.getElementById('viewer-message');
  const appending = Boolean(append) && Boolean(mapModelState);
  const placed = appending ? mapModelState.placed : new Map(); // fileId -> { node, point, center, diag, baseScale, groupNode }
  const nodeToFile = appending ? mapModelState.nodeToFile : new Map(); // container NodeId -> fileId
  const scale = appending
    ? mapModelState.scale
    : Number(partScale) > 0
      ? Number(partScale)
      : 1;
  // Appending keeps the size the models on screen are already drawn at; only a
  // fresh overlay takes it from the caller. Mirrors how `scale` is handled.
  const effectiveTargetDiag = appending
    ? mapModelState.targetDiag
    : Number(targetDiag) > 0
      ? Number(targetDiag)
      : MODEL_TARGET_DIAG;
  // Appending keeps the mode the models on screen are already drawn in; only a
  // fresh overlay takes it from the caller.
  if (!appending) {
    mapModelState = {
      placed, nodeToFile, scale, trueScale: Boolean(trueScale), refDiag: 0,
      targetDiag: effectiveTargetDiag,
    };
  }

  const root = model.getAbsoluteRootNode();
  const displayable = (parts || []).filter((p) => p && p.scsUrl && !placed.has(p.fileId));
  let loaded = 0;
  // Record a finished part in the shared state. Runs on the pool's callers, so
  // Map writes stay on the single JS thread; picking/legend/slider all read
  // these two Maps, so they must be populated as each part lands.
  const commit = (result) => {
    if (!result) return; // a missing/unreadable SCS just drops that one model
    placed.set(result.fileId, result.st);
    nodeToFile.set(result.st.node, result.fileId);
    loaded += 1;
    if (message && !message.hidden) {
      setMessage(message, `Loading models ${loaded}/${displayable.length}\u2026`, false);
    }
  };
  // Bounded parallel pool: at most MAP_MODEL_CONCURRENCY subtree loads are in
  // flight at once. Each part is one loadOneMapModel call, so the per-part
  // attach-invisibly -> measure -> place -> reveal order (the cf1bda1 flicker
  // fix) is preserved intact -- only the NUMBER of parts loading concurrently
  // changes. Workers pull from a shared cursor so faster parts do not wait on
  // slower ones.
  let next = 0;
  const runWorker = async () => {
    while (next < displayable.length) {
      const part = displayable[next++];
      commit(await loadOneMapModel(model, root, part, scale, effectiveTargetDiag));
    }
  };
  const workers = Math.max(1, Math.min(MAP_MODEL_CONCURRENCY, displayable.length));
  await Promise.all(Array.from({ length: workers }, () => runWorker()));

  clearMessage(message);
  // The common-scale reference is the median of the models on screen, so it
  // moves as models arrive. Recomputing it per part would make every model
  // jump each time a bigger or smaller one landed, so it is taken ONCE, here,
  // when the batch is complete -- the models load at whatever reference was in
  // force and settle together in a single re-place at the end.
  if (mapModelState && mapModelState.trueScale) {
    recomputeMapModelRefDiag();
    applyModelMatrices(model, mapModelState.placed, mapModelState.scale).catch(() => {});
  }
  // Matrices were applied per part during loading, so there is no batch apply
  // here; the Part slider still calls applyModelMatrices() to re-place every
  // model at a new scale.
  // When appending, picking is already wired to these very Maps (the closure
  // holds nodeToFile itself), so the parts just added are pickable without
  // touching the operators again -- re-wiring would stack a second handler.
  if (!appending) {
    try {
      setupModelPicking(hwv, nodeToFile, { onModelClick, onModelActivate, onPointClick, onPointActivate });
    } catch {
      // If picking cannot be wired up, clicks simply do nothing.
    }
  }
  // No fitWorld here: the overlay must keep the point cloud's own camera.
}

// Unload overlaid models: delete their nodes and forget them, so a category the
// legend switched off frees its geometry instead of merely being hidden. The
// point cloud, the grid and the camera are untouched.
export async function removeMapModels(fileIds) {
  const hwv = currentViewer && currentViewer.viewer;
  if (!hwv || !mapModelState || !fileIds || fileIds.length === 0) return;
  const model = hwv.model;
  for (const fileId of fileIds) {
    const st = mapModelState.placed.get(fileId);
    if (!st) continue;
    mapModelState.placed.delete(fileId);
    mapModelState.nodeToFile.delete(st.node);
    try {
      await model.deleteNode(st.node);
    } catch {
      // A node that cannot be deleted is already gone from our bookkeeping, so
      // it can no longer be picked or re-placed; leave it at that.
    }
  }
  // Same as in addMapModels: the median moved because the population did, and
  // it is taken once for the whole batch so the survivors re-place together.
  if (mapModelState.trueScale) {
    recomputeMapModelRefDiag();
    applyModelMatrices(model, mapModelState.placed, mapModelState.scale).catch(() => {});
  }
}

// The one divisor the common-scale mode uses. The MEDIAN diagonal of the
// models on screen, so a typical part still lands near the overlay's target
// diagonal and
// the Scale slider can take it from there; the biggest parts overflow their
// cell and the smallest shrink, which is the point of the mode. Deliberately
// not the max: with this index's size range that would make most of the map
// invisible.
function recomputeMapModelRefDiag() {
  if (!mapModelState) return;
  const diags = [];
  for (const st of mapModelState.placed.values()) {
    const d = Number(st.diag);
    if (Number.isFinite(d) && d > 0) diags.push(d);
  }
  if (diags.length === 0) {
    mapModelState.refDiag = 0;
    return;
  }
  diags.sort((a, b) => a - b);
  mapModelState.refDiag = diags[Math.floor(diags.length / 2)];
}

// Switch the overlaid models between "each to the same on-screen size" (the
// default) and "one common scale, real ratios preserved". Purely a client-side
// re-placement: no reload, no server call.
export function setMapModelTrueScale(on) {
  const hwv = currentViewer && currentViewer.viewer;
  if (!hwv || !mapModelState) return;
  mapModelState.trueScale = Boolean(on);
  recomputeMapModelRefDiag();
  applyModelMatrices(hwv.model, mapModelState.placed, mapModelState.scale).catch(() => {});
}

// Load ONE part's SCS as a placed, visible model under `root`, following the
// cf1bda1 order so the raw un-placed geometry is never drawn: attach it
// invisibly, measure its box, apply its final matrix, then reveal it. Returns
// { fileId, st } for the caller to record in the shared state, or null if the
// SCS was missing/unreadable. This is the unit of work run by addMapModels'
// bounded parallel pool.
//
// `targetDiag` is the on-screen diagonal to draw this model at, measured from
// the layout by the caller (see addMapModels); it is a parameter rather than
// the module constant because the right size depends on how tightly the
// current projection packs its points.
async function loadOneMapModel(model, root, part, scale, targetDiag) {
  const effectiveTargetDiag = Number(targetDiag) > 0 ? Number(targetDiag) : MODEL_TARGET_DIAG;
  const node = model.createNode(root, `model_${part.fileId}`);
  let roots;
  try {
    // Attach the geometry INVISIBLY. Freshly loaded geometry is drawn the
    // instant it lands, and no matrix has been applied yet, so it would show
    // at its real CAD millimetre size near the scene origin -- a part ~1000x
    // the map's MODEL_TARGET_DIAG, briefly filling the viewport. Loading
    // invisibly means that raw, un-placed state is never drawn; we measure it,
    // place it, and reveal it below, so a model only ever appears already
    // scaled and positioned. Several of these run at once (see the pool in
    // addMapModels); concurrent SCS-file subtree loads are reliable on this
    // path (measured 4-wide on tmcad), so the ordering within one part is what
    // matters here, not serializing across parts.
    const cfg = new LoadSubtreeConfig();
    cfg.attachInvisibly = true;
    roots = await model.loadSubtreeFromScsFile(node, part.scsUrl, cfg);
  } catch {
    return null; // a missing/unreadable SCS just drops that one model
  }
  if (!roots || !roots.length) return null;
  let center = { x: 0, y: 0, z: 0 };
  let diag = 1;
  try {
    // getNodesBounding returns the world-space geometric box and, unlike
    // getNodeRealBounding (which takes a view "for determining visibility"),
    // does not depend on the node being visible -- so it is valid for the
    // attach-invisibly geometry above.
    const box = await model.getNodesBounding([node]);
    const c = box.center();
    center = { x: c.x, y: c.y, z: c.z };
    diag =
      Math.sqrt(
        (box.max.x - box.min.x) ** 2 +
          (box.max.y - box.min.y) ** 2 +
          (box.max.z - box.min.z) ** 2,
      ) || 1;
  } catch {
    // Keep defaults; a model with no readable box still shows.
  }
  const st = {
    node,
    point: { x: part.x, y: part.y, z: part.z },
    center,
    // The part's own bounding-box diagonal, kept so the common-scale mode
    // can divide every model by ONE reference instead of by its own size.
    diag,
    baseScale: effectiveTargetDiag / diag,
    groupNode: part.groupNode || null,
    // The visibility last APPLIED to this model, so a later bulk change can
    // pass the viewer only what actually moves. It starts false because the
    // node is attached invisibly, and becomes true below only if the reveal
    // succeeds: a model whose reveal failed must look hidden here, or the
    // next change would skip it and leave it hidden with nothing to restore
    // it. Lives on the placed entry itself, which is how it can never fall
    // out of step with addMapModels/removeMapModels -- the flag is created
    // with the model and deleted with it.
    visible: false,
  };
  if (part.rgb && part.rgb.length === 3) {
    try {
      await model.setNodesFaceColor([node], new Color(part.rgb[0], part.rgb[1], part.rgb[2]));
    } catch {
      // Colour is best-effort.
    }
  }
  // Place this model NOW (before it is revealed): it must carry its final
  // scale and position before it is shown, so it never flashes at raw size.
  try {
    await applyModelMatrix(model, st, scale);
  } catch {
    // A model that could not be placed still gets revealed below rather than
    // being left stuck invisible.
  }
  // Reveal it, now that it sits at the right size and place. Pass false for
  // initiallyHiddenStayHidden so the attach-invisibly geometry actually shows;
  // once loading finishes every model is visible, so the legend's category
  // checkboxes (setMapModelGroupVisible) can hide/show them exactly as before.
  try {
    await model.setNodesVisibility([node], true, false);
    st.visible = true;
  } catch {
    // Visibility is best-effort; a model that cannot be shown is simply absent.
  }
  return { fileId: part.fileId, st };
}

// Add ONE model to an already-built 3D-model map, on demand (used when the user
// clicks a cloud point whose part has no model loaded yet). Reuses the exact
// per-part path loadOneMapModel/applyModelMatrix, so the new model follows the
// same attach-invisibly -> measure -> matrix -> reveal order and lands at the
// current Part-slider scale. Recording it in mapModelState.placed/nodeToFile is
// what makes the legend, the slider and picking apply to it automatically, just
// like the models loaded up front. Returns true if the model is now present.
export async function addMapModel(part) {
  const hwv = currentViewer && currentViewer.viewer;
  if (!hwv || !mapModelState || !part || !part.scsUrl) return false;
  // Already loaded (or being tracked): nothing to do.
  if (mapModelState.placed.has(part.fileId)) return true;
  const model = hwv.model;
  const root = model.getAbsoluteRootNode();
  let result;
  try {
    result = await loadOneMapModel(
      model, root, part, mapModelState.scale, mapModelState.targetDiag,
    );
  } catch {
    return false;
  }
  if (!result) return false;
  mapModelState.placed.set(result.fileId, result.st);
  mapModelState.nodeToFile.set(result.st.node, result.fileId);
  return true;
}

// Rescale the overlaid models (the Part slider). The camera is not touched.
export function setMapModelScale(scale) {
  const hwv = currentViewer && currentViewer.viewer;
  if (!hwv || !mapModelState) return;
  const s = Number(scale);
  mapModelState.scale = Number.isFinite(s) && s > 0 ? s : mapModelState.scale;
  applyModelMatrices(hwv.model, mapModelState.placed, mapModelState.scale).catch(() => {});
}

// Apply one visibility to a list of placed entries, and record it on them so
// the next request can tell what has to move. Nothing is recorded if the call
// throws: an unrecorded entry is applied again next time, which is the safe
// direction to fail in -- the unsafe one is a model that is hidden on screen
// but remembered as visible, because nothing would ever show it again.
function applyModelVisibility(entries, visible) {
  if (!entries.length) return;
  const hwv = currentViewer && currentViewer.viewer;
  if (!hwv) return;
  try {
    hwv.model.setNodesVisibility(entries.map((st) => st.node), visible);
    for (const st of entries) st.visible = visible;
  } catch {
    // Visibility is best-effort.
  }
}

// Show or hide every overlaid model that belongs to a legend category, so a
// legend checkbox hides the category's point-cloud nodes AND its 3D models
// together. `groupNode` is the SCS node name the model was tagged with (the
// same key the cloud legend uses).
//
// For ONE category only -- a legend checkbox, a single row's menu. Never call
// this in a loop over categories: each call walks the whole of placed (up to
// MAX_MODELS entries), so a loop over 2,048 categories is millions of
// iterations. Use setMapModelGroupsVisible() below, which does the same work
// in one pass.
export function setMapModelGroupVisible(groupNode, visible) {
  const hwv = currentViewer && currentViewer.viewer;
  if (!hwv || !mapModelState) return;
  const want = Boolean(visible);
  const changed = [];
  for (const st of mapModelState.placed.values()) {
    if (st.groupNode === groupNode && st.visible !== want) changed.push(st);
  }
  applyModelVisibility(changed, want);
}

// Set the visibility of EVERY overlaid model from one set of visible category
// names, in a single pass: models whose groupNode is in `visibleGroupNodes`
// are shown and all the rest are hidden, with at most two setNodesVisibility
// calls however many categories there are. Same shape as peekMapModelGroups()
// below and for the same reason -- one pass, then two visibility calls, never
// one per category.
//
// Only the models whose state CHANGES are handed over. The viewer walks each
// node's subtree on every call (traverse -> _walkNode -> _applyVisibilities),
// so passing all 800 placed models to switch 12 of them cost two seconds of
// scripting for work that was already done. With nothing to change, neither
// call is made.
//
// Returns how many models were shown and hidden and how many are placed, which
// the caller logs. This only toggles models that are ALREADY loaded; it never
// loads or unloads.
export function setMapModelGroupsVisible(visibleGroupNodes) {
  const hwv = currentViewer && currentViewer.viewer;
  if (!hwv || !mapModelState || !visibleGroupNodes) return { shown: 0, hidden: 0, placed: 0 };
  const show = [];
  const hide = [];
  for (const st of mapModelState.placed.values()) {
    const want = visibleGroupNodes.has(st.groupNode);
    if (st.visible === want) continue;
    (want ? show : hide).push(st);
  }
  applyModelVisibility(show, true);
  applyModelVisibility(hide, false);
  return { shown: show.length, hidden: hide.length, placed: mapModelState.placed.size };
}

// Hover preview for the overlaid models: show only the models whose category is
// in `groupNodes` (a Set of SCS node names) and hide the rest. Safe to drive
// from mapModelState.placed, unlike the point cloud's nodeMap, because every
// st.node is a placed model's own container -- there are no ancestors in here
// to hide by accident. One pass, then two visibility calls -- never one per
// category, which would be 64 calls per hover at 64 clusters, and 256 when the
// hovered thing is the root of Ward's tree. Differenced like the two above,
// which is what keeps a hover from re-stating the visibility of every model on
// screen. This only toggles the visibility of models that are ALREADY loaded;
// it must not trigger a model refresh, or every hover would unload and reload
// hundreds of parts. The preview is ended by shape-map.js's
// applyGroupVisibility(), which restores both the cloud and the models from the
// checkbox state -- and which sees the preview's own state here, so the restore
// moves back exactly what the preview moved.
export function peekMapModelGroups(groupNodes) {
  const hwv = currentViewer && currentViewer.viewer;
  if (!hwv || !mapModelState || !groupNodes) return;
  const show = [];
  const hide = [];
  for (const st of mapModelState.placed.values()) {
    const want = groupNodes.has(st.groupNode);
    if (st.visible === want) continue;
    (want ? show : hide).push(st);
  }
  applyModelVisibility(show, true);
  applyModelVisibility(hide, false);
}

// Turn the node highlight on or off for one overlaid model, used while hovering
// a legend row to point out that cluster's centre part. Returns whether the
// model was loaded to highlight at all, so the caller can fall back to the
// point-cloud marker when it was not.
//
// Deliberately NOT selectionManager: highlighting is independent of selection,
// so a hover leaves whatever the user has selected alone. highlightMapModel()
// below does use selection, because its job is to make a tile selection look
// exactly like a click on the model -- a different purpose.
export function setMapModelHighlighted(fileId, on) {
  const hwv = currentViewer && currentViewer.viewer;
  if (!hwv || !mapModelState) return false;
  const st = mapModelState.placed.get(fileId);
  if (!st) return false;
  // Returns a promise; nothing depends on when it settles, and a failure here
  // only means the hover is not illustrated.
  Promise.resolve(hwv.model.setNodesHighlighted([st.node], Boolean(on))).catch(() => {});
  return true;
}

// Repaint overlaid models in bands. Each band is { rgb: [r, g, b], fileIds },
// and every model in a band is coloured by ONE setNodesFaceColor call, so a
// recolour costs as many viewer calls as there are bands rather than one per
// model. File ids that are not loaded are skipped silently -- the map shows a
// budgeted subset of the parts, so that is the normal case, not an error.
// Returns how many models were actually coloured.
export function setMapModelColors(bands) {
  const hwv = currentViewer && currentViewer.viewer;
  if (!hwv || !mapModelState || !Array.isArray(bands)) return 0;
  let colored = 0;
  for (const band of bands) {
    if (!band || !Array.isArray(band.fileIds) || !Array.isArray(band.rgb)) continue;
    const nodes = [];
    for (const fileId of band.fileIds) {
      const st = mapModelState.placed.get(fileId);
      if (st) nodes.push(st.node);
    }
    if (nodes.length === 0) continue;
    const color = new Color(band.rgb[0], band.rgb[1], band.rgb[2]);
    // Returns a promise; colour is best-effort, exactly as in loadOneMapModel.
    Promise.resolve(hwv.model.setNodesFaceColor(nodes, color)).catch(() => {});
    colored += nodes.length;
  }
  return colored;
}

// Highlight one overlaid model WITHOUT moving the camera, so selecting a tile
// can point its part out on the map without yanking the view away from
// wherever the user is looking. Uses the viewer's own node selection, which is
// the same highlight a click on the model produces (see setupModelPicking), so
// a selected tile and a clicked model look identical. Returns whether the model
// was there to highlight.
export function highlightMapModel(fileId) {
  const hwv = currentViewer && currentViewer.viewer;
  if (!hwv || !mapModelState) return false;
  const st = mapModelState.placed.get(fileId);
  if (!st) return false;
  try {
    hwv.selectionManager.selectNode(st.node, SelectionMode.Set);
  } catch {
    return false;
  }
  return true;
}

// Drop the overlaid-model highlight (nothing selected on the map).
export function clearMapModelHighlight() {
  const hwv = currentViewer && currentViewer.viewer;
  if (!hwv) return;
  try {
    hwv.selectionManager.clear();
  } catch {
    // Best-effort: an already-empty selection is not an error worth reporting.
  }
}

// Which divisor a model's geometry is scaled by.
function modelBaseScale(st) {
  if (mapModelState && mapModelState.trueScale && mapModelState.refDiag > 0) {
    // One divisor for every model, so the drawn sizes keep their real
    // ratios.
    return (mapModelState.targetDiag || MODEL_TARGET_DIAG) / mapModelState.refDiag;
  }
  // Each model to the same on-screen size -- the default, and what the
  // embedding sees.
  return st.baseScale;
}

// Build the placement matrix for one overlaid model: scale it to the overlay's
// target diagonal times the current multiplier about its own centre, then
// translate so that scaled centre lands on its map point. World = scale*v + t,
// so t = point - scale*centre.
function modelMatrixFor(st, scaleMultiplier) {
  const scale = modelBaseScale(st) * scaleMultiplier;
  const m = new Matrix();
  m.setScaleComponent(scale, scale, scale);
  m.setTranslationComponent(
    st.point.x - scale * st.center.x,
    st.point.y - scale * st.center.y,
    st.point.z - scale * st.center.z,
  );
  return m;
}

// Apply one model's placement matrix. Split out from applyModelMatrices so a
// model can be positioned the instant it is loaded (before it is revealed),
// while the Part slider still re-applies every matrix in one batch below.
//
// setAsInitial is true because WebViewer.reset() (documented as "Reset the
// camera, visibility, and transforms to their default state") restores every
// node's INITIAL matrix. Without this, such a reset dropped each overlaid model
// back to the identity matrix: real CAD size at the scene origin, burying the
// [-1, 1] point cloud. The map's own Home button no longer calls reset() (see
// MAP_TOOLBAR_BUTTONS), so this is no longer what protects Home -- it is kept
// because it costs nothing and it makes the placement the models' real initial
// state, so ANY future path that resets the viewer leaves the map intact rather
// than exploding it. Passing true on every apply, including the Part slider's
// re-place, means "wherever the models are now is where a reset puts them
// back", which keeps the slider on screen in step with the models.
function applyModelMatrix(model, st, scaleMultiplier) {
  return model.setNodeMatrix(st.node, modelMatrixFor(st, scaleMultiplier), true);
}

// Place every overlaid model at its map point (the Part slider path). The
// formula is identical to the per-part placement done during loading; this is
// the batch entry point setMapModelScale() re-runs when the slider moves.
function applyModelMatrices(model, placed, scaleMultiplier) {
  const tasks = [];
  for (const st of placed.values()) {
    tasks.push(applyModelMatrix(model, st, scaleMultiplier));
  }
  return Promise.all(tasks);
}

// Wire up picking for the 3D-model map. A single click previews what was hit, a
// double click opens it; both work on either an overlaid MODEL (a face pick,
// resolved up to its container node -> file id) or a bare POINT of the cloud
// underneath (so the majority of parts, which have no model yet, are still
// clickable and can be loaded on demand -- see addMapModel). Uses our own click
// detection (a pointerdown/up pair that did not move) and one combined
// face+point pick. The picked model gets the viewer's own node highlight, but
// its faces and edges are NOT reddened (see below), so the whole part lights up
// as one instead of a single face.
function setupModelPicking(hwv, nodeToFile, { onModelClick, onModelActivate, onPointClick, onPointActivate } = {}) {
  const view = hwv.view;
  const container = view.domElements.getCanvasContainerElement();
  // Face | Point so a click resolves to a model face when one is under the
  // cursor, or to a cloud point when the click lands on the bare cloud.
  const pickConfig = new PickConfig(SelectionMask.Face | SelectionMask.Point);

  // Highlight the whole selected node, but not the individual face/edge that was
  // picked: the default reddens the single clicked face (and outlines edges),
  // which on these parts looks like a defect rather than a selection. Turning
  // off face/line element highlighting while leaving node highlighting on makes
  // the entire part light up as one. Node picking still returns the face's node,
  // so the mapping to a file id is unaffected.
  try {
    const sm = hwv.selectionManager;
    sm.setHighlightFaceElementSelection(false);
    sm.setHighlightLineElementSelection(false);
  } catch {
    // Best-effort; if the engine does not expose these the only cost is the
    // default per-face highlight.
  }

  // The built-in SelectionOperator fits the view to the model bounding box on
  // a double click. On the map that reframes the whole overlay -- and, because
  // it runs before our own async pick resolves, it moves the model out from
  // under the cursor, so the first double click only fitWorld's and a second is
  // needed to actually open the part. Disable it so a double click opens the
  // part on the first try without disturbing the map camera.
  try {
    const selectOp = view.operatorManager.getOperator(OperatorId.Select);
    if (selectOp && selectOp.setDoubleClickFitWorldEnabled) {
      selectOp.setDoubleClickFitWorldEnabled(false);
    }
  } catch {
    // Older engines may not expose this; the double click still works, it just
    // may need a second try after the fitWorld.
  }

  // Resolve a click into { kind: 'model', fileId } (an overlaid model),
  // { kind: 'point', nodeName, pointId, position } (a bare cloud point), or
  // null (empty space / the grid). Points are checked first: a point selection
  // also carries a node id, so testing isPointSelection() avoids mistaking it
  // for a model face.
  async function pick(clientX, clientY) {
    const point = toCanvasPoint(view, container, clientX, clientY);
    if (!point) return null;
    let item;
    try {
      item = await view.pickFromPoint(point, pickConfig);
    } catch {
      return null;
    }
    if (!item) return null;
    if (item.isPointSelection && item.isPointSelection()) {
      const entity = item.getPointEntity();
      if (!entity) return null;
      const pointId = entity.getPointId();
      const nodeId = item.getNodeId ? item.getNodeId() : null;
      const hasNodeId = nodeId != null && nodeId !== InvalidNodeId;
      const nodeName = hasNodeId ? nodeNameOf(hwv.model, nodeId) : null;
      return { kind: 'point', nodeName, pointId, position: entity.getPosition() };
    }
    const nodeId = item.getNodeId ? item.getNodeId() : null;
    if (nodeId == null || nodeId === InvalidNodeId) return null;
    const fileId = containerFileId(hwv.model, nodeId, nodeToFile);
    return fileId ? { kind: 'model', fileId } : null;
  }

  let downPos = null;
  container.addEventListener('pointerdown', (event) => {
    downPos = event.button === 0 ? { x: event.clientX, y: event.clientY } : null;
  });
  container.addEventListener('pointerup', async (event) => {
    const start = downPos;
    downPos = null;
    if (!start || event.button !== 0) return;
    if (Math.hypot(event.clientX - start.x, event.clientY - start.y) > CLICK_MOVE_TOLERANCE_PX) {
      return; // a camera drag, not a click
    }
    const result = await pick(event.clientX, event.clientY);
    if (!result) {
      // Empty space: report a null model pick so the caller closes the panel,
      // exactly like the point cloud's empty-click behaviour.
      if (onModelClick) onModelClick(null);
    } else if (result.kind === 'model') {
      if (onModelClick) onModelClick(result.fileId);
    } else if (onPointClick) {
      onPointClick(result.nodeName, result.pointId, result.position);
    }
  });
  container.addEventListener('dblclick', async (event) => {
    const result = await pick(event.clientX, event.clientY);
    if (!result) return;
    if (result.kind === 'model') {
      if (onModelActivate) onModelActivate(result.fileId);
    } else if (onPointActivate) {
      onPointActivate(result.nodeName, result.pointId, result.position);
    }
  });
}

// Resolve a picked geometry node up to the container node one of the models was
// loaded under, and return that model's file id (or null). Bounded so a broken
// tree can never loop.
function containerFileId(model, nodeId, nodeToFile) {
  let id = nodeId;
  for (let depth = 0; depth < 32 && id != null; depth++) {
    if (nodeToFile.has(id)) return nodeToFile.get(id);
    id = model.getNodeParent(id);
  }
  return null;
}

// Wire up point-cloud picking for the map: single click reports a pick,
// double click activates it. Both are driven by our own pointerdown/pointerup
// (click detection) and dblclick listeners on the canvas container, each
// issuing a fresh view.pickFromPoint() call with a PickConfig that only
// allows points -- not the shared Select operator's own click handling. This
// keeps the PickConfig used for every pick under this module's explicit
// control (a bare point cloud has no faces or lines to accidentally match)
// and means nothing is ever added to the selection manager, so there is no
// selection highlight to suppress.
function setupPointPicking(hwv, onPointClick, onPointActivate) {
  const view = hwv.view;
  const container = view.domElements.getCanvasContainerElement();
  const pickConfig = new PickConfig(SelectionMask.Point);

  // The built-in SelectionOperator fits the view to the model bounding box on a
  // double click. On the point-cloud map that reframes the whole cloud (back to
  // the iso overview) and, because it runs before our own async pick resolves,
  // it can also make the double click miss the point under the cursor, so the
  // part never opens. Disable it so a double click opens the part on the first
  // try without moving the camera -- the same fix applied to the 3D-model map
  // (see setupModelPicking).
  try {
    const selectOp = view.operatorManager.getOperator(OperatorId.Select);
    if (selectOp && selectOp.setDoubleClickFitWorldEnabled) {
      selectOp.setDoubleClickFitWorldEnabled(false);
    }
  } catch {
    // Older engines may not expose this; the double click still works, it just
    // may reframe the cloud first.
  }

  async function pick(clientX, clientY) {
    const point = toCanvasPoint(view, container, clientX, clientY);
    if (!point) return null;
    let item;
    try {
      item = await view.pickFromPoint(point, pickConfig);
    } catch {
      return null; // outside the canvas area, or the pick failed
    }
    if (!item || !item.isPointSelection || !item.isPointSelection()) return null;
    const entity = item.getPointEntity();
    if (!entity) return null;
    const pointId = entity.getPointId();
    // getNodeId() never returns null/undefined itself -- it falls back to
    // InvalidNodeId internally when the selection has no node, so that
    // fallback has to be checked for explicitly here (see resolvePoint() in
    // shape-map.js for what happens with a null node name).
    const nodeId = item.getNodeId ? item.getNodeId() : null;
    const hasNodeId = nodeId != null && nodeId !== InvalidNodeId;
    const nodeName = hasNodeId ? nodeNameOf(hwv.model, nodeId) : null;
    return { nodeName, pointId, position: entity.getPosition() };
  }

  let downPos = null;
  container.addEventListener('pointerdown', (event) => {
    downPos = event.button === 0 ? { x: event.clientX, y: event.clientY } : null;
  });
  container.addEventListener('pointerup', async (event) => {
    const start = downPos;
    downPos = null;
    if (!start || event.button !== 0) return;
    const dx = event.clientX - start.x;
    const dy = event.clientY - start.y;
    if (Math.hypot(dx, dy) > CLICK_MOVE_TOLERANCE_PX) return; // a camera drag, not a click
    const hit = await pick(event.clientX, event.clientY);
    // Always reported, even when nothing was hit (hit is null): the caller
    // tells a click on empty space (no position) apart from a pick it could
    // not resolve (a position with no matching /map point).
    if (onPointClick) {
      onPointClick(hit ? hit.nodeName : null, hit ? hit.pointId : null, hit ? hit.position : null);
    }
  });
  container.addEventListener('dblclick', async (event) => {
    const hit = await pick(event.clientX, event.clientY);
    if (hit && onPointActivate) onPointActivate(hit.nodeName, hit.pointId, hit.position);
  });
}

// Convert a client-space position (as reported by a DOM pointer/mouse event)
// into the canvas-space Point2 that pickFromPoint() expects, scaling by the
// canvas's own pixel size in case it differs from its CSS layout size (e.g.
// device pixel ratio) rather than assuming the two always match.
function toCanvasPoint(view, container, clientX, clientY) {
  const rect = container.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;
  const size = view.getCanvasSize();
  return new Point2(
    ((clientX - rect.left) / rect.width) * size.x,
    ((clientY - rect.top) / rect.height) * size.y,
  );
}

// Walk the assembly tree from the root and record every named node, keyed by
// name. Names come from the SCS (group_XXXX); a name may appear on more than one
// node (the category's child node and its body-instance node), so values are
// arrays. Used to toggle a whole category's visibility by name.
//
// WARNING: this map holds EVERY named node in the SCS, which includes the
// ancestors of the category nodes, not just the categories the legend knows
// about. Hiding a node in HOOPS hides its whole subtree, so anything that
// drives visibility from this map in bulk must state explicitly which names it
// may touch and leave the rest alone -- hiding "everything that is not X" would
// hide X's own ancestors and take X down with them, with no way back because
// the restore pass would only know to re-show the categories.
function buildNodeNameMap(model) {
  const map = new Map();
  if (!model) return map;
  const root = model.getAbsoluteRootNode();
  const stack = [root];
  while (stack.length) {
    const id = stack.pop();
    const name = model.getNodeName(id);
    if (name) {
      const ids = map.get(name);
      if (ids) ids.push(id);
      else map.set(name, [id]);
    }
    for (const child of model.getNodeChildren(id) || []) stack.push(child);
  }
  return map;
}

// Resolve the category name of a picked node: the geometry node the viewer
// reports may be the (named) body-instance node or an unnamed helper, so walk up
// until a named ancestor is found. Bounded so a broken tree can never loop.
function nodeNameOf(model, nodeId) {
  let id = nodeId;
  for (let depth = 0; depth < 16 && id != null; depth++) {
    const name = model.getNodeName(id);
    if (name) return name;
    id = model.getNodeParent(id);
  }
  return null;
}

// Switch the viewer to perspective THROUGH the camera service, so the
// toolbar's Camera menu shows which projection is actually in effect.
//
// view.setProjectionMode() changes the projection and nothing else. The two
// projection entries in the Camera menu are highlighted from
// CameraService.getProjectionMode(), which does read the live view -- but the
// menu is a Lit component that only re-renders when the service dispatches
// 'hoops-projection-mode-changed' or 'hoops-camera-service-reset'. Setting the
// mode on the view directly fires neither, so the menu kept whatever highlight
// it had when the service last reset it (Orthographic, the SDK default) while
// the map was in fact perspective. Clicking "Perspective Projection" could not
// clear it either: the service short-circuits when the requested mode is the
// one already in effect, so no event was dispatched then either.
//
// Going through the service does the same thing to the view AND dispatches the
// event, so the menu agrees with the scene. Best-effort: the service is a
// singleton registered by the components package and throws when it has no web
// viewer yet, so any failure falls back to the direct call -- a stale highlight
// is a smaller problem than a map left in orthographic.
function setPerspectiveProjection(hwv) {
  const view = hwv.view;
  try {
    const service = tryGetService('CameraService');
    if (service) {
      service.setProjectionMode('Perspective');
      if (view.getProjectionMode() === Projection.Perspective) return;
    }
  } catch {
    // Fall through to the direct call below.
  }
  view.setProjectionMode(Projection.Perspective);
}

// Apply the point-cloud display settings once the map is loaded, then frame it.
// Perspective (over the default orthographic) and the axis triad, because
// orthographic reads as flat for a point cloud that is meant to convey depth.
async function applyPointCloudView(hwv, overlayModels = false, initialCamera = null) {
  if (!hwv) return;
  const view = hwv.view;
  await view.setPointSize(
    overlayModels ? POINT_SIZE_MODELS : POINT_SIZE_CLOUD,
    POINT_SIZE_UNIT_CSS_PIXELS,
  );
  await view.setPointShape(POINT_SHAPE_DISK);
  await view.setEyeDomeLightingEnabled(true);
  setPerspectiveProjection(hwv);
  // The default lower-left anchor sits under the map legend; move it to the
  // lower-right, matching the loaded model view (see enableAxisTriad).
  await enableAxisTriad(hwv);
  // The SDK's own isometric orientation -- exactly what the toolbar's
  // Camera > Iso View button applies, which calls
  // view.setViewOrientation(ViewOrientation.Iso) with these same defaults.
  //
  // Using the SDK's orientation instead of a hand-built camera is what makes
  // the REST of the Camera menu behave. Left, Right, Top, Bottom, Front and
  // Back are all derived from the same view axes as Iso (front -X, up +Z here,
  // since this model carries no CAD up/front metadata -- the points are
  // server-computed, see api.js's Shape Map section). The camera this map used
  // before sat on the (1, 1, 1) octant, which is the mirror image of where
  // those defaults put Iso, so pressing "Left View" swung the camera to what
  // was, on screen, the right-hand side.
  //
  // With those defaults Iso puts the camera on the (-1, -1, 1) octant. The
  // fixed corner addGridPlane() below writes down -- X max, Y max, Z min -- is
  // the opposite one, chosen to match this orientation; changing it here means
  // changing it there too.
  //
  // Duration 0 because this runs while the map is still being revealed -- the
  // view should open at the iso frame, not animate into it.
  await view.setViewOrientation(ViewOrientation.Iso, 0);
  // Home re-applies the same iso orientation directly (see wireMapHomeButton),
  // but the SDK keeps its own notion of "the initial view" that resetCamera()
  // and WebViewer.reset() return to, and left unset that is the SDK's default
  // (an XY-front orthographic view). Point it at the frame above so any SDK
  // path that resets the camera lands where Home does. Read the camera back
  // after setViewOrientation() rather than building one, so the stored frame
  // carries the distance fitWorld() computed.
  view.setInitialCamera(view.getCamera());
  // Preserve the viewpoint across a View toggle: when re-drawing the map for a
  // point-cloud <-> 3D-models switch, the caller passes the camera captured
  // before the toggle, so the new map opens at the same viewpoint instead of
  // snapping back to the iso overview above. Home still resets to that overview
  // (setInitialCamera stays the iso frame); only the CURRENT camera is moved.
  if (initialCamera) {
    try {
      view.setCamera(initialCamera);
    } catch {
      // Best-effort; if the saved camera cannot be applied the iso frame stands.
    }
  }
}

// Author reference grids on three faces of the map's [-1, 1] cube, meeting
// at one corner like the walls of a box -- matching the HOOPS AI forum
// post's Shape Map screenshot, which grids more than just the floor.
// Authored the same way as the printing-plane grid in the HOOPS Web Viewer
// additive-manufacturing tutorial: one MeshData.addPolyline() call per grid
// line, a single MeshInstanceData per plane. Purely decorative: authored
// out-of-hierarchy (last createMeshInstance() argument) so it never appears
// in a model tree, and MeshInstanceCreationFlags.DoNotSelect keeps it out of
// the default Select operator's picking -- without it, the grid (being line
// geometry, unlike the bare point cloud that operator previously had
// nothing else to select) is both selectable and gets the normal whole-node
// highlight on click, exactly what showing a picked map point with its own
// Circle marker (see PointMarker) instead of a highlight is trying to
// avoid. Same idea as how the tutorial's own printing plane handles this
// (see its Selection page), done via a creation flag instead of removing
// the selection reactively.
//
// The grid is a spatial reference: it is how one point's position is read
// against another's. A reference that moves is not one, and computing the
// corner from the live camera made it move -- orbit, trigger a rebuild, and
// the grid jumps to another face. So the corner is fixed.
//
// X max, Y max, Z min is the octant opposite the iso camera
// applyPointCloudView() opens on ((-1, -1, 1)), which puts two walls behind
// the cloud and the floor under it at the default viewpoint. The price is
// that orbiting far enough brings the grid round to the front, where it reads
// as a wall rather than a backdrop. That is the accepted trade: standing
// still matters more than never being in the way.
//
// This no longer follows the camera, so a later change to the opening camera
// in applyPointCloudView() must revisit this corner by hand.
async function addGridPlane(hwv) {
  if (!hwv) return;
  const farCorner = {
    x: GRID_HALF_SIZE,
    y: GRID_HALF_SIZE,
    z: -GRID_HALF_SIZE,
  };
  for (const [axis, value] of [['x', farCorner.x], ['y', farCorner.y], ['z', farCorner.z]]) {
    await addGrid(hwv, axis, value);
  }
}

// One grid, on the plane where `fixedAxis` is held at `fixedValue` and the
// other two axes span the full [-GRID_HALF_SIZE, GRID_HALF_SIZE] range.
// Returns the instance's node id.
async function addGrid(hwv, fixedAxis, fixedValue) {
  const meshData = new MeshData();
  const step = (GRID_HALF_SIZE * 2) / GRID_DIVISIONS;
  for (let i = 0; i <= GRID_DIVISIONS; i++) {
    const position = -GRID_HALF_SIZE + i * step;
    meshData.addPolyline([
      ...gridPoint(fixedAxis, fixedValue, -GRID_HALF_SIZE, position),
      ...gridPoint(fixedAxis, fixedValue, GRID_HALF_SIZE, position),
    ]);
    meshData.addPolyline([
      ...gridPoint(fixedAxis, fixedValue, position, -GRID_HALF_SIZE),
      ...gridPoint(fixedAxis, fixedValue, position, GRID_HALF_SIZE),
    ]);
  }
  const meshId = await hwv.model.createMesh(meshData);
  const instanceData = new MeshInstanceData(meshId);
  instanceData.setLineColor(GRID_LINE_COLOR);
  instanceData.setCreationFlags(MeshInstanceCreationFlags.DoNotSelect);
  return hwv.model.createMeshInstance(instanceData, null, null, true);
}

// Place (u, v) into the two axes that are not `fixedAxis`, which is held at
// `fixedValue`; returns an [x, y, z] triple.
function gridPoint(fixedAxis, fixedValue, u, v) {
  if (fixedAxis === 'x') return [fixedValue, u, v];
  if (fixedAxis === 'y') return [u, fixedValue, v];
  return [u, v, fixedValue]; // 'z'
}

// Dispose the current model and show a plain message instead of a 3D scene.
// Used when there is nothing to view (e.g. the query is an uploaded file, which
// has no SCS to display).
export function showViewerMessage(text, isError = false) {
  disposeViewer();
  setMessage(document.getElementById('viewer-message'), text, isError);
}
