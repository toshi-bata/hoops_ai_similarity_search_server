// Vite build configuration for the reference web client.
//
// The build output goes to ../static/app so the Python server can serve it
// directly (see the /app mount in main.py). `base` is an absolute path so the
// bundled asset URLs resolve whether the page is opened at /app/ or
// /static/app/. The output is not part of the repository (.gitignore excludes
// /static/app/), so running this build is a required step before the server can
// serve the client; see the "Building" section of the README for why the bundle
// is not committed.
//
// The graphics engine WebAssembly file (engine.esm.wasm) is not part of the JS
// bundle; it is copied verbatim from the installed @ts3d-hoops/sc-engine
// package into static/app/engine/ by the small plugin below. Pinning
// sc-engine to the same version as web-viewer-components keeps the engine and
// the viewer in lock-step (both 2026.2.0).

import { defineConfig } from 'vite';
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';


const require = createRequire(import.meta.url);

// Copy engine.esm.wasm from node_modules into the build output so the engine
// ships with the committed client and never loads from a CDN. The package's
// "exports" map exposes the wasm directly, so resolve it by subpath.
function copyEngineAssets() {
  return {
    name: 'copy-hoops-engine',
    apply: 'build',
    closeBundle() {
      const src = require.resolve('@ts3d-hoops/sc-engine/engine.esm.wasm');
      if (!existsSync(src)) {
        throw new Error(`Engine wasm not found at ${src}. Run "npm ci" first.`);
      }
      const destDir = resolve(__dirname, '../static/app/engine');
      mkdirSync(destDir, { recursive: true });
      copyFileSync(src, resolve(destDir, 'engine.esm.wasm'));
    },
  };
}

export default defineConfig({
  root: __dirname,
  base: '/static/app/',
  build: {
    outDir: resolve(__dirname, '../static/app'),
    emptyOutDir: true,
    target: 'es2020',
  },
  plugins: [copyEngineAssets()],
});
