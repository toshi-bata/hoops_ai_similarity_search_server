# Third-Party Notices — compiled HOOPS components

UPDATE THESE FILES WHENEVER THE HOOPS COMPONENT VERSIONS CHANGE: re-fetch the
acknowledgment texts for the version this project is pinned to.

This directory holds the third-party acknowledgment texts for HOOPS components
this project ships. Some are compiled into the HOOPS binaries (embedded inside
the WebAssembly engine, the minified viewer bundle, and the statically linked
`libsc`) and are therefore **not** visible in the npm dependency graph (see
`../THIRD-PARTY-NOTICES.md` for that layer). Others are the acknowledgment texts
that ship with the HOOPS AI and HOOPS Exchange Python packages bundled into the
redistribution package.

The `.txt` files are transcribed verbatim as downloaded. No summary, paraphrase,
or legal interpretation is added.

## Files and what they correspond to

- `label_hwv.txt` — **HOOPS Web Viewer (HWV)**: the browser viewer and the
  WebAssembly engine. In this project that is the built web client under
  `static/app/` (the `@ts3d-hoops/*` npm packages and `engine.esm.wasm`).
- `label_sc.txt` — **HOOPS Visualize / Stream Cache (SC)**: the `libsc` library
  that is statically linked into `bin/scs_points` (the Shape Map point-cloud
  tool; see `../tools/scs_points/` and `../bin/README.md`).
- `label_hoops_ai.txt` — **HOOPS AI**: the acknowledgments published for the
  HOOPS AI Python package, which is bundled into the redistribution package.
- `label_hoops_exchange.txt` — **HOOPS Exchange**: the acknowledgments that ship
  with the HOOPS Exchange SDK; HOOPS Exchange is bundled into the redistribution
  package as a Python package.

The Converter and Server component notices published on the same page are
**intentionally not included**, because this project does not ship those
components.

## Source and retrieval

- `label_hwv.txt` — https://docs.techsoft3d.com/hoops/visualize-web/_downloads/label_hwv.txt
- `label_sc.txt` — https://docs.techsoft3d.com/hoops/visualize-web/_downloads/label_sc.txt
- Linked from: https://docs.techsoft3d.com/hoops/visualize-web/overview/acknowledgments.html
- Retrieved: 2026-09-01
- `label_hoops_ai.txt` — https://docs.techsoft3d.com/hoops/ai/resources/Acknowledgments.html
  (retrieved 2026-09-04)
- `label_hoops_exchange.txt` — copied from the pinned SDK at
  `C:\SDK\Exchange\HOOPS_Exchange_2026.2.0\samples\data\html\acknowledgments.txt`
  (retrieved 2026-09-04)

## Version note

No standalone equivalent acknowledgment text file was found inside the pinned
SDK at `C:\SDK\Communicator\HOOPS_Visualize_Web_2026.2.0`; there, the same text
is only embedded inside that SDK's minified viewer bundles, not published as a
separate file. These two files were therefore downloaded from the public HOOPS
Visualize Web documentation site. At retrieval time that page's version selector
reported "HOOPS Visualize Web 2026.2.0", but it is the public documentation page
rather than a file shipped in the pinned SDK, so it may be updated over time and
is not guaranteed to correspond byte-for-byte to the 2026.2.0 build. Do not
assume an exact match; re-verify against the pinned SDK version when it matters.

`label_hoops_exchange.txt` was copied directly from the pinned HOOPS Exchange
2026.2.0 SDK, so it corresponds to that build.

`label_hoops_ai.txt` was transcribed from the public HOOPS AI documentation
page, whose title reports "Acknowledgments — HOOPS AI 1.0". This project uses
HOOPS AI 1.1.0, so the page's stated version does not match the bundled version.
The page is also not shipped as a file inside the SDK and may be updated over
time. Do not assume an exact match; re-verify against the bundled HOOPS AI
version when it matters. The page's "Third-Party Dependencies" table is
generated from HOOPS AI's own dependency metadata, so it likewise reflects the
documented version rather than necessarily the bundled one.
