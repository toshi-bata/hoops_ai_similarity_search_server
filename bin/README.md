# Runtime binaries directory

This directory holds the compiled helper executables the server launches at
runtime. Its contents are **not** committed (the build output is git-ignored);
only this `README.md` and `.gitkeep` are tracked.

## What goes here

| File | Platform | Produced by |
| --- | --- | --- |
| `scs_points.exe` | Windows | `tools/scs_points` (build.bat) |
| `scs_points` | Linux | `tools/scs_points` (make) |

The server resolves these by fixed path (`<app_root>/bin/scs_points[.exe]`); the
location is not configurable.

## Why the build output is not committed

`scs_points` links a HOOPS license key into the binary. This repository will be
made public, and a key can be extracted from a compiled binary, so shipping the
executable here would leak the key. Anyone entitled to run the tool holds a HOOPS
SDK and can build it themselves (see `tools/scs_points/README.md`).

## Runtime dependencies

The HOOPS authoring libraries are linked statically, so on Windows the executable
needs only the Microsoft Visual C++ runtime — no HOOPS DLLs are placed here. If a
future change makes a DLL necessary, list it in this table and update the redist
packaging script.

## Redistribution

`tools/build_redist_package.ps1` copies this `bin/` directory into the package and
aborts if the executable is missing, so a redistributable build always ships the
tool.
