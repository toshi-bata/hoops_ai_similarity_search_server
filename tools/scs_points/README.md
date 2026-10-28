# scs_points

`scs_points` authors a HOOPS Web Viewer point-cloud `.scs` from a list of
coloured points. The similarity-search server calls it to turn a Shape Map into
a 3D point cloud that a future web client can open in the viewer.

The tool embeds a HOOPS license key, so it is **not** committed. Build it
yourself (any HOOPS Web Viewer license holder has the SDK) and drop the
executable in the repository's `bin/` directory.

## What it does

- Reads a point cloud, split into **category groups** (one k-means cluster, one
  tag, or "untagged"), from an input JSON file.
- Writes **one mesh / one named node per group**, colouring the whole node with
  the single RGB the server computed for that category (via a point material, not
  per-point colours). This lets the viewer show/hide or recolour a whole category
  by node without regenerating the file.
- Preserves order: groups, and the points inside each group, are written in the
  exact order received. A point's `(node name, index within node)` is the only
  link back to a part's `file_id` — the server maps it to `group.offset + index`
  in its `/map` array. No id is stored in the file.

## CLI

```
scs_points --input <points.json> --output <out.scs>
```

- Success: exit code `0`, one status line on stdout.
- Failure: non-zero exit code, one reason line on stderr.

Input JSON (group order and point order within a group are the contract):

```json
{
  "version": 2,
  "groups": [
    {
      "node_name": "group_0000",
      "r": 230, "g": 159, "b": 0,
      "points": [
        { "x": 0.0, "y": 0.0, "z": 0.0 }
      ]
    }
  ]
}
```

`node_name` is the stable key the viewer uses to map a picked node back to a
group — the tool never relies on node creation order. Coordinates are passed
straight through (the server sends values already normalised to `[-1, 1]`).
Colours are one 0-255 RGB triple per group.

## Build

The build resolves SDK headers and libraries from a single environment
variable, `HVISUALIZE_WEB_INSTALL_DIR`, which points at the root of the HOOPS
Visualize Web SDK (the folder that contains `authoring/libsc`).
No absolute SDK path is stored in the build scripts.

> **Match the SDK version to the web viewer engine.** Build against the same
> HOOPS Visualize Web version the client is pinned to (`web/package.json`:
> `@ts3d-hoops/* = 2026.2.0`). A point-cloud SCS authored by a **newer** libsc
> (e.g. `2026.6.0`) fails to load in the `2026.2.0` engine and the client shows
> *"Failed to load the map."*

### License header

`main.cpp` includes `hoops_license.h`, which defines `HOOPS_LICENSE`. Download a
valid `hoops_license.h` for your account from the Tech Soft 3D **Developer Zone**
and place it in this folder (`tools/scs_points/`). It is git-ignored, so your key
is never committed. Do **not** copy a placeholder into place — the build needs a
real key to author the SCS.

### Windows (MSVC, terminal)

Only the **MSVC compiler** is needed — the "Desktop development with C++"
workload of **Build Tools for Visual Studio 2022** is enough. The Visual Studio
IDE is not required (the HOOPS authoring libraries are MSVC-built static
libraries linked with `/MD`, so MinGW is not an option).

```powershell
copy _win_env.bat.example _win_env.bat   # then edit the SDK path
.\build.bat                              # writes ..\..\bin\scs_points.exe
.\build.bat clean                        # removes build_win\ and the exe
```

An ordinary PowerShell or Command Prompt window is fine; there is no need to
open a *Developer Command Prompt* first. `build.bat` locates the toolset with
`vswhere` (which ships with every Visual Studio 2017+ installer, including the
standalone Build Tools) and calls `vcvars64.bat` itself.

> **Why there is no `.sln` any more.** Linux builds through a single Makefile.
> Keeping a `.vcxproj` alongside a command line meant the correct set of
> compiler flags existed in two places on Windows, and only one of them would
> be kept current. There is one build path per platform now; the flags
> `build.bat` uses are the ones the retired `Release | x64` configuration had.

### Linux (Ubuntu)

`libsc` is a plain C/C++ SDK and this tool is a single-file C++17 console
program, so it builds with a hand-written `Makefile` and `g++` — **no Qt and no
qmake**. Only `build-essential` is required:

```
sudo apt-get install -y build-essential

cp _linux_env.sh.example _linux_env.sh   # then edit the SDK path
source _linux_env.sh
make                                     # writes ../../bin/scs_points
```

Override the SDK lib directory or library names on the command line if your SDK
layout differs, for example
`make SC_LIBDIR=$HVISUALIZE_WEB_INSTALL_DIR/authoring/libsc/lib/linux_x64`.

> **Confirmed against HOOPS Visualize Web 2026.2.0 on Ubuntu** (the version the
> web viewer engine is pinned to). The Linux SDK ships **static** archives
> (`libsc.a`, `libsc_utils.a`, `libsc_utils_text.a`, `libz.a`) and has **no
> `hps_core`** library (unlike Windows). The default `SC_LIBS` reflects this.
> The bundled set varies between SDK versions (`2026.6.0`, for example, also
> ships `libpng16.a`/`libpng.a`, needing
> `SC_LIBS="-lsc -lsc_utils -lsc_utils_text -lpng16 -lpng -lz -lpthread -ldl -lm"`);
> adjust via `SC_LIBDIR` / `SC_LIBS`.

## Output

The tool links the HOOPS authoring libraries statically, so `bin/` needs only
the executable (plus the usual Microsoft Visual C++ runtime on Windows). See
`bin/README.md`.
