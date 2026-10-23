<!-- A Japanese translation is maintained in umap_ja.md. When you change either file, update the other so both languages stay in sync. -->

# Optional: the UMAP layout

[English](umap.md) | [日本語](umap_ja.md) — back to the [README](../README.md).

The Shape Map flattens each part's 2,048-dimensional aggregate embedding to
three dimensions so it can be drawn. By default it does that with PCA. Install
`umap-learn` and UMAP becomes a second choice.

**Only the placement of the points changes.** The k-means clustering runs on the
full 2,048-dimensional vectors, never on the projection, so which cluster a part
belongs to — and the cluster's number, colour, member count and cohesion — is
identical under both layouts. UMAP changes where a point is drawn, nothing else.

**Installing it**

`umap-learn` brings numba, llvmlite, pynndescent and scikit-learn with it, and
it installs into the HOOPS AI virtual environment — the one that also holds the
SDK's own numpy, scipy, torch and faiss. The constraint that decides whether
that is safe belongs to numba: 0.68.0 declares `numpy>=1.22,<2.6`. Every numpy
available on Python 3.12 falls inside that window, and a dry run from a clean
environment at both ends of it (1.26.4, which is what this was verified
against, and 2.5.3, the newest) adds ten packages and leaves numpy and scipy
untouched. So on today's versions this is an ordinary install.

The four steps below are not there because it is expected to go wrong. They are
there because the one thing that would go wrong — pip deciding to move numpy to
satisfy numba, which would take torch and faiss with it — is invisible while it
happens and awkward to undo afterwards. Step 2 makes the answer for *your*
machine visible in a few seconds, and step 1 makes it reversible. That is worth
a minute even when the answer is almost certainly "nothing moved".

1. **Record what is installed now.** Write the file *outside* this checkout, so
   it is not mistaken for part of the repository.

   **Windows:**
   ```bat
   <Path\to\HOOPS_AI\install\dir>\.venv\Scripts\pip.exe freeze > %USERPROFILE%\sdk-packages-before-umap.txt
   ```

   **Linux:**
   ```bash
   /path/to/HOOPS_AI/install/dir/.venv/bin/pip freeze > ~/sdk-packages-before-umap.txt
   ```

2. **Ask pip what it would do, without doing it.** Needs pip 22.2 or newer; if
   your pip does not know `--dry-run`, skip this step — step 1 still covers you.
   Read the "Would install" list. On the versions this was verified with it
   names ten packages and numpy is not among them. If `numpy`, `scipy` or
   `torch` *do* appear on your machine, stop: pip is telling you it intends to
   move a package the SDK depends on, which means the SDK's numpy sits outside
   numba's window. The map layout is not worth that — leave it uninstalled.

   **Windows:**
   ```bat
   <Path\to\HOOPS_AI\install\dir>\.venv\Scripts\pip.exe install --dry-run -r requirements-umap.txt
   ```

   **Linux:**
   ```bash
   /path/to/HOOPS_AI/install/dir/.venv/bin/pip install --dry-run -r requirements-umap.txt
   ```

3. **Install it.**

   **Windows:**
   ```bat
   <Path\to\HOOPS_AI\install\dir>\.venv\Scripts\pip.exe install -r requirements-umap.txt
   ```

   **Linux:**
   ```bash
   /path/to/HOOPS_AI/install/dir/.venv/bin/pip install -r requirements-umap.txt
   ```

4. **Check the SDK still loads before you trust the install.** This imports what
   the server imports, and prints the numpy and scipy versions so they can be
   compared against step 1's file.

   **Windows:**
   ```bat
   <Path\to\HOOPS_AI\install\dir>\.venv\Scripts\python.exe -c "import numpy, scipy; from hoops_ai.ml import CADSearch; print('SDK ok -- numpy', numpy.__version__, 'scipy', scipy.__version__)"
   ```

   **Linux:**
   ```bash
   /path/to/HOOPS_AI/install/dir/.venv/bin/python -c "import numpy, scipy; from hoops_ai.ml import CADSearch; print('SDK ok -- numpy', numpy.__version__, 'scipy', scipy.__version__)"
   ```

**Restart the server afterwards.** Availability is decided by whether the
package can be imported, and that is settled once when the process starts.

To confirm it took effect, `GET /client-config` reports `umap_available: true`.
Without the package it reports `false` and the web client leaves the choice out
of the map's controls entirely, rather than offering an option that would answer
**503**. With it, a **Layout** control appears alongside the other map controls.
Either way the PCA path behaves exactly as before and every cached map and
`.scs` already on disk stays valid.

**Rolling it back**

Restore the versions recorded in step 1. `pip uninstall umap-learn` on its own
is **not** enough: it removes the package but leaves behind whatever numpy the
install moved into place, which is the part that breaks the SDK.

**Windows:**
```bat
<Path\to\HOOPS_AI\install\dir>\.venv\Scripts\pip.exe install -r %USERPROFILE%\sdk-packages-before-umap.txt
```

**Linux:**
```bash
/path/to/HOOPS_AI/install/dir/.venv/bin/pip install -r ~/sdk-packages-before-umap.txt
```

If a line in that file cannot be reinstalled, pip names it; reinstall the rest
and deal with that one on its own.

UMAP is not free: on an index of 7,928 parts at 2,048 dimensions the projection
takes about **25 seconds**. The result is cached, and changing the cluster count
(`k`) or the Outliers threshold does **not** redo it — those change the
clustering only. Drilling down does, since a drill-down re-projects the subset
on its own.

Three environment variables tune it: `HOOPS_AI_SHAPE_MAP_UMAP_N_NEIGHBORS`
(default `15`) sets the neighbourhood size, trading local detail for global
shape; `HOOPS_AI_SHAPE_MAP_UMAP_MIN_DIST` (default `0.1`) sets how tightly
points may pack together; and `HOOPS_AI_SHAPE_MAP_UMAP_SEED` (default `42`)
fixes `random_state`, which is what makes the layout reproducible and also what
makes umap-learn run single-threaded.

> **Why PCA is still the default.** A PCA layout is linear and deterministic:
> the same set of files always produces the same picture, and adding files moves
> it only slightly. UMAP builds its layout from a k-nearest-neighbour graph and
> offers no such guarantee — adding files can rotate or rearrange the whole
> picture. The cluster memberships and numbers do not change; the view of them
> does.
