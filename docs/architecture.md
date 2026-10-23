# Architecture

The [README](../README.md#architecture) lists *what* each module is. This note
explains *why* the boundaries are shaped the way they are, so that the next
person to change the code keeps the properties that make it safe to change.

## 1. The layering

Control flows in exactly one direction:

```
main.py  ->  routers/  ->  core.py (facade)  ->  domain modules  ->  config / locks / storage
```

A lower module never imports a higher one, and **no module imports `core`**.
The only two files allowed to `import core` are `main.py` (the composition
root) and `core.py` itself. Everything else reaches its siblings directly by
their own module name. This keeps the import graph acyclic: the facade depends
on the domains, the domains depend on the shared foundation (`config`, `locks`,
`storage`), and nothing points back up.

`config` resolves the data root once at import time. By default it is
`<app>/data`, so the runtime directories the server writes (`uploads/`,
`indexes/`, `out/`, …) stay under one `data/` subtree, separate from the source;
`HOOPS_AI_DATA_DIR` overrides the location.

## 2. Why three functions stay in the facade

Almost all logic lives in a single-purpose module. Three functions do not:
`remove_from_index`, `list_index_parts` and `delete_index`. Each orchestrates
work that spans *sibling* domains -- the index registry plus the tags, context
and assembly sidecars. Those sidecars sit beside the index-owning module, not
below it, so if an orchestrator were pushed down into any one of them that
module would have to call its siblings, which is an upward dependency and a
cycle. Keeping the three cross-domain orchestrators in the facade is what lets
every domain module stay one-way.

## 3. The re-export convention

Names fall into three classes, and the class decides how the facade exposes
them.

- **(C) stable names** are re-exported by identity through a
  `from <owner> import (...)` block. Callers in `routers/`, `tools/` and
  `tests/` reach them as `core.<name>`, and `core.<name>` *is* the owner's
  object (the layout test pins `core.X is owner.X`).
- **(A) names that a test swaps, or that a sibling module calls,** are **not**
  re-exported. They are reached as `<owner>.<name>` (a module attribute).
- **(B) names rebound with `global`** -- lazy singletons that a first caller
  installs -- are likewise not re-exported; the rebinding must stay inside the
  owning module.

**Why (A) names are never re-exported.** The search cache is the clearest
example. `search._search_cache_key` builds its key from the index's `.faiss`
modification time and the tag sidecar's ETag, and it reads that ETag by calling
`tags._read_tags_document` **as a module attribute**. A test that exercises
tag-filtered search swaps `tags._read_tags_document`. If the facade also
re-exported that name, `search` would still call the original object while the
test patched the copy on `core` -- the two would diverge. In production the same
divergence surfaces as *"a part was tagged but the search still returns the old
result"*, because the path that builds the cache key never saw the new sidecar.
Reaching it as `tags._read_tags_document` keeps a single source of truth.

## 4. Why the split is safe

The caches invalidate themselves. `search._search_cache_key` mixes the `.faiss`
`st_mtime_ns` with the tag sidecar ETag, so re-registering an index (which bumps
the file mtime) or tagging a part (which changes the ETag) makes the next lookup
miss and recompute -- no explicit invalidation call is needed.

Two paths are the deliberate exception, and both live in the facade because both
cross domains:

- `delete_index` actively evicts the named searchers, the assembly-matcher
  cache and the tag cache for the index it removes.
- `remove_from_index` prunes the tag and context sidecars for the removed parts
  and deletes their thumbnails and stream caches.

Everywhere else, letting the key go stale is the invalidation.

## 5. The guards

`tests/test_module_layout.py` pins the structure at runtime. When one turns red,
suspect the matching cause first:

- **`ImportHygiene`** walks every first-class module and fails if one imports
  `core` (the allowlist is `main.py` and `core.py`). Red means a domain module
  grew an `import core` -- the shared logic belongs lower down or in the facade,
  not in a sibling.
- **the `*Facade` classes** assert that each re-exported name is the same object
  as its owner (`core.X is owner.X`). Red means a re-export drifted: a name was
  renamed or removed in its owner, or an (A) name was re-exported by mistake.
- **`ModuleShadowGuard`** fails if a function binds a local name that shadows an
  imported module. Red means a variable or parameter was named after a module
  (see the first trap below).

## 6. Two traps worth knowing

- **Shadowing a module name.** A local variable or parameter named after an
  imported module hides that module inside the scope, so a `<module>.<attr>`
  call quietly resolves against the local value instead. `ModuleShadowGuard`
  exists to catch exactly this.
- **`from owner import X` binds a copy.** A name pulled in with
  `from owner import X` is an independent binding in the importer. If a test
  later swaps `owner.X`, the importer keeps calling the object it bound at import
  time and never sees the swap. Anything a test swaps -- or a sibling calls --
  must be reached as `owner.X`, never bound with `from owner import X`.

## 7. Line endings

The working tree uses CRLF and the git index stores LF (`core.autocrlf` is on),
so a normal checkout round-trips cleanly. `core.py` additionally carries a
UTF-8 BOM; every other module has none. When adding a file, write it noBOM +
CRLF (and keep `core.py` BOM + CRLF) -- never mix the two endings in one file.
`git ls-files --eol "*.py"` must report no `w/mixed` entry.
