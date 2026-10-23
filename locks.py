"""Central registry of the process-wide lock hierarchy.

The lock-ordering rules span several modules: a build lock in one area is taken
around per-index work handled in another. To keep the ordering in one place --
so anyone adding or acquiring a lock reads the same rules -- every shared lock
registry lives here rather than next to the code that happens to use it first.
This module depends on ``threading`` only and never imports ``core``.

LOCK ORDERING (read this before adding or acquiring a lock)

Acquire locks in this order, outermost first. Never acquire an earlier one
while holding a later one.

  1. Per-subject BUILD locks (serialise one expensive build per subject):
       _assembly_build_locks[name]   (assembly matcher k-means, ~20 s)
       _shape_map_build_locks[key]   (shape-map SVD, ~8 s)
     The heavy work runs while holding these but OUTSIDE the index lock, so a
     read-only index is not stalled by another index's build. The startup
     warmup thread also goes through these (via _get_assembly_matcher /
     compute_index_shape_map), so it obeys the same order.

     The per-.scs build lock (_scs_map_build_locks[key]) sits above the
     shape-map build locks: compute_index_shape_map_scs takes it as its
     OUTERMOST lock and never while holding any other, and the
     compute_index_shape_map it calls underneath takes the shape-map locks, so
     the order is always scs-build-lock -> shape-map locks, never the reverse.

  2. The per-index lock: _get_index_lock(name)
     Held for disk reads/writes of one index. A build lock may take this
     (build lock -> index lock); the reverse is forbidden.

  3. LEAF data-structure mutexes (never take another lock while holding one):
       _named_indexes_mutex, _named_searchers (guarded by the index lock),
       _assembly_matchers_mutex, _shape_map_cache_mutex, _embedding_cache_mutex,
       _tags_cache_mutex, _search_cache_mutex.

  The per-index tags lock (_get_tags_lock(name)) is a sibling of the index
  lock, not below it: a tag edit touches no vectors, so it must not block
  searches. Registered ids are resolved before the tags lock is taken, so the
  index lock is never acquired while holding it. Lock order: tags lock ->
  (nothing).

  The tag-vocabulary lock (_get_tag_vocabulary_lock()) is process-wide, not per
  index, because the file it guards is shared by the whole application. It is a
  leaf: it is taken after the tags lock has been released, and nothing is taken
  while holding it. Lock order: tag vocabulary lock -> (nothing).

  Registry mutexes (_index_locks_mutex, _tags_locks_mutex,
  _assembly_build_locks_mutex, _shape_map_build_locks_mutex,
  _scs_map_build_locks_mutex) and the prewarm mutexes are leaf locks held only
  to fetch/create a lock object or a small set; they are released before the
  returned lock is acquired.
"""

import threading

# Serialises writes to the shared progress stream so concurrent jobs do not
# interleave their tqdm output.
_PROGRESS_STDERR_LOCK = threading.Lock()

# per-index write lock; also held while resolving the cached searcher
_index_locks: dict[str, threading.Lock] = {}
_index_locks_mutex = threading.Lock()


def _get_index_lock(name: str) -> threading.Lock:
    with _index_locks_mutex:
        if name not in _index_locks:
            _index_locks[name] = threading.Lock()
        return _index_locks[name]


# Per-index lock held across the read-modify-write of tags.json. It is separate
# from the index lock on purpose: a tag edit touches no vectors, so it must not
# block searches. Reads take no lock at all -- the file is swapped in with
# os.replace, so a reader either sees the whole old file or the whole new one.
#
# Lock order: tags lock -> (nothing). Registered ids are resolved *before* the
# tags lock is taken, so the index lock is never acquired while holding it.
_tags_locks: dict[str, threading.Lock] = {}
_tags_locks_mutex = threading.Lock()


def _get_tags_lock(name: str) -> threading.Lock:
    with _tags_locks_mutex:
        if name not in _tags_locks:
            _tags_locks[name] = threading.Lock()
        return _tags_locks[name]


# Lock held across the read-modify-write of the tag vocabulary
# (DATA_ROOT/tag_vocabulary.json). One lock for the process, not one per index,
# because the file itself is shared by the whole application: tagging a part in
# index A and a part in index B at the same moment is a read-modify-write of
# the SAME file by two threads, and a per-index lock would let both run and
# lose one of the two names. The tags lock cannot be reused for it for the same
# reason -- it is per index -- and it is non-reentrant besides.
#
# Lock order: tag vocabulary lock -> (nothing). It is a leaf, taken last and
# never while another lock is held: the vocabulary is recorded only after the
# tags lock for the assignment has been released.
_TAG_VOCABULARY_LOCK = threading.Lock()


def _get_tag_vocabulary_lock() -> threading.Lock:
    """Return the process-wide tag-vocabulary lock.

    A function rather than a bare constant so callers reach it the same way
    they reach the per-subject locks above, and so a test can see which lock a
    writer asked for.
    """
    return _TAG_VOCABULARY_LOCK


# Per-index lock held for the whole of a matcher build, so two callers wanting
# the same matcher produce one build rather than two. Distinct from the index
# lock on purpose: building takes ~20 s, and holding the index lock that long
# would stall part search on an index that is only being read.
# Lock order is build lock -> index lock -> _assembly_matchers_mutex; nothing
# acquires them the other way round.
_assembly_build_locks: dict[str, threading.Lock] = {}
_assembly_build_locks_mutex = threading.Lock()


def _get_assembly_build_lock(name: str) -> threading.Lock:
    with _assembly_build_locks_mutex:
        if name not in _assembly_build_locks:
            _assembly_build_locks[name] = threading.Lock()
        return _assembly_build_locks[name]


# Per-subset build lock so concurrent requests for the same subset run the SVD
# once, not once each. Keyed by (faiss path, mtime_ns, subset_sig). The heavy
# work runs under this lock but OUTSIDE the per-index lock, exactly like the
# assembly matcher build.
_shape_map_build_locks: dict[tuple, threading.Lock] = {}
_shape_map_build_locks_mutex = threading.Lock()

# Per-key build lock so two requests for the same .scs generate it once,
# mirroring _shape_map_build_locks. It is the outermost lock of the pair (see
# the LOCK ORDERING note above).
_scs_map_build_locks: dict[tuple, threading.Lock] = {}
_scs_map_build_locks_mutex = threading.Lock()
