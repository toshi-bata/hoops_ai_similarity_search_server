"""Tests for the memory-bounded in-process caches (no SDK required).

These pin the self-review requirement that an extremely small cap makes the
caches evict (slower, more reloads) but never break: entries stay coherent, the
byte counters stay accurate, and the most-recently-used entry is never evicted.
"""

import os
import sys
import unittest
from collections import OrderedDict
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core
import index_registry
import embeddings


def _fake_entry(dim: int = 4) -> dict:
    return {
        "file_id": "x",
        "vector": np.zeros(dim, dtype=np.float32),
        "dim": dim,
        "model_name": "m",
        "num_bodies": 1,
        "filename": "f",
    }


class EmbeddingCacheCaps(unittest.TestCase):
    def setUp(self):
        self._orig = embeddings._embedding_memory_cache
        self._orig_bytes = embeddings._embedding_cache_bytes
        embeddings._embedding_memory_cache = OrderedDict()
        embeddings._embedding_cache_bytes = 0

    def tearDown(self):
        embeddings._embedding_memory_cache = self._orig
        embeddings._embedding_cache_bytes = self._orig_bytes

    def test_evicts_lru_but_keeps_one(self):
        # One entry is ~ dim*4 + 512 bytes. A 0 MB->min 1 MB cap is still large,
        # so force eviction by monkeypatching the cap to 0 bytes.
        with patch.object(embeddings, "_embed_cache_max_bytes", return_value=0):
            core._embedding_cache_put("a", _fake_entry())
            core._embedding_cache_put("b", _fake_entry())
            core._embedding_cache_put("c", _fake_entry())
        # Never empties: at least the most-recently-used entry survives.
        self.assertGreaterEqual(len(embeddings._embedding_memory_cache), 1)
        self.assertIn("c", embeddings._embedding_memory_cache)
        # Byte counter matches the surviving entries exactly.
        expected = sum(
            core._embed_entry_bytes(e) for e in embeddings._embedding_memory_cache.values()
        )
        self.assertEqual(embeddings._embedding_cache_bytes, expected)

    def test_get_marks_mru_and_reinsert_updates_counter(self):
        big_cap = 10 * 1024 * 1024
        with patch.object(embeddings, "_embed_cache_max_bytes", return_value=big_cap):
            core._embedding_cache_put("a", _fake_entry())
            core._embedding_cache_put("b", _fake_entry())
            # Touch "a" so it becomes most-recently-used.
            self.assertIsNotNone(core._embedding_cache_get("a"))
            self.assertEqual(list(embeddings._embedding_memory_cache)[-1], "a")
            # Re-inserting an existing key must not double-count bytes.
            before = embeddings._embedding_cache_bytes
            core._embedding_cache_put("a", _fake_entry())
            self.assertEqual(embeddings._embedding_cache_bytes, before)

    def test_clear_resets_counter(self):
        with patch.object(embeddings, "_embed_cache_max_bytes", return_value=10 * 1024 * 1024):
            core._embedding_cache_put("a", _fake_entry())
        core.clear_embedding_memory_cache()
        self.assertEqual(len(embeddings._embedding_memory_cache), 0)
        self.assertEqual(embeddings._embedding_cache_bytes, 0)


class NamedIndexCacheCaps(unittest.TestCase):
    def setUp(self):
        self._orig = index_registry._named_indexes
        index_registry._named_indexes = OrderedDict()

    def tearDown(self):
        index_registry._named_indexes = self._orig

    def test_evicts_lru_but_keeps_in_use(self):
        # Each fake index "weighs" 100 bytes; cap at 150 forces eviction to one.
        with patch.object(index_registry, "_index_entry_bytes", return_value=100), \
             patch.object(index_registry, "_index_cache_max_bytes", return_value=150):
            core._store_named_index("a", object())
            core._store_named_index("b", object())
            core._store_named_index("c", object())
        # Oldest evicted, the just-stored one (keep) is retained.
        self.assertIn("c", index_registry._named_indexes)
        self.assertLessEqual(len(index_registry._named_indexes), 2)

    def test_single_oversized_index_is_kept(self):
        with patch.object(index_registry, "_index_entry_bytes", return_value=10_000), \
             patch.object(index_registry, "_index_cache_max_bytes", return_value=100):
            core._store_named_index("big", object())
        # A lone index larger than the cap is kept resident, not thrashed away.
        self.assertIn("big", index_registry._named_indexes)


if __name__ == "__main__":
    unittest.main()
