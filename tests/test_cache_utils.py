"""
cache_utils.py ported verbatim. These tests pin the two properties the
whole disk-cache strategy depends on: (1) cache_root() resolves next to
this script, not the system temp/profile drive, and (2) module_version()
changes when the watched file's bytes change -- the property that makes
leaving the cache on by default safe during active bug-fixing (an edited
module auto-invalidates every cache entry that depended on its old logic).
"""
import os
import tempfile
import unittest
from pathlib import Path

import cache_utils


class TestCacheUtils(unittest.TestCase):
    def test_cache_root_is_next_to_script_not_system_temp(self):
        root = cache_utils.cache_root()
        self.assertEqual(root, Path(__file__).resolve().parent.parent / "_cache")
        self.assertTrue(root.is_dir())

    def test_module_version_changes_with_file_content(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "watched.py"
            f.write_text("VERSION = 1\n", encoding="utf-8")
            h1 = cache_utils.module_version(f)

            # module_version() is memoized per-process by path -- a second
            # call against the SAME path must return the cached hash even
            # after the file changes underneath it (matches how it's
            # actually used: hashed once per pipeline run, not re-hashed
            # mid-run). Verified separately below with a fresh path.
            f.write_text("VERSION = 2\n", encoding="utf-8")
            h1_again = cache_utils.module_version(f)
            self.assertEqual(h1, h1_again, "module_version should be memoized within one process")

            f2 = Path(td) / "watched2.py"
            f2.write_text("VERSION = 2\n", encoding="utf-8")
            h2 = cache_utils.module_version(f2)
            self.assertNotEqual(h1, h2, "different file content must produce a different hash")

    def test_get_set_round_trip(self):
        namespace = f"test_ns_{id(self)}"
        key_parts = ("a", "b", 123)
        self.assertIsNone(cache_utils.get(namespace, *key_parts))
        cache_utils.set(namespace, {"hello": "world"}, *key_parts)
        self.assertEqual(cache_utils.get(namespace, *key_parts), {"hello": "world"})

    def test_entry_dir_is_stable_for_same_key(self):
        namespace = f"test_ns_dir_{id(self)}"
        d1 = cache_utils.entry_dir(namespace, "x", "y")
        d2 = cache_utils.entry_dir(namespace, "x", "y")
        d3 = cache_utils.entry_dir(namespace, "x", "z")
        self.assertEqual(d1, d2)
        self.assertNotEqual(d1, d3)
        self.assertTrue(d1.is_dir())

    def test_file_identity_reflects_size_and_mtime(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "x.bin"
            f.write_bytes(b"abc")
            id1 = cache_utils.file_identity(f)
            f.write_bytes(b"abcdef")
            id2 = cache_utils.file_identity(f)
            self.assertNotEqual(id1, id2)


class TestEnforceSizeCap(unittest.TestCase):
    """The confirmed real problem: mesh_convert had no eviction at all and
    grew to 392GB. These pin the eviction sweep itself, not real
    mesh_convert entries."""

    def _make_entry(self, namespace, name, num_bytes, mtime):
        d = cache_utils.cache_root() / namespace
        d.mkdir(parents=True, exist_ok=True)
        entry_dir = d / name
        entry_dir.mkdir(exist_ok=True)
        f = entry_dir / "payload.bin"
        f.write_bytes(b"\0" * num_bytes)
        os.utime(entry_dir, (mtime, mtime))
        os.utime(f, (mtime, mtime))
        return entry_dir

    def test_under_cap_leaves_everything_alone(self):
        namespace = f"test_ns_cap_{id(self)}_under"
        e1 = self._make_entry(namespace, "e1", 100, mtime=1000.0)
        e2 = self._make_entry(namespace, "e2", 100, mtime=2000.0)
        cache_utils.enforce_size_cap(namespace, max_bytes=1_000_000)
        self.assertTrue(e1.is_dir())
        self.assertTrue(e2.is_dir())

    def test_over_cap_evicts_oldest_entries_first(self):
        namespace = f"test_ns_cap_{id(self)}_over"
        oldest = self._make_entry(namespace, "oldest", 1000, mtime=1000.0)
        middle = self._make_entry(namespace, "middle", 1000, mtime=2000.0)
        newest = self._make_entry(namespace, "newest", 1000, mtime=3000.0)
        # Cap under the total (3000 bytes) but over any single entry --
        # only the oldest should go.
        cache_utils.enforce_size_cap(namespace, max_bytes=2500)
        self.assertFalse(oldest.exists())
        self.assertTrue(middle.is_dir())
        self.assertTrue(newest.is_dir())

    def test_missing_namespace_is_a_noop(self):
        # Must not raise for a namespace that was never written to.
        cache_utils.enforce_size_cap(f"test_ns_never_created_{id(self)}", max_bytes=1)

    def test_logs_only_when_something_was_evicted(self):
        namespace = f"test_ns_cap_{id(self)}_log"
        self._make_entry(namespace, "e1", 1000, mtime=1000.0)
        logs = []
        cache_utils.enforce_size_cap(namespace, max_bytes=1_000_000, log_callback=lambda msg, lvl: logs.append(msg))
        self.assertEqual(logs, [])
        cache_utils.enforce_size_cap(namespace, max_bytes=1, log_callback=lambda msg, lvl: logs.append(msg))
        self.assertEqual(len(logs), 1)


if __name__ == "__main__":
    unittest.main()
