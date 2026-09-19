"""
pick_replacements.py -- the standalone browser that maps a conversion's
unresolved objects to X-Plane library paths. GUI aside, its data plumbing
(read unresolved_objects.json, scan library.txt EXPORTs, merge/save
object_replacements.json) is plain and testable.
"""
import json
import tempfile
import unittest
from pathlib import Path

import pick_replacements as pr


class TestKeyAndTitle(unittest.TestCase):
    def test_guid_key_strips_braces_and_case(self):
        self.assertEqual(pr.obj_key("Foo", "{A1B2-C3}"), "a1b2-c3")

    def test_title_key_when_no_guid_and_copy_suffix_stripped(self):
        self.assertEqual(pr.obj_key("ASOBO PAPI (copy 3)", ""), "asobo papi")


class TestLoadUnresolved(unittest.TestCase):
    def test_reads_objects_and_sorts_by_count(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "unresolved_objects.json").write_text(json.dumps({
                "objects": [
                    {"key": "k1", "title": "Rare", "guid": "", "count": 2},
                    {"key": "k2", "title": "Common", "guid": "{G}", "count": 40},
                ]}), encoding="utf-8")
            got = pr.load_unresolved(d)
            self.assertEqual([o["title"] for o in got], ["Common", "Rare"])

    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(pr.load_unresolved(Path(td)), [])


class TestScanLibraryPaths(unittest.TestCase):
    def test_extracts_lib_paths_from_export_lines(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            libdir = root / "Resources" / "default scenery" / "airport scenery"
            libdir.mkdir(parents=True)
            (libdir / "library.txt").write_text(
                "A\n800\nLIBRARY\n"
                "EXPORT lib/airport/lights/PAPI_4.obj some/real/PAPI_4.obj\n"
                "EXPORT_RATIO 0.5 lib/airport/lights/taxi_edge.obj some/taxi.obj\n"
                "# EXPORT lib/should/not/count.obj nope.obj\n"
                "EXPORT_EXCLUDE lib/airport/lights/rwy_edge.obj some/rwy.obj\n",
                encoding="utf-8")
            paths, nfiles = pr.scan_library_paths(root)
            self.assertEqual(nfiles, 1)
            self.assertIn("lib/airport/lights/PAPI_4.obj", paths)
            self.assertIn("lib/airport/lights/taxi_edge.obj", paths)
            self.assertIn("lib/airport/lights/rwy_edge.obj", paths)
            self.assertNotIn("lib/should/not/count.obj", paths)

    def test_none_root_is_safe(self):
        self.assertEqual(pr.scan_library_paths(None), ([], 0))


class TestSaveAndMerge(unittest.TestCase):
    def test_save_merges_and_blank_removes(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            pr.save_replacements(d, {"k1": "lib/a.obj", "k2": "SKIP"})
            reps = pr.load_existing(d)
            self.assertEqual(reps, {"k1": "lib/a.obj", "k2": "SKIP"})
            # re-save: change k1, drop k2 with blank, add k3
            pr.save_replacements(d, {"k1": "lib/a2.obj", "k2": "", "k3": "lib/c.obj"})
            reps = pr.load_existing(d)
            self.assertEqual(reps, {"k1": "lib/a2.obj", "k3": "lib/c.obj"})

    def test_saved_file_is_readable_by_the_converter_loader(self):
        import main
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            pr.save_replacements(d, {"{A-B}": "lib/x.obj", "some title": "SKIP"})
            reps = main.load_object_replacements(d)
            self.assertEqual(reps["a-b"], "lib/x.obj")
            self.assertEqual(reps["some title"], main.SKIP_SUBSTITUTION)


if __name__ == "__main__":
    unittest.main()
