"""
main.resolve_library_substitution -- keyword-based fallback for placements
whose real MSFS model is genuinely missing from the package's own
extracted/converted geometry (a base-game/library object that couldn't be
found or converted). Exact keyword match always checked; a fuzzy
(difflib) fallback only kicks in when explicitly opted into.
"""
import unittest

import main


class TestLibrarySubstitution(unittest.TestCase):
    def test_windsock_title_resolves_to_verified_library_path(self):
        lib_path = main.resolve_library_substitution("ASOBO_Windsock_01")
        self.assertEqual(lib_path, "lib/airport/landscape/windsock_lit.obj")

    def test_floodlight_title_resolves(self):
        lib_path = main.resolve_library_substitution("Generic_FloodLight_Tall")
        self.assertEqual(lib_path, "lib/airport/Common_Elements/Lighting/com_Flood_36m.obj")

    def test_copy_suffix_is_stripped_before_matching(self):
        lib_path = main.resolve_library_substitution("Windsock (Copy 3)")
        self.assertEqual(lib_path, "lib/airport/landscape/windsock_lit.obj")

    def test_unrelated_title_does_not_match_by_default(self):
        self.assertIsNone(main.resolve_library_substitution("Terminal_Building_Facade_02"))

    def test_none_title_returns_none(self):
        self.assertIsNone(main.resolve_library_substitution(None))

    def test_approximate_off_by_default_leaves_near_miss_unmatched(self):
        """"wind_soc" (a plausible truncation/typo) doesn't contain any
        exact keyword substring -- must NOT match unless approximate=True
        is explicitly passed, matching the GUI toggle's default-off state."""
        near_miss = "wind_soc_marker"
        self.assertIsNone(main.resolve_library_substitution(near_miss, approximate=False))

    def test_approximate_enabled_can_catch_a_near_miss(self):
        near_miss = "windsok"  # one-character typo of "windsock"
        result_off = main.resolve_library_substitution(near_miss, approximate=False)
        result_on = main.resolve_library_substitution(near_miss, approximate=True)
        self.assertIsNone(result_off)
        self.assertEqual(result_on, "lib/airport/landscape/windsock_lit.obj")

    def test_every_substitution_path_starts_with_lib_prefix(self):
        """Sanity check on the table itself -- every entry must be a real
        X-Plane virtual library path (lib/...), not a leftover local path."""
        for keyword, lib_path in main.LIBRARY_SUBSTITUTION_KEYWORDS:
            self.assertTrue(lib_path.startswith("lib/"), f"{keyword!r} -> {lib_path!r} isn't a lib/ path")


class TestUserReplacementPicks(unittest.TestCase):
    """resolve_library_substitution consults the user's saved
    object_replacements.json picks (via load_object_replacements) BEFORE
    the built-in keyword table -- keyed by GUID or normalized title, with
    a "SKIP" value meaning "place nothing, stop re-reporting"."""

    def test_guid_pick_wins_over_everything(self):
        reps = {"a1b2c3d4": "lib/airport/lights/PAPI_4.obj"}
        got = main.resolve_library_substitution(
            "ASOBO_Taxiway_Light", guid="{A1B2C3D4}", replacements=reps)
        self.assertEqual(got, "lib/airport/lights/PAPI_4.obj")

    def test_title_pick_used_when_no_guid_match(self):
        reps = {"asobo_papi_left": "lib/airport/lights/PAPI_L.obj"}
        got = main.resolve_library_substitution("ASOBO_PAPI_Left (copy 2)", replacements=reps)
        self.assertEqual(got, "lib/airport/lights/PAPI_L.obj")

    def test_skip_sentinel_returned_for_skip_value(self):
        reps = {"deadbeef": "SKIP"}
        got = main.resolve_library_substitution("whatever", guid="DEADBEEF", replacements=reps)
        self.assertEqual(got, main.SKIP_SUBSTITUTION)

    def test_user_pick_overrides_builtin_keyword(self):
        reps = {"asobo_windsock_01": "lib/airport/landscape/windsock_unlit.obj"}
        got = main.resolve_library_substitution("ASOBO_Windsock_01", replacements=reps)
        self.assertEqual(got, "lib/airport/landscape/windsock_unlit.obj")
        # without the pick, the built-in keyword still applies
        self.assertEqual(main.resolve_library_substitution("ASOBO_Windsock_01"),
                         "lib/airport/landscape/windsock_lit.obj")

    def test_no_replacements_dict_is_backward_compatible(self):
        self.assertEqual(main.resolve_library_substitution("ASOBO_Windsock_01", guid="{X}"),
                         "lib/airport/landscape/windsock_lit.obj")


class TestLoadObjectReplacements(unittest.TestCase):
    def _write(self, d, obj):
        import json
        (d / "object_replacements.json").write_text(json.dumps(obj), encoding="utf-8")

    def test_reads_replacements_block_and_normalizes_keys(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._write(d, {"replacements": {"{ABC}": "lib/x.obj", "Some Title": " lib/y.obj "}})
            reps = main.load_object_replacements(d)
            self.assertEqual(reps["abc"], "lib/x.obj")
            self.assertEqual(reps["some title"], "lib/y.obj")

    def test_empty_and_skip_values_become_skip_sentinel(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._write(d, {"replacements": {"k1": "", "k2": "skip", "k3": "SKIP"}})
            reps = main.load_object_replacements(d)
            self.assertEqual(reps["k1"], main.SKIP_SUBSTITUTION)
            self.assertEqual(reps["k2"], main.SKIP_SUBSTITUTION)
            self.assertEqual(reps["k3"], main.SKIP_SUBSTITUTION)

    def test_later_dir_wins_and_missing_file_is_silent(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            d1, d2 = Path(td1), Path(td2)
            self._write(d1, {"replacements": {"k": "lib/first.obj"}})
            self._write(d2, {"replacements": {"k": "lib/second.obj"}})
            self.assertEqual(main.load_object_replacements(d1, d2)["k"], "lib/second.obj")
            self.assertEqual(main.load_object_replacements(Path(td1) / "nope", d1)["k"], "lib/first.obj")


if __name__ == "__main__":
    unittest.main()
