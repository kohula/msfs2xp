"""
main._model_stem_basename / build_title_stem_index -- the GUID-miss
fallback that resolves a placement to a converted model by its TITLE
when guid_map doesn't map to a converted stem. The LHBP terminal doors
kept vanishing (extracted + titled, but the GUID path broke via a
container / cache / library quirk) -- not placed, and not offered in the
picker either. This recovers them as long as the door model itself was
converted.
"""
import unittest

import main


class TestModelStemBasename(unittest.TestCase):
    def test_strips_8hex_library_prefix(self):
        self.assertEqual(
            main._model_stem_basename("22b74d64_LHBP_Terminal_Ext_Door_001_LOD0"),
            "lhbp_terminal_ext_door_001")

    def test_strips_static_plus_32hex_suffix(self):
        self.assertEqual(
            main._model_stem_basename(
                "LHBP_Terminal_Ext_Door_004_Static_eca2fdd1863bd04391d2ba7d3b1f4903"),
            "lhbp_terminal_ext_door_004")

    def test_bare_title_normalizes_to_itself(self):
        self.assertEqual(
            main._model_stem_basename("LHBP_Terminal_Ext_Door_001"),
            "lhbp_terminal_ext_door_001")

    def test_empty(self):
        self.assertEqual(main._model_stem_basename(None), "")
        self.assertEqual(main._model_stem_basename(""), "")


class TestBuildTitleStemIndex(unittest.TestCase):
    def test_title_resolves_to_converted_stem(self):
        csm = {
            "22b74d64_LHBP_Terminal_Ext_Door_001_LOD0": ["a", "b"],
            "SomethingElse_9f0a1b2c": ["c"],
        }
        idx = main.build_title_stem_index(csm)
        self.assertEqual(idx.get(main._model_stem_basename("LHBP_Terminal_Ext_Door_001")),
                         "22b74d64_LHBP_Terminal_Ext_Door_001_LOD0")

    def test_static_variant_wins_a_collision(self):
        csm = {
            "22b74d64_LHBP_Terminal_Ext_Door_004_LOD0": ["x"],
            "LHBP_Terminal_Ext_Door_004_Static_eca2fdd1863bd04391d2ba7d3b1f4903": ["y"],
        }
        idx = main.build_title_stem_index(csm)
        self.assertEqual(idx["lhbp_terminal_ext_door_004"],
                         "LHBP_Terminal_Ext_Door_004_Static_eca2fdd1863bd04391d2ba7d3b1f4903")

    def test_unrelated_title_does_not_resolve(self):
        idx = main.build_title_stem_index({"22b74d64_LHBP_Terminal_Ext_Door_001_LOD0": ["a"]})
        self.assertIsNone(idx.get(main._model_stem_basename("SHS_Clutter_Snow_001")))


if __name__ == "__main__":
    unittest.main()
