"""
The confirmed real gap: inibuilds' EGLC package ships textures authored as
.TIF before KTX2 compression (e.g. "INI_ASPHALT_013_ALB.TIF.KTX2"), unlike
every previously-tested package's ".PNG.KTX2" convention -- both
clean_texture_stem (mesh_convert/convert.py, resolves a glTF material's
texture URI to the already-decoded file) and clean_texture_name (main.py,
names the decoded output during Step 2's bulk KTX2 pre-decode pass) lacked
.tif/.tif.ktx2/.tif.dds in their compound-extension list, leaving a stray
".tif" stuck in the middle of the cleaned name and breaking the match
between the two independently-computed names for the same texture.
"""
import unittest

from mesh_convert.convert import clean_texture_stem


class TestTextureExtensionCleaning(unittest.TestCase):
    def test_tif_ktx2_compound_extension_is_fully_stripped(self):
        self.assertEqual(clean_texture_stem("INI_ASPHALT_013_ALB.TIF.KTX2"), "ini_asphalt_013_alb")

    def test_tif_dds_compound_extension_is_fully_stripped(self):
        self.assertEqual(clean_texture_stem("Some_Texture.TIF.DDS"), "some_texture")

    def test_bare_tif_extension_is_stripped(self):
        self.assertEqual(clean_texture_stem("plain.tif"), "plain")

    def test_bare_tiff_extension_is_stripped(self):
        self.assertEqual(clean_texture_stem("plain.tiff"), "plain")

    def test_png_ktx2_still_works(self):
        """Confirms adding the new .tif variants didn't disturb the
        pre-existing, already-tested-in-production .png.ktx2 case."""
        self.assertEqual(clean_texture_stem("SHS_STUCCO_002_ALBD.PNG.KTX2"), "shs_stucco_002_albd")


if __name__ == "__main__":
    unittest.main()
