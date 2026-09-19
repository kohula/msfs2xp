"""
Any BLEND-alpha material (glass panes and other translucent surfaces --
tinted plastic, fabric, ...) must render double-sided in X-Plane, even
when the source glTF material is authored single-sided (MSFS relies on its
own shader to draw both faces anyway) -- otherwise the surface backface-
culls and fully hides whatever's behind it (a moving bus/GSE vehicle seen
through a terminal window) from one viewing direction.
"""
import base64
import io
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder, flat_quad  # noqa: E402

import mesh_convert


class TestDoubleSidedGlass(unittest.TestCase):
    def _convert_pane(self, material_name, alpha_mode, extensions=None, double_sided_in_source=False):
        b = GltfBuilder()
        tex = b.add_texture(b.add_image_data_uri((150, 150, 220, 180)))
        mat = b.add_material(material_name, base_color_texture_index=tex, alpha_mode=alpha_mode,
                              extensions=extensions)
        if not double_sided_in_source:
            # add_material has no doubleSided param -- glTF materials
            # default to single-sided (doubleSided omitted/false) unless
            # explicitly set, which is exactly the case being tested.
            pass
        positions, normals, uvs, indices = flat_quad(0, 2, 0, 2)
        mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
        b.add_node(mesh_index=mesh, name="Pane")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "pane.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()
            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)
            return result[0].read_text(encoding="utf-8")

    def test_asobo_glass_extension_material_is_double_sided(self):
        text = self._convert_pane("WindowGlass", "OPAQUE", extensions={"ASOBO_material_glass_v2": {}})
        self.assertIn("ATTR_no_cull", text)

    def test_plain_blend_alpha_material_without_glass_extension_is_double_sided(self):
        """The confirmed real gap: a genuinely translucent material with no
        ASOBO glass extension and no "glass" in its name at all -- the
        is_glass heuristic alone would miss this, but it's exactly as
        prone to the one-sided backface-cull artifact."""
        text = self._convert_pane("TintedPlastic", "BLEND")
        self.assertIn("ATTR_no_cull", text)

    def test_opaque_material_stays_single_sided(self):
        """Confirms this isn't a blanket double-sided-everything change --
        only alpha-blended (translucent) materials are affected."""
        text = self._convert_pane("PlainWall", "OPAQUE")
        self.assertNotIn("ATTR_no_cull", text)


class TestGlassAlphaModeRespectsSource(unittest.TestCase):
    """Confirmed real bug from a live EGLC package: an earlier version
    forced ANY "glass"-named/extension material into ATTR_blend
    regardless of its own authored alphaMode -- in TWO independent spots
    (the alpha_mode field assignment, and a separate is_glass check at
    OBJ8 write time that bypassed alpha_mode entirely). A real EGLC
    package ships dozens of "glass"-named materials deliberately authored
    OPAQUE (reflective/painted glass, e.g. "MT_GlassBlack") or MASK, some
    tagged ASOBO_material_invisible (LOD/collision placeholders never
    meant to render as glass at all) -- forcing all of them into BLEND
    overrode real artist intent, in both directions (some materials that
    should render solid ended up wrongly translucent, and vice versa via
    knock-on effects on which texture/alpha got baked in)."""

    def _convert_pane(self, material_name, alpha_mode, extensions=None, base_color_factor=None):
        b = GltfBuilder()
        tex = b.add_texture(b.add_image_data_uri((150, 150, 220, 180)))
        mat = b.add_material(material_name, base_color_texture_index=tex, alpha_mode=alpha_mode,
                              extensions=extensions, base_color_factor=base_color_factor)
        positions, normals, uvs, indices = flat_quad(0, 2, 0, 2)
        mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
        b.add_node(mesh_index=mesh, name="Pane")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "pane.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()
            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)
            return result[0].read_text(encoding="utf-8")

    def test_glass_named_material_authored_opaque_stays_opaque(self):
        text = self._convert_pane("MT_GlassBlack", "OPAQUE", base_color_factor=[0.04, 0.04, 0.04, 1.0])
        self.assertIn("ATTR_no_blend 0.5", text)
        self.assertNotIn("ATTR_blend\n", text)

    def test_glass_extension_material_authored_opaque_stays_opaque(self):
        text = self._convert_pane("ini_Glass_Opaque", "OPAQUE", extensions={"ASOBO_material_glass_v2": {}})
        self.assertIn("ATTR_no_blend 0.5", text)
        self.assertNotIn("ATTR_blend\n", text)

    def test_glass_named_material_authored_mask_stays_mask(self):
        text = self._convert_pane("ini_Glass_parallax_free", "MASK")
        self.assertNotIn("ATTR_blend\n", text)
        self.assertIn("ATTR_no_blend 0.500", text)  # MASK's own alphaCutoff-based line, not the OPAQUE fallback

    def test_glass_named_material_authored_blend_still_gets_alpha_floor(self):
        """The original, already-correct case must keep working: a real
        ASOBO glass material genuinely authored BLEND, with no useful
        alpha in its own baseColorFactor (glTF default 1.0/opaque) --
        still needs the translucency floor since the real transparency
        pattern lives in the texture's own alpha channel."""
        text = self._convert_pane("EGLC_Terminal_Glass", "BLEND", extensions={"ASOBO_material_glass_v2": {}})
        self.assertIn("ATTR_blend\n", text)
        self.assertIn("ATTR_shiny_rat 1.0", text)

    def test_parallax_window_material_converted_to_blend(self):
        """ASOBO_material_parallax_window fakes an interior room's depth
        entirely via shader trickery X-Plane has no equivalent for --
        left as its authored MASK, it would show as a flat opaque "fake
        room" picture (or a binary cutout of one) instead of real
        see-through glass. Converting to BLEND is closer to a real pane's
        look than either."""
        text = self._convert_pane("EGLC_Newham_Council_Windows", "MASK",
                                   extensions={"ASOBO_material_parallax_window": {}})
        self.assertIn("ATTR_blend\n", text)


class TestBlendDowngradeForEffectivelyOpaqueMaterials(unittest.TestCase):
    """Confirmed real, widespread pattern in a live SoFly/MSFS-2024-
    exported LHBP package: hundreds of materials -- plain walls, frames,
    generic "Material.004"-style unnamed ones, not just glass -- are
    authored alphaMode BLEND with baseColorFactor alpha at the glTF
    default of 1.0 (fully opaque), and no real transparency in their own
    texture's alpha channel either. That exporter appears to stamp BLEND
    near-universally rather than using it to signal real translucency,
    unlike the EGLC package the "respect alphaMode" fix was originally
    verified against. Routing solid walls through X-Plane's alpha-blend
    (depth-sort-dependent) render path instead of the opaque one is
    exactly what produces objects incorrectly disappearing behind things
    that were never meant to be transparent at all."""

    def _build_uniform_alpha_texture_uri(self, alpha=255):
        return GltfBuilder.make_data_uri((180, 180, 180, alpha), size=(4, 4))

    def _build_punch_through_texture_uri(self):
        """A real glass-pane-style texture: half the pixels fully
        transparent (the pane), half fully opaque (the frame) -- the
        actual shape EGLC_Terminal_Glass's real texture has."""
        img = Image.new("RGBA", (4, 4), (200, 200, 200, 255))
        for x in range(2):
            for y in range(4):
                img.putpixel((x, y), (200, 200, 200, 0))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _convert_pane(self, material_name, base_color_factor=None, texture_uri=None):
        b = GltfBuilder()
        texi = None
        if texture_uri is not None:
            imgi = b.add_image_uri(texture_uri)
            texi = b.add_texture(imgi)
        mat = b.add_material(material_name, base_color_texture_index=texi, alpha_mode="BLEND",
                              base_color_factor=base_color_factor)
        positions, normals, uvs, indices = flat_quad(0, 2, 0, 2)
        mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
        b.add_node(mesh_index=mesh, name="Pane")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "pane.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()
            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)
            return result[0].read_text(encoding="utf-8")

    def test_blend_material_opaque_factor_no_texture_downgrades_to_opaque(self):
        """"Wall.004"-shaped case: BLEND, baseColorFactor alpha 1.0, no
        texture at all -- nothing anywhere signals real transparency."""
        text = self._convert_pane("Wall", base_color_factor=[0.7, 0.7, 0.7, 1.0])
        self.assertIn("ATTR_no_blend 0.5", text)
        self.assertNotIn("ATTR_blend\n", text)

    def test_blend_material_default_factor_uniform_opaque_texture_downgrades(self):
        """No baseColorFactor at all (glTF default 1.0) and a texture
        whose own alpha channel is uniformly opaque -- still no real
        transparency signal anywhere."""
        text = self._convert_pane("Frame", texture_uri=self._build_uniform_alpha_texture_uri(255))
        self.assertIn("ATTR_no_blend 0.5", text)
        self.assertNotIn("ATTR_blend\n", text)

    def test_blend_material_opaque_factor_but_real_texture_transparency_stays_blend(self):
        """Opaque baseColorFactor (glTF default) but a texture with a real
        punch-through pane/frame alpha pattern -- exactly
        EGLC_Terminal_Glass's own shape. Must stay BLEND: the real
        transparency signal lives in the texture, not the factor."""
        text = self._convert_pane("EGLC_Terminal_Glass", texture_uri=self._build_punch_through_texture_uri())
        self.assertIn("ATTR_blend\n", text)
        self.assertNotIn("ATTR_no_blend 0.5", text)

    def test_blend_material_genuinely_translucent_factor_stays_blend(self):
        """The already-working case must keep working: a real translucent
        factor alpha (well below opaque) is respected regardless of any
        texture."""
        text = self._convert_pane("Glass transparent", base_color_factor=[0.79, 0.70, 0.63, 0.51])
        self.assertIn("ATTR_blend\n", text)


if __name__ == "__main__":
    unittest.main()
