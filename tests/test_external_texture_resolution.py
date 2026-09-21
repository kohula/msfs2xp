"""
mesh_convert.convert.extract_image's external_textures_dir fallback --
used to resolve a base-game/library object's texture referenced by an
external glTF URI but not bundled with the converting package itself (see
main.py, which now points this at the user's configured MSFS install root
instead of aliasing it to the package's own output texture folder).

Also pins _external_texture_index's memoization: the directory should only
be walked once (per worker process), not once per missing texture.
"""
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder  # noqa: E402

import mesh_convert
# mesh_convert/__init__.py does "from .convert import convert", which
# rebinds the mesh_convert.convert ATTRIBUTE to the function itself --
# importlib.import_module reaches the real submodule object directly
# (via sys.modules) instead of that shadowed attribute.
convert_module = importlib.import_module("mesh_convert.convert")


class TestExternalTextureResolution(unittest.TestCase):
    def setUp(self):
        convert_module._EXTERNAL_TEXTURE_INDEX_CACHE.clear()

    def _build_glb_with_external_texture(self, path: Path, uri_filename: str):
        b = GltfBuilder()
        imgi = b.add_image_uri(uri_filename)  # NOT a data: URI -- must be resolved externally
        texi = b.add_texture(imgi)
        mat = b.add_material("Mat", base_color_texture_index=texi)
        positions = [(-5, 0, -5), (5, 0, -5), (5, 4, -5), (-5, 4, -5)]  # non-flat-eligible-ish, doesn't matter here
        normals = [(0.0, 0.0, -1.0)] * 4
        uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        indices = [0, 1, 2, 0, 2, 3]
        mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
        b.add_node(mesh_index=mesh, name="Thing")
        path.write_bytes(b.build())

    def test_texture_found_deep_in_external_dir_is_copied_not_placeholder(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "model.glb"
            self._build_glb_with_external_texture(glb_path, "libtexture.PNG")

            # Simulates a real MSFS install: the actual texture lives many
            # folders deep, nowhere near the model or its own parent dirs.
            external_root = td / "msfs_install"
            deep_dir = external_root / "Official" / "OneStore" / "some-package" / "texture"
            deep_dir.mkdir(parents=True)
            real_texture_path = deep_dir / "libtexture.png"
            Image.new("RGBA", (4, 4), (10, 20, 30, 255)).save(real_texture_path, "PNG")

            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, external_root, "0.0", "0.0", "0.0")
            self.assertTrue(result)

            png_files = list(tex_dir.glob("*.png"))
            self.assertTrue(png_files, "expected the resolved texture to be copied into tex_dir")
            with Image.open(png_files[0]) as img:
                copied = img.convert("RGBA")
                self.assertEqual(copied.size, (4, 4), "a 2x2 flat placeholder would mean the external lookup failed")
                self.assertEqual(copied.getpixel((0, 0)), (10, 20, 30, 255))

    def test_missing_external_texture_falls_back_to_placeholder(self):
        """No matching file anywhere under external_textures_dir -- must
        still degrade to the flat placeholder, not raise or hang."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "model.glb"
            self._build_glb_with_external_texture(glb_path, "totally_missing_texture.png")

            external_root = td / "msfs_install"
            (external_root / "some" / "other" / "stuff").mkdir(parents=True)

            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, external_root, "0.0", "0.0", "0.0")
            self.assertTrue(result)
            png_files = list(tex_dir.glob("*.png"))
            self.assertTrue(png_files)
            with Image.open(png_files[0]) as placeholder:
                self.assertEqual(placeholder.size, (2, 2), "expected the flat 2x2 placeholder")

    def test_multiple_roots_searched_in_priority_order(self):
        """external_textures_dir now accepts a LIST of roots (main.py
        passes [source_package_root, msfs_install_root] -- confirmed real
        need: a texture can be genuinely shipped in the package but not
        anywhere the model's own relative URI or sibling-folder search
        would find it, e.g. a stale dev-machine path or one copy-pasted
        from an unrelated package's folder layout, so the WHOLE package
        needs to be searchable, with the base-game install as a further
        fallback). The first root in the list must win when both have a
        same-named file, matching "package's own texture beats a same-
        named base-game one"."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "model.glb"
            self._build_glb_with_external_texture(glb_path, "libtexture.PNG")

            package_root = td / "package"
            (package_root / "Scenery" / "somewhere" / "unexpected").mkdir(parents=True)
            Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(
                package_root / "Scenery" / "somewhere" / "unexpected" / "libtexture.png", "PNG")

            msfs_root = td / "msfs_install"
            (msfs_root / "Official" / "OneStore" / "some-package" / "texture").mkdir(parents=True)
            Image.new("RGBA", (4, 4), (0, 255, 0, 255)).save(
                msfs_root / "Official" / "OneStore" / "some-package" / "texture" / "libtexture.png", "PNG")

            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, [package_root, msfs_root], "0.0", "0.0", "0.0")
            self.assertTrue(result)
            png_files = list(tex_dir.glob("*.png"))
            self.assertTrue(png_files)
            with Image.open(png_files[0]) as img:
                self.assertEqual(img.convert("RGBA").getpixel((0, 0)), (255, 0, 0, 255),
                                  "the package root (listed first) must win over the base-game root")

    def test_texture_only_in_second_root_is_still_found(self):
        """Proves BOTH roots actually get searched, not just the first --
        a texture genuinely missing from the package but present in the
        configured MSFS install must still resolve."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "model.glb"
            self._build_glb_with_external_texture(glb_path, "libtexture.PNG")

            package_root = td / "package"
            package_root.mkdir()  # exists, but doesn't contain the texture at all

            msfs_root = td / "msfs_install"
            (msfs_root / "Official" / "texture").mkdir(parents=True)
            Image.new("RGBA", (4, 4), (0, 0, 255, 255)).save(
                msfs_root / "Official" / "texture" / "libtexture.png", "PNG")

            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, [package_root, msfs_root], "0.0", "0.0", "0.0")
            self.assertTrue(result)
            png_files = list(tex_dir.glob("*.png"))
            self.assertTrue(png_files)
            with Image.open(png_files[0]) as img:
                self.assertEqual(img.convert("RGBA").getpixel((0, 0)), (0, 0, 255, 255),
                                  "expected the second root's texture, not a placeholder")

    def test_duplicate_name_suffix_on_image_name_does_not_break_lookup(self):
        """Confirmed real bug from a live EGLC conversion: Blender/DCC
        exporters append a ".NNN" duplicate-name disambiguator AFTER the
        real extension when an image name is reused by multiple materials
        (e.g. glTF image name "ini_Kit_Lights_ALBD.png.001", uri some
        broken relative path). base_stem prefers the image's "name" field
        over its "uri", and clean_texture_stem's known-extension list
        didn't recognize ".png.001" as strippable, leaving the numeric
        suffix stuck onto the lookup key so it could never match the real
        file's clean stem ("ini_kit_lights_albd") in the external index --
        even though that real file was sitting right there in the
        package. Only images whose name has this disambiguator suffix are
        affected; plain names (no suffix) already worked."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            b = GltfBuilder()
            imgi = b.add_image_uri(
                "..\\..\\..\\..\\..\\INIMATERIALS 2024\\MODELS\\SOMEWHERE\\TEXTURES\\libtexture_ALBD.PNG.KTX2",
                name="libtexture_ALBD.png.001",
            )
            texi = b.add_texture(imgi)
            mat = b.add_material("Mat", base_color_texture_index=texi)
            positions = [(-5, 0, -5), (5, 0, -5), (5, 4, -5), (-5, 4, -5)]
            normals = [(0.0, 0.0, -1.0)] * 4
            uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            indices = [0, 1, 2, 0, 2, 3]
            mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
            b.add_node(mesh_index=mesh, name="Thing")
            glb_path = td / "model.glb"
            glb_path.write_bytes(b.build())

            package_root = td / "package"
            (package_root / "Scenery" / "modellib" / "texture").mkdir(parents=True)
            Image.new("RGBA", (4, 4), (200, 100, 50, 255)).save(
                package_root / "Scenery" / "modellib" / "texture" / "libtexture_ALBD.PNG.KTX2", "PNG")

            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, [package_root], "0.0", "0.0", "0.0")
            self.assertTrue(result)
            png_files = list(tex_dir.glob("*.png"))
            self.assertTrue(png_files)
            with Image.open(png_files[0]) as img:
                self.assertEqual(img.size, (4, 4), "a 2x2 flat placeholder means the duplicate-suffix name broke the lookup")
                self.assertEqual(img.convert("RGBA").getpixel((0, 0)), (200, 100, 50, 255))

    def test_external_index_is_built_only_once_per_directory(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            external_root = td / "msfs_install"
            (external_root / "sub").mkdir(parents=True)
            Image.new("RGBA", (2, 2), (1, 2, 3, 255)).save(external_root / "sub" / "tex.png", "PNG")

            index1 = convert_module._external_texture_index(external_root)
            self.assertIn("tex", index1)

            # Delete the directory entirely -- if the second call re-walked
            # the filesystem instead of returning the memoized index, it
            # would come back empty (or raise).
            (external_root / "sub" / "tex.png").unlink()

            index2 = convert_module._external_texture_index(external_root)
            self.assertIs(index1, index2, "expected the memoized dict object to be reused, not rebuilt")
            self.assertIn("tex", index2)


def _build_fake_ktx2_bc1(width, height):
    """A minimal, structurally-valid single-level BC1 KTX2 file --
    enough for decode_ktx2_bytes_to_png to actually decode it (unlike
    test_ktx2_dds_repackage.py's own _build_fake_ktx2, whose fixtures
    only need to parse, not decode)."""
    import struct
    blocks_w, blocks_h = max(1, (width + 3) // 4), max(1, (height + 3) // 4)
    compressed = b"\x00\xf8\xff\x07" + b"\x00" * 4  # one real, decodable BC1 block, repeated
    compressed = compressed * (blocks_w * blocks_h)
    magic = b"\xabKTX 20\xbb\r\n\x1a\n"
    vk_format_bc1_rgba = 133
    header = struct.pack("<17I", vk_format_bc1_rgba, 1, width, height, 0, 1, 1, 1, 0,
                          0, 0, 0, 0, 0, 0, 0, 0)
    offset = len(magic) + len(header) + 24
    level_index = struct.pack("<3Q", offset, len(compressed), len(compressed))
    return magic + header + level_index + compressed


class TestGuessTextureByMaterialName(unittest.TestCase):
    """convert()'s "no base color texture at all -- guess one from the
    material's own name" fallback (mesh_convert/convert.py, right after
    the normal-map extraction). CONFIRMED REAL CRASH this pins: the
    matched file came from _find_in_external_texture_roots, which
    indexes EVERY recognized extension (_EXTERNAL_TEXTURE_EXTENSIONS
    includes .ktx2/.dds/.tga/...) by clean stem -- a raw shutil.copyfile
    straight to a ".png"-named destination assumed the match was
    already a real PNG just because the DESTINATION happened to be
    named that. A real, undecoded .ktx2 file copied verbatim to a
    ".png" path is not a valid PNG at all -- confirmed in real X-Plane
    output: a hard crash ("THREAD FATAL ASSERT", a real IDAT CRC error)
    trying to load one."""

    def test_matched_ktx2_is_decoded_not_raw_copied(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            b = GltfBuilder()
            # No base_color_texture_index at all -- forces the
            # guess-by-material-name fallback.
            mat = b.add_material("FloodlightEmis")
            positions = [(-2, 0, -2), (2, 0, -2), (2, 0, 2), (-2, 0, 2)]
            normals = [(0.0, 1.0, 0.0)] * 4
            uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            indices = [0, 1, 2, 0, 2, 3]
            mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
            b.add_node(mesh_index=mesh, name="Light")

            glb_path = td / "model.glb"
            glb_path.write_bytes(b.build())

            external_root = td / "package"
            tex_src_dir = external_root / "texture"
            tex_src_dir.mkdir(parents=True)
            (tex_src_dir / "FloodlightEmis.ktx2").write_bytes(_build_fake_ktx2_bc1(4, 4))

            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, external_root, "0.0", "0.0", "0.0")
            self.assertTrue(result)

            out_path = tex_dir / "floodlightemis.png"
            self.assertTrue(out_path.exists(), "expected the guessed texture to be written")
            self.assertTrue(out_path.read_bytes().startswith(b"\x89PNG"),
                             "must be a real decoded PNG, not the raw KTX2 bytes copied verbatim")
            with Image.open(out_path) as img:
                img.load()  # raises if the PNG is actually corrupt


if __name__ == "__main__":
    unittest.main()
