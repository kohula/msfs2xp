"""
Confirmed real bug from a live conversion: X-Plane crashed outright
("oversubscribed dynamic bit lengths tree" -- a broken zlib/deflate
stream) reading a texture PNG that was genuinely corrupted on disk. Every
"does this output already exist" check in this pipeline used to trust
bare file existence (plus a minimum size) alone, at TWO separate layers:

  - mesh_convert.convert's own per-run reuse checks (extract_image's early
    short-circuit, its external-root candidate match, its final fallback-
    stub gate, and apply_color_factor/apply_alpha_factor/apply_emissive_
    factor's "already computed" checks).
  - main.cached_convert's persistent ACROSS-RUN disk cache (cache_utils),
    an entirely separate layer above mesh_convert.convert -- a cache HIT
    skips calling convert() at all, so fixing validation inside that
    module alone could never repair a bad file already sitting in this
    cache store from an earlier run.

Both must independently detect and refuse to trust a corrupted file,
rather than perpetuating it forever once it exists.
"""
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder  # noqa: E402

import cache_utils
import main
import mesh_convert
convert_module = importlib.import_module("mesh_convert.convert")


def _build_textured_glb(path: Path):
    b = GltfBuilder()
    imgi = b.add_image_data_uri((200, 120, 60, 255), name="BodyTex")
    texi = b.add_texture(imgi)
    mat = b.add_material("BodyMat", base_color_texture_index=texi)
    positions = [(-2, 0, -2), (2, 0, -2), (2, 3, -2), (-2, 3, -2)]
    normals = [(0.0, 0.0, -1.0)] * 4
    uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    indices = [0, 1, 2, 0, 2, 3]
    mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
    b.add_node(mesh_index=mesh, name="Body")
    path.write_bytes(b.build())


class TestConvertLevelTextureValidity(unittest.TestCase):
    def setUp(self):
        convert_module._EXTERNAL_TEXTURE_INDEX_CACHE.clear()

    def test_corrupted_existing_texture_is_regenerated_not_trusted(self):
        """extract_image's early "already extracted" short-circuit must not
        trust a file that merely exists and is over the size floor -- if it
        fails to actually decode, convert() must regenerate it instead of
        silently returning the broken file as if it were done."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "model.glb"
            _build_textured_glb(glb_path)

            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            # Plant a corrupted file at the exact path convert() will want
            # to write its BodyMat texture to, mimicking a file left behind
            # by some earlier interrupted/corrupted run.
            corrupt_path = tex_dir / "bodytex.png"
            corrupt_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 500)
            self.assertFalse(convert_module._is_valid_image(corrupt_path),
                              "test setup issue: this planted file should already be unreadable")

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, None, "0.0", "0.0", "0.0")
            self.assertTrue(result)

            with Image.open(corrupt_path) as img:
                img.load()
                self.assertEqual(img.size, (2, 2), "expected the real BodyTex flat-color image, not the planted junk")

    def test_valid_existing_texture_is_still_reused(self):
        """The opposite case: a genuinely valid pre-existing texture must
        still be trusted and left alone (no wasted re-decode work) -- the
        fix only needs to catch actually-broken files."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "model.glb"
            _build_textured_glb(glb_path)

            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            real_path = tex_dir / "bodytex.png"
            Image.new("RGBA", (64, 64), (9, 9, 9, 255)).save(real_path, "PNG")
            mtime_before = real_path.stat().st_mtime_ns

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, None, "0.0", "0.0", "0.0")
            self.assertTrue(result)
            self.assertEqual(real_path.stat().st_mtime_ns, mtime_before,
                              "a genuinely valid pre-existing texture should not have been rewritten")


class TestCachedConvertTextureValidity(unittest.TestCase):
    """main.cached_convert's own persistent disk cache -- a layer entirely
    above mesh_convert.convert. Uses a scratch cache_root() so this never
    touches the project's real _cache folder."""

    def setUp(self):
        self._scratch_dir = tempfile.TemporaryDirectory()
        self._real_cache_root = cache_utils.cache_root
        scratch_root = Path(self._scratch_dir.name) / "_cache"
        cache_utils.cache_root = lambda: scratch_root if scratch_root.mkdir(parents=True, exist_ok=True) or True else scratch_root
        convert_module._EXTERNAL_TEXTURE_INDEX_CACHE.clear()

    def tearDown(self):
        cache_utils.cache_root = self._real_cache_root
        self._scratch_dir.cleanup()

    def test_corrupted_cache_entry_is_not_replayed_forever(self):
        """The confirmed real bug: once a texture got cached (whatever the
        original cause of its corruption), main.cached_convert's own cache
        HIT path used to just shutil.copy2 it straight into every future
        run's tex_dir with zero validation -- completely bypassing
        mesh_convert.convert's own (now-fixed) checks, since a cache hit
        never calls convert() at all. A corrupted cache entry must be
        treated as a miss and reconverted."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "model.glb"
            _build_textured_glb(glb_path)

            obj_dir1 = td / "objects1"
            tex_dir1 = td / "textures1"
            obj_dir1.mkdir()
            tex_dir1.mkdir()

            main.cached_convert(glb_path, obj_dir1, tex_dir1, None, "0.0", "0.0", "0.0")

            # Corrupt every texture file this populated the cache store
            # with, simulating a bad file that made it into the cache from
            # some earlier interrupted/corrupted run.
            cache_root = cache_utils.cache_root()
            mesh_convert_ns = cache_root / "mesh_convert"
            corrupted_any = False
            for files_dir in mesh_convert_ns.glob("*_files"):
                for png in files_dir.glob("*.png"):
                    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 500)
                    corrupted_any = True
            self.assertTrue(corrupted_any, "test setup issue: expected at least one cached texture to corrupt")

            obj_dir2 = td / "objects2"
            tex_dir2 = td / "textures2"
            obj_dir2.mkdir()
            tex_dir2.mkdir()

            main.cached_convert(glb_path, obj_dir2, tex_dir2, None, "0.0", "0.0", "0.0")

            png_files = list(tex_dir2.glob("*.png"))
            self.assertTrue(png_files, "expected a texture to have been (re)produced in the second run")
            for p in png_files:
                with Image.open(p) as img:
                    img.load()  # must not raise -- a corrupted cache entry must never be trusted


if __name__ == "__main__":
    unittest.main()
