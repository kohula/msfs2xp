"""app_paths: writable data goes next to the program -- for an AppImage,
next to the .AppImage file, since the program itself runs from a
read-only mount -- and falls back to the per-user cache folder."""
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app_paths


class TestAppPaths(unittest.TestCase):
    def setUp(self):
        app_paths._DATA_ROOT = None

    def tearDown(self):
        app_paths._DATA_ROOT = None

    def test_appimage_data_goes_next_to_the_appimage_file(self):
        with tempfile.TemporaryDirectory() as td:
            image = Path(td) / "MSFS2XP-x86_64.AppImage"
            image.write_bytes(b"x")
            with mock.patch.dict(os.environ, {"APPIMAGE": str(image), "MSFS2XP_DATA_DIR": ""}):
                self.assertEqual(app_paths.program_dir(), Path(td).resolve())
                self.assertEqual(app_paths.data_root(), Path(td).resolve())
                self.assertEqual(app_paths.cache_dir(), Path(td).resolve() / "_cache")

    @unittest.skipIf(sys.platform == "win32" or os.geteuid() == 0, "needs POSIX permissions as non-root")
    def test_read_only_program_dir_falls_back_to_user_cache(self):
        with tempfile.TemporaryDirectory() as td:
            ro = Path(td) / "ro"
            ro.mkdir()
            (ro / "MSFS2XP.AppImage").write_bytes(b"x")
            ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
            try:
                env = {"APPIMAGE": str(ro / "MSFS2XP.AppImage"), "XDG_CACHE_HOME": str(Path(td) / "xdg"),
                       "MSFS2XP_DATA_DIR": ""}
                with mock.patch.dict(os.environ, env):
                    self.assertEqual(app_paths.data_root(), Path(td) / "xdg" / "msfs2xp")
            finally:
                ro.chmod(stat.S_IRWXU)

    def test_override(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(os.environ, {"MSFS2XP_DATA_DIR": td}):
                self.assertEqual(app_paths.data_root(), Path(td))
                self.assertEqual(app_paths.config_file(), Path(td) / "msfs2xp_config.json")

    def test_build_id_is_stable(self):
        self.assertEqual(app_paths.build_id(), app_paths.build_id())


if __name__ == "__main__":
    unittest.main()
