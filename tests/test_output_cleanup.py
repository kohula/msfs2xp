"""pipeline.clean_previous_output: a re-run into the same pack must not
keep last run's DSF tiles, apt.dat, objects or textures around."""
import tempfile
import unittest
from pathlib import Path

import pipeline


class TestCleanPreviousOutput(unittest.TestCase):
    def test_removes_generated_files_only(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "My Airport"
            files = [
                "Earth nav data/+40+010/+47+019.dsf",
                "Earth nav data/apt.dat",
                "Earth nav data/apt.dat.xp11",
                "objects/terminal_0.obj",
                "objects/terminal.originoffset.json",
                "objects/terminal_0.meshir.pkl",
                "polygons/asphalt.pol",
                "textures/asphalt.png",
                "textures/roof.dds",
                "plugin_data/msfs2xp_proximity.dat",
            ]
            keep = ["README.txt", "objects/notes.txt", "Earth nav data/readme.md", "plugins/x.lua"]
            for f in files + keep:
                (out / f).parent.mkdir(parents=True, exist_ok=True)
                (out / f).write_text("x")
            removed = pipeline.clean_previous_output(out, lambda *a: None)
            self.assertEqual(removed, len(files))
            for f in files:
                self.assertFalse((out / f).exists(), f)
            for f in keep:
                self.assertTrue((out / f).exists(), f)
            self.assertFalse((out / "Earth nav data" / "+40+010").exists(), "emptied tile folder removed")

    def test_refuses_custom_scenery_itself(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "Custom Scenery"
            out.mkdir()
            opts = pipeline.PipelineOptions(pkg_dir=td, out_dir=str(out))
            with self.assertRaises(ValueError):
                pipeline.run_pipeline(opts, pipeline.PipelineHooks())


if __name__ == "__main__":
    unittest.main()
