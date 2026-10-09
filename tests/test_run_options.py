"""
Run options around the conversion itself: the .log file
(PipelineOptions.write_log_file / --log-file), switching exclusion zones
off (--no-exclusions) and the version shown in the window and --version.
"""
import tempfile
import unittest
from pathlib import Path

import cli
import pipeline


class _Quiet(pipeline.PipelineHooks):
    def __init__(self):
        self.lines = []

    def log(self, text, level="info"):
        self.lines.append((level, text))


class TestLogFile(unittest.TestCase):
    def test_log_lines_and_options_are_written_even_when_the_run_fails(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "Custom Scenery" / "Test Airport"
            opts = pipeline.PipelineOptions(pkg_dir=str(Path(td) / "missing"), out_dir=str(out),
                                            write_log_file=True)
            hooks = _Quiet()
            original = pipeline._run_pipeline

            def fake(o, h):
                h.log("step one", "info")
                h.log("something odd", "warning")
                raise RuntimeError("boom")

            pipeline._run_pipeline = fake
            try:
                with self.assertRaises(RuntimeError):
                    pipeline.run_pipeline(opts, hooks)
            finally:
                pipeline._run_pipeline = original
            text = (out / pipeline.LOG_FILE_NAME).read_text(encoding="utf-8")
            self.assertIn("step one", text)
            self.assertIn("[warning] something odd", text)
            self.assertIn("Pipeline failed: boom", text)
            self.assertIn("write_log_file=True", text)
            self.assertIn(pipeline.app_version(), text.splitlines()[0])
            # the caller's own hooks still saw every line
            self.assertIn(("warning", "something odd"), hooks.lines)

    def test_no_file_unless_asked(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "Pack"
            opts = pipeline.PipelineOptions(pkg_dir=td, out_dir=str(out))
            original = pipeline._run_pipeline
            pipeline._run_pipeline = lambda o, h: h.log("hi")
            try:
                pipeline.run_pipeline(opts, _Quiet())
            finally:
                pipeline._run_pipeline = original
            self.assertFalse((out / pipeline.LOG_FILE_NAME).exists())


class TestCliSwitches(unittest.TestCase):
    def test_defaults(self):
        a = cli.build_parser().parse_args(["pkg", "-o", "out"])
        self.assertFalse(a.no_exclusions)
        self.assertFalse(a.log_file)

    def test_switches(self):
        a = cli.build_parser().parse_args(["pkg", "-o", "out", "--no-exclusions", "--log-file"])
        self.assertTrue(a.no_exclusions)
        self.assertTrue(a.log_file)


class TestVersion(unittest.TestCase):
    def test_version_comes_from_the_release_file(self):
        expected = (Path(pipeline.__file__).resolve().parent / "packaging" / "VERSION").read_text().strip()
        self.assertEqual(pipeline.app_version(), expected)


if __name__ == "__main__":
    unittest.main()
