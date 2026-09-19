"""
Single entry point for the regression suite: python tests/run_tests.py

Discovers and runs every tests/test_*.py, prints a summary, exits nonzero
on any failure -- the concrete mechanism for "review file formats
continuously": run this after any future change to re-verify every
previously-fixed bug's regression case in well under a minute.
"""
import sys
import unittest
from pathlib import Path

if __name__ == "__main__":
    start_dir = Path(__file__).resolve().parent
    project_root = start_dir.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(start_dir))

    suite = unittest.TestLoader().discover(start_dir=str(start_dir), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
