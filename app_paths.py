"""Where msfs2xp keeps its writable data: disk cache, scratch space and
settings.

By design that's next to the program itself -- the cache grows into the
tens of GB, and living next to the app keeps it off a small system drive
and makes it obvious where the space went:

- from source: next to the .py files;
- a Windows .exe: next to the .exe;
- an AppImage: next to the .AppImage FILE. The program inside an AppImage
  runs from a read-only mount (/tmp/.mount_XXXX/opt/...), so anchoring on
  sys.executable there pointed every write at a read-only folder -- the
  reason the AppImage only worked after --appimage-extract. The AppImage
  runtime exports the real file's path as $APPIMAGE.

If that folder isn't writable (an installed copy under /opt or Program
Files, a read-only share), the per-user cache folder is used instead:
$XDG_CACHE_HOME/msfs2xp (~/.cache/msfs2xp) on Linux, %LOCALAPPDATA%\\msfs2xp
on Windows. MSFS2XP_DATA_DIR overrides all of this.
"""

import hashlib
import os
import sys
import tempfile
from pathlib import Path

_DATA_ROOT = None


def program_dir():
    """The folder the program is launched from (see module docstring)."""
    appimage = os.environ.get("APPIMAGE")
    if appimage:
        return Path(appimage).resolve().parent
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _is_writable(folder):
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".msfs2xp_write_test_"):
            pass
        return True
    except OSError:
        return False


def _user_cache_dir():
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "msfs2xp"


def data_root():
    """Writable folder holding _cache/, _temp/ and the settings file."""
    global _DATA_ROOT
    if _DATA_ROOT is None:
        override = os.environ.get("MSFS2XP_DATA_DIR", "").strip()
        candidates = ([Path(override)] if override else []) + [program_dir(), _user_cache_dir()]
        _DATA_ROOT = next((c for c in candidates if _is_writable(c)), Path(tempfile.gettempdir()) / "msfs2xp")
        _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return _DATA_ROOT


def cache_dir():
    return data_root() / "_cache"


def temp_dir():
    return data_root() / "_temp"


def config_file():
    return data_root() / "msfs2xp_config.json"


def build_id():
    """Identity of the running build, for cache keys. From source the
    cache keys on each module's own content hash; a frozen build has no
    .py files to hash, so it keys on the executable (or AppImage) file
    itself -- a new release then never reuses an old release's cache."""
    exe = os.environ.get("APPIMAGE") or sys.executable
    try:
        st = os.stat(exe)
        ident = f"{exe}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        ident = exe
    return hashlib.blake2b(ident.encode("utf-8"), digest_size=8).hexdigest()
