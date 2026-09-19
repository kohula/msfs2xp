"""
Self-invalidating on-disk cache for msfs2xp.

Persists under a "_cache" folder next to this script, independent of
whatever output folder is picked for a given run -- the same source
package gets re-run against the same cache across sessions, which matters
a lot while iterating on bug fixes.

Deliberately NOT under %LOCALAPPDATA% (its original location): this cache
holds a full copy of every converted .obj/texture, which grows into the
tens of GB over repeated runs -- fine on a spacious secondary drive next
to the project, but exactly the kind of thing that quietly fills a small
C: system drive if left there. Living next to the scripts also makes it
obvious where the space is going and easy to delete by hand.

The one property that makes this safe to leave on by default: every cache
key folds in a content hash of the *processing module's own source file*
(see module_version()) in addition to the input file's identity. Edit
bgl_extractor.py or any mesh_convert/ submodule to fix a bug, and every
cache entry that depended on the old logic is automatically invalidated --
there is no way for this cache to keep serving pre-fix output after the
code changes.

Delete the cache_root() folder by hand at any time to force a fully clean
run.
"""

import hashlib
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

_MODULE_HASH_CACHE = {}

# GPU/_cache -- next to this script, not on whatever drive the system temp/
# profile folder happens to live on. Under a frozen PyInstaller EXE,
# __file__ resolves inside the ephemeral extraction temp dir (sys._MEIPASS)
# rather than where the real .exe sits on disk, so that case is anchored
# on sys.executable's own folder instead.
if getattr(sys, "frozen", False):
    _SCRIPT_DIR = Path(sys.executable).resolve().parent
else:
    _SCRIPT_DIR = Path(__file__).resolve().parent


def cache_root():
    root = _SCRIPT_DIR / "_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def module_version(module_path):
    """Content hash of a source file, memoized per-process. Folded into
    every cache key that depends on that module's logic -- see module
    docstring for why."""
    module_path = str(module_path)
    h = _MODULE_HASH_CACHE.get(module_path)
    if h is None:
        try:
            data = Path(module_path).read_bytes()
        except OSError:
            return "unknown"
        h = hashlib.blake2b(data, digest_size=8).hexdigest()
        _MODULE_HASH_CACHE[module_path] = h
    return h


def file_identity(path):
    """Cheap stand-in for a content hash: path + size + mtime. Hashing every
    .bgl/.glb's full bytes on every run would itself cost real time on a
    large package, and mtime already changes on any real edit/re-export."""
    st = os.stat(path)
    return f"{os.path.abspath(str(path))}|{st.st_size}|{st.st_mtime_ns}"


def _make_key(namespace, parts):
    raw = namespace + "||" + "||".join(str(p) for p in parts)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


def get(namespace, *parts):
    """Returns the cached value for this key, or None on a miss/corrupt entry."""
    key = _make_key(namespace, parts)
    meta_path = cache_root() / namespace / f"{key}.meta"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def set(namespace, value, *parts):
    """Stores value under this key. Written via a unique temp file + atomic
    os.replace() so concurrent worker processes never observe (or race to
    write) a partially-written cache entry."""
    key = _make_key(namespace, parts)
    ns_dir = cache_root() / namespace
    ns_dir.mkdir(parents=True, exist_ok=True)
    meta_path = ns_dir / f"{key}.meta"
    tmp_path = meta_path.with_name(f"{meta_path.name}.tmp_{os.getpid()}_{id(value)}")
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Windows can raise PermissionError on os.replace() if another
        # process momentarily has meta_path open (e.g. a concurrent get()
        # on the same key) -- retry briefly rather than losing this cache
        # write over a transient lock.
        for attempt in range(8):
            try:
                os.replace(tmp_path, meta_path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def enforce_size_cap(namespace, max_bytes, log_callback=None):
    """Evict a namespace's oldest entries (by mtime) until its total
    on-disk size is under max_bytes. A periodic sweep, not strict per-write
    enforcement -- meant to be called once per pipeline run, not on every
    set(), so it can't add per-call overhead to the hot path.

    Without this, `mesh_convert`'s cache never evicts and can grow
    unbounded, since its key includes the re-extracted temp .glb's mtime
    (a fresh one every run) -- this is a safety net for that disk-growth
    symptom, not a fix for the low hit rate itself."""
    ns_dir = cache_root() / namespace
    if not ns_dir.is_dir():
        return

    entries = []  # (mtime, path, size)
    for child in ns_dir.iterdir():
        try:
            if child.is_dir():
                size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
                mtime = child.stat().st_mtime
            else:
                st = child.stat()
                size, mtime = st.st_size, st.st_mtime
        except OSError:
            continue
        entries.append((mtime, child, size))

    total_size = sum(e[2] for e in entries)
    if total_size <= max_bytes:
        return

    entries.sort(key=lambda e: e[0])  # oldest first
    freed = 0
    removed = 0
    for _mtime, path, size in entries:
        if total_size - freed <= max_bytes:
            break
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except OSError:
            continue
        freed += size
        removed += 1

    if log_callback and removed:
        log_callback(
            f"Cache '{namespace}' was {total_size / 1e9:.1f} GB (over the "
            f"{max_bytes / 1e9:.0f} GB cap) -- evicted {removed} oldest entries, "
            f"freeing {freed / 1e9:.1f} GB.", "info")


def entry_dir(namespace, *parts):
    """A per-entry directory for caching output *files* (produced .obj,
    textures, extracted models, ...) alongside the pickled metadata above."""
    key = _make_key(namespace, parts)
    d = cache_root() / namespace / f"{key}_files"
    d.mkdir(parents=True, exist_ok=True)
    return d
