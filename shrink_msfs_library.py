#!/usr/bin/env python3
"""
Shrinks the MSFS installation down to the small subset of packages/files
msfs2xp_next's own base-game/library resolution (bgl_extractor.py's
build_install_guid_index/build_install_simobject_index, mesh_convert's
extract_image external-texture fallback) can ever actually read, and
7z-compresses the result -- so a user can point --msfs-install at this
small archive's extracted contents instead of a 90GB+ full install.

NOT scoped to any one airport package -- built from MSFS's own package
CATALOG (every package msfs2xp_next's install-wide scanning could touch
for ANY future conversion, not just whatever's currently being tested),
confirmed against a real install this session:

  - bgl_extractor.build_install_guid_index/build_install_simobject_index
    both do a full RECURSIVE scan from the install root for *.bgl and
    sim.cfg files respectively -- every package could theoretically
    contribute a GUID a converted package's placements reference.
  - In practice, only a small, well-defined set of packages ever provide
    the kind of GENERIC/REUSABLE objects a THIRD-PARTY airport package
    would reference by GUID (generic lights, windsocks, jetways, GSE,
    characters, generic building/prop kit pieces) -- real, hand-crafted
    real-world airport packages (asobo-airport-kjfk-new-york-jfk, etc),
    aircraft, cockpits, gameplay/tutorial/challenge content, and voice/
    sound/video packs are never referenced by an unrelated third-party
    scenery package's own placements at all.
  - fs-base-cgl (57GB in a real install) is entirely ~19,000 .cgl files --
    a proprietary format bgl_extractor.py never reads (only .bgl, .spb,
    sim.cfg) -- confirmed by direct inspection, not assumed. Excluded
    outright: zero functional loss, by far the single biggest win.
  - fs-base itself (14GB) DOES contain real .bgl files and real textures
    (DDS/PNG/TIF/BMP/JPG) alongside irrelevant weather/cloud/localization
    formats (.gvp/.locPak/.cld/.WX/.WTB/.xzp/.bin) -- kept, but filtered
    to only the extensions bgl_extractor/mesh_convert ever open.

Within every included package, only files with an extension the real
pipeline actually reads are kept (see _RELEVANT_EXTENSIONS) -- the same
filter applied uniformly, so a huge partially-relevant package like
fs-base gets pruned down to just its useful fraction instead of an all-
or-nothing per-package decision.

Read-only against the MSFS install: this script never writes, moves, or
deletes anything under the install itself -- only reads it, and only
ever writes into the new staging/output location.

Usage:
    python shrink_msfs_library.py [--msfs-install "D:\\Games\\Microsoft Flight Simulator"]
                                   [--output-dir <folder for the .7z>]
                                   [--dry-run]
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import bgl_extractor

# Every package name confirmed, by direct inspection of a real MSFS
# install's package catalog this session, to be genuine shared/generic
# content -- not tied to any one real-world airport, aircraft, or
# gameplay mode. Matched by package FOLDER NAME (not full path), so this
# works the same whether the real store subfolder is ".../Official/Steam"
# or ".../Official/OneStore" -- package names themselves are the same
# Asobo/Microsoft-authored content across storefronts.
_INCLUDED_PACKAGES = {
    "asobo-jetways",                 # generic jetway models
    "asobo-material-lib",            # shared material definitions
    "asobo-modellib-airport-generic",  # generic airport objects: lights, windsocks, signs, barriers
    "asobo-modellib-buildings",      # generic building/facade kit pieces
    "asobo-modellib-props",          # generic props (barriers, clutter, ...)
    "asobo-modellib-texture",        # shared texture library backing the modellib-* packages above
    "asobo-simobjects-animals",      # generic animal SimObjects
    "asobo-simobjects-boats",        # generic boat SimObjects
    "asobo-simobjects-characters",   # generic people (ground crew, marshallers, passengers, ...)
    "asobo-simobjects-landmarks",    # generic landmark objects
    "asobo-simobjects-misc",         # generic misc SimObjects (cones, GSE-adjacent clutter, ...)
    "asobo-simobjects-vehicles",     # generic ground vehicles (GSE, buses, fuel trucks, ...)
    "fs-base",                       # core defaults -- real .bgl + real textures, filtered (see below)
    "fs-base-cgl",                   # kept in the catalog so it's LOGGED as excluded, not silently
                                      # missing -- see _EXCLUDED_PACKAGES and the module docstring.
    "fs-base-material-lib",          # base material library
}

# Confirmed by direct inspection: fs-base-cgl is ~19,000 files, ALL
# ".cgl" -- a format bgl_extractor.py never reads (it only ever looks
# for .bgl/.spb/sim.cfg). Listed explicitly (not just "not in the
# include set") so a run's report is honest about the single biggest
# exclusion instead of it just quietly not appearing anywhere.
_EXCLUDED_PACKAGES_WITH_REASON = {
    "fs-base-cgl": "proprietary .cgl terrain/world-data format -- never read by bgl_extractor.py",
}

# Only files with one of these extensions are ever opened by
# bgl_extractor.py's install-wide scan (build_install_guid_index,
# build_install_simobject_index, _copy_simobject_model's sibling-texture
# search) or mesh_convert.convert's extract_image external-texture
# fallback -- everything else in an included package (weather/cloud data,
# localization packs, compiled shaders, ...) is dead weight for this
# tool's purposes specifically, even though it's real content MSFS itself
# needs at runtime.
_RELEVANT_EXTENSIONS = {
    ".bgl", ".spb",                                    # scenery/placement data
    ".gltf", ".glb", ".bin",                            # 3-D models + external glTF buffers
    ".xml", ".cfg",                                     # sim.cfg + ModelBehaviors/ModelInfo XML
    ".png", ".dds", ".jpg", ".jpeg", ".tga", ".ktx2",   # textures mesh_convert/bgl_extractor read
    ".tif", ".tiff", ".bmp",                            # textures seen in real fs-base content
}
# Package identity files -- not read by the pipeline at all, but tiny
# (a few KB) and worth keeping so the shrunk package still looks/behaves
# like a normal MSFS package to any other tool that might inspect it.
_ALWAYS_KEEP_FILENAMES = {"manifest.json", "layout.json"}


def _human_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}PB"


def find_package_root(msfs_install: Path):
    """Prefer an actual Official/<Store> packages folder (e.g.
    HLM_Packages/Official/Steam or Packages/Official/OneStore) over the
    bare install root, since that's where the per-package folders
    (matched against _INCLUDED_PACKAGES by name) actually live."""
    resolved = bgl_extractor._resolve_real_install_root(msfs_install)
    candidates = list(resolved.rglob("Official/*"))
    stores = [c for c in candidates if c.is_dir() and any(p.is_dir() for p in c.iterdir())]
    if stores:
        # Prefer whichever candidate actually contains a known package name.
        for store in stores:
            names = {p.name for p in store.iterdir() if p.is_dir()}
            if names & _INCLUDED_PACKAGES:
                return store
        return stores[0]
    return resolved


def plan_shrink(msfs_install: Path, log=print):
    """Returns (package_root, {package_name: [Path, ...]}, total_bytes,
    found_included, missing_included) without copying anything."""
    package_root = find_package_root(msfs_install)
    log(f"Package root: {package_root}")

    found = {}
    missing = sorted(_INCLUDED_PACKAGES - _EXCLUDED_PACKAGES_WITH_REASON.keys())
    total_bytes = 0

    for pkg_name in sorted(_INCLUDED_PACKAGES - _EXCLUDED_PACKAGES_WITH_REASON.keys()):
        pkg_dir = package_root / pkg_name
        if not pkg_dir.is_dir():
            continue
        missing.remove(pkg_name)
        keep_files = []
        for f in pkg_dir.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix.lower() in _RELEVANT_EXTENSIONS or f.name.lower() in _ALWAYS_KEEP_FILENAMES:
                keep_files.append(f)
                total_bytes += f.stat().st_size
        found[pkg_name] = keep_files
        log(f"  {pkg_name}: {len(keep_files)} file(s) kept, {_human_size(sum(f.stat().st_size for f in keep_files))}")

    for pkg_name, reason in _EXCLUDED_PACKAGES_WITH_REASON.items():
        pkg_dir = package_root / pkg_name
        if pkg_dir.is_dir():
            excl_size = sum(f.stat().st_size for f in pkg_dir.rglob("*") if f.is_file())
            log(f"  EXCLUDED {pkg_name} ({_human_size(excl_size)}): {reason}")

    if missing:
        log(f"  Note: {len(missing)} expected package(s) not found in this install (skipped): {missing}")

    return package_root, found, total_bytes


def copy_staged(package_root: Path, found: dict, staging_dir: Path, log=print):
    staging_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pkg_name, files in found.items():
        pkg_src = package_root / pkg_name
        for f in files:
            rel = f.relative_to(pkg_src)
            dst = staging_dir / pkg_name / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(f, dst)
            copied += 1
            if copied % 2000 == 0:
                log(f"  ...{copied} files copied so far")
    log(f"Copied {copied} file(s) into {staging_dir}")


def compress_to_7z(staging_dir: Path, archive_path: Path, log=print):
    import py7zr
    log(f"Compressing {staging_dir} -> {archive_path} (this can take a while for several GB)...")
    with py7zr.SevenZipFile(archive_path, "w") as archive:
        archive.writeall(staging_dir, arcname=staging_dir.name)
    log(f"Wrote {archive_path} ({_human_size(archive_path.stat().st_size)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--msfs-install", type=Path, default=Path(r"D:\Games\Microsoft Flight Simulator"),
                     help="MSFS installation root")
    ap.add_argument("--output-dir", type=Path, default=None,
                     help="Where to write the .7z (default: next to this script)")
    ap.add_argument("--keep-staging", action="store_true",
                     help="Don't delete the uncompressed staging folder after archiving")
    ap.add_argument("--dry-run", action="store_true", help="Report only, copy/compress nothing")
    args = ap.parse_args()

    if not args.msfs_install.is_dir():
        print(f"ERROR: {args.msfs_install} is not a directory", file=sys.stderr)
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir or script_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    package_root, found, total_bytes = plan_shrink(args.msfs_install)
    print(f"\nPlan: {sum(len(v) for v in found.values())} file(s), {_human_size(total_bytes)} total "
          f"across {len(found)} package(s).")

    if args.dry_run:
        print("Dry run -- nothing copied or compressed.")
        return

    staging_dir = output_dir / "msfs_shared_library"
    if staging_dir.exists():
        print(f"Removing previous staging folder {staging_dir}...")
        shutil.rmtree(staging_dir)

    copy_staged(package_root, found, staging_dir)

    archive_path = output_dir / "msfs_shared_library.7z"
    compress_to_7z(staging_dir, archive_path)

    if not args.keep_staging:
        shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"\nDone in {time.time() - t0:.0f}s. Point msfs2xp's 'MSFS 2020 (optional)' field at the "
          f"extracted contents of {archive_path.name} instead of the full install to resolve "
          f"base-game/library objects for ANY airport package, at a fraction of the size.")


if __name__ == "__main__":
    main()
