"""Headless entry point: run a conversion without the Tk window.

    python cli.py <package_dir> -o <output_pack_dir> [options]

Same pipeline the GUI runs (pipeline.run_pipeline); the replacement picker
is never shown here, so unresolved objects are only written to
unresolved_objects.json for a later run of pick_replacements.py.
"""

import argparse
import multiprocessing
import sys

from pipeline import PipelineHooks, PipelineOptions, run_pipeline


def build_parser():
    p = argparse.ArgumentParser(description="Convert an MSFS scenery package to an X-Plane scenery pack.")
    p.add_argument("package", help="MSFS package folder (the one with manifest.json)")
    p.add_argument("-o", "--out", required=True, help="output scenery pack folder (inside Custom Scenery)")
    p.add_argument("--msfs-install", default="", help="MSFS install root, for stock/ASOBO models")
    p.add_argument("--propdefs", default="", help="MSFS SDK Propdefs folder (optional)")
    p.add_argument("--xp11-legacy", action="store_true", help="write a pre-11.50 apt.dat")
    p.add_argument("--clean-run", action="store_true", help="wipe the disk cache and temp files first")
    p.add_argument("--no-cache", action="store_true", help="don't read or write the disk cache")
    p.add_argument("--animated-doors", action="store_true",
                   help="keep proximity/business-hours animations (default: static doors)")
    p.add_argument("--approximate-substitution", action="store_true",
                   help="fuzzy-match missing objects' names to X-Plane library objects")
    p.add_argument("--no-terrain-vectors", action="store_true", help="skip the TerrainVectorDb scan")
    p.add_argument("--pol-polygons", action="store_true",
                   help="write draped ground as .pol DSF polygons instead of draped .obj")
    p.add_argument("--native-runways", action="store_true",
                   help="apt.dat: draw X-Plane runways with markings instead of a transparent surface")
    p.add_argument("--glass-opacity", type=int, default=50, metavar="PCT",
                   help="how opaque building glass is drawn, 1-100 (default 50; 100 = solid)")
    p.add_argument("--max-texture", type=int, default=2048, metavar="PX",
                   help="largest texture side for the biggest buildings; smaller objects get less "
                        "(default 2048, 0 = keep source sizes)")
    p.add_argument("--painted-lines", action="store_true",
                   help="apt.dat: paint the MSFS painted-line records")
    p.add_argument("--keep-flat-objects", "--keep-runway-objects", dest="keep_runway_objects",
                   action="store_true",
                   help="keep flat objects lying on the airport ground as they are (by default small "
                        "ones -- covers, plates -- are removed and large ground sheets draped, since "
                        "they hover over X-Plane's terrain)")
    p.add_argument("--flatten", action="store_true",
                   help="apt.dat: flatten the terrain inside the airport boundary, as MSFS does "
                        "(fallback; by default X-Plane's terrain is kept)")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    opts = PipelineOptions(
        pkg_dir=a.package,
        out_dir=a.out,
        msfs_install_dir=a.msfs_install,
        propdefs_dir=a.propdefs,
        xp_version="xp11" if a.xp11_legacy else "xp12",
        clean_run=a.clean_run,
        disable_cache=a.no_cache,
        static_doors=not a.animated_doors,
        approximate_substitution=a.approximate_substitution,
        scan_terrain_vectors=not a.no_terrain_vectors,
        pol_polygons=a.pol_polygons,
        prompt_replacements=False,
        runway_surface="native" if a.native_runways else "transparent",
        native_painted_lines=a.painted_lines,
        glass_opacity=max(1, min(100, a.glass_opacity)),
        max_texture=max(0, a.max_texture),
        flatten_airport=a.flatten,
        remove_runway_clutter=not a.keep_runway_objects,
    )
    run_pipeline(opts, PipelineHooks())
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
