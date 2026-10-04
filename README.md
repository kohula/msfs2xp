# MSFS2XP

A converter that turns a Microsoft Flight Simulator (2020/2024) scenery
package into X-Plane 11/12 custom scenery: buildings, ground clutter,
taxi/apron pavement and markings, lighting, and a native `apt.dat`.

It's a pure-Python rewrite with no external binaries — no DSFTool, no
ImageMagick, no third-party DDS/KTX2 decoders shelled out to. The DSF
compiler, OBJ8 writer, and BGL/SPB readers are all implemented directly
in this codebase.

## What it converts

- **Models**: MSFS glTF/glb assets → X-Plane OBJ8, including animated
  parts (doors, barriers, gates, rotating fixtures), night lighting, and
  glass/translucent materials. Each model is converted at the most
  detailed MSFS level of detail that fits a triangle budget scaled to its
  size, small props get a draw distance, placement scale is honoured, and
  normal maps are written in X-Plane's `NORMAL_METALNESS` layout.
- **Glass**: drawn at a chosen opacity (default 50%, 100% = solid) since
  MSFS glass relies on reflections X-Plane doesn't draw, and drawn after a
  building's opaque parts so it never hides them.
- **Textures**: KTX2/DDS decoded or passed through only when X-Plane can
  load them as they are, then held to the size of the objects drawing
  them (largest side configurable, default 2048); unused ones removed.
- **Ground content**: draped pavement, painted lines and markings,
  merged across overlapping placements to avoid X-Plane's draw-order
  z-fighting between same-layer draped surfaces.
- **Terrain fit**: large buildings get a correction against real sampled
  X-Plane terrain, so a big footprint doesn't float or sink at its corners
  on sloped ground. Where the ground under something taller than 1 m
  varies by more than 1 m, it is lifted level to the highest ground and a
  skirt is added below its walls down to the terrain, instead of bending
  it (and its roof) to the slope. Objects meant to reach below the ground (a drain tile's
  channel, anything MSFS places below its ground level) have that drop
  baked into their geometry, since X-Plane won't sink an object below the
  terrain from a negative height. X-Plane's terrain itself is left as it
  is; seats, people and other props standing on a building's floor are
  set on that floor's level rather than on the ground under each of them.
- **Flat objects on the airport ground**: covers, plates and flush
  fixtures (under 0.5 m tall, no lights) inside the airport are removed,
  and large flat ground sheets are draped onto the terrain, since rigid
  flat objects hover over X-Plane's uneven ground
  (`--keep-flat-objects` / the GUI checkbox keeps them as they are).
- **Exclusions**: each converted object's own footprint (cut into
  north/east-aligned rectangles after rotation, +0.5 m), plus the
  package's own exclusion rectangles.
- **Flatten fallback** (off by default; GUI checkbox or `--flatten`): the
  apt.dat flattens the terrain inside the airport boundary (runways,
  taxiways, aprons, stands and the airport's own buildings and
  furnishings), as MSFS does, and models there are no longer
  terrain-fitted -- for an airport where objects still float or sink.
- **apt.dat**: built from the package's own MSFS airport record (MSFS
  2020 and 2024 layouts): runways with thresholds, approach lighting and
  VASI/PAPI, helipads, frequencies, taxiway signs, windsocks, taxiway
  light strings, the ATC taxi network (real edges, runway edges, taxiway
  names and size classes, hot zones, vehicle roads) and ramp starts with
  heading, type, size class and airlines. Runways and aprons are a
  transparent hard surface under the converted draped MSFS pavement by
  default (options: real X-Plane runways with markings; MSFS painted
  lines). A matched stock Global Airports block only lends ATC flows,
  metadata, the beacon and truck routes -- and is used on its own only
  when the package's airport record can't be decoded.
- **SimObject placements**: people, vehicles, and GSE decoded from the
  package's SimPropContainer (`.spb`) data -- read from each file's own
  property table, or with the full decompiler when a Propdefs folder is
  configured (see [Third-party code](#third-party-code) below).

Two companion FlyWithLua scripts (`plugins/`) drive behavior X-Plane has
no native dataref for: proximity-triggered animations (doors/barriers
that open when the aircraft gets close) and day/night-gated blink
lighting. See each plugin's own `README.txt`.

## Requirements

- Python 3.10+ (developed against 3.12)
- `pip install -r requirements.txt` (numpy, Pillow, texture2ddecoder,
  zstandard; `py7zr` is needed for terrain-fit against 7z-compressed
  default scenery, `pyopencl` is optional GPU acceleration — both
  feature-detected, the app runs without either)
- An X-Plane 11 or 12 install, for terrain sampling and as the deploy
  target
- Optionally, Microsoft's own MSFS SDK "Propdefs" XML data: SimProp
  containers are read without it, but children that reference another
  object by title need it — point Settings → "Propdefs folder" (or the
  `MSFS2XP_PROPDEFS_DIR` env var) at your own copy. Not bundled here;
  see [Third-party code](#third-party-code).

## Running it

From source:

```
python main.py
```

Headless (same pipeline, no window):

```
python cli.py <msfs package folder> -o "<X-Plane>/Custom Scenery/<pack name>"
```

`python cli.py --help` lists the options (glass opacity, max texture
size, native runways, painted lines, ...). The output folder must be the
pack's own folder; earlier output in it is cleared first.

Or build a standalone Windows executable:

```
exe.bat
```

which produces `dist/MSFS2XP.exe` (see that script's own comments for
what it bundles and why). On Linux, `packaging/build_appimage.sh` builds
`dist/MSFS2XP-x86_64.AppImage`. Pushing a `v*` tag -- or changing the version in
`packaging/VERSION` -- builds both on GitHub Actions and publishes them as
a release (`.github/workflows/release.yml`).

The cache, scratch space and settings live next to the program -- next to
the `.exe`, or next to the `.AppImage` file (the AppImage itself is
mounted read-only) -- or in the per-user cache folder (`~/.cache/msfs2xp`,
`%LOCALAPPDATA%\msfs2xp`) when that folder isn't writable.
`MSFS2XP_DATA_DIR` overrides it.

## Known limitations

- MSFS's own vector-polygon ground/vegetation system (`TerrainVectorDb`)
  isn't converted to geometry.
- A source model's own MSFS-side "TILTED" per-vertex terrain conforming
  has no direct OBJ8 equivalent; this project applies its own rigid-tilt
  correction instead (see above), which is close but not per-vertex.
- Placement pitch/bank aren't applied (DSF placements only carry a
  heading).
- MSFS jetways stay converted models; X-Plane's own docking jetways
  (apt.dat 1500) aren't generated.

## Development notes

Parts of this project were built with AI assistance (Claude).

## Third-party code

`spb2xml/` is a Python port of a third-party `.spb` decompiler tool —
see [`spb2xml/NOTICE.md`](spb2xml/NOTICE.md) for full attribution.
Microsoft/Asobo's own MSFS SDK Propdefs data is **not** included or
bundled anywhere in this project or its packaged builds; you need your
own copy from the MSFS SDK.

## License

MIT — see [LICENSE](LICENSE).
