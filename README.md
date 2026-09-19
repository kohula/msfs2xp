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
  glass/translucent materials.
- **Ground content**: draped pavement, painted lines and markings,
  merged across overlapping placements to avoid X-Plane's draw-order
  z-fighting between same-layer draped surfaces.
- **Terrain fit**: large buildings get a rigid-tilt correction against
  real sampled X-Plane terrain, so a big footprint doesn't float or sink
  at its corners on sloped ground.
- **apt.dat**: runway positions, the ATC taxi-route network, and ramp
  starts are rebuilt from the package's own MSFS BGL data (not just
  copied from X-Plane's stock real-world block), so ATC/AI ground
  routing matches a custom-rebuilt airport. Ground-service-vehicle
  routing and stand metadata fall back to the nearest matching stock
  entry where MSFS has no equivalent data.
- **SimObject placements**: people, vehicles, and GSE decoded from the
  package's SimPropContainer (`.spb`) data (see [Third-party
  code](#third-party-code) below).

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
- Microsoft's own MSFS SDK "Propdefs" XML data, if you want SimObject
  (people/vehicle/GSE) extraction — point Settings → "Propdefs folder"
  (or the `MSFS2XP_PROPDEFS_DIR` env var) at your own copy. Not
  bundled here; see [Third-party code](#third-party-code).

## Running it

From source:

```
python main.py
```

Or build a standalone Windows executable:

```
exe.bat
```

which produces `dist/MSFS2XP.exe` (see that script's own comments for
what it bundles and why).

## Known limitations

- MSFS's own vector-polygon ground/vegetation system (`TerrainVectorDb`)
  isn't converted to geometry.
- A source model's own MSFS-side "TILTED" per-vertex terrain conforming
  has no direct OBJ8 equivalent; this project applies its own rigid-tilt
  correction instead (see above), which is close but not per-vertex.
- Runway/pavement lighting is currently inherited from X-Plane's own
  stock airport data, not decoded from MSFS.

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
