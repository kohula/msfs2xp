Test build: heights from the terrain X-Plane actually draws, building
parts fitted together, vehicles driving their path on the terrain.

## What's new

- **Terrain heights from X-Plane's own mesh**: every height the converter
  works out (terrain fit, props on floors, paths) now comes from the
  terrain mesh X-Plane draws, not the elevation raster it was built from.
  Between mesh points the drawn ground can sit well off the raster, which
  left objects floating or sunk by small amounts here and there. Falls
  back to the raster where a tile's mesh can't be read.
- **Building parts fit together**: parts of one building placed as
  separate objects (wings, interiors, facades) take the building's own
  terrain correction -- warped with it, or raised/lowered to its level --
  instead of each being corrected on its own.
- **Moving vehicles**: wheels follow the vehicle along its path while
  spinning (every animated level of a looping animation is kept), and
  the path itself is raised or lowered onto the X-Plane ground at each
  point.
- **Below-zero geometry**: the negative-AGL experiment from 1.1.9 is
  removed; models stand on their MSFS zero point, unchanged.

## Windows

Two downloads, the same program:

- `MSFS2XP-win64.zip` -- unzip it anywhere and run `MSFS2XP\MSFS2XP.exe`.
  Recommended: Windows Defender's machine-learning check sometimes
  blocks the single-file exe as "potentially unwanted" (a false positive
  common to self-extracting Python apps); the folder version starts
  faster too.
- `MSFS2XP.exe` -- one file, as before.

The cache, scratch space and settings are kept next to the .exe.

## Linux

```
chmod +x MSFS2XP-x86_64.AppImage
./MSFS2XP-x86_64.AppImage
```

(Browsers strip the executable bit on download -- `chmod +x`, or right-click
-> Properties -> Permissions -> "Allow executing file as program".)

No more `--appimage-extract`: the program runs straight from the AppImage
and keeps its cache, scratch space and settings in `_cache/`, `_temp/` and
`msfs2xp_config.json` **next to the .AppImage file** (or in
`~/.cache/msfs2xp` if that folder isn't writable; `MSFS2XP_DATA_DIR`
overrides both). It also no longer needs `libfuse2` -- any system with
FUSE (`fusermount` or `fusermount3`) works. Built on Ubuntu 22.04, so it
runs on distributions from that age onward.
