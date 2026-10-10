Test build: buildings placed from one MSFS container stay together on
sloped terrain.

## What's new

- **Containers placed as one build**: everything an MSFS SimPropContainer
  (.spb) places (a terminal's shell, glass, floors, pillars, seats,
  people) used to be put on the X-Plane ground under each object's own
  anchor and terrain-fitted on its own. On sloped terrain that moved each
  piece by a different amount, so windows and interiors sat above the
  floor on one side while pillars missed it on the other. Now the
  objects of one container (those within 50 m of each other) share one
  ground level, the middle of the terrain under them, and every object
  keeps the height MSFS gives it relative to the others. They are not
  terrain-fitted one by one any more. Ground decals still drape on the
  terrain, and the terrain itself is never flattened. Where the ground
  under such a group varies by more than 5 m (a row of lamp posts up a
  hill), the objects are placed one by one as before.
- Seats and other props of such a container keep their MSFS heights
  instead of being moved onto a floor separately.
- Terrain-fitted copies that no placement uses are removed from the
  output.

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
