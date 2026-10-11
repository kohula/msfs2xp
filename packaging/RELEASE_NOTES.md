Test build: one shared ground grid for every object's height.

## What's new

- **One ground grid for all heights**: every placed object is now fitted
  onto X-Plane's terrain through one shared grid of terrain heights (posts
  every 4 m around the converted scenery). Each vertex moves by the
  grid's height at its own position, whichever object it belongs to, so
  things that touch in MSFS still touch: a seat stays on its floor, a
  deck on its pillars, glass in its frame, people on their seats.
  Objects standing on the ground stand on X-Plane's ground, and nothing
  of the terrain is flattened.
- Large faces (floors, decks, roofs) are subdivided where the ground
  bends under them, so they follow the grid between their corners just
  like the small objects standing on them. Walls and pillars are never
  cut up.
- A building on steep ground (more than 3 % slope) is kept level, with
  a skirt down to the terrain, as before; the grid is levelled under it,
  so whatever is placed inside it stays level with it too.
- This replaces the earlier per-object fixes (an averaged shift per
  object, shift sharing between parts, container levelling from 1.2.1,
  props moved onto floors), which picked a different correction for
  each object and so pulled touching parts apart.
- Small flat ground clutter is no longer removed inside buildings.
- The placement report (msfs2xp_placements.csv) shows each placement's
  grid lift instead of the old floor column.

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
