Test build: models keep MSFS's zero point, detail-map texture fix, WED
winding errors fixed.

## What's new

- **Models keep MSFS's zero point**: the converter used to lift every
  model so its lowest point stood on the ground and compensate with a
  negative height. Where that compensation got lost, an object with parts
  below zero (a drain tile's channel, a foundation) stood on its lowest
  point and hovered. Models now stand on the ground at their own zero
  point, as in MSFS; anything below it stays below the ground.
- **Detail-map textures**: a material without its own normal map used the
  MSFS detail map's small tiling normal map, stretched over the whole
  surface -- walls came out as a large black-and-white blotch pattern, and
  signs looked oddly metallic. Detail-map normals are no longer used; a
  detail colour texture used on its own keeps its tiling.
- **apt.dat apron and boundary outlines** are always written
  counter-clockwise (WED: "Taxiway 'Apron' is wound clock wise").
- `msfs2xp_placements.csv` (from 1.1.5) is still written: one row per
  converted object with how its height was worked out.

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
